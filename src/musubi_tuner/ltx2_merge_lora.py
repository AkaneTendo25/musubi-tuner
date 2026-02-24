from __future__ import annotations

import argparse
import json
import logging
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import torch
from safetensors import safe_open

from musubi_tuner.utils.safetensors_utils import mem_eff_save_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FormatName = Literal["lora_ab", "lora_down_up"]

FORMAT_SUFFIX: Dict[FormatName, Tuple[str, str]] = {
    "lora_ab": (".lora_A.weight", ".lora_B.weight"),
    "lora_down_up": (".lora_down.weight", ".lora_up.weight"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple LTX-2 LoRA files into a single LoRA.")
    parser.add_argument("--lora_weight", type=str, nargs="+", required=True, help="Input LoRA paths to merge in order.")
    parser.add_argument(
        "--lora_multiplier",
        type=float,
        nargs="*",
        default=None,
        help="LoRA multipliers aligned with --lora_weight. If omitted, all multipliers are 1.0.",
    )
    parser.add_argument("--save_merged_lora", type=str, required=True, help="Output merged LoRA safetensors path.")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Output tensor dtype. auto promotes across input LoRAs.",
    )
    parser.add_argument(
        "--emit_alpha",
        action="store_true",
        help="Always emit <module>.alpha keys. For lora_down/up format this is always enabled.",
    )
    parser.add_argument(
        "--merge_method",
        type=str,
        choices=["concat", "orthogonal"],
        default="concat",
        help=(
            "LoRA merge method. "
            "'concat' keeps all ranks by concatenation (default). "
            "'orthogonal' performs a two-LoRA orthogonalized merge with SVD refactorization."
        ),
    )
    parser.add_argument(
        "--orthogonal_k_fraction",
        type=float,
        default=0.5,
        help=(
            "Orthogonal merge only: fraction of top singular directions projected out "
            "bilaterally before combining modules. Range [0, 1]."
        ),
    )
    parser.add_argument(
        "--orthogonal_rank_mode",
        type=str,
        choices=["sum", "max", "min"],
        default="sum",
        help=(
            "Orthogonal merge only: target rank before clipping by matrix dimensions. "
            "'sum'=rank1+rank2, 'max'=max(rank1,rank2), 'min'=min(rank1,rank2)."
        ),
    )
    return parser.parse_args()


def _resolve_multipliers(weights: List[str], multipliers: List[float] | None) -> List[float]:
    if multipliers is None or len(multipliers) == 0:
        return [1.0] * len(weights)
    if len(multipliers) == 1 and len(weights) > 1:
        return [multipliers[0]] * len(weights)
    if len(multipliers) != len(weights):
        raise ValueError(
            f"--lora_multiplier count ({len(multipliers)}) must be 1 or match --lora_weight count ({len(weights)})."
        )
    return list(multipliers)


def _detect_format(keys: Iterable[str]) -> FormatName:
    has_ab = False
    has_down_up = False
    for key in keys:
        if key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"):
            has_ab = True
        if key.endswith(".lora_down.weight") or key.endswith(".lora_up.weight"):
            has_down_up = True
        if has_ab and has_down_up:
            break
    if has_ab and has_down_up:
        raise ValueError("Mixed LoRA formats detected in a single file (both A/B and down/up).")
    if has_ab:
        return "lora_ab"
    if has_down_up:
        return "lora_down_up"
    raise ValueError("No supported LoRA keys found. Expected lora_A/lora_B or lora_down/lora_up keys.")


def _module_prefixes(keys: Iterable[str], a_suffix: str, b_suffix: str) -> set[str]:
    a_prefixes = {k[: -len(a_suffix)] for k in keys if k.endswith(a_suffix)}
    b_prefixes = {k[: -len(b_suffix)] for k in keys if k.endswith(b_suffix)}
    both = a_prefixes & b_prefixes
    if not both:
        raise ValueError("No complete LoRA modules found (missing A/B or down/up pairs).")
    missing = (a_prefixes ^ b_prefixes)
    if missing:
        logger.warning("Ignoring %d incomplete module entries.", len(missing))
    return both


def _to_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _promote_dtype(dtypes: List[torch.dtype]) -> torch.dtype:
    if not dtypes:
        return torch.float32
    out = dtypes[0]
    for dt in dtypes[1:]:
        out = torch.promote_types(out, dt)
    if out == torch.float64:
        # Keep output practical for LoRA checkpoints.
        return torch.float32
    return out


def _read_alpha(reader, prefix: str, rank: int) -> float:
    alpha_key = f"{prefix}.alpha"
    if alpha_key not in reader.keys():
        return float(rank)
    alpha_val = reader.get_tensor(alpha_key)
    return float(alpha_val.detach().float().item())


def _flatten_b_rank_dim_last(b: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    # Move LoRA rank dim (dim=1) to the end so arbitrary conv/linear shapes can be flattened to [M, rank].
    b_rank_last = b.movedim(1, -1).contiguous()
    prefix_shape = tuple(int(x) for x in b_rank_last.shape[:-1])
    b_flat = b_rank_last.reshape(-1, int(b.shape[1]))
    return b_flat, prefix_shape


def _unflatten_b_rank_dim_last(b_flat: torch.Tensor, prefix_shape: Tuple[int, ...], rank: int) -> torch.Tensor:
    b_rank_last = b_flat.reshape(*prefix_shape, rank).contiguous()
    return b_rank_last.movedim(-1, 1).contiguous()


def _project_out_top_components(source: torch.Tensor, reference: torch.Tensor, k_fraction: float) -> torch.Tensor:
    if k_fraction <= 0.0:
        return source
    if source.shape[1] != reference.shape[1]:
        raise ValueError(
            f"Projection dimension mismatch: source={tuple(source.shape)} reference={tuple(reference.shape)}"
        )

    _u, singular_values, vh = torch.linalg.svd(reference, full_matrices=False)
    if singular_values.numel() == 0:
        return source

    k = int(round(float(singular_values.numel()) * float(k_fraction)))
    if k <= 0:
        return source
    k = min(k, singular_values.numel())

    basis = vh[:k, :].transpose(0, 1).contiguous()  # [N, k]
    return source - (source @ basis @ basis.transpose(0, 1))


def _resolve_target_rank(rank1: int, rank2: int, mode: str, max_rank: int) -> int:
    if mode == "sum":
        target = rank1 + rank2
    elif mode == "max":
        target = max(rank1, rank2)
    elif mode == "min":
        target = min(rank1, rank2)
    else:
        raise ValueError(f"Unsupported orthogonal rank mode: {mode}")
    return max(1, min(int(target), int(max_rank)))


def _merge_orthogonal_factors(
    a1: torch.Tensor,
    b1: torch.Tensor,
    a2: torch.Tensor,
    b2: torch.Tensor,
    *,
    k_fraction: float,
    rank_mode: str,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rank1 = int(a1.shape[0])
    rank2 = int(a2.shape[0])

    if rank1 != int(b1.shape[1]) or rank2 != int(b2.shape[1]):
        raise ValueError(
            f"Rank mismatch in orthogonal merge: "
            f"(A1={rank1}, B1={int(b1.shape[1])}) (A2={rank2}, B2={int(b2.shape[1])})"
        )

    a1_flat = a1.reshape(rank1, -1).to(dtype=torch.float32)
    a2_flat = a2.reshape(rank2, -1).to(dtype=torch.float32)

    b1_flat, b1_prefix = _flatten_b_rank_dim_last(b1.to(dtype=torch.float32))
    b2_flat, b2_prefix = _flatten_b_rank_dim_last(b2.to(dtype=torch.float32))

    if a1_flat.shape[1] != a2_flat.shape[1]:
        raise ValueError(
            f"Orthogonal merge requires matching A flattened dimensions. Got {a1_flat.shape[1]} vs {a2_flat.shape[1]}"
        )
    if b1_flat.shape[0] != b2_flat.shape[0] or b1_prefix != b2_prefix:
        raise ValueError(
            "Orthogonal merge requires matching B flattened output dimensions. "
            f"Got b1_flat={tuple(b1_flat.shape)} b2_flat={tuple(b2_flat.shape)}"
        )

    w1 = b1_flat @ a1_flat
    w2 = b2_flat @ a2_flat

    w1_ortho = _project_out_top_components(w1, w2, k_fraction)
    w2_ortho = _project_out_top_components(w2, w1, k_fraction)
    merged_w = w1_ortho + w2_ortho

    u, s, vh = torch.linalg.svd(merged_w, full_matrices=False)
    if s.numel() == 0:
        raise ValueError("Orthogonal merge produced an empty SVD spectrum.")

    target_rank = _resolve_target_rank(rank1, rank2, rank_mode, int(s.numel()))
    s_root = torch.sqrt(s[:target_rank].clamp_min(0.0))

    b_new_flat = (u[:, :target_rank] * s_root.unsqueeze(0)).contiguous()
    a_new_flat = (s_root.unsqueeze(1) * vh[:target_rank, :]).contiguous()

    a_new = a_new_flat.reshape((target_rank,) + tuple(int(x) for x in a1.shape[1:])).to(dtype=output_dtype).contiguous()
    b_new = _unflatten_b_rank_dim_last(b_new_flat, b1_prefix, target_rank).to(dtype=output_dtype).contiguous()
    return a_new, b_new


def main() -> None:
    args = _parse_args()
    input_paths = [str(Path(p)) for p in args.lora_weight]
    multipliers = _resolve_multipliers(input_paths, args.lora_multiplier)
    merge_method = str(args.merge_method)
    orthogonal_k_fraction = float(args.orthogonal_k_fraction)
    orthogonal_rank_mode = str(args.orthogonal_rank_mode)

    if not (0.0 <= orthogonal_k_fraction <= 1.0):
        raise ValueError(f"--orthogonal_k_fraction must be in [0, 1]. Got: {orthogonal_k_fraction}")
    if merge_method == "orthogonal" and len(input_paths) != 2:
        raise ValueError("--merge_method orthogonal currently supports exactly 2 input LoRAs.")

    logger.info("Merging %d LoRA file(s) with method=%s.", len(input_paths), merge_method)
    for path, mult in zip(input_paths, multipliers):
        logger.info("  %s (multiplier=%s)", path, mult)
    if merge_method == "orthogonal":
        logger.info(
            "Orthogonal settings: k_fraction=%s, rank_mode=%s",
            orthogonal_k_fraction,
            orthogonal_rank_mode,
        )

    with ExitStack() as stack:
        readers = [stack.enter_context(safe_open(path, framework="pt", device="cpu")) for path in input_paths]
        keysets = [set(reader.keys()) for reader in readers]

        formats = [_detect_format(keys) for keys in keysets]
        if len(set(formats)) != 1:
            raise ValueError(f"All input LoRAs must use the same format. Got: {formats}")
        fmt = formats[0]
        a_suffix, b_suffix = FORMAT_SUFFIX[fmt]

        module_sets = [_module_prefixes(keys, a_suffix, b_suffix) for keys in keysets]
        all_modules = sorted(set().union(*module_sets))
        shared_modules = set.intersection(*module_sets)
        logger.info(
            "Detected format: %s | modules: union=%d, shared=%d",
            fmt,
            len(all_modules),
            len(shared_modules),
        )

        sampled_a_dtypes = []
        for reader, module_names in zip(readers, module_sets):
            first = next(iter(module_names))
            sampled_a_dtypes.append(reader.get_tensor(first + a_suffix).dtype)

        if args.dtype == "auto":
            output_dtype = _promote_dtype(sampled_a_dtypes)
        else:
            output_dtype = _to_dtype(args.dtype)
        logger.info("Output dtype: %s", output_dtype)

        emit_alpha = args.emit_alpha or fmt == "lora_down_up"
        merged_sd: Dict[str, torch.Tensor] = {}

        for idx, module in enumerate(all_modules, start=1):
            module_parts: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
            expected_a_tail = None
            expected_b_head_tail = None

            for reader_idx, reader in enumerate(readers):
                a_key = module + a_suffix
                b_key = module + b_suffix
                if a_key not in keysets[reader_idx] or b_key not in keysets[reader_idx]:
                    continue

                a = reader.get_tensor(a_key).to(dtype=torch.float32)
                b = reader.get_tensor(b_key).to(dtype=torch.float32)
                rank_a = int(a.shape[0])
                rank_b = int(b.shape[1])
                if rank_a != rank_b:
                    raise ValueError(f"Rank mismatch in {input_paths[reader_idx]} for module {module}: A={rank_a}, B={rank_b}")

                a_tail = tuple(a.shape[1:])
                b_head_tail = (int(b.shape[0]),) + tuple(b.shape[2:])
                if expected_a_tail is None:
                    expected_a_tail = a_tail
                    expected_b_head_tail = b_head_tail
                elif expected_a_tail != a_tail or expected_b_head_tail != b_head_tail:
                    raise ValueError(f"Shape mismatch across LoRAs for module {module}")

                alpha = _read_alpha(reader, module, rank_a)
                scale = float(multipliers[reader_idx]) * (alpha / float(rank_a))
                b = b.mul(scale)

                module_parts.append((reader_idx, a, b))

            if not module_parts:
                continue

            used_orthogonal = False
            merged_a = None
            merged_b = None

            if merge_method == "orthogonal" and len(module_parts) == 2:
                (_, a1, b1), (_, a2, b2) = module_parts
                try:
                    merged_a, merged_b = _merge_orthogonal_factors(
                        a1,
                        b1,
                        a2,
                        b2,
                        k_fraction=orthogonal_k_fraction,
                        rank_mode=orthogonal_rank_mode,
                        output_dtype=output_dtype,
                    )
                    used_orthogonal = True
                except Exception as e:
                    logger.warning(
                        "Orthogonal merge fallback to concat for module '%s': %s",
                        module,
                        str(e),
                    )

            if merged_a is None or merged_b is None:
                a_parts = [a for _, a, _ in module_parts]
                b_parts = [b for _, _, b in module_parts]
                merged_a = torch.cat(a_parts, dim=0).to(dtype=output_dtype).contiguous()
                merged_b = torch.cat(b_parts, dim=1).to(dtype=output_dtype).contiguous()

            merged_sd[module + a_suffix] = merged_a
            merged_sd[module + b_suffix] = merged_b
            if emit_alpha:
                merged_sd[f"{module}.alpha"] = torch.tensor(float(merged_a.shape[0]), dtype=torch.float32)

            if idx % 100 == 0 or idx == len(all_modules):
                if merge_method == "orthogonal":
                    logger.info(
                        "Merged modules: %d / %d (last_method=%s)",
                        idx,
                        len(all_modules),
                        "orthogonal" if used_orthogonal else "concat",
                    )
                else:
                    logger.info("Merged modules: %d / %d", idx, len(all_modules))

    output_path = Path(args.save_merged_lora)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "merged_with": "musubi_tuner.ltx2_merge_lora",
        "merge_format": fmt,
        "merge_method": merge_method,
        "merge_sources": json.dumps(input_paths),
        "merge_multipliers": json.dumps(multipliers),
    }
    if merge_method == "orthogonal":
        metadata["orthogonal_k_fraction"] = str(orthogonal_k_fraction)
        metadata["orthogonal_rank_mode"] = orthogonal_rank_mode
    mem_eff_save_file(merged_sd, str(output_path), metadata=metadata)
    logger.info("Saved merged LoRA: %s", output_path)


if __name__ == "__main__":
    main()
