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


def main() -> None:
    args = _parse_args()
    input_paths = [str(Path(p)) for p in args.lora_weight]
    multipliers = _resolve_multipliers(input_paths, args.lora_multiplier)

    logger.info("Merging %d LoRA file(s).", len(input_paths))
    for path, mult in zip(input_paths, multipliers):
        logger.info("  %s (multiplier=%s)", path, mult)

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
            a_parts = []
            b_parts = []
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

                a_parts.append(a)
                b_parts.append(b)

            if not a_parts:
                continue

            merged_a = torch.cat(a_parts, dim=0).to(dtype=output_dtype).contiguous()
            merged_b = torch.cat(b_parts, dim=1).to(dtype=output_dtype).contiguous()
            merged_sd[module + a_suffix] = merged_a
            merged_sd[module + b_suffix] = merged_b
            if emit_alpha:
                merged_sd[f"{module}.alpha"] = torch.tensor(float(merged_a.shape[0]), dtype=torch.float32)

            if idx % 100 == 0 or idx == len(all_modules):
                logger.info("Merged modules: %d / %d", idx, len(all_modules))

    output_path = Path(args.save_merged_lora)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "merged_with": "musubi_tuner.ltx2_merge_lora",
        "merge_format": fmt,
        "merge_sources": json.dumps(input_paths),
        "merge_multipliers": json.dumps(multipliers),
    }
    mem_eff_save_file(merged_sd, str(output_path), metadata=metadata)
    logger.info("Saved merged LoRA: %s", output_path)


if __name__ == "__main__":
    main()
