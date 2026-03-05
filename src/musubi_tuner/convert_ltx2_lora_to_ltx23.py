import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from musubi_tuner.ltx_2.model.transformer.model_configurator import (
    LTXAudioOnlyModelConfigurator,
    LTXModelConfigurator,
    LTXVideoOnlyModelConfigurator,
)
from musubi_tuner.networks.lora_ltx2 import LTX2_TARGET_REPLACE_MODULES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetSpec:
    module_name: str
    out_dim: int
    in_dim: int


def _group_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    groups: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        if "." not in key:
            groups.setdefault(key, {})[key] = value
            continue
        group = key.split(".", 1)[0]
        groups.setdefault(group, {})[key] = value
    return groups


def _is_standard_lora_group(keys: Iterable[str]) -> bool:
    key_set = set(keys)
    return any(k.endswith(".lora_down.weight") for k in key_set) and any(k.endswith(".lora_up.weight") for k in key_set)


def _load_target_config(model_path: str | None, target_config_path: str | None) -> dict:
    if target_config_path is not None:
        return json.loads(Path(target_config_path).read_text(encoding="utf-8"))
    if model_path is None:
        raise ValueError("Either target_model path or target_config path must be provided.")
    with safe_open(model_path, framework="pt") as f:
        metadata = f.metadata() or {}
    if "config" not in metadata:
        raise RuntimeError(
            f"Target model metadata does not contain 'config': {model_path}. "
            "Provide --target_config with a JSON config file."
        )
    return json.loads(metadata["config"])


def _normalize_config_for_configurators(config: dict) -> dict:
    cfg = dict(config)
    transformer = dict(cfg.get("transformer", {}))

    defaults = {
        "dropout": 0.0,
        "attention_bias": True,
        "num_vector_embeds": None,
        "activation_fn": "gelu-approximate",
        "num_embeds_ada_norm": 1000,
        "use_linear_projection": False,
        "only_cross_attention": False,
        "cross_attention_norm": True,
        "double_self_attention": False,
        "upcast_attention": False,
        "standardization_norm": "rms_norm",
        "norm_elementwise_affine": False,
        "qk_norm": "rms_norm",
        "positional_embedding_type": "rope",
        "use_middle_indices_grid": True,
        "use_audio_video_cross_attention": True,
        "share_ff": False,
        "av_cross_ada_norm": True,
    }
    for key, value in defaults.items():
        transformer.setdefault(key, value)
    cfg["transformer"] = transformer
    return cfg


def _build_meta_model(config: dict, variant: str) -> torch.nn.Module:
    configurator_map = {
        "video": LTXVideoOnlyModelConfigurator,
        "av": LTXModelConfigurator,
        "audio": LTXAudioOnlyModelConfigurator,
    }
    configurator = configurator_map[variant]
    with torch.device("meta"):
        return configurator.from_config(config)


def _discover_target_specs(model: torch.nn.Module) -> Dict[str, TargetSpec]:
    specs: Dict[str, TargetSpec] = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ not in LTX2_TARGET_REPLACE_MODULES:
            continue
        for child_name, child in module.named_modules():
            is_linear = child.__class__.__name__ == "Linear"
            is_conv2d = child.__class__.__name__ == "Conv2d"
            if not (is_linear or is_conv2d):
                continue
            if is_conv2d and tuple(getattr(child, "kernel_size", ())) != (1, 1):
                continue

            original_name = (name + "." if name else "") + child_name
            lora_name = f"lora_unet_{original_name}".replace(".", "_")

            weight = child.weight
            out_dim = int(weight.shape[0])
            in_dim = int(weight.shape[1])
            specs[lora_name] = TargetSpec(module_name=original_name, out_dim=out_dim, in_dim=in_dim)
    return specs


def _resolve_target_spec(group_name: str, target_specs: Dict[str, TargetSpec]) -> Tuple[str | None, TargetSpec | None]:
    direct = target_specs.get(group_name)
    if direct is not None:
        return group_name, direct

    if group_name.startswith("lora_unet_model_"):
        alt_name = "lora_unet_" + group_name[len("lora_unet_model_") :]
        alt = target_specs.get(alt_name)
        if alt is not None:
            return alt_name, alt
    elif group_name.startswith("lora_unet_"):
        alt_name = "lora_unet_model_" + group_name[len("lora_unet_") :]
        alt = target_specs.get(alt_name)
        if alt is not None:
            return alt_name, alt
    return None, None


def _pick_variant_auto(config: dict, lora_groups: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[str, Dict[str, TargetSpec]]:
    candidates = ["video", "av", "audio"]
    standard_groups = [g for g, kv in lora_groups.items() if g.startswith("lora_unet_") and _is_standard_lora_group(kv.keys())]
    best_variant = None
    best_specs: Dict[str, TargetSpec] = {}
    best_matches = -1

    for variant in candidates:
        try:
            model = _build_meta_model(config, variant)
            specs = _discover_target_specs(model)
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping variant %s (failed to build meta model): %s", variant, e)
            continue

        matches = sum(1 for g in standard_groups if _resolve_target_spec(g, specs)[1] is not None)
        logger.info("Variant %s: %s direct LoRA group matches", variant, matches)
        if matches > best_matches:
            best_matches = matches
            best_variant = variant
            best_specs = specs

    if best_variant is None:
        raise RuntimeError("Could not build any target variant model from provided checkpoint metadata.")
    return best_variant, best_specs


def _adapt_delta_pad_crop(delta: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
    adapted = torch.zeros((out_dim, in_dim), dtype=delta.dtype, device=delta.device)
    copy_out = min(out_dim, delta.shape[0])
    copy_in = min(in_dim, delta.shape[1])
    adapted[:copy_out, :copy_in] = delta[:copy_out, :copy_in]
    return adapted


def _factorize_delta(delta: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # SVD factorization to LoRA form: up @ down ~= delta
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    k = min(rank, int(s.shape[0]))
    s_sqrt = torch.sqrt(torch.clamp(s[:k], min=0))
    up = u[:, :k] * s_sqrt.unsqueeze(0)
    down = s_sqrt.unsqueeze(1) * vh[:k, :]

    if k < rank:
        up = torch.cat([up, torch.zeros((up.shape[0], rank - k), dtype=up.dtype, device=up.device)], dim=1)
        down = torch.cat([down, torch.zeros((rank - k, down.shape[1]), dtype=down.dtype, device=down.device)], dim=0)
    return down, up


def _convert_one_group(
    group_name: str,
    group_items: Dict[str, torch.Tensor],
    target_spec: TargetSpec,
    allow_reshape: bool,
) -> Tuple[Dict[str, torch.Tensor], str]:
    down_key = next(k for k in group_items if k.endswith(".lora_down.weight"))
    up_key = next(k for k in group_items if k.endswith(".lora_up.weight"))
    down = group_items[down_key]
    up = group_items[up_key]

    if down.ndim != 2 or up.ndim != 2:
        raise ValueError(f"{group_name}: expected 2D lora_down/lora_up tensors, got {down.shape} and {up.shape}")
    if down.shape[0] != up.shape[1]:
        raise ValueError(f"{group_name}: rank mismatch down/up: {down.shape} vs {up.shape}")

    source_rank = int(down.shape[0])
    source_in = int(down.shape[1])
    source_out = int(up.shape[0])
    target_in = target_spec.in_dim
    target_out = target_spec.out_dim

    converted: Dict[str, torch.Tensor] = {}

    if source_in == target_in and source_out == target_out:
        converted[down_key] = down
        converted[up_key] = up
        for k, v in group_items.items():
            if k not in (down_key, up_key):
                converted[k] = v
        return converted, "exact"

    if not allow_reshape:
        raise ValueError(
            f"{group_name}: shape mismatch source(out={source_out},in={source_in}) -> target(out={target_out},in={target_in})"
        )

    work_dtype = torch.float32
    delta = up.to(work_dtype) @ down.to(work_dtype)  # [out, in]
    adapted_delta = _adapt_delta_pad_crop(delta, target_out, target_in)
    new_down, new_up = _factorize_delta(adapted_delta, source_rank)

    converted[down_key] = new_down.to(dtype=down.dtype)
    converted[up_key] = new_up.to(dtype=up.dtype)
    for k, v in group_items.items():
        if k not in (down_key, up_key):
            converted[k] = v
    return converted, "reshaped"


def _rename_group_keys(group_items: Dict[str, torch.Tensor], old_group: str, new_group: str) -> Dict[str, torch.Tensor]:
    renamed: Dict[str, torch.Tensor] = {}
    for key, value in group_items.items():
        if key == old_group:
            new_key = new_group
        else:
            new_key = key.replace(old_group, new_group, 1)
        renamed[new_key] = value
    return renamed


def convert_ltx2_lora_to_ltx23(
    input_path: str,
    output_path: str,
    target_model_path: str | None,
    target_config_path: str | None = None,
    target_variant: str = "auto",
    allow_reshape: bool = True,
    strict: bool = False,
    dry_run: bool = False,
    report_path: str | None = None,
) -> Dict[str, int]:
    logger.info("Loading input LoRA: %s", input_path)
    state_dict = load_file(input_path)
    with safe_open(input_path, framework="pt") as f:
        metadata = f.metadata() or {}

    groups = _group_keys(state_dict)
    target_config = _normalize_config_for_configurators(_load_target_config(target_model_path, target_config_path))

    if target_variant == "auto":
        resolved_variant, target_specs = _pick_variant_auto(target_config, groups)
    else:
        resolved_variant = target_variant
        target_specs = _discover_target_specs(_build_meta_model(target_config, target_variant))

    logger.info("Using target variant: %s (discovered %s target modules)", resolved_variant, len(target_specs))

    out_sd: Dict[str, torch.Tensor] = {}
    stats = {
        "groups_total": 0,
        "groups_lora_unet": 0,
        "groups_exact": 0,
        "groups_reshaped": 0,
        "groups_skipped_missing_target": 0,
        "groups_skipped_incompatible": 0,
        "groups_passthrough_nonstandard": 0,
        "keys_written": 0,
    }
    missing_targets: list[str] = []

    for group_name, group_items in groups.items():
        stats["groups_total"] += 1

        if not group_name.startswith("lora_unet_"):
            out_sd.update(group_items)
            stats["groups_passthrough_nonstandard"] += 1
            continue

        stats["groups_lora_unet"] += 1

        if not _is_standard_lora_group(group_items.keys()):
            out_sd.update(group_items)
            stats["groups_passthrough_nonstandard"] += 1
            continue

        resolved_name, target_spec = _resolve_target_spec(group_name, target_specs)
        if target_spec is None:
            missing_targets.append(group_name)
            stats["groups_skipped_missing_target"] += 1
            if strict:
                raise RuntimeError(f"Missing target module for LoRA group: {group_name}")
            continue

        try:
            converted_group, mode = _convert_one_group(
                group_name=group_name,
                group_items=group_items,
                target_spec=target_spec,
                allow_reshape=allow_reshape,
            )
        except Exception as e:  # noqa: BLE001
            if strict:
                raise
            logger.warning("Skipping incompatible group %s: %s", group_name, e)
            stats["groups_skipped_incompatible"] += 1
            continue
        converted_group = _rename_group_keys(converted_group, group_name, group_name)
        out_sd.update(converted_group)
        if mode == "exact":
            stats["groups_exact"] += 1
        else:
            stats["groups_reshaped"] += 1

    stats["keys_written"] = len(out_sd)

    if strict and stats["groups_reshaped"] > 0:
        raise RuntimeError("Strict mode enabled but reshaping was required for one or more LoRA groups.")

    logger.info(
        "Conversion summary: exact=%s reshaped=%s missing_target=%s passthrough=%s",
        stats["groups_exact"],
        stats["groups_reshaped"],
        stats["groups_skipped_missing_target"],
        stats["groups_passthrough_nonstandard"],
    )
    if missing_targets:
        preview = ", ".join(missing_targets[:10])
        logger.warning("Missing target groups (first 10): %s", preview)

    report = dict(stats)
    report["target_variant"] = resolved_variant
    report["target_modules_discovered"] = len(target_specs)
    report["missing_target_groups"] = missing_targets

    if report_path:
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote conversion report: %s", report_path)

    if dry_run:
        logger.info("Dry-run mode: output file not written.")
        return stats

    metadata = dict(metadata)
    metadata["ss_ltx_lora_converter"] = "ltx2_to_ltx23_v1"
    metadata["ss_ltx_lora_target_variant"] = str(resolved_variant)
    metadata["ss_ltx_lora_groups_exact"] = str(stats["groups_exact"])
    metadata["ss_ltx_lora_groups_reshaped"] = str(stats["groups_reshaped"])
    metadata["ss_ltx_lora_groups_missing_target"] = str(stats["groups_skipped_missing_target"])

    logger.info("Saving converted LoRA: %s", output_path)
    save_file(out_sd, output_path, metadata=metadata)
    logger.info("Done.")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LTX-2 LoRA to LTX-23-compatible LoRA")
    parser.add_argument("--input", type=str, required=True, help="Input LoRA safetensors path")
    parser.add_argument("--output", type=str, required=True, help="Output LoRA safetensors path")
    parser.add_argument(
        "--target_model",
        type=str,
        required=False,
        default=None,
        help="Target LTX-23 transformer safetensors path (used for module/shape discovery from config metadata)",
    )
    parser.add_argument(
        "--target_config",
        type=str,
        required=False,
        default=None,
        help="JSON file with full model config. Use this when target_model has no embedded config metadata.",
    )
    parser.add_argument(
        "--target_variant",
        type=str,
        choices=["auto", "video", "av", "audio"],
        default="auto",
        help="Target transformer variant. auto picks the best match against input LoRA keys.",
    )
    parser.add_argument(
        "--no_reshape",
        action="store_true",
        help="Disable SVD reshape fallback for mismatched layer shapes; mismatches are skipped (or fail with --strict).",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on missing target groups or reshaped groups.")
    parser.add_argument("--dry_run", action="store_true", help="Run conversion checks without writing output file.")
    parser.add_argument("--report", type=str, default=None, help="Optional JSON report output path.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    convert_ltx2_lora_to_ltx23(
        input_path=args.input,
        output_path=args.output,
        target_model_path=args.target_model,
        target_config_path=args.target_config,
        target_variant=args.target_variant,
        allow_reshape=not args.no_reshape,
        strict=args.strict,
        dry_run=args.dry_run,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
