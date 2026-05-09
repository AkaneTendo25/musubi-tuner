"""LyCORIS helpers for LTX-2 training.

This module keeps LyCORIS-specific runtime/config logic out of ltx2_train_network.py.
"""

from __future__ import annotations

import argparse
import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from musubi_tuner.modules.nf4_optimization_utils import is_nf4_module


def is_lycoris_requested(args: argparse.Namespace) -> bool:
    network_module_name = str(getattr(args, "network_module", "") or "")
    return "lycoris" in network_module_name.lower()


def _collect_quantized_base_flags(args: argparse.Namespace, default_nf4_block_size: int) -> List[str]:
    flags: List[str] = []
    if getattr(args, "fp8_base", False):
        flags.append("fp8_base")
    if getattr(args, "fp8_scaled", False):
        flags.append("fp8_scaled")
    if getattr(args, "fp8_w8a8", False):
        flags.append(f"fp8_w8a8:{getattr(args, 'w8a8_mode', 'int8')}")
    if getattr(args, "nf4_base", False):
        flags.append(f"nf4_base:block{int(getattr(args, 'nf4_block_size', default_nf4_block_size))}")
    return flags


def validate_lycoris_quantized_base_compatibility(
    args: argparse.Namespace,
    logger_instance: logging.Logger,
    default_nf4_block_size: int,
) -> None:
    if not is_lycoris_requested(args):
        return

    check_mode = str(getattr(args, "lycoris_quantized_base_check_mode", "warn") or "warn").lower()
    if check_mode not in {"off", "warn", "error"}:
        raise ValueError(
            f"lycoris_quantized_base_check_mode must be one of ['off', 'warn', 'error']. Got: {check_mode}"
        )
    if check_mode == "off":
        return

    flags = _collect_quantized_base_flags(args, default_nf4_block_size)
    if not flags:
        return

    msg = (
        "LyCORIS with quantized base model is enabled (%s). This path relies on lycoris-lora "
        "modules and can be slower/less stable than custom quantization-aware LoKr wrappers. "
        "For a quality baseline, try disabling quantized-base flags first."
    ) % ", ".join(flags)
    if check_mode == "error":
        raise ValueError(msg)
    logger_instance.warning(msg)


def _is_quantized_runtime_module(module: Optional[torch.nn.Module]) -> bool:
    if module is None:
        return False
    try:
        if is_nf4_module(module):
            return True
    except Exception:
        pass
    if hasattr(module, "scale_weight"):
        return True
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.dtype.itemsize == 1:
        return True
    return False


def _resolve_adapter_origin_module(adapter_module: torch.nn.Module) -> Optional[torch.nn.Module]:
    candidates: List[torch.nn.Module] = []
    for attr in ("org_module", "org_module_ref", "org_modules", "org_layer", "module"):
        value = getattr(adapter_module, attr, None)
        if isinstance(value, torch.nn.Module):
            candidates.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, torch.nn.Module):
                    candidates.append(item)
    if not candidates:
        return None
    return candidates[0]


def iter_active_adapter_bindings(transformer) -> Iterator[Tuple[str, torch.nn.Module, torch.nn.Module, str]]:
    lo_ra_module = None
    try:
        from musubi_tuner.networks.lora import LoRAModule

        lo_ra_module = LoRAModule
    except Exception:
        pass
    lycoris_base_module = None
    try:
        from lycoris.modules.base import LycorisBaseModule

        lycoris_base_module = LycorisBaseModule
    except Exception:
        pass

    seen = set()
    for name, module in transformer.named_modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            continue
        bound = getattr(module.forward, "__self__", None)
        if bound is None:
            continue

        adapter_type = None
        if lo_ra_module is not None and isinstance(bound, lo_ra_module):
            adapter_type = "lora"
        elif lycoris_base_module is not None and isinstance(bound, lycoris_base_module):
            adapter_type = "lycoris"
        if adapter_type is None:
            continue

        bid = id(bound)
        if bid in seen:
            continue
        seen.add(bid)
        yield name, module, bound, adapter_type


def summarize_active_adapters(transformer) -> Dict[str, int]:
    summary: Dict[str, int] = {
        "total": 0,
        "lora": 0,
        "lycoris": 0,
        "attn1": 0,
        "attn2": 0,
        "ff": 0,
        "audio": 0,
        "av_cross": 0,
        "quantized_origin": 0,
        "lycoris_quantized_origin": 0,
    }
    blocks = set()
    for name, _, bound, adapter_type in iter_active_adapter_bindings(transformer):
        lname = name.lower()
        summary["total"] += 1
        summary[adapter_type] += 1
        if ".attn1." in lname:
            summary["attn1"] += 1
        if ".attn2." in lname:
            summary["attn2"] += 1
        if ".ff." in lname:
            summary["ff"] += 1
        if ".audio_" in lname or "audio_to_video" in lname or "video_to_audio" in lname:
            summary["audio"] += 1
        if "audio_to_video" in lname or "video_to_audio" in lname:
            summary["av_cross"] += 1

        m = re.search(r"transformer_blocks\.(\d+)", name)
        if m:
            blocks.add(int(m.group(1)))

        origin = _resolve_adapter_origin_module(bound)
        if _is_quantized_runtime_module(origin):
            summary["quantized_origin"] += 1
            if adapter_type == "lycoris":
                summary["lycoris_quantized_origin"] += 1

    summary["block_count"] = len(blocks)
    return summary


def validate_lycoris_runtime(
    args: argparse.Namespace,
    accelerator,
    transformer: Optional[torch.nn.Module],
    network: Optional[torch.nn.Module],
    logger_instance: logging.Logger,
) -> None:
    if not is_lycoris_requested(args):
        return
    if transformer is None or network is None:
        logger_instance.warning("LyCORIS validation skipped because transformer/network is unavailable")
        return

    unwrapped_transformer = transformer
    unwrapped_network = network
    if accelerator is not None:
        try:
            unwrapped_transformer = accelerator.unwrap_model(transformer)
        except Exception:
            pass
        try:
            unwrapped_network = accelerator.unwrap_model(network)
        except Exception:
            pass

    summary = summarize_active_adapters(unwrapped_transformer)
    if summary["lycoris"] <= 0:
        raise RuntimeError(
            "LyCORIS module was requested but no active LyCORIS adapters were detected on transformer layers. "
            "Check --lycoris_config preset/module targets."
        )

    configured = getattr(unwrapped_network, "loras", None)
    configured_count = len(configured) if isinstance(configured, list) else None
    if configured_count is not None and configured_count > 0 and summary["lycoris"] != configured_count:
        logger_instance.warning(
            "LyCORIS attach count mismatch: configured=%d active=%d. Verify target filters/presets.",
            configured_count,
            summary["lycoris"],
        )

    logger_instance.info(
        "LyCORIS attach summary: active=%d blocks=%d attn1=%d attn2=%d ff=%d audio=%d quantized_origins=%d",
        summary["lycoris"],
        summary["block_count"],
        summary["attn1"],
        summary["attn2"],
        summary["ff"],
        summary["audio"],
        summary["lycoris_quantized_origin"],
    )


def ensure_adapters_enabled_for_sampling(transformer) -> int:
    lora_count = 0
    for _, _, bound, _ in iter_active_adapter_bindings(transformer):
        if hasattr(bound, "enabled"):
            try:
                bound.enabled = True
            except Exception:
                pass
        lora_count += 1
    return lora_count


def get_adapter_norm_samples(transformer, limit: int = 5) -> List[str]:
    try:
        from musubi_tuner.networks.lora import LoRAModule
    except Exception:
        LoRAModule = None

    stats = []
    for name, _, bound, adapter_type in iter_active_adapter_bindings(transformer):
        if LoRAModule is not None and isinstance(bound, LoRAModule):
            try:
                up = bound.lora_up
                down = bound.lora_down
                if isinstance(up, torch.nn.ModuleList):
                    up_norm = sum(u.weight.norm().item() for u in up)
                else:
                    up_norm = up.weight.norm().item()
                if isinstance(down, torch.nn.ModuleList):
                    down_norm = sum(d.weight.norm().item() for d in down)
                else:
                    down_norm = down.weight.norm().item()
                stats.append(
                    f"{name}: up_norm={up_norm:.6f}, down_norm={down_norm:.6f}, mult={float(getattr(bound, 'multiplier', 1.0)):.3f}"
                )
            except Exception:
                pass
            if len(stats) >= limit:
                break
            continue

        if adapter_type == "lycoris":
            try:
                params = [(pn, p) for pn, p in bound.named_parameters() if isinstance(p, torch.nn.Parameter)]
                if len(params) == 0:
                    continue
                parts = []
                for pn, p in params[:2]:
                    parts.append(f"{pn}_norm={p.detach().float().norm().item():.6f}")
                stats.append(
                    f"{name}: {' '.join(parts)}, mult={float(getattr(bound, 'multiplier', 1.0)):.3f}"
                )
            except Exception:
                pass
        if len(stats) >= limit:
            break
    return stats


def process_lycoris_config(args: argparse.Namespace, logger_instance: logging.Logger) -> None:
    """Process optional LyCORIS TOML config and merge into runtime args."""
    uses_lycoris_module = is_lycoris_requested(args)

    if args.network_args is None:
        args.network_args = []

    if getattr(args, "lycoris_config", None):
        if not uses_lycoris_module:
            raise ValueError("--lycoris_config requires --network_module lycoris.kohya")

        from musubi_tuner.networks.network_config import (
            parse_network_args_enhanced,
            parse_toml_config,
            validate_network_config,
        )
        from musubi_tuner.networks.lycoris_extensions import (
            build_network_kwargs_from_config,
            config_to_lycoris_preset,
            get_config_init_params,
            log_network_config,
        )

        logger_instance.info("Loading LyCORIS config from: %s", args.lycoris_config)
        config = parse_toml_config(args.lycoris_config)

        existing_args = parse_network_args_enhanced(args.network_args)
        for key, value in existing_args.items():
            if "." not in key:
                continue

            parts = key.split(".")
            if parts[0] == "modules" and len(parts) >= 3:
                module_name = parts[1]
                param_name = parts[2]
                config.setdefault("modules", {}).setdefault(module_name, {})[param_name] = value
            elif parts[0] == "init" and len(parts) >= 2:
                param_name = parts[1]
                config.setdefault("init", {})[param_name] = value

        filtered_network_args = []
        stripped_count = 0
        for arg in args.network_args:
            if arg.startswith("modules.") or arg.startswith("init."):
                stripped_count += 1
                continue
            filtered_network_args.append(arg)
        if stripped_count > 0:
            args.network_args = filtered_network_args
            logger_instance.info(
                "Consumed %d nested TOML override args from --network_args",
                stripped_count,
            )

        validate_network_config(config)
        log_network_config(config, logger_instance)

        preset = config_to_lycoris_preset(config)
        if preset:
            args._network_config_preset = preset
            logger_instance.info("LyCORIS TOML preset prepared for network creation")

        config_kwargs = build_network_kwargs_from_config(
            config,
            base_dim=getattr(args, "network_dim", None),
            base_alpha=getattr(args, "network_alpha", None),
        )
        for key, value in config_kwargs.items():
            arg_str = f"{key}={value}"
            if not any(arg.startswith(f"{key}=") for arg in args.network_args):
                args.network_args.append(arg_str)
                logger_instance.info("Added network arg from LyCORIS config: %s", arg_str)

        init_params = dict(get_config_init_params(config))
    else:
        init_params = {}

    if getattr(args, "init_lokr_norm", None) is not None:
        init_params["lokr_norm"] = args.init_lokr_norm

    if init_params:
        args._network_init_params = init_params
        logger_instance.info("Network initialization params: %s", args._network_init_params)


def _build_attention_only_lycoris_preset(base_preset: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    preset: Dict[str, Any] = dict(base_preset or {})
    module_algo_map = dict(preset.get("module_algo_map") or {})

    attention_config = module_algo_map.get("Attention")
    if attention_config is None:
        attention_config = module_algo_map.get("BasicAVTransformerBlock")

    preset["unet_target_module"] = ["Attention"]
    preset.pop("target_module", None)
    preset.pop("name_algo_map", None)
    preset["use_fnmatch"] = False

    if attention_config is not None:
        preset["module_algo_map"] = {"Attention": dict(attention_config)}
    else:
        preset.pop("module_algo_map", None)

    return preset


def apply_lycoris_preset_before_network_creation(
    args: argparse.Namespace,
    logger_instance: logging.Logger,
) -> None:
    """Apply/patch LyCORIS preset behavior for LTX-2 before network creation."""
    if not is_lycoris_requested(args):
        network_module_name = str(getattr(args, "network_module", "") or "")
        logger_instance.warning(
            "Ignoring LyCORIS preset because --network_module=%s",
            network_module_name or "<unset>",
        )
        return

    try:
        from lycoris.kohya import LycorisNetworkKohya
    except Exception as e:
        logger_instance.warning(
            "Failed to import lycoris.kohya for preset application. "
            "Install with: pip install lycoris-lora. Error: %s",
            e,
        )
        return

    if not getattr(LycorisNetworkKohya, "_ltx2_apply_preset_patched", False):
        original_apply_preset = LycorisNetworkKohya.apply_preset.__func__
        LycorisNetworkKohya._ltx2_original_preset_state = {
            "ENABLE_CONV": LycorisNetworkKohya.ENABLE_CONV,
            "UNET_TARGET_REPLACE_MODULE": list(getattr(LycorisNetworkKohya, "UNET_TARGET_REPLACE_MODULE", [])),
            "UNET_TARGET_REPLACE_NAME": list(getattr(LycorisNetworkKohya, "UNET_TARGET_REPLACE_NAME", [])),
            "TEXT_ENCODER_TARGET_REPLACE_MODULE": list(getattr(LycorisNetworkKohya, "TEXT_ENCODER_TARGET_REPLACE_MODULE", [])),
            "TEXT_ENCODER_TARGET_REPLACE_NAME": list(getattr(LycorisNetworkKohya, "TEXT_ENCODER_TARGET_REPLACE_NAME", [])),
            "MODULE_ALGO_MAP": dict(getattr(LycorisNetworkKohya, "MODULE_ALGO_MAP", {})),
            "NAME_ALGO_MAP": dict(getattr(LycorisNetworkKohya, "NAME_ALGO_MAP", {})),
            "USE_FNMATCH": bool(getattr(LycorisNetworkKohya, "USE_FNMATCH", False)),
        }

        def _apply_preset_with_ltx2_targets(cls, preset):
            preset_dict = dict(preset or {})
            if "target_module" in preset_dict and "unet_target_module" not in preset_dict:
                preset_dict["unet_target_module"] = list(preset_dict.get("target_module", []))
            preset_dict.pop("target_module", None)
            return original_apply_preset(cls, preset_dict)

        LycorisNetworkKohya.apply_preset = classmethod(_apply_preset_with_ltx2_targets)
        LycorisNetworkKohya._ltx2_apply_preset_patched = True
        logger_instance.info("Patched LyCORIS preset application for LTX-2 target modules")

    original_state = getattr(LycorisNetworkKohya, "_ltx2_original_preset_state", None)
    if isinstance(original_state, dict):
        LycorisNetworkKohya.ENABLE_CONV = original_state["ENABLE_CONV"]
        LycorisNetworkKohya.UNET_TARGET_REPLACE_MODULE = list(original_state["UNET_TARGET_REPLACE_MODULE"])
        LycorisNetworkKohya.UNET_TARGET_REPLACE_NAME = list(original_state["UNET_TARGET_REPLACE_NAME"])
        LycorisNetworkKohya.TEXT_ENCODER_TARGET_REPLACE_MODULE = list(original_state["TEXT_ENCODER_TARGET_REPLACE_MODULE"])
        LycorisNetworkKohya.TEXT_ENCODER_TARGET_REPLACE_NAME = list(original_state["TEXT_ENCODER_TARGET_REPLACE_NAME"])
        LycorisNetworkKohya.MODULE_ALGO_MAP = dict(original_state["MODULE_ALGO_MAP"])
        LycorisNetworkKohya.NAME_ALGO_MAP = dict(original_state["NAME_ALGO_MAP"])
        LycorisNetworkKohya.USE_FNMATCH = original_state["USE_FNMATCH"]

    preset = getattr(args, "_network_config_preset", None)
    if str(getattr(args, "lora_target_preset", "") or "").lower() == "lycoris":
        preset = _build_attention_only_lycoris_preset(preset)
        logger_instance.info("Using LyCORIS target preset: lycoris (Attention modules only)")
    if not preset:
        return

    try:
        LycorisNetworkKohya.apply_preset(preset)
        logger_instance.info("Applied LyCORIS preset before network creation")
    except Exception as e:
        logger_instance.warning("Failed to apply LyCORIS preset before network creation: %s", e)
