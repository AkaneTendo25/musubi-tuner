from __future__ import annotations

import importlib
import json
import logging
from typing import Any, Dict, Iterable, Optional

import torch
from safetensors import safe_open

from musubi_tuner.networks import lora
from musubi_tuner.networks import lora_ltx2

logger = logging.getLogger(__name__)


def load_lora_metadata(path: str) -> Dict[str, str]:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            return dict(handle.metadata() or {})
    except Exception:
        logger.warning("Could not read LoRA metadata from %s", path, exc_info=True)
        return {}


def parse_lora_network_args(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Ignoring invalid ss_network_args JSON in LoRA metadata")
        return {}
    if not isinstance(parsed, dict):
        logger.warning("Ignoring non-dict ss_network_args metadata in LoRA file")
        return {}
    return {str(key): value for key, value in parsed.items()}


def infer_lora_network_module(metadata: Dict[str, str], weights_sd: Dict[str, torch.Tensor]) -> str:
    network_module = str(metadata.get("ss_network_module") or "").strip()
    if network_module:
        if network_module in {"networks.lokr", "musubi_tuner.networks.lokr"}:
            has_lokr = any(".lokr_" in key for key in weights_sd.keys())
            has_plain_oft = any(".oft_R." in key for key in weights_sd.keys()) and not has_lokr
            if has_plain_oft:
                return "networks.lora_ltx2"
        return network_module

    if any(".lokr_" in key for key in weights_sd.keys()):
        return "networks.lokr"

    return "networks.lora_ltx2"


def import_lora_network_module(network_module: str):
    normalized = network_module.strip()
    if normalized in {"lora_ltx2", "networks.lora_ltx2", "musubi_tuner.networks.lora_ltx2"}:
        return lora_ltx2
    if normalized in {"lora", "networks.lora", "musubi_tuner.networks.lora"}:
        return lora
    if normalized in {"lokr", "networks.lokr", "musubi_tuner.networks.lokr"}:
        return importlib.import_module("musubi_tuner.networks.lokr")
    if normalized.startswith("musubi_tuner."):
        return importlib.import_module(normalized)
    if normalized.startswith("networks."):
        return importlib.import_module(f"musubi_tuner.{normalized}")
    return importlib.import_module(normalized)


def lora_module_count(net: torch.nn.Module) -> int:
    return len(getattr(net, "text_encoder_loras", [])) + len(getattr(net, "unet_loras", []))


def iter_lora_modules(net: torch.nn.Module) -> Iterable[torch.nn.Module]:
    yield from getattr(net, "text_encoder_loras", [])
    yield from getattr(net, "unet_loras", [])


def transformer_has_scaled_fp8_linears(transformer: torch.nn.Module) -> bool:
    for module in transformer.modules():
        scale_weight = getattr(module, "scale_weight", None)
        weight = getattr(module, "weight", None)
        if isinstance(scale_weight, torch.Tensor) and isinstance(weight, torch.Tensor) and weight.dtype.itemsize == 1:
            return True
    return False


def apply_lora_network_for_inference(
    net: torch.nn.Module,
    transformer: torch.nn.Module,
    lora_sd: Dict[str, torch.Tensor],
    *,
    merge_device: torch.device,
    non_blocking: bool = True,
    prefer_merge: bool = True,
) -> str:
    if prefer_merge and not transformer_has_scaled_fp8_linears(transformer):
        net.merge_to(None, transformer, lora_sd, device=merge_device, non_blocking=non_blocking)
        return "merged"

    net.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    if hasattr(net, "load_weights_state_dict"):
        info = net.load_weights_state_dict(lora_sd, False)
    else:
        info = net.load_state_dict(lora_sd, False)
    net.eval()
    net.requires_grad_(False)
    for module in iter_lora_modules(net):
        try:
            module.to(merge_device)
        except Exception:
            pass
        if hasattr(module, "enabled"):
            module.enabled = True
    setattr(transformer, "_musubi_inference_lora_network", net)
    setattr(transformer, "_musubi_inference_lora_load_info", str(info))
    return "live"
