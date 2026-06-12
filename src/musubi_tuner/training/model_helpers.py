"""Default model/checkpoint helpers for NetworkTrainer."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
from typing import Any, Dict

import torch
from safetensors.torch import load_file

from musubi_tuner import convert_lora
import musubi_tuner.networks.lora as lora_module

logger = logging.getLogger(__name__)


def get_checkpoint_metadata(self, args: argparse.Namespace) -> Dict[str, Any]:
    """Return extra metadata to include in LoRA safetensors. Override in subclasses."""
    return {}


def post_save_checkpoint_hook(self, args, ckpt_file, ckpt_name, accelerator, force_sync_upload=False, **kwargs):
    """Hook called after checkpoint is saved. Override in subclasses for architecture-specific processing."""
    pass


def i2v_training(self) -> bool:
    return self._i2v_training


def control_training(self) -> bool:
    return self._control_training


def resolve_network_module(self, network_module):
    if isinstance(network_module, str):
        return importlib.import_module(network_module)
    return network_module


def convert_weight_keys(self, weights_sd: dict[str, torch.Tensor], network_module: lora_module):
    if not weights_sd:
        return weights_sd

    network_module_obj = self._resolve_network_module(network_module)
    module_converter = getattr(network_module_obj, "convert_weight_keys", None)
    if callable(module_converter):
        converted = module_converter(weights_sd)
        if converted is not None:
            return converted

    keys = list(weights_sd.keys())
    if keys[0].startswith("lora_"):
        return weights_sd
    if keys[0].startswith("diffusion_model.") or keys[0].startswith("transformer."):
        logger.info("converting LoRA weights from diffusers format to default format")
        return convert_lora.convert_from_diffusers("lora_unet_", weights_sd)
    return weights_sd


def load_network_weights(self, file: str, network_module: lora_module) -> dict[str, torch.Tensor]:
    if os.path.splitext(file)[1] == ".safetensors":
        weights_sd = load_file(file)
    else:
        weights_sd = torch.load(file, map_location="cpu")
    return self.convert_weight_keys(weights_sd, network_module)


def load_network_state_dict(
    network: torch.nn.Module,
    weights_sd: dict[str, torch.Tensor],
    strict: bool = False,
):
    custom_loader = getattr(network, "load_weights_state_dict", None)
    if callable(custom_loader):
        return custom_loader(weights_sd, strict)
    return network.load_state_dict(weights_sd, strict)
