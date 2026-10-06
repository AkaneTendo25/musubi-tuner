"""LoRA adapter for the Kandinsky 6 joint video/audio DiT."""

from __future__ import annotations

import ast
from typing import Dict, List, Optional

import torch
from torch import nn

import musubi_tuner.networks.lora as lora


KANDINSKY6_TARGET_REPLACE_MODULES = ["TransformerEncoderBlock", "FusedTransformerDecoderBlock"]


def _patterns(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return list(value)


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    # Modulation and output heads are deliberately excluded: attention and FFN projections
    # cover both the video and audio paths while keeping adapter size manageable.
    exclude_patterns = _patterns(kwargs.get("exclude_patterns"))
    exclude_patterns.extend((r".*modulation.*", r".*out_layer\.out_layer.*"))
    kwargs["exclude_patterns"] = exclude_patterns
    network = lora.create_network(
        KANDINSKY6_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )
    if not network.unet_loras:
        raise RuntimeError("Kandinsky 6 LoRA found zero target modules")
    return network


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    return lora.create_network_from_weights(
        None, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
