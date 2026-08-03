from __future__ import annotations

import ast

import torch
from torch import nn

from musubi_tuner.networks import lora

MINIMAX_H3_TARGET_REPLACE_MODULES = ["MiniMaxH3TransformerBlock"]


def create_arch_network(
    multiplier: float,
    network_dim: int | None,
    network_alpha: float | None,
    vae: nn.Module,
    text_encoders: list[nn.Module],
    unet: nn.Module,
    neuron_dropout: float | None = None,
    **kwargs,
):
    exclude_patterns = kwargs.get("exclude_patterns")
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    # Keep timestep and modality calibration frozen. Attention and feed-forward
    # projections inside each transformer block remain adapter targets.
    exclude_patterns.extend((r".*(adaln_proj|modulation).*", r".*norm.*"))
    kwargs["exclude_patterns"] = exclude_patterns

    return lora.create_network(
        MINIMAX_H3_TARGET_REPLACE_MODULES,
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


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: dict[str, torch.Tensor],
    text_encoders: list[nn.Module] | None = None,
    unet: nn.Module | None = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    return lora.create_network_from_weights(
        MINIMAX_H3_TARGET_REPLACE_MODULES,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        **kwargs,
    )
