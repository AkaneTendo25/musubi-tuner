# LoRA module for Cosmos3-Nano / Cosmos3 Omni MoT

import ast
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import musubi_tuner.networks.lora as lora

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


COSMOS3_TARGET_REPLACE_MODULES = ["MoTDecoderLayer"]
COSMOS3_DEFAULT_INCLUDE_PATTERNS = [
    r".*self_attn\.(q_proj_moe_gen|k_proj_moe_gen|v_proj_moe_gen|o_proj_moe_gen)$",
    r".*mlp_moe_gen\.(gate_proj|up_proj|down_proj)$",
]


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
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    exclude_patterns.append(r".*(embed_tokens|lm_head|norm|time_embedder|time_proj|rotary_emb).*")
    exclude_patterns.append(r".*")
    kwargs["exclude_patterns"] = exclude_patterns

    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is None:
        kwargs["include_patterns"] = COSMOS3_DEFAULT_INCLUDE_PATTERNS
    elif isinstance(include_patterns, str):
        parsed_include_patterns = ast.literal_eval(include_patterns)
        kwargs["include_patterns"] = parsed_include_patterns

    return lora.create_network(
        COSMOS3_TARGET_REPLACE_MODULES,
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
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    return lora.create_network_from_weights(
        COSMOS3_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
