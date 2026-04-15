import ast
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import musubi_tuner.networks.lora as lora


MOVA_TARGET_REPLACE_MODULES = ["DiTBlock", "ConditionalCrossAttentionBlock"]


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
    target_scope = kwargs.pop("target_scope", "official")

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    elif isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)

    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is None:
        include_patterns = []
    elif isinstance(include_patterns, str):
        include_patterns = ast.literal_eval(include_patterns)

    exclude_patterns.extend(
        [
            r".*(norm).*",
            r".*(modulation).*",
        ]
    )

    if target_scope == "official":
        include_patterns.append(r".*(self_attn\.(q|k|v|o)|cross_attn\.(q|k|v|o)|inner\.(q|k|v|o))$")
        exclude_patterns.append(r".*(cross_attn\.(k_img|v_img)).*")
    elif target_scope == "attention_plus_image":
        include_patterns.append(
            r".*(self_attn\.(q|k|v|o)|cross_attn\.(q|k|v|o|k_img|v_img)|inner\.(q|k|v|o))$"
        )
    elif target_scope == "all_linear":
        pass
    else:
        raise ValueError(f"Unsupported MOVA LoRA target_scope: {target_scope}")

    kwargs["exclude_patterns"] = exclude_patterns
    if include_patterns:
        kwargs["include_patterns"] = include_patterns

    return lora.create_network(
        MOVA_TARGET_REPLACE_MODULES,
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
        MOVA_TARGET_REPLACE_MODULES,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        **kwargs,
    )
