# LoRA module for Cosmos Predict 2.5

import ast
from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


# Target the transformer Block modules which contain self_attn, cross_attn, and mlp
COSMOS_TARGET_REPLACE_MODULES = ["CosmosBlock"]


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
    # add default exclude patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    # Exclude non-attention/MLP modules: embeddings, norms, adaln modulation layers
    # LoRA targets: q_proj, k_proj, v_proj, output_proj (in self_attn and cross_attn), mlp.layer1, mlp.layer2
    exclude_patterns.append(r".*(x_embedder|pos_embedder|extra_pos_embedder|t_embedder|t_embedding_norm|final_layer|crossattn_proj|img_context_proj).*")
    exclude_patterns.append(r".*adaln_modulation.*")
    exclude_patterns.append(r".*layer_norm.*")
    exclude_patterns.append(r".*(q_norm|k_norm|v_norm).*")

    kwargs["exclude_patterns"] = exclude_patterns

    return lora.create_network(
        COSMOS_TARGET_REPLACE_MODULES,
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
        COSMOS_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
