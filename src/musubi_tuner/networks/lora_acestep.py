# LoRA module for ACE-Step 1.5 DiT decoder

import ast
import re
from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


# Pass None to search all modules, filtering by include/exclude patterns
ACESTEP_TARGET_REPLACE_MODULES = None


def _prepare_include_patterns(include_patterns: Optional[str]) -> List[str]:
    if include_patterns is None:
        # Default to attention projection layers as in the reference ACE-Step trainer
        return [
            r".*q_proj.*",
            r".*k_proj.*",
            r".*v_proj.*",
            r".*o_proj.*",
        ]
    return ast.literal_eval(include_patterns)


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
    """Create LoRA network for ACE-Step decoder.

    Args:
        multiplier: LoRA multiplier
        network_dim: LoRA rank
        network_alpha: LoRA alpha
        vae: VAE model (not used but required for interface)
        text_encoders: Text encoders (not used but required for interface)
        unet: AceStepConditionGenerationModel
        neuron_dropout: Dropout rate
        **kwargs: Additional arguments

    Returns:
        LoRANetwork instance
    """
    include_patterns = _prepare_include_patterns(kwargs.get("include_patterns", None))
    kwargs["include_patterns"] = include_patterns

    # Exclude encoder, embeddings, and normalization layers
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = [
            r".*encoder.*",  # Don't train condition encoder
            r".*embedding.*",  # Skip embeddings
            r".*norm.*",  # Skip normalization layers
            r".*layernorm.*",
            r".*ln.*",
        ]
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)
        exclude_patterns.extend([r".*encoder.*", r".*embedding.*", r".*norm.*"])
    kwargs["exclude_patterns"] = exclude_patterns

    # Target the decoder component of the model
    # ACE-Step model has model.decoder for the DiT
    if hasattr(unet, "decoder"):
        target_model = unet.decoder
        logger.info("Targeting ACE-Step decoder for LoRA injection")
    else:
        target_model = unet
        logger.info("Targeting entire model for LoRA injection")

    # Log model structure for debugging
    logger.info(f"Model type: {type(target_model)}")
    for name, module in target_model.named_modules():
        if "proj" in name.lower():
            logger.info(f"  Found proj module: {name} -> {type(module).__name__}")

    # Enable verbose to see what modules are being targeted
    kwargs["verbose"] = True

    return lora.create_network(
        ACESTEP_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        target_model,
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
    """Create LoRA network from saved weights.

    Args:
        multiplier: LoRA multiplier
        weights_sd: State dict with LoRA weights
        text_encoders: Text encoders (optional)
        unet: Target model (optional)
        for_inference: Whether loading for inference
        **kwargs: Additional arguments

    Returns:
        LoRANetwork instance
    """
    # Target the decoder if available
    if unet is not None and hasattr(unet, "decoder"):
        target_model = unet.decoder
    else:
        target_model = unet

    return lora.create_network_from_weights(
        None,  # Search all modules
        multiplier,
        weights_sd,
        text_encoders,
        target_model,
        for_inference,
        **kwargs,
    )
