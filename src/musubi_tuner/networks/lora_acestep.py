# LoRA module for ACE-Step 1.5 DiT decoder
#
# This module creates LoRA with ComfyUI-compatible key names.
# The key format is: lora_unet_decoder_{module_path}.{lora_down|lora_up|alpha}
# This matches the ACE-Step model structure in ComfyUI: decoder.layers.N.self_attn.q_proj etc.

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
ACESTEP_TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
ACESTEP_EXCLUDED_SUBSTRINGS = ("encoder", "embedding", "norm", "layernorm", ".ln", "_ln")


def _prepare_include_patterns(include_patterns: Optional[str]) -> List[str]:
    if include_patterns is None:
        return []
    return ast.literal_eval(include_patterns)


def _discover_target_module_names(target_model: nn.Module) -> List[str]:
    """Discover the exact projection modules to target for ACE-Step LoRA."""
    target_names: List[str] = []

    for name, module in target_model.named_modules():
        if module.__class__.__name__ != "Linear":
            continue
        lower_name = name.lower()
        if any(token in lower_name for token in ACESTEP_EXCLUDED_SUBSTRINGS):
            continue
        if lower_name.endswith(ACESTEP_TARGET_SUFFIXES):
            target_names.append(name)

    target_names = sorted(set(target_names))
    if not target_names:
        raise ValueError("No ACE-Step LoRA target modules were discovered in the decoder")

    return target_names


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
    verbose = kwargs.get("verbose", False)

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

    discovered_target_names = _discover_target_module_names(target_model)
    if include_patterns:
        unmatched_patterns = []
        for pattern in include_patterns:
            regex = re.compile(pattern)
            if not any(regex.match(name) for name in discovered_target_names):
                unmatched_patterns.append(pattern)
        if unmatched_patterns:
            raise ValueError(f"ACE-Step include_patterns matched no discovered decoder targets: {unmatched_patterns}")
        kwargs["include_patterns"] = include_patterns
    else:
        kwargs["include_patterns"] = [rf"^{re.escape(name)}$" for name in discovered_target_names]

    kwargs["verbose"] = verbose
    logger.info(
        f"Discovered {len(discovered_target_names)} ACE-Step LoRA target modules; "
        f"example targets: {discovered_target_names[:4]}"
    )

    # Use "lora_unet_decoder" prefix for ComfyUI compatibility
    # This produces keys like: lora_unet_decoder_layers_0_self_attn_q_proj.lora_down.weight
    # which map to ComfyUI model path: decoder.layers.0.self_attn.q_proj.weight
    network = lora.create_network(
        ACESTEP_TARGET_REPLACE_MODULES,
        "lora_unet_decoder",  # ComfyUI-compatible prefix
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        target_model,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )

    logger.info("Created ACE-Step LoRA with ComfyUI-compatible key format (lora_unet_decoder_*)")

    return network


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    """Create LoRA network from saved weights.

    Supports both old format (lora_unet_*) and new ComfyUI-compatible format (lora_unet_decoder_*).
    Old format weights are automatically converted to new format for compatibility.

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
    # Check if this is old format (lora_unet_ without decoder_) and convert if needed
    sample_key = next(iter(weights_sd.keys()), "")
    if sample_key.startswith("lora_unet_") and not sample_key.startswith("lora_unet_decoder_"):
        logger.info("Converting old LoRA format to ComfyUI-compatible format")
        converted_sd = {}
        for key, value in weights_sd.items():
            if key.startswith("lora_unet_"):
                new_key = "lora_unet_decoder_" + key[len("lora_unet_"):]
                converted_sd[new_key] = value
            else:
                converted_sd[key] = value
        weights_sd = converted_sd

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
