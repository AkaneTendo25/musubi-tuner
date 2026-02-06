# Convert ACE-Step LoRA from kohya format to ComfyUI format
#
# Key mapping:
#   kohya format:  lora_unet_layers_0_self_attn_q_proj.lora_down.weight
#   ComfyUI format: lora_unet_decoder_layers_0_self_attn_q_proj.lora_down.weight
#
# The difference is that ComfyUI expects the full model path including "decoder."
# since the ACE-Step model structure is: model.decoder.layers.0.self_attn.q_proj

import argparse
import os
from safetensors.torch import save_file
from safetensors import safe_open

import logging

from musubi_tuner.utils.model_utils import precalculate_safetensors_hashes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_acestep_lora_to_comfy(state_dict: dict, metadata: dict = None) -> tuple[dict, dict]:
    """Convert ACE-Step LoRA state dict from kohya format to ComfyUI format.

    Args:
        state_dict: LoRA state dict in kohya format
        metadata: Optional metadata dict

    Returns:
        Tuple of (converted_state_dict, updated_metadata)
    """
    converted_dict = {}
    count = 0

    for key, value in state_dict.items():
        new_key = key

        # Add "decoder_" after "lora_unet_" prefix to match ComfyUI model structure
        # lora_unet_layers_0_... -> lora_unet_decoder_layers_0_...
        # lora_unet_condition_embedder... -> lora_unet_decoder_condition_embedder...
        if key.startswith("lora_unet_"):
            # Extract the part after "lora_unet_"
            suffix = key[len("lora_unet_"):]
            new_key = f"lora_unet_decoder_{suffix}"
            count += 1

        converted_dict[new_key] = value

    logger.info(f"Converted {count} keys to ComfyUI format")

    # Update metadata if provided
    if metadata is not None:
        metadata = metadata.copy()
        metadata["ss_comfy_format"] = "true"

    return converted_dict, metadata


def convert_comfy_to_acestep_lora(state_dict: dict, metadata: dict = None) -> tuple[dict, dict]:
    """Convert ACE-Step LoRA state dict from ComfyUI format to kohya format (reverse).

    Args:
        state_dict: LoRA state dict in ComfyUI format
        metadata: Optional metadata dict

    Returns:
        Tuple of (converted_state_dict, updated_metadata)
    """
    converted_dict = {}
    count = 0

    for key, value in state_dict.items():
        new_key = key

        # Remove "decoder_" after "lora_unet_" prefix
        # lora_unet_decoder_layers_0_... -> lora_unet_layers_0_...
        if key.startswith("lora_unet_decoder_"):
            suffix = key[len("lora_unet_decoder_"):]
            new_key = f"lora_unet_{suffix}"
            count += 1

        converted_dict[new_key] = value

    logger.info(f"Converted {count} keys to kohya format")

    # Update metadata if provided
    if metadata is not None:
        metadata = metadata.copy()
        if "ss_comfy_format" in metadata:
            del metadata["ss_comfy_format"]

    return converted_dict, metadata


def save_comfy_format(
    state_dict: dict,
    output_path: str,
    metadata: dict = None,
):
    """Save LoRA in ComfyUI format.

    Args:
        state_dict: LoRA state dict (kohya format will be converted)
        output_path: Output file path
        metadata: Optional metadata dict
    """
    # Convert to ComfyUI format
    comfy_dict, comfy_metadata = convert_acestep_lora_to_comfy(state_dict, metadata)

    # Calculate hashes
    if comfy_metadata is not None:
        model_hash, legacy_hash = precalculate_safetensors_hashes(comfy_dict, comfy_metadata)
        comfy_metadata["sshs_model_hash"] = model_hash
        comfy_metadata["sshs_legacy_hash"] = legacy_hash

    # Save
    save_file(comfy_dict, output_path, metadata=comfy_metadata)
    logger.info(f"Saved ComfyUI format LoRA to {output_path}")


def main(args):
    # Load source safetensors
    logger.info(f"Loading source file {args.src_path}")
    state_dict = {}
    with safe_open(args.src_path, framework="pt") as f:
        metadata = f.metadata()
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    logger.info("Converting...")

    if args.reverse:
        # ComfyUI to kohya format
        converted_dict, converted_metadata = convert_comfy_to_acestep_lora(state_dict, metadata)
    else:
        # kohya to ComfyUI format
        converted_dict, converted_metadata = convert_acestep_lora_to_comfy(state_dict, metadata)

    # Calculate hash
    if converted_metadata is not None:
        logger.info("Calculating hashes and creating metadata...")
        model_hash, legacy_hash = precalculate_safetensors_hashes(converted_dict, converted_metadata)
        converted_metadata["sshs_model_hash"] = model_hash
        converted_metadata["sshs_legacy_hash"] = legacy_hash

    # Save destination safetensors
    logger.info(f"Saving destination file {args.dst_path}")
    save_file(converted_dict, args.dst_path, metadata=converted_metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ACE-Step LoRA format")
    parser.add_argument("src_path", type=str, help="source path (kohya format by default)")
    parser.add_argument("dst_path", type=str, help="destination path (ComfyUI format by default)")
    parser.add_argument("--reverse", action="store_true", help="reverse conversion (ComfyUI to kohya)")
    args = parser.parse_args()
    main(args)
