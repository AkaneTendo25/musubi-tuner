"""
Convert LTX-2 LoRA from training format to ComfyUI format

Training format:
  - Keys: lora_unet_model_transformer_blocks_0_attn1_to_k.lora_down.weight
  - Uses underscores as separators
  - Has separate .alpha keys

ComfyUI format:
  - Keys: diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight
  - Uses dots as separators
  - No alpha keys (not used by ComfyUI)
  - Renames lora_down -> lora_A, lora_up -> lora_B
"""

import safetensors.torch
import argparse
import os
from pathlib import Path


def convert_key_to_comfy(key):
    """
    Convert a training format key to ComfyUI format

    Example:
        lora_unet_model_transformer_blocks_0_attn1_to_k.lora_down.weight
        -> diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight
    """
    # Skip alpha keys - ComfyUI doesn't use them
    if '.alpha' in key:
        return None

    # Split into main part and weight part
    parts = key.split('.')
    if len(parts) < 2:
        print(f"Warning: Unexpected key format: {key}")
        return None

    main_part = parts[0]  # e.g., lora_unet_model_transformer_blocks_0_attn1_to_k
    weight_part = '.'.join(parts[1:])  # e.g., lora_down.weight

    # Remove the lora_unet_ prefix and handle the wrapper's module structure.
    # Transformer keys: lora_unet_model_transformer_blocks_... (wrapper.model.transformer_blocks)
    # Connector keys:   lora_unet_embeddings_connector_... (wrapper.embeddings_connector)
    if main_part.startswith('lora_unet_model_'):
        # Standard transformer path: strip wrapper.model prefix
        main_part = main_part[len('lora_unet_model_'):]
    elif main_part.startswith('lora_unet_'):
        # Connector or other wrapper-level module
        main_part = main_part[len('lora_unet_'):]
    else:
        print(f"Warning: Key doesn't start with 'lora_unet_': {key}")
        return None

    # Map connector attribute names to ComfyUI model names
    # Training wrapper: self.embeddings_connector -> ComfyUI: video_embeddings_connector
    if main_part.startswith('embeddings_connector_'):
        main_part = 'video_' + main_part
    # audio_embeddings_connector is already correct

    # Convert underscores to dots for the hierarchy
    # We need to be careful with numeric parts
    # transformer_blocks_0_attn1_to_k -> transformer_blocks.0.attn1.to_k

    # Strategy: Replace patterns carefully
    # Replace _NUMBER_ with .NUMBER.
    # Replace other underscores based on context

    converted = main_part

    # IMPORTANT ORDER: Handle audio patterns BEFORE general attn patterns!
    # This prevents _attn1_ from matching inside audio_attn1_
    import re

    # Step 0: Handle connector module paths
    # video_embeddings_connector_transformer_1d_blocks_0_... -> video_embeddings_connector.transformer_1d_blocks.0....
    # audio_embeddings_connector_transformer_1d_blocks_0_... -> audio_embeddings_connector.transformer_1d_blocks.0....
    converted = converted.replace('video_embeddings_connector_', 'video_embeddings_connector.')
    converted = converted.replace('audio_embeddings_connector_', 'audio_embeddings_connector.')
    converted = converted.replace('transformer_1d_blocks_', 'transformer_1d_blocks.')

    # Step 1: Basic block structure (main transformer)
    converted = converted.replace('transformer_blocks_', 'transformer_blocks.')

    # Step 2: Handle audio/video attention patterns FIRST (keep underscores in these)
    converted = converted.replace('_audio_attn1_', '.audio_attn1.')
    converted = converted.replace('_audio_attn2_', '.audio_attn2.')
    converted = converted.replace('_audio_to_video_attn_', '.audio_to_video_attn.')
    converted = converted.replace('_video_to_audio_attn_', '.video_to_audio_attn.')
    converted = converted.replace('_audio_ff_', '.audio_ff.')

    # Step 3: Now handle regular (non-audio) attention patterns
    converted = converted.replace('_attn1_', '.attn1.')
    converted = converted.replace('_attn2_', '.attn2.')

    # Step 4: Handle projection layers (use regex to avoid matching inside audio_to_video, video_to_audio)
    # Only match _to_k/_to_q/_to_v at word boundaries (end of string or followed by .)
    converted = re.sub(r'_to_k($|\.)', r'.to_k\1', converted)
    converted = re.sub(r'_to_q($|\.)', r'.to_q\1', converted)
    converted = re.sub(r'_to_v($|\.)', r'.to_v\1', converted)
    converted = re.sub(r'_to_out\.', r'.to_out.', converted)
    converted = re.sub(r'_to_out_', r'.to_out.', converted)

    # Step 5: Handle feedforward layers
    converted = converted.replace('_ff_net_', '.ff.net.')
    converted = converted.replace('_ff_', '.ff.')
    converted = converted.replace('_net_', '.net.')
    converted = converted.replace('_proj', '.proj')

    # Step 6: Fix remaining numeric suffixes
    converted = re.sub(r'to_out_(\d+)', r'to_out.\1', converted)
    converted = re.sub(r'net_(\d+)', r'net.\1', converted)

    # Build the final key
    comfy_key = f"diffusion_model.{converted}"

    # Convert weight naming: lora_down -> lora_A, lora_up -> lora_B
    if 'lora_down' in weight_part:
        weight_part = weight_part.replace('lora_down', 'lora_A')
    elif 'lora_up' in weight_part:
        weight_part = weight_part.replace('lora_up', 'lora_B')

    comfy_key = f"{comfy_key}.{weight_part}"

    return comfy_key


def convert_lora_to_comfy(input_path, output_path=None, verbose=False):
    """
    Convert a LoRA file from training format to ComfyUI format

    Args:
        input_path: Path to the input LoRA file
        output_path: Path to save the converted LoRA (optional)
        verbose: Print detailed conversion info

    Returns:
        Path to the output file
    """
    print(f"Loading LoRA from: {input_path}")

    # Load the trained LoRA
    trained_state_dict = safetensors.torch.load_file(input_path)

    print(f"Input LoRA has {len(trained_state_dict)} keys")

    # Collect alpha and rank per LoRA module to fold scale into weights
    lora_alpha = {}
    lora_rank = {}
    for key, tensor in trained_state_dict.items():
        if key.endswith(".lora_down.weight"):
            lora_name = key.rsplit(".", 2)[0]
            if lora_name not in lora_rank:
                lora_rank[lora_name] = tensor.shape[0]
        elif key.endswith(".alpha"):
            lora_name = key.rsplit(".", 1)[0]
            lora_alpha[lora_name] = tensor

    # Convert keys
    comfy_state_dict = {}
    skipped_alpha = 0
    folded_alpha = 0
    converted = 0
    failed = 0

    for key, tensor in trained_state_dict.items():
        new_key = convert_key_to_comfy(key)

        if new_key is None:
            if '.alpha' in key:
                skipped_alpha += 1
                if verbose:
                    print(f"Skipping alpha key: {key}")
            else:
                failed += 1
                print(f"Failed to convert key: {key}")
        else:
            # Fold alpha scale into lora_B (up) weights, since Comfy ignores alpha keys.
            if key.endswith(".lora_up.weight"):
                lora_name = key.rsplit(".", 2)[0]
                alpha = lora_alpha.get(lora_name, None)
                rank = lora_rank.get(lora_name, None)
                if alpha is not None and rank is not None and rank != 0:
                    scale = float(alpha.item()) / float(rank)
                    tensor = tensor * scale
                    folded_alpha += 1
                    if verbose:
                        print(f"Folded alpha for {lora_name}: alpha={float(alpha.item())} rank={rank} scale={scale}")
            comfy_state_dict[new_key] = tensor
            converted += 1
            if verbose:
                print(f"Converted: {key} -> {new_key}")

    print("\nConversion summary:")
    print(f"  Converted: {converted} keys")
    print(f"  Skipped alpha keys: {skipped_alpha}")
    print(f"  Folded alpha into lora_B: {folded_alpha}")
    print(f"  Failed: {failed} keys")
    print(f"  Output LoRA has {len(comfy_state_dict)} keys")

    # Determine output path
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}.comfy{input_file.suffix}"

    # Load metadata from the original file
    metadata = None
    try:
        with safetensors.safe_open(input_path, framework="pt") as f:
            metadata = f.metadata()
        if metadata:
            print(f"Preserving {len(metadata)} metadata entries")
    except Exception as e:
        print(f"Warning: Could not read metadata: {e}")

    # Save the converted LoRA
    print(f"\nSaving ComfyUI-compatible LoRA to: {output_path}")
    safetensors.torch.save_file(comfy_state_dict, output_path, metadata=metadata)

    print("[OK] Conversion complete!")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert LTX-2 LoRA from training format to ComfyUI format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input LoRA file (training format)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save the converted LoRA (default: <input>.comfy.safetensors)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed conversion information"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return 1

    try:
        output_path = convert_lora_to_comfy(args.input, args.output, args.verbose)
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
