"""Merge one or more LTX-2 LoRA files into a base model checkpoint.

Loads the model, merges LoRA weights, and saves — same as inference-time merging.

Usage:
    python ltx2_merge_lora_to_model.py ^
        --dit base_model.safetensors ^
        --lora_weight stage1_output/last.safetensors ^
        --save_merged_model merged_model.safetensors
"""

from __future__ import annotations

import argparse
import json
import logging

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from musubi_tuner.ltx2_lora_utils import (
    import_lora_network_module,
    infer_lora_network_module,
    load_lora_metadata,
    parse_lora_network_args,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LTX-2 LoRA weights into a base model checkpoint")
    parser.add_argument("--dit", type=str, required=True, help="LTX-2 base model checkpoint (.safetensors)")
    parser.add_argument("--lora_weight", type=str, nargs="+", required=True, help="LoRA weight path(s)")
    parser.add_argument(
        "--lora_multiplier", type=float, nargs="*", default=None,
        help="Per-LoRA multipliers aligned with --lora_weight. If omitted, all are 1.0.",
    )
    parser.add_argument("--save_merged_model", type=str, required=True, help="Output merged model path (.safetensors)")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for merge computation (default: cuda if available)",
    )
    parser.add_argument("--audio_video", action="store_true", help="Load as audio-video model")
    parser.add_argument("--audio_only", action="store_true", help="Load as audio-only model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lora_paths = args.lora_weight
    multipliers = args.lora_multiplier
    if multipliers is None:
        multipliers = [1.0] * len(lora_paths)
    elif len(multipliers) == 1 and len(lora_paths) > 1:
        multipliers = [multipliers[0]] * len(lora_paths)
    elif len(multipliers) != len(lora_paths):
        raise ValueError(
            f"--lora_multiplier count ({len(multipliers)}) must be 1 or match --lora_weight count ({len(lora_paths)})"
        )

    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Preserve original metadata for model config auto-detection on reload
    with safe_open(args.dit, framework="pt") as f:
        original_metadata = f.metadata() or {}

    # Load transformer
    from musubi_tuner.ltx2_model_loading import load_ltx2_model

    logger.info("Loading LTX-2 model from %s", args.dit)
    transformer = load_ltx2_model(
        model_path=args.dit,
        device=device,
        load_device=device,
        torch_dtype=torch.bfloat16,
        attn_mode="torch",
        audio_video=args.audio_video or args.audio_only,
        audio_only_model=args.audio_only,
    )
    transformer.eval()

    # Merge each LoRA (same path as ltx2_generate_video.py inference merging)
    for i, (lora_path, mult) in enumerate(zip(lora_paths, multipliers)):
        logger.info("Merging LoRA [%d/%d]: %s (multiplier=%.4f)", i + 1, len(lora_paths), lora_path, mult)
        metadata = load_lora_metadata(lora_path)
        lora_sd = load_file(lora_path)
        network_module_name = infer_lora_network_module(metadata, lora_sd)
        network_module = import_lora_network_module(network_module_name)
        network_args = parse_lora_network_args(metadata.get("ss_network_args"))
        net = network_module.create_arch_network_from_weights(
            multiplier=mult, weights_sd=lora_sd, unet=transformer, for_inference=True,
            **network_args,
        )
        net.merge_to(None, transformer, lora_sd, device=device, non_blocking=True)
        del lora_sd, net

    # Save merged transformer with original metadata preserved
    logger.info("Saving merged model to %s", args.save_merged_model)
    merged_sd = transformer.model.state_dict()

    metadata = dict(original_metadata)
    metadata["merged_loras"] = json.dumps(lora_paths)
    metadata["merged_multipliers"] = json.dumps(multipliers)

    from musubi_tuner.utils.safetensors_utils import mem_eff_save_file
    mem_eff_save_file(merged_sd, args.save_merged_model, metadata=metadata)
    logger.info("Done: %s (%d keys)", args.save_merged_model, len(merged_sd))


if __name__ == "__main__":
    main()
