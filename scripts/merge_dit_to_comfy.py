#!/usr/bin/env python3
"""Merge a finetuned DIT checkpoint with the original LTX-2 checkpoint for ComfyUI.

Takes a DIT-only checkpoint (model.X keys) saved by ltx2_train.py and:
1. Renames keys: model.X -> model.diffusion_model.X
2. Optionally merges with the original checkpoint to produce a complete file
   (VAE, audio VAE, vocoder, text_embedding_projection, etc.)

Usage:
    python scripts/merge_dit_to_comfy.py ^
        --dit_checkpoint output/ltx2_finetune-step00000100.safetensors ^
        --original_checkpoint E:/ComfyUI_windows_portable/ComfyUI/models/checkpoints/ltx-2-19b-dev.safetensors ^
        --output merged_comfy.safetensors

    # Rename keys only (no merge, smaller file):
    python scripts/merge_dit_to_comfy.py ^
        --dit_checkpoint output/ltx2_finetune-step00000100.safetensors ^
        --output comfy_dit_only.safetensors
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tqdm import tqdm
from safetensors.torch import load_file, save_file
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


def main():
    parser = argparse.ArgumentParser(description="Merge finetuned DIT with original LTX-2 checkpoint for ComfyUI")
    parser.add_argument("--dit_checkpoint", type=str, required=True, help="Path to finetuned DIT checkpoint")
    parser.add_argument("--original_checkpoint", type=str, default=None, help="Path to original LTX-2 checkpoint (for merging)")
    parser.add_argument("--output", type=str, required=True, help="Output path for the merged/converted checkpoint")
    args = parser.parse_args()

    # Load finetuned DIT
    print(f"Loading finetuned DIT: {args.dit_checkpoint}")
    dit_sd = load_file(args.dit_checkpoint, device="cpu")
    print(f"  {len(dit_sd)} keys loaded")

    # Rename keys: model.X -> model.diffusion_model.X
    print("Renaming keys to ComfyUI format...")
    renamed = {}
    renamed_count = 0
    for key, value in dit_sd.items():
        if key.startswith("model."):
            renamed["model.diffusion_model." + key[len("model."):]] = value
            renamed_count += 1
        else:
            renamed[key] = value
    del dit_sd
    print(f"  Renamed {renamed_count} keys (model.X -> model.diffusion_model.X)")

    extra_metadata = {}

    # Merge with original checkpoint if provided
    if args.original_checkpoint:
        print(f"Merging with original checkpoint: {args.original_checkpoint}")
        with MemoryEfficientSafeOpen(args.original_checkpoint) as f:
            all_keys = f.keys()
            missing_keys = [k for k in all_keys if k not in renamed]
            print(f"  {len(missing_keys)} non-overlapping keys to copy from original")

            # Restore original dtypes for overlapping keys (e.g. scale_shift_table F32 -> BF16 from --full_bf16)
            dtype_fixed = 0
            for key in all_keys:
                if key in renamed:
                    orig_dtype = f.header[key]["dtype"]
                    cur_dtype = renamed[key].dtype
                    # Map safetensors dtype string to torch dtype for comparison
                    st_to_torch = {"F32": "torch.float32", "F16": "torch.float16", "BF16": "torch.bfloat16"}
                    if st_to_torch.get(orig_dtype) and str(cur_dtype) != st_to_torch[orig_dtype]:
                        orig_tensor = f.get_tensor(key)
                        renamed[key] = renamed[key].to(orig_tensor.dtype)
                        dtype_fixed += 1
            if dtype_fixed:
                print(f"  Restored original dtype for {dtype_fixed} keys")

            for key in tqdm(missing_keys, desc="  Copying"):
                renamed[key] = f.get_tensor(key)
            orig_meta = f.metadata()
            if orig_meta and "config" in orig_meta:
                extra_metadata["config"] = orig_meta["config"]
        print(f"  Merged checkpoint has {len(renamed)} keys")
    else:
        print("No --original_checkpoint provided, saving DIT-only with ComfyUI key format")

    # Copy training metadata from the DIT checkpoint
    with MemoryEfficientSafeOpen(args.dit_checkpoint) as f:
        dit_meta = f.metadata()
    if dit_meta:
        extra_metadata.update(dit_meta)

    # Save
    print(f"Saving to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_file(renamed, args.output, extra_metadata if extra_metadata else None)
    size_gb = os.path.getsize(args.output) / (1024**3)
    print(f"Done! Output size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
