#!/usr/bin/env python3
"""Pre-quantize LTX-2 model weights to NF4 for faster loading and smaller disk footprint.

The quantized model preserves all non-transformer weights (VAE, projections, norms, etc.)
unchanged and only applies NF4 4-bit compression to transformer block Linear weights.

Usage:
    python ltx2_quantize_model.py \\
        --input_model path/to/ltx-2.3-22b-dev.safetensors \\
        --output_model path/to/ltx-2.3-22b-dev-nf4.safetensors

    With LoftQ pre-computation (requires --network_dim):
    python ltx2_quantize_model.py \\
        --input_model path/to/ltx-2.3-22b-dev.safetensors \\
        --output_model path/to/ltx-2.3-22b-dev-nf4.safetensors \\
        --loftq_init --network_dim 32

The quantized model can then be used with --nf4_base (quantization is auto-skipped):
    python ltx2_train_network.py --ltx2_checkpoint path/to/quantized.safetensors --nf4_base ...
    python ltx2_train_network.py --ltx2_checkpoint path/to/quantized.safetensors --nf4_base --loftq_init ...

Notes:
    - The --nf4_base flag is still required at training/inference time so the monkey
      patch is applied, but the expensive quantization step is skipped.
    - LoftQ data is saved as a companion file (e.g. model-nf4.loftq_r32.safetensors).
      The training code auto-detects it when --loftq_init is used.
    - LoftQ is rank-specific: if you change --network_dim, re-run with the new rank.
"""

import argparse
import logging
import os
import time

import safetensors
import torch
from tqdm import tqdm

from musubi_tuner.ltx2_train_network import KEEP_FP8_HIGH_PRECISION_TOKENS
from musubi_tuner.modules.nf4_optimization_utils import (
    DEFAULT_NF4_BLOCK_SIZE,
    quantize_nf4_block,
    dequantize_nf4_block,
)
from musubi_tuner.modules.loftq_init import loftq_initialize
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)

_NF4_EXCLUDE_PATTERNS = KEEP_FP8_HIGH_PRECISION_TOKENS
_NF4_TARGET_PATTERNS = ("transformer_blocks",)


def _is_quantizable(key: str, value: torch.Tensor, block_size: int) -> bool:
    """Check if a tensor key should be NF4-quantized."""
    is_target = any(t in key for t in _NF4_TARGET_PATTERNS) and key.endswith(".weight")
    is_excluded = any(e in key for e in _NF4_EXCLUDE_PATTERNS)
    if not is_target or is_excluded or value.ndim != 2:
        return False
    _, in_features = value.shape
    return in_features % block_size == 0


def loftq_path_for_model(output_model: str, network_dim: int) -> str:
    """Return the companion LoftQ file path for a given model and rank."""
    base, ext = os.path.splitext(output_model)
    return f"{base}.loftq_r{network_dim}{ext}"


def quantize_model(
    input_model: str,
    output_model: str,
    block_size: int,
    calc_device: str,
    loftq_init: bool = False,
    loftq_iters: int = 2,
    network_dim: int = 0,
):
    """Load model, quantize transformer blocks to NF4, save with metadata.

    Optionally pre-computes LoftQ initialization and saves as a companion file.
    """

    if not os.path.isfile(input_model):
        raise FileNotFoundError(f"Input model not found: {input_model}")

    if loftq_init and network_dim <= 0:
        raise ValueError("--loftq_init requires --network_dim > 0")

    # Read original metadata (contains model config)
    with safetensors.safe_open(input_model, framework="pt") as f:
        original_metadata = f.metadata() or {}

    if original_metadata.get("nf4_quantized") == "true":
        logger.error("Input model is already NF4 quantized — nothing to do.")
        return

    device = torch.device(calc_device)
    logger.info("Quantization device: %s", device)
    logger.info("Block size: %d", block_size)
    if loftq_init:
        logger.info("LoftQ enabled: rank=%d, iterations=%d", network_dim, loftq_iters)

    state_dict: dict[str, torch.Tensor] = {}
    loftq_data: dict[str, torch.Tensor] = {}
    quantized_count = 0
    skipped_count = 0
    passthrough_count = 0

    t0 = time.time()

    with MemoryEfficientSafeOpen(input_model) as f:
        keys = list(f.keys())
        logger.info("Total tensors in model: %d", len(keys))

        for key in tqdm(keys, desc="Quantizing", unit="tensor"):
            value = f.get_tensor(key)

            if not _is_quantizable(key, value, block_size):
                state_dict[key] = value
                if value.ndim == 2 and any(t in key for t in _NF4_TARGET_PATTERNS) and key.endswith(".weight"):
                    skipped_count += 1  # divisibility issue
                else:
                    passthrough_count += 1
                continue

            out_features, in_features = value.shape
            original_dtype = value.dtype

            # LoftQ: compute LoRA init from full-precision weight BEFORE quantizing
            if loftq_init:
                # Build lora_name matching the convention in lora.py
                # Keys have prefix like model.diffusion_model.transformer_blocks.0.attn.to_q.weight
                # After rename: transformer_blocks.0.attn.to_q.weight
                # lora_name: lora_unet_transformer_blocks_0_attn_to_q
                module_path = key.rsplit(".weight", 1)[0]
                # Strip the model.diffusion_model. prefix if present (same as LTXV_MODEL_COMFY_RENAMING_MAP)
                if module_path.startswith("model.diffusion_model."):
                    module_path = module_path[len("model.diffusion_model."):]
                lora_name = f"lora_unet_{module_path}".replace(".", "_")

                try:
                    lora_A, lora_B = loftq_initialize(
                        value,
                        quantize_fn=quantize_nf4_block,
                        dequantize_fn=dequantize_nf4_block,
                        lora_rank=network_dim,
                        block_size=block_size,
                        num_iterations=loftq_iters,
                        device=device,
                    )
                    loftq_data[f"{lora_name}.lora_A"] = lora_A.cpu().contiguous()
                    loftq_data[f"{lora_name}.lora_B"] = lora_B.cpu().contiguous()
                except Exception as e:
                    logger.warning("LoftQ init failed for %s: %s", key, e)

            # Quantize on calc_device
            value_on_device = value.to(device)
            packed, scale = quantize_nf4_block(value_on_device, block_size)

            state_dict[key] = packed.cpu()
            state_dict[key.replace(".weight", ".scale_weight")] = scale.to(dtype=original_dtype).cpu()
            state_dict[key.replace(".weight", ".nf4_shape")] = torch.tensor(
                [out_features, in_features], dtype=torch.int64
            )

            quantized_count += 1
            if device.type == "cuda" and quantized_count % 20 == 0:
                clean_memory_on_device(device)

    elapsed = time.time() - t0
    logger.info(
        "Quantization complete in %.1fs — quantized: %d, skipped: %d, passthrough: %d",
        elapsed,
        quantized_count,
        skipped_count,
        passthrough_count,
    )

    # Build output metadata: preserve original config + add NF4 markers
    output_metadata = dict(original_metadata)
    output_metadata["nf4_quantized"] = "true"
    output_metadata["nf4_block_size"] = str(block_size)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_model)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save quantized model
    logger.info("Saving quantized model to %s ...", output_model)
    from safetensors.torch import save_file

    save_file(state_dict, output_model, metadata=output_metadata)

    # Report sizes
    input_size = os.path.getsize(input_model) / (1024**3)
    output_size = os.path.getsize(output_model) / (1024**3)
    logger.info(
        "Size: %.2f GB -> %.2f GB (%.1f%% of original, saved %.2f GB)",
        input_size,
        output_size,
        output_size / input_size * 100,
        input_size - output_size,
    )

    # Save LoftQ companion file
    if loftq_init and loftq_data:
        loftq_file = loftq_path_for_model(output_model, network_dim)
        loftq_metadata = {
            "loftq_rank": str(network_dim),
            "loftq_iters": str(loftq_iters),
            "nf4_block_size": str(block_size),
            "num_modules": str(len(loftq_data) // 2),
        }
        save_file(loftq_data, loftq_file, metadata=loftq_metadata)
        loftq_size = os.path.getsize(loftq_file) / (1024**3)
        logger.info(
            "LoftQ data saved to %s (%.2f GB, %d modules, rank=%d)",
            loftq_file,
            loftq_size,
            len(loftq_data) // 2,
            network_dim,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Pre-quantize LTX-2 model weights to NF4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_model", required=True, help="Path to original .safetensors model")
    parser.add_argument("--output_model", required=True, help="Path for quantized output .safetensors")
    parser.add_argument(
        "--nf4_block_size",
        type=int,
        default=DEFAULT_NF4_BLOCK_SIZE,
        help=f"Block size for NF4 quantization (default: {DEFAULT_NF4_BLOCK_SIZE})",
    )
    parser.add_argument(
        "--calc_device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for quantization computation (default: cuda if available)",
    )
    parser.add_argument(
        "--loftq_init",
        action="store_true",
        help="Pre-compute LoftQ initialization (requires --network_dim)",
    )
    parser.add_argument(
        "--loftq_iters",
        type=int,
        default=2,
        help="Number of LoftQ alternating iterations (default: 2)",
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=0,
        help="LoRA rank for LoftQ initialization",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    quantize_model(
        args.input_model,
        args.output_model,
        args.nf4_block_size,
        args.calc_device,
        loftq_init=args.loftq_init,
        loftq_iters=args.loftq_iters,
        network_dim=args.network_dim,
    )
    logger.info("Done! Use the quantized model with --nf4_base to skip re-quantization.")


if __name__ == "__main__":
    main()
