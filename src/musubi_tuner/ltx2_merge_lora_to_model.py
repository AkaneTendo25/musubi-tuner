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
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from musubi_tuner.ltx2_lora_utils import (
    import_lora_network_module,
    infer_lora_network_module,
    load_lora_metadata,
)
from musubi_tuner.utils.safetensors_utils import LazyTensorForSave, MemoryEfficientSafeOpen, mem_eff_save_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_COMFY_TRANSFORMER_PREFIX = "model.diffusion_model."


def _normalize_checkpoint_key(key: str) -> str:
    if key.startswith(_COMFY_TRANSFORMER_PREFIX):
        return key[len(_COMFY_TRANSFORMER_PREFIX) :]
    return key


def _clone_for_save_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor | LazyTensorForSave:
    if tensor.dtype == dtype:
        return tensor

    return LazyTensorForSave(
        shape=tuple(tensor.shape),
        dtype=dtype,
        materialize_fn=lambda: tensor.to(dtype=dtype),
    )


def _lazy_original_tensor(
    base_model_path: str,
    key: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> LazyTensorForSave:
    def materialize() -> torch.Tensor:
        with MemoryEfficientSafeOpen(base_model_path) as reader:
            return reader.get_tensor(key)

    return LazyTensorForSave(shape=shape, dtype=dtype, materialize_fn=materialize)


def _build_full_model_state_dict(
    merged_transformer_sd: dict[str, torch.Tensor],
    base_model_path: str,
) -> tuple[dict[str, torch.Tensor | LazyTensorForSave], int, int]:
    """Replace transformer keys in the original checkpoint and stream-copy the rest."""

    tensors: dict[str, torch.Tensor | LazyTensorForSave] = {}
    replaced_normalized_keys: set[str] = set()
    original_key_count = 0

    with MemoryEfficientSafeOpen(base_model_path) as reader:
        for original_key in reader.keys():
            original_key_count += 1
            normalized_key = _normalize_checkpoint_key(original_key)
            original_tensor_meta = reader.header[original_key]
            original_dtype = reader._get_torch_dtype(original_tensor_meta["dtype"])  # noqa: SLF001
            original_shape = tuple(int(dim) for dim in original_tensor_meta["shape"])
            if original_dtype is None:
                raise ValueError(f"Unsupported safetensors dtype for key {original_key}: {original_tensor_meta['dtype']}")

            if normalized_key in merged_transformer_sd:
                tensors[original_key] = _clone_for_save_dtype(merged_transformer_sd[normalized_key], original_dtype)
                replaced_normalized_keys.add(normalized_key)
            else:
                tensors[original_key] = _lazy_original_tensor(base_model_path, original_key, original_shape, original_dtype)

    original_keys = set(tensors.keys())
    has_comfy_layout = any(key.startswith(_COMFY_TRANSFORMER_PREFIX) for key in original_keys)
    for key, tensor in merged_transformer_sd.items():
        if key in replaced_normalized_keys:
            continue
        output_key = f"{_COMFY_TRANSFORMER_PREFIX}{key}" if has_comfy_layout else key
        if output_key in original_keys:
            continue
        tensors[output_key] = tensor

    return tensors, len(replaced_normalized_keys), original_key_count


def _validate_output_path(base_model_path: str, output_path: str) -> None:
    if os.path.abspath(base_model_path) == os.path.abspath(output_path):
        raise ValueError("--save_merged_model must not overwrite --dit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LTX-2 LoRA weights into a base model checkpoint")
    parser.add_argument("--dit", type=str, required=True, help="LTX-2 base model checkpoint (.safetensors)")
    parser.add_argument("--lora_weight", type=str, nargs="+", required=True, help="LoRA weight path(s)")
    parser.add_argument(
        "--lora_multiplier",
        type=float,
        nargs="*",
        default=None,
        help="Per-LoRA multipliers aligned with --lora_weight. If omitted, all are 1.0.",
    )
    parser.add_argument("--save_merged_model", type=str, required=True, help="Output merged model path (.safetensors)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for merge computation (default: cuda if available)",
    )
    parser.add_argument("--audio_video", action="store_true", help="Load as audio-video model")
    parser.add_argument("--audio_only", action="store_true", help="Load as audio-only model")
    parser.add_argument(
        "--save_full_model",
        action="store_true",
        help=(
            "Save a full checkpoint by replacing transformer weights in --dit and copying all other "
            "tensors from the original file. Preserves ComfyUI-style key layout when --dit uses it."
        ),
    )
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
        raise ValueError(f"--lora_multiplier count ({len(multipliers)}) must be 1 or match --lora_weight count ({len(lora_paths)})")

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
        lora_sd = load_file(lora_path)
        metadata = load_lora_metadata(lora_path)
        network_module_name = infer_lora_network_module(metadata, lora_sd)
        network_module = import_lora_network_module(network_module_name)
        net = network_module.create_arch_network_from_weights(
            multiplier=mult,
            weights_sd=lora_sd,
            unet=transformer,
            for_inference=True,
        )
        logger.info("Resolved LoRA network module: %s", network_module_name)
        net.merge_to(None, transformer, lora_sd, device=device, non_blocking=True)
        del lora_sd, net

    # Save merged transformer with original metadata preserved
    logger.info("Saving merged model to %s", args.save_merged_model)
    merged_sd = transformer.model.state_dict()

    metadata = dict(original_metadata)
    metadata["merged_loras"] = json.dumps(lora_paths)
    metadata["merged_multipliers"] = json.dumps(multipliers)
    metadata["merged_full_model"] = str(bool(args.save_full_model))

    if args.save_full_model:
        _validate_output_path(args.dit, args.save_merged_model)
        merged_sd, replaced_keys, original_key_count = _build_full_model_state_dict(merged_sd, args.dit)
        logger.info(
            "Full model save: replaced %d transformer keys and preserved %d original checkpoint keys",
            replaced_keys,
            original_key_count - replaced_keys,
        )

    mem_eff_save_file(merged_sd, args.save_merged_model, metadata=metadata)
    logger.info("Done: %s (%d keys)", args.save_merged_model, len(merged_sd))


if __name__ == "__main__":
    main()
