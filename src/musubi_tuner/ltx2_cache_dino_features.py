#!/usr/bin/env python3
"""Cache DINOv2 per-frame features for CREPA dino mode.

Extracts patch tokens from a frozen DINOv2 encoder for each video frame and
saves them as safetensors files alongside the latent caches. At training
time these features are loaded from disk (zero VRAM from DINOv2).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Sequence, cast

import numpy as np
import torch
from safetensors.torch import save_file

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_LTX2,
    BaseDataset,
    VideoDataset,
)
from musubi_tuner.utils.safetensors_utils import save_file_atomic

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# DINOv2 model name → token dimension
DINO_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

# DINOv2 expected input size
DINO_INPUT_SIZE = 518

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_dino_model(model_name: str, *, dino_repo_path: str | None = None, torch_hub_dir: str | None = None) -> torch.nn.Module:
    """Load DINOv2 from a local torch.hub repo clone, or from torch hub."""

    if torch_hub_dir:
        hub_dir = Path(torch_hub_dir).expanduser()
        hub_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_dir))
        logger.info("Using torch hub directory: %s", hub_dir)

    if dino_repo_path:
        repo_path = Path(dino_repo_path).expanduser()
        if not repo_path.is_dir():
            raise ValueError(f"DINOv2 repo path is not a directory: {repo_path}")
        if not (repo_path / "hubconf.py").is_file():
            raise ValueError(f"DINOv2 repo path must contain hubconf.py: {repo_path}")
        logger.info("Loading DINOv2 from local repo: %s", repo_path)
        return torch.hub.load(str(repo_path), model_name, source="local")

    logger.info("Loading DINOv2 from torch hub: facebookresearch/dinov2")
    return torch.hub.load("facebookresearch/dinov2", model_name)


def _load_datasets(args: argparse.Namespace) -> Sequence[BaseDataset]:
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info("Load dataset config from %s", args.dataset_config)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_LTX2)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = list(dataset_group.datasets)

    if user_config.get("validation_datasets"):
        logger.info("Load validation datasets from dataset config")
        validation_user_config = {
            "general": user_config.get("general", {}),
            "datasets": user_config.get("validation_datasets", []),
        }
        validation_blueprint = blueprint_generator.generate(
            validation_user_config, args, architecture=ARCHITECTURE_LTX2
        )
        validation_dataset_group = config_utils.generate_dataset_group_by_blueprint(
            validation_blueprint.dataset_group
        )
        datasets.extend(validation_dataset_group.datasets)

    return cast(Sequence[BaseDataset], datasets)


def dino_cache_path_from_latent_cache_path(latent_cache_path: str) -> str:
    """Derive DINOv2 feature cache path from a latent cache path.

    ``*_ltx2.safetensors`` → ``*_ltx2_dino.safetensors``
    """
    suffix = "_ltx2.safetensors"
    if latent_cache_path.endswith(suffix):
        return latent_cache_path[: -len(suffix)] + "_ltx2_dino.safetensors"
    stem, _ext = os.path.splitext(latent_cache_path)
    return stem + "_ltx2_dino.safetensors"


def _preprocess_frames(
    frames: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Resize frames to 518x518 and normalise with ImageNet stats.

    Args:
        frames: uint8 numpy array [T, H, W, C].

    Returns:
        Tensor [T, 3, 518, 518] on *device* with *dtype*.
    """
    # [T, H, W, C] → [T, C, H, W] float32 in [0, 1]
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    # Resize to DINO_INPUT_SIZE x DINO_INPUT_SIZE
    t = torch.nn.functional.interpolate(t, size=(DINO_INPUT_SIZE, DINO_INPUT_SIZE), mode="bilinear", align_corners=False)
    # Normalise per-channel
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device).view(1, 3, 1, 1)
    t = (t - mean) / std
    return t.to(device=device, dtype=dtype)


@torch.no_grad()
def extract_dino_features(
    model: torch.nn.Module,
    frames: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 16,
) -> torch.Tensor:
    """Run DINOv2 on *frames* and return patch tokens.

    Args:
        model: Frozen DINOv2 model.
        frames: uint8 numpy [T, H, W, C].
        batch_size: Sub-batch size for DINOv2 forward passes.

    Returns:
        float16 tensor [T, N_patches, D_dino] on CPU.
        For vitb14 @ 518px: N_patches = (518//14)^2 = 1369.
    """
    preprocessed = _preprocess_frames(frames, device, dtype)  # [T, 3, 518, 518]
    T = preprocessed.shape[0]
    patch_tokens = []
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        batch = preprocessed[start:end]
        out = model.forward_features(batch)
        patches = out["x_norm_patchtokens"]  # [B, N_patches, D]
        patch_tokens.append(patches.cpu().to(torch.float16))
    return torch.cat(patch_tokens, dim=0)  # [T, N_patches, D_dino]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache DINOv2 per-frame features for CREPA dino mode")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset config .toml file")
    parser.add_argument(
        "--dino_model",
        type=str,
        default="dinov2_vitb14",
        choices=list(DINO_DIMS.keys()),
        help="DINOv2 model variant (default: dinov2_vitb14)",
    )
    parser.add_argument("--dino_batch_size", type=int, default=16, help="Frames per DINOv2 forward pass (default: 16)")
    parser.add_argument(
        "--dino_repo_path",
        type=str,
        default=None,
        help="Local facebookresearch/dinov2 repo clone containing hubconf.py. Uses torch.hub source='local'.",
    )
    parser.add_argument(
        "--torch_hub_dir",
        type=str,
        default=None,
        help="Torch hub cache directory. Useful for offline/pre-populated DINOv2 cache environments.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device (default: cuda if available)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip items that already have dino caches")
    parser.add_argument(
        "--atomic_cache_writes",
        action="store_true",
        help="Write cache files to a temporary sibling file and atomically replace the final path after a successful save.",
    )
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers for dataset loading")
    # These are required by _load_datasets / BlueprintGenerator but unused here
    parser.add_argument("--vae", type=str, default=None)
    parser.add_argument("--vae_dtype", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--skip_existing_latents", action="store_true", default=False)
    parser.add_argument("--keep_cache", action="store_true", default=False)
    parser.add_argument("--debug_mode", type=str, default=None)
    parser.add_argument("--console_width", type=int, default=80)
    parser.add_argument("--console_back", type=str, default=None)
    parser.add_argument("--console_num_images", type=int, default=None)
    parser.add_argument("--disable_cudnn_backend", action="store_true", default=False)

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_dim = DINO_DIMS[args.dino_model]

    logger.info("Loading DINOv2 model: %s (dim=%d)", args.dino_model, dino_dim)
    dino = load_dino_model(
        args.dino_model,
        dino_repo_path=args.dino_repo_path,
        torch_hub_dir=args.torch_hub_dir,
    )
    dino = dino.to(device=device, dtype=torch.float32)
    dino.eval()
    for p in dino.parameters():
        p.requires_grad_(False)

    datasets = _load_datasets(args)
    num_workers = args.num_workers if args.num_workers is not None else max(1, (os.cpu_count() or 2) - 1)

    total_cached = 0
    total_skipped = 0

    for ds in datasets:
        if not isinstance(ds, VideoDataset):
            logger.info("Skipping non-video dataset: %s", type(ds).__name__)
            continue

        logger.info("Processing dataset: %s", getattr(ds, "video_directory", "unknown"))
        for _bucket_key, batch in ds.retrieve_latent_cache_batches(num_workers):
            for item_info in batch:
                cache_path = item_info.latent_cache_path
                if cache_path is None:
                    continue

                dino_path = dino_cache_path_from_latent_cache_path(cache_path)

                if args.skip_existing and os.path.exists(dino_path):
                    total_skipped += 1
                    continue

                # item_info.content is numpy [T, H, W, C] uint8 (set by retrieve_latent_cache_batches)
                frames = item_info.content
                if frames is None:
                    logger.warning("No content for %s, skipping", item_info.item_key)
                    continue

                try:
                    patch_tokens = extract_dino_features(
                        dino, frames, device, torch.float32, args.dino_batch_size
                    )  # [T, N_patches, D_dino] float16
                except Exception as e:
                    logger.warning("Failed to extract DINOv2 features for %s: %s", item_info.item_key, e)
                    continue

                os.makedirs(os.path.dirname(dino_path), exist_ok=True)
                metadata = {
                    "dino_model": args.dino_model,
                    "dino_dim": str(dino_dim),
                    "num_frames": str(patch_tokens.shape[0]),
                    "num_patches": str(patch_tokens.shape[1]),
                }
                if args.atomic_cache_writes:
                    save_file_atomic({"dino_features": patch_tokens}, dino_path, metadata=metadata)
                else:
                    save_file({"dino_features": patch_tokens}, dino_path, metadata=metadata)
                total_cached += 1

    logger.info("Done. Cached: %d, Skipped: %d", total_cached, total_skipped)


if __name__ == "__main__":
    main()
