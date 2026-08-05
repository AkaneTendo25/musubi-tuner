from __future__ import annotations

import argparse
import logging
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from musubi_tuner import cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.image_video_dataset import VideoDataset
from musubi_tuner.minimax_h3.crepa import DINO_DIMS
from musubi_tuner.minimax_h3.dataset import create_h3_dataset_group

logger = logging.getLogger(__name__)
_INPUT_SIZE = 518
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def dino_cache_path(latent_cache_path: str | Path) -> Path:
    path = Path(latent_cache_path)
    suffix = "_mmh3.safetensors"
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected MiniMax H3 latent-cache name: {path}")
    return path.with_name(path.name[: -len(suffix)] + "_mmh3_dino.safetensors")


def _load_model(name: str, repo: Path | None, hub_dir: Path | None) -> torch.nn.Module:
    if hub_dir is not None:
        hub_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_dir))
    if repo is not None:
        if not (repo / "hubconf.py").is_file():
            raise ValueError("--dino_repo must point to a DINOv2 checkout containing hubconf.py")
        return torch.hub.load(str(repo), name, source="local")
    return torch.hub.load("facebookresearch/dinov2", name)


@torch.inference_mode()
def extract_features(model: torch.nn.Module, frames: np.ndarray, device: torch.device, batch_size: int) -> torch.Tensor:
    images = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255)
    images = F.interpolate(images, size=(_INPUT_SIZE, _INPUT_SIZE), mode="bilinear", align_corners=False)
    mean = images.new_tensor(_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(_STD).view(1, 3, 1, 1)
    images = (images - mean) / std
    outputs = []
    for start in range(0, images.shape[0], batch_size):
        result = model.forward_features(images[start : start + batch_size].to(device))
        outputs.append(result["x_norm_patchtokens"].to(device="cpu", dtype=torch.float16))
    return torch.cat(outputs)


def create_parser() -> argparse.ArgumentParser:
    parser = cache_latents.setup_parser_common()
    parser.description = "Cache DINOv2 frame features for MiniMax H3 CREPA"
    parser.add_argument("--dino_model", choices=tuple(DINO_DIMS), default="dinov2_vitb14")
    parser.add_argument("--dino_batch_size", type=int, default=8)
    parser.add_argument("--dino_repo", type=Path)
    parser.add_argument("--torch_hub_dir", type=Path)
    parser.add_argument(
        "--atomic_cache_writes",
        action="store_true",
        help="replace each completed cache atomically so interrupted writes cannot leave a partial final file",
    )
    return parser


def _save_features(path: Path, features: torch.Tensor, metadata: dict[str, str], *, atomic: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not atomic:
        save_file({"h3_dino_features": features}, str(path), metadata=metadata)
        return
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        save_file({"h3_dino_features": features}, str(temporary), metadata=metadata)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    if args.dino_batch_size <= 0:
        raise ValueError("--dino_batch_size must be positive")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    user_config = config_utils.load_user_config(args.dataset_config)
    dataset_group, _ = create_h3_dataset_group(user_config, args)
    datasets = list(dataset_group.datasets)
    if user_config.get("validation_datasets"):
        validation_config = {
            "general": user_config.get("general", {}),
            "datasets": user_config["validation_datasets"],
        }
        validation_group, _ = create_h3_dataset_group(validation_config, args)
        datasets.extend(validation_group.datasets)
    model = _load_model(args.dino_model, args.dino_repo, args.torch_hub_dir).eval().to(device)
    model.requires_grad_(False)
    workers = args.num_workers if args.num_workers is not None else 1
    cached = skipped = 0
    for dataset in datasets:
        if not isinstance(dataset, VideoDataset):
            continue
        for _, batch in dataset.retrieve_latent_cache_batches(workers):
            for item in batch:
                output = dino_cache_path(item.latent_cache_path)
                if args.skip_existing and output.is_file():
                    skipped += 1
                    continue
                features = extract_features(model, item.content, device, args.dino_batch_size)
                metadata = {
                    "dino_model": args.dino_model,
                    "frames": str(features.shape[0]),
                    "patches": str(features.shape[1]),
                    "channels": str(features.shape[2]),
                }
                _save_features(output, features, metadata, atomic=args.atomic_cache_writes)
                cached += 1
    logger.info("DINO feature caching complete: %d written, %d skipped", cached, skipped)


if __name__ == "__main__":
    main()
