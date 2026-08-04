from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import torch

from musubi_tuner import cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.backend import create_latent_encoder
from musubi_tuner.minimax_h3.cache import normalize_batch_tensors, save_latent_cache_minimax_h3
from musubi_tuner.minimax_h3.dataset import attach_h3_media, create_h3_dataset_group

logger = logging.getLogger(__name__)


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = "Cache MiniMax H3 latents with Musubi's dataset and cache pipeline"
    parser.add_argument(
        "--audio_vae",
        type=Path,
        help="H3 audio VAE checkpoint or Comfy model directory (required for videos, omitted for images)",
    )
    parser.set_defaults(vae_dtype="float32")
    return parser


def create_parser() -> argparse.ArgumentParser:
    return setup_parser(cache_latents.setup_parser_common())


def main(argv: Sequence[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    device_name = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    logger.info("Load dataset config from %s", args.dataset_config)
    user_config = config_utils.load_user_config(args.dataset_config)
    dataset_group, dataset_adapter = create_h3_dataset_group(user_config, args)
    datasets = dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images)
        return

    if dataset_adapter.requires_video and args.vae is None:
        parser.error("--vae is required for H3 visual targets or references")
    if dataset_adapter.requires_audio and args.audio_vae is None:
        parser.error("--audio_vae is required for H3 video datasets; omit it for image-only datasets")

    encoder = create_latent_encoder(
        video_vae=Path(args.vae) if args.vae is not None else None,
        audio_vae=args.audio_vae,
        device=str(device),
        dtype=args.vae_dtype or "float32",
    )

    def encode(batch: list[ItemInfo]) -> None:
        attach_h3_media(batch, dataset_adapter)
        results = normalize_batch_tensors(encoder.encode_latents(batch), len(batch), "latent encoder")
        for item, tensors in zip(batch, results):
            save_latent_cache_minimax_h3(item, tensors)

    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
