from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.backend import create_backend
from musubi_tuner.minimax_h3.cache import normalize_batch_tensors, save_latent_cache_minimax_h3
from musubi_tuner.minimax_h3.dataset import attach_h3_media, create_h3_dataset_group
from musubi_tuner.minimax_h3.load_options import H3LoadOptions, add_h3_load_arguments

logger = logging.getLogger(__name__)


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = "Cache MiniMax H3 latents with Musubi's dataset and cache pipeline"
    parser.add_argument("--model", type=Path, help="H3 model directory or checkpoint")
    add_h3_load_arguments(parser, include_text_encoder=False)
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

    if args.model is None:
        parser.error("--model is required unless --debug_mode is used")

    load_options = H3LoadOptions.from_namespace(args, dtype=args.vae_dtype or "float32")
    backend = create_backend(model=args.model, device=str(device), load_options=load_options)

    def encode(batch: list[ItemInfo]) -> None:
        attach_h3_media(batch, dataset_adapter)
        results = normalize_batch_tensors(backend.encode_latents(batch), len(batch), "latent encoder")
        for item, tensors in zip(batch, results):
            save_latent_cache_minimax_h3(item, tensors)

    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
