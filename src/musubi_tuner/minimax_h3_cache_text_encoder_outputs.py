from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.backend import create_backend
from musubi_tuner.minimax_h3.cache import normalize_batch_tensors, save_text_encoder_output_cache_minimax_h3
from musubi_tuner.minimax_h3.dataset import attach_h3_media, create_h3_dataset_group
from musubi_tuner.minimax_h3.load_options import H3LoadOptions, add_h3_load_arguments

logger = logging.getLogger(__name__)


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = "Cache MiniMax H3 conditioning with Musubi's dataset and cache pipeline"
    parser.add_argument("--model", type=Path, required=True, help="H3 model directory or checkpoint")
    parser.add_argument("--text_encoder_dtype", default="bfloat16")
    parser.add_argument(
        "--cache_guidance_empty",
        action="store_true",
        help="also cache H3's empty-text conditioning for the optional guidance-consistent training objective",
    )
    add_h3_load_arguments(parser)
    return parser


def create_parser() -> argparse.ArgumentParser:
    return setup_parser(cache_text_encoder_outputs.setup_parser_common())


def main(argv: Sequence[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    device_name = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    logger.info("Load dataset config from %s", args.dataset_config)
    user_config = config_utils.load_user_config(args.dataset_config)
    dataset_group, dataset_adapter = create_h3_dataset_group(user_config, args)
    datasets = dataset_group.datasets

    all_cache_files, all_cache_paths = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)
    load_options = H3LoadOptions.from_namespace(args, dtype=args.text_encoder_dtype)
    backend = create_backend(model=args.model, device=str(device), load_options=load_options)

    def encode(batch: list[ItemInfo]) -> None:
        attach_h3_media(batch, dataset_adapter)
        results = normalize_batch_tensors(
            backend.encode_conditioning(batch, include_empty=args.cache_guidance_empty), len(batch), "conditioning encoder"
        )
        for item, tensors in zip(batch, results):
            save_text_encoder_output_cache_minimax_h3(item, tensors)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files,
        all_cache_paths,
        encode,
        requires_content=bool(getattr(backend, "conditioning_requires_content", False)),
    )
    cache_text_encoder_outputs.post_process_cache_files(datasets, all_cache_files, all_cache_paths, args.keep_cache)


if __name__ == "__main__":
    main()
