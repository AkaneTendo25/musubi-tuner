from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import torch

from musubi_tuner import cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.assets import default_text_encoder_assets
from musubi_tuner.minimax_h3.backend import create_conditioning_encoder
from musubi_tuner.minimax_h3.cache import normalize_batch_tensors, save_text_encoder_output_cache_minimax_h3
from musubi_tuner.minimax_h3.dataset import attach_h3_media, create_h3_dataset_group

logger = logging.getLogger(__name__)


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = "Cache MiniMax H3 conditioning with Musubi's dataset and cache pipeline"
    parser.add_argument("--text_encoder", type=Path, required=True, help="H3 Qwen3-VL checkpoint or Comfy model directory")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=default_text_encoder_assets(),
        help="H3 tokenizer/processor directory; defaults to the metadata bundled with Musubi",
    )
    parser.add_argument(
        "--task",
        choices=("t2va", "i2va", "fl2va", "ref2va", "ref2va_omni"),
        default="t2va",
        help=(
            "conditioning presentation: text only, first-frame I2V, first+last-frame FL2VA, strict ordered references, "
            "or experimental zero-or-more Ref2VA references"
        ),
    )
    parser.add_argument("--text_encoder_dtype", default="bfloat16")
    parser.add_argument(
        "--text_encoder_quantization",
        choices=("none", "int8", "nf4", "nvfp4_awq"),
        default="none",
        help=(
            "text-encoder weight mode: INT8/NF4 quantize the BF16 checkpoint while loading; "
            "NVFP4_AWQ loads the matching Comfy-Org checkpoint directly"
        ),
    )
    parser.add_argument(
        "--cache_guidance_empty",
        action="store_true",
        help="also cache H3's empty-text conditioning for the optional guidance-consistent training objective",
    )
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
    encoder = create_conditioning_encoder(
        text_encoder=args.text_encoder,
        tokenizer=args.tokenizer,
        task=args.task,
        device=str(device),
        dtype=args.text_encoder_dtype,
        quantization=args.text_encoder_quantization,
    )

    def encode(batch: list[ItemInfo]) -> None:
        attach_h3_media(batch, dataset_adapter)
        results = normalize_batch_tensors(
            encoder.encode_conditioning(batch, include_empty=args.cache_guidance_empty), len(batch), "conditioning encoder"
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
        requires_content=encoder.conditioning_requires_content,
    )
    cache_text_encoder_outputs.post_process_cache_files(datasets, all_cache_files, all_cache_paths, args.keep_cache)


if __name__ == "__main__":
    main()
