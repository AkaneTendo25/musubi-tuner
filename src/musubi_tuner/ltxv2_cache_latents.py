#!/usr/bin/env python3
"""
Cache video latents for LTXV2 training.

Uses the standard musubi-tuner dataset config so cached files match the trainer.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from typing import List, Sequence, cast

import logging
import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_LTXV2,
    BaseDataset,
    ItemInfo,
    save_latent_cache_ltxv2,
)
from musubi_tuner.utils.model_utils import str_to_dtype


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _amp_context(device: torch.device, dtype: torch.dtype):
    if device.type in {"cuda", "xpu"}:
        try:
            from torch.amp import autocast as torch_autocast  # type: ignore[attr-defined]

            return torch_autocast(device_type=device.type, dtype=dtype)
        except (ImportError, AttributeError):
            from torch.cuda.amp import autocast as torch_autocast

            return torch_autocast(dtype=dtype)
    return nullcontext()


def _load_datasets(args: argparse.Namespace) -> Sequence[BaseDataset]:
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info("Load dataset config from %s", args.dataset_config)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_LTXV2)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    return cast(Sequence[BaseDataset], dataset_group.datasets)


def encode_and_save_batch(vae, batch: List[ItemInfo]) -> None:
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if contents.ndim == 4:
        contents = contents.unsqueeze(1)

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()
    vae_param = next(vae.parameters())
    device = vae_param.device
    vae_dtype = vae_param.dtype
    contents = contents.to(device=device, dtype=vae_dtype)
    contents = contents / 127.5 - 1.0

    frames = contents.shape[2]
    remainder = (frames - 1) % 8
    if remainder != 0:
        pad = 8 - remainder
        last = contents[:, :, -1:, :, :].expand(-1, -1, pad, -1, -1)
        contents = torch.cat([contents, last], dim=2)

    height, width = contents.shape[-2:]
    if height < 8 or width < 8:
        item = batch[0]
        raise ValueError(f"Image or video size too small: {item.item_key} (and others), size: {item.original_size}")

    with _amp_context(device, vae_dtype), torch.no_grad():
        latents = vae(contents)
        latents = latents.to(device=device, dtype=vae_dtype)

    for idx, item in enumerate(batch):
        save_latent_cache_ltxv2(item, latents[idx])


def main() -> None:
    parser = cache_latents.setup_parser_common()
    parser = ltxv2_setup_parser(parser)
    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = _load_datasets(args)

    if args.debug_mode is not None:
        cache_latents.show_datasets(list(datasets), args.debug_mode, args.console_width, args.console_back, args.console_num_images)
        return

    assert args.vae is not None, "VAE checkpoint is required"

    vae_dtype = torch.bfloat16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.video_vae import VideoEncoderConfigurator, VAE_ENCODER_COMFY_KEYS_FILTER

    vae = SingleGPUModelBuilder(
        model_path=str(args.vae),
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=vae_dtype)
    vae.eval()

    def encode_fn(batch: List[ItemInfo]) -> None:
        encode_and_save_batch(vae, batch)

    cache_latents.encode_datasets(list(datasets), encode_fn, args)


def ltxv2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--ltxv2_audio_video",
        action="store_true",
        help="Enable audio-video caching. Video latents are the same; audio latents are cached separately.",
    )
    return parser


if __name__ == "__main__":
    main()
