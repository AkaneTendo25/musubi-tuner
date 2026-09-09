from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors import safe_open

from musubi_tuner import cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.backend import create_latent_encoder
from musubi_tuner.minimax_h3.cache import (
    H3_REFERENCE_KINDS_KEY,
    normalize_batch_tensors,
    reference_key_suffix,
    save_latent_cache_minimax_h3,
)
from musubi_tuner.minimax_h3.dataset import TARGET_AUDIO_FINGERPRINT_KEY, attach_h3_media, create_h3_dataset_group
from musubi_tuner.minimax_h3.image_training import (
    H3_ONE_FRAME_CACHE_FORMAT,
    H3_ONE_FRAME_CONTROL_INDICES_KEY,
    H3_ONE_FRAME_LATENT_FINGERPRINT_KEY,
    H3_ONE_FRAME_TARGET_INDEX_KEY,
    add_image_training_arguments,
    cache_matches_fingerprint,
)
from musubi_tuner.minimax_h3.references import (
    REFERENCE_FINGERPRINT_KEY,
    REFERENCE_IMAGE_SHORT_EDGE,
    REFERENCE_IMAGE_SIZE_MODES,
    REFERENCE_VIDEO_FPS,
    REFERENCE_VIDEO_MAX_PIXELS,
    REFERENCE_VIDEO_SHORT_EDGE,
    reference_assets,
)

logger = logging.getLogger(__name__)
H3_LOSS_MASK_POOLING_KEY = "loss_mask_pooling_mode"
H3_LOSS_MASK_POOLING_CODES = {"max": 0, "average": 1, "nearest": 2}


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = "Cache MiniMax H3 latents with Musubi's dataset and cache pipeline"
    parser.add_argument(
        "--task",
        choices=("t2va", "i2va", "fl2va", "l2va", "ref2va", "ref2va_omni"),
        default="t2va",
        help="conditioning task used to interpret one-frame image controls",
    )
    parser.add_argument(
        "--audio_vae",
        type=Path,
        help="H3 audio VAE checkpoint or Comfy model directory (required whenever a target or reference includes audio)",
    )
    parser.add_argument(
        "--h3_loss_mask_pooling",
        choices=("max", "average", "nearest"),
        default="max",
        help="reduce pixel masks to H3 latents with conservative max, smooth average, or nearest sampling",
    )
    parser.add_argument(
        "--reference_image_size_mode",
        choices=REFERENCE_IMAGE_SIZE_MODES,
        default="short_edge",
        help="Ref2VA image sizing: released short-edge preprocessing or target-bucket area matching",
    )
    parser.add_argument(
        "--reference_image_max_pixels",
        type=int,
        default=0,
        help="optional target-area reference pixel cap; 0 uses the target bucket area",
    )
    parser.add_argument(
        "--reference_image_short_edge",
        type=int,
        default=REFERENCE_IMAGE_SHORT_EDGE,
        help=(
            "scale every Ref2VA reference image so its short edge reaches this many pixels; lowering it shortens the "
            "reference rows the DiT attends to. A non-default value is recorded in the reference cache keys"
        ),
    )
    parser.add_argument(
        "--reference_video_short_edge",
        type=int,
        default=REFERENCE_VIDEO_SHORT_EDGE,
        help="Ref2VA reference-video short edge; lower values reduce reference rows and compute",
    )
    parser.add_argument(
        "--reference_video_max_pixels",
        type=int,
        default=REFERENCE_VIDEO_MAX_PIXELS,
        help="maximum pixels per Ref2VA reference-video frame after aspect-preserving resize",
    )
    parser.add_argument(
        "--reference_video_fps",
        type=float,
        default=REFERENCE_VIDEO_FPS,
        help=(
            "subsample every Ref2VA reference video to this many frames per source second so the whole clip "
            "conditions the model; 0 (default) keeps the released behaviour of truncating to the target length"
        ),
    )
    parser.set_defaults(vae_dtype="float32")
    add_image_training_arguments(parser)
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
        parser.error("--audio_vae is required for H3 audio targets or references")

    encoder = create_latent_encoder(
        video_vae=Path(args.vae) if args.vae is not None else None,
        audio_vae=args.audio_vae,
        device=str(device),
        dtype=args.vae_dtype or "float32",
        reference_image_short_edge=args.reference_image_short_edge,
        reference_image_size_mode=args.reference_image_size_mode,
        reference_image_max_pixels=args.reference_image_max_pixels,
        reference_video_short_edge=args.reference_video_short_edge,
        reference_video_max_pixels=args.reference_video_max_pixels,
        reference_video_fps=args.reference_video_fps,
        loss_mask_pooling=args.h3_loss_mask_pooling,
    )

    def encode(batch: list[ItemInfo]) -> None:
        attach_h3_media(batch, dataset_adapter)
        results = normalize_batch_tensors(encoder.encode_latents(batch), len(batch), "latent encoder")
        for item, tensors in zip(batch, results):
            if "video_loss_mask" in tensors:
                tensors[f"{H3_LOSS_MASK_POOLING_KEY}_int64"] = torch.tensor(
                    H3_LOSS_MASK_POOLING_CODES[args.h3_loss_mask_pooling], dtype=torch.long
                )
            save_latent_cache_minimax_h3(item, tensors)

    def existing_cache_valid(item: ItemInfo, path: str) -> bool:
        attach_h3_media((item,), dataset_adapter)
        if getattr(item, "h3_one_frame", False):
            fingerprint = item.h3_cache_metadata[H3_ONE_FRAME_LATENT_FINGERPRINT_KEY]
            if not cache_matches_fingerprint(path, fingerprint, H3_ONE_FRAME_LATENT_FINGERPRINT_KEY):
                return False
            try:
                with safe_open(path, framework="pt", device="cpu") as handle:
                    keys = set(handle.keys())
                    if (handle.metadata() or {}).get("h3_cache_format") != H3_ONE_FRAME_CACHE_FORMAT:
                        return False
                    target_key = f"{H3_ONE_FRAME_TARGET_INDEX_KEY}_int64"
                    if target_key not in keys or int(handle.get_tensor(target_key)) != item.h3_one_frame_target_index:
                        return False
                    control_key = f"{H3_ONE_FRAME_CONTROL_INDICES_KEY}_int64"
                    expected = tuple(item.h3_one_frame_control_indices)
                    if bool(expected) != (control_key in keys):
                        return False
                    if expected and tuple(int(value) for value in handle.get_tensor(control_key)) != expected:
                        return False
            except (OSError, RuntimeError, ValueError):
                return False
        if args.h3_image_mode != "none" and not cache_matches_fingerprint(path, item.h3_cache_metadata["sample_fingerprint"]):
            return False
        target_audio_fingerprint = getattr(item, "h3_cache_metadata", {}).get(TARGET_AUDIO_FINGERPRINT_KEY)
        if target_audio_fingerprint is not None and not cache_matches_fingerprint(
            path, target_audio_fingerprint, TARGET_AUDIO_FINGERPRINT_KEY
        ):
            return False
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                if "video_loss_mask" in keys:
                    key = f"{H3_LOSS_MASK_POOLING_KEY}_int64"
                    if key not in keys or int(handle.get_tensor(key)) != H3_LOSS_MASK_POOLING_CODES[args.h3_loss_mask_pooling]:
                        return False
        except (OSError, RuntimeError, ValueError):
            return False
        if not reference_assets(item):
            return True
        if not cache_matches_fingerprint(path, item.h3_cache_metadata[REFERENCE_FINGERPRINT_KEY], REFERENCE_FINGERPRINT_KEY):
            return False
        suffix = reference_key_suffix(
            args.reference_image_short_edge,
            args.reference_image_size_mode,
            args.reference_image_max_pixels,
            args.reference_video_short_edge,
            args.reference_video_max_pixels,
            args.reference_video_fps,
        )
        kinds_key = f"varlen_{H3_REFERENCE_KINDS_KEY}{suffix}_int64"
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                return kinds_key in set(handle.keys())
        except (OSError, RuntimeError, ValueError):
            return False

    cache_latents.encode_datasets(
        datasets,
        encode,
        args,
        existing_cache_valid=existing_cache_valid,
    )


if __name__ == "__main__":
    main()
