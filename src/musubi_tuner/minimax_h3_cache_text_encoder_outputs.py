from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors import safe_open

from musubi_tuner import cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.assets import default_text_encoder_assets
from musubi_tuner.minimax_h3.backend import create_conditioning_encoder
from musubi_tuner.minimax_h3.cache import (
    H3_CONDITIONING_TASK_IDS,
    H3_CONDITIONING_TASK_KEY,
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_KEYFRAME_VISUALS_KEY,
    H3_MAX_CAPTION_TOKENS_KEY,
    H3_QWEN_CONTROL_VISUALS_KEY,
    H3_REFERENCE_IMAGE_MAX_PIXELS_KEY,
    H3_REFERENCE_IMAGE_SHORT_EDGE_KEY,
    H3_REFERENCE_IMAGE_SIZE_MODE_KEY,
    H3_REFERENCE_VIDEO_FPS_KEY,
    H3_REFERENCE_VIDEO_MAX_PIXELS_KEY,
    H3_REFERENCE_VIDEO_SHORT_EDGE_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
    H3_TEXT_VISUAL_MAX_PIXELS_KEY,
    QWEN_CONTROL_FINGERPRINT_KEY,
    format_keyframe_visuals,
    logical_cache_key,
    normalize_batch_tensors,
    parse_keyframe_visuals,
    qwen_control_assets,
    qwen_control_dropout_key,
    save_text_encoder_output_cache_minimax_h3,
)
from musubi_tuner.minimax_h3.dataset import attach_h3_media, create_h3_dataset_group
from musubi_tuner.minimax_h3.image_training import add_image_training_arguments, cache_matches_fingerprint
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
        "--task",
        choices=("t2va", "i2va", "fl2va", "l2va", "ref2va", "ref2va_omni"),
        default="t2va",
        help=(
            "conditioning presentation: text only, first-frame I2V, first+last-frame FL2VA, last-frame L2V, "
            "strict ordered references, or experimental zero-or-more Ref2VA references"
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
        "--h3_max_caption_tokens",
        type=int,
        default=0,
        help="optionally truncate only caption text to this many tokens; 0 keeps the full caption",
    )
    parser.add_argument(
        "--h3_text_encoder_blocks_to_stream",
        type=int,
        default=0,
        help="stream this many of the 50 frozen Qwen3-VL layers from CPU during encoding (CUDA only)",
    )
    parser.add_argument(
        "--h3_nvfp4_scaled_mm",
        "--nvfp4_scaled_mm",
        action="store_true",
        help="use Blackwell W4A4 scaled_mm for a native NVFP4/AWQ text encoder (PyTorch 2.10+)",
    )
    parser.add_argument(
        "--reference_image_short_edge",
        type=int,
        default=REFERENCE_IMAGE_SHORT_EDGE,
        help=(
            "scale every Ref2VA reference image so its short edge reaches this many pixels before it is presented to "
            "the understanding encoder; keep it equal to the value given to latent caching"
        ),
    )
    parser.add_argument(
        "--reference_video_short_edge",
        type=int,
        default=REFERENCE_VIDEO_SHORT_EDGE,
        help="Ref2VA reference-video short edge; keep it equal to latent caching",
    )
    parser.add_argument(
        "--reference_video_max_pixels",
        type=int,
        default=REFERENCE_VIDEO_MAX_PIXELS,
        help="maximum pixels per Ref2VA reference-video frame; keep it equal to latent caching",
    )
    parser.add_argument(
        "--reference_video_fps",
        type=float,
        default=REFERENCE_VIDEO_FPS,
        help=(
            "subsample every Ref2VA reference video to this many frames per source second; 0 (default) keeps the "
            "released truncation. Keep it equal to latent caching and training"
        ),
    )
    parser.add_argument(
        "--h3_keyframe_visuals",
        type=str,
        default="",
        help=(
            "EXPERIMENTAL: comma-separated target-video frames -- 'first', 'last', or a frame index -- presented to "
            "the Qwen3-VL conditioner as picture spans, so a --task t2va cache shows the conditioner the frames "
            "--h3_keyframe_anchors will pin at training time. --task t2va only"
        ),
    )
    parser.add_argument(
        "--h3_qwen_control_dropout",
        action="store_true",
        help=(
            "EXPERIMENTAL: also cache the control-free presentation of every item that carries qwen_control_* assets, "
            "so training can drop the controls per step with --h3_qwen_control_dropout_rate"
        ),
    )
    parser.add_argument(
        "--cache_guidance_empty",
        action="store_true",
        help="also cache H3's empty-text conditioning for the optional guidance-consistent training objective",
    )
    add_image_training_arguments(parser, text_visual=True)
    return parser


def create_parser() -> argparse.ArgumentParser:
    return setup_parser(cache_text_encoder_outputs.setup_parser_common())


def main(argv: Sequence[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.h3_image_mode != "none" and args.task != "fl2va":
        parser.error("--h3_image_mode requires --task fl2va")
    if args.h3_text_visual_max_pixels < 0:
        parser.error("--h3_text_visual_max_pixels must be non-negative")
    if args.h3_max_caption_tokens < 0:
        parser.error("--h3_max_caption_tokens must be non-negative")
    try:
        keyframe_visuals = parse_keyframe_visuals(args.h3_keyframe_visuals)
    except ValueError as error:
        parser.error(str(error))
    if keyframe_visuals and args.task != "t2va":
        parser.error("--h3_keyframe_visuals requires --task t2va")
    if keyframe_visuals:
        logger.info("Presenting target frames %s to the conditioner as keyframe visuals", format_keyframe_visuals(keyframe_visuals))
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
        blocks_to_stream=args.h3_text_encoder_blocks_to_stream,
        nvfp4_scaled_mm=args.h3_nvfp4_scaled_mm,
        reference_image_short_edge=args.reference_image_short_edge,
        max_caption_tokens=args.h3_max_caption_tokens,
        reference_image_size_mode=args.reference_image_size_mode,
        reference_image_max_pixels=args.reference_image_max_pixels,
        reference_video_short_edge=args.reference_video_short_edge,
        reference_video_max_pixels=args.reference_video_max_pixels,
        reference_video_fps=args.reference_video_fps,
        text_visual_max_pixels=args.h3_text_visual_max_pixels,
        keyframe_visuals=keyframe_visuals,
    )

    def encode(batch: list[ItemInfo]) -> None:
        attach_h3_media(batch, dataset_adapter)
        results = normalize_batch_tensors(
            encoder.encode_conditioning(
                batch,
                include_empty=args.cache_guidance_empty,
                include_qwen_control_dropout=args.h3_qwen_control_dropout,
            ),
            len(batch),
            "conditioning encoder",
        )
        for item, tensors in zip(batch, results):
            save_text_encoder_output_cache_minimax_h3(item, tensors)

    def existing_cache_valid(item: ItemInfo, path: str) -> bool:
        attach_h3_media((item,), dataset_adapter)
        if args.h3_image_mode != "none" and not cache_matches_fingerprint(path, item.h3_cache_metadata["sample_fingerprint"]):
            return False
        if reference_assets(item) and not cache_matches_fingerprint(
            path, item.h3_cache_metadata[REFERENCE_FINGERPRINT_KEY], REFERENCE_FINGERPRINT_KEY
        ):
            return False
        # Qwen control visuals live only in the text presentation, so their file
        # identity is a parallel fingerprint: editing or swapping a control file
        # rebuilds this cache and leaves the latent cache untouched.
        qwen_controls = qwen_control_assets(item)
        if qwen_controls and not cache_matches_fingerprint(
            path, item.h3_cache_metadata[QWEN_CONTROL_FINGERPRINT_KEY], QWEN_CONTROL_FINGERPRINT_KEY
        ):
            return False
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                logical_keys = {logical_cache_key(key) for key in keys}
                if bool(qwen_controls) != (H3_QWEN_CONTROL_VISUALS_KEY in keys):
                    return False
                if H3_CONDITIONING_TASK_KEY not in keys:
                    return False
                if int(handle.get_tensor(H3_CONDITIONING_TASK_KEY)) != H3_CONDITIONING_TASK_IDS[args.task]:
                    return False
                if args.task in {"ref2va", "ref2va_omni"}:
                    if H3_REFERENCE_VIDEO_SHORT_EDGE_KEY not in keys or H3_REFERENCE_VIDEO_MAX_PIXELS_KEY not in keys:
                        return False
                    if int(handle.get_tensor(H3_REFERENCE_VIDEO_SHORT_EDGE_KEY)) != args.reference_video_short_edge:
                        return False
                    if int(handle.get_tensor(H3_REFERENCE_VIDEO_MAX_PIXELS_KEY)) != args.reference_video_max_pixels:
                        return False
                    # A cache written before temporal subsampling existed carries no
                    # fps identity and therefore describes the truncation path only.
                    if H3_REFERENCE_VIDEO_FPS_KEY not in keys:
                        if args.reference_video_fps != REFERENCE_VIDEO_FPS:
                            return False
                    elif float(handle.get_tensor(H3_REFERENCE_VIDEO_FPS_KEY)) != float(args.reference_video_fps):
                        return False
                    if (
                        H3_REFERENCE_IMAGE_SHORT_EDGE_KEY not in keys
                        or H3_REFERENCE_IMAGE_SIZE_MODE_KEY not in keys
                        or H3_REFERENCE_IMAGE_MAX_PIXELS_KEY not in keys
                    ):
                        return False
                    if int(handle.get_tensor(H3_REFERENCE_IMAGE_SHORT_EDGE_KEY)) != args.reference_image_short_edge:
                        return False
                    expected_size_mode = 0 if args.reference_image_size_mode == "short_edge" else 1
                    if int(handle.get_tensor(H3_REFERENCE_IMAGE_SIZE_MODE_KEY)) != expected_size_mode:
                        return False
                    if int(handle.get_tensor(H3_REFERENCE_IMAGE_MAX_PIXELS_KEY)) != args.reference_image_max_pixels:
                        return False
                if args.cache_guidance_empty and not {H3_EMPTY_TEXT_HIDDEN_KEY, H3_EMPTY_TEXT_TOKEN_TAGS_KEY} <= logical_keys:
                    return False
                # The control-free twin is part of the cache identity in one
                # direction only: a run that asks for dropout needs it, while a
                # cache that carries it serves a run that does not, because the
                # extra keys are simply never read.
                if args.h3_qwen_control_dropout and qwen_controls:
                    required = {qwen_control_dropout_key(H3_TEXT_HIDDEN_KEY), qwen_control_dropout_key(H3_TEXT_TOKEN_TAGS_KEY)}
                    if args.cache_guidance_empty:
                        required |= {
                            qwen_control_dropout_key(H3_EMPTY_TEXT_HIDDEN_KEY),
                            qwen_control_dropout_key(H3_EMPTY_TEXT_TOKEN_TAGS_KEY),
                        }
                    if not required <= logical_keys:
                        return False
                # The presented frame list is cache identity: changing it changes
                # what the conditioner saw, and its absence means the cache was
                # written text-only.
                cached_visuals = [handle.get_tensor(key) for key in keys if logical_cache_key(key) == H3_KEYFRAME_VISUALS_KEY]
                if len(cached_visuals) > 1:
                    return False
                if bool(keyframe_visuals) != bool(cached_visuals):
                    return False
                if cached_visuals and tuple(int(value) for value in cached_visuals[0]) != keyframe_visuals:
                    return False
                if H3_TEXT_VISUAL_MAX_PIXELS_KEY not in keys:
                    if args.h3_text_visual_max_pixels != 0:
                        return False
                elif int(handle.get_tensor(H3_TEXT_VISUAL_MAX_PIXELS_KEY)) != args.h3_text_visual_max_pixels:
                    return False
                if H3_MAX_CAPTION_TOKENS_KEY not in keys:
                    return args.h3_max_caption_tokens == 0
                return int(handle.get_tensor(H3_MAX_CAPTION_TOKENS_KEY)) == args.h3_max_caption_tokens
        except (OSError, RuntimeError, ValueError):
            return False

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files,
        all_cache_paths,
        encode,
        requires_content=encoder.conditioning_requires_content,
        existing_cache_valid=existing_cache_valid,
        faster_check=args.faster_check,
    )
    encoder.close()
    cache_text_encoder_outputs.post_process_cache_files(datasets, all_cache_files, all_cache_paths, args.keep_cache)


if __name__ == "__main__":
    main()
