#!/usr/bin/env python3
"""
Cache text encoder outputs for LTX-2 training.

Uses the standard musubi-tuner dataset config so cached files match the trainer.
"""

from __future__ import annotations

import argparse

import logging
from contextlib import nullcontext
import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_LTX2,
    ItemInfo,
    save_text_encoder_output_cache_ltx2_official,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch_official_gemma(
    text_encoder,
    batch: list[ItemInfo],
    *,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    audio_video: bool,
) -> None:
    if autocast_dtype is not None and device.type == "cuda":
        autocast_context = torch.cuda.amp.autocast(dtype=autocast_dtype)
    else:
        autocast_context = nullcontext()

    with torch.no_grad(), autocast_context:
        for item in batch:
            if audio_video:
                out = text_encoder(item.caption, padding_side="left")
                video_embed = out.video_encoding
                audio_embed = out.audio_encoding
                mask = out.attention_mask
            else:
                out = text_encoder(item.caption, padding_side="left")
                video_embed = out.video_encoding
                audio_embed = None
                mask = out.attention_mask

            video_embed = video_embed.squeeze(0).detach().cpu()
            mask = mask.squeeze(0).detach().cpu()
            audio_embed_out = audio_embed.squeeze(0).detach().cpu() if audio_embed is not None else None

            save_text_encoder_output_cache_ltx2_official(
                item,
                video_prompt_embeds=video_embed,
                audio_prompt_embeds=audio_embed_out,
                prompt_attention_mask=mask,
            )


def main() -> None:
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = ltx2_setup_parser(parser)
    args = parser.parse_args()

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info("Load dataset config from %s", args.dataset_config)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_LTX2)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    autocast_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else None

    if args.gemma_root is None:
        raise ValueError("--gemma_root is required for LTX-2 Gemma text caching")
    if args.ltx2_checkpoint is None and getattr(args, "ltx2_text_encoder_checkpoint", None) is None:
        raise ValueError("--ltx2_checkpoint is required for LTX-2 Gemma text caching")
    from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from musubi_tuner.ltx_2.text_encoders.gemma.encoders.av_encoder import (
        AVGemmaTextEncoderModelConfigurator,
        AV_GEMMA_TEXT_ENCODER_KEY_OPS,
    )
    from musubi_tuner.ltx_2.text_encoders.gemma.encoders.base_encoder import module_ops_from_gemma_root
    from musubi_tuner.ltx_2.text_encoders.gemma.encoders.video_only_encoder import (
        VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS,
        VideoGemmaTextEncoderModelConfigurator,
    )

    text_encoder_checkpoint = (
        args.ltx2_text_encoder_checkpoint
        if getattr(args, "ltx2_text_encoder_checkpoint", None) is not None
        else args.ltx2_checkpoint
    )

    configurator = AVGemmaTextEncoderModelConfigurator if args.ltx2_audio_video else VideoGemmaTextEncoderModelConfigurator
    key_ops = AV_GEMMA_TEXT_ENCODER_KEY_OPS if args.ltx2_audio_video else VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS

    text_encoder = SingleGPUModelBuilder(
        model_path=str(text_encoder_checkpoint),
        model_class_configurator=configurator,
        model_sd_ops=key_ops,
        module_ops=module_ops_from_gemma_root(args.gemma_root),
    ).build(device=device, dtype=dtype)
    text_encoder.eval()

    # If connector weights are missing, SingleGPUModelBuilder returns a meta-device model.
    # That will make caching appear to hang or behave unpredictably. Fail fast with a clear error.
    meta_params = [name for name, p in text_encoder.named_parameters() if p.device.type == "meta"]
    meta_bufs = [name for name, b in text_encoder.named_buffers() if b.device.type == "meta"]
    if meta_params or meta_bufs:
        raise ValueError(
            "LTX-2 Gemma text encoder has uninitialized (meta) parameters/buffers. "
            "Your --ltx2_checkpoint likely does not contain the Gemma connector weights required for caching. "
            f"meta_params={meta_params[:10]} meta_bufs={meta_bufs[:10]}"
        )

    def encode_fn(batch: list[ItemInfo]) -> None:
        encode_and_save_batch_official_gemma(
            text_encoder,
            batch,
            device=device,
            autocast_dtype=autocast_dtype,
            audio_video=args.ltx2_audio_video,
        )

    # Text caching is CPU-heavy (tokenization, python-side preprocessing). On Windows, high num_workers
    # often hurts throughput or appears to hang due to thread contention. Default to 1 unless specified.
    num_workers = 1 if args.num_workers is None else args.num_workers

    cache_text_encoder_outputs.process_text_encoder_batches(
        num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_fn,
    )

    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


def ltx2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--ltx2_checkpoint",
        type=str,
        default=None,
        help="LTX-2 checkpoint (.safetensors) containing Gemma connector weights (Gemma backend only)",
    )
    parser.add_argument(
        "--ltx2_text_encoder_checkpoint",
        type=str,
        default=None,
        help="Optional separate checkpoint (.safetensors) used only for Gemma text encoder connector weights. Defaults to --ltx2_checkpoint.",
    )
    parser.add_argument(
        "--gemma_root",
        type=str,
        default=None,
        help="Local directory containing Gemma weights/tokenizer (Gemma backend only)",
    )
    parser.add_argument(
        "--ltx2_audio_video",
        action="store_true",
        help="If set, cache concatenated [video_ctx, audio_ctx] along last dim (required by LTXAV).",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
    )
    return parser


if __name__ == "__main__":
    main()
