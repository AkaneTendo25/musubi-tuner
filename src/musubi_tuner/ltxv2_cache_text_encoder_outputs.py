#!/usr/bin/env python3
"""
Cache text encoder outputs for LTXV2 training.

Uses the standard musubi-tuner dataset config so cached files match the trainer.
"""

from __future__ import annotations

import argparse
from typing import List

import logging
from contextlib import nullcontext
import torch
from transformers import AutoTokenizer, T5EncoderModel

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_LTXV2,
    ItemInfo,
    save_text_encoder_output_cache_ltxv2,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(
    tokenizer,
    text_encoder,
    batch: list[ItemInfo],
    device: torch.device,
    max_length: int,
    autocast_dtype: torch.dtype | None,
    audio_video: bool,
) -> None:
    prompts = [item.caption for item in batch]
    tokens = tokenizer(
        prompts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = tokens["input_ids"].to(device=device)
    attention_mask = tokens["attention_mask"].to(device=device)

    if autocast_dtype is not None and device.type == "cuda":
        autocast_context = torch.cuda.amp.autocast(dtype=autocast_dtype)
    else:
        autocast_context = nullcontext()

    def _encode_hidden_states(model, ids, mask):
        outputs = model(input_ids=ids, attention_mask=mask, output_hidden_states=False, return_dict=True)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        if hasattr(model, "model"):
            inner = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=False, return_dict=True)
            if hasattr(inner, "last_hidden_state") and inner.last_hidden_state is not None:
                return inner.last_hidden_state
        raise RuntimeError("Unable to extract last hidden states from the selected text encoder model")

    with torch.no_grad(), autocast_context:
        hidden = _encode_hidden_states(text_encoder, input_ids, attention_mask)

    if audio_video:
        hidden = torch.cat([hidden, hidden], dim=-1)

    embeddings = hidden.detach().cpu()
    masks = attention_mask.detach().cpu()

    for item, embed, mask in zip(batch, embeddings, masks):
        save_text_encoder_output_cache_ltxv2(item, embed, mask)


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
                embed = torch.cat([out.video_encoding, out.audio_encoding], dim=-1)
                mask = out.attention_mask
            else:
                out = text_encoder(item.caption, padding_side="left")
                embed = out.video_encoding
                mask = out.attention_mask

            embed = embed.squeeze(0).detach().cpu()
            mask = mask.squeeze(0).detach().cpu()
            save_text_encoder_output_cache_ltxv2(item, embed, mask)


def main() -> None:
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = ltxv2_setup_parser(parser)
    args = parser.parse_args()

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info("Load dataset config from %s", args.dataset_config)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_LTXV2)
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

    if args.text_encoder_backend == "t5":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        tokenizer.padding_side = "right"
        logger.info("Loading text encoder from %s (backend=t5)", args.text_encoder)
        text_encoder = T5EncoderModel.from_pretrained(args.text_encoder, torch_dtype=dtype, device_map=str(device))
        text_encoder.eval()

        def encode_fn(batch: list[ItemInfo]) -> None:
            encode_and_save_batch(
                tokenizer,
                text_encoder,
                batch,
                device,
                args.max_length,
                autocast_dtype,
                args.ltxv2_audio_video,
            )
    elif args.text_encoder_backend == "gemma":
        if args.gemma_root is None:
            raise ValueError("--gemma_root is required when --text_encoder_backend gemma")
        if args.ltx2_checkpoint is None:
            raise ValueError("--ltx2_checkpoint is required when --text_encoder_backend gemma")

        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from ltx_core.text_encoders.gemma.encoders.av_encoder import (
            AVGemmaTextEncoderModelConfigurator,
            AV_GEMMA_TEXT_ENCODER_KEY_OPS,
        )
        from ltx_core.text_encoders.gemma.encoders.base_encoder import module_ops_from_gemma_root
        from ltx_core.text_encoders.gemma.encoders.video_only_encoder import (
            VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS,
            VideoGemmaTextEncoderModelConfigurator,
        )

        configurator = AVGemmaTextEncoderModelConfigurator if args.ltxv2_audio_video else VideoGemmaTextEncoderModelConfigurator
        key_ops = AV_GEMMA_TEXT_ENCODER_KEY_OPS if args.ltxv2_audio_video else VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS

        text_encoder = SingleGPUModelBuilder(
            model_path=str(args.ltx2_checkpoint),
            model_class_configurator=configurator,
            model_sd_ops=key_ops,
            module_ops=module_ops_from_gemma_root(args.gemma_root),
        ).build(device=device, dtype=dtype)
        text_encoder.eval()

        def encode_fn(batch: list[ItemInfo]) -> None:
            encode_and_save_batch_official_gemma(
                text_encoder,
                batch,
                device=device,
                autocast_dtype=autocast_dtype,
                audio_video=args.ltxv2_audio_video,
            )
    else:
        raise ValueError(f"Unknown text_encoder_backend: {args.text_encoder_backend}")

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
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


def ltxv2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=True,
        help="Text encoder weights (HF directory or ID). Use with --text_encoder_backend.",
    )
    parser.add_argument(
        "--text_encoder_backend",
        type=str,
        default="t5",
        choices=["t5", "gemma"],
        help="Text encoder backend to use for caching.",
    )
    parser.add_argument("--tokenizer", type=str, default="google/t5-v1_1-xxl", help="Tokenizer repo or path (T5 only)")
    parser.add_argument(
        "--ltx2_checkpoint",
        type=str,
        default=None,
        help="LTX-2 checkpoint (.safetensors) containing Gemma connector weights (Gemma backend only)",
    )
    parser.add_argument(
        "--gemma_root",
        type=str,
        default=None,
        help="Local directory containing Gemma weights/tokenizer (Gemma backend only)",
    )
    parser.add_argument("--max_length", type=int, default=256, help="Maximum token sequence length")
    parser.add_argument(
        "--ltxv2_audio_video",
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
