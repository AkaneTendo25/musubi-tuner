#!/usr/bin/env python3
"""
Cache text encoder outputs for LTX-2 training.

Uses the standard musubi-tuner dataset config so cached files match the trainer.
"""

from __future__ import annotations

import argparse
import os

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
from musubi_tuner.ltx_2.env import apply_ltx2_tweaks


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_SAMPLE_PROMPTS_CACHE = "ltx2_sample_prompts_cache.pt"
DEFAULT_PRESERVATION_CACHE = "ltx2_preservation_cache.pt"


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


def _encode_prompt_text_ltx2(
    text_encoder,
    prompt_text: str,
    *,
    audio_video: bool,
    ltx_mode: str,
    autocast_dtype: torch.dtype | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if autocast_dtype is not None and device.type == "cuda":
        autocast_context = torch.cuda.amp.autocast(dtype=autocast_dtype)
    else:
        autocast_context = nullcontext()
    with torch.no_grad(), autocast_context:
        out = text_encoder(prompt_text, padding_side="left")
        if ltx_mode == "audio":
            embed = out.audio_encoding if hasattr(out, "audio_encoding") else out.video_encoding
        elif audio_video:
            embed = torch.cat([out.video_encoding, out.audio_encoding], dim=-1)
        else:
            embed = out.video_encoding
        mask = out.attention_mask
    return embed.squeeze(0).detach().cpu(), mask.squeeze(0).detach().cpu()


def _resolve_default_sample_prompts_cache(datasets: list) -> str:
    if not datasets:
        raise ValueError("No datasets available to resolve sample prompt cache directory")
    cache_dir = getattr(datasets[0], "cache_directory", None)
    if not cache_dir:
        raise ValueError("First dataset has no cache_directory; set cache_directory in dataset config")
    return os.path.join(cache_dir, DEFAULT_SAMPLE_PROMPTS_CACHE)


def _precache_sample_prompts(
    args: argparse.Namespace,
    *,
    datasets: list,
    text_encoder,
    audio_video: bool,
    ltx_mode: str,
    autocast_dtype: torch.dtype | None,
    device: torch.device,
) -> None:
    from musubi_tuner.hv_train_network import load_prompts

    if args.sample_prompts is None:
        raise ValueError("--sample_prompts is required when --precache_sample_prompts is set")

    prompts = load_prompts(args.sample_prompts)
    if not prompts:
        raise ValueError(f"No prompts found in {args.sample_prompts}")

    cache_path = args.sample_prompts_cache or _resolve_default_sample_prompts_cache(datasets)

    prompt_cache: list[dict] = []
    for prompt_dict in prompts:
        param = prompt_dict.copy()
        prompt_text = param.get("prompt", "")
        prompt_embeds, prompt_mask = _encode_prompt_text_ltx2(
            text_encoder,
            prompt_text,
            audio_video=audio_video,
            ltx_mode=ltx_mode,
            autocast_dtype=autocast_dtype,
            device=device,
        )
        cache_entry = {
            "prompt": prompt_text,
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_mask,
        }

        negative_prompt = param.get("negative_prompt")
        if negative_prompt:
            neg_embeds, neg_mask = _encode_prompt_text_ltx2(
                text_encoder,
                negative_prompt,
                audio_video=audio_video,
                ltx_mode=ltx_mode,
                autocast_dtype=autocast_dtype,
                device=device,
            )
            cache_entry["negative_prompt"] = negative_prompt
            cache_entry["negative_prompt_embeds"] = neg_embeds
            cache_entry["negative_prompt_attention_mask"] = neg_mask

        prompt_cache.append(cache_entry)

    payload = {
        "version": 2,
        "ltx_mode": ltx_mode,
        "audio_video": audio_video,
        "prompt_cache": prompt_cache,
    }
    torch.save(payload, cache_path)
    logger.info("Saved precached sample prompts to %s", cache_path)


def _precache_preservation_prompts(
    args: argparse.Namespace,
    *,
    datasets: list,
    text_encoder,
    audio_video: bool,
    autocast_dtype: torch.dtype | None,
    device: torch.device,
) -> None:
    """Encode blank/class prompts for preservation techniques and save to disk."""
    blank = getattr(args, "blank_preservation", False)
    dop = getattr(args, "dop", False)
    dop_class = getattr(args, "dop_class_prompt", "") or ""

    if not blank and not dop:
        logger.warning("--precache_preservation_prompts set but neither --blank_preservation nor --dop enabled, skipping.")
        return

    cache_path = getattr(args, "preservation_prompts_cache", None)
    if not cache_path:
        if not datasets:
            raise ValueError("No datasets available to resolve preservation cache directory")
        cache_dir = getattr(datasets[0], "cache_directory", None)
        if not cache_dir:
            raise ValueError("First dataset has no cache_directory; set cache_directory in dataset config")
        cache_path = os.path.join(cache_dir, DEFAULT_PRESERVATION_CACHE)

    payload: dict = {"version": 1, "audio_video": audio_video}

    # Always encode as video-only for preservation (even in AV mode)
    def _encode_video_only(prompt_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        embed, mask = _encode_prompt_text_ltx2(
            text_encoder, prompt_text,
            audio_video=audio_video, ltx_mode="video",  # force video-only encoding
            autocast_dtype=autocast_dtype, device=device,
        )
        # In AV mode the encoder still concatenates; take video half
        if audio_video and embed.shape[-1] % 2 == 0:
            embed = embed[..., : embed.shape[-1] // 2]
        return embed, mask

    if blank:
        embed, mask = _encode_video_only("")
        payload["blank_embed"] = embed
        payload["blank_mask"] = mask
        logger.info("Preservation cache: encoded blank prompt  embed=%s", tuple(embed.shape))

    if dop:
        if not dop_class:
            logger.warning("--dop set but no --dop_class_prompt provided, encoding empty string.")
        embed, mask = _encode_video_only(dop_class)
        payload["dop_embed"] = embed
        payload["dop_mask"] = mask
        payload["dop_class_prompt"] = dop_class
        logger.info("Preservation cache: encoded DOP class prompt %r  embed=%s", dop_class, tuple(embed.shape))

    torch.save(payload, cache_path)
    logger.info("Saved preservation prompt cache to %s", cache_path)


def main() -> None:
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = ltx2_setup_parser(parser)
    args = parser.parse_args()
    apply_ltx2_tweaks(args)

    short_map = {"v": "video", "a": "audio", "va": "av"}
    if getattr(args, "ltx_mode", None) in short_map:
        args.ltx_mode = short_map[args.ltx_mode]

    ltx_mode = getattr(args, "ltx_mode", "video")
    # For audio-only or AV mode, we need the AV encoder to get audio encodings
    audio_video = ltx_mode in ("av", "audio")

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info("Load dataset config from %s", args.dataset_config)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_LTX2)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = list(train_dataset_group.datasets)

    if user_config.get("validation_datasets"):
        logger.info("Load validation datasets from dataset config")
        validation_user_config = {
            "general": user_config.get("general", {}),
            "datasets": user_config.get("validation_datasets", []),
        }
        validation_blueprint = blueprint_generator.generate(
            validation_user_config, args, architecture=ARCHITECTURE_LTX2
        )
        validation_dataset_group = config_utils.generate_dataset_group_by_blueprint(
            validation_blueprint.dataset_group
        )
        datasets.extend(validation_dataset_group.datasets)

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32



    if getattr(args, "gemma_load_in_8bit", False) or getattr(args, "gemma_load_in_4bit", False):
        if device.type != "cuda":
            raise ValueError("Gemma 8-bit/4-bit loading requires --device cuda")

    autocast_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else None

    if getattr(args, "require_gemma_root", False):
        if args.gemma_root is None:
            raise ValueError("--gemma_root is required for LTX-2 Gemma text caching")
    elif args.gemma_root is None:
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

    configurator = AVGemmaTextEncoderModelConfigurator if audio_video else VideoGemmaTextEncoderModelConfigurator
    key_ops = AV_GEMMA_TEXT_ENCODER_KEY_OPS if audio_video else VIDEO_ONLY_GEMMA_TEXT_ENCODER_KEY_OPS

    bnb_compute_dtype = None
    bnb_compute_dtype_arg = getattr(args, "gemma_bnb_4bit_compute_dtype", "auto")
    if bnb_compute_dtype_arg == "auto":
        bnb_compute_dtype = dtype
    elif bnb_compute_dtype_arg == "fp16":
        bnb_compute_dtype = torch.float16
    elif bnb_compute_dtype_arg == "bf16":
        bnb_compute_dtype = torch.bfloat16
    elif bnb_compute_dtype_arg == "fp32":
        bnb_compute_dtype = torch.float32

    text_encoder = SingleGPUModelBuilder(
        model_path=str(text_encoder_checkpoint),
        model_class_configurator=configurator,
        model_sd_ops=key_ops,
        module_ops=module_ops_from_gemma_root(
            args.gemma_root,
            torch_dtype=dtype,
            load_in_8bit=bool(getattr(args, "gemma_load_in_8bit", False)),
            load_in_4bit=bool(getattr(args, "gemma_load_in_4bit", False)),
            bnb_4bit_quant_type=str(getattr(args, "gemma_bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=not bool(getattr(args, "gemma_bnb_4bit_disable_double_quant", False)),
            bnb_4bit_compute_dtype=bnb_compute_dtype,
        ),
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
            audio_video=audio_video,
        )

    # Text caching is CPU-heavy (tokenization, python-side preprocessing). On Windows, high num_workers
    # often hurts throughput or appears to hang due to thread contention. Default to 1 unless specified.
    num_workers = 1 if args.num_workers is None else args.num_workers

    if getattr(args, "precache_sample_prompts", False):
        _precache_sample_prompts(
            args,
            datasets=datasets,
            text_encoder=text_encoder,
            audio_video=audio_video,
            ltx_mode=ltx_mode,
            autocast_dtype=autocast_dtype,
            device=device,
        )

    if getattr(args, "precache_preservation_prompts", False):
        _precache_preservation_prompts(
            args,
            datasets=datasets,
            text_encoder=text_encoder,
            audio_video=audio_video,
            autocast_dtype=autocast_dtype,
            device=device,
        )

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
        help="Path to LTX-2 checkpoint (.safetensors)",
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
        "--ltx2_mode", "--ltx_mode",
        dest="ltx_mode",
        type=str,
        default="video",
        choices=["video", "av", "audio", "v", "a", "va"],
        help="Caching modality: 'video' (default), 'av' for audio+video, 'audio' for audio-only.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
    )
    parser.add_argument(
        "--precache_sample_prompts",
        action="store_true",
        help="Also cache Gemma embeddings for sample prompts and save to --sample_prompts_cache.",
    )
    parser.add_argument(
        "--sample_prompts",
        type=str,
        default=None,
        help="Sample prompt file used for --precache_sample_prompts.",
    )
    parser.add_argument(
        "--sample_prompts_cache",
        type=str,
        default=None,
        help=(
            "Path to write precached sample prompt embeddings (.pt). Defaults to "
            "the first dataset's cache_directory/ltx2_sample_prompts_cache.pt"
        ),
    )

    # -- Preservation prompt precaching --
    parser.add_argument(
        "--precache_preservation_prompts",
        action="store_true",
        help="Cache Gemma embeddings for preservation prompts (blank/DOP class) and save to --preservation_prompts_cache.",
    )
    parser.add_argument(
        "--blank_preservation",
        action="store_true",
        help="Include blank prompt in preservation cache (for --blank_preservation during training).",
    )
    parser.add_argument(
        "--dop",
        action="store_true",
        help="Include DOP class prompt in preservation cache (for --dop during training).",
    )
    parser.add_argument(
        "--dop_class_prompt",
        type=str,
        default="",
        help="Class prompt for DOP preservation, e.g. 'woman' (without trigger word).",
    )
    parser.add_argument(
        "--preservation_prompts_cache",
        type=str,
        default=None,
        help=(
            "Path to write precached preservation prompt embeddings (.pt). Defaults to "
            "the first dataset's cache_directory/ltx2_preservation_cache.pt"
        ),
    )

    parser.add_argument(
        "--gemma_load_in_8bit",
        action="store_true",
        help="Load Gemma LLM in 8-bit (bitsandbytes). CUDA only.",
    )
    parser.add_argument(
        "--gemma_load_in_4bit",
        action="store_true",
        help="Load Gemma LLM in 4-bit (bitsandbytes). CUDA only.",
    )
    parser.add_argument(
        "--gemma_bnb_4bit_quant_type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
        help="bitsandbytes 4-bit quant type (nf4 or fp4)",
    )
    parser.add_argument(
        "--gemma_bnb_4bit_disable_double_quant",
        action="store_true",
        help="Disable bitsandbytes double quant for 4-bit loading.",
    )
    parser.add_argument(
        "--gemma_bnb_4bit_compute_dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="Compute dtype for 4-bit (auto uses --mixed_precision dtype)",
    )
    return parser


if __name__ == "__main__":
    main()
