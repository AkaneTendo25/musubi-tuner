#!/usr/bin/env python3
"""LTX-2 inference script.

Uses the same sampling infrastructure as the training script (sample_image_inference)
with all the same arg names, so you can copy settings from a training config directly.

Supports single-stage and two-stage inference, LoRA merging, block swap,
staged offloading (text encoder → transformer → VAE), audio+video generation,
image-to-video conditioning, and video-to-video reference conditioning.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from types import SimpleNamespace
from typing import List, Optional

import torch
from accelerate import Accelerator
from safetensors.torch import load_file

from musubi_tuner.hv_generate_video import setup_parser_compile
from musubi_tuner.ltx2_defaults import get_ltx2_sampling_preset
from musubi_tuner.ltx2_lora_utils import (
    apply_lora_network_for_inference,
    import_lora_network_module,
    infer_lora_network_module,
    load_lora_metadata,
)
from musubi_tuner.model_defaults import default_gemma_root_path, default_ltx2_checkpoint_path
from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer
from musubi_tuner.networks import lora_ltx2
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _configure_windows_cli_encoding() -> None:
    """Keep Unicode help text printable when Windows stdout is not UTF-8."""
    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _apply_reference_conditioning_overrides(
    prompts: list[dict],
    *,
    reference_image: str | None = None,
    reference_video: str | None = None,
) -> tuple[str, bool] | None:
    if not reference_image and not reference_video:
        return None

    if reference_image and reference_video:
        logger.warning(
            "Both --reference_image and --reference_video given; --reference_video takes priority (V2V), --reference_image ignored."
        )

    ref_path = reference_video or reference_image

    # Auto-detect: if an --reference_image path is actually a video by ext, use V2V slot.
    try:
        from musubi_tuner.dataset.image_video_dataset import VIDEO_EXTENSIONS

        video_exts = {e.lower() for e in VIDEO_EXTENSIONS}
    except Exception:
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    ext = os.path.splitext(ref_path)[1].lower()
    use_v2v = bool(reference_video) or (ext in video_exts)

    if not os.path.isfile(ref_path):
        raise FileNotFoundError(f"Reference path does not exist: {ref_path}")

    for prompt_dict in prompts:
        # Explicit CLI references should override prompt-file paths and cached reference latents.
        prompt_dict.pop("image_path", None)
        prompt_dict.pop("conditioning_latent", None)
        prompt_dict.pop("v2v_ref_path", None)
        prompt_dict.pop("v2v_ref_latent", None)
        if use_v2v:
            prompt_dict["v2v_ref_path"] = ref_path
        else:
            prompt_dict["image_path"] = ref_path

    return ref_path, use_v2v


# ---------------------------------------------------------------------------
# Arg parsing — mirrors training script arg names so configs are portable
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LTX-2 video generation (inference only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- Core model --
    parser.add_argument(
        "--ltx2_checkpoint",
        type=str,
        default=default_ltx2_checkpoint_path(),
        help="Path to LTX-2 checkpoint (.safetensors). Also used for VAE unless --vae is set.",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="Path to separate VAE checkpoint (defaults to --ltx2_checkpoint).",
    )
    parser.add_argument(
        "--vae_dtype",
        type=str,
        default=None,
        help="VAE dtype (default: bfloat16). Options: bfloat16, float16, float32.",
    )
    parser.add_argument(
        "--ltx2_mode",
        "--ltx_mode",
        dest="ltx_mode",
        type=str,
        default="video",
        choices=["video", "av", "audio", "v", "a", "va"],
        help="Generation modality: 'video' (default), 'av' for audio+video, 'audio' for audio-only.",
    )
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--device", type=str, default=None, help="Force device to cpu or cuda")
    parser.add_argument(
        "--ltx_version",
        type=str,
        default="2.3",
        choices=["2.0", "2.3"],
        help="Target LTX version for default sampling presets.",
    )

    # -- Gemma text encoder --
    parser.add_argument(
        "--gemma_root", type=str, default=default_gemma_root_path(), help="Local directory containing Gemma weights/tokenizer"
    )
    parser.add_argument(
        "--gemma_safetensors",
        type=str,
        default=None,
        help="Single Gemma safetensors file (for example, an fp8 export). No --gemma_root needed.",
    )
    parser.add_argument("--gemma_load_in_8bit", action="store_true", help="Load Gemma in 8-bit (bitsandbytes). CUDA only.")
    parser.add_argument("--gemma_load_in_4bit", action="store_true", help="Load Gemma in 4-bit (bitsandbytes). CUDA only.")
    parser.add_argument("--gemma_bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--gemma_bnb_4bit_disable_double_quant", action="store_true")
    parser.add_argument(
        "--gemma_fp8_weight_offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When using FP8 Gemma safetensors, offload FP8 linear weights to CPU RAM. "
            "Defaults to the LTX2_GEMMA_SAFETENSORS_WEIGHT_OFFLOAD environment variable when omitted."
        ),
    )

    # -- LoRA (merged into transformer for inference) --
    parser.add_argument("--lora_weight", type=str, nargs="*", default=None, help="LoRA weight path(s) to merge")
    parser.add_argument(
        "--lora_multiplier", type=float, nargs="*", default=None, help="LoRA multiplier(s), aligned with --lora_weight order"
    )
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns")

    # -- Prompt input --
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt (enables CFG)")
    parser.add_argument(
        "--sample_prompts",
        "--from_file",
        type=str,
        default=None,
        dest="sample_prompts",
        help="Read prompts from a .txt file (one per line or TOML-style; same format as training --sample_prompts)",
    )

    # -- Output --
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--output_name", type=str, default="ltx2_gen", help="Base output filename prefix")

    # -- Generation controls (become sample_parameter defaults) --
    parser.add_argument(
        "--sampling_preset",
        "--sample_sampling_preset",
        type=str,
        default="defaults",
        choices=["legacy", "defaults", "ltx20", "ltx23", "ltx23_hq", "distilled_two_stage"],
        help="Generation defaults. Use 'legacy' for the old 512x768/20-step/no-guidance defaults.",
    )
    parser.add_argument(
        "--use_default_negative_prompt",
        "--sample_use_default_negative_prompt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the default LTX negative prompt when CFG is enabled and --negative_prompt is omitted.",
    )
    parser.add_argument("--height", type=int, default=None, help="Output height in pixels (rounded to multiple of 32)")
    parser.add_argument("--width", type=int, default=None, help="Output width in pixels (rounded to multiple of 32)")
    parser.add_argument(
        "--frame_count",
        "--sample_num_frames",
        type=int,
        default=None,
        dest="frame_count",
        help="Number of frames (rounded to 8k+1)",
    )
    parser.add_argument("--frame_rate", type=float, default=None, help="Output FPS")
    parser.add_argument("--sample_steps", type=int, default=None, help="Number of denoising steps")
    parser.add_argument(
        "--sample_sigma_schedule",
        type=str,
        default="auto",
        choices=["auto", "ltx", "ltx23_distilled"],
        help=(
            "Sigma schedule. 'auto' uses LTX token-shifted sigmas, and the exact "
            "LTX-2.3 distilled sigmas for the distilled_two_stage preset."
        ),
    )
    parser.add_argument(
        "--sample_sampler",
        type=str,
        default="auto",
        choices=["auto", "euler", "res_2s"],
        help="Sampler. 'auto' uses res_2s for full presets and Euler for distilled_two_stage.",
    )
    parser.add_argument("--guidance_scale", type=float, default=None, help="Guidance scale (1.0 = no guidance)")
    parser.add_argument("--cfg_scale", type=float, default=None, help="CFG scale (overrides guidance_scale when set)")
    parser.add_argument("--discrete_flow_shift", type=float, default=5.0, help="Flow matching shift parameter")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (None = random)")
    parser.add_argument("--video_cfg_scale", type=float, default=None, help="Video CFG scale")
    parser.add_argument("--audio_cfg_scale", type=float, default=None, help="Audio CFG scale")
    parser.add_argument("--video_modality_scale", type=float, default=None, help="Video A2V modality guidance scale")
    parser.add_argument("--audio_modality_scale", type=float, default=None, help="Audio V2A modality guidance scale")
    parser.add_argument("--video_rescale_scale", type=float, default=None, help="Video CFG rescale scale")
    parser.add_argument("--audio_rescale_scale", type=float, default=None, help="Audio CFG rescale scale")
    parser.add_argument(
        "--stg_scale",
        type=float,
        default=None,
        help="Spatio-Temporal Guidance scale (0.0 = disabled). Recommended default is 1.0. "
        "Costs one extra transformer forward per denoising step.",
    )
    parser.add_argument(
        "--stg_blocks",
        type=int,
        nargs="*",
        default=None,
        help="Transformer block indices to perturb for STG (None = all blocks).",
    )
    parser.add_argument(
        "--stg_mode",
        type=str,
        default=None,
        choices=["video", "audio", "both"],
        help="Which self-attention modality to perturb for STG.",
    )
    parser.add_argument(
        "--rescale_scale",
        type=float,
        default=None,
        help="CFG\u2605 rescaling after CFG+STG. LTX-2.3 default is 0.9; 0 disables.",
    )
    parser.add_argument("--av_bimodal_cfg", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--av_bimodal_scale", type=float, default=None)

    # -- Attention / DiT quantization --
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "flash2", "flash3", "torch", "xformers", "sdpa"],
        help="Attention backend",
    )
    # Training-style boolean attention flags (for config portability)
    parser.add_argument("--flash_attn", action="store_true", help="Use FlashAttention (same as --attn_mode flash)")
    parser.add_argument("--flash3", action="store_true", help="Use FlashAttention 3 (same as --attn_mode flash3)")
    parser.add_argument("--sdpa", action="store_true", help="Use SDPA (same as --attn_mode sdpa)")
    parser.add_argument("--xformers", action="store_true", help="Use xformers (same as --attn_mode xformers)")
    parser.add_argument("--fp8_base", action="store_true", help="Use FP8 cast for DiT weights")
    parser.add_argument("--fp8_scaled", action="store_true", help="Use scaled FP8 (requires fp8_base)")
    parser.add_argument(
        "--fp8_keep_blocks",
        type=str,
        default=None,
        help="Comma-separated transformer block indices to keep in high precision with --fp8_scaled, e.g. 0,1,2,45.",
    )
    parser.add_argument("--fp8_w8a8", action="store_true", help="Use W8A8 quantization (requires fp8_scaled)")
    parser.add_argument("--w8a8_mode", type=str, default="int8", choices=["int8", "fp8"])
    parser.add_argument("--fp8_upcast", action="store_true")
    parser.add_argument("--fp8_upcast_stochastic", action="store_true")
    parser.add_argument("--fp8_upcast_seed", type=int, default=0)
    parser.add_argument("--nf4_base", action="store_true", help="Use NF4 quantization for DiT")
    parser.add_argument("--nf4_block_size", type=int, default=64)
    parser.add_argument("--loftq_init", action="store_true")
    parser.add_argument("--loftq_iters", type=int, default=2)
    parser.add_argument("--awq_calibration", action="store_true")
    parser.add_argument("--awq_alpha", type=float, default=0.25)
    parser.add_argument("--awq_num_batches", type=int, default=8)
    parser.add_argument("--network_dim", type=int, default=0, help="LoRA rank (needed for loftq_init)")
    parser.add_argument("--split_attn_target", type=str, nargs="*", default=None)
    parser.add_argument("--split_attn_mode", type=str, default=None)
    parser.add_argument("--split_attn_chunk_size", type=int, default=0)
    parser.add_argument("--ffn_chunk_target", type=str, nargs="*", default=None)
    parser.add_argument("--ffn_chunk_size", type=int, default=0)

    # -- Block swap / VRAM management --
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="Number of DiT blocks to swap to CPU")
    parser.add_argument("--use_pinned_memory_for_block_swap", action="store_true")
    parser.add_argument(
        "--sample_with_offloading",
        action="store_true",
        help="Staged offloading: text encoder → CPU, then transformer → CPU before VAE decode. Saves ~10GB VRAM.",
    )

    # -- I2V / V2V conditioning --
    parser.add_argument(
        "--sample_i2v_token_timestep_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LTX I2V token timestep masking during sampling",
    )
    parser.add_argument(
        "--reference_downscale", type=int, default=1, help="Spatial downscale factor for V2V references (1=same res)"
    )
    parser.add_argument("--reference_frames", type=int, default=1, help="Number of V2V reference frames")
    parser.add_argument("--sample_include_reference", action="store_true", help="Show V2V reference side-by-side in output")
    parser.add_argument(
        "--reference_image",
        type=str,
        default=None,
        help="Path to reference image for I2V conditioning (single frame). "
        "If path points to a video file by extension, treated as V2V.",
    )
    parser.add_argument(
        "--reference_video", type=str, default=None, help="Path to reference video file for V2V conditioning (multi-frame)."
    )

    # -- Audio (AV mode) --
    parser.add_argument("--sample_disable_audio", action="store_true", help="Disable audio decoding in AV mode")
    parser.add_argument("--sample_audio_only", action="store_true", help="Audio-only output (skip video decode)")
    parser.add_argument("--sample_merge_audio", action="store_true", help="Mux audio into video (_av.mp4)")
    parser.add_argument(
        "--sample_audio_subprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode audio in a subprocess. Use --no-sample_audio_subprocess to decode in-process.",
    )

    # -- Two-stage inference --
    parser.add_argument(
        "--sample_two_stage", action="store_true", help="Two-stage: generate at half res, upsample, refine with distilled LoRA."
    )
    parser.add_argument(
        "--spatial_upsampler_path", type=str, default=None, help="Path to spatial upsampler model for two-stage inference."
    )
    parser.add_argument("--distilled_lora_path", type=str, default=None, help="Path to distilled LoRA for two-stage refinement.")
    parser.add_argument("--sample_stage2_steps", type=int, default=3, help="Number of stage-2 refinement steps (default: 3)")
    parser.add_argument(
        "--sample_stage1_distilled_lora_multiplier",
        type=float,
        default=None,
        help="Stage-1 distilled LoRA multiplier for two-stage. Default: 0.25 with res_2s, 0.0 with Euler.",
    )
    parser.add_argument(
        "--sample_stage2_distilled_lora_multiplier",
        type=float,
        default=None,
        help="Stage-2 distilled LoRA multiplier for two-stage. Default: 0.5 with res_2s, 1.0 with Euler.",
    )

    # -- Tiled VAE decode --
    parser.add_argument("--sample_tiled_vae", action="store_true", help="Enable tiled VAE decoding")
    parser.add_argument("--sample_vae_tile_size", type=int, default=512, help="Spatial tile size (px)")
    parser.add_argument("--sample_vae_tile_overlap", type=int, default=64, help="Spatial tile overlap (px)")
    parser.add_argument("--sample_vae_temporal_tile_size", type=int, default=0, help="Temporal tile size (0=no)")
    parser.add_argument("--sample_vae_temporal_tile_overlap", type=int, default=8, help="Temporal tile overlap")

    # -- Flash attn override --
    parser.add_argument(
        "--sample_disable_flash_attn", action="store_true", help="Disable FlashAttention during sampling (force SDPA)"
    )

    # -- Precached embeddings (skip Gemma at inference) --
    parser.add_argument(
        "--use_precached_sample_prompts",
        "--precache_sample_prompts",
        action="store_true",
        dest="use_precached_sample_prompts",
        help="Use precached Gemma embeddings instead of loading Gemma at inference time.",
    )
    parser.add_argument("--sample_prompts_cache", type=str, default=None, help="Path to precached sample prompt embeddings (.pt)")
    parser.add_argument("--use_precached_sample_latents", action="store_true", help="Use precached I2V conditioning latents.")
    parser.add_argument("--sample_latents_cache", type=str, default=None, help="Path to precached I2V conditioning latents (.pt)")

    # -- Compile --
    setup_parser_compile(parser)

    # -- Parse and validate --
    args = parser.parse_args()

    # Normalize mode aliases
    short_map = {"v": "video", "a": "audio", "va": "av"}
    if getattr(args, "ltx_mode", None) in short_map:
        args.ltx_mode = short_map[args.ltx_mode]

    if args.prompt is None and args.sample_prompts is None:
        raise ValueError("Either --prompt or --sample_prompts (--from_file) must be specified")
    if args.gemma_root is None and not args.gemma_safetensors and not args.use_precached_sample_prompts:
        raise ValueError("--gemma_root or --gemma_safetensors is required (unless using --use_precached_sample_prompts)")
    if args.gemma_load_in_8bit and args.gemma_load_in_4bit:
        raise ValueError("--gemma_load_in_8bit and --gemma_load_in_4bit cannot be enabled together")
    if args.gemma_safetensors and (args.gemma_load_in_8bit or args.gemma_load_in_4bit):
        raise ValueError("--gemma_safetensors cannot be combined with --gemma_load_in_4bit/8bit")

    preset = get_ltx2_sampling_preset(args.sampling_preset, ltx_version=args.ltx_version)
    if preset is None:
        args.height = 512 if args.height is None else args.height
        args.width = 768 if args.width is None else args.width
        args.frame_count = 45 if args.frame_count is None else args.frame_count
        args.frame_rate = 25.0 if args.frame_rate is None else args.frame_rate
        args.sample_steps = 20 if args.sample_steps is None else args.sample_steps
        args.guidance_scale = 1.0 if args.guidance_scale is None else args.guidance_scale
        args.stg_scale = 0.0 if args.stg_scale is None else args.stg_scale
        args.stg_mode = "video" if args.stg_mode is None else args.stg_mode
        args.rescale_scale = 0.0 if args.rescale_scale is None else args.rescale_scale
        if args.use_default_negative_prompt is None:
            args.use_default_negative_prompt = False
    else:
        args.height = preset.height if args.height is None else args.height
        args.width = preset.width if args.width is None else args.width
        args.frame_count = preset.frame_count if args.frame_count is None else args.frame_count
        args.frame_rate = preset.frame_rate if args.frame_rate is None else args.frame_rate
        args.sample_steps = preset.sample_steps if args.sample_steps is None else args.sample_steps
        args.guidance_scale = preset.video_cfg_scale if args.guidance_scale is None else args.guidance_scale
        args.video_cfg_scale = preset.video_cfg_scale if args.video_cfg_scale is None else args.video_cfg_scale
        args.audio_cfg_scale = preset.audio_cfg_scale if args.audio_cfg_scale is None else args.audio_cfg_scale
        args.stg_scale = preset.stg_scale if args.stg_scale is None else args.stg_scale
        args.stg_blocks = preset.stg_blocks if args.stg_blocks is None else args.stg_blocks
        args.stg_mode = preset.stg_mode if args.stg_mode is None else args.stg_mode
        args.rescale_scale = preset.video_rescale_scale if args.rescale_scale is None else args.rescale_scale
        args.video_rescale_scale = preset.video_rescale_scale if args.video_rescale_scale is None else args.video_rescale_scale
        args.audio_rescale_scale = preset.audio_rescale_scale if args.audio_rescale_scale is None else args.audio_rescale_scale
        args.video_modality_scale = preset.video_modality_scale if args.video_modality_scale is None else args.video_modality_scale
        args.audio_modality_scale = preset.audio_modality_scale if args.audio_modality_scale is None else args.audio_modality_scale
        if args.av_bimodal_cfg is None:
            args.av_bimodal_cfg = preset.video_modality_scale != 1.0 or preset.audio_modality_scale != 1.0
        if args.av_bimodal_scale is None:
            args.av_bimodal_scale = preset.video_modality_scale
        if args.use_default_negative_prompt is None:
            args.use_default_negative_prompt = bool(preset.negative_prompt)
        if args.negative_prompt is None and args.use_default_negative_prompt:
            args.negative_prompt = preset.negative_prompt
        if args.sampling_preset == "distilled_two_stage":
            args.sample_two_stage = True

    args.video_cfg_scale = args.guidance_scale if args.video_cfg_scale is None else args.video_cfg_scale
    args.audio_cfg_scale = args.guidance_scale if args.audio_cfg_scale is None else args.audio_cfg_scale
    args.video_rescale_scale = args.rescale_scale if args.video_rescale_scale is None else args.video_rescale_scale
    args.audio_rescale_scale = args.rescale_scale if args.audio_rescale_scale is None else args.audio_rescale_scale
    args.video_modality_scale = 1.0 if args.video_modality_scale is None else args.video_modality_scale
    args.audio_modality_scale = 1.0 if args.audio_modality_scale is None else args.audio_modality_scale
    args.av_bimodal_cfg = False if args.av_bimodal_cfg is None else args.av_bimodal_cfg
    args.av_bimodal_scale = args.video_modality_scale if args.av_bimodal_scale is None else args.av_bimodal_scale
    if args.cfg_scale is None and args.video_cfg_scale != args.guidance_scale:
        args.cfg_scale = args.video_cfg_scale

    return args


# ---------------------------------------------------------------------------
# Attention flag setup — same as training script's load_transformer expects
# ---------------------------------------------------------------------------


def _configure_attention_flags(args: argparse.Namespace) -> None:
    # If a training-style boolean flag was explicitly set, it takes priority
    if args.flash_attn or args.flash3 or args.sdpa or args.xformers:
        # Flags already set by argparse; ensure the others are False
        args.flash_attn = bool(args.flash_attn)
        args.flash3 = bool(args.flash3)
        args.sdpa = bool(args.sdpa)
        args.xformers = bool(args.xformers)
        return

    # Otherwise derive from --attn_mode
    attn_mode = (args.attn_mode or "torch").lower()
    args.sdpa = attn_mode == "sdpa"
    args.flash_attn = attn_mode in {"flash", "flash2"}
    args.flash3 = attn_mode == "flash3"
    args.xformers = attn_mode == "xformers"


# ---------------------------------------------------------------------------
# LoRA merging
# ---------------------------------------------------------------------------


def _merge_lora_weights(
    transformer: torch.nn.Module,
    weights: list[str],
    multipliers: Optional[list[float]],
    include_patterns: Optional[list[str]],
    exclude_patterns: Optional[list[str]],
    ltx2_checkpoint: Optional[str] = None,
    text_encoder: Optional[torch.nn.Module] = None,
) -> None:
    for idx, path in enumerate(weights):
        multiplier = multipliers[idx] if multipliers and len(multipliers) > idx else 1.0
        logger.info("Merging LoRA: %s (multiplier=%.3f)", path, multiplier)
        lora_sd = load_file(path)
        metadata = load_lora_metadata(path)
        network_module_name = infer_lora_network_module(metadata, lora_sd)
        network_module = import_lora_network_module(network_module_name)

        # Auto-detect connector LoRA and attach connectors to wrapper for merge
        has_connector_lora = any("embeddings_connector" in k for k in lora_sd.keys())
        if has_connector_lora and not getattr(transformer, "has_connectors", lambda: False)():
            if ltx2_checkpoint is None:
                logger.warning(
                    "LoRA contains connector weights but --ltx2_checkpoint not provided; connector LoRA will be skipped."
                )
            else:
                from musubi_tuner.ltx_2.loader.sft_loader import SafetensorsModelStateDictLoader

                config = SafetensorsModelStateDictLoader().metadata(str(ltx2_checkpoint))
                video_conn, audio_conn = lora_ltx2.load_connectors_from_checkpoint(
                    str(ltx2_checkpoint),
                    config,
                    audio_video=True,
                    device=next(transformer.parameters()).device,
                    dtype=next(transformer.parameters()).dtype,
                )
                transformer.load_connectors(video_conn, audio_conn)
                logger.info("Attached connectors to wrapper for connector LoRA merge")

        net = network_module.create_arch_network_from_weights(
            multiplier,
            lora_sd,
            unet=transformer,
            for_inference=True,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
        module_count = len(getattr(net, "text_encoder_loras", [])) + len(getattr(net, "unet_loras", []))
        if module_count == 0:
            raise ValueError(
                f"LoRA {path} matched zero modules using {network_module_name}. "
                "Check --include_patterns/--exclude_patterns and adapter metadata."
            )
        apply_mode = apply_lora_network_for_inference(
            net,
            transformer,
            lora_sd,
            merge_device=next(transformer.parameters()).device,
            non_blocking=True,
            prefer_merge=True,
        )
        logger.info("Applied LoRA via %s using %s", apply_mode, network_module_name)

        # Copy merged connector weights back to text encoder if available
        if has_connector_lora and text_encoder is not None and getattr(transformer, "has_connectors", lambda: False)():
            if hasattr(text_encoder, "embeddings_connector"):
                text_encoder.embeddings_connector.load_state_dict(transformer.embeddings_connector.state_dict(), strict=False)
                logger.info("Copied merged connector LoRA weights to text encoder (video)")
            if hasattr(text_encoder, "audio_embeddings_connector") and hasattr(transformer, "audio_embeddings_connector"):
                text_encoder.audio_embeddings_connector.load_state_dict(
                    transformer.audio_embeddings_connector.state_dict(), strict=False
                )
                logger.info("Copied merged connector LoRA weights to text encoder (audio)")


# ---------------------------------------------------------------------------
# Prompt list construction — same format as training sampling
# ---------------------------------------------------------------------------


def _build_prompt_list(
    trainer: LTX2NetworkTrainer,
    args: argparse.Namespace,
    accelerator: Accelerator,
) -> List[dict]:
    """Build a list of sample_parameter dicts, same format sample_image_inference expects."""
    if args.sample_prompts:
        # File-based prompts: reuse training's prompt loading + defaults
        args.sample_num_frames = args.frame_count
        prompts = trainer.process_sample_prompts(args, accelerator, args.sample_prompts) or []
        return prompts

    # Single prompt from CLI
    prompt = args.prompt or ""
    sample = {
        "prompt": prompt,
        "negative_prompt": args.negative_prompt or "",
        "height": args.height,
        "width": args.width,
        "frame_count": args.frame_count,
        "frame_rate": args.frame_rate,
        "sample_steps": args.sample_steps,
        "sigma_schedule": args.sample_sigma_schedule,
        "sample_sampler": args.sample_sampler,
        "sampling_preset": args.sampling_preset,
        "guidance_scale": args.guidance_scale,
        "discrete_flow_shift": args.discrete_flow_shift,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "enum": 0,
    }
    return [sample]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _configure_windows_cli_encoding()
    args = parse_args()

    # Wire up aliases that the training code expects
    args.dit = args.ltx2_checkpoint
    args.sample_sampling_preset = args.sampling_preset
    args.sample_use_default_negative_prompt = args.use_default_negative_prompt
    if args.vae is None:
        args.vae = args.ltx2_checkpoint
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    _configure_attention_flags(args)

    use_cpu = args.device == "cpu"
    mixed_precision = args.mixed_precision if args.mixed_precision != "no" else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision, cpu=use_cpu)
    device = accelerator.device

    # Initialize trainer (only using its model loading + sampling methods, not training)
    trainer = LTX2NetworkTrainer()
    trainer.blocks_to_swap = int(args.blocks_to_swap or 0)
    trainer.handle_model_specific_args(args)

    # -- Load transformer --
    loading_device = "cpu" if trainer.blocks_to_swap > 0 else device
    transformer = trainer.load_transformer(
        accelerator=SimpleNamespace(device=device),
        args=args,
        dit_path=args.ltx2_checkpoint,
        attn_mode=args.attn_mode,
        split_attn=False,
        loading_device=loading_device,
        dit_weight_dtype=None,
    )

    # -- Merge LoRAs --
    if args.lora_weight:
        _merge_lora_weights(
            transformer,
            args.lora_weight,
            args.lora_multiplier,
            args.include_patterns,
            args.exclude_patterns,
            ltx2_checkpoint=str(args.ltx2_checkpoint),
        )
        clean_memory_on_device(device)

    # -- Block swap --
    if trainer.blocks_to_swap > 0:
        logger.info("Block swap: %d blocks to CPU from %s", trainer.blocks_to_swap, device)
        transformer.enable_block_swap(
            trainer.blocks_to_swap,
            device,
            supports_backward=False,
            use_pinned_memory=bool(args.use_pinned_memory_for_block_swap),
        )
        if hasattr(transformer, "move_to_device_except_swap_blocks"):
            transformer.move_to_device_except_swap_blocks(device)
        if hasattr(transformer, "switch_block_swap_for_inference"):
            transformer.switch_block_swap_for_inference()

    # -- Compile --
    if args.compile:
        transformer = trainer.compile_transformer(args, transformer)

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = _build_prompt_list(trainer, args, accelerator)
    if not prompts:
        logger.error("No prompts to generate. Exiting.")
        return

    if args.reference_image or args.reference_video:
        try:
            ref_path, use_v2v = _apply_reference_conditioning_overrides(
                prompts,
                reference_image=args.reference_image,
                reference_video=args.reference_video,
            )
        except FileNotFoundError as exc:
            logger.error(str(exc))
            return
        logger.info("Reference conditioning: %s via %s slot", ref_path, "V2V (v2v_ref_path)" if use_v2v else "I2V (image_path)")

    logger.info("Generating %d sample(s)...", len(prompts))

    # Set sampling gate args so should_sample_images() passes at step 0
    args.sample_at_first = True
    args.sample_every_n_steps = None
    args.sample_every_n_epochs = None

    # Delegate to the training script's sample_images(), which handles:
    # - batch prompt encoding (loads Gemma once for all prompts)
    # - staged offloading (transformer ↔ VAE on GPU)
    # - VAE pre-loading for offloading mode
    # - distributed inference support
    # - per-prompt error recovery
    # - RNG state preservation
    # - audio component management
    trainer.sample_images(
        accelerator=accelerator,
        args=args,
        epoch=0,
        steps=0,
        vae=None,  # lazy-loaded by sample_image_inference()
        transformer=transformer,
        sample_parameters=prompts,
        dit_dtype=trainer.dit_dtype or torch.float32,
    )

    logger.info("Generation complete. Outputs saved to %s", os.path.join(args.output_dir, "sample"))


if __name__ == "__main__":
    main()
