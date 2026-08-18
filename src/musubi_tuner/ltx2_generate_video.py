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
import json
import logging
import math
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
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
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
    reference_videos: list[str] | None = None,
) -> tuple[list[str], bool] | None:
    reference_videos = list(reference_videos or ())
    if not reference_image and not reference_video and not reference_videos:
        return None

    if reference_videos and (reference_image or reference_video):
        raise ValueError("--reference_videos cannot be combined with --reference_image or --reference_video")
    if reference_image and reference_video:
        logger.warning(
            "Both --reference_image and --reference_video given; --reference_video takes priority (V2V), --reference_image ignored."
        )

    ref_paths = reference_videos or [reference_video or reference_image]

    # Auto-detect: if an --reference_image path is actually a video by ext, use V2V slot.
    try:
        from musubi_tuner.dataset.image_video_dataset import VIDEO_EXTENSIONS

        video_exts = {e.lower() for e in VIDEO_EXTENSIONS}
    except Exception:
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    ext = os.path.splitext(ref_paths[0])[1].lower()
    use_v2v = bool(reference_video or reference_videos) or (ext in video_exts)

    for ref_path in ref_paths:
        if not os.path.isfile(ref_path):
            raise FileNotFoundError(f"Reference path does not exist: {ref_path}")

    for prompt_dict in prompts:
        # Explicit CLI references should override prompt-file paths and cached reference latents.
        prompt_dict.pop("image_path", None)
        prompt_dict.pop("conditioning_latent", None)
        prompt_dict.pop("v2v_ref_path", None)
        prompt_dict.pop("v2v_ref_paths", None)
        prompt_dict.pop("v2v_ref_latent", None)
        if use_v2v:
            if len(ref_paths) == 1:
                prompt_dict["v2v_ref_path"] = ref_paths[0]
            else:
                prompt_dict["v2v_ref_paths"] = ref_paths
        else:
            prompt_dict["image_path"] = ref_paths[0]

    return ref_paths, use_v2v


def _apply_composable_conditioning_overrides(prompts: list[dict], args: argparse.Namespace) -> None:
    """Inject inference-time composable-conditioning inputs into each prompt dict.

    Writes the per-prompt keys the sampler already reads (ref_audio_path / inpaint_mask_path /
    v2v_ref_path / inpaint_invert / inpaint_threshold). No-op when the corresponding flag is unset,
    so a run without these flags is byte-identical to prior behavior. Strategy and AV cross-attention
    mode are consumed straight from args by handle_model_specific_args and need no per-prompt key.
    """
    reference_audio = getattr(args, "reference_audio", None)
    inpaint_mask = getattr(args, "inpaint_mask", None)
    inpaint_source = getattr(args, "inpaint_source", None) or getattr(args, "reference_video", None)
    inpaint_invert = getattr(args, "inpaint_invert", None)
    inpaint_threshold = getattr(args, "inpaint_threshold", None)
    extend_prefix = getattr(args, "extend_prefix", None)
    extend_prefix_frames = getattr(args, "extend_prefix_frames", None)
    drive_video = getattr(args, "drive_video", None)

    # Audio-lock (a2v / audio_extend / audio_inpaint) resolves to one (path, mode) the sampler pins.
    audio_lock_path = audio_lock_mode = audio_lock_seconds = audio_lock_interval = None
    if getattr(args, "drive_audio", None):
        audio_lock_path, audio_lock_mode = args.drive_audio, "drive"
    elif getattr(args, "audio_prefix", None):
        audio_lock_path, audio_lock_mode = args.audio_prefix, "prefix"
        audio_lock_seconds = getattr(args, "audio_prefix_seconds", None)
    elif getattr(args, "audio_inpaint_audio", None):
        audio_lock_path, audio_lock_mode = args.audio_inpaint_audio, "inpaint"
        _iv = getattr(args, "audio_inpaint_interval", None)
        if _iv:
            try:
                _s, _e = (float(v) for v in str(_iv).split(","))
                audio_lock_interval = (_s, _e)
            except Exception as exc:
                raise ValueError(f"--audio_inpaint_interval must be 'start,end' seconds; got {_iv!r}") from exc
    if audio_lock_path and not os.path.isfile(audio_lock_path):
        raise FileNotFoundError(f"Audio-lock audio does not exist: {audio_lock_path}")

    # Outpaint lowers onto the inpaint path: keep the region from the source, generate the surround.
    outpaint_region = getattr(args, "outpaint_region", None)
    if outpaint_region:
        x, y, w, h = _parse_outpaint_region(outpaint_region)
        outpaint_source = getattr(args, "outpaint_source", None) or inpaint_source
        if not outpaint_source:
            raise ValueError("--outpaint_region needs a source video via --outpaint_source (or --reference_video)")
        if not os.path.isfile(outpaint_source):
            raise FileNotFoundError(f"Outpaint source does not exist: {outpaint_source}")
        os.makedirs(args.output_dir, exist_ok=True)
        inpaint_mask = _build_outpaint_mask_png(
            (x, y, w, h), int(args.width), int(args.height), os.path.join(args.output_dir, "outpaint_mask.png")
        )
        inpaint_source = outpaint_source

    if reference_audio and not os.path.isfile(reference_audio):
        raise FileNotFoundError(f"Reference audio does not exist: {reference_audio}")
    if extend_prefix and not os.path.isfile(extend_prefix):
        raise FileNotFoundError(f"Extend prefix does not exist: {extend_prefix}")
    if drive_video and not os.path.isfile(drive_video):
        raise FileNotFoundError(f"Driving video does not exist: {drive_video}")
    if inpaint_mask:
        if not os.path.isfile(inpaint_mask):
            raise FileNotFoundError(f"Inpaint mask does not exist: {inpaint_mask}")
        if not inpaint_source:
            raise ValueError("--inpaint_mask needs a source video via --inpaint_source (or --reference_video)")
        if not os.path.isfile(inpaint_source):
            raise FileNotFoundError(f"Inpaint source does not exist: {inpaint_source}")

    for prompt_dict in prompts:
        if reference_audio:
            prompt_dict["ref_audio_path"] = reference_audio
        if inpaint_mask:
            prompt_dict["inpaint_mask_path"] = inpaint_mask
            # Inpaint runs inside the v2v denoiser, so the source must populate the v2v ref slot.
            prompt_dict["v2v_ref_path"] = inpaint_source
            if inpaint_invert is not None:
                prompt_dict["inpaint_invert"] = bool(inpaint_invert)
            if inpaint_threshold is not None:
                prompt_dict["inpaint_threshold"] = float(inpaint_threshold)
        if extend_prefix:
            prompt_dict["extend_prefix_path"] = extend_prefix
            if extend_prefix_frames is not None:
                prompt_dict["extend_prefix_frames"] = int(extend_prefix_frames)
        if drive_video:
            prompt_dict["drive_video_path"] = drive_video
        if audio_lock_mode:
            prompt_dict["audio_lock_audio_path"] = audio_lock_path
            prompt_dict["audio_lock_mode"] = audio_lock_mode
            if audio_lock_seconds is not None:
                prompt_dict["audio_lock_seconds"] = float(audio_lock_seconds)
            if audio_lock_interval is not None:
                prompt_dict["audio_lock_interval"] = audio_lock_interval


def _apply_reference_target_ranges_from_adapters(prompts: list[dict], weights: list[str]) -> None:
    """Apply the ordered reference-routing contract stored by a trained adapter."""

    encoded_ranges = {raw for path in weights if (raw := load_lora_metadata(path).get("ss_ltx2_reference_target_frame_ranges"))}
    if not encoded_ranges:
        return
    if len(encoded_ranges) != 1:
        raise ValueError("The supplied LoRAs contain different reference target-frame ranges")

    try:
        ranges = json.loads(next(iter(encoded_ranges)))
    except json.JSONDecodeError as exc:
        raise ValueError("LoRA has invalid ss_ltx2_reference_target_frame_ranges metadata") from exc
    if not isinstance(ranges, list):
        raise ValueError("LoRA reference target-frame ranges metadata must be a list")

    from musubi_tuner.ltx2_conditioning_routing import normalize_reference_target_frame_ranges

    for prompt in prompts:
        ref_paths = prompt.get("v2v_ref_paths")
        if ref_paths is None:
            ref_paths = [prompt["v2v_ref_path"]] if prompt.get("v2v_ref_path") else []
        if not isinstance(ref_paths, (list, tuple)) or not all(isinstance(path, str) and path for path in ref_paths):
            raise ValueError("Prompt v2v_ref_paths must be a non-empty ordered list of paths")
        if not ref_paths:
            raise ValueError("Routed LoRA metadata requires V2V reference input")

        normalized = normalize_reference_target_frame_ranges(ranges, reference_count=len(ref_paths))
        routed_ranges = [list(value) for value in normalized]
        explicit_ranges = prompt.get("reference_target_frame_ranges")
        explicit_range = prompt.get("reference_target_frame_range")
        if explicit_ranges is not None and [list(value) for value in explicit_ranges] != routed_ranges:
            raise ValueError("Prompt reference_target_frame_ranges do not match the loaded LoRA metadata")
        if explicit_range is not None:
            if len(routed_ranges) != 1 or list(explicit_range) != routed_ranges[0]:
                raise ValueError("Prompt reference_target_frame_range does not match the loaded LoRA metadata")
        prompt["reference_target_frame_ranges"] = routed_ranges
        if len(routed_ranges) == 1:
            prompt["reference_target_frame_range"] = routed_ranges[0]


def _parse_outpaint_region(spec: str) -> tuple[int, int, int, int]:
    """Parse 'x,y,w,h' (pixels) into ints."""
    try:
        x, y, w, h = (int(round(float(v))) for v in spec.split(","))
    except Exception as exc:
        raise ValueError(f"--outpaint_region must be 'x,y,w,h' in pixels; got {spec!r}") from exc
    if w <= 0 or h <= 0:
        raise ValueError(f"--outpaint_region width/height must be positive; got {spec!r}")
    return x, y, w, h


def _build_outpaint_mask_png(region: tuple[int, int, int, int], width: int, height: int, out_path: str) -> str:
    """Write a still keep-mask: white (kept) inside the region, black (generated) outside.

    The inpaint clean-paste pins white-region tokens to the source and generates the rest, so a white
    interior rectangle realizes outpaint (keep the crop, generate the surround). A still broadcasts to
    every frame in the sampler.
    """
    import numpy as np
    from PIL import Image

    x, y, w, h = region
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    mask = np.zeros((height, width), dtype=np.uint8)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255
    Image.fromarray(mask, mode="L").save(out_path)
    return out_path


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
        "--ltx2_text_encoder_checkpoint",
        type=str,
        default=None,
        help="Packed LTX-2.5 Gemma text encoder and projection checkpoint.",
    )
    parser.add_argument(
        "--ltx2_audio_vae",
        type=str,
        default=None,
        help="Separate audio VAE/vocoder checkpoint for audio or AV generation.",
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
        choices=["2.0", "2.3", "2.5"],
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
        choices=["legacy", "defaults", "ltx20", "ltx23", "ltx23_hq", "ltx25", "ltx25_full", "distilled_two_stage"],
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
        choices=["auto", "ltx", "ltx_latent", "ltx23_distilled"],
        help=(
            "Sigma schedule. 'auto' uses LTX token-shifted sigmas, 'ltx_latent' is an "
            "explicit latent-aware LTX schedule, and the exact LTX-2.3 distilled sigmas "
            "are used for the distilled_two_stage preset."
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
        choices=["flash", "flash2", "flash3", "torch", "xformers", "sdpa", "sageattn", "cudnn"],
        help="Attention backend",
    )
    # Training-style boolean attention flags (for config portability)
    parser.add_argument("--flash_attn", action="store_true", help="Use FlashAttention (same as --attn_mode flash)")
    parser.add_argument("--flash3", action="store_true", help="Use FlashAttention 3 (same as --attn_mode flash3)")
    parser.add_argument(
        "--cudnn_attn",
        action="store_true",
        help="Use cuDNN-prioritized SDPA (same as --attn_mode cudnn). Outside torch.compile, PyTorch may fall "
        "back per shape to flash/efficient/math; under torch.compile the normal SDPA call is used.",
    )
    parser.add_argument("--sdpa", action="store_true", help="Use SDPA (same as --attn_mode sdpa)")
    parser.add_argument(
        "--ltx2_causal_temporal_attention",
        action="store_true",
        help=("Restrict video self-attention to the current and earlier video frames. Requires SDPA and is off by default."),
    )
    parser.add_argument(
        "--ltx2_soft_av_alignment",
        action="store_true",
        help="Bias AV cross-attention toward temporally aligned tokens. Requires AV mode and SDPA; off by default.",
    )
    parser.add_argument(
        "--ltx2_soft_av_alignment_sigma",
        type=float,
        default=1.0,
        help="Gaussian temporal width in seconds for --ltx2_soft_av_alignment. Default 1.0.",
    )
    parser.add_argument("--xformers", action="store_true", help="Use xformers (same as --attn_mode xformers)")
    parser.add_argument(
        "--sage_attn",
        action="store_true",
        help="Use SageAttention (inference only; requires the sageattention package; same as --attn_mode sageattn)",
    )
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
    int4_convrot_mode = parser.add_mutually_exclusive_group()
    int4_convrot_mode.add_argument(
        "--int4_convrot_base",
        action="store_true",
        help="Load a converter-produced packed int4cr checkpoint for W4A8 inference.",
    )
    int4_convrot_mode.add_argument(
        "--int4_convrot_dynamic",
        action="store_true",
        help="Quantize a bf16/fp16 checkpoint to INT4 ConvRot on load for W4A8 inference.",
    )
    parser.add_argument(
        "--int4_convrot_groupsize",
        default="auto",
        help="Dynamic INT4 ConvRot rotation group size or comma list (default: auto).",
    )
    parser.add_argument("--int4_convrot_no_mse_clip", action="store_true")
    parser.add_argument("--int4_convrot_scale_refine_steps", type=int, default=0)
    parser.add_argument(
        "--int4_convrot_group_scales",
        type=int,
        default=0,
        metavar="SIZE",
        help="Dynamic per-group INT4 weight scales (for example 128; 0/off by default).",
    )
    parser.add_argument(
        "--int4_convrot_group_ratio_q8",
        action="store_true",
        help="Use exact-mapping int16 Q8.8 storage for dynamic group ratios (requires group scales).",
    )
    parser.add_argument("--quantize_device", type=str, default=None, help="Device used for dynamic weight quantization.")
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
    parser.add_argument(
        "--reference_videos",
        type=str,
        nargs="+",
        default=None,
        help="Ordered V2V reference paths for multi-reference inference. "
        "Cannot be combined with --reference_image or --reference_video.",
    )

    # -- Composable conditioning (inference): surface the strategy/audio/inpaint knobs the sampler
    #    already supports. All default to current behavior, so omitting them is byte-identical. --
    parser.add_argument(
        "--ic_lora_strategy",
        type=str,
        default="auto",
        help="IC-LoRA inference strategy: auto|none|v2v|av_ic|video_ref_only_av|audio_ref_ic. "
        "'auto' infers from --lora_target_preset (default -> none). Set explicitly to drive a trained "
        "reference / audio-reference LoRA (e.g. av_ic, audio_ref_ic). av_ic/video_ref_only_av/audio_ref_ic "
        "require --ltx2_mode av.",
    )
    parser.add_argument(
        "--av_cross_attention_mode",
        type=str,
        default="both",
        help="AV cross-attention direction inside av_ic: both|a2v_only|v2a_only|none. Selects which "
        "modality attends across the other; only affects --ic_lora_strategy av_ic.",
    )
    parser.add_argument(
        "--reference_audio",
        type=str,
        default=None,
        help="Path to a reference audio clip for audio-reference conditioning (maps to ref_audio_path). "
        "Requires --ltx2_mode av and --ic_lora_strategy audio_ref_ic (or av_ic); ignored otherwise.",
    )
    parser.add_argument(
        "--inpaint_mask",
        type=str,
        default=None,
        help="Path to an inpaint mask (image/video) for masked V2V generation (maps to inpaint_mask_path). "
        "Needs a source video via --inpaint_source (or --reference_video) to clean-paste the kept region.",
    )
    parser.add_argument(
        "--inpaint_source",
        type=str,
        default=None,
        help="Source video the inpaint kept-region is clean-pasted from (drives the v2v slot). "
        "Defaults to --reference_video when omitted.",
    )
    parser.add_argument(
        "--inpaint_invert",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Invert the inpaint mask polarity at inference (maps to inpaint_invert).",
    )
    parser.add_argument(
        "--inpaint_threshold",
        type=float,
        default=None,
        help="Binarization threshold for the inpaint mask in [0,1] (maps to inpaint_threshold).",
    )
    parser.add_argument(
        "--extend_prefix",
        type=str,
        default=None,
        help="Path to a prefix video to continue (extend). Its first frames are encoded and locked at "
        "the start of generation; the rest is generated. No-v2v path only (do not combine with "
        "--reference_video).",
    )
    parser.add_argument(
        "--extend_prefix_frames",
        type=int,
        default=None,
        help="Pixel frames of --extend_prefix to lock (default: ~half of --frame_count). Must be < frame_count.",
    )
    parser.add_argument(
        "--outpaint_region",
        type=str,
        default=None,
        help="Outpaint keep-region 'x,y,w,h' in pixels: the region is kept from --outpaint_source and the "
        "surrounding area is generated. Realized via the inpaint path (auto-built rectangular keep-mask).",
    )
    parser.add_argument(
        "--outpaint_source",
        type=str,
        default=None,
        help="Source video whose kept region is preserved during outpaint. Defaults to --reference_video.",
    )
    parser.add_argument(
        "--drive_video",
        type=str,
        default=None,
        help="Video-to-audio (v2a): lock this full video and generate its audio. Requires --ltx2_mode av "
        "(optionally --av_cross_attention_mode v2a_only). Do not combine with --reference_video.",
    )
    parser.add_argument(
        "--drive_audio",
        type=str,
        default=None,
        help="Audio-to-video (a2v): lock this full audio and generate video from it. Requires --ltx2_mode av "
        "(optionally --av_cross_attention_mode a2v_only).",
    )
    parser.add_argument(
        "--audio_prefix",
        type=str,
        default=None,
        help="Audio extend: lock the start of this audio and generate the continuation. Requires --ltx2_mode av. "
        "Use --audio_prefix_seconds to set how much is locked.",
    )
    parser.add_argument(
        "--audio_prefix_seconds",
        type=float,
        default=None,
        help="Seconds of --audio_prefix to lock (default: ~half the clip).",
    )
    parser.add_argument(
        "--audio_inpaint_audio",
        type=str,
        default=None,
        help="Audio inpaint: keep this audio everywhere except --audio_inpaint_interval, which is regenerated. "
        "Requires --ltx2_mode av.",
    )
    parser.add_argument(
        "--audio_inpaint_interval",
        type=str,
        default=None,
        help="'start,end' seconds: the audio span to regenerate during --audio_inpaint_audio (the rest is kept).",
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
    parser.add_argument(
        "--temporal_upsampler_path",
        type=str,
        default=None,
        help="Path to temporal upsampler (fps x2); chains with the spatial one (temporal first), stage 1 runs at half fps.",
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
    if (args.int4_convrot_base or args.int4_convrot_dynamic) and (args.fp8_base or args.fp8_scaled or args.nf4_base):
        raise ValueError("INT4 ConvRot inference is mutually exclusive with FP8 and NF4 base modes")
    from musubi_tuner.modules.int4_convrot_utils import validate_int4_convrot_scale_group_size

    validate_int4_convrot_scale_group_size(args.int4_convrot_group_scales)
    if args.int4_convrot_scale_refine_steps < 0:
        raise ValueError("--int4_convrot_scale_refine_steps must be >= 0")
    if (
        args.int4_convrot_scale_refine_steps or args.int4_convrot_group_scales or args.int4_convrot_group_ratio_q8
    ) and not args.int4_convrot_dynamic:
        raise ValueError(
            "--int4_convrot_scale_refine_steps/--int4_convrot_group_scales/--int4_convrot_group_ratio_q8 apply only with "
            "--int4_convrot_dynamic; prepacked checkpoints already carry their scales"
        )
    if args.int4_convrot_group_ratio_q8 and not args.int4_convrot_group_scales:
        raise ValueError("--int4_convrot_group_ratio_q8 requires --int4_convrot_group_scales")
    if args.ltx2_causal_temporal_attention:
        if args.ltx_mode == "audio":
            raise ValueError("--ltx2_causal_temporal_attention requires a video generation path")
        if not (args.sdpa or args.attn_mode == "sdpa"):
            raise ValueError("--ltx2_causal_temporal_attention requires --sdpa or --attn_mode sdpa")
    if not math.isfinite(args.ltx2_soft_av_alignment_sigma) or args.ltx2_soft_av_alignment_sigma <= 0.0:
        raise ValueError("--ltx2_soft_av_alignment_sigma must be finite and greater than zero")
    if args.ltx2_soft_av_alignment:
        if args.ltx_mode != "av":
            raise ValueError("--ltx2_soft_av_alignment requires --ltx2_mode av")
        if not (args.sdpa or args.attn_mode == "sdpa"):
            raise ValueError("--ltx2_soft_av_alignment requires --sdpa or --attn_mode sdpa")

    preset = get_ltx2_sampling_preset(args.sampling_preset, ltx_version=args.ltx_version)
    if args.sampling_preset == "defaults" and args.ltx_version == "2.5":
        args.sampling_preset = "ltx25"
        args.sample_sampling_preset = "ltx25"
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
        if args.sampling_preset in {"ltx25", "distilled_two_stage"}:
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
    if (
        args.flash_attn
        or args.flash3
        or args.sdpa
        or args.xformers
        or getattr(args, "sage_attn", False)
        or getattr(args, "cudnn_attn", False)
    ):
        # Flags already set by argparse; ensure the others are False
        args.flash_attn = bool(args.flash_attn)
        args.flash3 = bool(args.flash3)
        args.sdpa = bool(args.sdpa)
        args.xformers = bool(args.xformers)
        args.sage_attn = bool(getattr(args, "sage_attn", False))
        args.cudnn_attn = bool(getattr(args, "cudnn_attn", False))
        return

    # Otherwise derive from --attn_mode
    attn_mode = (args.attn_mode or "torch").lower()
    args.sdpa = attn_mode == "sdpa"
    args.flash_attn = attn_mode in {"flash", "flash2"}
    args.flash3 = attn_mode == "flash3"
    args.xformers = attn_mode == "xformers"
    args.sage_attn = attn_mode in {"sageattn", "sage_attention", "sage"}
    args.cudnn_attn = attn_mode in {"cudnn", "pytorch_cudnn"}


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
        # Forward CLI generation params into file prompts.
        for prompt_index, _p in enumerate(prompts):
            if isinstance(_p, dict):
                _p["frame_count"] = args.frame_count
                _p["frame_rate"] = args.frame_rate
                _p["sample_steps"] = args.sample_steps
                _p["seed"] = args.seed + prompt_index
                if getattr(args, "guidance_scale", None) is not None:
                    _p["guidance_scale"] = args.guidance_scale
                if getattr(args, "height", None) is not None:
                    _p["height"] = args.height
                if getattr(args, "width", None) is not None:
                    _p["width"] = args.width
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

    # Standalone AV generation should produce a playable file with its generated
    # audio, rather than silently leaving audio disabled or as a sidecar WAV.
    if args.ltx_mode in {"av", "va"} and not args.sample_disable_audio:
        args.sample_merge_audio = True

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
            BlockSwapConfig(
                device=device,
                supports_backward=False,
                use_pinned_memory=bool(args.use_pinned_memory_for_block_swap),
                h2d_only=True,
            ),
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

    if args.reference_image or args.reference_video or args.reference_videos:
        try:
            ref_paths, use_v2v = _apply_reference_conditioning_overrides(
                prompts,
                reference_image=args.reference_image,
                reference_video=args.reference_video,
                reference_videos=args.reference_videos,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error(str(exc))
            return
        logger.info(
            "Reference conditioning: %s via %s slot",
            ref_paths,
            "V2V (v2v_ref_path[s])" if use_v2v else "I2V (image_path)",
        )

    try:
        _apply_reference_target_ranges_from_adapters(prompts, list(args.lora_weight or ()))
    except ValueError as exc:
        logger.error(str(exc))
        return

    if (
        args.reference_audio
        or args.inpaint_mask
        or args.extend_prefix
        or args.outpaint_region
        or args.drive_video
        or args.drive_audio
        or args.audio_prefix
        or args.audio_inpaint_audio
    ):
        try:
            _apply_composable_conditioning_overrides(prompts, args)
        except (FileNotFoundError, ValueError) as exc:
            logger.error(str(exc))
            return

    logger.info("Generating %d sample(s)...", len(prompts))
    sample_output_dir = os.path.join(args.output_dir, "sample")
    outputs_before = set()
    if os.path.isdir(sample_output_dir):
        outputs_before = {entry.path for entry in os.scandir(sample_output_dir) if entry.is_file()}

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

    outputs_after = (
        {entry.path for entry in os.scandir(sample_output_dir) if entry.is_file()} if os.path.isdir(sample_output_dir) else set()
    )
    if not outputs_after.difference(outputs_before):
        raise RuntimeError("Generation failed: no output file was produced. See the sampling error above.")
    logger.info("Generation complete. Outputs saved to %s", sample_output_dir)


if __name__ == "__main__":
    main()
