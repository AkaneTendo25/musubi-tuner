"""LTX-2 argument parser and training entry point."""

import argparse
import math
import os
import sys
import logging

from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner.ltx_2.env import apply_ltx2_tweaks
from musubi_tuner.ltx2_model_parallel import add_ltx2_model_parallel_args
from musubi_tuner.ltx2_remote_stage import add_ltx2_remote_stage_args
from musubi_tuner.ltx2_lycoris_runtime import (
    apply_lycoris_preset_before_network_creation,
    is_lycoris_requested,
    process_lycoris_config,
)
from musubi_tuner.ltx2_train_network import IC_LORA_STRATEGIES
from musubi_tuner.model_defaults import default_gemma_root_path, default_ltx2_checkpoint_path

logger = logging.getLogger(__name__)


def ltx2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add LTX-2-specific arguments to parser"""

    parser.set_defaults(network_module="networks.lora_ltx2")

    parser.add_argument(
        "--ltx2_checkpoint",
        type=str,
        default=default_ltx2_checkpoint_path(),
        help="Path to LTX-2 checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--gemma_root",
        type=str,
        default=default_gemma_root_path(),
        help="Local directory containing Gemma weights/tokenizer (used for sample prompts)",
    )
    parser.add_argument(
        "--gemma_safetensors",
        type=str,
        default=None,
        help="Path to a single Gemma safetensors file (for example, an fp8 export). Loads weights, config, and tokenizer from one file. No --gemma_root needed.",
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
        "--gemma_bnb_use_local_rank",
        action="store_true",
        help=(
            "For Gemma 8-bit/4-bit loading, place the quantized model on this process's LOCAL_RANK device. "
            "Useful for multi-GPU accelerate launches; disabled by default to preserve existing loading behavior."
        ),
    )
    parser.add_argument(
        "--gemma_fp8_weight_offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When using FP8 Gemma safetensors, offload FP8 linear weights to CPU RAM. "
            "Defaults to the LTX2_GEMMA_SAFETENSORS_WEIGHT_OFFLOAD environment variable when omitted."
        ),
    )

    parser.add_argument(
        "--ltx2_mode",
        "--ltx_mode",
        dest="ltx_mode",
        type=str,
        default="v",
        choices=["video", "av", "audio", "v", "a", "va"],
        help="Training modality.",
    )
    parser.add_argument(
        "--ltx_version",
        type=str,
        default="2.3",
        choices=["2.0", "2.3"],
        help=(
            "Target LTX major trainer behavior. "
            "2.0 keeps legacy defaults; 2.3 enables 2.3-oriented defaults when mode is not explicitly overridden."
        ),
    )
    parser.add_argument(
        "--ltx_version_check_mode",
        type=str,
        default="warn",
        choices=["off", "warn", "error"],
        help=(
            "How strictly to enforce --ltx_version vs checkpoint metadata consistency. "
            "'warn' logs mismatches, 'error' stops startup, 'off' disables checks."
        ),
    )
    parser.add_argument(
        "--ltx2_audio_only_model",
        action="store_true",
        help="Load physically audio-only LTX-2 transformer (omit video modules). Requires --ltx2_mode audio.",
    )
    add_ltx2_model_parallel_args(parser)
    add_ltx2_remote_stage_args(parser)
    parser.add_argument(
        "--split_attn_target",
        type=str,
        default=None,
        choices=["none", "all", "self", "cross", "text_cross", "av_cross", "video", "audio"],
        help=("Enable split attention for selected modules. Targets: none/all/self/cross/text_cross/av_cross/video/audio."),
    )
    parser.add_argument(
        "--split_attn_mode",
        type=str,
        default=None,
        choices=["batch", "query"],
        help="Split attention mode: batch (split by batch) or query (split by query length).",
    )
    parser.add_argument(
        "--split_attn_chunk_size",
        type=int,
        default=0,
        help="Chunk size for split_attn_mode=query. 0 uses the internal default (1024).",
    )
    parser.add_argument(
        "--ffn_chunk_target",
        type=str,
        default=None,
        choices=["none", "all", "video", "audio"],
        help="Enable FFN chunking for selected modules. Targets: none/all/video/audio.",
    )
    parser.add_argument(
        "--ffn_chunk_size",
        type=int,
        default=0,
        help="Chunk size for FFN chunking. 0 disables chunking.",
    )
    parser.add_argument(
        "--lora_target_preset",
        type=str,
        default="t2v",
        choices=[
            "t2v",
            "v2v",
            "video_sa",
            "video_sa_ff",
            "video_sa_ca_ff",
            "audio",
            "audio_v2a",
            "audio_ref_ic",
            "av_ic",
            "video_ref_only_av",
            "full",
            "lycoris",
        ],
        help=(
            "LoRA target preset: "
            "'t2v' = text-to-video (all attention, default), "
            "'v2v' = video-to-video/IC-LoRA (all attention + feed-forward), "
            "'video_sa' = video self-attention only, "
            "'video_sa_ff' = video self-attention + video feed-forward, "
            "'video_sa_ca_ff' = video self-attention + cross-attention + feed-forward, "
            "'audio' = audio-only (audio attn/ffn, no AV cross-modal), "
            "'audio_v2a' = audio preset plus video_to_audio_attn (audio attn/ffn + video_to_audio_attn), "
            "'audio_ref_ic' = ID-LoRA-style AV preset "
            "(audio attn/ffn + audio/video cross-modal both directions), "
            "'av_ic' = AV IC preset (use --av_cross_attention_mode / --av_multi_ref for AV variants), "
            "'video_ref_only_av' = AV preset for video-reference conditioning without reference audio, "
            "'lycoris' = LyCORIS attention-only target preset for LTX-2, "
            "'full' = all linear layers for LoRA targeting. "
            "Can be overridden by --network_args include_patterns=..."
        ),
    )
    parser.add_argument(
        "--train_connectors",
        action="store_true",
        help="Also apply LoRA to text connector modules (Embeddings1DConnector) in addition to "
        "transformer blocks. Requires caching with --cache_before_connector.",
    )
    parser.add_argument(
        "--ic_lora_strategy",
        type=str,
        default="auto",
        choices=list(IC_LORA_STRATEGIES),
        help=(
            "IC-LoRA conditioning strategy. "
            "'auto' infers the strategy from --lora_target_preset "
            "(uses 'v2v' for v2v, "
            "'audio_ref_ic' when --lora_target_preset=audio_ref_ic, "
            "'av_ic' when --lora_target_preset=av_ic, "
            "'video_ref_only_av' when --lora_target_preset=video_ref_only_av, else 'none'). "
            "'v2v' uses reference-video conditioning. "
            "'audio_ref_ic' uses reference-audio conditioning (ID-LoRA-style) in AV or audio-only mode. "
            "'av_ic' uses combined video+audio reference conditioning (requires --ltx2_mode av; "
            "use --av_cross_attention_mode / --av_multi_ref for AV variants). "
            "'video_ref_only_av' uses reference-video conditioning while still training target AV generation "
            "(requires --ltx2_mode av)."
        ),
    )
    parser.add_argument(
        "--av_cross_attention_mode",
        type=str,
        default="both",
        choices=["both", "a2v_only", "v2a_only", "none"],
        help=(
            "For --ic_lora_strategy av_ic: which AV cross-modal directions remain enabled. "
            "'both' keeps default bidirectional AV IC, "
            "'a2v_only' keeps audio-to-video only, "
            "'v2a_only' keeps video-to-audio only, "
            "'none' disables both AV cross-modal directions."
        ),
    )
    parser.add_argument(
        "--av_multi_ref",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For --ic_lora_strategy av_ic: mark the run as using multi-reference AV IC. "
            "The backend already accepts multiple provided reference tensors; this flag exposes the intent in UI/metadata."
        ),
    )
    parser.add_argument(
        "--audio_ref_use_negative_positions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For --ic_lora_strategy audio_ref_ic: place reference-audio token positions in negative time.",
    )
    parser.add_argument(
        "--audio_ref_mask_cross_attention_to_reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For --ic_lora_strategy audio_ref_ic: mask A2V cross-attention so video attends only to target audio, "
            "not reference-audio tokens."
        ),
    )
    parser.add_argument(
        "--audio_ref_mask_reference_from_text_attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For --ic_lora_strategy audio_ref_ic: block reference-audio tokens from attending to text tokens "
            "(target-audio tokens still attend to text; currently ignored in av_ic)."
        ),
    )
    parser.add_argument(
        "--audio_ref_identity_guidance_scale",
        type=float,
        default=0.0,
        help=(
            "For --ic_lora_strategy audio_ref_ic sampling: identity guidance scale. "
            "Runs an extra forward pass without reference audio to isolate and amplify "
            "the speaker identity contribution. 0.0 disables. Recommended: 3.0."
        ),
    )
    parser.add_argument(
        "--av_bimodal_cfg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable audio-video bimodal CFG during sampling. Runs an extra forward pass "
            "with cross-modal attention (A2V, V2A) disabled to strengthen independent "
            "modality generation. Used by the ID-LoRA inference path."
        ),
    )
    parser.add_argument(
        "--av_bimodal_scale",
        type=float,
        default=None,
        help="Scale for AV bimodal CFG. Applied as (scale-1) * (cond - bimodal). Default: 3.0.",
    )
    parser.add_argument(
        "--sample_sampling_preset",
        "--sampling_preset",
        type=str,
        default="defaults",
        choices=["legacy", "defaults", "ltx20", "ltx23", "ltx23_hq", "distilled_two_stage"],
        help=(
            "Sampling defaults for validation previews. 'defaults' selects the preset for --ltx_version; "
            "use 'legacy' to bypass version preset defaults."
        ),
    )
    parser.add_argument(
        "--sample_use_default_negative_prompt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the default LTX negative prompt when a CFG sample has no --n/negative_prompt.",
    )
    parser.add_argument(
        "--sample_sigma_schedule",
        type=str,
        default="auto",
        choices=["auto", "ltx", "ltx23_distilled"],
        help=(
            "Sigma schedule for LTX-2 previews. 'auto' uses LTX token-shifted sigmas, "
            "and uses the exact LTX-2.3 distilled sigmas for the distilled_two_stage preset."
        ),
    )
    parser.add_argument(
        "--sample_sampler",
        type=str,
        default="auto",
        choices=["auto", "euler", "res_2s"],
        help="Sampler for LTX-2 previews. 'auto' uses res_2s for full presets and Euler for distilled_two_stage.",
    )
    parser.add_argument(
        "--video_cfg_scale",
        type=float,
        default=None,
        help="Video CFG scale for LTX-2 sampling. Defaults to cfg_scale/guidance_scale unless a preset sets it.",
    )
    parser.add_argument(
        "--audio_cfg_scale",
        type=float,
        default=None,
        help="Audio CFG scale for LTX-2 sampling. Defaults to cfg_scale/guidance_scale unless a preset sets it.",
    )
    parser.add_argument(
        "--video_modality_scale",
        type=float,
        default=None,
        help="Video A2V modality guidance scale. Default preset uses 3.0.",
    )
    parser.add_argument(
        "--audio_modality_scale",
        type=float,
        default=None,
        help="Audio V2A modality guidance scale. Default preset uses 3.0.",
    )
    parser.add_argument(
        "--video_rescale_scale",
        type=float,
        default=None,
        help="Video CFG rescale scale. Defaults to --rescale_scale unless a preset sets it.",
    )
    parser.add_argument(
        "--audio_rescale_scale",
        type=float,
        default=None,
        help="Audio CFG rescale scale. Defaults to --rescale_scale unless a preset sets it.",
    )
    parser.add_argument(
        "--stg_scale",
        type=float,
        default=None,
        help=(
            "Spatio-Temporal Guidance (STG) scale. None inherits from --sample_sampling_preset; 0.0 disables STG. "
            "When > 0, runs a perturbed forward with self-attention skipped at --stg_blocks "
            "and steers x0 by stg_scale * (cond - perturbed). Default preset uses 1.0."
        ),
    )
    parser.add_argument(
        "--stg_blocks",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Transformer block indices to perturb for STG. None = all blocks. "
            "Default preset targets a single late block, e.g. --stg_blocks 29."
        ),
    )
    parser.add_argument(
        "--stg_mode",
        type=str,
        default=None,
        choices=["video", "audio", "both"],
        help=(
            "Which modality to perturb for STG. None inherits from --sample_sampling_preset; "
            "'video' skips video self-attn, 'audio' skips audio self-attn, 'both' skips both (AV mode)."
        ),
    )
    parser.add_argument(
        "--rescale_scale",
        type=float,
        default=0.0,
        help=(
            "CFG★ rescaling strength after CFG+STG. 0.0 disables (default). "
            "LTX-2.3 default is 0.9; prevents oversaturation from "
            "amplified guidance by rescaling prediction toward cond.std()."
        ),
    )
    parser.add_argument(
        "--separate_audio_buckets",
        action="store_true",
        default=None,
        help="Split LTX-2 buckets by audio presence to avoid mixed audio/non-audio batches.",
    )
    parser.add_argument(
        "--accumulation_group_by",
        type=str,
        default="none",
        choices=["none", "frames", "bucket", "dataset"],
        help=(
            "Opt-in dataloader ordering for gradient accumulation windows. "
            "'frames' keeps each accumulation window on one frame count, "
            "'bucket' keeps it on one full bucket key (resolution/frame/audio), "
            "'dataset' keeps it within one dataset section, and 'none' uses normal shuffling."
        ),
    )
    parser.add_argument(
        "--accumulation_group_remainder",
        type=str,
        default="drop",
        choices=["drop", "pad", "allow_mixed"],
        help=(
            "How --accumulation_group_by handles buckets that do not divide evenly by the accumulation window. "
            "'drop' skips incomplete windows, 'pad' repeats same-group batches to fill them, "
            "and 'allow_mixed' keeps all batches but may mix groups in the final windows."
        ),
    )
    parser.add_argument(
        "--caption_field",
        type=str,
        default=None,
        help=(
            "For JSONL datasets, use this metadata field as the training caption instead of 'caption'. "
            "Useful for I2V/reference datasets that store separate fields such as target_caption "
            "and reference_caption. Directory datasets still use caption_extension files."
        ),
    )
    parser.add_argument(
        "--audio_bucket_strategy",
        type=str,
        default=None,
        choices=["pad", "truncate"],
        help=(
            "Audio duration bucketing strategy. "
            "'pad' (default): round-to-nearest bucket boundary, pad shorter clips and mask loss. "
            "'truncate': floor to bucket boundary, truncate all clips to bucket length (no padding/masking needed)."
        ),
    )
    parser.add_argument(
        "--audio_bucket_interval",
        type=float,
        default=None,
        help="Audio bucket step size in seconds (default: 2.0). Controls how finely audio clips are grouped by duration.",
    )
    parser.add_argument(
        "--video_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to the video diffusion loss.",
    )
    parser.add_argument(
        "--audio_loss_weight",
        type=float,
        default=1.0,
        help="Weight applied to the audio diffusion loss.",
    )
    parser.add_argument(
        "--audio_loss_balance_mode",
        type=str,
        default="none",
        choices=["none", "inv_freq", "ema_mag", "uncertainty", "ogm_ge"],
        help=(
            "Optional dynamic balancing for audio loss. "
            "'none' keeps static --audio_loss_weight; "
            "'inv_freq' scales audio weight by inverse EMA of audio-batch frequency; "
            "'ema_mag' matches audio loss magnitude to a target fraction of video loss; "
            "'uncertainty' uses learnable log-variance scalars per modality "
            "(Kendall et al., CVPR 2018), no hyperparameters required; "
            "'ogm_ge' attenuates the lower-loss / faster-learning modality on each step "
            "and can optionally inject GE noise into its gradients."
        ),
    )
    parser.add_argument(
        "--audio_loss_balance_beta",
        type=float,
        default=0.01,
        help="EMA update factor for audio-batch frequency when --audio_loss_balance_mode=inv_freq.",
    )
    parser.add_argument(
        "--audio_loss_balance_eps",
        type=float,
        default=0.05,
        help="Minimum denominator for inverse-frequency audio weighting (prevents extreme weights).",
    )
    parser.add_argument(
        "--audio_loss_balance_min",
        type=float,
        default=0.05,
        help="Minimum clamp for effective audio loss weight after inverse-frequency scaling.",
    )
    parser.add_argument(
        "--audio_loss_balance_max",
        type=float,
        default=4.0,
        help="Maximum clamp for effective audio loss weight after inverse-frequency scaling.",
    )
    parser.add_argument(
        "--audio_loss_balance_ema_init",
        type=float,
        default=1.0,
        help="Initial EMA value used by audio loss balancing modes.",
    )
    parser.add_argument(
        "--audio_loss_balance_target_ratio",
        type=float,
        default=0.33,
        help="Target audio/video loss magnitude ratio when --audio_loss_balance_mode=ema_mag.",
    )
    parser.add_argument(
        "--audio_loss_balance_ema_decay",
        type=float,
        default=0.99,
        help="EMA decay for loss magnitude tracking when --audio_loss_balance_mode=ema_mag.",
    )
    parser.add_argument(
        "--uncertainty_lr",
        type=float,
        default=None,
        help="Learning rate for uncertainty weighting log-variance parameters. "
        "Defaults to --learning_rate. Only used with --audio_loss_balance_mode=uncertainty.",
    )
    parser.add_argument(
        "--ogm_ge_alpha",
        type=float,
        default=0.3,
        help="OGM-GE modulation strength. Higher values attenuate the dominant modality more strongly. "
        "Only used with --audio_loss_balance_mode=ogm_ge.",
    )
    parser.add_argument(
        "--ogm_ge_noise_std",
        type=float,
        default=0.0,
        help="Optional GE gradient-noise scale for OGM-GE. 0 disables noise injection. "
        "Only used with --audio_loss_balance_mode=ogm_ge.",
    )
    parser.add_argument(
        "--independent_audio_timestep",
        action="store_true",
        help="Sample independent timesteps for audio noising/conditioning in AV and audio modes.",
    )
    parser.add_argument(
        "--preserve_audio_timing",
        action="store_true",
        default=None,
        help=(
            "Preserve original audio duration by skipping audio time-stretching and duration alignment. "
            "Useful when you want audio timing/tempo to remain unchanged even if video frame count differs."
        ),
    )
    parser.add_argument(
        "--audio_only_sequence_resolution",
        type=int,
        default=64,
        help=(
            "Virtual pixel resolution used to derive sequence length for shifted_logit_normal "
            "in --ltx_mode audio. Set 0 to use cached virtual geometry."
        ),
    )
    parser.add_argument(
        "--shifted_logit_mode",
        type=str,
        default=None,
        choices=["legacy", "stretched"],
        help=(
            "Shifted logit-normal sigma sampler mode. "
            "'legacy' keeps historical behavior; 'stretched' enables the current Mar-2026 sampling path. "
            "If unset, defaults by --ltx_version (2.0->legacy, 2.3->stretched)."
        ),
    )
    parser.add_argument(
        "--shifted_logit_eps",
        type=float,
        default=1e-3,
        help="Numerical epsilon used by --shifted_logit_mode stretched (reflection floor and uniform lower bound).",
    )
    parser.add_argument(
        "--shifted_logit_uniform_prob",
        type=float,
        default=0.1,
        help="Uniform fallback probability used by --shifted_logit_mode stretched.",
    )
    parser.add_argument(
        "--shifted_logit_shift",
        type=float,
        default=None,
        help=(
            "Override the auto-calculated logit-normal shift value. "
            "Lower values bias toward low noise / fine details, higher values toward high noise / global structure. "
            "If unset, shift is computed dynamically from sequence length using the LTX-2 linear formula "
            "(anchored at 0.95 for 1024 tokens and 2.05 for 4096 tokens). "
            "By default non-audio training extrapolates outside those anchor values for shorter/longer sequences; "
            "--ltx2_mode audio clamps auto-computed shifts to the configured min/max shift bounds."
        ),
    )
    parser.add_argument(
        "--shifted_logit_clamp_auto_shift",
        action="store_true",
        help=(
            "Clamp auto-computed shifted_logit_normal shifts to "
            "[--shifted_logit_min_shift, --shifted_logit_max_shift]. "
            "Does not affect explicit --shifted_logit_shift overrides."
        ),
    )
    parser.add_argument(
        "--shifted_logit_min_shift",
        type=float,
        default=0.95,
        help="Lower clamp bound for auto-computed shifted_logit_normal shifts.",
    )
    parser.add_argument(
        "--shifted_logit_max_shift",
        type=float,
        default=2.05,
        help="Upper clamp bound for auto-computed shifted_logit_normal shifts.",
    )
    parser.add_argument(
        "--audio_silence_regularizer",
        action="store_true",
        help="Use synthetic silence audio latents for AV batches that are missing audio latents.",
    )
    parser.add_argument(
        "--audio_silence_regularizer_weight",
        type=float,
        default=1.0,
        help="Multiplier applied to audio loss on synthetic-silence fallback batches.",
    )
    parser.add_argument(
        "--audio_supervision_mode",
        type=str,
        default="off",
        choices=["off", "warn", "error"],
        help=(
            "Monitor AV audio supervision quality. "
            "'warn' logs periodic warnings when supervised-audio ratio is too low; "
            "'error' stops training; 'off' disables checks."
        ),
    )
    parser.add_argument(
        "--audio_supervision_warmup_steps",
        type=int,
        default=50,
        help="Number of expected AV batches to observe before audio supervision checks begin.",
    )
    parser.add_argument(
        "--audio_supervision_check_interval",
        type=int,
        default=50,
        help="Run audio supervision checks every N expected AV batches.",
    )
    parser.add_argument(
        "--audio_supervision_min_ratio",
        type=float,
        default=0.9,
        help="Minimum required supervised/expected audio ratio for AV training.",
    )
    parser.add_argument(
        "--min_audio_batches_per_accum",
        type=int,
        default=0,
        help=(
            "Minimum number of audio-bearing microbatches per gradient accumulation window. "
            "0 disables quota sampling and preserves existing random sampling behavior."
        ),
    )
    parser.add_argument(
        "--audio_batch_probability",
        type=float,
        default=None,
        help=(
            "Probability of selecting an audio-bearing batch when both audio/non-audio batches remain. "
            "Mutually exclusive with --min_audio_batches_per_accum. "
            "Unset keeps existing random sampling behavior."
        ),
    )

    parser.add_argument(
        "--lycoris_config",
        type=str,
        default=None,
        help=(
            "Path to LyCORIS TOML configuration file. Use this for module-level algorithm settings without bundled example files."
        ),
    )
    parser.add_argument(
        "--init_lokr_norm",
        type=float,
        default=None,
        help=(
            "Initialize LoKR network with perturbed normal distribution (e.g., 1e-3). "
            "Helps training stability. Only applies when using LoKR algorithm."
        ),
    )
    parser.add_argument(
        "--lycoris_quantized_base_check_mode",
        type=str,
        default="warn",
        choices=["off", "warn", "error"],
        help=(
            "LyCORIS-only compatibility check when base-model quantization flags are enabled. "
            "'warn' logs a warning, 'error' stops startup, 'off' disables checks."
        ),
    )
    parser.add_argument(
        "--ltx2_first_frame_conditioning_p",
        type=float,
        default=0.1,
        help="Probability of first-frame conditioning during training (keep frame 0 clean and set its timestep to 0).",
    )
    # ---- Endpoint-keyframe training (orthogonal to --ic_lora_strategy) -----
    parser.add_argument(
        "--keyframe_endpoint_training",
        action="store_true",
        help=(
            "Enable endpoint-keyframe training: extract first/last/random-interior latent "
            "frames of the target and append them as APPEND-GUIDE keyframe tokens (soft "
            "guidance — model is steered toward but not constrained to match them). For "
            "exact frame replacement (hard lock) use the dataset-level latent_idx workflow "
            "instead. Composes with any --ic_lora_strategy. Off by default."
        ),
    )
    parser.add_argument(
        "--keyframe_first_frame_p",
        type=float,
        default=1.0,
        help=(
            "Per-sample probability of appending the first latent frame as a keyframe when "
            "--keyframe_endpoint_training is set (independent Bernoulli per item in the "
            "batch). Default 1.0 (always)."
        ),
    )
    parser.add_argument(
        "--keyframe_last_frame_p",
        type=float,
        default=1.0,
        help=(
            "Per-sample probability of appending the last latent frame as a keyframe when "
            "--keyframe_endpoint_training is set (independent Bernoulli per item in the "
            "batch). Default 1.0 (always)."
        ),
    )
    parser.add_argument(
        "--keyframe_random_interior_p",
        type=float,
        default=0.0,
        help=(
            "Per-sample probability of appending random interior latent frames as keyframes "
            "when --keyframe_endpoint_training is set. Interior indices are shared across the "
            "batch; only the dropout decision is per-sample. Default 0.0 (off)."
        ),
    )
    parser.add_argument(
        "--keyframe_max_random_interior",
        type=int,
        default=0,
        help=(
            "Maximum number of random interior latent frames to append per batch when "
            "--keyframe_random_interior_p triggers. Default 0 (none)."
        ),
    )
    parser.add_argument(
        "--video_anchor_training",
        action="store_true",
        help=(
            "Enable video-anchor training: copy selected target latent frames back into the noisy input, "
            "zero their timesteps, and exclude them from loss. Off by default."
        ),
    )
    parser.add_argument(
        "--video_anchor_probability",
        type=float,
        default=0.5,
        help=("Per-sample probability of applying video-anchor training when --video_anchor_training is set. Default 0.5."),
    )
    parser.add_argument(
        "--video_anchor_count",
        type=int,
        default=1,
        help=("Number of random anchors to add per sample when --video_anchor_strategy includes random anchors. Default 1."),
    )
    parser.add_argument(
        "--video_anchor_strategy",
        type=str,
        default="endpoints_random",
        choices=["endpoints", "random", "endpoints_random"],
        help=(
            "Anchor placement strategy for --video_anchor_training. "
            "'endpoints' keeps first/last frames only, 'random' samples anchors uniformly, "
            "and 'endpoints_random' combines both."
        ),
    )
    parser.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う",
    )
    parser.add_argument(
        "--fp8_keep_blocks",
        type=str,
        default=None,
        help=(
            "Comma-separated transformer block indices to keep in high precision when --fp8_scaled is enabled. "
            "Example: --fp8_keep_blocks 0,1,2,45. Ranges like 0-2,45 are also accepted."
        ),
    )
    parser.add_argument(
        "--fp8_w8a8",
        action="store_true",
        help="Enable W8A8 activation quantization (saves VRAM by not storing dequantized weights "
        "in autograd graph). Requires --fp8_scaled and LoRA training.",
    )
    parser.add_argument(
        "--w8a8_mode",
        type=str,
        default="int8",
        choices=["int8", "fp8"],
        help="W8A8 quantization format: int8 (Turing+, default) or fp8 (Ada Lovelace+).",
    )
    parser.add_argument(
        "--fp8_gemm",
        action="store_true",
        help=(
            "Full-parameter fine-tuning in FP8: replace attention/FFN Linear layers with "
            "FP8 forward/backward GEMMs (torch._scaled_mm, per-tensor dynamic scaling). "
            "Requires full-FT and FP8 tensor cores (compute capability >= 8.9; Ada/Hopper). "
            "Optimizer-agnostic. Mutually exclusive with --qgalore_full_ft / LoRA / --fp8_scaled."
        ),
    )
    parser.add_argument(
        "--fp8_gemm_targets",
        type=str,
        default="video",
        help="Which LTX-2 Linear layers to run in FP8 (same tokens as --qgalore_targets: video/audio/attn/ff/blocks/all).",
    )
    parser.add_argument(
        "--fp8_gemm_grad_dtype",
        type=str,
        default="e4m3",
        choices=["e4m3", "e5m2"],
        help="FP8 format for gradients. e4m3 (default) is more accurate; e5m2 has wider range (safer against gradient spikes).",
    )
    parser.add_argument(
        "--fp8_gemm_min_numel",
        type=int,
        default=16384,
        help="Skip Linear layers with fewer than this many weight elements when applying --fp8_gemm.",
    )
    parser.add_argument(
        "--fp8_gemm_compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Region-compile the FP8 GEMM (torch.compile) to fuse the per-tensor scaling. "
        "On by default; ~halves the FP8 GEMM step time with no quality change. Use --no-fp8_gemm_compile to disable.",
    )
    parser.add_argument(
        "--int8_weights",
        action="store_true",
        help=(
            "Full-parameter fine-tuning with int8-quantized resident weights (1 byte/param, no bf16 master); "
            "the optimizer updates them in place with stochastic rounding. Forward/backward dequantize to bf16, "
            "so no FP8 tensor cores are required (runs on Ampere). Requires --fused_backward_pass and a factored "
            "optimizer (Adafactor) for the memory win. Mutually exclusive with --fp8_gemm / --qgalore_full_ft / --fp8_scaled."
        ),
    )
    parser.add_argument(
        "--int8_weights_targets",
        type=str,
        default="video",
        help="Which LTX-2 Linear layers to store in int8 (same tokens as --qgalore_targets: video/audio/attn/ff/blocks/all).",
    )
    parser.add_argument(
        "--int8_weights_min_numel",
        type=int,
        default=16384,
        help="Skip Linear layers with fewer than this many weight elements when applying --int8_weights.",
    )
    parser.add_argument(
        "--int8_weights_group_size",
        type=int,
        default=0,
        help="int8 quantization granularity for --int8_weights: 0 = row-wise (one scale per output channel); "
        ">0 = group-wise along the input dim (e.g. 128/256 — finer scales, lower error and less update drift).",
    )
    parser.add_argument(
        "--int8_weights_outlier_quantile",
        type=float,
        default=1.0,
        help="set the int8 scale from a per-row/group quantile of |w| instead of the absmax (default 1.0 = absmax). "
        "Values like 0.999 clip the top 0.1%% of weights to ±127·scale, giving the bulk a tighter grid — "
        "useful when a heavy outlier tail inflates absmax and coarsens every other weight.",
    )
    parser.add_argument(
        "--int8_weights_sparse_ratio",
        type=float,
        default=0.0,
        help="Keep the top fraction of |w| as an exact fp32 side-vector, excluded from the int8 grid so "
        "outliers don't coarsen the bulk weights. Default 0.0 = off; try 0.01. Alternative to "
        "--int8_weights_outlier_quantile (which clips the tail); use one or the other.",
    )
    parser.add_argument(
        "--nf4_base",
        action="store_true",
        help="use NF4 4-bit quantization for base DiT model (reduces VRAM ~75%%)",
    )
    parser.add_argument(
        "--nf4_block_size",
        type=int,
        default=32,
        help="block size for NF4 quantization (default 32)",
    )
    parser.add_argument(
        "--quantize_device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "gpu"],
        help="Device for NF4/FP8 quantization math (default: cuda). Overrides LTX2_NF4_CALC_DEVICE / LTX2_FP8_CALC_DEVICE env vars.",
    )
    parser.add_argument(
        "--loftq_init",
        action="store_true",
        help="use LoftQ initialization for LoRA (compensates NF4 quantization error, requires --nf4_base)",
    )
    parser.add_argument(
        "--loftq_iters",
        type=int,
        default=2,
        help="number of LoftQ alternating iterations (default 2)",
    )
    parser.add_argument(
        "--awq_calibration",
        action="store_true",
        help="experimental: use AWQ-style activation-aware calibration for NF4 (requires --nf4_base)",
    )
    parser.add_argument(
        "--awq_alpha",
        type=float,
        default=0.25,
        help="AWQ scaling strength (0=no effect, 1=full activation-aware, default 0.25)",
    )
    parser.add_argument(
        "--awq_num_batches",
        type=int,
        default=8,
        help="number of synthetic calibration batches for AWQ (default 8)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Default sample height for LTX-2 preview generation.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Default sample width for LTX-2 preview generation.",
    )
    parser.add_argument(
        "--sample_num_frames",
        type=int,
        default=45,
        help="Default frame count for LTX-2 preview generation.",
    )
    parser.add_argument(
        "--sample_with_offloading",
        action="store_true",
        help="Offload LTX-2 DiT to CPU between sampling prompts to save VRAM.",
    )
    parser.add_argument(
        "--precache_sample_prompts",
        action="store_true",
        help="Use precached Gemma embeddings for sample prompts (no Gemma load during training).",
    )
    parser.add_argument(
        "--use_precached_sample_prompts",
        action="store_true",
        help="Use precached Gemma embeddings for sample prompts (no Gemma load during training).",
    )
    parser.add_argument(
        "--sample_prompts_cache",
        type=str,
        default=None,
        help=(
            "Path to precached sample prompt embeddings (.pt). Defaults to "
            "the first dataset's cache_directory/ltx2_sample_prompts_cache.pt"
        ),
    )
    parser.add_argument(
        "--use_precached_sample_latents",
        action="store_true",
        help="Use precached I2V conditioning latents for sample prompts (no VAE encoder load during training).",
    )
    parser.add_argument(
        "--sample_latents_cache",
        type=str,
        default=None,
        help="Path to precached I2V conditioning latents (.pt) for sample prompts.",
    )
    parser.add_argument(
        "--sample_disable_audio",
        action="store_true",
        help="Disable audio decoding during LTX-2 preview sampling (AV mode).",
    )
    parser.add_argument(
        "--sample_audio_only",
        action="store_true",
        help="Generate audio-only previews during sampling (skip video decode/save).",
    )
    parser.add_argument(
        "--sample_disable_flash_attn",
        action="store_true",
        help="Disable FlashAttention during LTX-2 preview sampling (use SDPA).",
    )
    parser.add_argument(
        "--sample_i2v_token_timestep_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use LTX I2V token timestep masking during sampling "
            "(conditioned first-frame tokens use timestep=0 via video_conditioning_mask)."
        ),
    )
    parser.add_argument(
        "--sample_audio_subprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Decode audio previews in a separate subprocess (default: enabled). "
            "This prevents native crashes / OOM segfaults when loading the audio "
            "decoder on low-VRAM GPUs. Use --no-sample_audio_subprocess to decode "
            "audio in-process (requires enough GPU memory for audio decoder + vocoder)."
        ),
    )
    parser.add_argument(
        "--sample_merge_audio",
        action="store_true",
        help="Mux sample audio into the sample video (outputs *_av.mp4).",
    )
    parser.add_argument(
        "--sample_include_reference",
        action="store_true",
        help="Show V2V reference side-by-side with generated output in sample videos.",
    )
    parser.add_argument(
        "--reference_downscale",
        type=int,
        default=1,
        help="Spatial downscale factor for V2V references (1=same res, 2=half). Must be >= 1.",
    )
    parser.add_argument(
        "--reference_frames",
        type=int,
        default=1,
        help="Number of reference frames to use for V2V sampling. Images are repeated to fill this count.",
    )

    # Two-stage inference arguments
    parser.add_argument(
        "--sample_two_stage",
        action="store_true",
        help="Enable two-stage inference: generate at half resolution, then upsample and refine.",
    )
    parser.add_argument(
        "--spatial_upsampler_path",
        type=str,
        default=None,
        help="Path to spatial upsampler model (ltx-2-spatial-upscaler-x2-1.0.safetensors) for two-stage inference.",
    )
    parser.add_argument(
        "--distilled_lora_path",
        type=str,
        default=None,
        help="Path to distilled LoRA (ltx-2-19b-distilled-lora-384.safetensors) for two-stage refinement.",
    )
    parser.add_argument(
        "--sample_stage2_steps",
        type=int,
        default=3,
        help="Number of denoising steps for stage 2 refinement (default: 3 steps with 4 sigma values).",
    )
    parser.add_argument(
        "--sample_stage1_distilled_lora_multiplier",
        type=float,
        default=None,
        help=("Optional distilled LoRA multiplier for two-stage stage 1. If omitted, res_2s two-stage uses 0.25; Euler uses 0.0."),
    )
    parser.add_argument(
        "--sample_stage2_distilled_lora_multiplier",
        type=float,
        default=None,
        help=("Optional distilled LoRA multiplier for two-stage stage 2. If omitted, res_2s two-stage uses 0.5; Euler uses 1.0."),
    )

    parser.add_argument(
        "--sample_tiled_vae",
        action="store_true",
        help="Enable tiled VAE decoding during sampling to reduce VRAM usage.",
    )
    parser.add_argument(
        "--sample_vae_tile_size",
        type=int,
        default=512,
        help="Spatial tile size in pixels for tiled VAE decode (default: 512).",
    )
    parser.add_argument(
        "--sample_vae_tile_overlap",
        type=int,
        default=64,
        help="Spatial tile overlap in pixels for tiled VAE decode (default: 64).",
    )
    parser.add_argument(
        "--sample_vae_temporal_tile_size",
        type=int,
        default=0,
        help="Temporal tile size in frames for tiled VAE decode. 0=no temporal tiling (default: 0).",
    )
    parser.add_argument(
        "--sample_vae_temporal_tile_overlap",
        type=int,
        default=8,
        help="Temporal tile overlap in frames for tiled VAE decode (default: 8).",
    )
    parser.add_argument(
        "--blockwise_checkpointing",
        action="store_true",
        help="Enable block-wise weight offloading during backward (ultra-low VRAM).",
    )
    parser.add_argument(
        "--blocks_to_checkpoint",
        type=int,
        default=-1,
        help="Number of blocks to checkpoint. -1 = all (default), 0 = none, N = last N blocks. "
        "Use with --blockwise_checkpointing to trade VRAM for speed on 12-16GB cards.",
    )
    parser.add_argument(
        "--no_convert_to_comfy",
        action="store_false",
        dest="convert_to_comfy",
        default=True,
        help="Disable automatic conversion of saved LoRA to ComfyUI format. "
        "By default, both original and ComfyUI checkpoints are saved.",
    )
    parser.add_argument(
        "--save_original_lora",
        action="store_true",
        default=True,
        help="(Default: True) Keep the original non-Comfy LoRA alongside the ComfyUI-converted checkpoint. "
        "Use --no_save_original_lora to disable.",
    )
    parser.add_argument(
        "--no_save_original_lora",
        action="store_false",
        dest="save_original_lora",
        help="Delete the original LoRA after ComfyUI conversion, keeping only *.comfy.safetensors.",
    )

    # -- Preservation / regularization flags --
    parser.add_argument(
        "--blank_preservation",
        action="store_true",
        help="Regularize LoRA to not change blank-prompt output (MSE between LoRA ON/OFF with empty prompt).",
    )
    parser.add_argument(
        "--blank_preservation_args",
        type=str,
        nargs="*",
        help="Key=value args for blank preservation, e.g. multiplier=0.5",
    )
    parser.add_argument(
        "--dop",
        action="store_true",
        help="Differential Output Preservation: regularize LoRA to not change class-prompt output.",
    )
    parser.add_argument(
        "--dop_args",
        type=str,
        nargs="*",
        help="Key=value args for DOP, e.g. class=woman multiplier=1.0",
    )
    parser.add_argument(
        "--prior_divergence",
        action="store_true",
        help="Encourage LoRA output to diverge from base model on training prompts.",
    )
    parser.add_argument(
        "--prior_divergence_args",
        type=str,
        nargs="*",
        help="Key=value args for prior divergence, e.g. multiplier=0.1",
    )
    parser.add_argument(
        "--use_precached_preservation",
        action="store_true",
        help="Load preservation embeddings from precached .pt file instead of loading Gemma. "
        "Run ltx2_cache_text_encoder_outputs.py with --precache_preservation_prompts first.",
    )
    parser.add_argument(
        "--preservation_prompts_cache",
        type=str,
        default=None,
        help="Path to precached preservation prompt embeddings (.pt). "
        "Defaults to <cache_directory>/ltx2_preservation_cache.pt. Requires --use_precached_preservation.",
    )
    parser.add_argument(
        "--audio_dop",
        action="store_true",
        help="Audio DOP: preserve base model audio predictions on non-audio batches. "
        "Only active in AV mode (--ltx2_mode av). Adds +2 forwards and +1 backward on non-audio steps.",
    )
    parser.add_argument(
        "--audio_dop_args",
        type=str,
        nargs="*",
        help="Key=value args for audio DOP, e.g. multiplier=0.5",
    )

    # -- TARP / DCR (arXiv:2603.18600) --
    parser.add_argument(
        "--tarp",
        action="store_true",
        help="Enable TARP windowed cross-attention for audio-video temporal locality (arXiv:2603.18600).",
    )
    parser.add_argument(
        "--tarp_args",
        type=str,
        nargs="*",
        help="Key=value args for TARP, e.g. window_multiplier=3",
    )
    parser.add_argument(
        "--dcr",
        action="store_true",
        help="Enable DCR per-sample gradient routing for mixed audio/video batches (arXiv:2603.18600).",
    )
    parser.add_argument(
        "--dcr_args",
        type=str,
        nargs="*",
        help="Key=value args for DCR, e.g. reference_detach=true",
    )
    parser.add_argument(
        "--av_cross_grad_surgery",
        action="store_true",
        help=(
            "Enable branch-aware gradient scaling for AV cross-modal K/V projections. "
            "Only active in --ltx2_mode av. With no args, uses the OmniNFT A2V schedule "
            "a2v=0:0,1-10:0.1,40-47:0.3."
        ),
    )
    parser.add_argument(
        "--av_cross_grad_surgery_args",
        type=str,
        nargs="*",
        help=("Key=value args for AV cross grad surgery, e.g. a2v=0:0,1-10:0.1,40-47:0.3 v2a=40-47:0.3 projections=k,v"),
    )
    parser.add_argument(
        "--av_attention_loss_weighting",
        action="store_true",
        help="Enable AV cross-attention-derived loss weighting. Requires --ltx2_mode av.",
    )
    parser.add_argument(
        "--av_attention_loss_max",
        type=float,
        default=1.5,
        help="Maximum per-token multiplier for --av_attention_loss_weighting. Default: 1.5.",
    )
    parser.add_argument(
        "--av_attention_loss_warmup_steps",
        type=int,
        default=400,
        help="Warmup steps before --av_attention_loss_weighting reaches --av_attention_loss_max. Default: 400.",
    )

    # -- Audio Metrics --
    parser.add_argument(
        "--audio_metrics",
        action="store_true",
        help="Enable audio quality metrics logging. Tier 1 (latent-space) runs every step at ~0 cost. "
        "Tier 2 (mel-space) and Tier 3 (embedding-space) are opt-in via --audio_metrics_args.",
    )
    parser.add_argument(
        "--audio_metrics_args",
        type=str,
        nargs="*",
        help="Key=value args for audio metrics, e.g. mel_metrics=true fad=true clap_similarity=true "
        "latent_fd_compute_every=50 mel_compute_every=100",
    )

    # -- CREPA (Cross-frame Representation Alignment) --
    parser.add_argument(
        "--crepa",
        action="store_true",
        help="Enable CREPA temporal consistency regularization (arxiv 2506.09229). "
        "Aligns DiT hidden states across video frames via a small projector MLP.",
    )
    parser.add_argument(
        "--crepa_args",
        type=str,
        nargs="*",
        help="Key=value args for CREPA, e.g. student_block_idx=16 teacher_block_idx=32 "
        "lambda_crepa=0.1 tau=1.0 num_neighbors=2 schedule=constant normalize=true "
        "similarity_threshold=0.9 similarity_ema_decay=0.99 threshold_mode=permanent",
    )
    parser.add_argument(
        "--self_flow",
        action="store_true",
        help="Enable Self-Flow regularization (dual-timestep noising + EMA-teacher feature alignment). "
        "Supported for --ltx_mode video and --ltx_mode av (video branch only in av). "
        "Single-frame image-like samples are supported via --ltx_mode video.",
    )
    parser.add_argument(
        "--self_flow_args",
        type=str,
        nargs="*",
        help="Key=value args for Self-Flow, e.g. student_block_idx=16 teacher_block_idx=32 "
        "lambda_self_flow=0.1 temporal_mode=hybrid lambda_temporal=0.1 lambda_delta=0.05 "
        "temporal_tau=1.0 num_neighbors=2 temporal_granularity=patch patch_spatial_radius=1 "
        "patch_match_mode=soft patch_match_temperature=0.2 delta_num_steps=2 "
        "motion_weighting=teacher_delta motion_weight_strength=0.5 "
        "temporal_schedule=linear temporal_warmup_steps=200 temporal_max_steps=2000 mask_ratio=0.1 "
        "frame_level_mask=false teacher_mode=base mask_focus_loss=false max_loss=0.0 "
        "student_block_stochastic_range=2 teacher_momentum=0.999 "
        "dual_timestep=true student_block_ratio=0.3 teacher_block_ratio=0.7 projector_lr=5e-5",
    )

    # -- Latent temporal objectives --
    parser.add_argument(
        "--latent_temporal_weighting",
        action="store_true",
        help=(
            "Enable latent motion weighting for the video denoising loss. "
            "Clean latent frame deltas are converted into per-frame loss weights."
        ),
    )
    parser.add_argument(
        "--latent_temporal_weighting_args",
        type=str,
        nargs="*",
        help=("Key=value args for latent temporal weighting, e.g. alpha=0.5 mode=log normalize=mean clip_min=0.5 clip_max=2.0"),
    )
    parser.add_argument(
        "--latent_delta_loss",
        action="store_true",
        help=(
            "Enable latent temporal derivative matching for video training. "
            "By default this matches predicted x0 frame deltas to clean latent frame deltas."
        ),
    )
    parser.add_argument(
        "--latent_delta_loss_args",
        type=str,
        nargs="*",
        help=(
            "Key=value args for latent delta loss, e.g. "
            "weight=0.03 order=1 target=x0 sigma_min=0.05 sigma_max=0.85 "
            "second_order_weight=0.5 loss_type=mse"
        ),
    )

    # -- HFATO (High-Frequency Awareness Training Objective, ViBe) --
    parser.add_argument(
        "--hfato",
        action="store_true",
        help="Enable HFATO: degrades clean latents via downsample-upsample before "
        "noise addition, then supervises model to reconstruct original clean "
        "latents.  Forces high-frequency detail recovery (ViBe, arxiv 2603.23326).",
    )
    parser.add_argument(
        "--hfato_args",
        type=str,
        nargs="*",
        help="Key=value args for HFATO, e.g. scale_factor=0.5 interpolation=bilinear probability=1.0",
    )

    # -- Per-module learning rate groups --
    parser.add_argument(
        "--audio_lr",
        type=float,
        default=None,
        help="Learning rate for audio LoRA modules (audio_attn, audio_ff, cross-modal). "
        "Overridden by more specific --lr_args patterns. Defaults to --learning_rate.",
    )
    parser.add_argument(
        "--lr_args",
        type=str,
        nargs="*",
        default=None,
        help="Per-module learning rate overrides (pattern=lr). Patterns are matched via regex "
        "against LoRA module names. Example: --lr_args audio_attn=1e-6 audio_ff=1e-6 "
        "video_to_audio=1e-5",
    )
    parser.add_argument(
        "--lr_group_warmup_args",
        type=str,
        nargs="*",
        default=None,
        help="Optional per-group warmup overrides (pattern=steps) applied on top of the selected "
        "scheduler family. Patterns match optimizer group names such as unet_audio, "
        "unet_video, or unet_audio_attn. Example: --lr_group_warmup_args audio=500 video=1500",
    )

    # -- Per-module rank (dim) overrides --
    parser.add_argument(
        "--audio_dim",
        type=int,
        default=None,
        help="LoRA rank (dim) for audio modules (names containing 'audio_'). "
        "Defaults to --network_dim. Allows lower rank for audio to reduce overfitting.",
    )
    parser.add_argument(
        "--audio_alpha",
        type=float,
        default=None,
        help="LoRA alpha for audio modules. Defaults to --network_alpha. "
        "Typically set equal to --audio_dim for consistent scaling.",
    )

    # -- Caption dropout --
    parser.add_argument(
        "--caption_dropout_rate",
        type=float,
        default=0.0,
        help="Probability of dropping ALL text conditioning for each sample (0.0 = disabled). "
        "Zeros out both video and audio text embeddings and mask. "
        "For per-modality dropout, use --video_caption_dropout_rate / --audio_caption_dropout_rate.",
    )
    parser.add_argument(
        "--video_caption_dropout_rate",
        type=float,
        default=0.0,
        help="Probability of dropping video text conditioning per sample while keeping audio (0.0 = disabled). "
        "Applied independently before --caption_dropout_rate. AV mode only.",
    )
    parser.add_argument(
        "--audio_caption_dropout_rate",
        type=float,
        default=0.0,
        help="Probability of dropping audio text conditioning per sample while keeping video (0.0 = disabled). "
        "Applied independently before --caption_dropout_rate. AV mode only.",
    )
    parser.add_argument(
        "--tread",
        action="store_true",
        help=(
            "Enable TREAD token routing for LTX training. Bare --tread uses defaults: "
            "selection_ratio=0.5, start/end=3/-4 for LTX-2.3 or 2/-2 for LTX-2.0. "
            "Training-only; default target is video, with target=audio/both available for audio-enabled LTX paths."
        ),
    )
    parser.add_argument(
        "--tread_args",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional key=value settings for --tread. "
            "Examples: --tread --tread_args target=video selection_ratio=0.5 start_layer_idx=3 end_layer_idx=-4. "
            "Aliases: modality, ratio, start, end."
        ),
    )

    # -- Cross-Task Synergy (Harmony) --
    parser.add_argument(
        "--cts_lambda_video_driven",
        type=float,
        default=0.0,
        help="Weight for video-driven audio auxiliary loss (clean video + noisy audio). "
        "0 = disabled. Harmony (2025) uses 0.3. Requires --ltx2_mode av with audio data.",
    )
    parser.add_argument(
        "--cts_lambda_audio_driven",
        type=float,
        default=0.0,
        help="Weight for audio-driven video auxiliary loss (noisy video + clean audio). "
        "0 = disabled. Harmony (2025) uses 0.1. Requires --ltx2_mode av with audio data.",
    )

    # -- Modality freezing (G2D) --
    parser.add_argument(
        "--modality_freeze_check_interval",
        type=int,
        default=0,
        help="Check modality freeze state every N steps (0 = disabled). "
        "When audio loss EMA / video loss EMA drops below --modality_freeze_ratio_threshold, "
        "audio LoRA params are frozen. Vice versa for video.",
    )
    parser.add_argument(
        "--modality_freeze_ratio_threshold",
        type=float,
        default=0.5,
        help="Audio/video loss EMA ratio below which the lower-loss modality's LoRA is frozen. "
        "Default 0.5: freeze audio when audio_loss < 0.5 * video_loss.",
    )
    parser.add_argument(
        "--modality_freeze_warmup_steps",
        type=int,
        default=100,
        help="Steps before modality freezing can activate (default: 100).",
    )
    parser.add_argument(
        "--modality_freeze_ema_decay",
        type=float,
        default=0.99,
        help="EMA decay for per-modality loss tracking in the freezer (default: 0.99).",
    )

    return parser


def main() -> None:
    """Main training entry point"""
    parser = setup_parser_common()
    parser = ltx2_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)
    if hasattr(args, "ltx_mode"):
        short_map = {"v": "video", "a": "audio", "va": "av"}
        if args.ltx_mode in short_map:
            args.ltx_mode = short_map[args.ltx_mode]
    apply_ltx2_tweaks(args)
    args.differential_guidance_scale = float(getattr(args, "differential_guidance_scale", 3.0))
    if not math.isfinite(args.differential_guidance_scale):
        raise ValueError("--differential_guidance_scale must be finite.")
    if bool(getattr(args, "differential_guidance", False)) and (
        getattr(args, "ltx_mode", "video") == "audio" or bool(getattr(args, "ltx2_audio_only_model", False))
    ):
        raise ValueError("--differential_guidance requires a video/main prediction loss and cannot be used in audio-only mode.")
    if getattr(args, "auto_blocks_to_checkpoint", False):
        if getattr(args, "blockwise_checkpointing", False) and int(getattr(args, "blocks_to_swap", 0) or 0) > 0:
            if int(getattr(args, "blocks_to_checkpoint", -1)) == -1:
                args.blocks_to_checkpoint = int(getattr(args, "blocks_to_swap", 0) or 0)
            logger.warning(
                "Using blockwise checkpointing with block swap enabled (slower but lower VRAM). "
                "blocks_to_checkpoint=%s blocks_to_swap=%s",
                args.blocks_to_checkpoint,
                args.blocks_to_swap,
            )

    blocks_to_swap = int(getattr(args, "blocks_to_swap", 0) or 0)
    has_bw_checkpointing = getattr(args, "blockwise_checkpointing", False)
    # Auto-enable LTX2_SWAP_TRAIN_FULL for proper block swapping during training
    if blocks_to_swap > 0:
        current_val = os.environ.get("LTX2_SWAP_TRAIN_FULL")
        # Always set to "1" for training with block swap (override any previous value)
        os.environ["LTX2_SWAP_TRAIN_FULL"] = "1"
        if current_val is None:
            logger.info("Auto-enabled LTX2_SWAP_TRAIN_FULL=1 (blocks_to_swap=%d)", blocks_to_swap)
        elif current_val != "1":
            logger.info("Overriding LTX2_SWAP_TRAIN_FULL from '%s' to '1' (blocks_to_swap=%d)", current_val, blocks_to_swap)
        else:
            logger.info("LTX2_SWAP_TRAIN_FULL=1 already set (blocks_to_swap=%d)", blocks_to_swap)

    explicit_lora_preset = any(arg == "--lora_target_preset" or arg.startswith("--lora_target_preset=") for arg in sys.argv)
    explicit_ic_strategy = any(arg == "--ic_lora_strategy" or arg.startswith("--ic_lora_strategy=") for arg in sys.argv)

    if getattr(args, "dit", None) is not None and args.dit != args.ltx2_checkpoint:
        logger.warning("Ignoring --dit for LTX-2; using --ltx2_checkpoint instead")
    args.dit = args.ltx2_checkpoint

    if getattr(args, "vae", None) is not None and args.vae != args.ltx2_checkpoint:
        logger.warning("Ignoring --vae for LTX-2; using --ltx2_checkpoint instead")
    args.vae = args.ltx2_checkpoint

    if getattr(args, "weighting_scheme", None) not in {None, "none"}:
        logger.warning("Ignoring --weighting_scheme for LTX-2; forcing weighting_scheme=none")
    args.weighting_scheme = "none"

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    uses_lycoris_module = is_lycoris_requested(args)
    requested_ic_strategy = str(getattr(args, "ic_lora_strategy", "auto") or "auto").lower()

    # Inject lora_target_preset into network_args (LTX-2 specific, non-LyCORIS only)
    if getattr(args, "ltx_mode", "video") == "audio" and not explicit_lora_preset and not uses_lycoris_module:
        if args.network_args is None:
            args.network_args = []
        if not any(arg.startswith("include_patterns=") for arg in args.network_args):
            args.lora_target_preset = "audio"
    elif requested_ic_strategy == "audio_ref_ic" and not explicit_lora_preset and not uses_lycoris_module:
        if args.network_args is None:
            args.network_args = []
        if not any(arg.startswith("include_patterns=") for arg in args.network_args):
            args.lora_target_preset = "audio_ref_ic"
            logger.info("Using lora_target_preset=audio_ref_ic for --ic_lora_strategy audio_ref_ic")
    elif requested_ic_strategy == "av_ic" and not explicit_lora_preset and not uses_lycoris_module:
        if args.network_args is None:
            args.network_args = []
        if not any(arg.startswith("include_patterns=") for arg in args.network_args):
            args.lora_target_preset = "av_ic"
            logger.info("Using lora_target_preset=av_ic for --ic_lora_strategy av_ic")
    elif requested_ic_strategy == "video_ref_only_av" and not explicit_lora_preset and not uses_lycoris_module:
        if args.network_args is None:
            args.network_args = []
        if not any(arg.startswith("include_patterns=") for arg in args.network_args):
            args.lora_target_preset = "video_ref_only_av"
            logger.info("Using lora_target_preset=video_ref_only_av for --ic_lora_strategy video_ref_only_av")

    if (
        explicit_ic_strategy
        and requested_ic_strategy == "audio_ref_ic"
        and getattr(args, "ltx_mode", "video") not in {"av", "audio"}
    ):
        logger.warning("--ic_lora_strategy audio_ref_ic works in --ltx2_mode av or audio; current mode is %s", args.ltx_mode)
    if explicit_ic_strategy and requested_ic_strategy == "av_ic" and getattr(args, "ltx_mode", "video") != "av":
        logger.warning("--ic_lora_strategy %s requires --ltx2_mode av; current mode is %s", requested_ic_strategy, args.ltx_mode)
    if explicit_ic_strategy and requested_ic_strategy == "video_ref_only_av" and getattr(args, "ltx_mode", "video") != "av":
        logger.warning("--ic_lora_strategy video_ref_only_av requires --ltx2_mode av; current mode is %s", args.ltx_mode)

    if (
        explicit_lora_preset
        and getattr(args, "lora_target_preset", None) == "audio_ref_ic"
        and getattr(args, "ltx_mode", "video") == "audio"
    ):
        logger.warning(
            "--lora_target_preset audio_ref_ic in --ltx2_mode audio trains cross-modal layers that only "
            "affect the (dummy) video branch; consider --lora_target_preset audio instead."
        )
    if (
        explicit_lora_preset
        and getattr(args, "lora_target_preset", None) == "audio_v2a"
        and getattr(args, "ltx_mode", "video") == "audio"
    ):
        logger.warning(
            "--lora_target_preset audio_v2a in --ltx2_mode audio trains video_to_audio_attn against "
            "dummy/unsupervised video context (audio mode has no real video tokens). Prefer "
            "--lora_target_preset audio when --ltx2_mode is audio, or use --ltx2_mode av if you intend "
            "to train video_to_audio_attn."
        )

    lora_target_preset = getattr(args, "lora_target_preset", None)
    if uses_lycoris_module:
        if args.network_args is not None:
            filtered_args = [arg for arg in args.network_args if not arg.startswith("lora_target_preset=")]
            if len(filtered_args) != len(args.network_args):
                args.network_args = filtered_args
                logger.info("Removed lora_target_preset from --network_args for LyCORIS module compatibility")
        if explicit_lora_preset and lora_target_preset == "lycoris":
            logger.info("Using LyCORIS target preset: lycoris (Attention modules only)")
        elif explicit_lora_preset and lora_target_preset is not None:
            logger.warning(
                "--lora_target_preset %s does not affect LyCORIS networks. "
                "Use --lora_target_preset lycoris for the attention-only LTX-2 LyCORIS preset.",
                lora_target_preset,
            )
    elif lora_target_preset is not None:
        if args.network_args is None:
            args.network_args = []
        # Only add if not already specified in network_args
        if not any(arg.startswith("lora_target_preset=") for arg in args.network_args):
            args.network_args.append(f"lora_target_preset={lora_target_preset}")
            logger.info(f"Using LoRA target preset: {lora_target_preset}")

    # Inject connector_lora into network_args
    if getattr(args, "train_connectors", False):
        if uses_lycoris_module:
            logger.warning(
                "--train_connectors has no effect with LyCORIS networks. "
                "Connector LoRA is only supported with standard LoRA. Ignoring."
            )
            args.train_connectors = False
        else:
            if args.network_args is None:
                args.network_args = []
            if not any(arg.startswith("connector_lora=") for arg in args.network_args):
                args.network_args.append("connector_lora=True")
                logger.info("Connector LoRA enabled: connectors will be targeted by LoRA")

    # Inject audio_dim/audio_alpha into network_args (regular LoRA only, not LyCORIS)
    if not uses_lycoris_module:
        audio_dim = getattr(args, "audio_dim", None)
        audio_alpha = getattr(args, "audio_alpha", None)
        if audio_dim is not None or audio_alpha is not None:
            if args.network_args is None:
                args.network_args = []
            if audio_dim is not None and not any(arg.startswith("audio_dim=") for arg in args.network_args):
                args.network_args.append(f"audio_dim={audio_dim}")
            if audio_alpha is not None and not any(arg.startswith("audio_alpha=") for arg in args.network_args):
                args.network_args.append(f"audio_alpha={audio_alpha}")
            logger.info(f"Per-modality LoRA rank: audio_dim={audio_dim}, audio_alpha={audio_alpha}")

    process_lycoris_config(args, logger)
    apply_lycoris_preset_before_network_creation(args, logger)

    from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer

    trainer = LTX2NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
