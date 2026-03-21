from __future__ import annotations

import os
from dataclasses import dataclass

from musubi_tuner.ltx_2.model.ltx2_custom_offloading_utils import (
    set_fp8_offload_keep_fp8,
    set_fp8_offload_restore_bf16,
    set_fp8_offload_restore_stochastic,
    set_fp8_offload_upcast,
)


@dataclass(frozen=True)
class LTX2Env:
    # Apply audio_lengths mask to audio loss.
    # recommended=True
    use_audio_length_mask: bool = True

    # Upcast FP8 Linear weights to compute dtype during forward for stability.
    # recommended=True
    fp8_upcast: bool = False

    # Add stochastic rounding during FP8 upcast (extra stability, more noise).
    # recommended=False
    fp8_upcast_stochastic: bool = False

    # Seed for stochastic FP8 upcast (only used if fp8_upcast_stochastic=True).
    # recommended=0
    fp8_upcast_seed: int = 0

    # Allow FP8 weights to be offloaded to CPU by upcasting to bf16.
    # recommended=True
    fp8_offload_upcast: bool = False

    # Keep FP8-offloaded weights in bf16 on GPU after restore (avoid FP8 round-trip).
    # recommended=True
    fp8_offload_restore_bf16: bool = False

    # Keep FP8 weights in FP8 on CPU (avoid bf16 upcast and GPU re-cast cost).
    # recommended=False
    fp8_offload_keep_fp8: bool = False

    # Allow FP8 CPU offload only when blockwise checkpointing is enabled.
    # recommended=False
    blockwise_fp8_offload_upcast: bool = True

    # Add stochastic noise before restoring FP8 weights (experimental).
    # recommended=False
    fp8_offload_restore_stochastic: bool = False

    # Keep attention weights on GPU during block swap (stability, higher VRAM).
    # recommended=True
    swap_keep_attn: bool = False

    # Keep cross-attn weights on GPU during swap (stability).
    # recommended=True
    swap_keep_cross_attn: bool = False

    # Keep audio-related weights on GPU during swap (stability).
    # recommended=True
    swap_keep_audio: bool = False

    # Force PyTorch attention for audio ctx + cross-attn in swapped blocks.
    # recommended=True
    attn_stability: bool = False

    # Force FP32 attention for audio ctx + cross-attn in swapped blocks.
    # recommended=False
    attn_stability_fp32: bool = False

    # Keep prompt-AdaLN modulation in float32 for consistency with the other
    # AdaLN stability fixes.
    # recommended=True
    prompt_adaln_fp32: bool = True

    # Combined safe swap attention preset (keep attn + PyTorch + FP32 retry).
    # recommended=True
    safe_swap_attn: bool = False

    # FP8 swap safety preset (strict sync + safe swap attn).
    # recommended=True
    fp8_swap_safe: bool = True

    # Ensure FP8 weights are on device after swap-in.
    # recommended=True
    swap_fp8_sync: bool = False

    # Strict CUDA sync after FP8 swap-in.
    # recommended=False
    swap_fp8_sync_strict: bool = True

    # Force PyTorch attention for swapped blocks.
    # recommended=False
    swap_force_pytorch_attn: bool = False

    # Retry attention in FP32 if non-finite detected.
    # recommended=True
    attn_fp32_retry: bool = False

    # Force PyTorch attention for audio context.
    # recommended=False
    force_pytorch_audio_ctx_attn: bool = False

    # Force FP32 for audio context attention.
    # recommended=False
    audio_ctx_attn_fp32: bool = False

    # Force PyTorch for cross-attn.
    # recommended=False
    force_pytorch_cross_attn: bool = False

    # Force FP32 for cross-attn.
    # recommended=False
    cross_attn_fp32: bool = False

    # Apply cross-attn forcing only in swapped blocks.
    # recommended=True
    cross_attn_swap_only: bool = True

    # Apply audio-ctx forcing only in swapped blocks.
    # recommended=True
    audio_ctx_attn_swap_only: bool = True

    # Async prefetch stream for swap (faster but can be unstable).
    # recommended=False
    swap_async_prefetch: bool = False

    # Pinned memory for swap/offload (faster, can destabilize).
    # recommended=False
    swap_pinned: bool = False

    # Swap RMSNorm/LayerNorm weights to CPU during block swap (VRAM saver).
    # recommended=False
    swap_norms: bool = False

    # Log sublayer-level non-finite diagnostics.
    # recommended=False
    nan_sublayer_diag: bool = False

    # Log block-level non-finite diagnostics.
    # recommended=False
    nan_block_diag: bool = False

    # Log NaN debug stats (latents/text/etc).
    # recommended=False
    nan_diag: bool = False

    # Trainer loss diagnostics (LTX2_LOSS_DIAG / LTX2_LOSS_DIAG_EVERY).
    # recommended=False
    loss_diag: bool = False

    # Frequency for loss diagnostics.
    # recommended=10
    loss_diag_every: int = 10

    # Swap/offloader diagnostics.
    # recommended=False
    swap_diag: bool = False

    # Offloader debug prints (LTX2_OFFLOADER_DEBUG).
    # recommended=False
    offloader_debug: bool = False

    # Extra debug flag used by hv_train_network (LTX2_DEBUG).
    # recommended=False
    debug: bool = False

    # Log V2A (video-to-audio) internal stats.
    # recommended=False
    v2a_diag: bool = False

    # Align audio latents to video duration during training.
    # recommended=False
    align_audio_latents_train: bool = False

    # Align audio latents to video duration during cache_latents.
    # recommended=False
    align_audio_latents_cache: bool = False

    # Use video_prompt_embeds as AV fallback when audio missing.
    # recommended=False
    av_use_video_prompt_embeds: bool = False

    # Use 5D video loss mask (broadcast-friendly).
    # recommended=False
    video_loss_mask_5d: bool = False

    # Align transformer outputs to proj_out device.
    # recommended=False
    align_output_device: bool = False

    # Skip steps on non-finite tensors instead of erroring.
    # recommended=False
    skip_nonfinite_steps: bool = True

    # Auto-set blocks_to_checkpoint when blockwise+swap are both enabled.
    # recommended=False
    auto_blocks_to_checkpoint: bool = False

    # Require gemma_root (no gemma_safetensors-only usage).
    # recommended=True
    require_gemma_root: bool = True

    # Clamp FFN outputs to prevent bf16 overflow. Value is the clamp bound.
    # 60000 leaves headroom below bf16 max (~65504). 0 = disabled.
    # recommended=60000
    ffn_clamp: float = 60000.0

    # Skip no-op attention masks to enable Flash Attention on cross-attn.
    # recommended=False
    skip_noop_attn_mask: bool = False

    # Max FPS difference (after ceiling source FPS) below which resampling is skipped.
    # e.g. threshold=1: 23.976->ceil=24 vs 25, diff=1, skip. 30 vs 25, diff=5, resample.
    # recommended=1
    fps_resampling_threshold: int = 1

    # Number of blocks to prefetch ahead during block swap (1 = current behavior).
    # Higher values overlap more transfers with compute but use more VRAM.
    # recommended=1
    swap_prefetch_window: int = 1

    # Use pre-allocated pinned slab pool instead of lazy per-parameter _pinned_buffer_cache.
    # recommended=False
    swap_slab_pool: bool = False


DEFAULT_ENV = LTX2Env()


def _set_env_bool(key: str, enabled: bool) -> None:
    os.environ[key] = "1" if enabled else "0"


def get_ltx2_env() -> LTX2Env:
    return DEFAULT_ENV


def apply_ltx2_tweaks(args) -> None:
    t = DEFAULT_ENV

    args.use_audio_length_mask = t.use_audio_length_mask

    args.fp8_upcast = t.fp8_upcast
    args.fp8_upcast_stochastic = t.fp8_upcast_stochastic
    args.fp8_upcast_seed = int(t.fp8_upcast_seed)

    args.fp8_offload_upcast = t.fp8_offload_upcast
    args.fp8_offload_restore_bf16 = t.fp8_offload_restore_bf16
    args.fp8_offload_restore_stochastic = t.fp8_offload_restore_stochastic
    args.fp8_offload_keep_fp8 = t.fp8_offload_keep_fp8

    args.swap_keep_attn = t.swap_keep_attn
    args.swap_keep_cross_attn = t.swap_keep_cross_attn
    args.swap_keep_audio = t.swap_keep_audio
    args.attn_stability = t.attn_stability
    args.attn_stability_fp32 = t.attn_stability_fp32
    args.prompt_adaln_fp32 = t.prompt_adaln_fp32
    args.safe_swap_attn = t.safe_swap_attn
    args.fp8_swap_safe = t.fp8_swap_safe
    args.swap_fp8_sync = t.swap_fp8_sync
    args.swap_fp8_sync_strict = t.swap_fp8_sync_strict
    args.swap_force_pytorch_attn = t.swap_force_pytorch_attn
    args.attn_fp32_retry = t.attn_fp32_retry
    args.force_pytorch_audio_ctx_attn = t.force_pytorch_audio_ctx_attn
    args.audio_ctx_attn_fp32 = t.audio_ctx_attn_fp32
    args.force_pytorch_cross_attn = t.force_pytorch_cross_attn
    args.cross_attn_fp32 = t.cross_attn_fp32
    args.cross_attn_swap_only = t.cross_attn_swap_only
    args.audio_ctx_attn_swap_only = t.audio_ctx_attn_swap_only

    args.swap_async_prefetch = t.swap_async_prefetch
    args.swap_no_async_prefetch = not t.swap_async_prefetch
    cli_swap_pinned = bool(getattr(args, "use_pinned_memory_for_block_swap", False))
    effective_swap_pinned = bool(t.swap_pinned or cli_swap_pinned)
    args.swap_pinned = effective_swap_pinned
    args.swap_no_pinned = not effective_swap_pinned
    args.swap_norms = t.swap_norms

    args.nan_sublayer_diag = t.nan_sublayer_diag
    args.nan_block_diag = t.nan_block_diag

    args.align_audio_latents_train = t.align_audio_latents_train
    args.align_audio_latents_cache = t.align_audio_latents_cache
    args.av_use_video_prompt_embeds = t.av_use_video_prompt_embeds
    args.video_loss_mask_5d = t.video_loss_mask_5d
    args.align_output_device = t.align_output_device
    args.skip_nonfinite_steps = t.skip_nonfinite_steps
    args.auto_blocks_to_checkpoint = t.auto_blocks_to_checkpoint
    args.require_gemma_root = t.require_gemma_root
    args.skip_noop_attn_mask = t.skip_noop_attn_mask

    # Apply global FP8 offload behavior.
    fp8_offload_enabled = t.fp8_offload_upcast
    if (
        getattr(args, "blockwise_checkpointing", False)
        and t.blockwise_fp8_offload_upcast
    ):
        fp8_offload_enabled = True
    set_fp8_offload_upcast(fp8_offload_enabled)
    set_fp8_offload_restore_bf16(t.fp8_offload_restore_bf16)
    set_fp8_offload_restore_stochastic(t.fp8_offload_restore_stochastic)
    set_fp8_offload_keep_fp8(t.fp8_offload_keep_fp8)

    # Apply env vars consumed in transformer/offloading code.
    _set_env_bool("LTX2_SWAP_KEEP_ATTN", t.swap_keep_attn)
    _set_env_bool("LTX2_SWAP_KEEP_CROSS_ATTN", t.swap_keep_cross_attn)
    _set_env_bool("LTX2_SWAP_SKIP_AUDIO", t.swap_keep_audio)
    _set_env_bool("LTX2_SWAP_ASYNC_PREFETCH", t.swap_async_prefetch)
    _set_env_bool("LTX2_SWAP_PINNED", effective_swap_pinned)
    _set_env_bool("LTX2_SWAP_FP8_SYNC", t.swap_fp8_sync)
    _set_env_bool("LTX2_SWAP_FP8_SYNC_STRICT", t.swap_fp8_sync_strict)
    _set_env_bool("LTX2_FP8_OFFLOAD_KEEP_FP8", t.fp8_offload_keep_fp8)
    _set_env_bool("LTX2_SWAP_STRICT_SYNC", t.fp8_swap_safe)
    _set_env_bool("LTX2_SWAP_FORCE_PYTORCH_ATTN", t.swap_force_pytorch_attn)
    _set_env_bool(
        "LTX2_FORCE_PYTORCH_AUDIO_CTX_ATTN",
        t.force_pytorch_audio_ctx_attn or t.attn_stability or t.safe_swap_attn,
    )
    _set_env_bool(
        "LTX2_AUDIO_CTX_ATTN_FP32", t.audio_ctx_attn_fp32 or t.attn_stability_fp32
    )
    _set_env_bool(
        "LTX2_FORCE_PYTORCH_CROSS_ATTN",
        t.force_pytorch_cross_attn or t.attn_stability or t.safe_swap_attn,
    )
    _set_env_bool("LTX2_CROSS_ATTN_FP32", t.cross_attn_fp32 or t.attn_stability_fp32)
    _set_env_bool("LTX2_PROMPT_ADALN_FP32", t.prompt_adaln_fp32)
    _set_env_bool("LTX2_CROSS_ATTN_SWAP_ONLY", t.cross_attn_swap_only)
    _set_env_bool("LTX2_AUDIO_CTX_ATTN_SWAP_ONLY", t.audio_ctx_attn_swap_only)
    _set_env_bool(
        "LTX2_ATTN_FP32_RETRY", t.attn_fp32_retry or t.safe_swap_attn or t.fp8_swap_safe
    )
    _set_env_bool("LTX2_NAN_SUBLAYER_DIAG", t.nan_sublayer_diag)
    _set_env_bool("LTX2_NAN_BLOCK_DIAG", t.nan_block_diag)
    _set_env_bool("LTX2_NAN_DIAG", t.nan_diag)
    _set_env_bool("LTX2_LOSS_DIAG", t.loss_diag)
    os.environ["LTX2_LOSS_DIAG_EVERY"] = str(int(t.loss_diag_every))
    _set_env_bool("LTX2_SWAP_DIAG", t.swap_diag)
    _set_env_bool("LTX2_OFFLOADER_DEBUG", t.offloader_debug)
    _set_env_bool("LTX2_DEBUG", t.debug)
    _set_env_bool("LTX2_V2A_DIAG", t.v2a_diag)
    _set_env_bool("LTX2_ALIGN_OUTPUT_DEVICE", t.align_output_device)
    _set_env_bool("LTX2_REQUIRE_GEMMA_ROOT", t.require_gemma_root)
    os.environ["LTX2_FFN_CLAMP"] = str(t.ffn_clamp)
    _set_env_bool("LTX2_SKIP_NOOP_ATTN_MASK", t.skip_noop_attn_mask)
    os.environ["LTX2_SWAP_PREFETCH_WINDOW"] = str(int(t.swap_prefetch_window))
    _set_env_bool("LTX2_SWAP_SLAB_POOL", t.swap_slab_pool)
