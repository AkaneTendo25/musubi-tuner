from __future__ import annotations

import os
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class LTX2Env:
    # Apply audio_lengths mask to audio loss.
    use_audio_length_mask: bool = True

    # Upcast FP8 Linear weights to compute dtype during forward for stability.
    fp8_upcast: bool = False

    # Add stochastic rounding during FP8 upcast (extra stability, more noise).
    fp8_upcast_stochastic: bool = False

    # Keep FP8 weights in FP8 (do not override to bf16 compute dtype).
    keep_fp8_weights: bool = False

    # Seed for stochastic FP8 upcast (only used if fp8_upcast_stochastic=True).
    fp8_upcast_seed: int = 0

    # Allow FP8 weights to be offloaded to CPU by upcasting to bf16.
    fp8_offload_upcast: bool = False

    # Keep FP8-offloaded weights in bf16 on GPU after restore (avoid FP8 round-trip).
    fp8_offload_restore_bf16: bool = False

    # Allow FP8 weights to live on CPU without upcasting (Wan-style behavior).
    fp8_offload_allow_fp8_cpu: bool = False

    # Use the generic (Wan-style) block offloader instead of LTX-2 offloader.
    use_generic_offloader: bool = True

    # Force forward-only swapping during training (reduces VRAM, slower).
    swap_forward_only_train: bool = False

    # Allow FP8 CPU offload only when blockwise checkpointing is enabled.
    blockwise_fp8_offload_upcast: bool = False

    # Add stochastic noise before restoring FP8 weights (experimental).
    fp8_offload_restore_stochastic: bool = False

    # Keep attention weights on GPU during block swap (stability, higher VRAM).
    swap_keep_attn: bool = False

    # Keep cross-attn weights on GPU during swap (stability).
    swap_keep_cross_attn: bool = False

    # Keep audio-related weights on GPU during swap (stability).
    swap_keep_audio: bool = False

    # Use generic full-block swap (move all params/buffers of swap blocks to CPU).
    swap_full_block: bool = True

    # Force PyTorch attention for audio ctx + cross-attn in swapped blocks.
    attn_stability: bool = False

    # Force FP32 attention for audio ctx + cross-attn in swapped blocks.
    attn_stability_fp32: bool = False

    # Combined safe swap attention preset (keep attn + PyTorch + FP32 retry).
    safe_swap_attn: bool = False

    # FP8 swap safety preset (strict sync + safe swap attn).
    fp8_swap_safe: bool = False

    # Ensure FP8 weights are on device after swap-in.
    swap_fp8_sync: bool = False

    # Strict CUDA sync after FP8 swap-in.
    swap_fp8_sync_strict: bool = False

    # Force PyTorch attention for swapped blocks.
    swap_force_pytorch_attn: bool = False

    # Retry attention in FP32 if non-finite detected.
    attn_fp32_retry: bool = False

    # Force PyTorch attention for audio context.
    force_pytorch_audio_ctx_attn: bool = False

    # Force FP32 for audio context attention.
    audio_ctx_attn_fp32: bool = False

    # Force PyTorch for cross-attn.
    force_pytorch_cross_attn: bool = False

    # Force FP32 for cross-attn.
    cross_attn_fp32: bool = False

    # Apply cross-attn forcing only in swapped blocks.
    cross_attn_swap_only: bool = True

    # Apply audio-ctx forcing only in swapped blocks.
    audio_ctx_attn_swap_only: bool = True

    # Async prefetch stream for swap (faster but can be unstable).
    swap_async_prefetch: bool = False

    # Pinned memory for swap/offload (faster, can destabilize).
    swap_pinned: bool = False

    # Log sublayer-level non-finite diagnostics.
    nan_sublayer_diag: bool = False

    # Log block-level non-finite diagnostics.
    nan_block_diag: bool = False

    # Log NaN debug stats (latents/text/etc).
    nan_diag: bool = False

    # Trainer loss diagnostics (LTX2_LOSS_DIAG / LTX2_LOSS_DIAG_EVERY).
    loss_diag: bool = False

    # Frequency for loss diagnostics.
    loss_diag_every: int = 10

    # Swap/offloader diagnostics.
    swap_diag: bool = True    

    # Offloader debug prints (LTX2_OFFLOADER_DEBUG).
    offloader_debug: bool = False

    # Extra debug flag used by hv_train_network (LTX2_DEBUG).
    debug: bool = False

    # Log V2A (video-to-audio) internal stats.
    v2a_diag: bool = False

    # Log split-attn configuration details when enabled.
    split_attn_log: bool = True

    # Log per-block VRAM peak deltas (slow, debug only).
    vram_block_diag: bool = False

    # Align audio latents to video duration during training.
    align_audio_latents_train: bool = False

    # Align audio latents to video duration during cache_latents.
    align_audio_latents_cache: bool = False

    # Use video_prompt_embeds as AV fallback when audio missing.
    av_use_video_prompt_embeds: bool = False

    # Use 5D video loss mask (broadcast-friendly).
    video_loss_mask_5d: bool = False

    # Align transformer outputs to proj_out device.
    align_output_device: bool = False

    # Skip steps on non-finite tensors instead of erroring.
    skip_nonfinite_steps: bool = True

    # Auto-set blocks_to_checkpoint when blockwise+swap are both enabled.
    auto_blocks_to_checkpoint: bool = False

    # Require gemma_root (no gemma_safetensors-only usage).
    require_gemma_root: bool = True


def _set_env_bool(key: str, enabled: bool) -> None:
    os.environ[key] = "1" if enabled else "0"


def _set_env_int(key: str, value: int) -> None:
    os.environ[key] = str(int(value))


def _set_env_bool_default(key: str, enabled: bool) -> None:
    if key not in os.environ:
        _set_env_bool(key, enabled)


def _set_env_int_default(key: str, value: int) -> None:
    if key not in os.environ:
        _set_env_int(key, value)


def _env_bool(key: str, default: bool) -> bool:
    if key not in os.environ:
        return default
    value = os.environ.get(key, "")
    value_lower = value.strip().lower()
    if value_lower in {"1", "true", "yes", "on"}:
        return True
    if value_lower in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(key: str, default: int) -> int:
    if key not in os.environ:
        return default
    try:
        return int(os.environ.get(key, "").strip())
    except ValueError:
        return default
# Training presets - can be selected via --ltx2_preset CLI flag
_ACTIVE_PRESET: str = "auto"


_ENV_KEY_OVERRIDES = {
    "swap_keep_attn": "LTX2_SWAP_KEEP_ATTN",
    "swap_keep_cross_attn": "LTX2_SWAP_KEEP_CROSS_ATTN",
    "swap_keep_audio": "LTX2_SWAP_SKIP_AUDIO",
    "swap_full_block": "LTX2_SWAP_FULL_BLOCK",
    "swap_async_prefetch": "LTX2_SWAP_ASYNC_PREFETCH",
    "swap_pinned": "LTX2_SWAP_PINNED",
    "swap_fp8_sync": "LTX2_SWAP_FP8_SYNC",
    "swap_fp8_sync_strict": "LTX2_SWAP_FP8_SYNC_STRICT",
    "fp8_swap_safe": "LTX2_SWAP_STRICT_SYNC",
    "swap_force_pytorch_attn": "LTX2_SWAP_FORCE_PYTORCH_ATTN",
    "force_pytorch_audio_ctx_attn": "LTX2_FORCE_PYTORCH_AUDIO_CTX_ATTN",
    "audio_ctx_attn_fp32": "LTX2_AUDIO_CTX_ATTN_FP32",
    "force_pytorch_cross_attn": "LTX2_FORCE_PYTORCH_CROSS_ATTN",
    "cross_attn_fp32": "LTX2_CROSS_ATTN_FP32",
    "cross_attn_swap_only": "LTX2_CROSS_ATTN_SWAP_ONLY",
    "audio_ctx_attn_swap_only": "LTX2_AUDIO_CTX_ATTN_SWAP_ONLY",
    "attn_fp32_retry": "LTX2_ATTN_FP32_RETRY",
    "nan_sublayer_diag": "LTX2_NAN_SUBLAYER_DIAG",
    "nan_block_diag": "LTX2_NAN_BLOCK_DIAG",
    "nan_diag": "LTX2_NAN_DIAG",
    "loss_diag": "LTX2_LOSS_DIAG",
    "loss_diag_every": "LTX2_LOSS_DIAG_EVERY",
    "swap_diag": "LTX2_SWAP_DIAG",
    "offloader_debug": "LTX2_OFFLOADER_DEBUG",
    "debug": "LTX2_DEBUG",
    "v2a_diag": "LTX2_V2A_DIAG",
    "align_output_device": "LTX2_ALIGN_OUTPUT_DEVICE",
    "require_gemma_root": "LTX2_REQUIRE_GEMMA_ROOT",
    "split_attn_log": "LTX2_SPLIT_ATTN_LOG",
    "vram_block_diag": "LTX2_VRAM_BLOCK_DIAG",
}

_INT_FIELDS = {"fp8_upcast_seed", "loss_diag_every"}


def _env_key_for_field(field_name: str) -> str:
    return _ENV_KEY_OVERRIDES.get(field_name, f"LTX2_{field_name.upper()}")


def get_preset_env(preset: str) -> LTX2Env:
    """Get LTX2Env configuration for the specified preset.

    Presets:
    - "auto": Current defaults (balanced stability/performance)
    - "fast": Maximum performance (might be less stable)
    - "safe": Maximum stability (recommended for problematic training runs)
    """
    if preset == "auto":
        return LTX2Env()

    elif preset == "fast":
        return LTX2Env(
            use_audio_length_mask=False,  
            fp8_offload_upcast=False,  
            fp8_offload_restore_bf16=False,  
            fp8_swap_safe=False,  
            swap_fp8_sync_strict=False,  
            swap_async_prefetch=True,  
            swap_pinned=True,  
            align_audio_latents_train=False,  
            align_audio_latents_cache=False,  
            av_use_video_prompt_embeds=False,  
            video_loss_mask_5d=False,  
            align_output_device=False,  
            skip_nonfinite_steps=False,  
            auto_blocks_to_checkpoint=False,  
            require_gemma_root=True,  
        )

    elif preset == "safe":
        return LTX2Env(
            use_audio_length_mask=True,  
            fp8_upcast=True,  
            fp8_offload_upcast=True,  
            fp8_offload_restore_bf16=True,  
            swap_keep_attn=True,  
            swap_keep_cross_attn=True,  
            swap_keep_audio=True,  
            attn_stability=True,  
            safe_swap_attn=True,  
            fp8_swap_safe=True,  
            swap_fp8_sync=True,  
            attn_fp32_retry=True,  
            swap_async_prefetch=False,  
            swap_pinned=False,  
            skip_nonfinite_steps=True,
            require_gemma_root=True,  
        )

    else:
        raise ValueError(
            f"Unknown preset: {preset}. Valid presets: 'auto', 'fast', 'safe'"
        )


def set_ltx2_preset(preset: str) -> None:
    """Set the active LTX2 training preset.

    Args:
        preset: One of "auto", "fast", or "safe"
    """
    global _ACTIVE_PRESET
    _ACTIVE_PRESET = preset
    preset_env = get_preset_env(preset)
    for field in fields(LTX2Env):
        name = field.name
        value = getattr(preset_env, name)
        env_key = _env_key_for_field(name)
        if name in _INT_FIELDS:
            _set_env_int_default(env_key, int(value))
        else:
            _set_env_bool_default(env_key, bool(value))


def get_ltx2_env() -> LTX2Env:
    base = LTX2Env()
    kwargs = {}
    for field in fields(LTX2Env):
        name = field.name
        env_key = _env_key_for_field(name)
        default_value = getattr(base, name)
        if name in _INT_FIELDS:
            kwargs[name] = _env_int(env_key, int(default_value))
        else:
            kwargs[name] = _env_bool(env_key, bool(default_value))
    return LTX2Env(**kwargs)


def apply_ltx2_tweaks(args) -> None:
    t = get_ltx2_env()

    args.use_audio_length_mask = t.use_audio_length_mask

    args.fp8_upcast = t.fp8_upcast
    args.fp8_upcast_stochastic = t.fp8_upcast_stochastic
    args.fp8_upcast_seed = int(t.fp8_upcast_seed)
    args.keep_fp8_weights = t.keep_fp8_weights

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

    # Ensure env vars exist for downstream consumers without overriding explicit settings.
    for field in fields(LTX2Env):
        name = field.name
        value = getattr(t, name)
        env_key = _env_key_for_field(name)
        if name in _INT_FIELDS:
            _set_env_int_default(env_key, int(value))
        else:
            _set_env_bool_default(env_key, bool(value))

    # Log effective LTX-2 env configuration once.
    if not getattr(apply_ltx2_tweaks, "_logged_env", False):
        entries = []
        for field in fields(LTX2Env):
            name = field.name
            env_key = _env_key_for_field(name)
            entries.append(f"{env_key}={os.environ.get(env_key, '')}")
        logger = None
        try:
            import logging
            logger = logging.getLogger(__name__)
        except Exception:
            logger = None
        if logger is not None:
            logger.info("LTX-2 effective env: %s", ", ".join(entries))
        else:
            print("LTX-2 effective env:", ", ".join(entries))
        apply_ltx2_tweaks._logged_env = True
