"""LTX-2 LoRA Training Implementation."""

import argparse
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from accelerate import Accelerator
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from musubi_tuner.hv_train_network import (
    NetworkTrainer,
)
from musubi_tuner.audio_supervision import (
    AudioSupervisionState,
    format_audio_supervision_alert,
    normalize_audio_supervision_mode,
    reset_audio_supervision_state,
    update_and_check_audio_supervision,
)
from musubi_tuner.utils import model_utils
from musubi_tuner.ltx_2.model.transformer.fp8_device_utils import ensure_fp8_modules_on_device
from musubi_tuner.modules.nf4_optimization_utils import (
    is_nf4_module,
    DEFAULT_NF4_BLOCK_SIZE,
)
from musubi_tuner.ltx_2.env import apply_ltx2_tweaks
from musubi_tuner.ltx2_text_conditioning import (
    select_audio_text_embeds_for_audio_mode,
    select_video_text_embeds_for_video_mode,
    select_video_text_embeds_for_av_no_audio,
)
from musubi_tuner.self_flow import build_self_flow_video_context, prepare_self_flow_audio_view
from musubi_tuner.ltx2_lycoris_runtime import (
    validate_lycoris_quantized_base_compatibility,
    validate_lycoris_runtime,
)

# LTX-2 latent normalization defaults.
# These are identity stats (mean=0, std=1). We keep them as a safe fallback and
# override them from the loaded VAE if it exposes per-channel statistics.
LTX2_LATENTS_MEAN = [0.0]
LTX2_LATENTS_STD = [1.0]

DEFAULT_SAMPLE_PROMPTS_CACHE = "ltx2_sample_prompts_cache.pt"
DEFAULT_SAMPLE_LATENTS_CACHE = "ltx2_sample_latents_cache.pt"
IC_LORA_STRATEGIES = (
    "auto",
    "none",
    "v2v",
    "audio_ref_only_ic",
    "av_ic",
    "video_ref_only_av",
)
AV_CROSS_ATTENTION_MODES = ("both", "a2v_only", "v2a_only", "none")


def infer_ic_lora_strategy_from_preset(lora_target_preset: Optional[str]) -> str:
    """Infer IC-LoRA strategy from LoRA target preset for backward-compatible auto mode."""
    preset = str(lora_target_preset or "").lower()
    if preset == "v2v":
        return "v2v"
    if preset == "audio_ref_only_ic":
        return "audio_ref_only_ic"
    if preset == "av_ic":
        return "av_ic"
    if preset == "video_ref_only_av":
        return "video_ref_only_av"
    return "none"


def _normalize_av_cross_attention_mode(value: Optional[str]) -> str:
    mode = str(value or "both").lower()
    if mode not in AV_CROSS_ATTENTION_MODES:
        raise ValueError(
            f"av_cross_attention_mode must be one of {list(AV_CROSS_ATTENTION_MODES)}. Got: {mode}"
        )
    return mode


def _extract_tensor_payload(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, dict):
        value = value.get("latents")
    return value if isinstance(value, torch.Tensor) else None


def _expand_reference_tensor(ref_tensor: torch.Tensor, expected_ndim: int) -> List[torch.Tensor]:
    if ref_tensor.dim() == expected_ndim:
        return [ref_tensor]
    if ref_tensor.dim() == expected_ndim + 1:
        return [ref_tensor[:, i, ...] for i in range(int(ref_tensor.shape[1]))]
    raise ValueError(
        f"Expected reference tensor with ndim {expected_ndim} or {expected_ndim + 1}, got {ref_tensor.dim()}"
    )


def _collect_reference_tensors(batch: Dict[str, Any], base_key: str, expected_ndim: int) -> List[torch.Tensor]:
    refs: List[torch.Tensor] = []
    base_tensor = _extract_tensor_payload(batch.get(base_key))
    if base_tensor is not None:
        refs.extend(_expand_reference_tensor(base_tensor, expected_ndim))

    suffix_pattern = re.compile(rf"^{re.escape(base_key)}(?:[_-].+)$")
    for key in sorted(batch.keys()):
        if key == base_key or not suffix_pattern.match(str(key)):
            continue
        extra_tensor = _extract_tensor_payload(batch.get(key))
        if extra_tensor is not None:
            refs.extend(_expand_reference_tensor(extra_tensor, expected_ndim))
    return refs


def _merge_reference_tensors(refs: List[torch.Tensor], concat_dim: int) -> Optional[torch.Tensor]:
    if not refs:
        return None
    if len(refs) == 1:
        return refs[0]
    return torch.cat(refs, dim=concat_dim)


def _coerce_video_loss_mask(
    mask: Any,
    *,
    latents: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    as_tokens: bool,
) -> Optional[torch.Tensor]:
    payload = _extract_tensor_payload(mask)
    mask_tensor = payload if payload is not None else (mask if isinstance(mask, torch.Tensor) else None)
    if mask_tensor is None:
        return None

    bsz, _channels, frames, height, width = latents.shape
    mask = mask_tensor.to(device=device, dtype=torch.float32)
    if mask.dim() == 2:
        if mask.shape[0] != bsz:
            raise ValueError(f"video_loss_mask batch mismatch: mask={tuple(mask.shape)} latents={tuple(latents.shape)}")
        if as_tokens:
            if mask.shape[1] != frames:
                mask = F.interpolate(mask.unsqueeze(1), size=frames, mode="linear", align_corners=False).squeeze(1)
            return mask.view(bsz, frames, 1, 1).expand(bsz, frames, height, width).reshape(bsz, -1).to(dtype=dtype)
        if mask.shape[1] != frames:
            mask = F.interpolate(mask.unsqueeze(1), size=frames, mode="linear", align_corners=False).squeeze(1)
        return mask.view(bsz, 1, frames, 1, 1).to(dtype=dtype)

    if mask.dim() == 4:
        mask = mask.unsqueeze(1)
    if mask.dim() != 5:
        raise ValueError(f"video_loss_mask must be [B,F], [B,F,H,W], or [B,1,F,H,W], got {tuple(mask.shape)}")
    if mask.shape[0] != bsz:
        raise ValueError(f"video_loss_mask batch mismatch: mask={tuple(mask.shape)} latents={tuple(latents.shape)}")
    if mask.shape[1] != 1:
        mask = mask.mean(dim=1, keepdim=True)
    if tuple(mask.shape[2:]) != (frames, height, width):
        mask = F.interpolate(mask, size=(frames, height, width), mode="trilinear", align_corners=False)
    mask = mask.clamp(0.0, 1.0)
    if as_tokens:
        return mask.flatten(2).squeeze(1).to(dtype=dtype)
    return mask.to(dtype=dtype)


def _coerce_audio_loss_mask(
    mask: Any,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    payload = _extract_tensor_payload(mask)
    mask_tensor = payload if payload is not None else (mask if isinstance(mask, torch.Tensor) else None)
    if mask_tensor is None:
        return None
    mask = mask_tensor.to(device=device, dtype=torch.float32)
    if mask.dim() == 1:
        mask = mask.view(1, -1)
    if mask.dim() != 2:
        raise ValueError(f"audio_loss_mask must be [B,T] or [T], got {tuple(mask.shape)}")
    if mask.shape[0] == 1 and batch_size != 1:
        mask = mask.expand(batch_size, -1)
    if mask.shape[0] != batch_size:
        raise ValueError(f"audio_loss_mask batch mismatch: mask={tuple(mask.shape)} batch={batch_size}")
    if mask.shape[1] < seq_len:
        pad = torch.ones((batch_size, seq_len - int(mask.shape[1])), device=device, dtype=mask.dtype)
        mask = torch.cat([mask, pad], dim=1)
    elif mask.shape[1] > seq_len:
        mask = mask[:, :seq_len]
    return mask.clamp(0.0, 1.0).to(dtype=dtype)


def _combine_loss_masks(existing: Optional[torch.Tensor], extra: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if extra is None:
        return existing
    if existing is None:
        return extra
    extra = extra.to(device=existing.device)
    if existing.dim() == 2 and extra.dim() == 5:
        existing = existing.view(existing.shape[0], 1, existing.shape[1], 1, 1)
    elif existing.dim() == 5 and extra.dim() == 2:
        extra = extra.view(extra.shape[0], 1, extra.shape[1], 1, 1)
    elif existing.dim() == 2 and extra.dim() == 3 and extra.shape[-1] == 1:
        extra = extra.squeeze(-1)
    return existing.to(dtype=extra.dtype) * extra


def _compose_target_audio_loss_mask(
    target_audio_loss_mask: Optional[torch.Tensor],
    cached_audio_loss_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    target_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if target_audio_loss_mask is None or int(target_audio_loss_mask.shape[1]) != target_seq_len:
        target_audio_loss_mask = torch.ones((batch_size, target_seq_len), device=device, dtype=torch.bool)
    return _combine_loss_masks(target_audio_loss_mask, cached_audio_loss_mask)


def _compose_audio_ref_only_ic_loss_mask(
    target_audio_loss_mask: Optional[torch.Tensor],
    cached_audio_loss_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    target_seq_len: int,
    ref_seq_len: int,
    device: torch.device,
    audio_lengths: Any = None,
) -> torch.Tensor:
    if target_audio_loss_mask is None:
        target_audio_loss_mask = torch.ones((batch_size, target_seq_len), device=device, dtype=torch.bool)
    if isinstance(audio_lengths, dict):
        audio_lengths = audio_lengths.get("lengths")
    if isinstance(audio_lengths, torch.Tensor):
        if audio_lengths.dim() == 0:
            audio_lengths = audio_lengths.view(1)
        if audio_lengths.numel() == 1 and batch_size != 1:
            audio_lengths = audio_lengths.expand(batch_size)
        if audio_lengths.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: audio_latents batch={batch_size} vs audio_lengths batch={audio_lengths.shape[0]}"
            )
        audio_lengths = audio_lengths.to(device=device, dtype=torch.int64).clamp(min=0, max=target_seq_len)
        t = torch.arange(target_seq_len, device=device).view(1, -1)
        target_audio_loss_mask = t < audio_lengths.view(-1, 1)
    target_audio_loss_mask = _combine_loss_masks(target_audio_loss_mask, cached_audio_loss_mask)
    ref_audio_loss_mask = torch.zeros(
        (batch_size, ref_seq_len),
        device=device,
        dtype=target_audio_loss_mask.dtype,
    )
    return torch.cat([ref_audio_loss_mask, target_audio_loss_mask], dim=1)


def _resolve_batch_captions(batch: Dict[str, Any]) -> Optional[list[str]]:
    captions = batch.get("captions")
    if captions is None:
        return None
    if isinstance(captions, str):
        return [captions]
    if isinstance(captions, (list, tuple)):
        if not captions:
            return []
        for caption in captions:
            if not isinstance(caption, str):
                return None
        return list(captions)
    return None

# Model loading functions moved to ltx2_model_loading.py
# Parser functions moved to ltx2_args.py
# Sampling/inference methods moved to ltx2_sampling.py (LTX2SamplingMixin)


from musubi_tuner.ltx2_sampling import LTX2SamplingMixin
from musubi_tuner.ltx2_model_loading import (  # noqa: F401 — re-exported for external callers
    detect_ltx2_dtype,
    detect_ltx2_config,
    infer_ltx_version_from_checkpoint_config,
    load_ltx2_model,
    KEEP_FP8_HIGH_PRECISION_TOKENS,
)


def ltx2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Lazy re-export to avoid a circular import with musubi_tuner.ltx2_args."""
    from musubi_tuner.ltx2_args import ltx2_setup_parser as _ltx2_setup_parser

    return _ltx2_setup_parser(parser)


def main() -> None:
    """Lazy re-export to avoid a circular import with musubi_tuner.ltx2_args."""
    from musubi_tuner.ltx2_args import main as _main

    _main()


class LTX2NetworkTrainer(LTX2SamplingMixin, NetworkTrainer):
    """Trainer for LTX-2 models with LoRA support"""

    def __init__(self) -> None:
        super().__init__()
        self._text_encoder = None
        self._dit_attn_mode: Optional[str] = None
        self._latent_norm_cache: Dict = {}
        self._warned_missing_audio = False
        self._warned_ignored_ref_latents = False
        self._warned_text_encoder_fallback = False
        self._audio_supervision_state = AudioSupervisionState()

        # Initialize latent normalization
        mean = torch.tensor(LTX2_LATENTS_MEAN, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(LTX2_LATENTS_STD, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = std.clamp_min(1e-6)
        self._latent_norm_base: Tuple[torch.Tensor, torch.Tensor] = (mean, std.reciprocal())

        self._flow_target: str = "noise"  # LTX-2 predicts noise
        self._num_timesteps: int = 1000
        self._audio_video: bool = False
        self._i2v_training: bool = False
        self._ic_lora_strategy: str = "none"
        self._ltx_mode: str = "video"
        self._ltx_version: str = "2.3"
        self._ltx2_audio_only_model: bool = False
        self._logged_audio_only_timestep_shift: bool = False
        self._audio_only_sequence_resolution: int = 64
        self._ltx2_checkpoint_config: Optional[Dict[str, Any]] = None
        self.default_guidance_scale = 3.0
        self._audio_preview_config: Optional[Dict[str, int | float]] = None
        self._timestep_logging_context: Optional[Dict[str, torch.Tensor]] = None

        # Preservation / regularization (off by default)
        self._preservation_active: bool = False
        self._preservation_helper = None
        self._last_dit_inputs: Optional[Dict[str, Any]] = None

        # TARP / DCR (off by default)
        self._tarp_enabled: bool = False
        self._dcr_enabled: bool = False
        self._tarp_window_multiplier: int = 3
        self._dcr_reference_detach: bool = True

        # CREPA (off by default)
        self._crepa = None
        # Self-Flow (off by default)
        self._self_flow = None
        self._self_flow_active: bool = False
        self._self_flow_step_context: Optional[Dict[str, Any]] = None
        # Audio metrics (off by default)
        self._audio_metrics = None
        # HFATO (off by default)
        self._hfato_config = None  # Optional[HFATOConfig]

    @staticmethod
    def _apply_caption_dropout(
        text_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        caption_dropout_rate: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        text_embeds = text_embeds.clone()
        if text_mask is not None:
            text_mask = text_mask.clone()

        for i in range(text_embeds.shape[0]):
            if random.random() < caption_dropout_rate:
                text_embeds[i] = 0
                if text_mask is not None:
                    text_mask[i] = False
                    if text_mask.shape[-1] > 0:
                        text_mask[i, 0] = True

        return text_embeds, text_mask

    def _build_audio_ref_transformer_overrides(
        self,
        *,
        args: argparse.Namespace,
        transformer,
        video_latents: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        audio_model_latents: torch.Tensor,
        ref_audio_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Build optional transformer overrides for audio_ref_only_ic training/sampling."""
        overrides: Dict[str, torch.Tensor] = {}

        if ref_audio_seq_len <= 0:
            return overrides

        total_audio_seq_len = int(audio_model_latents.shape[2])
        if total_audio_seq_len <= 0:
            return overrides
        ref_tokens = max(0, min(int(ref_audio_seq_len), total_audio_seq_len))
        bsz = int(audio_model_latents.shape[0])

        mask_dtype = dtype if dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16) else torch.float32
        neg_inf = torch.finfo(mask_dtype).min

        # Always generate separate ref/target position arrays for audio_ref_only_ic.
        # ID-LoRA reference generates positions independently: both ref and target start
        # at time=0 by default; with negative positions enabled, ref is shifted to t<0.
        # Without this override the wrapper would compute positions for the combined
        # sequence, placing target after ref (non-overlapping) — a layout mismatch.
        # Position override is always applied for audio_ref_only_ic (not gated by a flag).
        use_negative_positions = bool(getattr(args, "audio_ref_use_negative_positions", False))
        if ref_tokens > 0:
            from musubi_tuner.ltx_2.types import AudioLatentShape

            audio_patchifier = getattr(transformer, "_audio_patchifier", None)
            if audio_patchifier is None and hasattr(transformer, "module"):
                audio_patchifier = getattr(transformer.module, "_audio_patchifier", None)
            if audio_patchifier is None:
                if use_negative_positions:
                    logger.warning("audio_ref position override requested but audio patchifier is unavailable; skipping")
            else:
                channels = int(audio_model_latents.shape[1])
                mel_bins = int(audio_model_latents.shape[3])
                tgt_tokens = total_audio_seq_len - ref_tokens

                # Generate SEPARATE position arrays for ref and target (matches ID-LoRA reference).
                # Target positions always start at 0 (aligned with video time).
                ref_shape = AudioLatentShape(batch=bsz, channels=channels, frames=ref_tokens, mel_bins=mel_bins)
                ref_positions = audio_patchifier.get_patch_grid_bounds(ref_shape, device=device).to(dtype=mask_dtype)

                if use_negative_positions:
                    # Shift ref into negative time with a one-step gap for clean positional separation.
                    _hop = getattr(audio_patchifier, "hop_length", 160)
                    _ds = getattr(audio_patchifier, "audio_latent_downsample_factor", 4)
                    _sr = getattr(audio_patchifier, "sample_rate", 16000)
                    time_per_latent = float(_hop) * float(_ds) / float(_sr)
                    ref_duration = ref_positions[:, :, -1:, 1:2]
                    ref_positions = ref_positions - ref_duration - time_per_latent

                # else: ref positions start at 0 (same as target) — matches ID-LoRA default

                tgt_shape = AudioLatentShape(batch=bsz, channels=channels, frames=max(tgt_tokens, 1), mel_bins=mel_bins)
                tgt_positions = audio_patchifier.get_patch_grid_bounds(tgt_shape, device=device).to(dtype=mask_dtype)
                if tgt_tokens <= 0:
                    tgt_positions = tgt_positions[:, :, :0, :]  # empty slice

                audio_positions = torch.cat([ref_positions, tgt_positions], dim=2)
                overrides["audio_positions_override"] = audio_positions.to(
                    device=device,
                    dtype=audio_model_latents.dtype,
                )

        if bool(getattr(args, "audio_ref_mask_cross_attention_to_reference", False)):
            video_seq_len = int(video_latents.shape[2]) * int(video_latents.shape[3]) * int(video_latents.shape[4])
            if video_seq_len > 0:
                a2v_mask = torch.zeros((bsz, video_seq_len, total_audio_seq_len), device=device, dtype=mask_dtype)
                a2v_mask[:, :, :ref_tokens] = neg_inf
                overrides["a2v_cross_attention_mask"] = a2v_mask

        if bool(getattr(args, "audio_ref_mask_reference_from_text_attention", False)):
            text_seq_len = int(text_embeds.shape[1])
            audio_context_mask = torch.zeros(
                (bsz, total_audio_seq_len, text_seq_len),
                device=device,
                dtype=mask_dtype,
            )

            if text_mask is not None:
                if not isinstance(text_mask, torch.Tensor):
                    raise TypeError(f"Expected text_mask to be a torch.Tensor, got: {type(text_mask)}")
                tm = text_mask
                if tm.dim() == 1:
                    tm = tm.unsqueeze(0)
                if tm.dim() != 2:
                    raise ValueError(f"Expected text_mask to be 2D [B, seq_len], got shape: {tuple(tm.shape)}")
                if tm.shape[0] == 1 and bsz != 1:
                    tm = tm.expand(bsz, tm.shape[1])
                if tm.shape[0] != bsz:
                    raise ValueError(f"Batch mismatch for text_mask: got {tm.shape[0]}, expected {bsz}")
                if tm.shape[1] != text_seq_len:
                    if tm.shape[1] > text_seq_len:
                        tm = tm[:, -text_seq_len:]
                    else:
                        tm = F.pad(tm, (text_seq_len - tm.shape[1], 0), value=1)
                tm = tm.to(device=device)
                valid_text = tm.to(torch.bool) if tm.dtype == torch.bool else (tm > 0)
                key_bias = torch.zeros((bsz, text_seq_len), device=device, dtype=mask_dtype)
                key_bias[~valid_text] = neg_inf
                audio_context_mask = key_bias.unsqueeze(1).expand(-1, total_audio_seq_len, -1).clone()

            audio_context_mask[:, :ref_tokens, :] = neg_inf
            overrides["audio_context_mask"] = audio_context_mask

        return overrides

    # ------------------------------------------------------------------
    # Preservation / regularization hooks
    # ------------------------------------------------------------------

    def pre_train_hook(self, args: argparse.Namespace, accelerator: Accelerator,
                       transformer=None, network=None) -> None:
        self._setup_preservation(args, accelerator)
        self._setup_tarp_dcr(args)
        self._setup_crepa(args, accelerator, transformer)
        self._setup_self_flow(args, accelerator, transformer, network)
        self._setup_audio_metrics(args)
        self._setup_hfato(args)
        self._apply_network_initialization(args, network)
        validate_lycoris_runtime(args, accelerator, transformer, network, logger)

    def _setup_audio_metrics(self, args: argparse.Namespace) -> None:
        """Parse audio metrics CLI flags.  No-op when --audio_metrics is not set."""
        if not getattr(args, "audio_metrics", False):
            return
        from musubi_tuner.audio_metrics import AudioMetricsConfig, AudioMetricsModule, parse_audio_metrics_args, _config_from_kwargs
        kw = parse_audio_metrics_args(getattr(args, "audio_metrics_args", None))
        config = _config_from_kwargs(kw) if kw else AudioMetricsConfig()
        self._audio_metrics = AudioMetricsModule(config)
        self._audio_metrics_checkpoint = getattr(args, "ltx2_checkpoint", None)
        self._audio_metrics_decoder = None  # lazy-loaded
        logger.info("Audio metrics enabled: latent_fd=%s temporal_coherence=%s av_latent_sync=%s mel=%s clap=%s",
                     config.latent_fd, config.temporal_coherence, config.av_latent_sync,
                     config.mel_metrics, config.clap_similarity)

    def _get_audio_decoder_for_metrics(self) -> torch.nn.Module | None:
        """Lazy-load AudioDecoder for mel-space metrics.  Cached on CPU."""
        if self._audio_metrics_decoder is not None:
            return self._audio_metrics_decoder
        ckpt = getattr(self, "_audio_metrics_checkpoint", None)
        if ckpt is None:
            return None
        try:
            from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
            from musubi_tuner.ltx_2.model.audio_vae.model_configurator import (
                AudioDecoderConfigurator,
                AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            )
            decoder = SingleGPUModelBuilder(
                model_path=str(ckpt),
                model_class_configurator=AudioDecoderConfigurator,
                model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            ).build(device=torch.device("cpu"), dtype=torch.bfloat16)
            decoder.eval()
            decoder.requires_grad_(False)
            self._audio_metrics_decoder = decoder
            logger.info("Audio metrics: loaded AudioDecoder for mel-space metrics")
            return decoder
        except Exception as e:
            logger.warning("Audio metrics: failed to load AudioDecoder: %s", e)
            return None

    def _setup_hfato(self, args: argparse.Namespace) -> None:
        """Parse HFATO CLI flags.  No-op when --hfato is not set."""
        if not getattr(args, "hfato", False):
            return
        ic_strategy = getattr(args, "ic_lora_strategy", "none")
        if ic_strategy not in ("none", "auto"):
            raise ValueError(f"--hfato is incompatible with --ic_lora_strategy {ic_strategy} (v2v/ref_latents path uses standard loss)")

        from musubi_tuner.hfato import HFATOConfig, parse_hfato_args

        kw = parse_hfato_args(getattr(args, "hfato_args", None))
        cfg_kwargs = {}
        float_keys = {"scale_factor", "probability"}
        for k, v in kw.items():
            if k in float_keys:
                cfg_kwargs[k] = float(v)
            elif k == "interpolation":
                if v not in ("bilinear", "nearest", "bicubic"):
                    raise ValueError(f"HFATO interpolation must be bilinear|nearest|bicubic, got: {v}")
                cfg_kwargs[k] = v
            else:
                raise ValueError(f"Unknown HFATO arg: {k}")

        config = HFATOConfig(**cfg_kwargs)
        if not (0.0 < config.scale_factor <= 1.0):
            raise ValueError(f"HFATO scale_factor must be in (0, 1], got: {config.scale_factor}")
        if not (0.0 < config.probability <= 1.0):
            raise ValueError(f"HFATO probability must be in (0, 1], got: {config.probability}")

        self._hfato_config = config
        logger.info(
            "HFATO enabled: scale_factor=%.2f interpolation=%s probability=%.2f",
            config.scale_factor, config.interpolation, config.probability,
        )

    def _setup_tarp_dcr(self, args: argparse.Namespace) -> None:
        """Parse TARP / DCR CLI flags.  No-op when no flags are set."""
        from musubi_tuner.preservation import parse_preservation_args

        self._tarp_enabled = bool(getattr(args, "tarp", False))
        self._dcr_enabled = bool(getattr(args, "dcr", False))

        tarp_kw = parse_preservation_args(getattr(args, "tarp_args", None))
        self._tarp_window_multiplier = int(tarp_kw.get("window_multiplier", 3))

        dcr_kw = parse_preservation_args(getattr(args, "dcr_args", None))
        self._dcr_reference_detach = dcr_kw.get("reference_detach", "true").lower() in ("true", "1", "yes")

        if self._tarp_enabled:
            if self._ltx_mode != "av":
                raise ValueError("--tarp requires --ltx2_mode av")
            logger.info("TARP enabled: window_multiplier=%d", self._tarp_window_multiplier)
        if self._dcr_enabled:
            if self._ltx_mode != "av":
                raise ValueError("--dcr requires --ltx2_mode av")
            logger.info("DCR enabled: per-sample routing, reference_detach=%s", self._dcr_reference_detach)

    def _setup_preservation(self, args: argparse.Namespace, accelerator: Accelerator) -> None:
        """Parse preservation CLI flags and prepare helper.  No-op when no flags are set."""
        blank = getattr(args, "blank_preservation", False)
        dop = getattr(args, "dop", False)
        prior_div = getattr(args, "prior_divergence", False)
        audio_dop = getattr(args, "audio_dop", False)

        if not (blank or dop or prior_div or audio_dop):
            return

        from musubi_tuner.preservation import PreservationConfig, PreservationHelper, parse_preservation_args

        # Validate audio_dop requirements
        if audio_dop:
            if self._ltx_mode != "av":
                raise ValueError("--audio_dop requires --ltx2_mode av (audio-video mode)")
            if getattr(args, "audio_silence_regularizer", False):
                logger.warning(
                    "Both --audio_dop and --audio_silence_regularizer are active. "
                    "The silence regularizer converts non-audio batches to audio batches, "
                    "so audio DOP will never fire. These are mutually exclusive."
                )

        blank_kw = parse_preservation_args(getattr(args, "blank_preservation_args", None))
        dop_kw = parse_preservation_args(getattr(args, "dop_args", None))
        prior_kw = parse_preservation_args(getattr(args, "prior_divergence_args", None))
        audio_dop_kw = parse_preservation_args(getattr(args, "audio_dop_args", None))

        cfg = PreservationConfig(
            blank_preservation=blank,
            blank_multiplier=float(blank_kw.get("multiplier", 1.0)),
            dop=dop,
            dop_multiplier=float(dop_kw.get("multiplier", 1.0)),
            dop_class_prompt=dop_kw.get("class", ""),
            prior_divergence=prior_div,
            prior_divergence_multiplier=float(prior_kw.get("multiplier", 0.1)),
            audio_dop=audio_dop,
            audio_dop_multiplier=float(audio_dop_kw.get("multiplier", 1.0)),
        )

        # Warn about DOP without class prompt (acts identical to blank preservation)
        if dop and not cfg.dop_class_prompt:
            logger.warning(
                "DOP enabled but no class prompt specified (--dop_args class=<prompt>). "
                "This will use an empty prompt, which is identical to blank preservation."
            )

        helper = PreservationHelper(cfg)
        helper.encode_prompts(self, args, accelerator)

        self._preservation_helper = helper
        self._preservation_active = True

        # Log VRAM impact: each technique adds extra transformer forward passes per step
        extra_fwd = 0
        extra_bwd = 0
        if blank:
            extra_fwd += 2  # no-grad OFF + with-grad ON
            extra_bwd += 1
        if dop:
            extra_fwd += 2
            extra_bwd += 1
        if prior_div:
            extra_fwd += 1  # no-grad OFF only
        if audio_dop:
            extra_fwd += 2  # no-grad OFF + with-grad ON (non-audio steps only)
            extra_bwd += 1
        logger.info(
            "Preservation enabled: blank=%s (x%.2f), dop=%s (class=%r, x%.2f), prior_div=%s (x%.3f), audio_dop=%s (x%.2f)",
            cfg.blank_preservation, cfg.blank_multiplier,
            cfg.dop, cfg.dop_class_prompt, cfg.dop_multiplier,
            cfg.prior_divergence, cfg.prior_divergence_multiplier,
            cfg.audio_dop, cfg.audio_dop_multiplier,
        )
        logger.warning(
            "Preservation adds +%d forward passes and +%d backward passes per training step. "
            "This significantly increases VRAM usage and step time.%s",
            extra_fwd, extra_bwd,
            " Audio DOP costs apply only on non-audio steps." if audio_dop else "",
        )

    def _setup_crepa(self, args: argparse.Namespace, accelerator: Accelerator,
                     transformer=None) -> None:
        """Parse CREPA CLI flags and install hooks.  No-op when ``--crepa`` is not set."""
        if not getattr(args, "crepa", False):
            return
        if transformer is None:
            logger.warning("CREPA enabled but transformer not available — skipping setup")
            return

        from musubi_tuner.crepa import CREPAConfig, CREPAModule, parse_crepa_args

        kw = parse_crepa_args(getattr(args, "crepa_args", None))

        # Build config — convert types from string values
        cfg_kwargs: Dict[str, Any] = {}
        _int_keys = {"student_block_idx", "teacher_block_idx", "num_neighbors", "warmup_steps", "max_steps"}
        _float_keys = {"lambda_crepa", "tau"}
        _bool_keys = {"normalize"}
        for k, v in kw.items():
            if k in _int_keys:
                cfg_kwargs[k] = int(v)
            elif k in _float_keys:
                cfg_kwargs[k] = float(v)
            elif k in _bool_keys:
                cfg_kwargs[k] = v.lower() in ("true", "1", "yes")
            else:
                cfg_kwargs[k] = v

        # Auto-fill max_steps for schedule
        if "max_steps" not in cfg_kwargs and hasattr(args, "max_train_steps"):
            cfg_kwargs["max_steps"] = args.max_train_steps

        config = CREPAConfig(**cfg_kwargs)

        unwrapped = accelerator.unwrap_model(transformer)
        module = CREPAModule(config, unwrapped)

        # Determine dtype from model
        first_param = next(iter(unwrapped.parameters()), None)
        dtype = first_param.dtype if first_param is not None else torch.float32

        module.setup(accelerator.device, dtype)

        # Try to load existing projector weights from state directory (for resume)
        if getattr(args, "resume", None):
            proj_path = os.path.join(args.resume, "crepa_projector.safetensors")
            if os.path.exists(proj_path):
                from safetensors.torch import load_file
                sd = load_file(proj_path)
                module.load_state_dict(sd)
                logger.info("CREPA: resumed projector weights from %s", proj_path)

        self._crepa = module

    def _setup_self_flow(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer=None,
        network=None,
    ) -> None:
        """Parse Self-Flow flags and install helper. No-op when ``--self_flow`` is not set."""
        if not getattr(args, "self_flow", False):
            return
        if transformer is None:
            logger.warning("Self-Flow enabled but transformer is unavailable — skipping setup")
            return
        if self._ltx_mode not in {"video", "av"}:
            raise ValueError("--self_flow currently supports --ltx_mode video or av (video branch only in av)")

        from musubi_tuner.self_flow import (
            SelfFlowConfig,
            SelfFlowModule,
            parse_self_flow_args,
        )

        kw = parse_self_flow_args(getattr(args, "self_flow_args", None))

        cfg_kwargs: Dict[str, Any] = {}
        int_keys = {
            "student_block_idx",
            "teacher_block_idx",
            "teacher_update_interval",
            "projector_hidden_multiplier",
            "num_neighbors",
            "patch_spatial_radius",
            "delta_num_steps",
            "temporal_warmup_steps",
            "temporal_max_steps",
            "student_block_stochastic_range",
        }
        float_keys = {
            "student_block_ratio",
            "teacher_block_ratio",
            "lambda_self_flow",
            "lambda_temporal",
            "lambda_delta",
            "temporal_tau",
            "patch_match_temperature",
            "motion_weight_strength",
            "mask_ratio",
            "max_loss",
            "teacher_momentum",
            "projector_lr",
            "lambda_audio",
        }
        bool_keys = {
            "dual_timestep",
            "tokenwise_timestep",
            "frame_level_mask",
            "mask_focus_loss",
            "offload_teacher_features",
            "offload_teacher_params",
        }
        for k, v in kw.items():
            if k in int_keys:
                cfg_kwargs[k] = int(v)
            elif k in float_keys:
                cfg_kwargs[k] = float(v)
            elif k in bool_keys:
                cfg_kwargs[k] = v.lower() in ("true", "1", "yes", "on")
            else:
                cfg_kwargs[k] = v

        if "temporal_max_steps" not in cfg_kwargs and hasattr(args, "max_train_steps"):
            cfg_kwargs["temporal_max_steps"] = args.max_train_steps

        config = SelfFlowConfig(**cfg_kwargs)
        if config.mask_ratio < 0.0 or config.mask_ratio > 0.5:
            raise ValueError("Self-Flow mask_ratio must be in [0, 0.5]")
        if config.teacher_momentum < 0.0 or config.teacher_momentum >= 1.0:
            raise ValueError("Self-Flow teacher_momentum must be in [0, 1)")
        if config.student_block_ratio is not None and not (0.0 < config.student_block_ratio < 1.0):
            raise ValueError("Self-Flow student_block_ratio must be in (0, 1)")
        if config.teacher_block_ratio is not None and not (0.0 < config.teacher_block_ratio < 1.0):
            raise ValueError("Self-Flow teacher_block_ratio must be in (0, 1)")
        if config.projector_lr is not None and config.projector_lr <= 0.0:
            raise ValueError("Self-Flow projector_lr must be > 0")
        if config.teacher_mode not in {"base", "ema", "partial_ema"}:
            raise ValueError("Self-Flow teacher_mode must be one of: base, ema, partial_ema")
        if config.student_block_stochastic_range < 0:
            raise ValueError("Self-Flow student_block_stochastic_range must be >= 0")
        if config.max_loss < 0.0:
            raise ValueError("Self-Flow max_loss must be >= 0")
        if config.loss_type not in {"negative_cosine", "one_minus_cosine"}:
            raise ValueError("Self-Flow loss_type must be one of: negative_cosine, one_minus_cosine")
        if config.temporal_mode not in {"off", "frame", "delta", "hybrid"}:
            raise ValueError("Self-Flow temporal_mode must be one of: off, frame, delta, hybrid")
        if config.temporal_granularity not in {"frame", "patch"}:
            raise ValueError("Self-Flow temporal_granularity must be one of: frame, patch")
        if config.patch_spatial_radius < 0:
            raise ValueError("Self-Flow patch_spatial_radius must be >= 0")
        if config.patch_match_mode not in {"hard", "soft"}:
            raise ValueError("Self-Flow patch_match_mode must be one of: hard, soft")
        if config.patch_match_temperature <= 0.0:
            raise ValueError("Self-Flow patch_match_temperature must be > 0")
        if config.delta_num_steps < 1:
            raise ValueError("Self-Flow delta_num_steps must be >= 1")
        if config.motion_weighting not in {"none", "teacher_delta"}:
            raise ValueError("Self-Flow motion_weighting must be one of: none, teacher_delta")
        if config.motion_weight_strength < 0.0:
            raise ValueError("Self-Flow motion_weight_strength must be >= 0")
        if config.lambda_temporal < 0.0:
            raise ValueError("Self-Flow lambda_temporal must be >= 0")
        if config.lambda_delta < 0.0:
            raise ValueError("Self-Flow lambda_delta must be >= 0")
        if config.temporal_tau <= 0.0:
            raise ValueError("Self-Flow temporal_tau must be > 0")
        if config.num_neighbors < 0:
            raise ValueError("Self-Flow num_neighbors must be >= 0")
        if config.temporal_schedule not in {"constant", "linear", "cosine"}:
            raise ValueError("Self-Flow temporal_schedule must be one of: constant, linear, cosine")
        if config.temporal_warmup_steps < 0:
            raise ValueError("Self-Flow temporal_warmup_steps must be >= 0")
        if config.temporal_max_steps < 0:
            raise ValueError("Self-Flow temporal_max_steps must be >= 0")

        unwrapped_transformer = accelerator.unwrap_model(transformer)
        if network is not None:
            unwrapped_network = accelerator.unwrap_model(network)
            self_flow_network = unwrapped_network
        else:
            # Full fine-tuning mode: transformer itself is the EMA target.
            # teacher_mode=base is incompatible (requires LoRA multipliers to create the gap).
            if str(config.teacher_mode).lower() == "base":
                raise ValueError(
                    "Self-Flow teacher_mode=base requires a LoRA network — it works by zeroing LoRA multipliers "
                    "to produce a base-model teacher pass. For full fine-tuning use teacher_mode=ema "
                    "(EMA over all transformer weights) or teacher_mode=partial_ema (EMA over teacher block only)."
                )
            self_flow_network = unwrapped_transformer
            logger.info(
                "Self-Flow: no LoRA network detected — using transformer as EMA target (teacher_mode=%s). "
                "teacher_mode=partial_ema is recommended to limit shadow-param memory to one block.",
                config.teacher_mode,
            )
        self._self_flow_network = self_flow_network
        module = SelfFlowModule(config, unwrapped_transformer)

        first_param = next(iter(unwrapped_transformer.parameters()), None)
        dtype = first_param.dtype if first_param is not None else torch.float32
        if isinstance(dtype, torch.dtype) and dtype.itemsize == 1:
            if args.mixed_precision == "fp16":
                dtype = torch.float16
            elif args.mixed_precision == "bf16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        module.setup(accelerator.device, dtype)
        module.init_teacher(self_flow_network)

        if getattr(args, "resume", None):
            proj_path = os.path.join(args.resume, "self_flow_projector.safetensors")
            if os.path.exists(proj_path):
                from safetensors.torch import load_file

                sd = load_file(proj_path)
                module.load_state_dict(sd)
                logger.info("Self-Flow: resumed projector weights from %s", proj_path)
            teacher_path = os.path.join(args.resume, "self_flow_teacher_ema.safetensors")
            if os.path.exists(teacher_path):
                from safetensors.torch import load_file

                teacher_sd = load_file(teacher_path)
                module.load_teacher_state_dict(teacher_sd)
                logger.info("Self-Flow: resumed EMA teacher state from %s", teacher_path)

        self._self_flow = module
        self._self_flow_active = True
        logger.warning(
            "Self-Flow is experimental and adds one extra teacher forward pass per step; expect higher VRAM/time cost."
        )

    def _apply_network_initialization(self, args: argparse.Namespace, network=None) -> None:
        """Apply network initialization customizations.

        Called after network creation to apply special initialization,
        for example LoKR perturbed normal.
        """
        if network is None:
            return

        # Apply special initialization if configured
        if hasattr(args, "_network_init_params"):
            init_params = args._network_init_params

            # LoKR perturbed normal initialization
            if "lokr_norm" in init_params:
                scale = init_params["lokr_norm"]
                logger.info(f"Applying LoKR perturbed normal initialization (scale={scale})")
                try:
                    from musubi_tuner.networks.lycoris_extensions import init_lokr_network_with_perturbed_normal
                    init_lokr_network_with_perturbed_normal(network, scale=scale)
                except Exception as e:
                    logger.warning(f"Failed to apply LoKR initialization: {e}")

    def compute_prior_divergence_addition(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        video_pred: torch.Tensor,
        network_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Return ``-MSE(video_pred, prior_pred) * mult`` or None."""
        if not self._preservation_active or self._preservation_helper is None:
            return None
        cfg = self._preservation_helper.config
        if not cfg.prior_divergence:
            return None
        dit_inputs = self._last_dit_inputs
        if dit_inputs is None:
            return None

        prior_pred = self._preservation_helper.compute_prior_divergence(
            self, transformer, network, accelerator, dit_inputs, network_dtype,
        )
        div_loss = -F.mse_loss(video_pred.float(), prior_pred.float()) * cfg.prior_divergence_multiplier
        if not torch.isfinite(div_loss):
            logger.warning("Prior divergence loss is non-finite (%.4g), skipping.", div_loss.item())
            return None
        return div_loss

    def preservation_backward(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        network_dtype: torch.dtype,
    ) -> Dict[str, float]:
        """Run preservation backward passes for blank and DOP.  Returns loss dict for logging."""
        if not self._preservation_active or self._preservation_helper is None:
            return {}
        dit_inputs = self._last_dit_inputs
        self._last_dit_inputs = None  # clear for next step
        if dit_inputs is None:
            return {}

        losses: Dict[str, float] = {}
        helper = self._preservation_helper
        cfg = helper.config

        if cfg.blank_preservation:
            val = helper.compute_preservation_backward(
                "blank", self, transformer, network, accelerator, dit_inputs, network_dtype,
            )
            losses["loss/blank_pres"] = val

        if cfg.dop:
            val = helper.compute_preservation_backward(
                "dop", self, transformer, network, accelerator, dit_inputs, network_dtype,
            )
            losses["loss/dop"] = val

        if cfg.audio_dop and self._ltx_mode == "av":
            is_non_audio_batch = dit_inputs.get("audio_model_timesteps") is None
            if is_non_audio_batch:
                av_inputs = self._build_audio_dop_inputs(args, accelerator, transformer, dit_inputs, network_dtype)
                if av_inputs is not None:
                    val = helper.compute_audio_dop_backward(
                        self, transformer, network, accelerator, av_inputs, network_dtype,
                    )
                    losses["loss/audio_dop"] = val

        return losses

    def _compute_validation_preservation_loss(
        self,
        technique: str,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        dit_inputs: Dict[str, Any],
        network_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        helper = self._preservation_helper
        if helper is None:
            return None

        cfg = helper.config
        if technique == "blank":
            embed, mask, mult = cfg.blank_embed, cfg.blank_mask, cfg.blank_multiplier
        elif technique == "dop":
            embed, mask, mult = cfg.dop_embed, cfg.dop_mask, cfg.dop_multiplier
        else:
            raise ValueError(f"Unknown preservation technique: {technique}")

        if embed is None or mask is None:
            return None

        device = accelerator.device
        bsz = dit_inputs["model_timesteps"].shape[0]
        pres_embed = embed.unsqueeze(0).expand(bsz, -1, -1).to(device=device, dtype=network_dtype)
        pres_mask = mask.unsqueeze(0).expand(bsz, -1).to(device=device)

        model_input = dit_inputs["model_input"]
        if isinstance(model_input, (list, tuple)):
            model_input = model_input[0]

        pres_inputs = {
            "model_input": model_input,
            "model_timesteps": dit_inputs["model_timesteps"],
            "text_embeds": pres_embed,
            "text_mask": pres_mask,
            "frame_rate": dit_inputs["frame_rate"],
            "transformer_options": dit_inputs["transformer_options"],
        }

        helper._prepare_block_swap(transformer, accelerator)
        network.set_multiplier(0.0)
        try:
            with torch.no_grad(), accelerator.autocast():
                prior_pred = transformer(
                    pres_inputs["model_input"],
                    timestep=pres_inputs["model_timesteps"],
                    context=pres_inputs["text_embeds"],
                    attention_mask=pres_inputs["text_mask"],
                    frame_rate=pres_inputs["frame_rate"],
                    transformer_options=pres_inputs["transformer_options"],
                )
            if isinstance(prior_pred, (list, tuple)):
                prior_pred = prior_pred[0]
            prior_pred = prior_pred.detach()
        finally:
            network.set_multiplier(1.0)

        helper._prepare_block_swap(transformer, accelerator)
        with torch.no_grad(), accelerator.autocast():
            pres_pred = transformer(
                pres_inputs["model_input"],
                timestep=pres_inputs["model_timesteps"],
                context=pres_inputs["text_embeds"],
                attention_mask=pres_inputs["text_mask"],
                frame_rate=pres_inputs["frame_rate"],
                transformer_options=pres_inputs["transformer_options"],
            )
        if isinstance(pres_pred, (list, tuple)):
            pres_pred = pres_pred[0]

        pres_loss = F.mse_loss(pres_pred.float(), prior_pred.float()) * mult
        if not torch.isfinite(pres_loss):
            logger.warning("Validation preservation %s loss is non-finite (%.4g), skipping.", technique, pres_loss.item())
            return None
        return pres_loss

    def _compute_validation_audio_dop_loss(
        self,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        av_inputs: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        helper = self._preservation_helper
        if helper is None:
            return None

        mult = float(helper.config.audio_dop_multiplier)
        helper._prepare_block_swap(transformer, accelerator)
        network.set_multiplier(0.0)
        try:
            with torch.no_grad(), accelerator.autocast():
                prior_pred = transformer(
                    av_inputs["model_input"],
                    timestep=av_inputs["model_timesteps"],
                    audio_timestep=av_inputs["audio_timestep"],
                    context=av_inputs["text_embeds"],
                    attention_mask=av_inputs["text_mask"],
                    frame_rate=av_inputs["frame_rate"],
                    transformer_options=av_inputs["transformer_options"],
                )
            if not isinstance(prior_pred, (list, tuple)) or len(prior_pred) < 2:
                logger.warning("Validation audio DOP: transformer did not return [video, audio] — skipping.")
                return None
            audio_prior = prior_pred[1].detach()
        finally:
            network.set_multiplier(1.0)

        helper._prepare_block_swap(transformer, accelerator)
        with torch.no_grad(), accelerator.autocast():
            lora_pred = transformer(
                av_inputs["model_input"],
                timestep=av_inputs["model_timesteps"],
                audio_timestep=av_inputs["audio_timestep"],
                context=av_inputs["text_embeds"],
                attention_mask=av_inputs["text_mask"],
                frame_rate=av_inputs["frame_rate"],
                transformer_options=av_inputs["transformer_options"],
            )
        if not isinstance(lora_pred, (list, tuple)) or len(lora_pred) < 2:
            logger.warning("Validation audio DOP: transformer did not return [video, audio] — skipping.")
            return None

        adop_loss = F.mse_loss(lora_pred[1].float(), audio_prior.float()) * mult
        if not torch.isfinite(adop_loss):
            logger.warning("Validation audio DOP loss is non-finite (%.4g), skipping.", adop_loss.item())
            return None
        return adop_loss

    def compute_validation_extra_loss(
        self,
        args,
        accelerator,
        transformer,
        network,
        batch,
        global_step: int,
        network_dtype,
    ):
        if not self._preservation_active or self._preservation_helper is None:
            return None, {}

        dit_inputs = self._last_dit_inputs
        if dit_inputs is None:
            return None, {}

        helper = self._preservation_helper
        cfg = helper.config
        total_loss = None
        metrics: Dict[str, float] = {}

        def _accumulate(name: str, value: Optional[torch.Tensor]) -> None:
            nonlocal total_loss
            if value is None:
                return
            total_loss = value if total_loss is None else total_loss + value
            metrics[name] = float(value.detach().item())

        if cfg.blank_preservation:
            _accumulate(
                "loss/blank_pres",
                self._compute_validation_preservation_loss(
                    "blank", accelerator, transformer, network, dit_inputs, network_dtype
                ),
            )

        if cfg.dop:
            _accumulate(
                "loss/dop",
                self._compute_validation_preservation_loss(
                    "dop", accelerator, transformer, network, dit_inputs, network_dtype
                ),
            )

        if cfg.audio_dop and self._ltx_mode == "av":
            is_non_audio_batch = dit_inputs.get("audio_model_timesteps") is None
            if is_non_audio_batch:
                av_inputs = self._build_audio_dop_inputs(args, accelerator, transformer, dit_inputs, network_dtype)
                if av_inputs is not None:
                    _accumulate(
                        "loss/audio_dop",
                        self._compute_validation_audio_dop_loss(
                            accelerator, transformer, network, av_inputs
                        ),
                    )

        return total_loss, metrics

    def compute_self_flow_addition(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer: torch.nn.Module,
        network: torch.nn.Module,
        network_dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Compute Self-Flow loss addition and logging values for the current step."""
        if not self._self_flow_active or self._self_flow is None:
            return None, {}
        if not bool(getattr(args, "self_flow", False)):
            return None, {}

        dit_inputs = self._last_dit_inputs
        sf_ctx = self._self_flow_step_context
        if dit_inputs is None or sf_ctx is None:
            self._self_flow.cleanup_step()
            return None, {}
        loss = self._self_flow.compute_loss_from_cached_features(
            num_latent_frames=sf_ctx.get("num_latent_frames"),
            latent_height=sf_ctx.get("latent_height"),
            latent_width=sf_ctx.get("latent_width"),
            token_mask=sf_ctx.get("dual_timestep_mask"),
        )

        metrics: Dict[str, float] = {}
        if loss is not None:
            metrics["loss/self_flow"] = float(loss.detach().item())
        cosine = self._self_flow.last_cosine
        if cosine is not None:
            metrics["self_flow/cosine"] = float(cosine)
        frame_cosine = self._self_flow.last_frame_cosine
        if frame_cosine is not None:
            metrics["self_flow/frame_cosine"] = float(frame_cosine)
        delta_cosine = self._self_flow.last_delta_cosine
        if delta_cosine is not None:
            metrics["self_flow/delta_cosine"] = float(delta_cosine)
        audio_cosine = self._self_flow.last_audio_cosine
        if audio_cosine is not None:
            metrics["self_flow/audio_cosine"] = float(audio_cosine)
        metrics["self_flow/lambda_self_flow"] = float(self._self_flow.current_lambda_self_flow)
        metrics["self_flow/lambda_audio"] = float(self._self_flow._current_lambda_audio)
        metrics["self_flow/lambda_temporal"] = float(self._self_flow.current_lambda_temporal)
        metrics["self_flow/lambda_delta"] = float(self._self_flow.current_lambda_delta)
        ema_drift = self._self_flow.last_ema_drift
        if ema_drift is not None:
            metrics["self_flow/ema_drift"] = float(ema_drift)
        if "masked_token_ratio" in sf_ctx:
            metrics["self_flow/masked_token_ratio"] = float(sf_ctx["masked_token_ratio"])
        if "tau_mean" in sf_ctx:
            metrics["self_flow/tau_mean"] = float(sf_ctx["tau_mean"])
        if "tau_min_mean" in sf_ctx:
            metrics["self_flow/tau_min_mean"] = float(sf_ctx["tau_min_mean"])
        if "audio_masked_token_ratio" in sf_ctx:
            metrics["self_flow/audio_masked_token_ratio"] = float(sf_ctx["audio_masked_token_ratio"])
        if "audio_tau_mean" in sf_ctx:
            metrics["self_flow/audio_tau_mean"] = float(sf_ctx["audio_tau_mean"])
        if "audio_tau_min_mean" in sf_ctx:
            metrics["self_flow/audio_tau_min_mean"] = float(sf_ctx["audio_tau_min_mean"])

        self._self_flow.cleanup_step()
        self._self_flow_step_context = None
        return loss, metrics


    def _load_ltx2_checkpoint_config(self, args: argparse.Namespace) -> Dict[str, Any]:
        if self._ltx2_checkpoint_config is not None:
            return self._ltx2_checkpoint_config

        from musubi_tuner.ltx_2.loader.sft_loader import SafetensorsModelStateDictLoader

        checkpoint_path = getattr(args, "ltx2_checkpoint", None)
        if checkpoint_path is None:
            raise ValueError("--ltx2_checkpoint is required to inspect checkpoint metadata")

        self._ltx2_checkpoint_config = SafetensorsModelStateDictLoader().metadata(str(checkpoint_path))
        return self._ltx2_checkpoint_config

    def _validate_ltx_version_consistency(self, args: argparse.Namespace) -> None:
        check_mode = str(getattr(args, "ltx_version_check_mode", "warn") or "warn").lower()
        if check_mode == "off":
            return
        if check_mode not in {"warn", "error"}:
            raise ValueError(
                f"Invalid ltx_version_check_mode={check_mode!r}. Expected one of: off, warn, error."
            )

        try:
            config = self._load_ltx2_checkpoint_config(args)
            detected_version, markers = infer_ltx_version_from_checkpoint_config(config)
        except Exception as exc:
            message = f"Failed to inspect checkpoint metadata for --ltx_version consistency check: {exc}"
            if check_mode == "error":
                raise ValueError(message) from exc
            logger.warning(message)
            return

        target_version = str(getattr(args, "ltx_version", self._ltx_version))
        if detected_version != target_version:
            marker_text = ", ".join(markers) if markers else "no explicit 2.3 markers"
            message = (
                f"--ltx_version={target_version} does not match checkpoint metadata (detected {detected_version}; "
                f"markers: {marker_text})."
            )
            if check_mode == "error":
                raise ValueError(message)
            logger.warning(message)
            return

        logger.info("LTX version check: --ltx_version=%s matches checkpoint metadata.", target_version)

    def _get_video_temporal_downsample(self) -> int:
        vae = getattr(self, "vae", None)
        return int(getattr(vae, "temporal_downsample_factor", 8))

    def _calculate_expected_audio_latent_length(
        self,
        args: argparse.Namespace,
        transformer,
        latent_frames: int,
        frame_rate: float,
    ) -> int:
        audio_cfg = self._get_audio_preview_config(args, transformer)
        video_temporal_factor = self._get_video_temporal_downsample()
        video_frames = max((latent_frames - 1) * video_temporal_factor + 1, 1)
        duration_s = float(video_frames) / max(float(frame_rate), 1.0)
        latents_per_second = (
            float(audio_cfg["sample_rate"])
            / float(audio_cfg["hop_length"])
            / float(audio_cfg["audio_latent_downsample_factor"])
        )
        return max(int(duration_s * latents_per_second), 1)

    def _adjust_audio_latent_duration(
        self,
        audio_latents: torch.Tensor,
        expected_length: int,
    ) -> torch.Tensor:
        actual_length = int(audio_latents.shape[2])
        if actual_length == expected_length:
            return audio_latents
        if actual_length > expected_length:
            logger.warning(
                "Trimming audio latents from %s to %s frames to match video duration.",
                actual_length,
                expected_length,
            )
            return audio_latents[:, :, :expected_length, :]
        padding_length = expected_length - actual_length
        logger.warning(
            "Padding audio latents from %s to %s frames (+%s) to match video duration.",
            actual_length,
            expected_length,
            padding_length,
        )
        padding = torch.zeros(
            audio_latents.shape[0],
            audio_latents.shape[1],
            padding_length,
            audio_latents.shape[3],
            device=audio_latents.device,
            dtype=audio_latents.dtype,
        )
        return torch.cat([audio_latents, padding], dim=2)

    def _build_empty_audio_latents(
        self,
        args: argparse.Namespace,
        transformer,
        latents: torch.Tensor,
        frame_rate: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        audio_cfg = self._get_audio_preview_config(args, transformer)
        expected_length = self._calculate_expected_audio_latent_length(
            args,
            transformer,
            latent_frames=int(latents.shape[2]),
            frame_rate=frame_rate,
        )
        return torch.zeros(
            (
                latents.shape[0],
                int(audio_cfg["channels"]),
                expected_length,
                int(audio_cfg["mel_bins"]),
            ),
            device=device,
            dtype=dtype,
        )

    def _build_audio_dop_inputs(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        dit_inputs: Dict[str, Any],
        network_dtype: torch.dtype,
    ) -> Optional[Dict[str, Any]]:
        """Build AV inputs for audio DOP from a non-audio batch's dit_inputs.

        Takes the current step's noisy video, constructs silence audio latents,
        noises them at the video sigma, duplicates text embeddings to 2×cc,
        and returns a dict ready for the transformer.
        """
        device = accelerator.device

        # Extract video tensor from model_input
        model_input = dit_inputs["model_input"]
        if isinstance(model_input, (list, tuple)):
            video_input = model_input[0]
        else:
            video_input = model_input

        # Get video sigma from timesteps
        model_timesteps = dit_inputs["model_timesteps"]
        sigma = model_timesteps[:, 0] if model_timesteps.dim() > 1 else model_timesteps

        # Get frame rate
        frame_rate = dit_inputs["frame_rate"]
        if isinstance(frame_rate, torch.Tensor):
            fr_float = frame_rate.item() if frame_rate.numel() == 1 else frame_rate[0].item()
        else:
            fr_float = float(frame_rate)

        # Build silence audio latents (zeros) with correct shape
        try:
            silence_audio = self._build_empty_audio_latents(
                args=args,
                transformer=transformer,
                latents=video_input,
                frame_rate=fr_float,
                device=device,
                dtype=network_dtype,
            )
        except Exception as e:
            logger.warning("Audio DOP: failed to build silence latents: %s", e)
            return None

        # Noise the silence audio using flow matching with video sigma
        audio_noise = torch.randn_like(silence_audio)
        sigma_audio = sigma.view(-1, 1, 1, 1).to(dtype=silence_audio.dtype)
        noisy_silence = (1.0 - sigma_audio) * silence_audio + sigma_audio * audio_noise
        del silence_audio, audio_noise

        # Build AV model_input: [noisy_video, noisy_silence_audio]
        av_model_input = [video_input, noisy_silence]

        # Duplicate text embeddings to 2×cc for AV forward
        text_embeds = dit_inputs["text_embeds"]
        if isinstance(text_embeds, torch.Tensor):
            # In non-audio batches, text_embeds is video-only (1×cc).
            # Duplicate to 2×cc so the wrapper can split into video + audio connectors.
            av_text_embeds = torch.cat([text_embeds, text_embeds], dim=-1)
        else:
            av_text_embeds = text_embeds

        # Audio timestep = video sigma (coupled timesteps for silence)
        audio_timestep = model_timesteps

        return {
            "model_input": av_model_input,
            "model_timesteps": model_timesteps,
            "audio_timestep": audio_timestep,
            "text_embeds": av_text_embeds,
            "text_mask": dit_inputs["text_mask"],
            "frame_rate": frame_rate,
            "transformer_options": dit_inputs["transformer_options"],
        }

    def _normalize_timesteps_for_model(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Normalize timesteps to the model's expected 0..1 sigma range."""
        if timesteps.numel() == 0:
            return timesteps

        return timesteps / 1000.0

    def _sample_independent_audio_timesteps(
        self,
        args: argparse.Namespace,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample audio timesteps in the same sigma range used by video timesteps."""
        min_timestep = getattr(args, "min_timestep", None)
        max_timestep = getattr(args, "max_timestep", None)
        min_sigma = (float(min_timestep) / 1000.0) if min_timestep is not None else 0.0
        max_sigma = (float(max_timestep) / 1000.0) if max_timestep is not None else 1.0
        if max_sigma < min_sigma:
            raise ValueError(f"Invalid timestep range: min_sigma={min_sigma} > max_sigma={max_sigma}")
        sigmas = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigmas = sigmas * (max_sigma - min_sigma) + min_sigma
        return sigmas.to(device=device, dtype=dtype).view(batch_size, 1)

    def _get_timestep_distribution_logging_payload(
        self,
        args: argparse.Namespace,
        timesteps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        payload = super()._get_timestep_distribution_logging_payload(args, timesteps)
        ctx = self._timestep_logging_context
        self._timestep_logging_context = None
        if not isinstance(ctx, dict):
            return payload

        audio_model_timesteps = ctx.get("audio_model_timesteps")
        if (
            bool(getattr(args, "independent_audio_timestep", False))
            and isinstance(audio_model_timesteps, torch.Tensor)
            and audio_model_timesteps.numel() > 0
        ):
            payload["audio"] = audio_model_timesteps.detach().to(dtype=torch.float32) * 1000.0
        shifted_logit_shifts = ctx.get("shifted_logit_shift")
        if isinstance(shifted_logit_shifts, torch.Tensor) and shifted_logit_shifts.numel() > 0:
            payload["shifted_logit_shift"] = shifted_logit_shifts.detach().to(dtype=torch.float32)
        return payload

    def _ensure_fp8_buffers_on_device(self, model: torch.nn.Module) -> None:
        if not any(True for _ in model.parameters()):
            return
        target_device = next(model.parameters()).device

        # If block swap is enabled, we must NOT call ensure_fp8_modules_on_device on the entire model
        # because it would move all swapped blocks from CPU to GPU, defeating block swapping.
        # Instead, process only non-swapped parts of the model.
        base_model = model.model if hasattr(model, "model") else model
        blocks_to_swap = getattr(base_model, "blocks_to_swap", 0) or 0

        if blocks_to_swap > 0 and hasattr(base_model, "transformer_blocks"):
            # Process non-block components (patchify, adaln, caption_projection, etc.)
            for name, child in base_model.named_children():
                if name == "transformer_blocks":
                    continue  # Skip transformer blocks - they are managed by block swap
                ensure_fp8_modules_on_device(child, target_device)

            # Only process non-swapped blocks (those that should always be on GPU)
            num_blocks = len(base_model.transformer_blocks)
            swap_start = max(0, num_blocks - blocks_to_swap)
            for idx, block in enumerate(base_model.transformer_blocks):
                if idx < swap_start:
                    # This block should be on GPU - ensure FP8 modules are on device
                    ensure_fp8_modules_on_device(block, target_device)
                # Skip swapped blocks - they are managed by the block swap mechanism
        else:
            # No block swap - process entire model as before
            ensure_fp8_modules_on_device(model, target_device)

    def _ensure_nf4_buffers_on_device(self, model: torch.nn.Module) -> None:
        """Move NF4 scale_weight buffers to the same device as the model weights.

        NF4 uint8 packed weights move naturally between CPU/GPU, but the
        scale_weight buffers (float) must be co-located with the weight for
        the dequantize forward to work.  This mirrors _ensure_fp8_buffers_on_device
        but uses the is_nf4_module check instead of FP8 dtype detection.
        """
        if not any(True for _ in model.parameters()):
            return
        target_device = next(model.parameters()).device

        base_model = model.model if hasattr(model, "model") else model
        blocks_to_swap = getattr(base_model, "blocks_to_swap", 0) or 0

        def _sync_nf4_buffers(module: torch.nn.Module, device: torch.device) -> None:
            for submodule in module.modules():
                if is_nf4_module(submodule):
                    sw = getattr(submodule, "scale_weight", None)
                    if isinstance(sw, torch.Tensor) and sw.device != device:
                        submodule.scale_weight = sw.to(device)
                    w = getattr(submodule, "weight", None)
                    if isinstance(w, torch.Tensor) and w.device != device:
                        submodule.weight = w.to(device)

        if blocks_to_swap > 0 and hasattr(base_model, "transformer_blocks"):
            for name, child in base_model.named_children():
                if name == "transformer_blocks":
                    continue
                _sync_nf4_buffers(child, target_device)
            num_blocks = len(base_model.transformer_blocks)
            swap_start = max(0, num_blocks - blocks_to_swap)
            for idx, block in enumerate(base_model.transformer_blocks):
                if idx < swap_start:
                    _sync_nf4_buffers(block, target_device)
        else:
            _sync_nf4_buffers(model, target_device)

    class _DeferredVAE:
        def __init__(self) -> None:
            self._deferred = True
            self.temporal_downsample_factor = 8
            self.spatial_downsample_factor = 32

        def to_device(self, _device) -> None:
            return None

        def to_dtype(self, _dtype) -> None:
            return None

        def eval(self) -> None:
            return None

        def requires_grad_(self, _requires_grad: bool = True):
            return self

    @staticmethod
    def _shifted_logit_normal_shift_for_sequence_length(
        seq_length: int,
        *,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
    ) -> float:
        """Calculate shift value for shifted logit-normal timestep sampling.

        This matches the LTX-2 trainer implementation where the shift
        is linearly interpolated based on sequence length.
        """
        m = (max_shift - min_shift) / float(max_tokens - min_tokens)
        b = min_shift - m * float(min_tokens)
        return m * float(seq_length) + b

    @staticmethod
    def _shifted_logit_normal_shift_for_sequence_lengths(
        seq_lengths: torch.Tensor,
        *,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
    ) -> torch.Tensor:
        m = (max_shift - min_shift) / float(max_tokens - min_tokens)
        b = min_shift - m * float(min_tokens)
        return seq_lengths.to(dtype=torch.float32) * float(m) + float(b)

    @staticmethod
    def _sample_shifted_logit_normal_sigmas(
        batch_size: int,
        shifts: torch.Tensor,
        *,
        std: float = 1.0,
        mode: str = "legacy",
        eps: float = 1e-3,
        uniform_prob: float = 0.1,
        uniform_samples: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample sigmas for shifted_logit_normal.

        Modes:
        - legacy: historical behavior, sigma = sigmoid(N(shift, std)).
        - stretched: percentile stretch behavior with optional uniform fallback.
        """
        if shifts.ndim != 1 or shifts.shape[0] != batch_size:
            raise ValueError(f"shifts must be shape [batch_size], got {tuple(shifts.shape)} for batch_size={batch_size}")

        shifts = shifts.to(dtype=torch.float32)
        std = float(std)
        mode = str(mode).lower()

        if uniform_samples is None:
            normal_samples = torch.randn((batch_size,), device=shifts.device, dtype=torch.float32) * std + shifts
        else:
            uniform_samples = uniform_samples.to(device=shifts.device, dtype=torch.float32).view(-1)
            if uniform_samples.numel() == 1 and batch_size > 1:
                uniform_samples = uniform_samples.expand(batch_size)
            if uniform_samples.numel() != batch_size:
                raise ValueError(
                    f"uniform_samples must be shape [batch_size], got {tuple(uniform_samples.shape)} "
                    f"for batch_size={batch_size}"
                )
            uniform_samples = uniform_samples.clamp(1e-7, 1.0 - 1e-7)
            standard_normal = math.sqrt(2.0) * torch.erfinv(2.0 * uniform_samples - 1.0)
            normal_samples = standard_normal * std + shifts
        logitnormal_samples = torch.sigmoid(normal_samples)
        if mode in {"legacy", "classic", "old"}:
            return logitnormal_samples
        if mode not in {"stretched", "v2", "normalized"}:
            raise ValueError(f"Invalid shifted_logit_mode={mode!r}. Expected one of: legacy, stretched.")

        # Constants for 99.9th and 0.5th normal percentiles.
        eps = min(max(float(eps), 0.0), 0.499)
        uniform_prob = min(max(float(uniform_prob), 0.0), 1.0)
        normal_999_percentile = 3.0902 * std
        normal_005_percentile = -2.5758 * std
        percentile_999 = torch.sigmoid(shifts + normal_999_percentile)
        percentile_005 = torch.sigmoid(shifts + normal_005_percentile)
        denom = (percentile_999 - percentile_005).clamp(min=1e-6)

        stretched = (logitnormal_samples - percentile_005) / denom
        stretched = torch.where(stretched >= eps, stretched, 2 * eps - stretched)
        stretched = stretched.clamp(0.0, 1.0)

        if uniform_prob <= 0.0:
            return stretched
        uniform = (1.0 - eps) * torch.rand((batch_size,), device=shifts.device, dtype=torch.float32) + eps
        if uniform_prob >= 1.0:
            return uniform
        prob = torch.rand((batch_size,), device=shifts.device, dtype=torch.float32)
        return torch.where(prob > uniform_prob, stretched, uniform)

    def _apply_shifted_logit_auto_shift_bounds(
        self,
        args: argparse.Namespace,
        shifts: torch.Tensor,
        *,
        force_clamp: bool = False,
    ) -> torch.Tensor:
        if getattr(args, "shifted_logit_shift", None) is not None:
            return shifts
        if not force_clamp and not bool(getattr(args, "shifted_logit_clamp_auto_shift", False)):
            return shifts

        min_shift = float(getattr(args, "shifted_logit_min_shift", 0.95))
        max_shift = float(getattr(args, "shifted_logit_max_shift", 2.05))
        return shifts.clamp(min=min_shift, max=max_shift)

    def _record_timestep_logging_tensor(self, name: str, value: torch.Tensor) -> None:
        ctx = self._timestep_logging_context if isinstance(self._timestep_logging_context, dict) else {}
        ctx[name] = value.detach()
        self._timestep_logging_context = ctx

    def _resolve_shifted_logit_mode(self, args: argparse.Namespace) -> str:
        explicit_mode = getattr(args, "shifted_logit_mode", None)
        if explicit_mode is not None:
            mode = str(explicit_mode).lower()
            if mode in {"legacy", "stretched"}:
                return mode
            raise ValueError(f"Invalid shifted_logit_mode={explicit_mode!r}. Expected one of: legacy, stretched.")

        # Route defaults by selected LTX version for backward compatibility.
        ltx_version = str(getattr(args, "ltx_version", self._ltx_version))
        return "stretched" if ltx_version == "2.3" else "legacy"

    def _resolve_audio_only_sequence_lengths(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        latents_info = self.get_current_batch_latents_info()
        if not isinstance(latents_info, dict):
            return None

        def _as_batch_int_tensor(value: Any) -> Optional[torch.Tensor]:
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return value.view(1).to(device=device, dtype=torch.int64).expand(batch_size)
                if value.numel() == batch_size:
                    return value.to(device=device, dtype=torch.int64).view(batch_size)
                return None
            if isinstance(value, (int, float)):
                return torch.full((batch_size,), int(value), device=device, dtype=torch.int64)
            return None

        num_frames = _as_batch_int_tensor(latents_info.get("num_frames"))
        if num_frames is None:
            return None

        # Audio-only mode does not optimize video loss; use a minimal virtual spatial
        # area by default to avoid over-scaling shifted_logit_normal with large
        # (irrelevant) video resolutions.
        seq_res = int(getattr(self, "_audio_only_sequence_resolution", 64))
        if seq_res > 0:
            spatial_downsample = int(getattr(getattr(self, "vae", None), "spatial_downsample_factor", 32))
            latent_hw = max(seq_res // max(spatial_downsample, 1), 1)
            return num_frames * latent_hw * latent_hw

        height = _as_batch_int_tensor(latents_info.get("height"))
        width = _as_batch_int_tensor(latents_info.get("width"))
        if height is None or width is None:
            return None
        return num_frames * height * width

    def _resolve_shifted_logit_normal_shift(
        self,
        args: argparse.Namespace,
        seq_len: int,
    ) -> float:
        """Resolve shifted-logit-normal shift for the current mode.

        Audio-only mode requires duration-aware video latents so seq_len
        reflects target token geometry.
        """
        if self._ltx_mode == "audio" and int(seq_len) <= 1:
            raise ValueError(
                "Audio-only training requires sequence-aware video latent geometry (seq_len>1). "
                "Re-cache latents with ltx2_cache_latents.py using --ltx2_mode audio "
                "to generate duration-aware geometry."
            )

        shift = self._shifted_logit_normal_shift_for_sequence_length(seq_len)
        shift_tensor = torch.tensor([float(shift)], dtype=torch.float32)
        shift_tensor = self._apply_shifted_logit_auto_shift_bounds(
            args,
            shift_tensor,
            force_clamp=self._ltx_mode == "audio",
        )
        shift = float(shift_tensor[0].item())
        if self._ltx_mode == "audio" and not self._logged_audio_only_timestep_shift:
            logger.info(
                "LTX-2 audio-only mode: using shifted_logit_normal shift %.4f from seq_len=%s.",
                shift,
                int(seq_len),
            )
            self._logged_audio_only_timestep_shift = True
        return shift

    def get_noisy_model_input_and_timesteps(
        self,
        args: argparse.Namespace,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: Optional[List[float]],
        noise_scheduler,
        device: torch.device,
        dtype: torch.dtype,
    ):
        # For non-video latents, use parent implementation
        if latents.dim() != 5:
            return super().get_noisy_model_input_and_timesteps(args, noise, latents, timesteps, noise_scheduler, device, dtype)

        if latents.device != device:
            latents = latents.to(device=device)
        if noise.device != device:
            noise = noise.to(device=device)
        self._self_flow_step_context = None
        self._timestep_logging_context = None

        batch_size = latents.shape[0]
        frames, height, width = latents.shape[2], latents.shape[3], latents.shape[4]
        seq_len = int(frames * height * width)
        timestep_uniforms: Optional[torch.Tensor] = None
        if timesteps is not None:
            timestep_uniforms = torch.as_tensor(timesteps, device=device, dtype=torch.float32).view(-1)
            if timestep_uniforms.numel() == 1 and batch_size > 1:
                timestep_uniforms = timestep_uniforms.expand(batch_size)
            if timestep_uniforms.numel() != batch_size:
                raise ValueError(
                    f"timesteps must contain {batch_size} values for LTX-2 sampling, "
                    f"got {timestep_uniforms.numel()}"
                )
            timestep_uniforms = timestep_uniforms.clamp(0.0, 1.0)
        audio_seq_lens = None
        if self._ltx_mode == "audio":
            audio_seq_lens = self._resolve_audio_only_sequence_lengths(batch_size, device)
            if audio_seq_lens is not None and torch.any(audio_seq_lens <= 1):
                raise ValueError(
                    "Audio-only training requires sequence-aware video latent geometry (seq_len>1). "
                    "Re-cache latents with ltx2_cache_latents.py using --ltx2_mode audio."
                )

        # Get timestep sampling mode (default to shifted_logit_normal for LTX-2)
        timestep_sampling = getattr(args, "timestep_sampling", "shifted_logit_normal")

        # For LTX-2, treat "sigma" as "shifted_logit_normal" (backward compatibility)
        if timestep_sampling == "sigma":
            timestep_sampling = "shifted_logit_normal"

        if timestep_sampling not in {"shifted_logit_normal", "uniform"}:
            # For other sampling modes, use parent implementation
            return super().get_noisy_model_input_and_timesteps(args, noise, latents, timesteps, noise_scheduler, device, dtype)

        min_timestep = getattr(args, "min_timestep", None)
        max_timestep = getattr(args, "max_timestep", None)
        min_sigma = (float(min_timestep) / 1000.0) if min_timestep is not None else 0.0
        max_sigma = (float(max_timestep) / 1000.0) if max_timestep is not None else 1.0
        if max_sigma < min_sigma:
            raise ValueError(f"Invalid timestep range: min_sigma={min_sigma} > max_sigma={max_sigma}")

        def _sample_base_uniform(*, use_provided: bool = True) -> Optional[torch.Tensor]:
            if use_provided and timestep_uniforms is not None:
                return timestep_uniforms
            if self.num_timestep_buckets is not None and self.num_timestep_buckets > 1:
                return torch.tensor(
                    [self.get_bucketed_timestep() for _ in range(batch_size)],
                    device=device,
                    dtype=torch.float32,
                )
            return None

        def _sample_raw_sigmas(base_uniform: Optional[torch.Tensor] = None) -> torch.Tensor:
            if timestep_sampling == "shifted_logit_normal":
                if self._ltx_mode == "audio":
                    if audio_seq_lens is not None:
                        shifts = self._shifted_logit_normal_shift_for_sequence_lengths(audio_seq_lens)
                        shifts = self._apply_shifted_logit_auto_shift_bounds(args, shifts, force_clamp=True)
                        if not self._logged_audio_only_timestep_shift:
                            logger.info(
                                "LTX-2 audio-only mode: shifted_logit_normal seq_len min=%s max=%s mean=%.2f, "
                                "shift min=%.4f max=%.4f mean=%.4f.",
                                int(audio_seq_lens.min().item()),
                                int(audio_seq_lens.max().item()),
                                float(audio_seq_lens.to(dtype=torch.float32).mean().item()),
                                float(shifts.min().item()),
                                float(shifts.max().item()),
                                float(shifts.mean().item()),
                            )
                            self._logged_audio_only_timestep_shift = True
                    else:
                        shift = self._resolve_shifted_logit_normal_shift(args, seq_len)
                        shifts = torch.full((batch_size,), float(shift), device=device, dtype=torch.float32)
                else:
                    shift = self._shifted_logit_normal_shift_for_sequence_length(seq_len)
                    shifts = torch.full((batch_size,), float(shift), device=device, dtype=torch.float32)
                    shifts = self._apply_shifted_logit_auto_shift_bounds(args, shifts)
                # Apply manual shift override if set
                shifted_logit_shift_override = getattr(args, "shifted_logit_shift", None)
                if shifted_logit_shift_override is not None:
                    shifts = torch.full((batch_size,), float(shifted_logit_shift_override), device=device, dtype=torch.float32)
                self._record_timestep_logging_tensor("shifted_logit_shift", shifts)
                std = getattr(args, "logit_std", 1.0)
                shifted_logit_mode = self._resolve_shifted_logit_mode(args)
                shifted_logit_eps = getattr(args, "shifted_logit_eps", 1e-3)
                shifted_logit_uniform_prob = getattr(args, "shifted_logit_uniform_prob", 0.1)
                sampled = self._sample_shifted_logit_normal_sigmas(
                    batch_size,
                    shifts,
                    std=std,
                    mode=shifted_logit_mode,
                    eps=shifted_logit_eps,
                    uniform_prob=shifted_logit_uniform_prob,
                    uniform_samples=base_uniform,
                )
            else:
                if base_uniform is None:
                    sampled = torch.rand((batch_size,), device=device, dtype=torch.float32)
                else:
                    sampled = base_uniform.to(device=device, dtype=torch.float32).view(-1)
                    if sampled.numel() == 1 and batch_size > 1:
                        sampled = sampled.expand(batch_size)
                    if sampled.numel() != batch_size:
                        raise ValueError(
                            f"base_uniform must contain {batch_size} values for LTX-2 uniform sampling, "
                            f"got {sampled.numel()}"
                        )
                    sampled = sampled.clamp(0.0, 1.0)
            return sampled

        def _sample_sigmas(*, use_provided: bool = True) -> torch.Tensor:
            if bool(getattr(args, "preserve_distribution_shape", False)) and (
                min_timestep is not None or max_timestep is not None
            ):
                max_loops = 1000
                available_sigmas: List[torch.Tensor] = []
                for _ in range(max_loops):
                    sampled = _sample_raw_sigmas(_sample_base_uniform(use_provided=False))
                    for sigma in sampled:
                        sigma_value = float(sigma.item())
                        if min_sigma <= sigma_value <= max_sigma:
                            available_sigmas.append(sigma)
                        if len(available_sigmas) == batch_size:
                            break
                    if len(available_sigmas) == batch_size:
                        break
                if len(available_sigmas) < batch_size:
                    logger.warning(
                        f"Could not sample {batch_size} valid LTX-2 timesteps in {max_loops} loops; "
                        "falling back to clamped samples."
                    )
                    sampled = _sample_raw_sigmas(_sample_base_uniform(use_provided=use_provided))
                    return sampled.clamp(min=min_sigma, max=max_sigma)
                return torch.stack(available_sigmas, dim=0).to(device=device, dtype=torch.float32)

            sampled = _sample_raw_sigmas(_sample_base_uniform(use_provided=use_provided))
            if min_timestep is not None or max_timestep is not None:
                sampled = sampled * (max_sigma - min_sigma) + min_sigma
            return sampled

        sigmas = _sample_sigmas()

        # Optional Self-Flow dual-timestep noising for video and AV modes.
        if (
            self._self_flow_active
            and self._self_flow is not None
            and self._ltx_mode in {"video", "av"}
            and bool(getattr(args, "self_flow", False))
            and bool(getattr(self._self_flow.config, "dual_timestep", True))
        ):
            sigmas_alt = _sample_sigmas(use_provided=False)
            t_tokens = sigmas.view(batch_size, 1).expand(batch_size, seq_len)
            s_tokens = sigmas_alt.view(batch_size, 1).expand(batch_size, seq_len)

            mask_ratio = float(getattr(self._self_flow.config, "mask_ratio", 0.10))
            mask_ratio = max(0.0, min(0.5, mask_ratio))
            if bool(getattr(self._self_flow.config, "frame_level_mask", False)):
                # Mask whole frames rather than individual tokens.
                frame_mask = torch.rand((batch_size, frames), device=device, dtype=torch.float32) < mask_ratio
                mask = frame_mask.unsqueeze(-1).expand(batch_size, frames, height * width).reshape(batch_size, seq_len)
            else:
                mask = torch.rand((batch_size, seq_len), device=device, dtype=torch.float32) < mask_ratio

            tau_tokens = torch.where(mask, s_tokens, t_tokens)
            tau_min = torch.minimum(sigmas, sigmas_alt)

            tau_latent = tau_tokens.view(batch_size, frames, height, width).unsqueeze(1)
            tau_min_latent = tau_min.view(batch_size, 1, 1, 1, 1)

            noisy_model_input = (1.0 - tau_latent) * latents.to(dtype=torch.float32) + tau_latent * noise.to(dtype=torch.float32)
            teacher_noisy = (1.0 - tau_min_latent) * latents.to(dtype=torch.float32) + tau_min_latent * noise.to(dtype=torch.float32)

            if bool(getattr(self._self_flow.config, "tokenwise_timestep", True)):
                timesteps_out = tau_tokens.to(device=device, dtype=torch.float32) * 1000.0
            else:
                timesteps_out = tau_tokens.mean(dim=1).to(device=device, dtype=torch.float32) * 1000.0
            teacher_timesteps = tau_min.to(device=device, dtype=torch.float32) * 1000.0

            self._self_flow_step_context = build_self_flow_video_context(
                base_sigmas=sigmas,
                alt_sigmas=sigmas_alt,
                teacher_noisy_model_input=teacher_noisy,
                teacher_model_timesteps=teacher_timesteps,
                dual_timestep_mask=mask,
                tau_tokens=tau_tokens,
                tau_min=tau_min,
                num_latent_frames=int(frames),
                latent_height=int(height),
                latent_width=int(width),
            )
            return noisy_model_input, timesteps_out

        sigmas_expanded = sigmas.view(-1, 1, 1, 1, 1)
        noisy_model_input = (1.0 - sigmas_expanded) * latents.to(dtype=torch.float32) + sigmas_expanded * noise.to(dtype=torch.float32)
        timesteps_out = sigmas.to(device=device, dtype=torch.float32) * 1000.0
        return noisy_model_input, timesteps_out

    # ======== Model-specific properties and configuration ========

    @property
    def architecture(self) -> str:
        """Returns architecture identifier"""
        from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LTX2

        return ARCHITECTURE_LTX2

    @property
    def architecture_full_name(self) -> str:
        """Returns full architecture name with version"""
        from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LTX2_FULL

        return ARCHITECTURE_LTX2_FULL

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        """Handle LTX-2-specific command line arguments"""
        self.dit_dtype = detect_ltx2_dtype(args.ltx2_checkpoint)
        if self.dit_dtype is not None and self.dit_dtype.itemsize == 1:
            if args.mixed_precision == "fp16":
                compute_dtype = torch.float16
            elif args.mixed_precision == "bf16":
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = torch.float32
            logger.warning(
                "LTX-2 weights are fp8; overriding compute dtype to %s for training stability.",
                compute_dtype,
            )
            self.dit_dtype = compute_dtype
        elif self.dit_dtype == torch.float32 and args.mixed_precision in ["fp16", "bf16"]:
            compute_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
            logger.warning(
                "LTX-2 weights are fp32; casting compute dtype to %s to reduce memory usage.",
                compute_dtype,
            )
            self.dit_dtype = compute_dtype

        if getattr(args, "nf4_base", False) and getattr(args, "fp8_base", False):
            raise ValueError("--nf4_base and --fp8_base are mutually exclusive")
        if getattr(args, "loftq_init", False) and not getattr(args, "nf4_base", False):
            raise ValueError("--loftq_init requires --nf4_base")
        if getattr(args, "awq_calibration", False) and not getattr(args, "nf4_base", False):
            raise ValueError("--awq_calibration requires --nf4_base")

        if getattr(args, "fp8_scaled", False):
            assert getattr(args, "fp8_base", False), "fp8_scaled requires fp8_base / fp8_scaledはfp8_baseが必要です"

        if getattr(args, "fp8_scaled", False) and self.dit_dtype is not None and self.dit_dtype.itemsize == 1:
            raise ValueError(
                "DiT weights is already in fp8 format, cannot scale to fp8. Please use fp16/bf16 weights / DiTの重みはすでにfp8形式です。fp8にスケーリングできません。fp16/bf16の重みを使用してください"
            )

        if getattr(args, "fp8_w8a8", False):
            if not getattr(args, "fp8_scaled", False):
                raise ValueError("--fp8_w8a8 requires --fp8_scaled")
            if not getattr(args, "network_module", None):
                raise ValueError("--fp8_w8a8 requires LoRA training (--network_module)")
            if getattr(args, "fp8_upcast", False):
                raise ValueError("--fp8_w8a8 and --fp8_upcast are mutually exclusive")

        validate_lycoris_quantized_base_compatibility(args, logger, DEFAULT_NF4_BLOCK_SIZE)

        if getattr(args, "save_original_lora", True) and not getattr(args, "convert_to_comfy", True):
            logger.info("--no_convert_to_comfy is set; original LoRA is always saved (--save_original_lora has no extra effect).")

        if self.dit_dtype == torch.float16:
            assert args.mixed_precision in ["fp16", "no"], "LTX-2 weights are fp16; mixed precision must be fp16 or no"
        elif self.dit_dtype == torch.bfloat16:
            assert args.mixed_precision in ["bf16", "no"], "LTX-2 weights are bf16; mixed precision must be bf16 or no"

        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)

        ltx_mode = getattr(args, "ltx_mode", "video")
        if ltx_mode not in {"video", "av", "audio"}:
            raise ValueError(f"Invalid ltx_mode: {ltx_mode}")
        self._ltx_mode = ltx_mode

        ltx_version = str(getattr(args, "ltx_version", "2.3"))
        if ltx_version not in {"2.0", "2.3"}:
            raise ValueError(f"Invalid ltx_version: {ltx_version}. Expected '2.0' or '2.3'.")
        self._ltx_version = ltx_version
        args.ltx_version = ltx_version
        ltx_version_check_mode = str(getattr(args, "ltx_version_check_mode", "warn") or "warn").lower()
        if ltx_version_check_mode not in {"off", "warn", "error"}:
            raise ValueError(
                f"ltx_version_check_mode must be one of ['off', 'warn', 'error']. Got: {ltx_version_check_mode}"
            )
        args.ltx_version_check_mode = ltx_version_check_mode
        self._validate_ltx_version_consistency(args)

        self._audio_video = self._ltx_mode in {"av", "audio"}
        self._train_connectors = bool(getattr(args, "train_connectors", False))
        self._ltx2_audio_only_model = bool(getattr(args, "ltx2_audio_only_model", False))
        if self._ltx2_audio_only_model and self._ltx_mode != "audio":
            raise ValueError("--ltx2_audio_only_model requires --ltx2_mode audio")
        self.default_guidance_scale = 1.0
        audio_only_sequence_resolution = int(getattr(args, "audio_only_sequence_resolution", 64))
        if audio_only_sequence_resolution != 0 and audio_only_sequence_resolution < 32:
            raise ValueError(
                "audio_only_sequence_resolution must be 0 (use cached virtual geometry) "
                f"or >= 32, got {audio_only_sequence_resolution}."
            )
        self._audio_only_sequence_resolution = audio_only_sequence_resolution

        args.weighting_scheme = "none"

        audio_balance_mode = str(getattr(args, "audio_loss_balance_mode", "none") or "none").lower()
        if audio_balance_mode not in {"none", "inv_freq", "ema_mag", "uncertainty", "ogm_ge"}:
            raise ValueError(
                f"audio_loss_balance_mode must be one of ['none', 'inv_freq', 'ema_mag', 'uncertainty', 'ogm_ge']. Got: {audio_balance_mode}"
            )
        args.audio_loss_balance_mode = audio_balance_mode

        audio_balance_beta = float(getattr(args, "audio_loss_balance_beta", 0.01))
        audio_balance_eps = float(getattr(args, "audio_loss_balance_eps", 0.05))
        audio_balance_min = float(getattr(args, "audio_loss_balance_min", 0.05))
        audio_balance_max = float(getattr(args, "audio_loss_balance_max", 4.0))
        audio_balance_ema_init = float(getattr(args, "audio_loss_balance_ema_init", 1.0))
        audio_balance_target_ratio = float(getattr(args, "audio_loss_balance_target_ratio", 0.33))
        audio_balance_ema_decay = float(getattr(args, "audio_loss_balance_ema_decay", 0.99))
        ogm_ge_alpha = float(getattr(args, "ogm_ge_alpha", 0.3))
        ogm_ge_noise_std = float(getattr(args, "ogm_ge_noise_std", 0.0))

        if not (0.0 < audio_balance_beta <= 1.0):
            raise ValueError(f"audio_loss_balance_beta must be in (0, 1]. Got: {audio_balance_beta}")
        if audio_balance_eps <= 0.0:
            raise ValueError(f"audio_loss_balance_eps must be > 0. Got: {audio_balance_eps}")
        if audio_balance_min < 0.0:
            raise ValueError(f"audio_loss_balance_min must be >= 0. Got: {audio_balance_min}")
        if audio_balance_max <= 0.0:
            raise ValueError(f"audio_loss_balance_max must be > 0. Got: {audio_balance_max}")
        if audio_balance_max < audio_balance_min:
            raise ValueError(
                f"audio_loss_balance_max must be >= audio_loss_balance_min. Got: min={audio_balance_min}, max={audio_balance_max}"
            )
        if audio_balance_mode == "inv_freq":
            if not (0.0 < audio_balance_ema_init <= 1.0):
                raise ValueError(f"audio_loss_balance_ema_init must be in (0, 1] for inv_freq. Got: {audio_balance_ema_init}")
        else:
            if audio_balance_ema_init <= 0.0:
                raise ValueError(f"audio_loss_balance_ema_init must be > 0. Got: {audio_balance_ema_init}")
        if audio_balance_target_ratio < 0.0:
            raise ValueError(f"audio_loss_balance_target_ratio must be >= 0. Got: {audio_balance_target_ratio}")
        if not (0.0 < audio_balance_ema_decay < 1.0):
            raise ValueError(f"audio_loss_balance_ema_decay must be in (0, 1). Got: {audio_balance_ema_decay}")
        if ogm_ge_alpha < 0.0:
            raise ValueError(f"ogm_ge_alpha must be >= 0. Got: {ogm_ge_alpha}")
        if ogm_ge_noise_std < 0.0:
            raise ValueError(f"ogm_ge_noise_std must be >= 0. Got: {ogm_ge_noise_std}")

        args.audio_loss_balance_beta = audio_balance_beta
        args.audio_loss_balance_eps = audio_balance_eps
        args.audio_loss_balance_min = audio_balance_min
        args.audio_loss_balance_max = audio_balance_max
        args.audio_loss_balance_ema_init = audio_balance_ema_init
        args.audio_loss_balance_target_ratio = audio_balance_target_ratio
        args.audio_loss_balance_ema_decay = audio_balance_ema_decay
        args.ogm_ge_alpha = ogm_ge_alpha
        args.ogm_ge_noise_std = ogm_ge_noise_std

        shifted_logit_mode = getattr(args, "shifted_logit_mode", None)
        if shifted_logit_mode is not None:
            shifted_logit_mode = str(shifted_logit_mode).lower()
            if shifted_logit_mode not in {"legacy", "stretched"}:
                raise ValueError(
                    f"shifted_logit_mode must be one of ['legacy', 'stretched']. Got: {shifted_logit_mode}"
                )
            args.shifted_logit_mode = shifted_logit_mode

        shifted_logit_eps = float(getattr(args, "shifted_logit_eps", 1e-3))
        shifted_logit_uniform_prob = float(getattr(args, "shifted_logit_uniform_prob", 0.1))
        if shifted_logit_eps < 0.0:
            raise ValueError(f"shifted_logit_eps must be >= 0. Got: {shifted_logit_eps}")
        if not (0.0 <= shifted_logit_uniform_prob <= 1.0):
            raise ValueError(
                f"shifted_logit_uniform_prob must be within [0, 1]. Got: {shifted_logit_uniform_prob}"
            )
        shifted_logit_min_shift = float(getattr(args, "shifted_logit_min_shift", 0.95))
        shifted_logit_max_shift = float(getattr(args, "shifted_logit_max_shift", 2.05))
        if not math.isfinite(shifted_logit_min_shift):
            raise ValueError(f"shifted_logit_min_shift must be finite. Got: {shifted_logit_min_shift}")
        if not math.isfinite(shifted_logit_max_shift):
            raise ValueError(f"shifted_logit_max_shift must be finite. Got: {shifted_logit_max_shift}")
        if shifted_logit_max_shift < shifted_logit_min_shift:
            raise ValueError(
                "shifted_logit_max_shift must be >= shifted_logit_min_shift. "
                f"Got: min={shifted_logit_min_shift}, max={shifted_logit_max_shift}"
            )
        args.shifted_logit_eps = shifted_logit_eps
        args.shifted_logit_uniform_prob = shifted_logit_uniform_prob
        args.shifted_logit_clamp_auto_shift = bool(getattr(args, "shifted_logit_clamp_auto_shift", False))
        args.shifted_logit_min_shift = shifted_logit_min_shift
        args.shifted_logit_max_shift = shifted_logit_max_shift

        args.independent_audio_timestep = bool(getattr(args, "independent_audio_timestep", False))
        args.audio_silence_regularizer = bool(getattr(args, "audio_silence_regularizer", False))
        audio_silence_regularizer_weight = float(getattr(args, "audio_silence_regularizer_weight", 1.0))
        if audio_silence_regularizer_weight < 0.0:
            raise ValueError(
                f"audio_silence_regularizer_weight must be >= 0. Got: {audio_silence_regularizer_weight}"
            )
        args.audio_silence_regularizer_weight = audio_silence_regularizer_weight

        audio_supervision_mode = normalize_audio_supervision_mode(
            getattr(args, "audio_supervision_mode", "off")
        )
        audio_supervision_warmup_steps = int(getattr(args, "audio_supervision_warmup_steps", 50))
        audio_supervision_check_interval = int(getattr(args, "audio_supervision_check_interval", 50))
        audio_supervision_min_ratio = float(getattr(args, "audio_supervision_min_ratio", 0.9))
        if audio_supervision_warmup_steps < 0:
            raise ValueError(
                f"audio_supervision_warmup_steps must be >= 0. Got: {audio_supervision_warmup_steps}"
            )
        if audio_supervision_check_interval <= 0:
            raise ValueError(
                f"audio_supervision_check_interval must be > 0. Got: {audio_supervision_check_interval}"
            )
        if not (0.0 <= audio_supervision_min_ratio <= 1.0):
            raise ValueError(
                f"audio_supervision_min_ratio must be in [0, 1]. Got: {audio_supervision_min_ratio}"
            )
        args.audio_supervision_mode = audio_supervision_mode
        args.audio_supervision_warmup_steps = audio_supervision_warmup_steps
        args.audio_supervision_check_interval = audio_supervision_check_interval
        args.audio_supervision_min_ratio = audio_supervision_min_ratio

        reset_audio_supervision_state(self._audio_supervision_state)

        ic_lora_strategy = str(getattr(args, "ic_lora_strategy", "auto") or "auto").lower()
        if ic_lora_strategy not in IC_LORA_STRATEGIES:
            raise ValueError(
                f"ic_lora_strategy must be one of {list(IC_LORA_STRATEGIES)}. Got: {ic_lora_strategy}"
            )
        if ic_lora_strategy == "auto":
            ic_lora_strategy = infer_ic_lora_strategy_from_preset(getattr(args, "lora_target_preset", "t2v"))

        if ic_lora_strategy == "audio_ref_only_ic" and self._ltx_mode not in {"av", "audio"}:
            raise ValueError("--ic_lora_strategy audio_ref_only_ic requires --ltx2_mode av or audio")
        if ic_lora_strategy == "av_ic" and self._ltx_mode != "av":
            raise ValueError(f"--ic_lora_strategy {ic_lora_strategy} requires --ltx2_mode av")
        if ic_lora_strategy == "video_ref_only_av" and self._ltx_mode != "av":
            raise ValueError("--ic_lora_strategy video_ref_only_av requires --ltx2_mode av")

        self._ic_lora_strategy = ic_lora_strategy
        args.ic_lora_strategy = ic_lora_strategy
        args.av_cross_attention_mode = _normalize_av_cross_attention_mode(
            getattr(args, "av_cross_attention_mode", "both")
        )
        args.av_multi_ref = bool(getattr(args, "av_multi_ref", False))
        args.audio_ref_use_negative_positions = bool(getattr(args, "audio_ref_use_negative_positions", False))
        args.audio_ref_mask_cross_attention_to_reference = bool(
            getattr(args, "audio_ref_mask_cross_attention_to_reference", False)
        )
        args.audio_ref_mask_reference_from_text_attention = bool(
            getattr(args, "audio_ref_mask_reference_from_text_attention", False)
        )
        if ic_lora_strategy == "av_ic" and args.audio_ref_mask_reference_from_text_attention:
            logger.warning(
                "%s: --audio_ref_mask_reference_from_text_attention is not supported "
                "(Modality API uses 2D context_mask). The flag will be ignored.",
                ic_lora_strategy,
            )
            args.audio_ref_mask_reference_from_text_attention = False
        if ic_lora_strategy not in {"av_ic", "video_ref_only_av"}:
            if args.av_cross_attention_mode != "both":
                logger.warning(
                    "av_cross_attention_mode=%s is set but --ic_lora_strategy is '%s'; the option will be ignored.",
                    args.av_cross_attention_mode,
                    ic_lora_strategy,
                )
            if args.av_multi_ref:
                logger.warning(
                    "av_multi_ref is enabled but --ic_lora_strategy is '%s'; the flag only affects av_ic metadata/UI and will be ignored.",
                    ic_lora_strategy,
                )
        args.audio_ref_identity_guidance_scale = float(getattr(args, "audio_ref_identity_guidance_scale", 0.0) or 0.0)
        if args.audio_ref_identity_guidance_scale < 0.0:
            raise ValueError(
                f"audio_ref_identity_guidance_scale must be >= 0. Got: {args.audio_ref_identity_guidance_scale}"
            )
        if ic_lora_strategy not in ("audio_ref_only_ic", "av_ic"):
            if (
                args.audio_ref_use_negative_positions
                or args.audio_ref_mask_cross_attention_to_reference
                or args.audio_ref_mask_reference_from_text_attention
                or args.audio_ref_identity_guidance_scale > 0.0
            ):
                logger.warning(
                    "audio_ref_* options are set but --ic_lora_strategy is '%s'; options will be ignored.",
                    ic_lora_strategy,
                )
        else:
            # Warn about recommended settings (based on ID-LoRA reference config).
            ic_label = ic_lora_strategy
            first_frame_p = float(getattr(args, "ltx2_first_frame_conditioning_p", 0.0))
            if first_frame_p < 0.01 and self._ltx_mode == "av":
                logger.warning(
                    "%s: --ltx2_first_frame_conditioning_p is %.2f (effectively off). "
                    "The ID-LoRA reference uses 0.9 — first-frame conditioning provides face identity "
                    "while the LoRA provides voice identity. Set --ltx2_first_frame_conditioning_p 0.9 "
                    "for best results.",
                    ic_label,
                    first_frame_p,
                )
            if not args.audio_ref_use_negative_positions:
                logger.warning(
                    "%s: --audio_ref_use_negative_positions is off. "
                    "The ID-LoRA reference enables this for clean positional separation "
                    "between reference and target audio tokens.",
                    ic_label,
                )
            if not args.audio_ref_mask_cross_attention_to_reference and self._ltx_mode == "av":
                logger.warning(
                    "%s: --audio_ref_mask_cross_attention_to_reference is off. "
                    "The ID-LoRA reference enables this during training so video attends "
                    "only to target audio (not reference). Masks are automatically disabled "
                    "during sampling/inference.",
                    ic_label,
                )
            if not args.audio_ref_mask_reference_from_text_attention:
                if ic_lora_strategy == "av_ic":
                    logger.warning(
                        "%s: --audio_ref_mask_reference_from_text_attention is not supported "
                        "in this AV IC mode (Modality API uses 2D context_mask). The flag is ignored.",
                        ic_label,
                    )
                else:
                    logger.warning(
                        "%s: --audio_ref_mask_reference_from_text_attention is off. "
                        "The ID-LoRA reference enables this during training to prevent reference "
                        "audio from attending to text (which describes target speech, not reference). "
                        "Masks are automatically disabled during sampling/inference.",
                        ic_label,
                    )

        # IC-LoRA strategies enable I2V-capable sampling flow in trainer.
        self._i2v_training = ic_lora_strategy in {
            "v2v",
            "audio_ref_only_ic",
            "av_ic",
            "video_ref_only_av",
        }

        apply_ltx2_tweaks(args)

    @property
    def i2v_training(self) -> bool:
        """True when training v2v / IC-LoRA (enables I2V conditioning in sampling)"""
        return self._i2v_training

    @property
    def control_training(self) -> bool:
        """LTX-2 doesn't currently support control conditioning"""
        return False

    def get_checkpoint_metadata(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Return LTX-2-specific metadata for LoRA safetensors (v2v mode info, etc.)."""
        md: Dict[str, Any] = {}
        preset = getattr(args, "lora_target_preset", None)
        if preset:
            md["ss_lora_target_preset"] = preset
        if self._ic_lora_strategy and self._ic_lora_strategy != "none":
            md["ss_ic_lora_strategy"] = self._ic_lora_strategy
        if self._ic_lora_strategy == "v2v":
            md["ss_v2v_training"] = True
        elif self._ic_lora_strategy == "self_ref_v2v":
            md["ss_self_ref_v2v_training"] = True
        elif self._ic_lora_strategy == "audio_ref_only_ic":
            md["ss_audio_ref_only_ic_training"] = True
        elif self._ic_lora_strategy == "av_ic":
            md["ss_av_ic_training"] = True
            av_cross_attention_mode = _normalize_av_cross_attention_mode(
                getattr(args, "av_cross_attention_mode", "both")
            )
            if av_cross_attention_mode != "both":
                md["ss_av_cross_attention_mode"] = av_cross_attention_mode
            if bool(getattr(args, "av_multi_ref", False)):
                md["ss_av_multi_ref"] = True
        elif self._ic_lora_strategy == "video_ref_only_av":
            md["ss_video_ref_only_av_training"] = True
        elif self._i2v_training:
            md["ss_i2v_training"] = True
        ref_downscale = max(1, getattr(args, "reference_downscale", 1))
        if ref_downscale != 1:
            md["ss_reference_downscale_factor"] = ref_downscale
        return md
    def post_save_checkpoint_hook(self, args, ckpt_file, ckpt_name, accelerator, force_sync_upload=False):
        """Convert saved LoRA to ComfyUI format."""
        if not getattr(args, 'convert_to_comfy', True):
            return

        try:
            from musubi_tuner.ltx_2.convert_lora_to_comfy import convert_lora_to_comfy
            comfy_ckpt_name = ckpt_name.replace('.safetensors', '.comfy.safetensors')
            comfy_ckpt_file = os.path.join(args.output_dir, comfy_ckpt_name)
            convert_lora_to_comfy(ckpt_file, comfy_ckpt_file, verbose=False)
            accelerator.print(f"Saved ComfyUI-compatible LoRA: {comfy_ckpt_file}")

            # Upload ComfyUI version to HuggingFace if enabled
            if args.huggingface_repo_id is not None:
                from musubi_tuner.utils import huggingface_utils
                huggingface_utils.upload(args, comfy_ckpt_file, "/" + comfy_ckpt_name, force_sync_upload=force_sync_upload)

            if not getattr(args, "save_original_lora", True):
                if os.path.exists(ckpt_file):
                    try:
                        os.remove(ckpt_file)  # --no_save_original_lora: keep only ComfyUI LoRA
                        accelerator.print(f"Removed original LoRA checkpoint (--no_save_original_lora): {ckpt_file}")
                    except Exception as e:
                        accelerator.print(f"Warning: Failed to remove original checkpoint '{ckpt_file}': {e}")
        except Exception as e:
            accelerator.print(f"Warning: Failed to convert LoRA to ComfyUI format: {e}")

    # ======== Model loading ========

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        """Load LTX-2 transformer model

        Args:
            accelerator: HF Accelerator instance
            args: Training arguments
            dit_path: Path to LTX-2 weights
            attn_mode: Attention implementation
            split_attn: Whether to split attention (ignored for LTX-2)
            loading_device: Device to load weights to
            dit_weight_dtype: Weight data type

        Returns:
            Loaded LTX-2 transformer model
        """
        # Determine attention mode from args
        if args.sdpa:
            attn_mode = "torch"
        elif args.flash_attn:
            attn_mode = "flash"
        elif args.flash3:
            attn_mode = "flash3"
        elif args.xformers:
            attn_mode = "xformers"
        else:
            attn_mode = "torch"

        self._dit_attn_mode = attn_mode

        torch_dtype_to_use = dit_weight_dtype or self.dit_dtype or torch.float32
        if dit_weight_dtype is None:
            logger.info("LTX-2 weight dtype not set; using %s for loading", torch_dtype_to_use)
        transformer = load_ltx2_model(
            model_path=dit_path,
            device=accelerator.device,
            load_device=loading_device,
            torch_dtype=torch_dtype_to_use,
            attn_mode=attn_mode,
            audio_video=self._audio_video,
            audio_only_model=self._ltx2_audio_only_model,
            split_attn_target=getattr(args, "split_attn_target", None),
            split_attn_mode=getattr(args, "split_attn_mode", None),
            split_attn_chunk_size=int(getattr(args, "split_attn_chunk_size", 0) or 0),
            ffn_chunk_target=getattr(args, "ffn_chunk_target", None),
            ffn_chunk_size=int(getattr(args, "ffn_chunk_size", 0) or 0),
            fp8_scaled=bool(getattr(args, "fp8_scaled", False)),
            fp8_w8a8=bool(getattr(args, "fp8_w8a8", False)),
            w8a8_mode=str(getattr(args, "w8a8_mode", "int8")),
            fp8_upcast=bool(getattr(args, "fp8_upcast", False)),
            fp8_upcast_stochastic=bool(getattr(args, "fp8_upcast_stochastic", False)),
            fp8_upcast_seed=int(getattr(args, "fp8_upcast_seed", 0)),
            fp8_keep_blocks=getattr(args, "fp8_keep_blocks", None),
            nf4_base=bool(getattr(args, "nf4_base", False)),
            nf4_block_size=int(getattr(args, "nf4_block_size", DEFAULT_NF4_BLOCK_SIZE)),
            loftq_init=bool(getattr(args, "loftq_init", False)),
            loftq_iters=int(getattr(args, "loftq_iters", 2)),
            lora_rank=int(getattr(args, "network_dim", 0) or 0),
            quantize_device=getattr(args, "quantize_device", None),
            awq_calibration=bool(getattr(args, "awq_calibration", False)),
            awq_alpha=float(getattr(args, "awq_alpha", 0.25)),
            awq_num_batches=int(getattr(args, "awq_num_batches", 8)),
        )

        transformer.eval()
        transformer.requires_grad_(False)

        # Connector LoRA: load connectors from checkpoint and attach to wrapper
        if self._train_connectors:
            from musubi_tuner.networks.lora_ltx2 import load_connectors_from_checkpoint
            from musubi_tuner.ltx_2.loader.sft_loader import SafetensorsModelStateDictLoader

            connector_config = SafetensorsModelStateDictLoader().metadata(str(dit_path))
            video_connector, audio_connector = load_connectors_from_checkpoint(
                str(dit_path),
                connector_config,
                audio_video=self._audio_video,
                device=accelerator.device,
                dtype=torch_dtype_to_use or torch.bfloat16,
            )
            video_connector.eval()
            video_connector.requires_grad_(False)
            if audio_connector is not None:
                audio_connector.eval()
                audio_connector.requires_grad_(False)
            transformer.load_connectors(video_connector, audio_connector)
            logger.info("Connector LoRA: attached connectors to transformer wrapper")

        return transformer

    def compile_transformer(self, args: argparse.Namespace, transformer):
        base_model = transformer.model if hasattr(transformer, "model") else transformer
        target_blocks = []
        if hasattr(base_model, "transformer_blocks"):
            target_blocks.append(base_model.transformer_blocks)
        return model_utils.compile_transformer(args, transformer, target_blocks, disable_linear=self.blocks_to_swap > 0)

    def _load_vae_impl(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        """Load VAE for LTX2"""
        logger.info(f"Loading VAE from {vae_path}")
        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.model.video_vae import VideoDecoderConfigurator, VAE_DECODER_COMFY_KEYS_FILTER

        class _LTX2VideoVAE(torch.nn.Module):
            def __init__(self, decoder: torch.nn.Module):
                super().__init__()
                self.decoder = decoder

                first_param = next(self.decoder.parameters())
                self.device = first_param.device
                self.dtype = first_param.dtype

                # LTX Video VAE configuration compresses frames by 8 (except the first frame) and spatial dims by 32.
                self.temporal_downsample_factor = 8
                self.spatial_downsample_factor = 32

                stats = getattr(self.decoder, "per_channel_statistics", None)
                self.latents_mean = None
                self.latents_std = None
                if stats is not None:
                    try:
                        self.latents_mean = stats.get_buffer("mean-of-means").detach().cpu()
                        self.latents_std = stats.get_buffer("std-of-means").detach().cpu()
                    except Exception:
                        self.latents_mean = None
                        self.latents_std = None

            def to_device(self, device: torch.device | str) -> None:
                self.device = torch.device(device)
                self.decoder.to(self.device)

            def to_dtype(self, dtype: torch.dtype) -> None:
                self.dtype = dtype
                self.decoder.to(dtype=dtype)

            def eval(self) -> None:
                self.decoder.eval()

            def requires_grad_(self, requires_grad: bool = True):
                self.decoder.requires_grad_(requires_grad)
                return self

            def decode(self, zs):
                outs = []
                for z in zs:
                    if z.dim() == 4:
                        z = z.unsqueeze(0)
                    z = z.to(device=self.device, dtype=self.dtype)
                    video = self.decoder(z)
                    outs.append(video.squeeze(0))
                return outs

            def tiled_decode(self, z, tiling_config=None):
                """Decode latents using tiled processing to reduce VRAM usage.
                
                Args:
                    z: Latent tensor [C, T, H, W] or [B, C, T, H, W]
                    tiling_config: TilingConfig object for spatial/temporal tiling
                    
                Returns:
                    Decoded video tensor [B, C, T, H, W]
                """
                if z.dim() == 4:
                    z = z.unsqueeze(0)
                z = z.to(device=self.device, dtype=self.dtype)
                
                # Collect all chunks from tiled decode generator
                chunks = []
                for frame_chunk in self.decoder.tiled_decode(z, tiling_config):
                    chunks.append(frame_chunk)
                
                # Concatenate along temporal dimension
                video = torch.cat(chunks, dim=2)  # [B, C, T, H, W]
                return video

        decoder = SingleGPUModelBuilder(
            model_path=str(vae_path),
            model_class_configurator=VideoDecoderConfigurator,
            model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
        ).build(device=torch.device("cpu"), dtype=vae_dtype)
        decoder.eval()
        decoder.requires_grad_(False)

        vae = _LTX2VideoVAE(decoder)
        self._update_latent_norm_base_from_vae(vae)
        return vae


    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        use_precached = bool(getattr(args, "use_precached_sample_prompts", False)) or bool(
            getattr(args, "precache_sample_prompts", False)
        )
        if getattr(args, "sample_prompts", None) or use_precached:
            logger.info("LTX-2 sampling: deferring VAE load until sampling")
            return self._DeferredVAE()
        return self._load_vae_impl(args, vae_dtype, vae_path)

    def _update_latent_norm_base_from_vae(self, vae) -> None:
        """Update latent normalization statistics from VAE config"""
        latents_mean = getattr(vae, "latents_mean", None)
        latents_std = getattr(vae, "latents_std", None)

        if latents_mean is None or latents_std is None:
            # Some VAE wrappers expose mean/std instead of latents_mean/latents_std
            latents_mean = getattr(vae, "mean", None)
            latents_std = getattr(vae, "std", None)

        if latents_mean is None or latents_std is None:
            config = getattr(vae, "config", None)
            if config is None:
                return
            latents_mean = getattr(config, "latents_mean", None)
            latents_std = getattr(config, "latents_std", None)

        if latents_mean is None or latents_std is None:
            return

        if isinstance(latents_mean, torch.Tensor):
            mean = latents_mean.to(dtype=torch.float32).view(1, -1, 1, 1, 1)
        else:
            mean = torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)

        if isinstance(latents_std, torch.Tensor):
            std = latents_std.to(dtype=torch.float32).view(1, -1, 1, 1, 1).clamp_min(1e-6)
        else:
            std = torch.tensor(latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1).clamp_min(1e-6)
        self._latent_norm_base = (mean, std.reciprocal())
        self._latent_norm_cache.clear()

    # ======== Training loop methods ========

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ) -> Tuple[object, torch.Tensor]:
        """Forward pass through LTX-2 (video or audio-video) model

        Args:
            args: Training arguments
            accelerator: HF Accelerator
            transformer: LTX-2 model
            latents: Video latents [B, 128, T, H, W]
            batch: Batch data including text embeddings
            noise: Noise tensor (same shape as latents)
            noisy_model_input: Noisy latents [B, 128, T, H, W]
            timesteps: Diffusion timesteps (normalized 0-1 for flow matching)
            network_dtype: Network precision

        Returns:
            Tuple of (model_prediction, target) for loss computation
        """
        diag_enabled = os.getenv("LTX2_NAN_DIAG", "0") == "1"
        skip_nonfinite = bool(getattr(args, "skip_nonfinite_steps", False))
        nonfinite_flag = {"hit": False, "tag": None}

        def _check_finite(tag: str, tensor: Optional[torch.Tensor]) -> None:
            if not skip_nonfinite or tensor is None:
                return
            if not torch.isfinite(tensor).all():
                bad = (~torch.isfinite(tensor)).sum().item()
                logger.error("%s has non-finite values (count=%s).", tag, bad)
                nonfinite_flag["hit"] = True
                nonfinite_flag["tag"] = tag
                return

        def _log_stats(tag: str, tensor: Optional[torch.Tensor]) -> None:
            if not diag_enabled or tensor is None:
                return
            with torch.no_grad():
                t = tensor.detach().float()
                logger.info(
                    "DIAG %s: shape=%s min=%.6f max=%.6f mean=%.6f std=%.6f",
                    tag,
                    tuple(t.shape),
                    float(t.min().item()),
                    float(t.max().item()),
                    float(t.mean().item()),
                    float(t.std().item()),
                )

        if not isinstance(batch, dict):
            raise TypeError(f"Expected batch to be a dict, got: {type(batch)}")

        def _resolve_loss_weight(batch_key: str, arg_key: str, default: float = 1.0) -> float:
            batch_value = batch.get(batch_key)
            if batch_value is not None:
                try:
                    return float(batch_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{batch_key} must be a float-compatible scalar, got {batch_value!r}") from exc
            return float(getattr(args, arg_key, default))

        if latents is None or not isinstance(latents, torch.Tensor):
            raise TypeError(f"Expected latents to be a torch.Tensor, got: {type(latents)}")
        if latents.dim() != 5:
            raise ValueError(f"Expected latents to be 5D [B, C, F, H, W], got shape: {tuple(latents.shape)}")
        in_channels = getattr(transformer, "in_channels", None)
        if in_channels is None and hasattr(transformer, "patchify_proj"):
            in_channels = getattr(getattr(transformer, "patchify_proj", None), "in_features", None)
        if in_channels is not None and latents.shape[1] != int(in_channels):
            raise ValueError(
                f"Latents channel mismatch: got {latents.shape[1]}, expected {int(in_channels)} (transformer.in_channels)"
            )
        if not torch.isfinite(latents).all():
            raise ValueError("Non-finite (NaN/Inf) detected in latents")
        _log_stats("latents", latents)

        if timesteps is None or not isinstance(timesteps, torch.Tensor):
            raise TypeError(f"Expected timesteps to be a torch.Tensor, got: {type(timesteps)}")

        conditions = batch.get("conditions")
        if conditions is not None:
            if not isinstance(conditions, dict):
                raise TypeError(f"Expected batch['conditions'] to be a dict, got: {type(conditions)}")
            if self._ltx_mode == "audio":
                text_embeds = conditions.get("audio_prompt_embeds")
                if text_embeds is None:
                    text_embeds = conditions.get("prompt_embeds")
                if text_embeds is None:
                    text_embeds = conditions.get("video_prompt_embeds")
            elif self._audio_video:
                video_prompt_embeds = conditions.get("video_prompt_embeds")
                audio_prompt_embeds = conditions.get("audio_prompt_embeds")
                if video_prompt_embeds is not None and audio_prompt_embeds is not None:
                    if not isinstance(video_prompt_embeds, torch.Tensor) or video_prompt_embeds.dim() != 3:
                        raise ValueError(
                            f"conditions['video_prompt_embeds'] must be a 3D tensor [B, seq_len, dim], "
                            f"got {type(video_prompt_embeds).__name__} "
                            f"{tuple(video_prompt_embeds.shape) if isinstance(video_prompt_embeds, torch.Tensor) else ''}"
                        )
                    if not isinstance(audio_prompt_embeds, torch.Tensor) or audio_prompt_embeds.dim() != 3:
                        raise ValueError(
                            f"conditions['audio_prompt_embeds'] must be a 3D tensor [B, seq_len, dim], "
                            f"got {type(audio_prompt_embeds).__name__} "
                            f"{tuple(audio_prompt_embeds.shape) if isinstance(audio_prompt_embeds, torch.Tensor) else ''}"
                        )
                    if video_prompt_embeds.shape[:2] != audio_prompt_embeds.shape[:2]:
                        raise ValueError(
                            f"video_prompt_embeds {tuple(video_prompt_embeds.shape)} and audio_prompt_embeds "
                            f"{tuple(audio_prompt_embeds.shape)} must have the same batch and seq_len dimensions. "
                            "Caches may have been created with different sequence length settings or different checkpoints."
                        )
                    # Per-modality caption dropout: independently zero video/audio embeddings
                    if getattr(self, "training", False):
                        v_drop = float(getattr(args, "video_caption_dropout_rate", 0.0))
                        a_drop = float(getattr(args, "audio_caption_dropout_rate", 0.0))
                        if v_drop > 0.0 or a_drop > 0.0:
                            video_prompt_embeds = video_prompt_embeds.clone()
                            audio_prompt_embeds = audio_prompt_embeds.clone()
                            for i in range(video_prompt_embeds.shape[0]):
                                if v_drop > 0.0 and random.random() < v_drop:
                                    video_prompt_embeds[i] = 0
                                if a_drop > 0.0 and random.random() < a_drop:
                                    audio_prompt_embeds[i] = 0
                    text_embeds = torch.cat([video_prompt_embeds, audio_prompt_embeds], dim=-1)
                else:
                    text_embeds = conditions.get("prompt_embeds")
            else:
                text_embeds = conditions.get("video_prompt_embeds")

            text_mask = conditions.get("prompt_attention_mask")
        else:
            text_embeds = batch.get("text")
            text_mask = batch.get("text_mask")

        if text_embeds is None:
            use_full_ft_fallback = bool(getattr(args, "full_ft_train_text_encoder", False)) and bool(
                getattr(args, "full_ft_text_encoder_fallback", False)
            )
            enable_prompt_grad = bool(getattr(args, "full_ft_train_text_encoder", False))
            if not use_full_ft_fallback:
                raise ValueError(
                    "Cached text embeddings missing from batch. Expected either batch['conditions'] (Gemma cache format) "
                    "or 'text'/'text_mask' (legacy musubi format)."
                )

            captions = _resolve_batch_captions(batch)
            if captions is None:
                raise ValueError(
                    "Cached text embeddings missing from batch and fallback is enabled, but no captions were found. "
                    "Set --full_ft_train_text_encoder with --full_ft_text_encoder_fallback and ensure batches include "
                    "per-sample captions."
                )
            if len(captions) != latents.shape[0]:
                raise ValueError(
                    "Captions count mismatch for runtime text encoding. Expected one caption per sample in the batch. "
                    f"captions={len(captions)} latents_batch={latents.shape[0]}"
                )
            if not self._warned_text_encoder_fallback:
                logger.warning(
                    "Runtime-fallbacking Gemma text encoder for %d captions because cached embeddings are missing.",
                    len(captions),
                )
                self._warned_text_encoder_fallback = True

            if not isinstance(conditions, dict):
                conditions = {}
            if self._text_encoder is None:
                text_encoder_dtype = self._build_text_encoder(args, accelerator)
            else:
                text_encoder_dtype = torch.float32
                first_param = next(self._text_encoder.parameters(), None)
                if first_param is not None:
                    text_encoder_dtype = first_param.dtype

            encoded_pairs = []
            for caption in captions:
                embed, mask = self._encode_prompt_text(
                    accelerator,
                    caption,
                    text_encoder_dtype,
                    allow_grad=enable_prompt_grad,
                )
                if not isinstance(embed, torch.Tensor) or not isinstance(mask, torch.Tensor):
                    raise TypeError("Gemma runtime text encoding must return tensors for both embedding and attention mask.")
                if embed.dim() != 2:
                    raise ValueError(
                        f"Runtime text embedding must be [seq_len, hidden], got {tuple(embed.shape)} for caption={caption!r}"
                    )
                if mask.dim() != 1:
                    raise ValueError(
                        f"Runtime text mask must be 1D, got {tuple(mask.shape)} for caption={caption!r}"
                    )
                if embed.shape[0] != mask.shape[0]:
                    raise ValueError(
                        f"Runtime text embedding length and mask length mismatch for caption={caption!r}: "
                        f"embed={tuple(embed.shape)} mask={tuple(mask.shape)}"
                    )
                encoded_pairs.append((embed, mask))

            embed_dim = int(encoded_pairs[0][0].shape[1])
            max_seq_len = max(int(embed.shape[0]) for embed, _ in encoded_pairs)
            fallback_device = accelerator.device if enable_prompt_grad else torch.device("cpu")
            fallback_text_embeds = torch.zeros(
                (latents.shape[0], max_seq_len, embed_dim),
                device=fallback_device,
                dtype=encoded_pairs[0][0].dtype,
            )
            fallback_text_mask = torch.zeros(
                (latents.shape[0], max_seq_len),
                device=fallback_device,
                dtype=encoded_pairs[0][1].dtype,
            )
            for sample_idx, (embed, mask) in enumerate(encoded_pairs):
                if int(embed.shape[1]) != embed_dim:
                    raise ValueError(
                        "Runtime text embedding hidden size must match across all captions in a batch. "
                        f"caption={captions[sample_idx]!r} has hidden={int(embed.shape[1])}, expected={embed_dim}"
                    )
                seq_len = int(embed.shape[0])
                fallback_text_embeds[sample_idx, :seq_len, :] = embed
                fallback_text_mask[sample_idx, :seq_len] = mask
            text_embeds = fallback_text_embeds
            text_mask = fallback_text_mask
            conditions["prompt_embeds"] = text_embeds

        base_model = transformer.model if hasattr(transformer, "model") else transformer
        expected_video_dim = int(getattr(base_model, "cross_attention_dim", 0) or 0)
        expected_audio_dim = int(getattr(base_model, "audio_cross_attention_dim", 0) or 0)

        if self._ltx_mode == "video" and isinstance(text_embeds, torch.Tensor):
            video_source = text_embeds
            if conditions is not None:
                prompt_embeds = conditions.get("prompt_embeds")
                if isinstance(prompt_embeds, torch.Tensor):
                    video_source = prompt_embeds
            text_embeds = select_video_text_embeds_for_video_mode(
                video_source,
                expected_video_dim=expected_video_dim,
                expected_audio_dim=expected_audio_dim,
            )

        if self._ltx_mode == "audio" and isinstance(text_embeds, torch.Tensor):
            text_embeds = select_audio_text_embeds_for_audio_mode(
                text_embeds,
                conditions,
                expected_audio_dim=expected_audio_dim,
                expected_video_dim=expected_video_dim,
            )

        # LTX-2.3 (caption_proj_before_connector=True) expects already-projected context
        # dimensions for each modality. In audio mode this must be audio_prompt_embeds
        # (audio_cross_attention_dim), not generic/video prompt embeds.
        if self._ltx_mode == "audio" and bool(getattr(base_model, "caption_proj_before_connector", False)):
            if expected_audio_dim > 0 and text_embeds.shape[-1] != expected_audio_dim:
                raise ValueError(
                    "Audio mode received text embeddings with incompatible hidden size for this checkpoint. "
                    f"Expected audio_prompt_embeds dim={expected_audio_dim}, got dim={text_embeds.shape[-1]}. "
                    "This usually means text encoder cache was created without audio embeddings. "
                    "Re-run ltx2_cache_text_encoder_outputs.py with --ltx2_mode audio (or av) using the same "
                    "--ltx2_checkpoint, then train again."
                )

        if not isinstance(text_embeds, torch.Tensor):
            raise TypeError(f"Expected text embeddings to be a torch.Tensor, got: {type(text_embeds)}")
        if text_embeds.dim() != 3:
            raise ValueError(f"Expected text embeddings to be 3D [B, seq_len, hidden_dim], got shape: {tuple(text_embeds.shape)}")
        if text_embeds.shape[0] != latents.shape[0]:
            raise ValueError(f"Batch size mismatch: latents batch={latents.shape[0]} vs text batch={text_embeds.shape[0]}")

        text_embeds = text_embeds.to(device=accelerator.device, dtype=network_dtype)
        _log_stats("text_embeds", text_embeds)

        # Check for NaN values
        if torch.isnan(text_embeds).any():
            raise ValueError("NaN detected in cached text embeddings!")

        if text_mask is not None:
            if not isinstance(text_mask, torch.Tensor):
                raise TypeError(f"Expected text_mask to be a torch.Tensor, got: {type(text_mask)}")
            if text_mask.dim() != 2:
                raise ValueError(f"Expected text_mask to be 2D [B, seq_len], got shape: {tuple(text_mask.shape)}")
            if text_mask.shape[0] != latents.shape[0]:
                raise ValueError(f"Batch size mismatch: latents batch={latents.shape[0]} vs text_mask batch={text_mask.shape[0]}")
            if text_mask.shape[1] != text_embeds.shape[1]:
                raise ValueError(
                    f"text_mask seq_len ({text_mask.shape[1]}) does not match text_embeds seq_len ({text_embeds.shape[1]}). "
                    "This usually means the attention mask and text embedding caches were created with different "
                    "sequence length settings. Re-run the text encoder caching step."
                )
            text_mask = text_mask.to(device=accelerator.device)
            if args.gradient_checkpointing:
                text_mask = text_mask.to(torch.bool)

        # Caption dropout: zero out text conditioning with probability p (for CFG training)
        caption_dropout_rate = getattr(args, "caption_dropout_rate", 0.0)
        if caption_dropout_rate > 0.0 and getattr(self, "training", False):
            text_embeds, text_mask = self._apply_caption_dropout(text_embeds, text_mask, caption_dropout_rate)

        # Move latents to device
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        noise = noise.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)

        def _cached_video_loss_mask(*, as_tokens: bool) -> Optional[torch.Tensor]:
            return _coerce_video_loss_mask(
                batch.get("video_loss_mask"),
                latents=latents,
                device=accelerator.device,
                dtype=network_dtype,
                as_tokens=as_tokens,
            )

        def _cached_audio_loss_mask(seq_len: int, batch_size: int) -> Optional[torch.Tensor]:
            return _coerce_audio_loss_mask(
                batch.get("audio_loss_mask"),
                batch_size=batch_size,
                seq_len=seq_len,
                device=accelerator.device,
                dtype=network_dtype,
            )

        # Check for NaN in latents
        if torch.isnan(latents).any():
            raise ValueError("NaN detected in latents!")

        # Get frame rate from batch or use default
        frame_rate = batch.get("frame_rate", None)
        if frame_rate is None:
            latents_info = batch.get("latents")
            if isinstance(latents_info, dict):
                frame_rate = latents_info.get("fps", None)
        if frame_rate is None:
            frame_rate = 25
        if isinstance(frame_rate, torch.Tensor):
            frame_rate = frame_rate.item() if frame_rate.numel() == 1 else frame_rate[0].item()

        model_timesteps = timesteps.to(device=accelerator.device, dtype=network_dtype)

        model_timesteps = self._normalize_timesteps_for_model(model_timesteps)

        if model_timesteps.dim() == 0:
            model_timesteps = model_timesteps.unsqueeze(0)
        if model_timesteps.dim() == 1:
            model_timesteps = model_timesteps.unsqueeze(1)

        sigma = model_timesteps[:, 0]
        audio_model_timesteps = model_timesteps
        if self._ltx_mode in {"av", "audio"} and model_timesteps.dim() == 2 and model_timesteps.shape[1] > 1:
            # Self-Flow token-wise video timesteps can have a different token length than audio.
            # Keep audio timesteps per-sample unless explicitly overridden below.
            audio_model_timesteps = model_timesteps[:, :1]
        if self._ltx_mode in {"av", "audio"} and bool(getattr(args, "independent_audio_timestep", False)):
            audio_model_timesteps = self._sample_independent_audio_timesteps(
                args,
                batch_size=model_timesteps.shape[0],
                device=accelerator.device,
                dtype=network_dtype,
            )
        timestep_logging_context = (
            self._timestep_logging_context if isinstance(self._timestep_logging_context, dict) else {}
        )
        timestep_logging_context["audio_model_timesteps"] = audio_model_timesteps.detach()
        self._timestep_logging_context = timestep_logging_context
        audio_sigma = audio_model_timesteps[:, 0]
        ic_lora_strategy = str(
            getattr(
                args,
                "ic_lora_strategy",
                self._ic_lora_strategy
                or infer_ic_lora_strategy_from_preset(getattr(args, "lora_target_preset", "t2v")),
            )
            or "none"
        ).lower()
        audio_ref_only_ic_enabled = ic_lora_strategy == "audio_ref_only_ic"

        ref_latent_tensors = _collect_reference_tensors(batch, "ref_latents", expected_ndim=5)
        ref_latents = _merge_reference_tensors(ref_latent_tensors, concat_dim=2)

        if ref_latents is not None:
            if ic_lora_strategy not in ("v2v", "av_ic", "video_ref_only_av"):
                if not self._warned_ignored_ref_latents:
                    logger.warning(
                        "ref_latents were provided but --ic_lora_strategy is '%s'; ignoring reference-video conditioning.",
                        ic_lora_strategy,
                    )
                    self._warned_ignored_ref_latents = True
                ref_latents = None
            elif ic_lora_strategy == "v2v":
                if self._audio_video or self._ltx_mode != "video":
                    raise ValueError("Reference latent conditioning is only supported for video-only LTX-2 training")
                if not isinstance(ref_latents, torch.Tensor):
                    raise TypeError(f"Expected ref_latents to be a torch.Tensor, got: {type(ref_latents)}")
                if ref_latents.dim() != 5:
                    raise ValueError(f"Expected ref_latents to be 5D [B, C, F, H, W], got shape: {tuple(ref_latents.shape)}")
                if ref_latents.shape[0] != latents.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: latents batch={latents.shape[0]} vs ref_latents batch={ref_latents.shape[0]}"
                    )
                if ref_latents.shape[1] != latents.shape[1]:
                    raise ValueError(f"Channel mismatch: latents C={latents.shape[1]} vs ref_latents C={ref_latents.shape[1]}")
                ref_h, ref_w = int(ref_latents.shape[3]), int(ref_latents.shape[4])
                tgt_h, tgt_w = int(latents.shape[3]), int(latents.shape[4])
                if ref_h == tgt_h and ref_w == tgt_w:
                    reference_downscale_factor = 1
                else:
                    h_ratio = tgt_h / ref_h
                    w_ratio = tgt_w / ref_w
                    if abs(h_ratio - w_ratio) > 0.01 or abs(h_ratio - round(h_ratio)) > 0.01:
                        raise ValueError(
                            f"Spatial mismatch: latents HxW={tgt_h}x{tgt_w} vs ref_latents HxW={ref_h}x{ref_w}. "
                            f"Ratios h={h_ratio:.2f} w={w_ratio:.2f} are not consistent integer downscale factors."
                        )
                    reference_downscale_factor = round(h_ratio)
            else:
                # AV IC / video_ref_only_av: AV reference-video conditioning.
                if self._ltx_mode != "av":
                    raise ValueError(f"--ic_lora_strategy {ic_lora_strategy} requires --ltx2_mode av")
                if not isinstance(ref_latents, torch.Tensor):
                    raise TypeError(f"Expected ref_latents to be a torch.Tensor, got: {type(ref_latents)}")
                if ref_latents.dim() != 5:
                    raise ValueError(f"Expected ref_latents to be 5D [B, C, F, H, W], got shape: {tuple(ref_latents.shape)}")
                if ref_latents.shape[0] != latents.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: latents batch={latents.shape[0]} vs ref_latents batch={ref_latents.shape[0]}"
                    )
                if ref_latents.shape[1] != latents.shape[1]:
                    raise ValueError(f"Channel mismatch: latents C={latents.shape[1]} vs ref_latents C={ref_latents.shape[1]}")
                ref_h, ref_w = int(ref_latents.shape[3]), int(ref_latents.shape[4])
                tgt_h, tgt_w = int(latents.shape[3]), int(latents.shape[4])
                if ref_h == tgt_h and ref_w == tgt_w:
                    reference_downscale_factor = 1
                else:
                    h_ratio = tgt_h / ref_h
                    w_ratio = tgt_w / ref_w
                    if abs(h_ratio - w_ratio) > 0.01 or abs(h_ratio - round(h_ratio)) > 0.01:
                        raise ValueError(
                            f"av_ic spatial mismatch: latents HxW={tgt_h}x{tgt_w} vs ref_latents HxW={ref_h}x{ref_w}. "
                            f"Ratios h={h_ratio:.2f} w={w_ratio:.2f} are not consistent integer downscale factors."
                        )
                    reference_downscale_factor = round(h_ratio)

        if self._ltx_mode == "audio":
            audio_latents = batch.get("audio_latents")
            if isinstance(audio_latents, dict):
                audio_latents = audio_latents.get("latents")
            if audio_latents is None:
                raise ValueError("audio_latents are required for --ltx_mode audio")
            if not isinstance(audio_latents, torch.Tensor):
                raise TypeError(f"Expected audio_latents to be a torch.Tensor, got: {type(audio_latents)}")
            if audio_latents.dim() != 4:
                raise ValueError(f"Expected audio_latents to be 4D [B, C, T, F], got shape: {tuple(audio_latents.shape)}")
            if audio_latents.shape[0] != latents.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: latents batch={latents.shape[0]} vs audio_latents batch={audio_latents.shape[0]}"
                )

            audio_latents = audio_latents.to(device=accelerator.device, dtype=network_dtype)
            audio_noise = torch.randn_like(audio_latents)
            sigma_audio = audio_sigma.view(-1, 1, 1, 1)
            noisy_audio = (1.0 - sigma_audio) * audio_latents + sigma_audio * audio_noise

            # Compute target and loss mask BEFORE IC block so they can be concatenated with ref tokens.
            audio_target = audio_noise - audio_latents
            audio_seq_len = int(audio_latents.shape[2])
            audio_loss_mask = torch.ones(
                (audio_latents.shape[0], audio_seq_len),
                device=accelerator.device,
                dtype=torch.bool,
            )

            # Audio-only mode always masks padding to prevent loss on zero-padded
            # positions that arise from batching variable-length audio clips.
            audio_lengths = batch.get("audio_lengths")
            if isinstance(audio_lengths, dict):
                audio_lengths = audio_lengths.get("lengths")
            if isinstance(audio_lengths, torch.Tensor):
                if audio_lengths.dim() == 0:
                    audio_lengths = audio_lengths.view(1)
                if audio_lengths.dim() != 1:
                    raise ValueError(f"Expected audio_lengths to be 1D [B] or scalar, got shape: {tuple(audio_lengths.shape)}")
                if audio_lengths.numel() == 1 and audio_latents.shape[0] != 1:
                    audio_lengths = audio_lengths.expand(audio_latents.shape[0])
                if audio_lengths.shape[0] != audio_latents.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: audio_latents batch={audio_latents.shape[0]} vs audio_lengths batch={audio_lengths.shape[0]}"
                    )

                audio_lengths = audio_lengths.to(device=accelerator.device, dtype=torch.int64)
                audio_lengths = audio_lengths.clamp(min=0, max=audio_seq_len)
                t = torch.arange(audio_seq_len, device=accelerator.device).view(1, -1)
                audio_loss_mask = t < audio_lengths.view(-1, 1)
            audio_loss_mask = _combine_loss_masks(
                audio_loss_mask,
                _cached_audio_loss_mask(audio_seq_len, int(audio_latents.shape[0])),
            )

            video_latents = torch.zeros(
                (latents.shape[0], latents.shape[1], 1, 1, 1),
                device=accelerator.device,
                dtype=network_dtype,
            )

            audio_timestep_local = audio_model_timesteps
            resolved_transformer_options: Dict[str, Any] = {"patches_replace": {}}
            ref_audio_seq_len = 0

            if audio_ref_only_ic_enabled:
                ref_audio_latents = batch.get("ref_audio_latents")
                if isinstance(ref_audio_latents, dict):
                    ref_audio_latents = ref_audio_latents.get("latents")
                if ref_audio_latents is None:
                    raise ValueError(
                        "--ic_lora_strategy audio_ref_only_ic requires ref_audio_latents. "
                        "Set reference_audio_directory/reference_audio_cache_directory and cache reference audio latents."
                    )
                if not isinstance(ref_audio_latents, torch.Tensor):
                    raise TypeError(f"Expected ref_audio_latents to be a torch.Tensor, got: {type(ref_audio_latents)}")
                if ref_audio_latents.dim() != 4:
                    raise ValueError(
                        f"Expected ref_audio_latents to be 4D [B, C, T, F], got shape: {tuple(ref_audio_latents.shape)}"
                    )
                if ref_audio_latents.shape[0] != latents.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: latents batch={latents.shape[0]} vs ref_audio_latents batch={ref_audio_latents.shape[0]}"
                    )
                if ref_audio_latents.shape[1] != audio_latents.shape[1] or ref_audio_latents.shape[3] != audio_latents.shape[3]:
                    raise ValueError(
                        "ref_audio_latents channel/mel dimensions must match audio_latents. "
                        f"Got ref={tuple(ref_audio_latents.shape)} target={tuple(audio_latents.shape)}"
                    )

                ref_audio_latents = ref_audio_latents.to(device=accelerator.device, dtype=network_dtype)

                ref_audio_lengths = batch.get("ref_audio_lengths")
                if isinstance(ref_audio_lengths, dict):
                    ref_audio_lengths = ref_audio_lengths.get("lengths")
                if isinstance(ref_audio_lengths, torch.Tensor):
                    if ref_audio_lengths.dim() == 0:
                        ref_audio_lengths = ref_audio_lengths.view(1)
                    if ref_audio_lengths.numel() == 1 and ref_audio_latents.shape[0] != 1:
                        ref_audio_lengths = ref_audio_lengths.expand(ref_audio_latents.shape[0])
                    if ref_audio_lengths.shape[0] != ref_audio_latents.shape[0]:
                        raise ValueError(
                            "Batch size mismatch: ref_audio_lengths batch="
                            f"{ref_audio_lengths.shape[0]} vs ref_audio_latents batch={ref_audio_latents.shape[0]}"
                        )
                    ref_audio_lengths = ref_audio_lengths.to(device=accelerator.device, dtype=torch.int64)
                    if (ref_audio_lengths <= 0).any():
                        raise ValueError(
                            "ref_audio_lengths contains zeros; missing reference-audio caches in batch. "
                            "Ensure every training sample has cached reference audio."
                        )

                ref_audio_seq_len = int(ref_audio_latents.shape[2])
                tgt_seq_len = int(audio_latents.shape[2])
                noisy_audio = torch.cat([ref_audio_latents, noisy_audio], dim=2)

                audio_timestep_source = audio_timestep_local
                target_audio_timestep = (
                    audio_timestep_source
                    if audio_timestep_source.shape[1] == tgt_seq_len
                    else audio_timestep_source[:, :1].expand(audio_timestep_source.shape[0], tgt_seq_len)
                )
                ref_audio_timestep = torch.zeros(
                    (audio_model_timesteps.shape[0], ref_audio_seq_len),
                    device=accelerator.device,
                    dtype=network_dtype,
                )
                audio_timestep_local = torch.cat([ref_audio_timestep, target_audio_timestep], dim=1)

                zero_ref_target = torch.zeros_like(ref_audio_latents)
                audio_target = torch.cat([zero_ref_target, audio_target], dim=2)

                ref_audio_loss_mask = torch.zeros(
                    (audio_latents.shape[0], ref_audio_seq_len),
                    device=accelerator.device,
                    dtype=torch.bool,
                )
                audio_loss_mask = torch.cat([ref_audio_loss_mask, audio_loss_mask], dim=1)

                resolved_transformer_options = dict(resolved_transformer_options)
                resolved_transformer_options.update(
                    self._build_audio_ref_transformer_overrides(
                        args=args,
                        transformer=transformer,
                        video_latents=video_latents,
                        text_embeds=text_embeds,
                        text_mask=text_mask,
                        audio_model_latents=noisy_audio,
                        ref_audio_seq_len=ref_audio_seq_len,
                        device=accelerator.device,
                        dtype=network_dtype,
                    )
                )

            if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
                self._ensure_fp8_buffers_on_device(transformer)
            elif getattr(args, "nf4_base", False):
                self._ensure_nf4_buffers_on_device(transformer)
            with accelerator.autocast():
                model_pred = transformer(
                    [video_latents, noisy_audio],
                    timestep=model_timesteps,
                    audio_timestep=audio_timestep_local,
                    context=text_embeds,
                    attention_mask=text_mask,
                    frame_rate=frame_rate,
                    transformer_options=resolved_transformer_options,
                    audio_only=True,
                )

            video_pred = model_pred
            audio_pred = None
            if isinstance(model_pred, (list, tuple)):
                if len(model_pred) != 2:
                    raise ValueError(f"Expected audio-only model to return [video_pred, audio_pred], got {len(model_pred)} outputs")
                video_pred, audio_pred = model_pred
            if audio_pred is None:
                raise ValueError("Audio-only mode expected an audio prediction but got None")

            video_target = torch.zeros_like(video_pred)
            out_audio: Dict[str, Any] = {
                "video_pred": video_pred,
                "video_target": video_target,
                "video_loss_weight": 0.0,
            }

            out_audio.update(
                {
                    "audio_pred": audio_pred,
                    "audio_target": audio_target,
                    "audio_loss_mask": audio_loss_mask,
                    "audio_loss_weight": _resolve_loss_weight("audio_loss_weight", "audio_loss_weight"),
                    "audio_sigma": audio_sigma,
                }
            )
            if out_audio["audio_loss_weight"] < 0.0:
                raise ValueError(f"audio_loss_weight must be >= 0. Got: {out_audio['audio_loss_weight']}")

            self._last_dit_inputs = None  # audio-only path — skip preservation
            return out_audio, torch.tensor(0.0, device=accelerator.device)

        first_frame_p = float(getattr(args, "ltx2_first_frame_conditioning_p", 0.0))
        if not (0.0 <= first_frame_p <= 1.0):
            raise ValueError(f"ltx2_first_frame_conditioning_p must be in [0,1]. Got: {first_frame_p}")

        video_conditioning_enabled = None
        # Skip first-frame conditioning for single-frame samples (images)
        # since there are no subsequent frames to generate from frame 0
        num_frames = latents.shape[2]
        if first_frame_p > 0.0 and num_frames > 1:
            enable_conditioning = bool(torch.rand((), device=accelerator.device) < first_frame_p)
            if enable_conditioning:
                video_conditioning_enabled = torch.ones((latents.shape[0],), device=accelerator.device, dtype=torch.bool)

        model_noisy_video = noisy_model_input
        if video_conditioning_enabled is not None and model_noisy_video.shape[2] > 0:
            model_noisy_video = model_noisy_video.clone()
            model_noisy_video[video_conditioning_enabled, :, 0:1, :, :] = latents[video_conditioning_enabled, :, 0:1, :, :]

        if ref_latents is not None:
            from musubi_tuner.ltx_2.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
            from musubi_tuner.ltx_2.guidance.perturbations import BatchedPerturbationConfig
            from musubi_tuner.ltx_2.model.transformer.modality import Modality
            from musubi_tuner.ltx_2.types import SpatioTemporalScaleFactors, VideoLatentShape

            patchifier = VideoLatentPatchifier(patch_size=1)

            ref_latents = ref_latents.to(device=accelerator.device, dtype=network_dtype)
            ref_tokens = patchifier.patchify(ref_latents)
            target_tokens = patchifier.patchify(model_noisy_video)
            combined_tokens = torch.cat([ref_tokens, target_tokens], dim=1)

            bsz = combined_tokens.shape[0]
            ref_seq_len = ref_tokens.shape[1]
            target_seq_len = target_tokens.shape[1]

            ref_height = int(ref_latents.shape[3])
            ref_width = int(ref_latents.shape[4])
            tgt_height = int(latents.shape[3])
            tgt_width = int(latents.shape[4])

            ref_conditioning_mask = torch.ones((bsz, ref_seq_len), device=accelerator.device, dtype=torch.bool)

            target_conditioning_mask = torch.zeros((bsz, target_seq_len), device=accelerator.device, dtype=torch.bool)
            if video_conditioning_enabled is not None:
                first_frame_tokens = tgt_height * tgt_width
                if first_frame_tokens > 0:
                    target_conditioning_mask[video_conditioning_enabled, :first_frame_tokens] = True
            conditioning_mask = torch.cat([ref_conditioning_mask, target_conditioning_mask], dim=1)

            combined_timesteps = sigma.view(bsz, 1).expand(bsz, ref_seq_len + target_seq_len)
            combined_timesteps = torch.where(conditioning_mask, torch.zeros_like(combined_timesteps), combined_timesteps)

            frame_rate_v2v = frame_rate
            if frame_rate_v2v is None:
                frame_rate_v2v = 25

            ref_frames = int(ref_latents.shape[2])
            tgt_frames = int(latents.shape[2])

            ref_coords = patchifier.get_patch_grid_bounds(
                output_shape=VideoLatentShape(
                    batch=bsz,
                    channels=int(ref_latents.shape[1]),
                    frames=ref_frames,
                    height=ref_height,
                    width=ref_width,
                ),
                device=accelerator.device,
            )
            ref_positions = get_pixel_coords(
                latent_coords=ref_coords,
                scale_factors=SpatioTemporalScaleFactors.default(),
                causal_fix=True,
            ).to(dtype=network_dtype)
            ref_positions[:, 0, ...] = ref_positions[:, 0, ...] / float(frame_rate_v2v)
            if reference_downscale_factor != 1:
                ref_positions = ref_positions.clone()
                ref_positions[:, 1, ...] *= reference_downscale_factor
                ref_positions[:, 2, ...] *= reference_downscale_factor

            tgt_coords = patchifier.get_patch_grid_bounds(
                output_shape=VideoLatentShape(
                    batch=bsz,
                    channels=int(latents.shape[1]),
                    frames=tgt_frames,
                    height=tgt_height,
                    width=tgt_width,
                ),
                device=accelerator.device,
            )
            tgt_positions = get_pixel_coords(
                latent_coords=tgt_coords,
                scale_factors=SpatioTemporalScaleFactors.default(),
                causal_fix=True,
            ).to(dtype=network_dtype)
            tgt_positions[:, 0, ...] = tgt_positions[:, 0, ...] / float(frame_rate_v2v)

            combined_positions = torch.cat([ref_positions, tgt_positions], dim=2)

            video_modality = Modality(
                enabled=True,
                latent=combined_tokens,
                timesteps=combined_timesteps,
                positions=combined_positions,
                context=text_embeds,
                sigma=sigma,
                context_mask=text_mask,
            )

            perturbations = BatchedPerturbationConfig.empty(bsz)
            unwrapped_transformer = accelerator.unwrap_model(transformer)

            if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
                self._ensure_fp8_buffers_on_device(unwrapped_transformer)
            elif getattr(args, "nf4_base", False):
                self._ensure_nf4_buffers_on_device(unwrapped_transformer)
            with accelerator.autocast():
                if hasattr(unwrapped_transformer, "forward_modalities"):
                    pred_tokens, _ = unwrapped_transformer.forward_modalities(video_modality, None, perturbations)
                else:
                    base_model = (
                        unwrapped_transformer.model if hasattr(unwrapped_transformer, "model") else unwrapped_transformer
                    )
                    pred_tokens, _ = base_model(video_modality, None, perturbations)

            target_pred_tokens = pred_tokens[:, ref_seq_len:, :]
            target_velocity = patchifier.patchify(noise - latents)
            target_loss_mask = ~target_conditioning_mask
            target_loss_mask = _combine_loss_masks(target_loss_mask, _cached_video_loss_mask(as_tokens=True))

            out_v2v: Dict[str, Any] = {
                "video_pred": target_pred_tokens,
                "video_target": target_velocity,
                "video_loss_mask": target_loss_mask,
                "video_loss_weight": _resolve_loss_weight("video_loss_weight", "video_loss_weight"),
            }
            if out_v2v["video_loss_weight"] < 0.0:
                raise ValueError(f"video_loss_weight must be >= 0. Got: {out_v2v['video_loss_weight']}")

            self._last_dit_inputs = None  # reference-latent path — skip preservation
            return out_v2v, torch.tensor(0.0, device=accelerator.device)

        # ---- av_ic: combined video + audio IC-LoRA ----
        if ic_lora_strategy == "av_ic" and ref_latents is not None:
            from musubi_tuner.ltx_2.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
            from musubi_tuner.ltx_2.guidance.perturbations import BatchedPerturbationConfig
            from musubi_tuner.ltx_2.model.transformer.modality import Modality
            from musubi_tuner.ltx_2.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape
            from musubi_tuner.networks.lora_ltx2 import _split_av_context

            unwrapped_transformer = accelerator.unwrap_model(transformer)
            base_model = unwrapped_transformer.model if hasattr(unwrapped_transformer, "model") else unwrapped_transformer

            # --- Audio latents: retrieve, validate, noise ---
            av_ic_audio_latents = batch.get("audio_latents")
            if isinstance(av_ic_audio_latents, dict):
                av_ic_audio_latents = av_ic_audio_latents.get("latents")
            if av_ic_audio_latents is None:
                raise ValueError(f"--ic_lora_strategy {ic_lora_strategy} requires audio_latents in every AV batch")
            if not isinstance(av_ic_audio_latents, torch.Tensor):
                raise TypeError(f"Expected audio_latents to be a torch.Tensor, got: {type(av_ic_audio_latents)}")
            if av_ic_audio_latents.dim() != 4:
                raise ValueError(f"Expected audio_latents to be 4D [B, C, T, F], got shape: {tuple(av_ic_audio_latents.shape)}")
            av_ic_audio_latents = av_ic_audio_latents.to(device=accelerator.device, dtype=network_dtype)
            if getattr(args, "align_audio_latents_train", False):
                expected_length = self._calculate_expected_audio_latent_length(
                    args, transformer, latent_frames=int(latents.shape[2]), frame_rate=float(frame_rate),
                )
                av_ic_audio_latents = self._adjust_audio_latent_duration(av_ic_audio_latents, expected_length)

            av_ic_audio_noise = torch.randn_like(av_ic_audio_latents)
            av_ic_sigma_audio = audio_sigma.view(-1, 1, 1, 1)
            av_ic_noisy_audio = (1.0 - av_ic_sigma_audio) * av_ic_audio_latents + av_ic_sigma_audio * av_ic_audio_noise
            av_ic_audio_target_raw = av_ic_audio_noise - av_ic_audio_latents  # velocity target

            # --- Audio reference latents: retrieve & validate ---
            av_ic_ref_audio = _merge_reference_tensors(
                _collect_reference_tensors(batch, "ref_audio_latents", expected_ndim=4),
                concat_dim=2,
            )
            if av_ic_ref_audio is None:
                raise ValueError(
                    f"--ic_lora_strategy {ic_lora_strategy} requires ref_audio_latents. "
                    "Set reference_audio_directory/reference_audio_cache_directory and cache reference audio latents."
                )
            if not isinstance(av_ic_ref_audio, torch.Tensor):
                raise TypeError(f"Expected ref_audio_latents to be a torch.Tensor, got: {type(av_ic_ref_audio)}")
            if av_ic_ref_audio.dim() != 4:
                raise ValueError(f"Expected ref_audio_latents 4D [B, C, T, F], got shape: {tuple(av_ic_ref_audio.shape)}")
            if av_ic_ref_audio.shape[1] != av_ic_audio_latents.shape[1] or av_ic_ref_audio.shape[3] != av_ic_audio_latents.shape[3]:
                raise ValueError(
                    "ref_audio_latents channel/mel dimensions must match audio_latents. "
                    f"Got ref={tuple(av_ic_ref_audio.shape)} target={tuple(av_ic_audio_latents.shape)}"
                )
            av_ic_ref_audio = av_ic_ref_audio.to(device=accelerator.device, dtype=network_dtype)

            bsz = latents.shape[0]

            # ---- VIDEO SIDE: patchify ref + target, build positions & timesteps ----
            video_patchifier = VideoLatentPatchifier(patch_size=1)
            ref_latents = ref_latents.to(device=accelerator.device, dtype=network_dtype)
            ref_video_tokens = video_patchifier.patchify(ref_latents)
            target_video_tokens = video_patchifier.patchify(model_noisy_video)
            video_combined_tokens = torch.cat([ref_video_tokens, target_video_tokens], dim=1)

            ref_video_seq_len = ref_video_tokens.shape[1]
            tgt_video_seq_len = target_video_tokens.shape[1]

            # Conditioning mask: ref=True (timestep→0), target=False (timestep→sigma)
            ref_video_cond_mask = torch.ones((bsz, ref_video_seq_len), device=accelerator.device, dtype=torch.bool)
            tgt_video_cond_mask = torch.zeros((bsz, tgt_video_seq_len), device=accelerator.device, dtype=torch.bool)
            if video_conditioning_enabled is not None:
                first_frame_tokens = int(latents.shape[3]) * int(latents.shape[4])
                if first_frame_tokens > 0:
                    tgt_video_cond_mask[video_conditioning_enabled, :first_frame_tokens] = True
            video_cond_mask = torch.cat([ref_video_cond_mask, tgt_video_cond_mask], dim=1)

            video_combined_ts = sigma.view(bsz, 1).expand(bsz, ref_video_seq_len + tgt_video_seq_len)
            video_combined_ts = torch.where(video_cond_mask, torch.zeros_like(video_combined_ts), video_combined_ts)

            # Video positions
            av_ic_frame_rate = frame_rate if frame_rate is not None else 25
            ref_frames = int(ref_latents.shape[2])
            tgt_frames = int(latents.shape[2])
            tgt_h, tgt_w = int(latents.shape[3]), int(latents.shape[4])

            ref_coords = video_patchifier.get_patch_grid_bounds(
                output_shape=VideoLatentShape(
                    batch=bsz, channels=int(ref_latents.shape[1]),
                    frames=ref_frames, height=ref_h, width=ref_w,
                ),
                device=accelerator.device,
            )
            ref_video_pos = get_pixel_coords(
                latent_coords=ref_coords, scale_factors=SpatioTemporalScaleFactors.default(), causal_fix=True,
            ).to(dtype=network_dtype)
            ref_video_pos[:, 0, ...] = ref_video_pos[:, 0, ...] / float(av_ic_frame_rate)
            if reference_downscale_factor != 1:
                ref_video_pos = ref_video_pos.clone()
                ref_video_pos[:, 1, ...] *= reference_downscale_factor
                ref_video_pos[:, 2, ...] *= reference_downscale_factor

            tgt_coords = video_patchifier.get_patch_grid_bounds(
                output_shape=VideoLatentShape(
                    batch=bsz, channels=int(latents.shape[1]),
                    frames=tgt_frames, height=tgt_h, width=tgt_w,
                ),
                device=accelerator.device,
            )
            tgt_video_pos = get_pixel_coords(
                latent_coords=tgt_coords, scale_factors=SpatioTemporalScaleFactors.default(), causal_fix=True,
            ).to(dtype=network_dtype)
            tgt_video_pos[:, 0, ...] = tgt_video_pos[:, 0, ...] / float(av_ic_frame_rate)
            video_combined_pos = torch.cat([ref_video_pos, tgt_video_pos], dim=2)

            # ---- AUDIO SIDE: patchify ref + target, build positions & timesteps ----
            audio_patchifier = getattr(unwrapped_transformer, "_audio_patchifier", None)
            if audio_patchifier is None and hasattr(unwrapped_transformer, "module"):
                audio_patchifier = getattr(unwrapped_transformer.module, "_audio_patchifier", None)
            if audio_patchifier is None:
                raise ValueError(f"{ic_lora_strategy} requires an audio patchifier on the model (LTXAV model expected)")

            ref_audio_tokens = audio_patchifier.patchify(av_ic_ref_audio)        # [B, ref_T, C*F]
            tgt_audio_tokens = audio_patchifier.patchify(av_ic_noisy_audio)      # [B, tgt_T, C*F]
            audio_combined_tokens = torch.cat([ref_audio_tokens, tgt_audio_tokens], dim=1)

            ref_audio_seq_len = ref_audio_tokens.shape[1]
            tgt_audio_seq_len = tgt_audio_tokens.shape[1]

            # Audio timesteps: ref=0, target=audio_sigma
            tgt_audio_ts = (
                audio_model_timesteps
                if audio_model_timesteps.shape[1] == tgt_audio_seq_len
                else audio_model_timesteps[:, :1].expand(bsz, tgt_audio_seq_len)
            )
            ref_audio_ts = torch.zeros((bsz, ref_audio_seq_len), device=accelerator.device, dtype=network_dtype)
            audio_combined_ts = torch.cat([ref_audio_ts, tgt_audio_ts], dim=1)

            # Audio positions (separate for ref and target, same as _build_audio_ref_transformer_overrides)
            channels_audio = int(av_ic_ref_audio.shape[1])
            mel_bins = int(av_ic_ref_audio.shape[3])
            use_negative_positions = bool(getattr(args, "audio_ref_use_negative_positions", False))

            ref_audio_shape = AudioLatentShape(batch=bsz, channels=channels_audio, frames=ref_audio_seq_len, mel_bins=mel_bins)
            ref_audio_pos = audio_patchifier.get_patch_grid_bounds(ref_audio_shape, device=accelerator.device).to(dtype=network_dtype)
            if use_negative_positions:
                _hop = getattr(audio_patchifier, "hop_length", 160)
                _ds = getattr(audio_patchifier, "audio_latent_downsample_factor", 4)
                _sr = getattr(audio_patchifier, "sample_rate", 16000)
                time_per_latent = float(_hop) * float(_ds) / float(_sr)
                ref_duration = ref_audio_pos[:, :, -1:, 1:2]
                ref_audio_pos = ref_audio_pos - ref_duration - time_per_latent

            tgt_audio_shape = AudioLatentShape(batch=bsz, channels=channels_audio, frames=tgt_audio_seq_len, mel_bins=mel_bins)
            tgt_audio_pos = audio_patchifier.get_patch_grid_bounds(tgt_audio_shape, device=accelerator.device).to(dtype=network_dtype)
            audio_combined_pos = torch.cat([ref_audio_pos, tgt_audio_pos], dim=2)

            # ---- CONTEXT: split for video and audio ----
            video_ctx, audio_ctx = _split_av_context(base_model, text_embeds)

            # ---- OPTIONAL CROSS-MODAL MASKS ----
            mask_dtype = network_dtype if network_dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16) else torch.float32
            neg_inf = torch.finfo(mask_dtype).min

            av_cross_attention_mode = _normalize_av_cross_attention_mode(
                getattr(args, "av_cross_attention_mode", "both")
            )
            av_ic_a2v_enabled = av_cross_attention_mode in {"both", "a2v_only"}
            av_ic_v2a_enabled = av_cross_attention_mode in {"both", "v2a_only"}
            total_video_seq = ref_video_seq_len + tgt_video_seq_len
            total_audio_seq = ref_audio_seq_len + tgt_audio_seq_len

            a2v_mask = None
            if not av_ic_a2v_enabled:
                a2v_mask = torch.full((bsz, total_video_seq, total_audio_seq), neg_inf, device=accelerator.device, dtype=mask_dtype)
            elif bool(getattr(args, "audio_ref_mask_cross_attention_to_reference", False)):
                a2v_mask = torch.zeros((bsz, total_video_seq, total_audio_seq), device=accelerator.device, dtype=mask_dtype)
                a2v_mask[:, :, :ref_audio_seq_len] = neg_inf  # block video from attending to ref audio

            v2a_mask = None
            if not av_ic_v2a_enabled:
                v2a_mask = torch.full((bsz, total_audio_seq, total_video_seq), neg_inf, device=accelerator.device, dtype=mask_dtype)

            # ---- BUILD MODALITY OBJECTS & FORWARD ----
            video_modality = Modality(
                enabled=True,
                latent=video_combined_tokens,
                timesteps=video_combined_ts,
                positions=video_combined_pos,
                context=video_ctx,
                sigma=sigma,
                context_mask=text_mask,
                a2v_cross_attention_mask=a2v_mask,
            )

            audio_modality = Modality(
                enabled=True,
                latent=audio_combined_tokens,
                timesteps=audio_combined_ts,
                positions=audio_combined_pos,
                context=audio_ctx,
                sigma=audio_sigma,
                context_mask=text_mask,
            )

            perturbations = BatchedPerturbationConfig.empty(bsz)

            if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
                self._ensure_fp8_buffers_on_device(unwrapped_transformer)
            elif getattr(args, "nf4_base", False):
                self._ensure_nf4_buffers_on_device(unwrapped_transformer)
            with accelerator.autocast():
                if hasattr(unwrapped_transformer, "forward_modalities"):
                    video_pred_all, audio_pred_all = unwrapped_transformer.forward_modalities(
                        video_modality, audio_modality, perturbations,
                    )
                else:
                    video_pred_all, audio_pred_all = base_model(video_modality, audio_modality, perturbations)

            # ---- EXTRACT TARGET PREDICTIONS & COMPUTE LOSS TARGETS ----
            target_video_pred = video_pred_all[:, ref_video_seq_len:, :]
            target_audio_pred = audio_pred_all[:, ref_audio_seq_len:, :]

            video_velocity = video_patchifier.patchify(noise - latents)
            audio_velocity = audio_patchifier.patchify(av_ic_audio_target_raw)

            video_loss_mask = _combine_loss_masks(
                ~tgt_video_cond_mask,
                _cached_video_loss_mask(as_tokens=True),
            )
            audio_loss_mask = _compose_target_audio_loss_mask(
                None,
                _cached_audio_loss_mask(tgt_audio_seq_len, bsz),
                batch_size=bsz,
                target_seq_len=tgt_audio_seq_len,
                device=accelerator.device,
            )

            out_av_ic: Dict[str, Any] = {
                "video_pred": target_video_pred,
                "video_target": video_velocity,
                "video_loss_mask": video_loss_mask,
                "video_loss_weight": _resolve_loss_weight("video_loss_weight", "video_loss_weight"),
                "audio_pred": target_audio_pred,
                "audio_target": audio_velocity,
                "audio_loss_mask": audio_loss_mask,
                "audio_loss_weight": _resolve_loss_weight("audio_loss_weight", "audio_loss_weight"),
                "audio_sigma": audio_sigma,
            }
            self._last_dit_inputs = None  # av_ic path — skip preservation
            return out_av_ic, torch.tensor(0.0, device=accelerator.device)

        audio_latents = None
        audio_noise = None
        noisy_audio = None
        audio_enabled_for_batch = False
        audio_regularizer_active = False
        audio_expected_for_batch = self._ltx_mode == "av"
        audio_loss_mask = None
        audio_target = None
        audio_timestep_for_model = None
        teacher_noisy_audio_for_self_flow = None
        teacher_audio_timestep_for_self_flow = None
        ref_audio_latents = None
        ref_audio_seq_len = 0
        if self._ltx_mode == "av":
            audio_latents = batch.get("audio_latents")
            if isinstance(audio_latents, dict):
                audio_latents = audio_latents.get("latents")

            if audio_latents is None:
                if bool(getattr(args, "audio_silence_regularizer", False)):
                    audio_latents = self._build_empty_audio_latents(
                        args=args,
                        transformer=transformer,
                        latents=latents,
                        frame_rate=float(frame_rate),
                        device=accelerator.device,
                        dtype=network_dtype,
                    )
                    audio_regularizer_active = True
                    if not self._warned_missing_audio:
                        logger.warning(
                            "LTXAV mode: missing audio latents in this batch; using silence regularizer fallback."
                        )
                        self._warned_missing_audio = True
                else:
                    if not self._warned_missing_audio:
                        logger.warning(
                            "LTXAV mode: missing audio latents in this batch; skipping audio branch. "
                            "Provide cached audio latents to train audio generation."
                        )
                        self._warned_missing_audio = True
            if audio_latents is not None:
                if not isinstance(audio_latents, torch.Tensor):
                    raise TypeError(f"Expected audio_latents to be a torch.Tensor, got: {type(audio_latents)}")
                if audio_latents.dim() != 4:
                    raise ValueError(f"Expected audio_latents to be 4D [B, C, T, F], got shape: {tuple(audio_latents.shape)}")
                if audio_latents.shape[0] != latents.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: latents batch={latents.shape[0]} vs audio_latents batch={audio_latents.shape[0]}"
                    )
                audio_latents = audio_latents.to(device=accelerator.device, dtype=network_dtype)
                _check_finite("audio_latents", audio_latents)
                _log_stats("audio_latents", audio_latents)
                if getattr(args, "align_audio_latents_train", False):
                    expected_length = self._calculate_expected_audio_latent_length(
                        args,
                        transformer,
                        latent_frames=int(latents.shape[2]),
                        frame_rate=float(frame_rate),
                    )
                    audio_latents = self._adjust_audio_latent_duration(audio_latents, expected_length)
                audio_loss_mask = torch.ones(
                    (audio_latents.shape[0], audio_latents.shape[2]),
                    device=accelerator.device,
                    dtype=torch.bool,
                )

                audio_enabled_for_batch = True
                audio_noise = torch.randn_like(audio_latents)
                sigma_audio = audio_sigma.view(-1, 1, 1, 1)
                noisy_audio = (1.0 - sigma_audio) * audio_latents + sigma_audio * audio_noise
                sf_ctx = self._self_flow_step_context if self._self_flow_active else None
                if (
                    isinstance(sf_ctx, dict)
                    and bool(getattr(args, "self_flow", False))
                    and bool(getattr(getattr(self._self_flow, "config", None), "dual_timestep", True))
                ):
                    base_audio_sigmas = sf_ctx.get("base_sigmas")
                    alt_audio_sigmas = sf_ctx.get("alt_sigmas")
                    if bool(getattr(args, "independent_audio_timestep", False)):
                        base_audio_sigmas = audio_model_timesteps[:, :1].to(device=accelerator.device, dtype=network_dtype)
                        alt_audio_sigmas = self._sample_independent_audio_timesteps(
                            args,
                            batch_size=audio_model_timesteps.shape[0],
                            device=accelerator.device,
                            dtype=network_dtype,
                        )
                    if isinstance(base_audio_sigmas, torch.Tensor) and isinstance(alt_audio_sigmas, torch.Tensor):
                        audio_sf = prepare_self_flow_audio_view(
                            audio_latents=audio_latents,
                            audio_noise=audio_noise,
                            base_audio_sigmas=base_audio_sigmas,
                            alt_audio_sigmas=alt_audio_sigmas,
                            mask_ratio=float(getattr(self._self_flow.config, "mask_ratio", 0.10)),
                            device=accelerator.device,
                            dtype=network_dtype,
                        )
                        noisy_audio = audio_sf["student_noisy_audio"]
                        audio_timestep_for_model = audio_sf["student_audio_timesteps"]
                        teacher_noisy_audio_for_self_flow = audio_sf["teacher_noisy_audio"].detach()
                        teacher_audio_timestep_for_self_flow = audio_sf["teacher_audio_timesteps"].detach()
                        sf_ctx["audio_self_flow_mask"] = audio_sf["audio_mask"].detach()
                        sf_ctx["audio_masked_token_ratio"] = float(audio_sf["audio_masked_token_ratio"].detach().item())
                        sf_ctx["audio_tau_mean"] = float(audio_sf["audio_tau_mean"].detach().item())
                        sf_ctx["audio_tau_min_mean"] = float(audio_sf["audio_tau_min_mean"].detach().item())
                _check_finite("noisy_audio", noisy_audio)
                _log_stats("noisy_audio", noisy_audio)
                audio_target = audio_noise - audio_latents
                if audio_timestep_for_model is None:
                    audio_timestep_for_model = audio_model_timesteps

            if audio_ref_only_ic_enabled:
                ref_audio_latents = batch.get("ref_audio_latents")
                if isinstance(ref_audio_latents, dict):
                    ref_audio_latents = ref_audio_latents.get("latents")

                if not audio_enabled_for_batch or audio_latents is None or noisy_audio is None:
                    raise ValueError(
                        "--ic_lora_strategy audio_ref_only_ic requires target audio_latents in every AV batch"
                    )
                if ref_audio_latents is None:
                    raise ValueError(
                        "--ic_lora_strategy audio_ref_only_ic requires ref_audio_latents. "
                        "Set reference_audio_directory/reference_audio_cache_directory and cache reference audio latents."
                    )
                if not isinstance(ref_audio_latents, torch.Tensor):
                    raise TypeError(f"Expected ref_audio_latents to be a torch.Tensor, got: {type(ref_audio_latents)}")
                if ref_audio_latents.dim() != 4:
                    raise ValueError(
                        f"Expected ref_audio_latents to be 4D [B, C, T, F], got shape: {tuple(ref_audio_latents.shape)}"
                    )
                if ref_audio_latents.shape[0] != latents.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch: latents batch={latents.shape[0]} vs ref_audio_latents batch={ref_audio_latents.shape[0]}"
                    )
                if ref_audio_latents.shape[1] != audio_latents.shape[1] or ref_audio_latents.shape[3] != audio_latents.shape[3]:
                    raise ValueError(
                        "ref_audio_latents channel/mel dimensions must match audio_latents. "
                        f"Got ref={tuple(ref_audio_latents.shape)} target={tuple(audio_latents.shape)}"
                    )

                ref_audio_latents = ref_audio_latents.to(device=accelerator.device, dtype=network_dtype)
                _check_finite("ref_audio_latents", ref_audio_latents)
                _log_stats("ref_audio_latents", ref_audio_latents)

                ref_audio_lengths = batch.get("ref_audio_lengths")
                if isinstance(ref_audio_lengths, dict):
                    ref_audio_lengths = ref_audio_lengths.get("lengths")
                if isinstance(ref_audio_lengths, torch.Tensor):
                    if ref_audio_lengths.dim() == 0:
                        ref_audio_lengths = ref_audio_lengths.view(1)
                    if ref_audio_lengths.numel() == 1 and ref_audio_latents.shape[0] != 1:
                        ref_audio_lengths = ref_audio_lengths.expand(ref_audio_latents.shape[0])
                    if ref_audio_lengths.shape[0] != ref_audio_latents.shape[0]:
                        raise ValueError(
                            "Batch size mismatch: ref_audio_lengths batch="
                            f"{ref_audio_lengths.shape[0]} vs ref_audio_latents batch={ref_audio_latents.shape[0]}"
                        )
                    ref_audio_lengths = ref_audio_lengths.to(device=accelerator.device, dtype=torch.int64)
                    if (ref_audio_lengths <= 0).any():
                        raise ValueError(
                            "ref_audio_lengths contains zeros; missing reference-audio caches in batch. "
                            "Ensure every training sample has cached reference audio."
                        )

                ref_audio_seq_len = int(ref_audio_latents.shape[2])
                tgt_seq_len = int(audio_latents.shape[2])
                noisy_audio = torch.cat([ref_audio_latents, noisy_audio], dim=2)
                _check_finite("noisy_audio_with_reference", noisy_audio)

                audio_timestep_source = audio_timestep_for_model if isinstance(audio_timestep_for_model, torch.Tensor) else audio_model_timesteps
                target_audio_timestep = (
                    audio_timestep_source
                    if audio_timestep_source.shape[1] == tgt_seq_len
                    else audio_timestep_source[:, :1].expand(audio_timestep_source.shape[0], tgt_seq_len)
                )
                ref_audio_timestep = torch.zeros(
                    (audio_model_timesteps.shape[0], ref_audio_seq_len),
                    device=accelerator.device,
                    dtype=network_dtype,
                )
                audio_timestep_for_model = torch.cat([ref_audio_timestep, target_audio_timestep], dim=1)
                if (
                    isinstance(teacher_noisy_audio_for_self_flow, torch.Tensor)
                    and isinstance(teacher_audio_timestep_for_self_flow, torch.Tensor)
                ):
                    teacher_noisy_audio_for_self_flow = torch.cat(
                        [ref_audio_latents, teacher_noisy_audio_for_self_flow],
                        dim=2,
                    )
                    teacher_target_audio_timestep = (
                        teacher_audio_timestep_for_self_flow
                        if teacher_audio_timestep_for_self_flow.shape[1] == tgt_seq_len
                        else teacher_audio_timestep_for_self_flow[:, :1].expand(
                            teacher_audio_timestep_for_self_flow.shape[0], tgt_seq_len
                        )
                    )
                    teacher_audio_timestep_for_self_flow = torch.cat(
                        [ref_audio_timestep, teacher_target_audio_timestep],
                        dim=1,
                    )

                if audio_target is None:
                    raise ValueError("Internal error: audio_target must be initialized before audio_ref_only_ic composition")
                zero_ref_target = torch.zeros_like(ref_audio_latents)
                audio_target = torch.cat([zero_ref_target, audio_target], dim=2)

                audio_loss_mask = _compose_audio_ref_only_ic_loss_mask(
                    audio_loss_mask,
                    _cached_audio_loss_mask(tgt_seq_len, int(audio_latents.shape[0])),
                    batch_size=int(audio_latents.shape[0]),
                    target_seq_len=tgt_seq_len,
                    ref_seq_len=ref_audio_seq_len,
                    device=accelerator.device,
                    audio_lengths=batch.get("audio_lengths") if getattr(args, "use_audio_length_mask", False) else None,
                )

        if self._ltx_mode == "av" and not audio_enabled_for_batch:
            text_embeds = select_video_text_embeds_for_av_no_audio(
                text_embeds,
                conditions,
                expected_video_dim=expected_video_dim,
                expected_audio_dim=expected_audio_dim,
            )

        if bool(getattr(transformer, "training", False)) and self._ltx_mode == "av":
            supervision_alert = update_and_check_audio_supervision(
                self._audio_supervision_state,
                mode=str(getattr(args, "audio_supervision_mode", "off")),
                warmup_steps=int(getattr(args, "audio_supervision_warmup_steps", 50)),
                check_interval=int(getattr(args, "audio_supervision_check_interval", 50)),
                min_ratio=float(getattr(args, "audio_supervision_min_ratio", 0.9)),
                audio_expected_for_batch=audio_expected_for_batch,
                audio_supervised_for_batch=audio_enabled_for_batch and not audio_regularizer_active,
            )
            if supervision_alert is not None:
                message = format_audio_supervision_alert(supervision_alert)
                if str(getattr(args, "audio_supervision_mode", "off")) == "error":
                    raise ValueError(message)
                logger.warning("%s Running in warning mode; training will continue.", message)

        if skip_nonfinite and nonfinite_flag["hit"]:
            return {"_skip_step": True, "skip_reason": nonfinite_flag["tag"]}, torch.tensor(
                0.0, device=accelerator.device
            )

        caption_channels = getattr(transformer, "caption_channels", None)
        if caption_channels is None:
            base_model = transformer.model if hasattr(transformer, "model") else transformer
            _caption_proj = getattr(base_model, "caption_projection", None)
            if _caption_proj is not None:
                caption_channels = getattr(getattr(_caption_proj, "linear_1", None), "in_features", None)
        if caption_channels is not None:
            expected_last_dim = int(caption_channels) * (2 if audio_enabled_for_batch else 1)
            if text_embeds.shape[-1] != expected_last_dim:
                if (
                    self._ltx_mode == "av"
                    and audio_enabled_for_batch
                    and audio_regularizer_active
                    and text_embeds.shape[-1] * 2 == expected_last_dim
                ):
                    text_embeds = torch.cat([text_embeds, torch.zeros_like(text_embeds)], dim=-1)
                    expected_last_dim = text_embeds.shape[-1]
                else:
                    raise ValueError(
                        f"Text embedding dim mismatch for {'LTXAV' if self._audio_video else 'LTXV'}: "
                        f"got {text_embeds.shape[-1]}, expected {expected_last_dim}. "
                        f"(caption_channels={caption_channels})"
                    )
            if text_embeds.shape[-1] != expected_last_dim:
                raise ValueError(
                    f"Text embedding dim mismatch for {'LTXAV' if self._audio_video else 'LTXV'}: "
                    f"got {text_embeds.shape[-1]}, expected {expected_last_dim}. "
                    f"(caption_channels={caption_channels})"
                )

        if self._ltx_mode == "av" and bool(getattr(base_model, "caption_proj_before_connector", False)):
            expected_ctx_dim = expected_video_dim + expected_audio_dim if audio_enabled_for_batch else expected_video_dim
            if expected_ctx_dim > 0 and int(text_embeds.shape[-1]) != expected_ctx_dim:
                mode_name = "AV (video+audio)" if audio_enabled_for_batch else "AV-no-audio (video-only)"
                raise ValueError(
                    f"{mode_name} received text embeddings with incompatible hidden size for this checkpoint. "
                    f"Expected dim={expected_ctx_dim}, got dim={text_embeds.shape[-1]}. "
                    "Ensure caches contain modality-specific embeddings generated with the same --ltx2_checkpoint."
                )

        if self._ltx_mode == "video" and bool(getattr(base_model, "caption_proj_before_connector", False)):
            if expected_video_dim > 0 and int(text_embeds.shape[-1]) != expected_video_dim:
                raise ValueError(
                    f"Video mode received text embeddings with incompatible hidden size for this checkpoint. "
                    f"Expected dim={expected_video_dim}, got dim={text_embeds.shape[-1]}. "
                    "Ensure text encoder caches were generated with the same --ltx2_checkpoint."
                )

        if ic_lora_strategy == "video_ref_only_av" and ref_latents is not None:
            from musubi_tuner.ltx_2.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
            from musubi_tuner.ltx_2.guidance.perturbations import BatchedPerturbationConfig
            from musubi_tuner.ltx_2.model.transformer.modality import Modality
            from musubi_tuner.ltx_2.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape
            from musubi_tuner.networks.lora_ltx2 import _split_av_context

            if not audio_enabled_for_batch or audio_latents is None or noisy_audio is None or audio_target is None:
                raise ValueError("--ic_lora_strategy video_ref_only_av requires target audio_latents in every AV batch")

            unwrapped_transformer = accelerator.unwrap_model(transformer)
            base_model = unwrapped_transformer.model if hasattr(unwrapped_transformer, "model") else unwrapped_transformer

            bsz = latents.shape[0]
            video_patchifier = VideoLatentPatchifier(patch_size=1)
            ref_latents = ref_latents.to(device=accelerator.device, dtype=network_dtype)
            ref_video_tokens = video_patchifier.patchify(ref_latents)
            target_video_tokens = video_patchifier.patchify(model_noisy_video)
            video_combined_tokens = torch.cat([ref_video_tokens, target_video_tokens], dim=1)

            ref_video_seq_len = ref_video_tokens.shape[1]
            tgt_video_seq_len = target_video_tokens.shape[1]

            ref_video_cond_mask = torch.ones((bsz, ref_video_seq_len), device=accelerator.device, dtype=torch.bool)
            tgt_video_cond_mask = torch.zeros((bsz, tgt_video_seq_len), device=accelerator.device, dtype=torch.bool)
            if video_conditioning_enabled is not None:
                first_frame_tokens = int(latents.shape[3]) * int(latents.shape[4])
                if first_frame_tokens > 0:
                    tgt_video_cond_mask[video_conditioning_enabled, :first_frame_tokens] = True
            video_cond_mask = torch.cat([ref_video_cond_mask, tgt_video_cond_mask], dim=1)

            video_combined_ts = sigma.view(bsz, 1).expand(bsz, ref_video_seq_len + tgt_video_seq_len)
            video_combined_ts = torch.where(video_cond_mask, torch.zeros_like(video_combined_ts), video_combined_ts)

            ref_h, ref_w = int(ref_latents.shape[3]), int(ref_latents.shape[4])
            tgt_h, tgt_w = int(latents.shape[3]), int(latents.shape[4])
            frame_rate_vref = frame_rate if frame_rate is not None else 25
            ref_frames = int(ref_latents.shape[2])
            tgt_frames = int(latents.shape[2])

            ref_coords = video_patchifier.get_patch_grid_bounds(
                output_shape=VideoLatentShape(
                    batch=bsz, channels=int(ref_latents.shape[1]),
                    frames=ref_frames, height=ref_h, width=ref_w,
                ),
                device=accelerator.device,
            )
            ref_video_pos = get_pixel_coords(
                latent_coords=ref_coords, scale_factors=SpatioTemporalScaleFactors.default(), causal_fix=True,
            ).to(dtype=network_dtype)
            ref_video_pos[:, 0, ...] = ref_video_pos[:, 0, ...] / float(frame_rate_vref)
            if reference_downscale_factor != 1:
                ref_video_pos = ref_video_pos.clone()
                ref_video_pos[:, 1, ...] *= reference_downscale_factor
                ref_video_pos[:, 2, ...] *= reference_downscale_factor

            tgt_coords = video_patchifier.get_patch_grid_bounds(
                output_shape=VideoLatentShape(
                    batch=bsz, channels=int(latents.shape[1]),
                    frames=tgt_frames, height=tgt_h, width=tgt_w,
                ),
                device=accelerator.device,
            )
            tgt_video_pos = get_pixel_coords(
                latent_coords=tgt_coords, scale_factors=SpatioTemporalScaleFactors.default(), causal_fix=True,
            ).to(dtype=network_dtype)
            tgt_video_pos[:, 0, ...] = tgt_video_pos[:, 0, ...] / float(frame_rate_vref)
            video_combined_pos = torch.cat([ref_video_pos, tgt_video_pos], dim=2)

            audio_patchifier = getattr(unwrapped_transformer, "_audio_patchifier", None)
            if audio_patchifier is None and hasattr(unwrapped_transformer, "module"):
                audio_patchifier = getattr(unwrapped_transformer.module, "_audio_patchifier", None)
            if audio_patchifier is None:
                raise ValueError("video_ref_only_av requires an audio patchifier on the model (LTXAV model expected)")

            target_audio_tokens = audio_patchifier.patchify(noisy_audio)
            tgt_audio_seq_len = target_audio_tokens.shape[1]
            target_audio_ts = (
                audio_model_timesteps
                if audio_model_timesteps.shape[1] == tgt_audio_seq_len
                else audio_model_timesteps[:, :1].expand(bsz, tgt_audio_seq_len)
            )
            channels_audio = int(audio_latents.shape[1])
            mel_bins = int(audio_latents.shape[3])
            tgt_audio_shape = AudioLatentShape(batch=bsz, channels=channels_audio, frames=tgt_audio_seq_len, mel_bins=mel_bins)
            target_audio_pos = audio_patchifier.get_patch_grid_bounds(tgt_audio_shape, device=accelerator.device).to(dtype=network_dtype)

            video_ctx, audio_ctx = _split_av_context(base_model, text_embeds)
            av_cross_attention_mode = _normalize_av_cross_attention_mode(
                getattr(args, "av_cross_attention_mode", "both")
            )
            av_audio_to_video_enabled = av_cross_attention_mode in {"both", "a2v_only"}
            av_video_to_audio_enabled = av_cross_attention_mode in {"both", "v2a_only"}
            total_video_seq = ref_video_seq_len + tgt_video_seq_len
            total_audio_seq = tgt_audio_seq_len
            mask_dtype = network_dtype if network_dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16) else torch.float32
            neg_inf = torch.finfo(mask_dtype).min

            a2v_cross_attention_mask = None
            if not av_audio_to_video_enabled:
                a2v_cross_attention_mask = torch.full(
                    (bsz, total_video_seq, total_audio_seq),
                    neg_inf,
                    device=accelerator.device,
                    dtype=mask_dtype,
                )

            v2a_cross_attention_mask = None
            if not av_video_to_audio_enabled:
                v2a_cross_attention_mask = torch.full(
                    (bsz, total_audio_seq, total_video_seq),
                    neg_inf,
                    device=accelerator.device,
                    dtype=mask_dtype,
                )

            video_modality = Modality(
                enabled=True,
                latent=video_combined_tokens,
                timesteps=video_combined_ts,
                positions=video_combined_pos,
                context=video_ctx,
                sigma=sigma,
                context_mask=text_mask,
                a2v_cross_attention_mask=a2v_cross_attention_mask,
            )
            audio_modality = Modality(
                enabled=True,
                latent=target_audio_tokens,
                timesteps=target_audio_ts,
                positions=target_audio_pos,
                context=audio_ctx,
                sigma=audio_sigma,
                context_mask=text_mask,
                v2a_cross_attention_mask=v2a_cross_attention_mask,
            )

            perturbations = BatchedPerturbationConfig.empty(bsz)

            if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
                self._ensure_fp8_buffers_on_device(unwrapped_transformer)
            elif getattr(args, "nf4_base", False):
                self._ensure_nf4_buffers_on_device(unwrapped_transformer)
            with accelerator.autocast():
                if hasattr(unwrapped_transformer, "forward_modalities"):
                    video_pred_all, audio_pred_all = unwrapped_transformer.forward_modalities(
                        video_modality, audio_modality, perturbations,
                    )
                else:
                    video_pred_all, audio_pred_all = base_model(video_modality, audio_modality, perturbations)

            target_video_pred = video_pred_all[:, ref_video_seq_len:, :]
            target_audio_pred = audio_pred_all
            video_velocity = video_patchifier.patchify(noise - latents)
            audio_velocity = audio_patchifier.patchify(audio_target)

            target_audio_loss_mask = _compose_target_audio_loss_mask(
                audio_loss_mask,
                _cached_audio_loss_mask(tgt_audio_seq_len, bsz),
                batch_size=bsz,
                target_seq_len=tgt_audio_seq_len,
                device=accelerator.device,
            )

            out_video_ref_av: Dict[str, Any] = {
                "video_pred": target_video_pred,
                "video_target": video_velocity,
                "video_loss_mask": _combine_loss_masks(~tgt_video_cond_mask, _cached_video_loss_mask(as_tokens=True)),
                "video_loss_weight": _resolve_loss_weight("video_loss_weight", "video_loss_weight"),
                "audio_pred": target_audio_pred,
                "audio_target": audio_velocity,
                "audio_loss_mask": target_audio_loss_mask,
                "audio_loss_weight": _resolve_loss_weight("audio_loss_weight", "audio_loss_weight"),
                "audio_sigma": audio_sigma,
            }
            self._last_dit_inputs = None
            return out_video_ref_av, torch.tensor(0.0, device=accelerator.device)

        model_input = model_noisy_video
        if self._ltx_mode == "av" and audio_enabled_for_batch:
            model_input = [model_noisy_video, noisy_audio]
        _log_stats("noisy_video", model_noisy_video)
        _log_stats("timesteps", timesteps)

        video_conditioning_mask_tokens = None
        video_loss_mask = None
        if video_conditioning_enabled is not None:
            bsz, _c, frames, height, width = latents.shape
            seq_len = frames * height * width
            first_frame_tokens = height * width
            video_conditioning_mask_tokens = torch.zeros((bsz, seq_len), device=accelerator.device, dtype=torch.bool)
            if first_frame_tokens > 0:
                video_conditioning_mask_tokens[video_conditioning_enabled, :first_frame_tokens] = True
            transformer_options = {"patches_replace": {}, "video_conditioning_mask": video_conditioning_mask_tokens}

            if getattr(args, "video_loss_mask_5d", False):
                video_loss_mask = torch.ones((bsz, 1, frames, 1, 1), device=accelerator.device, dtype=torch.bool)
                if frames > 0:
                    video_loss_mask[video_conditioning_enabled, :, 0:1, :, :] = False
            else:
                video_loss_mask = torch.ones((bsz, frames), device=accelerator.device, dtype=torch.bool)
                if frames > 0:
                    video_loss_mask[video_conditioning_enabled, 0] = False
        video_loss_mask = _combine_loss_masks(video_loss_mask, _cached_video_loss_mask(as_tokens=False))

        resolved_transformer_options = transformer_options if video_conditioning_mask_tokens is not None else {"patches_replace": {}}
        if (
            self._ltx_mode == "av"
            and audio_ref_only_ic_enabled
            and audio_enabled_for_batch
            and noisy_audio is not None
            and ref_audio_seq_len > 0
        ):
            resolved_transformer_options = dict(resolved_transformer_options)
            resolved_transformer_options.update(
                self._build_audio_ref_transformer_overrides(
                    args=args,
                    transformer=transformer,
                    video_latents=model_noisy_video,
                    text_embeds=text_embeds,
                    text_mask=text_mask,
                    audio_model_latents=noisy_audio,
                    ref_audio_seq_len=ref_audio_seq_len,
                    device=accelerator.device,
                    dtype=network_dtype,
                )
            )

        # TARP / DCR injection (no-op when both flags are off)
        if (self._tarp_enabled or self._dcr_enabled) and audio_enabled_for_batch:
            resolved_transformer_options = dict(resolved_transformer_options)
            if self._tarp_enabled:
                resolved_transformer_options["tarp_config"] = {
                    "window_multiplier": self._tarp_window_multiplier,
                }
            if self._dcr_enabled:
                # Per-sample audio mask: 1.0 = normal gradient, 0.0 = detach
                # Detach when: (a) audio absent (zero-padded), or
                #              (b) audio is clean reference (timestep=0)
                _dcr_audio = torch.ones(latents.shape[0], device=accelerator.device)
                _dcr_audio_lengths = batch.get("audio_lengths")
                if isinstance(_dcr_audio_lengths, dict):
                    _dcr_audio_lengths = _dcr_audio_lengths.get("lengths")
                if isinstance(_dcr_audio_lengths, torch.Tensor):
                    if _dcr_audio_lengths.dim() == 0:
                        _dcr_audio_lengths = _dcr_audio_lengths.view(1)
                    if _dcr_audio_lengths.numel() == 1 and latents.shape[0] != 1:
                        _dcr_audio_lengths = _dcr_audio_lengths.expand(latents.shape[0])
                    _dcr_audio_lengths = _dcr_audio_lengths.to(device=accelerator.device, dtype=torch.int64)
                    _dcr_audio[_dcr_audio_lengths <= 0] = 0.0
                # Reference-stream detachment: sigma == 0 means clean conditioning
                if self._dcr_reference_detach:
                    if isinstance(audio_sigma, torch.Tensor):
                        _dcr_audio[audio_sigma == 0] = 0.0
                resolved_transformer_options["dcr_audio_mask"] = _dcr_audio.view(-1, 1, 1)

                # Per-sample video mask: detach video when it's the reference (sigma=0)
                if self._dcr_reference_detach and isinstance(sigma, torch.Tensor) and (sigma == 0).any():
                    _dcr_video = torch.ones(latents.shape[0], device=accelerator.device)
                    _dcr_video[sigma == 0] = 0.0
                    resolved_transformer_options["dcr_video_mask"] = _dcr_video.view(-1, 1, 1)

        # Store inputs for preservation / Self-Flow techniques (no-op when both are off)
        if self._preservation_active or self._self_flow_active:
            self._last_dit_inputs = {
                "model_input": model_input,
                "model_timesteps": model_timesteps,
                "audio_model_timesteps": audio_timestep_for_model if audio_enabled_for_batch else None,
                "text_embeds": text_embeds,
                "text_mask": text_mask,
                "frame_rate": frame_rate,
                "transformer_options": resolved_transformer_options,
            }

        if self._self_flow_active and self._self_flow is not None:
            self._self_flow.cleanup_step()
            network_for_self_flow = getattr(self, "_self_flow_network", None)
            is_train_step = bool(getattr(network_for_self_flow, "training", False)) if network_for_self_flow is not None else bool(
                getattr(transformer, "training", False)
            )
            if is_train_step and bool(getattr(args, "self_flow", False)):
                sf_ctx = self._self_flow_step_context
                if sf_ctx is not None:
                    teacher_noisy = sf_ctx.get("teacher_noisy_model_input")
                    teacher_timesteps = sf_ctx.get("teacher_model_timesteps")
                    if isinstance(teacher_noisy, torch.Tensor) and isinstance(teacher_timesteps, torch.Tensor):
                        teacher_noisy_input = teacher_noisy
                        if video_conditioning_enabled is not None and teacher_noisy_input.shape[2] > 0:
                            teacher_noisy_input = teacher_noisy_input.clone()
                            teacher_noisy_input[video_conditioning_enabled, :, 0:1, :, :] = latents[
                                video_conditioning_enabled, :, 0:1, :, :
                            ]

                        teacher_model_input_for_self_flow: Any = teacher_noisy_input.to(
                            device=accelerator.device, dtype=network_dtype
                        )
                        teacher_audio_timestep = None
                        if (
                            isinstance(model_input, (list, tuple))
                            and len(model_input) >= 2
                            and isinstance(model_input[1], torch.Tensor)
                        ):
                            teacher_audio_source = (
                                teacher_noisy_audio_for_self_flow
                                if isinstance(teacher_noisy_audio_for_self_flow, torch.Tensor)
                                else model_input[1]
                            )
                            teacher_audio_input = teacher_audio_source.to(device=accelerator.device, dtype=network_dtype)
                            teacher_model_input_for_self_flow = [teacher_model_input_for_self_flow, teacher_audio_input]
                            teacher_audio_timestep_source = (
                                teacher_audio_timestep_for_self_flow
                                if isinstance(teacher_audio_timestep_for_self_flow, torch.Tensor)
                                else audio_timestep_for_model
                            )
                            if isinstance(teacher_audio_timestep_source, torch.Tensor):
                                teacher_audio_timestep = teacher_audio_timestep_source.to(
                                    device=accelerator.device, dtype=network_dtype
                                )

                        teacher_timesteps_model = self._normalize_timesteps_for_model(
                            teacher_timesteps.to(device=accelerator.device, dtype=network_dtype)
                        )
                        if teacher_timesteps_model.dim() == 0:
                            teacher_timesteps_model = teacher_timesteps_model.unsqueeze(0)
                        if teacher_timesteps_model.dim() == 1:
                            teacher_timesteps_model = teacher_timesteps_model.unsqueeze(1)

                        if network_for_self_flow is not None:
                            self._self_flow.prepare_teacher_features(
                                accelerator=accelerator,
                                transformer=transformer,
                                network=network_for_self_flow,
                                teacher_model_input=teacher_model_input_for_self_flow,
                                teacher_timesteps=teacher_timesteps_model,
                                audio_timestep=teacher_audio_timestep,
                                text_embeds=text_embeds,
                                text_mask=text_mask,
                                frame_rate=frame_rate,
                                transformer_options=resolved_transformer_options,
                            )
                self._self_flow.mark_student_forward()

        # Connector LoRA: pass pre-connector features for on-the-fly connector processing
        if self._train_connectors and conditions is not None:
            video_features = conditions.get("video_features")
            if not isinstance(video_features, torch.Tensor):
                if not getattr(self, "_warned_missing_connector_features", False):
                    logger.error(
                        "--train_connectors is set but cache lacks 'video_features' keys. "
                        "Re-run caching with --cache_before_connector. "
                        "Connector LoRA will have NO effect on training."
                    )
                    self._warned_missing_connector_features = True
            if isinstance(video_features, torch.Tensor):
                resolved_transformer_options = dict(resolved_transformer_options)
                resolved_transformer_options["video_features"] = video_features.to(
                    device=accelerator.device, dtype=network_dtype
                )
                audio_features = conditions.get("audio_features")
                if isinstance(audio_features, torch.Tensor):
                    resolved_transformer_options["audio_features"] = audio_features.to(
                        device=accelerator.device, dtype=network_dtype
                    )
                resolved_transformer_options["features_attention_mask"] = text_mask

        if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
            self._ensure_fp8_buffers_on_device(transformer)
        with accelerator.autocast():
            model_pred = transformer(
                model_input,
                timestep=model_timesteps,
                audio_timestep=audio_timestep_for_model if audio_enabled_for_batch else None,
                context=text_embeds,
                attention_mask=text_mask,
                frame_rate=frame_rate,
                transformer_options=resolved_transformer_options,
            )

        video_pred = model_pred
        audio_pred = None
        if isinstance(model_pred, (list, tuple)):
            if len(model_pred) != 2:
                raise ValueError(f"Expected AV model to return [video_pred, audio_pred], got {len(model_pred)} outputs")
            video_pred, audio_pred = model_pred
        _check_finite("video_pred", video_pred)
        _check_finite("audio_pred", audio_pred)
        _log_stats("video_pred", video_pred)
        _log_stats("audio_pred", audio_pred)

        if skip_nonfinite and nonfinite_flag["hit"]:
            return {"_skip_step": True, "skip_reason": nonfinite_flag["tag"]}, torch.tensor(
                0.0, device=accelerator.device
            )

        video_target = noise - latents
        _check_finite("video_target", video_target)
        _log_stats("video_target", video_target)

        out: Dict[str, Any] = {
            "video_pred": video_pred,
            "video_target": video_target,
            "video_loss_mask": video_loss_mask,
            "video_loss_weight": _resolve_loss_weight("video_loss_weight", "video_loss_weight"),
        }

        # HFATO: pass metadata for x0-prediction loss in the training loop
        _hfato_data = batch.get("_hfato")
        if _hfato_data is not None:
            out["_hfato"] = {
                "noisy": noisy_model_input,
                "clean": _hfato_data["clean_latents"],
                "sigma": sigma,
            }

        if out["video_loss_weight"] < 0.0:
            raise ValueError(f"video_loss_weight must be >= 0. Got: {out['video_loss_weight']}")

        if self._ltx_mode == "av" and audio_enabled_for_batch:
            if audio_pred is None:
                raise ValueError("AV mode expected an audio prediction but got None")
            if audio_target is None:
                raise ValueError("Internal error: audio_target is missing in AV path")
            _check_finite("audio_target", audio_target)
            _log_stats("audio_target", audio_target)

            audio_seq_len = int(audio_target.shape[2])
            if audio_loss_mask is None:
                audio_loss_mask = torch.ones(
                    (audio_target.shape[0], audio_seq_len),
                    device=accelerator.device,
                    dtype=torch.bool,
                )

            if getattr(args, "use_audio_length_mask", False) and not audio_ref_only_ic_enabled:
                audio_lengths = batch.get("audio_lengths")
                if isinstance(audio_lengths, dict):
                    audio_lengths = audio_lengths.get("lengths")
                if isinstance(audio_lengths, torch.Tensor):
                    if audio_lengths.dim() == 0:
                        audio_lengths = audio_lengths.view(1)
                    if audio_lengths.dim() != 1:
                        raise ValueError(f"Expected audio_lengths to be 1D [B] or scalar, got shape: {tuple(audio_lengths.shape)}")
                    if audio_lengths.numel() == 1 and audio_target.shape[0] != 1:
                        audio_lengths = audio_lengths.expand(audio_target.shape[0])
                    if audio_lengths.shape[0] != audio_target.shape[0]:
                        raise ValueError(
                            f"Batch size mismatch: audio_target batch={audio_target.shape[0]} vs audio_lengths batch={audio_lengths.shape[0]}"
                        )

                    audio_lengths = audio_lengths.to(device=accelerator.device)
                    if audio_lengths.dtype.is_floating_point:
                        audio_lengths = audio_lengths.to(dtype=torch.int64)
                    else:
                        audio_lengths = audio_lengths.to(dtype=torch.int64)

                    audio_lengths = audio_lengths.clamp(min=0, max=audio_seq_len)
                    t = torch.arange(audio_seq_len, device=accelerator.device).view(1, -1)
                    audio_loss_mask = t < audio_lengths.view(-1, 1)
            audio_loss_mask = _combine_loss_masks(
                audio_loss_mask,
                _cached_audio_loss_mask(audio_seq_len, int(audio_target.shape[0])),
            )
            out.update(
                {
                    "audio_pred": audio_pred,
                    "audio_target": audio_target,
                    "audio_loss_mask": audio_loss_mask,
                    "audio_loss_weight": _resolve_loss_weight("audio_loss_weight", "audio_loss_weight")
                    * (
                        float(getattr(args, "audio_silence_regularizer_weight", 1.0))
                        if audio_regularizer_active
                        else 1.0
                    ),
                    "audio_sigma": audio_sigma,
                }
            )
            if out["audio_loss_weight"] < 0.0:
                raise ValueError(f"audio_loss_weight must be >= 0. Got: {out['audio_loss_weight']}")

            # Store data for Cross-Task Synergy (if enabled)
            cts_lambda_v = float(getattr(args, "cts_lambda_video_driven", 0.0))
            cts_lambda_a = float(getattr(args, "cts_lambda_audio_driven", 0.0))
            if cts_lambda_v > 0.0 or cts_lambda_a > 0.0:
                out["_cts"] = {
                    "noisy_video": model_noisy_video,
                    "clean_video": latents,
                    "noisy_audio": noisy_audio,
                    "clean_audio": audio_latents,
                    "video_timesteps": model_timesteps,
                    "audio_timesteps": audio_timestep_for_model,
                    "text_embeds": text_embeds,
                    "text_mask": text_mask,
                    "frame_rate": frame_rate,
                    "transformer_options": resolved_transformer_options,
                    "lambda_video_driven": cts_lambda_v,
                    "lambda_audio_driven": cts_lambda_a,
                }

        if diag_enabled:
            item_keys = batch.get("item_keys")
            if isinstance(item_keys, list) and item_keys:
                logger.info("DIAG item_keys: %s", item_keys[:5])
            latent_paths = batch.get("latent_cache_paths")
            if isinstance(latent_paths, list) and latent_paths:
                logger.info("DIAG latent_cache_paths: %s", latent_paths[:3])

        return out, torch.tensor(0.0, device=accelerator.device)

    def scale_shift_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Scale and shift latents for training (optional normalization)"""
        # LTX-2 typically doesn't require normalization, but can be enabled if needed
        return latents


# ======== Argument parser setup ========


# ======== Main training entry point ========




if __name__ == "__main__":
    main()
