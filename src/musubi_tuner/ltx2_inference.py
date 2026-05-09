"""LTX-2 Inference Module.

Handles video generation inference including single-stage and two-stage pipelines.
Two-stage inference generates at half resolution then upsamples for better quality.
"""

from __future__ import annotations

import gc
import logging
import os
import wave
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from musubi_tuner.ltx2_samplers import resolve_ltx2_sampler, res2s_midpoint, res2s_step
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)

# Stage 2 distilled sigma values for the LTX-2 two-stage path.
# These are a subset of the full distilled schedule, optimized for refinement
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


@dataclass
class InferenceConfig:
    """Configuration for LTX-2 inference."""

    # Basic generation parameters
    prompt: str = ""
    negative_prompt: Optional[str] = None
    width: int = 768
    height: int = 512
    frame_count: int = 45
    frame_rate: float = 25.0
    sample_steps: int = 40  # Default: 40 steps
    guidance_scale: float = 1.0
    cfg_scale: Optional[float] = 4.0  # Default: 4.0 CFG
    video_cfg_scale: Optional[float] = None
    audio_cfg_scale: Optional[float] = None
    discrete_flow_shift: float = 5.0
    seed: Optional[int] = None
    sigma_schedule: str = "auto"
    sample_sampler: str = "auto"
    sampling_preset: Optional[str] = None

    # Two-stage inference
    two_stage: bool = False
    spatial_upsampler_path: Optional[str] = None
    distilled_lora_path: Optional[str] = None
    stage2_steps: int = 3  # Stage 2 uses 3 steps (4 sigma values including 0.0)
    stage1_distilled_lora_multiplier: Optional[float] = None
    stage2_distilled_lora_multiplier: Optional[float] = None

    # Offloading
    offload_between_stages: bool = False

    # STG (Spatio-Temporal Guidance) — opt-in, inert when stg_scale == 0.0
    stg_scale: float = 0.0
    stg_blocks: Optional[List[int]] = None  # None = all blocks
    stg_mode: str = "video"  # "video" | "audio" | "both"

    # CFG★ rescaling (LTX-2.3 default is 0.9). 0.0 disables.
    rescale_scale: float = 0.0
    video_rescale_scale: Optional[float] = None
    audio_rescale_scale: Optional[float] = None
    video_modality_scale: float = 1.0
    audio_modality_scale: float = 1.0

    # Audio settings
    enable_audio: bool = False
    audio_only: bool = False

    # Embeddings (pre-computed)
    prompt_embeds: Optional[torch.Tensor] = None
    prompt_attention_mask: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_attention_mask: Optional[torch.Tensor] = None

    # I2V conditioning
    conditioning_latent: Optional[torch.Tensor] = None
    use_i2v_token_timestep_mask: bool = True

    # Latent-guide conditioning.
    # latent_idx_guides: list of LatentIndexGuide — replaces tokens at the given
    # latent frame slot with guide latents (generalizes I2V to arbitrary frames).
    # keyframe_guides: list of KeyframeGuide — appends extra tokens with custom
    # frame-index positional encoding (use frame_idx=-1 for global reference).
    latent_idx_guides: Optional[List["LatentIndexGuide"]] = None
    keyframe_guides: Optional[List["KeyframeGuide"]] = None

    # Extra options
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatentIndexGuide:
    """Replace tokens at a specific latent-frame slot with guide latents.

    Generalizes I2V: I2V is the special case latent_idx=0, strength=1.0.
    Mirrors VideoConditionByLatentIndex semantics on 5D latents (no patchify needed).
    """
    latent: torch.Tensor       # [B, C, T_g, H_lat, W_lat] — guide latents (already VAE-encoded)
    latent_idx: int            # Target latent-frame slot (must satisfy latent_idx + T_g <= total latent frames)
    strength: float = 1.0      # 1.0 = fully clean (no denoising), 0.0 = no conditioning


@dataclass
class KeyframeGuide:
    """Append extra tokens to the sequence with custom frame-index positional encoding.

    frame_idx=-1 is the canonical global-reference case (tokens visible to all frames
    via positional encoding, but not part of the temporal grid).
    Mirrors VideoConditionByKeyframeIndex semantics.
    """
    latent: torch.Tensor       # [B, C, T_g, H_lat, W_lat] — guide latents
    frame_idx: int             # Frame index used to offset positional encoding (-1 = global)
    strength: float = 1.0
    # When True, the appended token's temporal extent is collapsed to a single
    # pixel-frame (end = start + 1). Appropriate for still-image keyframes;
    # leave None to defer to the caller default in build_keyframe_extension.
    collapse_to_single_pixel_frame: Optional[bool] = None


class LTX2Inferencer:
    """LTX-2 inference handler with single and two-stage support."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        vae: Any,
        device: torch.device,
        dit_dtype: torch.dtype,
        audio_video_mode: bool = False,
    ):
        self.transformer = transformer
        self.vae = vae
        self.device = device
        self.dit_dtype = dit_dtype
        self._audio_video = audio_video_mode

        # Cached components
        self._spatial_upsampler: Optional[torch.nn.Module] = None
        self._distilled_lora_state: Optional[Dict[str, torch.Tensor]] = None
        self._original_lora_state: Optional[Dict[str, torch.Tensor]] = None
        self._audio_preview_config: Optional[Dict[str, Any]] = None

    def load_spatial_upsampler(
        self,
        upsampler_path: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.nn.Module:
        """Load the spatial upsampler model for two-stage inference."""
        if self._spatial_upsampler is not None:
            return self._spatial_upsampler

        device = device or self.device
        dtype = dtype or torch.bfloat16

        logger.info("Loading spatial upsampler from %s", upsampler_path)

        from musubi_tuner.ltx_2.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from musubi_tuner.ltx_2.model.upsampler import LatentUpsamplerConfigurator

        upsampler = SingleGPUModelBuilder(
            model_path=upsampler_path,
            model_class_configurator=LatentUpsamplerConfigurator,
        ).build(device=device, dtype=dtype)

        upsampler.eval()
        self._spatial_upsampler = upsampler
        return upsampler

    def load_distilled_lora(
        self,
        lora_path: str,
    ) -> Dict[str, torch.Tensor]:
        """Load distilled LoRA weights for stage 2 refinement."""
        if self._distilled_lora_state is not None:
            return self._distilled_lora_state

        logger.info("Loading distilled LoRA from %s", lora_path)

        from safetensors.torch import load_file
        state = load_file(lora_path)
        try:
            from musubi_tuner.ltx_2.convert_lora_to_comfy import (
                convert_lora_from_comfy_state_dict,
                is_comfy_lora_state_dict,
            )

            if is_comfy_lora_state_dict(state):
                logger.info("Converting distilled LoRA from external format")
                state = convert_lora_from_comfy_state_dict(state)
        except Exception:
            logger.exception("Failed to convert distilled LoRA weights from external format")
            raise

        self._distilled_lora_state = state
        return self._distilled_lora_state

    def _apply_distilled_lora(self, multiplier: float = 1.0) -> None:
        """Apply distilled LoRA weights to transformer (for stage 2)."""
        if self._distilled_lora_state is None:
            logger.warning("No distilled LoRA loaded; skipping application")
            return

        from musubi_tuner.networks import lora_ltx2

        # Create and merge LoRA
        net = lora_ltx2.create_arch_network_from_weights(
            multiplier,
            self._distilled_lora_state,
            unet=self.transformer,
            for_inference=True,
        )
        net.merge_to(
            None,
            self.transformer,
            self._distilled_lora_state,
            device=self.device,
            non_blocking=True,
        )
        logger.info("Applied distilled LoRA with multiplier %.2f", multiplier)

    def _remove_distilled_lora(self, multiplier: float = 1.0) -> None:
        """Remove distilled LoRA weights from transformer."""
        if self._distilled_lora_state is None:
            return

        from musubi_tuner.networks import lora_ltx2

        # Merge with negative multiplier to remove
        net = lora_ltx2.create_arch_network_from_weights(
            -multiplier,
            self._distilled_lora_state,
            unet=self.transformer,
            for_inference=True,
        )
        net.merge_to(
            None,
            self.transformer,
            self._distilled_lora_state,
            device=self.device,
            non_blocking=True,
        )
        logger.info("Removed distilled LoRA")

    def _get_vae_factors(self) -> Tuple[int, int]:
        """Get VAE temporal and spatial downsample factors."""
        temporal = int(getattr(self.vae, "temporal_downsample_factor", 8))
        spatial = int(getattr(self.vae, "spatial_downsample_factor", 32))
        return temporal, spatial

    def _init_latents(
        self,
        batch_size: int,
        channels: int,
        frames: int,
        height: int,
        width: int,
        generator: torch.Generator,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Initialize random latents."""
        return torch.randn(
            (batch_size, channels, frames, height, width),
            dtype=dtype,
            device=self.device,
            generator=generator,
        )

    def _get_expected_embed_dim(self) -> Optional[int]:
        """Get expected embedding dimension based on mode."""
        transformer = getattr(self.transformer, "model", self.transformer)
        caption_proj = getattr(transformer, "caption_projection", None)
        linear_1 = getattr(caption_proj, "linear_1", None) if caption_proj is not None else None
        in_features = getattr(linear_1, "in_features", None)
        if isinstance(in_features, int) and in_features > 0:
            return in_features * 2 if self._audio_video else in_features

        # Some LTX-2.3 checkpoints use connector/preprocessor paths where the
        # transformer does not expose caption_projection. In that case the
        # single-stage sampler passes Gemma embeddings through unchanged, so the
        # two-stage path should do the same instead of guessing and warning.
        return None

    def _prepare_prompt_embeds(
        self,
        config: InferenceConfig,
        do_cfg: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare prompt embeddings for inference."""
        prompt_embeds = config.prompt_embeds
        if prompt_embeds is None:
            raise ValueError("Prompt embeddings are required for inference")

        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)

        # Check embedding dimension matches model expectation
        expected_dim = self._get_expected_embed_dim()
        current_dim = prompt_embeds.shape[-1]

        if expected_dim is not None and current_dim != expected_dim:
            if current_dim * 2 == expected_dim:
                # Video-only embeddings (1920) but AV model expects (3840)
                # Pad with zeros for audio portion
                logger.warning(
                    "Prompt embeddings are video-only (%d) but model expects AV (%d). "
                    "Padding with zeros. For best results, re-cache embeddings in AV mode.",
                    current_dim, expected_dim
                )
                padding = torch.zeros(
                    *prompt_embeds.shape[:-1], current_dim,
                    dtype=prompt_embeds.dtype, device=prompt_embeds.device
                )
                prompt_embeds = torch.cat([prompt_embeds, padding], dim=-1)
            elif current_dim == expected_dim * 2:
                # AV embeddings but video-only model - slice to video portion
                logger.warning(
                    "Prompt embeddings are AV (%d) but model expects video-only (%d). "
                    "Using video portion only.",
                    current_dim, expected_dim
                )
                prompt_embeds = prompt_embeds[..., :expected_dim]
            else:
                logger.warning(
                    "Prompt embedding dimension mismatch: got %d, expected %d. "
                    "This may cause errors.",
                    current_dim, expected_dim
                )

        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dit_dtype)

        prompt_mask = config.prompt_attention_mask

        def _normalize_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if mask is None:
                return None
            if mask.dim() == 1:
                return mask.unsqueeze(0)
            if mask.dim() > 2:
                return mask.view(mask.shape[0], -1)
            return mask

        if do_cfg:
            neg_embeds = config.negative_prompt_embeds
            neg_mask = config.negative_prompt_attention_mask

            if neg_embeds is not None:
                if neg_embeds.dim() == 2:
                    neg_embeds = neg_embeds.unsqueeze(0)
                neg_embeds = neg_embeds.to(device=self.device, dtype=self.dit_dtype)
                prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)

                prompt_mask = _normalize_mask(prompt_mask)
                neg_mask = _normalize_mask(neg_mask)
                if prompt_mask is not None and neg_mask is not None:
                    prompt_mask = torch.cat([neg_mask, prompt_mask], dim=0)
                elif prompt_mask is not None:
                    prompt_mask = torch.cat([prompt_mask, prompt_mask], dim=0)
            else:
                logger.warning(
                    "CFG is enabled but negative_prompt_embeds are missing; "
                    "falling back to duplicated positive embeddings."
                )
                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds], dim=0)
                prompt_mask = _normalize_mask(prompt_mask)
                if prompt_mask is not None:
                    prompt_mask = torch.cat([prompt_mask, prompt_mask], dim=0)

        prompt_mask = _normalize_mask(prompt_mask)

        # Align mask length to embeddings
        if prompt_mask is not None:
            mask_len = prompt_mask.shape[-1]
            embed_len = prompt_embeds.shape[1]
            if mask_len != embed_len:
                if mask_len > embed_len:
                    prompt_mask = prompt_mask[:, -embed_len:]
                else:
                    pad = embed_len - mask_len
                    prompt_mask = F.pad(prompt_mask, (pad, 0), value=1)

        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(device=self.device, dtype=torch.int64)

        return prompt_embeds, prompt_mask

    def _denoise_loop(
        self,
        latents: torch.Tensor,
        sigmas: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: Optional[torch.Tensor],
        do_cfg: bool,
        cfg_scale: float,
        frame_rate: float,
        audio_latents: Optional[torch.Tensor] = None,
        audio_only: bool = False,
        progress_desc: str = "LTX-2 inference",
        conditioning_latent: Optional[torch.Tensor] = None,
        use_i2v_token_timestep_mask: bool = True,
        latent_idx_guides: Optional[List[LatentIndexGuide]] = None,
        keyframe_guides: Optional[List["KeyframeGuide"]] = None,
        stg_scale: float = 0.0,
        stg_blocks: Optional[List[int]] = None,
        stg_mode: str = "video",
        rescale_scale: float = 0.0,
        video_cfg_scale: Optional[float] = None,
        audio_cfg_scale: Optional[float] = None,
        video_rescale_scale: Optional[float] = None,
        audio_rescale_scale: Optional[float] = None,
        video_modality_scale: float = 1.0,
        audio_modality_scale: float = 1.0,
        sample_sampler: str = "euler",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the denoising loop with optional I2V conditioning.

        Args:
            conditioning_latent: Optional first-frame conditioning latent [B, C, 1, H, W].
                                If provided, first frame will be locked during denoising.
        """
        from musubi_tuner.ltx_2.model.ltx2_scheduler import EulerDiffusionStep, X0PredictionWrapper

        stepper = EulerDiffusionStep()
        effective_video_cfg = float(video_cfg_scale if video_cfg_scale is not None else cfg_scale)
        effective_audio_cfg = float(audio_cfg_scale if audio_cfg_scale is not None else cfg_scale)
        effective_video_rescale = float(video_rescale_scale if video_rescale_scale is not None else rescale_scale)
        effective_audio_rescale = float(audio_rescale_scale if audio_rescale_scale is not None else rescale_scale)

        # Build a unified guide list:
        #   - legacy conditioning_latent → single LatentIndexGuide(latent_idx=0, strength=1.0)
        #   - explicit latent_idx_guides → as-is
        # Both go through the same setup path: build 5D denoise_mask + clean_latent
        # plus per-token video_conditioning_mask covering all guide tokens.
        denoise_mask = None
        clean_latent = None
        i2v_conditioning_mask_tokens = None

        # Determine whether the I2V lock will actually be established. The raw
        # `conditioning_latent is not None` check is insufficient — a malformed
        # conditioning_latent may fail downstream validation, leaving no lock.
        # Use a pre-validated flag so the user-guide overlap guard doesn't
        # spuriously block valid guides when no lock exists.
        i2v_lock_active = False
        if conditioning_latent is not None:
            try:
                cl_shape = conditioning_latent.shape
                # Match do_inference and legacy semantics: conditioning_latent
                # is a single-frame I2V anchor. For multi-frame replacement,
                # callers must use explicit latent_idx_guides.
                if (
                    conditioning_latent.dim() == 5
                    and cl_shape[2] == 1
                    and latents.shape[2] >= 1
                    and cl_shape[1] == latents.shape[1]
                ):
                    i2v_lock_active = True
                elif conditioning_latent.dim() == 5 and cl_shape[2] != 1:
                    logger.warning(
                        "conditioning_latent has %d frames; expected 1. Use "
                        "latent_idx_guides for multi-frame replacement. Treating "
                        "as no I2V lock.",
                        cl_shape[2],
                    )
            except Exception:
                i2v_lock_active = False

        guides: List[LatentIndexGuide] = []
        # Only append conditioning_latent if pre-validation passed; otherwise
        # the warning above already informed the user and we skip silently here.
        if conditioning_latent is not None and i2v_lock_active:
            guides.append(LatentIndexGuide(latent=conditioning_latent, latent_idx=0, strength=1.0))
        if latent_idx_guides:
            for g in latent_idx_guides:
                # Skip user guides at latent_idx=0 only when a real lock will exist.
                if i2v_lock_active and int(getattr(g, "latent_idx", 0)) == 0:
                    logger.warning(
                        "latent_idx_guide at latent_idx=0 overlaps active I2V lock; "
                        "skipping to preserve the lock. Remove conditioning_latent if "
                        "you want this guide to take effect."
                    )
                    continue
                # Clamp strength to [0, 1]; out-of-range produces broken
                # denoise_mask = 1 - strength arithmetic and corrupts the
                # latent / clean_latent / mask blend.
                raw_strength = float(getattr(g, "strength", 1.0))
                if raw_strength < 0.0 or raw_strength > 1.0:
                    logger.warning(
                        "latent_idx_guide: strength=%.3f outside [0,1]; clamping.",
                        raw_strength,
                    )
                clamped_strength = max(0.0, min(1.0, raw_strength))
                # strength<=0 means no conditioning; skip entirely so the apply
                # step doesn't overwrite the latent slice with clean guide
                # content tagged as fully noisy (denoise_mask=1).
                if clamped_strength <= 0.0:
                    logger.warning(
                        "latent_idx_guide: strength=0 skipped (no conditioning effect)."
                    )
                    continue
                # Substitute the clamped strength so downstream resolve uses it.
                guides.append(LatentIndexGuide(
                    latent=g.latent,
                    latent_idx=int(getattr(g, "latent_idx", 0)),
                    strength=clamped_strength,
                ))

        if guides:
            try:
                bsz, ch, total_frames, h_lat, w_lat = latents.shape
                if total_frames < 1:
                    logger.warning("Latent guides: video latents have no temporal frames. Skipping.")
                    guides = []

                # Validate every guide and resize spatially if needed (two-stage stage-1).
                resolved: List[Tuple[torch.Tensor, int, float]] = []
                for gi, g in enumerate(guides):
                    cond = g.latent
                    if cond is None or cond.dim() != 5:
                        logger.warning("Guide %d: invalid shape %s, skipping.", gi, tuple(getattr(cond, "shape", ())))
                        continue
                    cond = cond.to(device=latents.device, dtype=latents.dtype)
                    if cond.shape[1] != ch:
                        logger.warning("Guide %d: channel mismatch (%d vs %d), skipping.", gi, cond.shape[1], ch)
                        continue
                    if cond.shape[-2:] != latents.shape[-2:]:
                        # Squeeze T into batch for resize, then restore.
                        b_g, c_g, t_g, _, _ = cond.shape
                        flat = cond.reshape(b_g * t_g, c_g, cond.shape[3], cond.shape[4])
                        flat = F.interpolate(flat, size=latents.shape[-2:], mode="bilinear", align_corners=False)
                        cond = flat.reshape(b_g, c_g, t_g, latents.shape[3], latents.shape[4])
                        logger.info(
                            "Guide %d: resized to %s for current denoising stage.",
                            gi, tuple(cond.shape),
                        )
                    t_g = int(cond.shape[2])
                    if t_g < 1:
                        logger.warning("Guide %d: zero temporal length, skipping.", gi)
                        continue
                    if g.latent_idx < 0 or g.latent_idx + t_g > total_frames:
                        logger.warning(
                            "Guide %d: latent_idx %d + T %d out of range (total frames %d), skipping.",
                            gi, g.latent_idx, t_g, total_frames,
                        )
                        continue
                    resolved.append((cond, int(g.latent_idx), float(g.strength)))

                if resolved:
                    denoise_mask = torch.ones_like(latents)
                    clean_latent = torch.zeros_like(latents)
                    for cond, idx, strength in resolved:
                        t_g = int(cond.shape[2])
                        latents[:, :, idx : idx + t_g, :, :] = cond
                        # denoise_mask: 1 - strength (1.0 = fully clean → mask=0, no denoising).
                        denoise_mask[:, :, idx : idx + t_g, :, :] = 1.0 - strength
                        clean_latent[:, :, idx : idx + t_g, :, :] = cond

                    if use_i2v_token_timestep_mask:
                        seq_len = total_frames * h_lat * w_lat
                        i2v_conditioning_mask_tokens = torch.zeros(
                            (bsz, seq_len), device=latents.device, dtype=torch.bool,
                        )
                        for _cond, idx, strength in resolved:
                            if strength < 1.0:
                                continue  # Only fully-clean guides get token-timestep zeroing.
                            tg_eff = int(_cond.shape[2])
                            tokens_per_frame = h_lat * w_lat
                            start_tok = idx * tokens_per_frame
                            stop_tok = (idx + tg_eff) * tokens_per_frame
                            i2v_conditioning_mask_tokens[:, start_tok:stop_tok] = True
                        logger.info(
                            "Latent guides: %d guide(s) applied at frame indices %s.",
                            len(resolved), [r[1] for r in resolved],
                        )
            except Exception as e:
                logger.error("Latent guides: failed to setup: %s", e, exc_info=True)
                denoise_mask = None
                clean_latent = None
                i2v_conditioning_mask_tokens = None
                i2v_lock_active = False

        # Preprocess keyframe guides into the dict format expected by LTX2Wrapper.
        # We resize spatially to match the current denoising stage if needed (two-stage).
        kf_guide_dicts: Optional[List[Dict[str, Any]]] = None
        if keyframe_guides:
            kf_guide_dicts = []
            for gi, kg in enumerate(keyframe_guides):
                gl = kg.latent
                if gl is None or gl.dim() != 5:
                    logger.warning("KeyframeGuide %d: invalid shape %s, skipping.",
                                   gi, tuple(getattr(gl, "shape", ())))
                    continue
                gl = gl.to(device=latents.device, dtype=latents.dtype)
                if gl.shape[1] != latents.shape[1]:
                    logger.warning("KeyframeGuide %d: channel mismatch (%d vs %d), skipping.",
                                   gi, gl.shape[1], latents.shape[1])
                    continue
                if gl.shape[-2:] != latents.shape[-2:]:
                    b_g, c_g, t_g, _, _ = gl.shape
                    flat = gl.reshape(b_g * t_g, c_g, gl.shape[3], gl.shape[4])
                    flat = F.interpolate(flat, size=latents.shape[-2:], mode="bilinear", align_corners=False)
                    gl = flat.reshape(b_g, c_g, t_g, latents.shape[3], latents.shape[4])
                kf_frame_idx = int(kg.frame_idx)
                kf_strength = float(kg.strength)
                # Use i2v_lock_active (not raw conditioning_latent) so a
                # malformed I2V setup doesn't spuriously block valid keyframes.
                if i2v_lock_active and kf_frame_idx == 0 and kf_strength >= 1.0:
                    logger.warning(
                        "KeyframeGuide %d at frame_idx=0 with strength=%.2f competes with the "
                        "conditioning_latent first-frame lock; skipping. Use strength<1.0 for an "
                        "auxiliary cue, or remove conditioning_latent if this guide should win.",
                        gi, kf_strength,
                    )
                    continue
                if i2v_lock_active and kf_frame_idx == 0 and 0.9 <= kf_strength < 1.0:
                    logger.warning(
                        "KeyframeGuide %d at frame_idx=0 with strength=%.2f is very close to the "
                        "locking threshold (1.0). It will pass through as an auxiliary cue but "
                        "its effect may be visually indistinguishable from a competing lock.",
                        gi, kf_strength,
                    )
                kf_dict: Dict[str, Any] = {
                    "latent": gl,
                    "frame_idx": kf_frame_idx,
                    "strength": kf_strength,
                }
                # Propagate collapse_to_single_pixel_frame if the source carries
                # it (still-image keyframe inference relies on this). Without
                # propagation the guide silently uses the wider latent-slot span.
                kf_collapse = getattr(kg, "collapse_to_single_pixel_frame", None)
                if kf_collapse is not None:
                    kf_dict["collapse_to_single_pixel_frame"] = bool(kf_collapse)
                kf_guide_dicts.append(kf_dict)
            if kf_guide_dicts:
                logger.info(
                    "Keyframe guides: %d guide(s) at frame indices %s.",
                    len(kf_guide_dicts), [g["frame_idx"] for g in kf_guide_dicts],
                )
            else:
                kf_guide_dicts = None

        def _kf_for_cfg(do_cfg_local: bool) -> Optional[List[Dict[str, Any]]]:
            """Duplicate each guide along the batch dim to match CFG-expanded inputs."""
            if kf_guide_dicts is None:
                return None
            if not do_cfg_local:
                return kf_guide_dicts
            out = []
            for g in kf_guide_dicts:
                lat = g["latent"]
                doubled = {**g, "latent": torch.cat([lat, lat], dim=0)}
                s = g.get("strength")
                if isinstance(s, torch.Tensor):
                    doubled["strength"] = torch.cat([s, s], dim=0)
                out.append(doubled)
            return out

        sampler_name = resolve_ltx2_sampler(sample_sampler, None)
        if sampler_name == "res_2s" and audio_latents is not None:
            logger.warning("LTX-2 inference: res_2s is not wired for audio two-stage refinement yet; using Euler.")
            sampler_name = "euler"
        logger.info("LTX-2 inference sampler: %s", sampler_name)

        def _predict_video_x0_res2s(video_state: torch.Tensor, sigma_value: torch.Tensor) -> torch.Tensor:
            latent_input = torch.cat([video_state, video_state], dim=0) if do_cfg else video_state
            latent_input = latent_input.to(dtype=self.dit_dtype)
            timestep = sigma_value.expand(latent_input.shape[0]).to(device=self.device, dtype=self.dit_dtype)
            options: Dict[str, Any] = {"patches_replace": {}}
            if i2v_conditioning_mask_tokens is not None:
                mask_tokens = i2v_conditioning_mask_tokens
                if do_cfg:
                    mask_tokens = torch.cat([mask_tokens, mask_tokens], dim=0)
                options["video_conditioning_mask"] = mask_tokens
            kf_for_main_res2s = _kf_for_cfg(do_cfg)
            if kf_for_main_res2s is not None:
                options["keyframe_guides"] = kf_for_main_res2s

            pred = self.transformer(
                latent_input,
                timestep=timestep.unsqueeze(1),
                context=prompt_embeds,
                attention_mask=prompt_mask,
                frame_rate=frame_rate,
                transformer_options=options,
                audio_only=audio_only,
            )
            video_pred_mid = pred[0] if isinstance(pred, (list, tuple)) else pred
            video_pred_mid = video_pred_mid.to(dtype=video_state.dtype)
            sigma_for_video_mid = denoise_mask * sigma_value if denoise_mask is not None else sigma_value

            if do_cfg:
                vel_uncond_mid, vel_cond_mid = video_pred_mid.chunk(2)
                x0_uncond_mid = X0PredictionWrapper.velocity_to_x0(video_state, vel_uncond_mid, sigma_for_video_mid)
                x0_cond_mid = X0PredictionWrapper.velocity_to_x0(video_state, vel_cond_mid, sigma_for_video_mid)
                video_x0_mid = x0_uncond_mid + effective_video_cfg * (x0_cond_mid - x0_uncond_mid)
            else:
                x0_cond_mid = X0PredictionWrapper.velocity_to_x0(video_state, video_pred_mid, sigma_for_video_mid)
                video_x0_mid = x0_cond_mid

            stg_video_active_mid = stg_scale > 0.0 and stg_mode in ("video", "both")
            if stg_video_active_mid:
                from musubi_tuner.ltx_2.guidance.perturbations import (
                    BatchedPerturbationConfig,
                    Perturbation,
                    PerturbationConfig,
                    PerturbationType,
                )

                stg_context = prompt_embeds[prompt_embeds.shape[0] // 2 :] if do_cfg else prompt_embeds
                stg_ctx_mask = prompt_mask[prompt_mask.shape[0] // 2 :] if do_cfg and prompt_mask is not None else prompt_mask
                stg_opts: Dict[str, Any] = {
                    "patches_replace": {},
                    "perturbations": BatchedPerturbationConfig(
                        [PerturbationConfig(perturbations=[
                            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=stg_blocks)
                        ])]
                    ),
                }
                if i2v_conditioning_mask_tokens is not None:
                    stg_opts["video_conditioning_mask"] = i2v_conditioning_mask_tokens
                kf_for_stg_res2s = _kf_for_cfg(False)
                if kf_for_stg_res2s is not None:
                    stg_opts["keyframe_guides"] = kf_for_stg_res2s
                stg_pred = self.transformer(
                    video_state.to(dtype=self.dit_dtype),
                    timestep=sigma_value.expand(1).to(device=self.device, dtype=self.dit_dtype).unsqueeze(1),
                    context=stg_context,
                    attention_mask=stg_ctx_mask,
                    frame_rate=frame_rate,
                    transformer_options=stg_opts,
                    audio_only=audio_only,
                )
                stg_video_vel = stg_pred[0] if isinstance(stg_pred, (list, tuple)) else stg_pred
                stg_video_vel = stg_video_vel.to(dtype=video_state.dtype)
                x0_video_ptb_mid = X0PredictionWrapper.velocity_to_x0(video_state, stg_video_vel, sigma_for_video_mid)
                video_x0_mid = video_x0_mid + stg_scale * (x0_cond_mid - x0_video_ptb_mid)

            if effective_video_rescale > 0.0 and x0_cond_mid is not None:
                pred_std_mid = video_x0_mid.std()
                if pred_std_mid > 1e-6:
                    factor_mid = x0_cond_mid.std() / pred_std_mid
                    factor_mid = effective_video_rescale * factor_mid + (1.0 - effective_video_rescale)
                    video_x0_mid = video_x0_mid * factor_mid
            return video_x0_mid

        for step_idx in tqdm(range(len(sigmas) - 1), desc=progress_desc, leave=False):
            sigma = sigmas[step_idx]

            # Expand for CFG
            latent_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
            latent_input = latent_input.to(dtype=self.dit_dtype)

            audio_input = None
            if audio_latents is not None:
                audio_input = torch.cat([audio_latents, audio_latents], dim=0) if do_cfg else audio_latents
                audio_input = audio_input.to(dtype=self.dit_dtype)

            timestep = sigma.expand(latent_input.shape[0]).to(device=self.device, dtype=self.dit_dtype)
            resolved_transformer_options = {"patches_replace": {}}
            if i2v_conditioning_mask_tokens is not None:
                video_conditioning_mask_tokens = i2v_conditioning_mask_tokens
                if do_cfg:
                    video_conditioning_mask_tokens = torch.cat(
                        [video_conditioning_mask_tokens, video_conditioning_mask_tokens],
                        dim=0,
                    )
                resolved_transformer_options["video_conditioning_mask"] = video_conditioning_mask_tokens
            kf_for_main = _kf_for_cfg(do_cfg)
            if kf_for_main is not None:
                resolved_transformer_options["keyframe_guides"] = kf_for_main

            # Model input
            if self._audio_video and audio_input is not None:
                model_input = [latent_input, audio_input]
            else:
                model_input = latent_input

            model_pred = self.transformer(
                model_input,
                timestep=timestep.unsqueeze(1),
                context=prompt_embeds,
                attention_mask=prompt_mask,
                frame_rate=frame_rate,
                transformer_options=resolved_transformer_options,
                audio_only=audio_only,
            )

            audio_pred = None
            if isinstance(model_pred, (list, tuple)):
                video_pred, audio_pred = model_pred
            else:
                video_pred = model_pred

            # IMPORTANT: Convert velocity to x0 FIRST, then apply CFG to x0
            # X0Model wraps the velocity model before guidance is applied.
            # and CFG is applied to denoised (x0) outputs, not velocity predictions
            video_pred = video_pred.to(dtype=latents.dtype)

            sigma_for_video = denoise_mask * sigma if denoise_mask is not None else sigma

            if do_cfg:
                # Split velocity predictions for CFG
                vel_uncond, vel_cond = video_pred.chunk(2)
                # Convert each to x0
                x0_uncond = X0PredictionWrapper.velocity_to_x0(latents, vel_uncond, sigma_for_video)
                x0_cond = X0PredictionWrapper.velocity_to_x0(latents, vel_cond, sigma_for_video)
                # Apply CFG to x0 using the LTX-2 formula
                video_x0 = x0_uncond + effective_video_cfg * (x0_cond - x0_uncond)
            else:
                x0_cond = X0PredictionWrapper.velocity_to_x0(latents, video_pred, sigma_for_video)
                video_x0 = x0_cond

            # Modality guidance: extra conditional forward with A2V/V2A attention skipped.
            if (
                audio_latents is not None
                and (video_modality_scale != 1.0 or audio_modality_scale != 1.0)
                and x0_cond is not None
            ):
                from musubi_tuner.ltx_2.guidance.perturbations import (
                    BatchedPerturbationConfig,
                    Perturbation,
                    PerturbationConfig,
                    PerturbationType,
                )

                mod_context = prompt_embeds[prompt_embeds.shape[0] // 2 :] if do_cfg else prompt_embeds
                mod_mask = prompt_mask[prompt_mask.shape[0] // 2 :] if do_cfg and prompt_mask is not None else prompt_mask
                mod_opts: Dict[str, Any] = {
                    "patches_replace": {},
                    "perturbations": BatchedPerturbationConfig(
                        [
                            PerturbationConfig(
                                perturbations=[
                                    Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                                    Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
                                ]
                            )
                        ]
                    ),
                }
                if i2v_conditioning_mask_tokens is not None:
                    mod_opts["video_conditioning_mask"] = i2v_conditioning_mask_tokens
                kf_for_mod = _kf_for_cfg(False)
                if kf_for_mod is not None:
                    mod_opts["keyframe_guides"] = kf_for_mod
                mod_pred = self.transformer(
                    [latents.to(dtype=self.dit_dtype), audio_latents.to(dtype=self.dit_dtype)],
                    timestep=sigma.expand(1).to(device=self.device, dtype=self.dit_dtype).unsqueeze(1),
                    context=mod_context,
                    attention_mask=mod_mask,
                    frame_rate=frame_rate,
                    transformer_options=mod_opts,
                    audio_only=audio_only,
                )
                if isinstance(mod_pred, (list, tuple)):
                    mod_video_vel, mod_audio_vel = mod_pred
                else:
                    mod_video_vel, mod_audio_vel = mod_pred, None
                mod_video_x0 = X0PredictionWrapper.velocity_to_x0(
                    latents, mod_video_vel.to(dtype=latents.dtype), sigma_for_video
                )
                video_x0 = video_x0 + (video_modality_scale - 1.0) * (x0_cond - mod_video_x0)
            else:
                mod_audio_vel = None

            # STG refinement: perturbed forward (self-attention skipped at chosen blocks),
            # then steer x0 toward (x0_cond - x0_perturbed). Opt-in via stg_scale > 0.
            x0_audio_ptb: Optional[torch.Tensor] = None
            aud_x0_cond_for_stg: Optional[torch.Tensor] = None
            stg_video_active = stg_scale > 0.0 and stg_mode in ("video", "both")
            stg_audio_active = (
                stg_scale > 0.0 and stg_mode in ("audio", "both") and audio_pred is not None
            )
            if stg_video_active or stg_audio_active:
                from musubi_tuner.ltx_2.guidance.perturbations import (
                    BatchedPerturbationConfig,
                    Perturbation,
                    PerturbationConfig,
                    PerturbationType,
                )

                pert_list: List[Perturbation] = []
                if stg_video_active:
                    pert_list.append(
                        Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=stg_blocks)
                    )
                if stg_audio_active:
                    pert_list.append(
                        Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=stg_blocks)
                    )

                if do_cfg:
                    _halves = prompt_embeds.shape[0] // 2
                    stg_context = prompt_embeds[_halves:]
                    stg_ctx_mask = prompt_mask[_halves:] if prompt_mask is not None else None
                else:
                    stg_context = prompt_embeds
                    stg_ctx_mask = prompt_mask

                stg_latent = latents.to(dtype=self.dit_dtype)
                stg_audio_lat = (
                    audio_latents.to(dtype=self.dit_dtype) if audio_latents is not None else None
                )
                stg_timestep = sigma.expand(stg_latent.shape[0]).to(
                    device=self.device, dtype=self.dit_dtype
                )

                stg_opts: Dict[str, Any] = {"patches_replace": {}}
                if i2v_conditioning_mask_tokens is not None:
                    stg_opts["video_conditioning_mask"] = i2v_conditioning_mask_tokens
                kf_for_stg = _kf_for_cfg(False)
                if kf_for_stg is not None:
                    stg_opts["keyframe_guides"] = kf_for_stg
                stg_opts["perturbations"] = BatchedPerturbationConfig(
                    [PerturbationConfig(perturbations=pert_list)] * stg_latent.shape[0]
                )

                if self._audio_video and stg_audio_lat is not None:
                    stg_input = [stg_latent, stg_audio_lat]
                else:
                    stg_input = stg_latent

                stg_pred = self.transformer(
                    stg_input,
                    timestep=stg_timestep.unsqueeze(1),
                    context=stg_context,
                    attention_mask=stg_ctx_mask,
                    frame_rate=frame_rate,
                    transformer_options=stg_opts,
                    audio_only=audio_only,
                )

                stg_audio_vel = None
                if isinstance(stg_pred, (list, tuple)):
                    stg_video_vel, stg_audio_vel = stg_pred
                else:
                    stg_video_vel = stg_pred

                if stg_video_active:
                    stg_video_vel = stg_video_vel.to(dtype=latents.dtype)
                    x0_video_ptb = X0PredictionWrapper.velocity_to_x0(
                        latents, stg_video_vel, sigma_for_video
                    )
                    video_x0 = video_x0 + stg_scale * (x0_cond - x0_video_ptb)

                if stg_audio_active and stg_audio_vel is not None and audio_latents is not None:
                    stg_audio_vel = stg_audio_vel.to(dtype=audio_latents.dtype)
                    x0_audio_ptb = X0PredictionWrapper.velocity_to_x0(
                        audio_latents, stg_audio_vel, sigma.item()
                    )

            # CFG★ rescaling: after CFG + STG, rescale prediction toward cond.std() to
            # prevent oversaturation from amplified guidance. Matches the MultiModalGuider behavior.
            if effective_video_rescale > 0.0 and x0_cond is not None:
                pred_std = video_x0.std()
                if pred_std > 1e-6:
                    factor = x0_cond.std() / pred_std
                    factor = effective_video_rescale * factor + (1.0 - effective_video_rescale)
                    video_x0 = video_x0 * factor

            if denoise_mask is not None and clean_latent is not None:
                # LTX-2 ordering: blend denoised x0 before Euler step.
                video_x0 = video_x0 * denoise_mask + clean_latent * (1.0 - denoise_mask)

            if sampler_name == "res_2s":
                midpoint = res2s_midpoint(latents, video_x0, sigmas[step_idx], sigmas[step_idx + 1])
                if midpoint is None:
                    latents = video_x0
                else:
                    midpoint_latents, midpoint_sigma = midpoint
                    midpoint_video_x0 = _predict_video_x0_res2s(midpoint_latents, midpoint_sigma)
                    if denoise_mask is not None and clean_latent is not None:
                        midpoint_video_x0 = midpoint_video_x0 * denoise_mask + clean_latent * (1.0 - denoise_mask)
                    latents = res2s_step(latents, video_x0, midpoint_video_x0, sigmas[step_idx], sigmas[step_idx + 1])
            else:
                latents = stepper.step(latents, video_x0, sigmas, step_idx)

            # CRITICAL: Hard-lock conditioned frames after Euler step
            # The Euler step performs gradual correction, but I2V requires absolute locking
            if denoise_mask is not None and clean_latent is not None:
                # Restore locked frames: where denoise_mask == 0.0, force latents = clean_latent
                latents = latents * denoise_mask + clean_latent * (1.0 - denoise_mask)

            if audio_pred is not None and audio_latents is not None:
                audio_pred = audio_pred.to(dtype=audio_latents.dtype)
                if do_cfg:
                    aud_vel_uncond, aud_vel_cond = audio_pred.chunk(2)
                    aud_x0_uncond = X0PredictionWrapper.velocity_to_x0(audio_latents, aud_vel_uncond, sigma.item())
                    aud_x0_cond = X0PredictionWrapper.velocity_to_x0(audio_latents, aud_vel_cond, sigma.item())
                    audio_x0 = aud_x0_uncond + effective_audio_cfg * (aud_x0_cond - aud_x0_uncond)
                else:
                    aud_x0_cond = X0PredictionWrapper.velocity_to_x0(audio_latents, audio_pred, sigma.item())
                    audio_x0 = aud_x0_cond
                if mod_audio_vel is not None:
                    mod_audio_x0 = X0PredictionWrapper.velocity_to_x0(
                        audio_latents, mod_audio_vel.to(dtype=audio_latents.dtype), sigma.item()
                    )
                    audio_x0 = audio_x0 + (audio_modality_scale - 1.0) * (aud_x0_cond - mod_audio_x0)
                if x0_audio_ptb is not None:
                    audio_x0 = audio_x0 + stg_scale * (aud_x0_cond - x0_audio_ptb)
                if effective_audio_rescale > 0.0:
                    pred_std = audio_x0.std()
                    if pred_std > 1e-6:
                        factor = aud_x0_cond.std() / pred_std
                        factor = effective_audio_rescale * factor + (1.0 - effective_audio_rescale)
                        audio_x0 = audio_x0 * factor
                audio_latents = stepper.step(audio_latents, audio_x0, sigmas, step_idx)

        # Free I2V conditioning tensors to reclaim memory
        if denoise_mask is not None or clean_latent is not None:
            del denoise_mask, clean_latent
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return latents, audio_latents

    def _upsample_latents(
        self,
        latents: torch.Tensor,
        upsampler: torch.nn.Module,
    ) -> torch.Tensor:
        """Upsample latents using the spatial upsampler."""
        from musubi_tuner.ltx_2.model.upsampler import upsample_video

        # Get per_channel_statistics for normalization
        # Try multiple locations: vae.encoder, vae.decoder, vae itself
        per_channel_stats = None
        for attr_path in ["encoder.per_channel_statistics", "decoder.per_channel_statistics", "per_channel_statistics"]:
            obj = self.vae
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr, None)
                    if obj is None:
                        break
                if obj is not None:
                    per_channel_stats = obj
                    break
            except Exception:
                continue

        if per_channel_stats is None:
            # Fallback: do simple upsampling without normalization
            logger.warning("VAE per_channel_statistics not available; upsampling without normalization")
            return upsampler(latents)

        # Create a simple object with per_channel_statistics for upsample_video
        class _EncoderProxy:
            def __init__(self, stats):
                self.per_channel_statistics = stats

        return upsample_video(latents, _EncoderProxy(per_channel_stats), upsampler)

    def generate(
        self,
        config: InferenceConfig,
        audio_decoder: Optional[torch.nn.Module] = None,
        vocoder: Optional[torch.nn.Module] = None,
        decode_video: bool = True,
        use_tiled_vae: bool = False,
        tiled_vae_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Generate video (and optionally audio) from config.

        Args:
            config: Inference configuration
            audio_decoder: Audio decoder module (optional)
            vocoder: Vocoder module (optional)
            decode_video: Whether to decode video latents
            use_tiled_vae: Use tiled VAE decoding
            tiled_vae_config: Configuration for tiled VAE

        Returns:
            Tuple of (video_tensor, audio_waveform), either may be None
        """
        from musubi_tuner.ltx_2.types import AudioLatentShape, VideoPixelShape

        # Setup CFG activation: enabled only when effective scale != 1.0.
        cfg_scale = config.cfg_scale if config.cfg_scale is not None else config.guidance_scale
        video_cfg_scale = config.video_cfg_scale if config.video_cfg_scale is not None else cfg_scale
        audio_cfg_scale = config.audio_cfg_scale if config.audio_cfg_scale is not None else cfg_scale
        do_cfg = float(video_cfg_scale) != 1.0 or float(audio_cfg_scale) != 1.0

        # Seed
        if config.seed is not None:
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed(config.seed)
            generator = torch.Generator(device=self.device).manual_seed(config.seed)
        else:
            generator = torch.Generator(device=self.device).manual_seed(torch.initial_seed())

        # Prepare prompt embeddings (handles dimension fixing internally)
        prompt_embeds, prompt_mask = self._prepare_prompt_embeds(config, do_cfg)

        # Calculate dimensions
        temporal_factor, spatial_factor = self._get_vae_factors()

        # For two-stage, generate at half resolution first
        if config.two_stage:
            gen_width = config.width // 2
            gen_height = config.height // 2
        else:
            gen_width = config.width
            gen_height = config.height

        # Align to VAE requirements
        gen_width = (gen_width // spatial_factor) * spatial_factor
        gen_height = (gen_height // spatial_factor) * spatial_factor
        frame_count = (config.frame_count - 1) // temporal_factor * temporal_factor + 1

        latent_frames = (frame_count - 1) // temporal_factor + 1
        latent_height = gen_height // spatial_factor
        latent_width = gen_width // spatial_factor
        in_channels = getattr(self.transformer, "in_channels", 128)

        # Initialize latents
        latents = self._init_latents(
            1, in_channels, latent_frames, latent_height, latent_width, generator
        )

        # I2V conditioning will be applied during denoising loop via denoise_mask

        # Initialize audio latents if needed
        audio_latents = None
        if config.enable_audio and self._audio_video:
            audio_cfg = config.extra.get("audio_config", {})
            if audio_cfg:
                video_shape = VideoPixelShape(
                    batch=1, frames=frame_count, height=gen_height, width=gen_width, fps=config.frame_rate
                )
                audio_shape = AudioLatentShape.from_video_pixel_shape(
                    video_shape,
                    channels=audio_cfg.get("channels", 8),
                    mel_bins=audio_cfg.get("mel_bins", 64),
                    sample_rate=audio_cfg.get("sample_rate", 24000),
                    hop_length=audio_cfg.get("hop_length", 160),
                    audio_latent_downsample_factor=audio_cfg.get("audio_latent_downsample_factor", 4),
                )
                audio_frames = max(int(audio_shape.frames), 1)
                audio_latents = torch.randn(
                    (1, audio_cfg.get("channels", 8), audio_frames, audio_cfg.get("mel_bins", 64)),
                    dtype=torch.float32,
                    device=self.device,
                    generator=generator,
                )

        # Stage 1: Main generation
        from musubi_tuner.ltx_2.components.schedulers import build_ltx2_sigmas
        sigmas = build_ltx2_sigmas(
            config.sample_steps,
            latent=latents,
            sigma_schedule=config.sigma_schedule,
            sampling_preset=config.sampling_preset,
        ).to(device=self.device, dtype=torch.float32)

        logger.info("Stage 1: Generating at %dx%d (%d frames, %d steps)",
                   gen_width, gen_height, frame_count, config.sample_steps)

        resolved_sampler = resolve_ltx2_sampler(config.sample_sampler, config.sampling_preset)
        if resolved_sampler == "res_2s":
            default_stage1_lora_multiplier = 0.25
            default_stage2_lora_multiplier = 0.5
        else:
            default_stage1_lora_multiplier = 0.0
            default_stage2_lora_multiplier = 1.0

        stage1_lora_multiplier = (
            default_stage1_lora_multiplier
            if config.stage1_distilled_lora_multiplier is None
            else float(config.stage1_distilled_lora_multiplier)
        )
        stage2_lora_multiplier = (
            default_stage2_lora_multiplier
            if config.stage2_distilled_lora_multiplier is None
            else float(config.stage2_distilled_lora_multiplier)
        )

        stage1_lora_applied = False
        if config.two_stage and config.distilled_lora_path and stage1_lora_multiplier != 0.0:
            if self._distilled_lora_state is None:
                self.load_distilled_lora(config.distilled_lora_path)
            logger.info("Applying stage-1 distilled LoRA multiplier %.2f", stage1_lora_multiplier)
            self._apply_distilled_lora(stage1_lora_multiplier)
            stage1_lora_applied = True

        with torch.no_grad():
            latents, audio_latents = self._denoise_loop(
                latents, sigmas, prompt_embeds, prompt_mask,
                do_cfg, cfg_scale, config.frame_rate,
                audio_latents=audio_latents,
                audio_only=config.audio_only,
                progress_desc="Stage 1" if config.two_stage else "LTX-2 inference",
                conditioning_latent=config.conditioning_latent,
                use_i2v_token_timestep_mask=bool(config.use_i2v_token_timestep_mask),
                latent_idx_guides=config.latent_idx_guides,
                keyframe_guides=config.keyframe_guides,
                stg_scale=config.stg_scale,
                stg_blocks=config.stg_blocks,
                stg_mode=config.stg_mode,
                rescale_scale=config.rescale_scale,
                video_cfg_scale=video_cfg_scale,
                audio_cfg_scale=audio_cfg_scale,
                video_rescale_scale=config.video_rescale_scale,
                audio_rescale_scale=config.audio_rescale_scale,
                video_modality_scale=config.video_modality_scale,
                audio_modality_scale=config.audio_modality_scale,
                sample_sampler=config.sample_sampler,
            )

        # Stage 2: Upsample and refine (if two-stage)
        if config.two_stage:
            if stage1_lora_applied:
                self._remove_distilled_lora(stage1_lora_multiplier)

            logger.info("Stage 2: Upsampling and refining to %dx%d", config.width, config.height)

            # Load upsampler if needed
            if self._spatial_upsampler is None and config.spatial_upsampler_path:
                self.load_spatial_upsampler(config.spatial_upsampler_path)

            if self._spatial_upsampler is None:
                raise ValueError("Spatial upsampler required for two-stage inference")

            # Optionally offload transformer to CPU while upsampling (saves VRAM)
            transformer_was_offloaded = False
            if config.offload_between_stages:
                if hasattr(self.transformer, "move_to_device_except_swap_blocks"):
                    logger.info("Offloading transformer for upsampling")
                    self.transformer.move_to_device_except_swap_blocks(torch.device("cpu"))
                    transformer_was_offloaded = True
                elif hasattr(self.transformer, "to"):
                    logger.info("Offloading transformer for upsampling")
                    self.transformer.to("cpu")
                    transformer_was_offloaded = True
                if transformer_was_offloaded:
                    clean_memory_on_device(self.device)

            # Upsample latents
            self._spatial_upsampler.to(self.device)
            with torch.no_grad():
                latents = self._upsample_latents(latents, self._spatial_upsampler)
            self._spatial_upsampler.to("cpu")
            clean_memory_on_device(self.device)

            # Restore transformer for stage 2
            if transformer_was_offloaded:
                logger.info("Restoring transformer for stage 2")
                if hasattr(self.transformer, "move_to_device_except_swap_blocks"):
                    self.transformer.move_to_device_except_swap_blocks(self.device)
                else:
                    self.transformer.to(self.device)

            # Apply distilled LoRA for stage 2
            if config.distilled_lora_path:
                if self._distilled_lora_state is None:
                    self.load_distilled_lora(config.distilled_lora_path)
                if stage2_lora_multiplier != 0.0:
                    logger.info("Applying stage-2 distilled LoRA multiplier %.2f", stage2_lora_multiplier)
                    self._apply_distilled_lora(stage2_lora_multiplier)

            # Stage 2 denoising with distilled sigmas
            stage2_sigmas = torch.tensor(
                STAGE_2_DISTILLED_SIGMA_VALUES[:config.stage2_steps + 1],
                device=self.device,
                dtype=torch.float32,
            )

            # Prepare stage 2 prompt (no CFG needed for distilled)
            stage2_embeds = config.prompt_embeds
            if stage2_embeds.dim() == 2:
                stage2_embeds = stage2_embeds.unsqueeze(0)

            # Fix embedding dimensions if needed (same as stage 1)
            expected_dim = self._get_expected_embed_dim()
            current_dim = stage2_embeds.shape[-1]
            if expected_dim is not None and current_dim != expected_dim:
                if current_dim * 2 == expected_dim:
                    # Pad video-only to AV
                    padding = torch.zeros(
                        *stage2_embeds.shape[:-1], current_dim,
                        dtype=stage2_embeds.dtype, device=stage2_embeds.device
                    )
                    stage2_embeds = torch.cat([stage2_embeds, padding], dim=-1)
                elif current_dim == expected_dim * 2:
                    # Slice AV to video-only
                    stage2_embeds = stage2_embeds[..., :expected_dim]

            stage2_embeds = stage2_embeds.to(device=self.device, dtype=self.dit_dtype)

            stage2_mask = config.prompt_attention_mask
            if stage2_mask is not None:
                if stage2_mask.dim() == 1:
                    stage2_mask = stage2_mask.unsqueeze(0)
                stage2_mask = stage2_mask.to(device=self.device, dtype=torch.int64)

            # Add noise at stage 2 starting sigma using flow matching formula:
            # noisy = (1 - sigma) * x0 + sigma * noise
            sigma = stage2_sigmas[0].item()
            video_noise = torch.randn(
                latents.shape, dtype=latents.dtype, device=latents.device, generator=generator
            )
            latents = (1.0 - sigma) * latents + sigma * video_noise

            # Also add noise to audio latents if present.
            if audio_latents is not None:
                audio_noise = torch.randn(
                    audio_latents.shape, dtype=audio_latents.dtype, device=audio_latents.device, generator=generator
                )
                audio_latents = (1.0 - sigma) * audio_latents + sigma * audio_noise

            with torch.no_grad():
                latents, audio_latents = self._denoise_loop(
                    latents, stage2_sigmas, stage2_embeds, stage2_mask,
                    do_cfg=False, cfg_scale=1.0, frame_rate=config.frame_rate,
                    audio_latents=audio_latents,
                    audio_only=config.audio_only,
                    progress_desc="Stage 2 refine",
                    conditioning_latent=config.conditioning_latent,
                    use_i2v_token_timestep_mask=bool(config.use_i2v_token_timestep_mask),
                    latent_idx_guides=config.latent_idx_guides,
                    keyframe_guides=config.keyframe_guides,
                    stg_scale=config.stg_scale,
                    stg_blocks=config.stg_blocks,
                    stg_mode=config.stg_mode,
                    video_cfg_scale=1.0,
                    audio_cfg_scale=1.0,
                    video_rescale_scale=0.0,
                    audio_rescale_scale=0.0,
                    video_modality_scale=1.0,
                    audio_modality_scale=1.0,
                    sample_sampler=config.sample_sampler,
                )

            # Remove distilled LoRA
            if (
                config.distilled_lora_path
                and self._distilled_lora_state is not None
                and stage2_lora_multiplier != 0.0
            ):
                self._remove_distilled_lora(stage2_lora_multiplier)

        # Decode video
        video = None
        if decode_video and not config.audio_only:
            self.vae.to_device(self.device)
            with torch.no_grad():
                if use_tiled_vae and tiled_vae_config:
                    from musubi_tuner.ltx_2.model.video_vae import TilingConfig, SpatialTilingConfig, TemporalTilingConfig

                    tile_cfg = TilingConfig(
                        spatial_config=SpatialTilingConfig(
                            tile_size_in_pixels=tiled_vae_config.get("tile_size", 512),
                            tile_overlap_in_pixels=tiled_vae_config.get("tile_overlap", 64),
                        ),
                        temporal_config=TemporalTilingConfig(
                            tile_size_in_frames=tiled_vae_config.get("temporal_tile_size", 9999),
                            tile_overlap_in_frames=tiled_vae_config.get("temporal_tile_overlap", 0),
                        ),
                    )
                    video = self.vae.tiled_decode(latents.squeeze(0), tile_cfg)
                else:
                    video = self.vae.decode([latents.squeeze(0)])
                    if isinstance(video, list) and video:
                        video = video[0]

                if video is not None:
                    if video.dim() == 4:
                        video = video.unsqueeze(0)
                    video = (video / 2 + 0.5).clamp(0, 1).to(torch.float32).cpu()

        # Decode audio
        audio_waveform = None
        if audio_latents is not None and audio_decoder is not None and vocoder is not None:
            audio_decoder.to(self.device)
            vocoder.to(self.device)
            with torch.no_grad():
                first_param = next(audio_decoder.parameters(), None)
                decode_dtype = first_param.dtype if first_param is not None else audio_latents.dtype
                audio_latents = audio_latents.to(device=self.device, dtype=decode_dtype)
                decoded_audio = audio_decoder(audio_latents)
                audio_waveform = vocoder(decoded_audio).squeeze(0).float().cpu()
            audio_decoder.to("cpu")
            vocoder.to("cpu")

        return video, audio_waveform


# ============ Utility Functions ============


def save_audio_wav(path: str, audio: torch.Tensor, sample_rate: int) -> None:
    """Save audio tensor to WAV file."""
    audio = audio.detach().cpu().float()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    if audio.shape[0] > 2:
        audio = audio[:2, :]

    audio_int16 = (audio.clamp(-1, 1) * 32767.0).to(torch.int16)
    interleaved = audio_int16.t().contiguous().numpy().tobytes()

    with wave.open(path, "wb") as wav:
        wav.setnchannels(audio_int16.shape[0])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(interleaved)


def mux_video_audio(video_path: str, audio_path: str, output_path: str) -> None:
    """Mux video and audio files into a single file."""
    if not os.path.exists(video_path) or not os.path.exists(audio_path):
        return

    try:
        import av
        import numpy as np
    except Exception as exc:
        logger.warning("Unable to mux audio/video (PyAV missing?): %s", exc)
        return

    with wave.open(audio_path, "rb") as wav_in:
        sample_rate = wav_in.getframerate()
        channels = wav_in.getnchannels()
        frames = wav_in.readframes(wav_in.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)
    else:
        audio = audio.reshape(-1, 1)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]

    container_in = av.open(video_path)
    video_stream_in = next((s for s in container_in.streams if s.type == "video"), None)
    if video_stream_in is None:
        container_in.close()
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    container_out = av.open(output_path, mode="w")
    video_stream_out = container_out.add_stream(
        "libx264",
        rate=video_stream_in.average_rate or video_stream_in.base_rate or 24,
    )
    video_stream_out.width = video_stream_in.width
    video_stream_out.height = video_stream_in.height
    video_stream_out.pix_fmt = "yuv420p"

    audio_stream = container_out.add_stream("aac", rate=sample_rate)
    audio_stream.codec_context.sample_rate = sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, sample_rate)

    for frame in container_in.decode(video_stream_in):
        for packet in video_stream_out.encode(frame):
            container_out.mux(packet)
    for packet in video_stream_out.encode():
        container_out.mux(packet)

    # PyAV s16p (planar) expects shape (channels, samples), must be C-contiguous
    audio_planar = np.ascontiguousarray(audio.T)
    audio_frame = av.AudioFrame.from_ndarray(audio_planar, format="s16p", layout="stereo")
    audio_frame.sample_rate = sample_rate
    for packet in audio_stream.encode(audio_frame):
        container_out.mux(packet)
    for packet in audio_stream.encode():
        container_out.mux(packet)

    container_out.close()
    container_in.close()


def cleanup_cuda(device: torch.device) -> None:
    """Clean up CUDA memory."""
    clean_memory_on_device(device)
    if device.type == "cuda":
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()
