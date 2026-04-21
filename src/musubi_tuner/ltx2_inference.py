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

from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)

# Stage 2 distilled sigma values (from LTX-2 official pipeline)
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
    sample_steps: int = 40  # Official default: 40 steps
    guidance_scale: float = 1.0
    cfg_scale: Optional[float] = 4.0  # Official default: 4.0 CFG
    discrete_flow_shift: float = 5.0
    seed: Optional[int] = None

    # Two-stage inference
    two_stage: bool = False
    spatial_upsampler_path: Optional[str] = None
    distilled_lora_path: Optional[str] = None
    stage2_steps: int = 3  # Stage 2 uses 3 steps (4 sigma values including 0.0)

    # Offloading
    offload_between_stages: bool = False

    # STG (Spatio-Temporal Guidance) — opt-in, inert when stg_scale == 0.0
    stg_scale: float = 0.0
    stg_blocks: Optional[List[int]] = None  # None = all blocks
    stg_mode: str = "video"  # "video" | "audio" | "both"

    # CFG★ rescaling (official pipeline uses 0.7 for LTX-2.3). 0.0 disables.
    rescale_scale: float = 0.0

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

    # Extra options
    extra: Dict[str, Any] = field(default_factory=dict)


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
        self._distilled_lora_state = load_file(lora_path)
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

    def _get_expected_embed_dim(self) -> int:
        """Get expected embedding dimension based on mode."""
        # Known dimensions for LTX-2
        VIDEO_ONLY_DIM = 1920
        AV_DIM = 3840
        return AV_DIM if self._audio_video else VIDEO_ONLY_DIM

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

        if current_dim != expected_dim:
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
                    "falling back to duplicated positive embeddings (deviates from official behavior)."
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
        stg_scale: float = 0.0,
        stg_blocks: Optional[List[int]] = None,
        stg_mode: str = "video",
        rescale_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the denoising loop with optional I2V conditioning.

        Args:
            conditioning_latent: Optional first-frame conditioning latent [B, C, 1, H, W].
                                If provided, first frame will be locked during denoising.
        """
        from musubi_tuner.ltx_2.model.ltx2_scheduler import EulerDiffusionStep, X0PredictionWrapper

        stepper = EulerDiffusionStep()

        # Setup I2V conditioning mask if provided
        denoise_mask = None
        clean_latent = None
        i2v_conditioning_mask_tokens = None
        if conditioning_latent is not None:
            try:
                # Validate conditioning_latent shape
                if conditioning_latent.dim() != 5:
                    logger.warning(
                        "I2V: conditioning_latent has wrong dimensions %s, expected [B,C,T,H,W]. Skipping I2V conditioning.",
                        tuple(conditioning_latent.shape),
                    )
                elif latents.shape[2] < 1:
                    logger.warning("I2V: Video latents have no temporal frames. Skipping I2V conditioning.")
                else:
                    cond_on_device = conditioning_latent.to(device=latents.device, dtype=latents.dtype)

                    if cond_on_device.shape[2] != 1:
                        logger.warning(
                            "I2V: conditioning_latent has %s frames, expected 1. Skipping I2V conditioning.",
                            cond_on_device.shape[2],
                        )
                        cond_on_device = None
                    elif cond_on_device.shape[1] != latents.shape[1]:
                        logger.warning(
                            "I2V: Channel dimension mismatch - conditioning %s vs latents %s. Skipping I2V conditioning.",
                            cond_on_device.shape[1],
                            latents.shape[1],
                        )
                        cond_on_device = None

                    # Two-stage stage-1 uses half-res latents. Resize conditioning latent if needed.
                    if cond_on_device is not None and cond_on_device.shape[-2:] != latents.shape[-2:]:
                        resized = F.interpolate(
                            cond_on_device.squeeze(2),
                            size=latents.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).unsqueeze(2)
                        logger.info(
                            "I2V: resized conditioning latent from %s to %s for current denoising stage",
                            tuple(cond_on_device.shape),
                            tuple(resized.shape),
                        )
                        cond_on_device = resized

                    if cond_on_device is not None and cond_on_device.shape[-2:] != latents.shape[-2:]:
                        logger.warning(
                            "I2V: Spatial dimension mismatch after resize attempt - conditioning %s vs latents %s. Skipping I2V conditioning.",
                            tuple(cond_on_device.shape[-2:]),
                            tuple(latents.shape[-2:]),
                        )
                        cond_on_device = None

                    if cond_on_device is not None:
                        # Initialize first frame with conditioning latent.
                        latents[:, :, 0:1, :, :] = cond_on_device

                        # Keep first frame locked across denoising.
                        denoise_mask = torch.ones_like(latents)
                        denoise_mask[:, :, 0:1, :, :] = 0.0

                        clean_latent = torch.zeros_like(latents)
                        clean_latent[:, :, 0:1, :, :] = cond_on_device

                        if use_i2v_token_timestep_mask:
                            bsz, _c, frames, h_lat, w_lat = latents.shape
                            seq_len = frames * h_lat * w_lat
                            first_frame_tokens = h_lat * w_lat
                            i2v_conditioning_mask_tokens = torch.zeros(
                                (bsz, seq_len),
                                device=latents.device,
                                dtype=torch.bool,
                            )
                            if first_frame_tokens > 0:
                                i2v_conditioning_mask_tokens[:, :first_frame_tokens] = True
                            logger.info("I2V: enabled token timestep mask for conditioned first-frame tokens")

                        logger.info(f"I2V: Initialized first frame conditioning (shape: {cond_on_device.shape})")
            except Exception as e:
                logger.error(f"I2V: Failed to setup conditioning: {e}", exc_info=True)
                denoise_mask = None
                clean_latent = None
                i2v_conditioning_mask_tokens = None

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
            # This matches the official LTX-2 pipeline where X0Model wraps velocity model
            # and CFG is applied to denoised (x0) outputs, not velocity predictions
            video_pred = video_pred.to(dtype=latents.dtype)

            sigma_for_video = denoise_mask * sigma if denoise_mask is not None else sigma

            if do_cfg:
                # Split velocity predictions for CFG
                vel_uncond, vel_cond = video_pred.chunk(2)
                # Convert each to x0
                x0_uncond = X0PredictionWrapper.velocity_to_x0(latents, vel_uncond, sigma_for_video)
                x0_cond = X0PredictionWrapper.velocity_to_x0(latents, vel_cond, sigma_for_video)
                # Apply CFG to x0 (official formula)
                video_x0 = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
            else:
                x0_cond = X0PredictionWrapper.velocity_to_x0(latents, video_pred, sigma_for_video)
                video_x0 = x0_cond

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
            # prevent oversaturation from amplified guidance. Matches official MultiModalGuider.
            if rescale_scale > 0.0 and x0_cond is not None:
                pred_std = video_x0.std()
                if pred_std > 1e-6:
                    factor = x0_cond.std() / pred_std
                    factor = rescale_scale * factor + (1.0 - rescale_scale)
                    video_x0 = video_x0 * factor

            if denoise_mask is not None and clean_latent is not None:
                # Official LTX-2 ordering: blend denoised x0 before Euler step.
                video_x0 = video_x0 * denoise_mask + clean_latent * (1.0 - denoise_mask)

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
                    audio_x0 = aud_x0_uncond + cfg_scale * (aud_x0_cond - aud_x0_uncond)
                else:
                    aud_x0_cond = X0PredictionWrapper.velocity_to_x0(audio_latents, audio_pred, sigma.item())
                    audio_x0 = aud_x0_cond
                if x0_audio_ptb is not None:
                    audio_x0 = audio_x0 + stg_scale * (aud_x0_cond - x0_audio_ptb)
                if rescale_scale > 0.0:
                    pred_std = audio_x0.std()
                    if pred_std > 1e-6:
                        factor = aud_x0_cond.std() / pred_std
                        factor = rescale_scale * factor + (1.0 - rescale_scale)
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

        # Setup (official-style CFG activation: enabled only when effective scale != 1.0)
        cfg_scale = config.cfg_scale if config.cfg_scale is not None else config.guidance_scale
        do_cfg = float(cfg_scale) != 1.0

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
        # Official pipeline does NOT pass latent to scheduler - uses default MAX_SHIFT_ANCHOR=4096
        from musubi_tuner.ltx_2.components.schedulers import LTX2Scheduler
        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=config.sample_steps).to(device=self.device, dtype=torch.float32)

        logger.info("Stage 1: Generating at %dx%d (%d frames, %d steps)",
                   gen_width, gen_height, frame_count, config.sample_steps)

        with torch.no_grad():
            latents, audio_latents = self._denoise_loop(
                latents, sigmas, prompt_embeds, prompt_mask,
                do_cfg, cfg_scale, config.frame_rate,
                audio_latents=audio_latents,
                audio_only=config.audio_only,
                progress_desc="Stage 1" if config.two_stage else "LTX-2 inference",
                conditioning_latent=config.conditioning_latent,
                use_i2v_token_timestep_mask=bool(config.use_i2v_token_timestep_mask),
                stg_scale=config.stg_scale,
                stg_blocks=config.stg_blocks,
                stg_mode=config.stg_mode,
                rescale_scale=config.rescale_scale,
            )

        # Stage 2: Upsample and refine (if two-stage)
        if config.two_stage:
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
                self._apply_distilled_lora()

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

            # Also add noise to audio latents if present (official pipeline does this)
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
                    stg_scale=config.stg_scale,
                    stg_blocks=config.stg_blocks,
                    stg_mode=config.stg_mode,
                )

            # Remove distilled LoRA
            if config.distilled_lora_path and self._distilled_lora_state is not None:
                self._remove_distilled_lora()

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
