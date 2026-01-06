# LoRA module for LTXV2 video transformer (video-only and audio-video)

from __future__ import annotations

import ast
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


class OfficialLTXV2Wrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        patch_size: int = 1,
        audio_patch_size: int = 1,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal_audio: bool = True,
    ) -> None:
        super().__init__()
        self.model = model

        from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier

        self._video_patchifier = VideoLatentPatchifier(patch_size=patch_size)
        self._audio_patchifier = AudioPatchifier(
            patch_size=audio_patch_size,
            sample_rate=sample_rate,
            hop_length=hop_length,
            audio_latent_downsample_factor=audio_latent_downsample_factor,
            is_causal=is_causal_audio,
        )

    def __getattr__(self, name: str):
        # Delegate attribute lookup to the wrapped model for compatibility with
        # existing musubi code paths (e.g. `in_channels`, `caption_projection`,
        # `audio_patchify_proj`, etc.) and LoRA audio-video auto-detection.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(
        self,
        x,
        *,
        timestep,
        context,
        attention_mask=None,
        frame_rate: int = 25,
        transformer_options=None,
        **kwargs,
    ):
        from ltx_core.components.patchifiers import get_pixel_coords
        from ltx_core.guidance.perturbations import BatchedPerturbationConfig
        from ltx_core.model.transformer.modality import Modality
        from ltx_core.types import AudioLatentShape, SpatioTemporalScaleFactors, VideoLatentShape

        if isinstance(x, (list, tuple)):
            if len(x) != 2:
                raise ValueError("Expected x to be [video_latents, audio_latents] for AV mode")
            video_latents, audio_latents = x
        else:
            video_latents, audio_latents = x, None

        if not isinstance(video_latents, torch.Tensor) or video_latents.dim() != 5:
            raise ValueError(f"Expected video latents shape [B, C, F, H, W], got: {getattr(video_latents, 'shape', None)}")

        bsz, vch, vframes, vheight, vwidth = video_latents.shape

        if isinstance(timestep, torch.Tensor):
            ts = timestep
        else:
            ts = torch.tensor(timestep, device=video_latents.device, dtype=video_latents.dtype)
        if ts.dim() == 0:
            ts = ts.view(1)
        if ts.dim() == 2 and ts.shape[1] == 1:
            sigma = ts[:, 0]
        elif ts.dim() == 1:
            sigma = ts
        else:
            raise ValueError(f"Unexpected timestep shape: {tuple(ts.shape)}")

        video_tokens = self._video_patchifier.patchify(video_latents)
        video_seq_len = video_tokens.shape[1]
        video_timesteps = sigma.view(bsz, 1).expand(bsz, video_seq_len)

        latent_coords = self._video_patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                batch=bsz,
                channels=vch,
                frames=vframes,
                height=vheight,
                width=vwidth,
            ),
            device=video_latents.device,
        )
        video_positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=SpatioTemporalScaleFactors.default(),
            causal_fix=True,
        ).to(dtype=video_latents.dtype)
        video_positions[:, 0, ...] = video_positions[:, 0, ...] / float(frame_rate)

        video_context = context
        audio_context = context
        if audio_latents is not None and isinstance(context, torch.Tensor) and context.shape[-1] % 2 == 0:
            half = context.shape[-1] // 2
            video_context = context[..., :half]
            audio_context = context[..., half:]

        video_modality = Modality(
            enabled=True,
            latent=video_tokens,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_context,
            context_mask=attention_mask,
        )

        audio_modality = None
        audio_shape = None
        if audio_latents is not None:
            if not isinstance(audio_latents, torch.Tensor) or audio_latents.dim() != 4:
                raise ValueError(f"Expected audio latents shape [B, C, T, F], got: {getattr(audio_latents, 'shape', None)}")

            absz, ach, at, af = audio_latents.shape
            if absz != bsz:
                raise ValueError(f"Batch mismatch: video B={bsz}, audio B={absz}")

            audio_tokens = self._audio_patchifier.patchify(audio_latents)
            audio_seq_len = audio_tokens.shape[1]
            audio_timesteps = sigma.view(bsz, 1).expand(bsz, audio_seq_len)

            audio_shape = AudioLatentShape(batch=bsz, channels=ach, frames=at, mel_bins=af)
            audio_positions = self._audio_patchifier.get_patch_grid_bounds(audio_shape, device=audio_latents.device)

            audio_modality = Modality(
                enabled=True,
                latent=audio_tokens,
                timesteps=audio_timesteps,
                positions=audio_positions.to(dtype=audio_latents.dtype),
                context=audio_context,
                context_mask=attention_mask,
            )

        perturbations = BatchedPerturbationConfig.empty(bsz)
        video_pred_tokens, audio_pred_tokens = self.model(video_modality, audio_modality, perturbations)

        video_pred = self._video_patchifier.unpatchify(
            video_pred_tokens,
            output_shape=VideoLatentShape(
                batch=bsz,
                channels=vch,
                frames=vframes,
                height=vheight,
                width=vwidth,
            ),
        )

        if audio_latents is None:
            return video_pred

        audio_pred = self._audio_patchifier.unpatchify(audio_pred_tokens, output_shape=audio_shape)
        return [video_pred, audio_pred]


def load_official_ltxv2_transformer(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    audio_video: bool = False,
) -> nn.Module:
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer.model_configurator import (
        LTXModelConfigurator,
        LTXVideoOnlyModelConfigurator,
        LTXV_MODEL_COMFY_RENAMING_MAP,
    )

    configurator = LTXModelConfigurator if audio_video else LTXVideoOnlyModelConfigurator
    return SingleGPUModelBuilder(
        model_path=str(model_path),
        model_class_configurator=configurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    ).build(device=device, dtype=dtype)


def load_official_ltxv2_wrapper(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    audio_video: bool = False,
    patch_size: int = 1,
) -> nn.Module:
    model = load_official_ltxv2_transformer(model_path, device=device, dtype=dtype, audio_video=audio_video)
    return OfficialLTXV2Wrapper(model, patch_size=patch_size)


# LTXV2 target modules for LoRA
# Based on LTXVModel and LTXAVModel architecture from the patch
LTXV2_TARGET_REPLACE_MODULES = [
    "BasicAVTransformerBlock",
    "PixArtAlphaTextProjection",
]


def _build_exclude_patterns(raw_patterns: Optional[str], audio_video: bool = False) -> List[str]:
    """Build exclude patterns for LTXV2 LoRA

    Args:
        raw_patterns: User-specified exclude patterns (as string repr of list)
        audio_video: If True, add audio-specific exclusions for LTXAV model
    """
    if raw_patterns is None:
        patterns: List[str] = []
    else:
        patterns = ast.literal_eval(raw_patterns)
        if not isinstance(patterns, list):
            raise ValueError("exclude_patterns must evaluate to a list")

    # Exclude these patterns (don't apply LoRA to these)
    # Core exclusions for all LTXV2 models
    patterns.extend(
        [
            r".*(norm|norm_out|scale_shift_table|patchify_proj|proj_out).*",
            r".*patchifier.*",
            r".*adaln_single\.emb.*",  # Timestep embedder internals
        ]
    )

    if audio_video:
        # Additional exclusions for LTXAV audio-video models
        patterns.extend(
            [
                r".*audio_scale_shift_table.*",
                r".*audio_patchify_proj.*",
                r".*audio_proj_out.*",
                r".*audio_norm_out.*",
                r".*audio_adaln_single\.emb.*",
                r".*a_patchifier.*",
                r".*av_ca.*adaln_single\.emb.*",  # AV cross-attention adaln embedders
            ]
        )

    return patterns


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    """Create LTXV2 LoRA network

    Args:
        multiplier: LoRA multiplier
        network_dim: LoRA rank dimension
        network_alpha: LoRA alpha for scaling
        vae: VAE module (unused but required by interface)
        text_encoders: Text encoder modules (unused but required)
        unet: LTXV2 transformer model
        neuron_dropout: Dropout for LoRA neurons
        **kwargs: Additional arguments including:
            - audio_video: bool, if True use LTXAV-specific exclusions

    Returns:
        LoRANetwork for LTXV2
    """
    # Check if this is an audio-video model by inspecting the unet class name
    audio_video = kwargs.pop("audio_video", False)
    if not audio_video and unet is not None:
        # Auto-detect LTXAV model
        audio_video = unet.__class__.__name__ == "LTXAVModel" or hasattr(unet, "audio_patchify_proj")

    kwargs["exclude_patterns"] = _build_exclude_patterns(kwargs.get("exclude_patterns"), audio_video=audio_video)

    return lora.create_network(
        LTXV2_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    """Create LTXV2 LoRA network from saved weights

    Args:
        multiplier: LoRA multiplier
        weights_sd: State dict of saved LoRA weights
        text_encoders: Text encoder modules (optional)
        unet: LTXV2 transformer model (optional)
        for_inference: Whether loading for inference
        **kwargs: Additional arguments including:
            - audio_video: bool, if True use LTXAV-specific exclusions

    Returns:
        LoRANetwork with loaded weights
    """
    # Check if this is an audio-video model
    audio_video = kwargs.pop("audio_video", False)
    if not audio_video:
        # Auto-detect from weights - LTXAV has audio-specific keys
        audio_video = any("audio_" in k for k in weights_sd.keys())
        if not audio_video and unet is not None:
            audio_video = unet.__class__.__name__ == "LTXAVModel" or hasattr(unet, "audio_patchify_proj")

    kwargs["exclude_patterns"] = _build_exclude_patterns(kwargs.get("exclude_patterns"), audio_video=audio_video)

    return lora.create_network_from_weights(
        LTXV2_TARGET_REPLACE_MODULES,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        **kwargs,
    )
