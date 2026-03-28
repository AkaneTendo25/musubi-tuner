"""
VACE inference utilities for LTX-2.

Provides helpers to load a trained VACE model and inject hints
during the denoising loop. Can be used standalone or integrated
with ltx2_generate_video.py via transformer_options.

Example usage:
    from musubi_tuner.ltx_vace.vace_inference import VaceInferenceHelper

    vace = VaceInferenceHelper(
        vace_model_path="path/to/vace.safetensors",
        vace_layers="0,4,8,12,16,20,24,28,32,36,40,44",
        vace_scale=1.0,
        dim=4096,
    )
    vace.to(device="cuda", dtype=torch.bfloat16)

    # During denoising, inject into transformer_options:
    transformer_options = vace.prepare_transformer_options(
        control_latents=vace_context,  # (B, vace_in_dim, F, H, W)
        proxy_x=noisy_latents_patchified,  # (B, seq_len, dim)
        text_context=text_embeds,
        text_mask=text_mask,
        timesteps=timestep_emb,
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from safetensors.torch import load_file

from musubi_tuner.ltx_vace.vace_model import VaceLTXModel, AudioVaceLTXModel, DEFAULT_VACE_LAYERS
from musubi_tuner.ltx_vace.vace_control_encoder import patchify_vace_context

logger = logging.getLogger(__name__)


class VaceInferenceHelper:
    """Helper for VACE-conditioned inference with LTX-2."""

    def __init__(
        self,
        vace_model_path: str,
        vace_layers: str | tuple[int, ...] = DEFAULT_VACE_LAYERS,
        vace_scale: float = 1.0,
        dim: int = 4096,
        context_dim: int = 4096,
        vace_in_dim: int = 1280,
        audio_context_dim: int | None = None,
    ):
        if isinstance(vace_layers, str):
            vace_layers = tuple(int(x.strip()) for x in vace_layers.split(","))

        self.vace_scale = vace_scale

        self.model = VaceLTXModel(
            vace_layers=vace_layers,
            vace_in_dim=vace_in_dim,
            dim=dim,
            context_dim=context_dim,
            audio_context_dim=audio_context_dim,
        )

        # Load weights
        state_dict = load_file(vace_model_path)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        logger.info(
            "Loaded VACE model from %s (%d blocks, scale=%.2f)",
            vace_model_path, len(vace_layers), vace_scale,
        )

    def to(self, device: str | torch.device = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.model = self.model.to(device=device, dtype=dtype)
        return self

    def prepare_transformer_options(
        self,
        control_latents: torch.Tensor,
        patchifier=None,
    ) -> dict:
        """Patchify VACE control latents and return transformer_options dict entries.

        The VACE model must be attached to the LTXModel (as ``_vace_model``)
        for hints to be computed inside ``LTXModel.forward()``.  This helper
        only prepares the raw tokens and scale for that forward pass.

        Args:
            control_latents: VACE context tensor (B, vace_in_dim, F, H, W).
            patchifier: Optional VideoLatentPatchifier for correct spatial token folding.

        Returns:
            Dict with keys ``"vace_context"`` and ``"vace_scale"`` to merge
            into ``transformer_options``.
        """
        vace_tokens = patchify_vace_context(control_latents, patchifier=patchifier)
        return {
            "vace_context": vace_tokens,
            "vace_scale": self.vace_scale,
        }

    def attach_to_model(self, ltx_model) -> None:
        """Attach the loaded VACE model to an LTXModel for forward-pass integration."""
        ltx_model._vace_model = self.model


class AudioVaceInferenceHelper:
    """Helper for audio VACE-conditioned inference with LTX-2."""

    def __init__(
        self,
        audio_vace_model_path: str,
        vace_layers: str | tuple[int, ...] = DEFAULT_VACE_LAYERS,
        vace_scale: float = 1.0,
        dim: int = 2048,
        context_dim: int = 2048,
        num_heads: int = 32,
        d_head: int = 64,
        vace_in_dim: int = 257,
    ):
        if isinstance(vace_layers, str):
            vace_layers = tuple(int(x.strip()) for x in vace_layers.split(","))

        self.vace_scale = vace_scale

        self.model = AudioVaceLTXModel(
            vace_layers=vace_layers,
            vace_in_dim=vace_in_dim,
            dim=dim,
            num_heads=num_heads,
            d_head=d_head,
            context_dim=context_dim,
        )

        # Load weights
        state_dict = load_file(audio_vace_model_path)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        logger.info(
            "Loaded audio VACE model from %s (%d blocks, scale=%.2f)",
            audio_vace_model_path, len(vace_layers), vace_scale,
        )

    def to(self, device: str | torch.device = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.model = self.model.to(device=device, dtype=dtype)
        return self

    def prepare_transformer_options(
        self,
        audio_vace_context_tokens: torch.Tensor,
    ) -> dict:
        """Return transformer_options dict entries for audio VACE.

        The audio VACE model must be attached to the LTXModel (as
        ``_audio_vace_model``).  This helper passes the pre-built token
        context for ``LTXModel.forward()`` to compute hints.

        Args:
            audio_vace_context_tokens: Audio VACE context tokens (B, T, vace_in_dim).
                Already built via ``prepare_audio_vace_context()``.

        Returns:
            Dict with keys ``"audio_vace_context"`` and ``"audio_vace_scale"``
            to merge into ``transformer_options``.
        """
        return {
            "audio_vace_context": audio_vace_context_tokens,
            "audio_vace_scale": self.vace_scale,
        }

    def attach_to_model(self, ltx_model) -> None:
        """Attach the loaded audio VACE model to an LTXModel."""
        ltx_model._audio_vace_model = self.model
