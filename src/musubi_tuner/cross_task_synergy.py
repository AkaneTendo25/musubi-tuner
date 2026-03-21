"""Cross-Task Synergy auxiliary losses for joint audio-video training.

Alongside the main joint denoising loss (both modalities noisy), computes
two uni-directional auxiliary losses where one modality is clean (timestep=0):

1. Audio-driven video: video noisy + audio clean → supervise video prediction
2. Video-driven audio: video clean + audio noisy → supervise audio prediction

This provides stable cross-modal alignment targets, addressing the
"correspondence drift" problem where concurrent noisy latents impede
alignment learning.

Reference: Harmony (2025), "Cross-Task Synergy training paradigm".
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_cross_task_synergy_losses(
    *,
    transformer: nn.Module,
    accelerator: Any,
    # Video inputs
    noisy_video: torch.Tensor,
    clean_video: torch.Tensor,
    video_target: torch.Tensor,
    video_timesteps: torch.Tensor,
    video_loss_mask: Optional[torch.Tensor],
    # Audio inputs
    noisy_audio: torch.Tensor,
    clean_audio: torch.Tensor,
    audio_target: torch.Tensor,
    audio_timesteps: Optional[torch.Tensor],
    audio_loss_mask: Optional[torch.Tensor],
    # Shared inputs
    text_embeds: torch.Tensor,
    text_mask: Optional[torch.Tensor],
    frame_rate: Any,
    transformer_options: Dict[str, Any],
    # Weights
    lambda_video_driven: float = 0.1,
    lambda_audio_driven: float = 0.3,
    # Loss function
    loss_fn: Any = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Compute Cross-Task Synergy auxiliary losses.

    Args:
        noisy_video: Noisy video latents (same as main forward pass)
        clean_video: Clean video latents (original, unnoised)
        video_target: Video velocity target (noise - latents)
        video_timesteps: Video timesteps for the noisy modality
        noisy_audio: Noisy audio latents
        clean_audio: Clean audio latents (original, unnoised)
        audio_target: Audio velocity target (noise - latents)
        audio_timesteps: Audio timesteps for the noisy modality
        audio_loss_mask: Audio loss mask
        text_embeds: Text conditioning
        text_mask: Text attention mask
        frame_rate: Frame rate
        transformer_options: Transformer options dict
        lambda_video_driven: Weight for video-driven audio loss (default: 0.3 per Harmony)
        lambda_audio_driven: Weight for audio-driven video loss (default: 0.1 per Harmony)
        loss_fn: Loss function (defaults to MSE)

    Returns:
        (loss, metrics) tuple. loss is None if both lambdas are 0.
    """
    if lambda_video_driven <= 0.0 and lambda_audio_driven <= 0.0:
        return None, {}

    if loss_fn is None:
        def loss_fn(pred, target):
            return torch.nn.functional.mse_loss(pred, target, reduction="none")

    metrics: Dict[str, float] = {}
    total_loss = torch.tensor(0.0, device=noisy_video.device)

    # Zero timesteps for clean modality
    zero_timestep = torch.zeros_like(video_timesteps[:, :1])

    # 1. Audio-driven video loss: clean audio + noisy video → supervise video
    if lambda_audio_driven > 0.0:
        with accelerator.autocast():
            pred = transformer(
                [noisy_video, clean_audio],
                timestep=video_timesteps,
                audio_timestep=zero_timestep,
                context=text_embeds,
                attention_mask=text_mask,
                frame_rate=frame_rate,
                transformer_options=transformer_options,
            )
        if isinstance(pred, (list, tuple)) and len(pred) >= 2:
            video_pred_driven = pred[0]
        else:
            video_pred_driven = pred

        driven_video_loss = loss_fn(video_pred_driven, video_target)
        if video_loss_mask is not None:
            mask = video_loss_mask.to(dtype=driven_video_loss.dtype, device=driven_video_loss.device)
            while mask.dim() < driven_video_loss.dim():
                mask = mask.unsqueeze(-1)
            denom = mask.mean()
            if denom.item() > 0:
                driven_video_loss = (driven_video_loss * mask).div(denom).mean()
            else:
                driven_video_loss = driven_video_loss.mean()
        else:
            driven_video_loss = driven_video_loss.mean()

        total_loss = total_loss + lambda_audio_driven * driven_video_loss
        metrics["loss/cts_audio_driven_video"] = driven_video_loss.detach().item()

    # 2. Video-driven audio loss: clean video + noisy audio → supervise audio
    if lambda_video_driven > 0.0:
        with accelerator.autocast():
            pred = transformer(
                [clean_video, noisy_audio],
                timestep=zero_timestep,
                audio_timestep=audio_timesteps,
                context=text_embeds,
                attention_mask=text_mask,
                frame_rate=frame_rate,
                transformer_options=transformer_options,
            )
        if isinstance(pred, (list, tuple)) and len(pred) >= 2:
            audio_pred_driven = pred[1]
        else:
            audio_pred_driven = None

        if audio_pred_driven is not None:
            driven_audio_loss = loss_fn(audio_pred_driven, audio_target)
            if audio_loss_mask is not None:
                mask = audio_loss_mask.to(dtype=driven_audio_loss.dtype, device=driven_audio_loss.device)
                while mask.dim() < driven_audio_loss.dim():
                    mask = mask.unsqueeze(-1)
                denom = mask.mean()
                if denom.item() > 0:
                    driven_audio_loss = (driven_audio_loss * mask).div(denom).mean()
                else:
                    driven_audio_loss = driven_audio_loss.mean()
            else:
                driven_audio_loss = driven_audio_loss.mean()

            total_loss = total_loss + lambda_video_driven * driven_audio_loss
            metrics["loss/cts_video_driven_audio"] = driven_audio_loss.detach().item()

    if not torch.isfinite(total_loss):
        logger.warning("Cross-Task Synergy loss is non-finite (%.4g), skipping", total_loss.item())
        return None, metrics

    metrics["loss/cross_task_synergy"] = total_loss.detach().item()
    return total_loss, metrics
