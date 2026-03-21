from __future__ import annotations

import torch


def update_audio_presence_ema(audio_presence_ema: float, balance_beta: float, has_audio_loss: bool) -> float:
    """Update EMA for audio-batch frequency."""
    audio_presence = 1.0 if has_audio_loss else 0.0
    ema = (1.0 - balance_beta) * float(audio_presence_ema) + balance_beta * audio_presence
    return min(max(ema, 0.0), 1.0)


def compute_inverse_frequency_audio_weight(
    base_audio_weight: float,
    audio_presence_ema: float,
    balance_eps: float,
    balance_min: float,
    balance_max: float,
) -> float:
    """Compute inverse-frequency-scaled and clamped audio loss weight."""
    denom = max(float(audio_presence_ema), float(balance_eps))
    weight = float(base_audio_weight) / denom
    return min(max(weight, float(balance_min)), float(balance_max))


def update_loss_ema(loss_ema: float, loss_value: float, ema_decay: float) -> float:
    """Update EMA for scalar loss values."""
    decay = float(ema_decay)
    value = float(loss_value)
    ema = decay * float(loss_ema) + (1.0 - decay) * value
    return max(ema, 1e-12)


def compute_uncertainty_weighted_loss(
    video_loss: torch.Tensor,
    audio_loss: torch.Tensor,
    log_var_video: torch.Tensor,
    log_var_audio: torch.Tensor,
) -> torch.Tensor:
    """Compute combined loss using homoscedastic uncertainty weighting.

    Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses
    for Scene Geometry and Semantics", CVPR 2018.

    loss = 0.5 * exp(-log_var_v) * L_v + 0.5 * log_var_v
         + 0.5 * exp(-log_var_a) * L_a + 0.5 * log_var_a

    The log_var regularization terms prevent the model from zeroing out
    either loss by making its variance arbitrarily large.
    """
    precision_v = torch.exp(-log_var_video)
    precision_a = torch.exp(-log_var_audio)
    loss = (
        0.5 * precision_v * video_loss + 0.5 * log_var_video
        + 0.5 * precision_a * audio_loss + 0.5 * log_var_audio
    )
    return loss


def compute_ema_magnitude_audio_weight(
    base_audio_weight: float,
    audio_loss_ema: float,
    video_loss_ema: float,
    target_audio_ratio: float,
    balance_min: float,
    balance_max: float,
) -> float:
    """Scale audio weight to match a target audio/video loss magnitude ratio."""
    target_audio_loss = float(target_audio_ratio) * max(float(video_loss_ema), 1e-12)
    dynamic_scale = target_audio_loss / max(float(audio_loss_ema), 1e-12)
    weight = float(base_audio_weight) * dynamic_scale
    return min(max(weight, float(balance_min)), float(balance_max))

