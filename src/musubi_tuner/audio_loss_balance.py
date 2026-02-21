from __future__ import annotations


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

