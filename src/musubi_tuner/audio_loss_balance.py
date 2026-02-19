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

