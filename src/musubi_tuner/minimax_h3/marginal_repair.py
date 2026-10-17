"""Training-only guidance reconstruction.

The released model supplies both frozen fields; this approximates a teacher and
does not recover MiniMax's unavailable original undistilled model.
"""

import math

import torch


def parse_scale_schedule(text: str) -> tuple[float, float]:
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except (ValueError, AttributeError) as error:
        raise ValueError("--h3_repair_scale must be constant or intercept,slope") from error
    if len(values) not in (1, 2) or not all(math.isfinite(value) for value in values):
        raise ValueError("--h3_repair_scale must contain one or two finite numbers")
    if (len(values) == 1 or values[1] == 0) and values[0] < 1:
        raise ValueError("constant --h3_repair_scale must be at least 1")
    return values[0], values[1] if len(values) == 2 else 0.0


def scale_at(schedule: tuple[float, float], sigma: torch.Tensor) -> torch.Tensor:
    return (schedule[0] + schedule[1] * sigma.float()).clamp_min(1.0)


def interpolate_schedule(values, sigma: torch.Tensor) -> torch.Tensor:
    """Linear at .40/.95 or piecewise linear at .40/.80/.95, clamped outside."""
    value = sigma.float()
    if len(values) == 2:
        return values[0] + (values[1] - values[0]) * ((value - 0.40) / 0.55).clamp(0, 1)
    low = values[0] + (values[1] - values[0]) * ((value - 0.40) / 0.40).clamp(0, 1)
    high = values[1] + (values[2] - values[1]) * ((value - 0.80) / 0.15).clamp(0, 1)
    return torch.where(value <= 0.80, low, high)


def guidance_recovered_target(prompted, reference, scale, multiplier=None):
    if prompted.shape != reference.shape:
        raise ValueError("repair prompted and reference predictions must have matching shapes")
    scale = torch.as_tensor(scale, device=prompted.device, dtype=torch.float32)
    if not bool(torch.isfinite(scale).all()) or bool((scale < 1).any()):
        raise ValueError("repair scale must be finite and at least 1")
    if scale.ndim and scale.numel() != prompted.shape[0]:
        raise ValueError("repair scale must have one value per sample")
    fraction = 1 - scale.reciprocal()
    if multiplier is not None:
        fraction = fraction * multiplier.to(device=prompted.device, dtype=torch.float32)
        if not bool(torch.isfinite(fraction).all()) or bool((fraction > 1 + 1e-7).any()):
            raise ValueError("--h3_repair_target_multiplier extrapolates past the reference branch")
    fraction = fraction.reshape(-1, *([1] * (prompted.ndim - 1)))
    return (prompted.detach().float() + fraction * (reference.detach().float() - prompted.detach().float())).to(prompted.dtype)
