"""Explicit sigma-bin targets for MiniMax H3 adapter reamplification."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from musubi_tuner.minimax_h3.training import H3ModelPrediction

REAMPLIFY_AUTO = "auto"


def parse_reamplify_schedule(value: str | None) -> tuple[float, ...] | str | None:
    """Parse one or more explicit field scales; an empty value disables the objective."""
    if value is None or not str(value).strip():
        return None
    if str(value).strip().lower() == "auto":
        return REAMPLIFY_AUTO
    try:
        schedule = tuple(float(item.strip()) for item in str(value).split(","))
    except ValueError as error:
        raise ValueError("--h3_reamplify_scale requires comma-separated finite factors greater than 1") from error
    if not schedule or any(not math.isfinite(item) or item <= 1.0 or item > torch.finfo(torch.float32).max for item in schedule):
        raise ValueError("--h3_reamplify_scale factors must be finite in FP32 and greater than 1")
    return schedule


def resolve_reamplify_schedule(ratios: dict[int, float], expected_bins: int) -> tuple[float, ...]:
    """Resolve a complete validation field-ratio grid into fixed reciprocal factors."""
    if type(expected_bins) is not int or expected_bins < 1:
        raise ValueError("automatic reamplification requires a positive validation-bin count")
    if not isinstance(ratios, dict) or any(type(index) is not int for index in ratios) or set(ratios) != set(range(expected_bins)):
        raise ValueError("automatic reamplification requires every contiguous validation field bin")
    values = []
    for index in range(expected_bins):
        ratio = ratios[index]
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("automatic reamplification field ratios must be finite and positive")
        factor = max(1.01, 1.0 / float(ratio))
        if not math.isfinite(factor) or factor > torch.finfo(torch.float32).max:
            raise ValueError("automatic reamplification factors must be finite in FP32")
        values.append(factor)
    return tuple(values)


def reamplify_scale(schedule: Sequence[float], base_sigma: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Select an equal-width base-sigma bin and broadcast it over ``reference``."""
    if not schedule:
        raise ValueError("reamplification requires at least one scale")
    values = torch.tensor(tuple(schedule), device=base_sigma.device, dtype=torch.float32)
    bins = torch.floor(base_sigma.float() * len(schedule)).to(torch.long).clamp_(0, len(schedule) - 1)
    selected = values[bins]
    while selected.ndim < reference.ndim:
        selected = selected.unsqueeze(-1)
    return selected


def reamplify_target(
    stock_empty: H3ModelPrediction,
    snapshot_prompt: H3ModelPrediction,
    snapshot_empty: H3ModelPrediction,
    schedule: Sequence[float],
    base_sigma: torch.Tensor,
) -> H3ModelPrediction:
    """Build ``stock_empty + scale * (snapshot_prompt - snapshot_empty)`` in FP32."""
    outputs: dict[str, torch.Tensor | None] = {}
    for modality in ("video", "audio"):
        stock = getattr(stock_empty, modality)
        prompt = getattr(snapshot_prompt, modality)
        empty = getattr(snapshot_empty, modality)
        if (stock is None) != (prompt is None) or (stock is None) != (empty is None):
            raise ValueError("reamplification branches must contain the same modalities")
        if stock is None:
            outputs[modality] = None
            continue
        if stock.shape != prompt.shape or stock.shape != empty.shape:
            raise ValueError(f"reamplification {modality} branches must have matching shapes")
        scale = reamplify_scale(schedule, base_sigma, stock)
        outputs[modality] = (stock.float() + scale * (prompt.float() - empty.float())).to(stock.dtype).detach()
    return H3ModelPrediction(video=outputs["video"], audio=outputs["audio"])
