from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from musubi_tuner.minimax_h3.architecture import AUDIO_FLOW_SHIFT, VIDEO_FLOW_SHIFT
from musubi_tuner.minimax_h3.training import LossBalance, shift_sigma

ValidationModality = Literal["video", "audio"]

_VIDEO = 0
_AUDIO = 1
_SUM = 0
_COUNT = 1


@contextmanager
def preserve_rng_state():
    """Prevent validation setup and forwards from advancing training RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def seed_validation_forward(seed: int) -> None:
    """Seed every RNG a stochastic H3 forward may consume."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_squared_error_sum(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Return the exact weighted squared-error numerator and valid count."""
    if prediction.shape != target.shape:
        raise ValueError("H3 validation prediction and target shapes differ")
    if mask is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        mask = mask.to(device=target.device, dtype=torch.bool)
        if mask.shape == target.shape:
            valid = mask
        else:
            if mask.ndim < 2 or mask.shape[0] != target.shape[0] or mask.shape[1:] != target.shape[-(mask.ndim - 1) :]:
                raise ValueError(f"H3 validation mask shape {tuple(mask.shape)} cannot broadcast to {tuple(target.shape)}")
            shape = (mask.shape[0], *([1] * (target.ndim - mask.ndim)), *mask.shape[1:])
            valid = mask.view(shape).expand_as(target)
    count = int(valid.sum().item())
    if count == 0:
        return prediction.new_zeros((), dtype=torch.float32), 0
    squared = (prediction - target).float().square()
    if sample_weight is not None:
        if sample_weight.shape != (target.shape[0],):
            raise ValueError("H3 validation sample weighting must contain one value per batch item")
        weight_shape = (target.shape[0], *([1] * (target.ndim - 1)))
        squared = squared * sample_weight.to(device=squared.device, dtype=torch.float32).view(weight_shape)
    return squared.masked_select(valid).sum(), count


@dataclass(frozen=True)
class H3ValidationSigmaBin:
    """One deterministic validation point on H3's synchronized schedules."""

    index: int
    base_sigma: float
    video_sigma: float
    audio_sigma: float


def validation_sigma_bins(
    count: int,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
    video_shift: float = VIDEO_FLOW_SHIFT,
    audio_shift: float = AUDIO_FLOW_SHIFT,
) -> tuple[H3ValidationSigmaBin, ...]:
    """Return midpoint bins in the shared, unshifted base-sigma coordinate."""
    if count <= 0:
        raise ValueError("H3 validation sigma-bin count must be positive")
    if not 0.0 <= minimum < maximum <= 1.0:
        raise ValueError("H3 validation sigma bounds must satisfy 0 <= minimum < maximum <= 1")
    if video_shift <= 0 or audio_shift <= 0:
        raise ValueError("H3 validation flow shifts must be positive")

    width = (maximum - minimum) / count
    base = torch.tensor([minimum + (index + 0.5) * width for index in range(count)], dtype=torch.float64)
    video = shift_sigma(base, video_shift)
    audio = shift_sigma(base, audio_shift)
    return tuple(
        H3ValidationSigmaBin(index, float(base[index]), float(video[index]), float(audio[index]))
        for index in range(count)
    )


def image_flow_shift(latent_height: int, latent_width: int) -> float:
    """Match the H3 image-training ``krea2_shift`` resolution schedule."""
    if latent_height <= 0 or latent_width <= 0:
        raise ValueError("H3 image latent dimensions must be positive")
    if latent_height % 2 or latent_width % 2:
        raise ValueError("H3 image latent dimensions must be divisible by the 1x2x2 DiT patch")
    token_count = (latent_height // 2) * (latent_width // 2)
    mu = 0.5 + (token_count - 256) * (1.15 - 0.5) / (6400 - 256)
    return math.exp(mu)


def image_validation_sigma(
    base_sigma: torch.Tensor,
    *,
    latent_height: int,
    latent_width: int,
    flow_shift: float | None = None,
) -> torch.Tensor:
    """Apply the explicit or resolution-aware image shift used by training."""
    shift = image_flow_shift(latent_height, latent_width) if flow_shift is None else flow_shift
    return shift_sigma(base_sigma, shift)


class H3ValidationAccumulator:
    """Accumulate exact error sums/counts for reduction across processes."""

    def __init__(
        self,
        bin_count: int,
        *,
        balance: LossBalance = "token",
        video_weight: float = 1.0,
        audio_weight: float = 1.0,
    ) -> None:
        if bin_count <= 0:
            raise ValueError("H3 validation bin count must be positive")
        if balance not in {"token", "modality"}:
            raise ValueError(f"unsupported H3 validation loss balance: {balance}")
        if video_weight < 0 or audio_weight < 0 or video_weight + audio_weight <= 0:
            raise ValueError("H3 validation modality weights must be non-negative and not both zero")
        self.balance = balance
        self.video_weight = float(video_weight)
        self.audio_weight = float(audio_weight)
        self._state = torch.zeros((bin_count, 2, 2), dtype=torch.float64)

    @property
    def bin_count(self) -> int:
        return self._state.shape[0]

    def add(self, bin_index: int, modality: ValidationModality, error_sum: torch.Tensor | float, element_count: int) -> None:
        if not 0 <= bin_index < self.bin_count:
            raise IndexError(f"H3 validation bin index {bin_index} is out of range")
        if modality not in {"video", "audio"}:
            raise ValueError(f"unsupported H3 validation modality: {modality}")
        if element_count < 0:
            raise ValueError("H3 validation element count must be non-negative")
        value = float(error_sum.detach().double().cpu()) if isinstance(error_sum, torch.Tensor) else float(error_sum)
        if element_count == 0:
            if value != 0.0:
                raise ValueError("H3 validation zero-element updates must have a zero error sum")
            return
        if not math.isfinite(value) or value < 0:
            raise ValueError("H3 validation error sum must be finite and non-negative")
        modality_index = _VIDEO if modality == "video" else _AUDIO
        self._state[bin_index, modality_index, _SUM] += value
        self._state[bin_index, modality_index, _COUNT] += element_count

    def reduction_tensor(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        """Return a standalone tensor suitable for distributed sum reduction."""
        return self._state.clone().to(device=device)

    def load_reduced_tensor(self, reduced: torch.Tensor) -> None:
        """Replace local state with an already sum-reduced tensor."""
        if reduced.shape != self._state.shape:
            raise ValueError(
                f"H3 validation reduction tensor must have shape {tuple(self._state.shape)}, got {tuple(reduced.shape)}"
            )
        reduced = reduced.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(reduced).all()) or bool((reduced < 0).any()):
            raise ValueError("H3 validation reduction tensor must contain finite non-negative values")
        counts = reduced[:, :, _COUNT]
        if not bool(torch.equal(counts, counts.round())):
            raise ValueError("H3 validation reduction counts must be integers")
        self._state.copy_(reduced)

    def _loss(self, state: torch.Tensor) -> float | None:
        sums = state[:, _SUM]
        counts = state[:, _COUNT]
        active = counts > 0
        weights = torch.tensor([self.video_weight, self.audio_weight], dtype=torch.float64)
        active_weights = weights * active
        if float(active_weights.sum()) == 0.0:
            return None
        if self.balance == "modality":
            means = torch.where(active, sums / counts.clamp_min(1), torch.zeros_like(sums))
            return float((means * active_weights).sum() / active_weights.sum())
        denominator = (counts * active_weights).sum()
        if float(denominator) == 0.0:
            return None
        return float((sums * active_weights).sum() / denominator)

    def metrics(self) -> dict[str, float]:
        """Finalize aggregate and per-bin metrics, omitting empty entries."""
        metrics: dict[str, float] = {}
        aggregate = self._state.sum(dim=0)
        loss = self._loss(aggregate)
        if loss is not None:
            metrics["loss"] = loss
        for name, modality_index in (("video", _VIDEO), ("audio", _AUDIO)):
            total = aggregate[modality_index]
            if float(total[_COUNT]) > 0:
                metrics[f"loss/{name}"] = float(total[_SUM] / total[_COUNT])
        for index, state in enumerate(self._state):
            bin_loss = self._loss(state)
            if bin_loss is not None:
                metrics[f"loss/bin_{index:02d}"] = bin_loss
        return metrics
