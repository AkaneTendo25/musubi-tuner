from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from musubi_tuner.minimax_h3.architecture import AUDIO_FLOW_SHIFT, VIDEO_FLOW_SHIFT

LossBalance = Literal["token", "modality"]
H3TrainingMode = Literal["fl2va", "ref2va"]


@dataclass(frozen=True)
class H3JointNoisyInputs:
    video: torch.Tensor | None
    audio: torch.Tensor | None
    video_target: torch.Tensor | None
    audio_target: torch.Tensor | None
    video_sigma: torch.Tensor
    audio_sigma: torch.Tensor
    video_timestep: torch.Tensor
    audio_timestep: torch.Tensor


@dataclass(frozen=True)
class H3ModelPrediction:
    video: torch.Tensor | None
    audio: torch.Tensor | None


@dataclass(frozen=True)
class H3JointLoss:
    loss: torch.Tensor
    video_loss: torch.Tensor
    audio_loss: torch.Tensor
    video_elements: int
    audio_elements: int


def _validate_sigma(sigma: torch.Tensor) -> None:
    if not sigma.is_floating_point():
        raise TypeError("H3 sigma must be floating point")
    if sigma.ndim != 1:
        raise ValueError(f"H3 sigma must have shape [batch], got {tuple(sigma.shape)}")
    if bool(((sigma < 0) | (sigma > 1)).any()):
        raise ValueError("H3 sigma values must be in [0, 1]")


def _shift_unchecked(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply H3's exponential flow shift to an unshifted noise level."""
    if shift <= 0:
        raise ValueError("H3 flow shift must be positive")
    _validate_sigma(sigma)
    return _shift_unchecked(sigma, shift)


def unshift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """Recover the unshifted noise level from an H3 shifted sigma."""
    if shift <= 0:
        raise ValueError("H3 flow shift must be positive")
    _validate_sigma(sigma)
    return sigma / (shift - (shift - 1.0) * sigma)


def map_sigma_between_shifts(sigma: torch.Tensor, *, source_shift: float, target_shift: float) -> torch.Tensor:
    """Map one modality's sigma to the synchronized point on another H3 schedule."""
    return shift_sigma(unshift_sigma(sigma, source_shift), target_shift)


def _expand_batch_values(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if values.shape != (target.shape[0],):
        raise ValueError(f"expected one value per batch item, got {tuple(values.shape)} for batch {target.shape[0]}")
    return values.to(device=target.device, dtype=target.dtype).view(target.shape[0], *([1] * (target.ndim - 1)))


def prepare_joint_noisy_inputs(
    video_latents: torch.Tensor | None,
    audio_latents: torch.Tensor | None,
    video_noise: torch.Tensor | None,
    audio_noise: torch.Tensor | None,
    base_sigma: torch.Tensor,
    *,
    video_shift: float = VIDEO_FLOW_SHIFT,
    audio_shift: float = AUDIO_FLOW_SHIFT,
) -> H3JointNoisyInputs:
    """Construct synchronized H3 video/audio flow inputs and data-pointing targets.

    ``base_sigma`` is the *unshifted* schedule coordinate shared by both
    modalities. Each modality derives its own sigma from it independently, so
    the two schedules stay synchronized for any pair of shifts. Passing an
    already-shifted sigma here would shift it twice.

    H3 uses ``x_t = (1 - sigma) * x0 + sigma * noise`` and predicts the
    data-pointing velocity ``x0 - noise``.
    """
    if video_latents is None and audio_latents is None:
        raise ValueError("H3 training requires at least one target modality")
    if (video_latents is None) != (video_noise is None):
        raise ValueError("H3 video latents and noise must either both be present or both be absent")
    if (audio_latents is None) != (audio_noise is None):
        raise ValueError("H3 audio latents and noise must either both be present or both be absent")
    if video_latents is not None and video_latents.shape != video_noise.shape:
        raise ValueError("H3 video latents and noise must have identical shapes")
    if audio_latents is not None and audio_latents.shape != audio_noise.shape:
        raise ValueError("H3 audio latents and noise must have identical shapes")
    present = video_latents if video_latents is not None else audio_latents
    if video_latents is not None and audio_latents is not None and video_latents.shape[0] != audio_latents.shape[0]:
        raise ValueError("H3 video and audio batch sizes must match")
    if video_shift <= 0 or audio_shift <= 0:
        raise ValueError("H3 flow shifts must be positive")
    _validate_sigma(base_sigma)
    if base_sigma.shape[0] != present.shape[0]:
        raise ValueError("H3 base sigma batch size must match the latents")

    base_sigma = base_sigma.float()
    video_sigma = _shift_unchecked(base_sigma, video_shift)
    audio_sigma = _shift_unchecked(base_sigma, audio_shift)
    video_sigma_expanded = _expand_batch_values(video_sigma, video_latents) if video_latents is not None else None
    audio_sigma_expanded = _expand_batch_values(audio_sigma, audio_latents) if audio_latents is not None else None

    return H3JointNoisyInputs(
        video=(1.0 - video_sigma_expanded) * video_latents + video_sigma_expanded * video_noise if video_latents is not None else None,
        audio=(1.0 - audio_sigma_expanded) * audio_latents + audio_sigma_expanded * audio_noise if audio_latents is not None else None,
        video_target=video_latents - video_noise if video_latents is not None else None,
        audio_target=audio_latents - audio_noise if audio_latents is not None else None,
        video_sigma=video_sigma,
        audio_sigma=audio_sigma,
        video_timestep=1.0 - video_sigma,
        audio_timestep=1.0 - audio_sigma,
    )


def guidance_consistent_prediction(
    guided: H3ModelPrediction,
    empty: H3ModelPrediction,
    guidance_scale: float,
    *,
    detach_empty: bool = True,
) -> H3ModelPrediction:
    """Estimate the raw conditional field from a guidance-distilled prediction.

    With ``g = u + s * (c - u)`` and ``g(empty) ~= u``, this returns
    ``c_hat = (g + (s - 1) * g(empty)) / s``.
    """
    if guidance_scale < 1.0:
        raise ValueError("H3 guidance distillation scale must be at least 1")
    if (guided.video is None) != (empty.video is None) or (guided.audio is None) != (empty.audio is None):
        raise ValueError("H3 prompt and empty predictions must contain the same modalities")
    if guided.video is not None and guided.video.shape != empty.video.shape:
        raise ValueError("H3 prompt and empty video predictions must have matching shapes")
    if guided.audio is not None and guided.audio.shape != empty.audio.shape:
        raise ValueError("H3 prompt and empty audio predictions must have matching shapes")
    empty_video = empty.video.detach() if detach_empty and empty.video is not None else empty.video
    empty_audio = empty.audio.detach() if detach_empty and empty.audio is not None else empty.audio
    empty_weight = guidance_scale - 1.0
    return H3ModelPrediction(
        video=(guided.video + empty_weight * empty_video) / guidance_scale if guided.video is not None else None,
        audio=(guided.audio + empty_weight * empty_audio) / guidance_scale if guided.audio is not None else None,
    )


def _broadcast_mask(mask: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(target, dtype=torch.bool)
    mask = mask.to(device=target.device, dtype=torch.bool)
    if mask.shape == target.shape:
        return mask
    if mask.ndim < 2 or mask.shape[0] != target.shape[0] or mask.shape[1:] != target.shape[-(mask.ndim - 1) :]:
        raise ValueError(f"loss mask shape {tuple(mask.shape)} cannot broadcast to {tuple(target.shape)}")
    shape = (mask.shape[0], *([1] * (target.ndim - mask.ndim)), *mask.shape[1:])
    return mask.view(shape).expand_as(target)


def _modality_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    sample_weight: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if prediction.shape != target.shape:
        raise ValueError(f"H3 prediction shape {tuple(prediction.shape)} does not match target {tuple(target.shape)}")
    valid = _broadcast_mask(mask, target)
    elements = int(valid.sum().item())
    if elements == 0:
        zero = prediction.sum() * 0.0
        return zero, zero, 0

    squared = (prediction - target).float().square()
    if sample_weight is not None:
        if sample_weight.shape != (target.shape[0],):
            raise ValueError("H3 sample weighting must contain one value per batch item")
        squared = squared * _expand_batch_values(sample_weight.float(), squared)
    total = squared.masked_select(valid).sum()
    return total / elements, total, elements


def joint_velocity_loss(
    prediction: H3ModelPrediction,
    inputs: H3JointNoisyInputs,
    *,
    video_mask: torch.Tensor | None = None,
    audio_mask: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    balance: LossBalance = "token",
    video_weight: float = 1.0,
    audio_weight: float = 1.0,
) -> H3JointLoss:
    """Reduce video and audio velocity errors with explicit modality balancing."""
    if balance not in {"token", "modality"}:
        raise ValueError(f"unsupported H3 loss balance: {balance}")
    if video_weight < 0 or audio_weight < 0 or video_weight + audio_weight <= 0:
        raise ValueError("H3 video/audio loss weights must be non-negative and not both zero")

    zero_source = prediction.video if prediction.video is not None else prediction.audio
    if zero_source is None:
        raise ValueError("H3 prediction contains no target modality")
    zero = zero_source.sum() * 0.0
    if prediction.video is None or inputs.video_target is None:
        if prediction.video is not None or inputs.video_target is not None:
            raise ValueError("H3 video prediction and target presence differ")
        video_mean, video_total, video_elements = zero, zero, 0
    else:
        video_mean, video_total, video_elements = _modality_loss(prediction.video, inputs.video_target, video_mask, sample_weight)
    if prediction.audio is None or inputs.audio_target is None:
        if prediction.audio is not None or inputs.audio_target is not None:
            raise ValueError("H3 audio prediction and target presence differ")
        audio_mean, audio_total, audio_elements = zero, zero, 0
    else:
        audio_mean, audio_total, audio_elements = _modality_loss(prediction.audio, inputs.audio_target, audio_mask, sample_weight)

    active_video_weight = video_weight if video_elements else 0.0
    active_audio_weight = audio_weight if audio_elements else 0.0
    if active_video_weight + active_audio_weight == 0:
        raise ValueError("H3 loss masks exclude every video and audio element")

    if balance == "modality":
        loss = (active_video_weight * video_mean + active_audio_weight * audio_mean) / (active_video_weight + active_audio_weight)
    else:
        weighted_elements = active_video_weight * video_elements + active_audio_weight * audio_elements
        loss = (active_video_weight * video_total + active_audio_weight * audio_total) / weighted_elements

    return H3JointLoss(loss, video_mean, audio_mean, video_elements, audio_elements)
