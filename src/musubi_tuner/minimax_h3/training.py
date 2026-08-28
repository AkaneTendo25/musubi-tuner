from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from musubi_tuner.minimax_h3.architecture import AUDIO_FLOW_SHIFT, VIDEO_FLOW_SHIFT

LossBalance = Literal["token", "modality"]
H3TrainingMode = Literal["fl2va", "ref2va", "ref2va_omni"]
ObservedModality = Literal["video", "audio"]

# Noise levels the released transformer uses for a modality it reads rather than
# generates. Video conditioning carries a trace of noise and rides timestep
# 0.999; audio conditioning is passed through untouched at timestep 1.0.
# ``x_t = (1 - sigma) * x0 + sigma * noise`` reproduces both exactly, so an
# observed modality is expressed by pinning its sigma rather than by a separate
# code path.
OBSERVED_VIDEO_SIGMA = 0.001
OBSERVED_AUDIO_SIGMA = 0.0


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
    video_frame_sigma: torch.Tensor | None = None


@dataclass(frozen=True)
class H3ModelPrediction:
    video: torch.Tensor | None
    audio: torch.Tensor | None


@dataclass(frozen=True)
class H3JointLoss:
    loss: torch.Tensor
    video_loss: torch.Tensor
    audio_loss: torch.Tensor
    # Counts of valid elements, for reporting and for telling an inactive
    # modality from an active one. They are not the denominator of the means
    # above whenever sample weighting is active.
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


def _expand_scale_values(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expand either per-item values or an explicitly broadcastable scale grid."""
    if values.ndim == 1:
        return _expand_batch_values(values, target)
    try:
        torch.broadcast_shapes(values.shape, target.shape)
    except RuntimeError as exc:
        raise ValueError(f"H3 guidance scale shape {tuple(values.shape)} cannot broadcast to {tuple(target.shape)}") from exc
    return values.to(device=target.device, dtype=target.dtype)


def prepare_joint_noisy_inputs(
    video_latents: torch.Tensor | None,
    audio_latents: torch.Tensor | None,
    video_noise: torch.Tensor | None,
    audio_noise: torch.Tensor | None,
    base_sigma: torch.Tensor,
    *,
    video_shift: float = VIDEO_FLOW_SHIFT,
    audio_shift: float = AUDIO_FLOW_SHIFT,
    observed: ObservedModality | None = None,
) -> H3JointNoisyInputs:
    """Construct synchronized H3 video/audio flow inputs and data-pointing targets.

    ``base_sigma`` is the *unshifted* schedule coordinate shared by both
    modalities. Each modality derives its own sigma from it independently, so
    the two schedules stay synchronized for any pair of shifts. Passing an
    already-shifted sigma here would shift it twice.

    H3 uses ``x_t = (1 - sigma) * x0 + sigma * noise`` and predicts the
    data-pointing velocity ``x0 - noise``.

    ``observed`` pins one modality to the noise level the released transformer
    uses for conditioning, leaving the other on the sampled schedule. Both
    modalities must be present, and the observed one carries no training signal,
    so its loss weight has to be zeroed by the caller.
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

    if observed is not None:
        if observed not in ("video", "audio"):
            raise ValueError(f"H3 observed modality must be 'video' or 'audio', got {observed!r}")
        if video_latents is None or audio_latents is None:
            raise ValueError("H3 observed-modality training requires both video and audio targets")

    base_sigma = base_sigma.float()
    video_sigma = _shift_unchecked(base_sigma, video_shift)
    audio_sigma = _shift_unchecked(base_sigma, audio_shift)
    if observed == "video":
        video_sigma = torch.full_like(base_sigma, OBSERVED_VIDEO_SIGMA)
    elif observed == "audio":
        audio_sigma = torch.full_like(base_sigma, OBSERVED_AUDIO_SIGMA)
    video_sigma_expanded = _expand_batch_values(video_sigma, video_latents) if video_latents is not None else None
    audio_sigma_expanded = _expand_batch_values(audio_sigma, audio_latents) if audio_latents is not None else None

    return H3JointNoisyInputs(
        video=(1.0 - video_sigma_expanded) * video_latents + video_sigma_expanded * video_noise
        if video_latents is not None
        else None,
        audio=(1.0 - audio_sigma_expanded) * audio_latents + audio_sigma_expanded * audio_noise
        if audio_latents is not None
        else None,
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
    guidance_scale: float | torch.Tensor,
    *,
    audio_guidance_scale: float | torch.Tensor | None = None,
    detach_empty: bool = True,
) -> H3ModelPrediction:
    """Estimate the raw conditional field from a guidance-distilled prediction.

    With ``g = u + s * (c - u)`` and ``g(empty) ~= u``, this returns
    ``c_hat = (g + (s - 1) * g(empty)) / s``.
    """
    audio_guidance_scale = guidance_scale if audio_guidance_scale is None else audio_guidance_scale
    for scale in (guidance_scale, audio_guidance_scale):
        if isinstance(scale, torch.Tensor):
            if bool((scale < 1.0).any()):
                raise ValueError("H3 guidance distillation scale must be at least 1")
        elif scale < 1.0:
            raise ValueError("H3 guidance distillation scale must be at least 1")
    if (guided.video is None) != (empty.video is None) or (guided.audio is None) != (empty.audio is None):
        raise ValueError("H3 prompt and empty predictions must contain the same modalities")
    if guided.video is not None and guided.video.shape != empty.video.shape:
        raise ValueError("H3 prompt and empty video predictions must have matching shapes")
    if guided.audio is not None and guided.audio.shape != empty.audio.shape:
        raise ValueError("H3 prompt and empty audio predictions must have matching shapes")
    empty_video = empty.video.detach() if detach_empty and empty.video is not None else empty.video
    empty_audio = empty.audio.detach() if detach_empty and empty.audio is not None else empty.audio
    video_scale = (
        _expand_scale_values(guidance_scale, guided.video)
        if isinstance(guidance_scale, torch.Tensor) and guided.video is not None
        else guidance_scale
    )
    audio_scale = (
        _expand_scale_values(audio_guidance_scale, guided.audio)
        if isinstance(audio_guidance_scale, torch.Tensor) and guided.audio is not None
        else audio_guidance_scale
    )
    return H3ModelPrediction(
        video=(guided.video + (video_scale - 1.0) * empty_video) / video_scale if guided.video is not None else None,
        audio=(guided.audio + (audio_scale - 1.0) * empty_audio) / audio_scale if guided.audio is not None else None,
    )


def _cfg_zero_alpha(reference: torch.Tensor, empty: torch.Tensor, eps: float) -> torch.Tensor:
    if reference.shape != empty.shape:
        raise ValueError("H3 CFG-Zero rescaling requires matching reference and empty prediction shapes")
    batch = empty.shape[0]
    reference_flat = reference.detach().float().reshape(batch, -1)
    empty_flat = empty.detach().float().reshape(batch, -1)
    dot = (reference_flat * empty_flat).sum(dim=1)
    squared_norm = empty_flat.square().sum(dim=1) + eps
    alpha = dot / squared_norm
    return alpha.reshape(batch, *([1] * (empty.ndim - 1)))


def cfg_zero_rescaled_empty(
    empty: H3ModelPrediction,
    reference: H3ModelPrediction,
    *,
    eps: float = 1e-8,
) -> H3ModelPrediction:
    """Project the null field onto the reference field before guidance is applied.

    Per sample and per modality, ``alpha = <reference, u> / (||u||^2 + eps)``
    keeps only the component of the null prediction that the conditional field
    actually opposes; an orthogonal null branch collapses to zero instead of
    being extrapolated away from.
    """
    if (empty.video is None) != (reference.video is None) or (empty.audio is None) != (reference.audio is None):
        raise ValueError("H3 CFG-Zero rescaling requires the same modalities in both predictions")
    video = None
    if empty.video is not None:
        alpha = _cfg_zero_alpha(reference.video, empty.video, eps)
        video = (empty.video.float() * alpha).to(empty.video.dtype)
    audio = None
    if empty.audio is not None:
        alpha = _cfg_zero_alpha(reference.audio, empty.audio, eps)
        audio = (empty.audio.float() * alpha).to(empty.audio.dtype)
    return H3ModelPrediction(video=video, audio=audio)


def guidance_scale_for_sigma(configured_scale: float | torch.Tensor, sigma: torch.Tensor, schedule: str) -> torch.Tensor:
    """Resolve the per-example guidance scale for one H3 modality.

    ``configured_scale`` is normally the single authoritative distillation
    scale. It may also be a tensor, which is how a per-sample scale drawn from
    ``--h3_guidance_scale_range`` reaches the schedule: the caller is then
    responsible for a shape that broadcasts against ``sigma`` (e.g. ``[batch,
    1]`` against a per-frame ``[1, frames]`` sigma). The float path is left
    bit-for-bit unchanged.
    """
    if isinstance(configured_scale, torch.Tensor):
        if not configured_scale.is_floating_point():
            raise TypeError("H3 guidance distillation scale must be floating point")
        if bool((configured_scale < 1.0).any()):
            raise ValueError("H3 guidance distillation scale must be at least 1")
        scale: float | torch.Tensor = configured_scale.to(device=sigma.device, dtype=sigma.dtype)
    else:
        if configured_scale < 1.0:
            raise ValueError("H3 guidance distillation scale must be at least 1")
        scale = configured_scale
    if schedule == "constant":
        if isinstance(scale, torch.Tensor):
            # ``full_like`` cannot carry a per-sample value; the multiply is the
            # broadcasting equivalent and keeps sigma's device and dtype.
            return scale * torch.ones_like(sigma)
        return torch.full_like(sigma, scale)
    if schedule == "sigma":
        return 1.0 + (scale - 1.0) * sigma
    raise ValueError(f"unsupported H3 guidance loss schedule: {schedule}")


def contrastive_guidance_target(
    target: H3ModelPrediction,
    empty: H3ModelPrediction,
    guidance_scale: float | torch.Tensor,
    *,
    audio_guidance_scale: float | torch.Tensor | None = None,
) -> H3ModelPrediction:
    """Construct the guided-field target equivalent to the normalized objective."""
    audio_guidance_scale = guidance_scale if audio_guidance_scale is None else audio_guidance_scale
    video_scale = (
        _expand_scale_values(guidance_scale, target.video)
        if isinstance(guidance_scale, torch.Tensor) and target.video is not None
        else guidance_scale
    )
    audio_scale = (
        _expand_scale_values(audio_guidance_scale, target.audio)
        if isinstance(audio_guidance_scale, torch.Tensor) and target.audio is not None
        else audio_guidance_scale
    )
    return H3ModelPrediction(
        video=(empty.video + video_scale * (target.video - empty.video)) if target.video is not None else None,
        audio=(empty.audio + audio_scale * (target.audio - empty.audio)) if target.audio is not None else None,
    )


def _broadcast_mask(mask: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(target, dtype=torch.bool)
    dtype = torch.float32 if mask.is_floating_point() else torch.bool
    mask = mask.to(device=target.device, dtype=dtype)
    if mask.shape == target.shape:
        return mask
    if mask.ndim < 2 or mask.shape[0] != target.shape[0] or mask.shape[1:] != target.shape[-(mask.ndim - 1) :]:
        raise ValueError(f"loss mask shape {tuple(mask.shape)} cannot broadcast to {tuple(target.shape)}")
    shape = (mask.shape[0], *([1] * (target.ndim - mask.ndim)), *mask.shape[1:])
    return mask.view(shape).expand_as(target)


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor | float) -> torch.Tensor:
    """Divide by a mean denominator that may be a vanishing weight sum.

    A plain element count is always positive here, so the float path is the
    unchanged division. A weight sum can legitimately reach zero (every item in
    the batch weighted 0), in which case the numerator is zero as well; clamping
    keeps that case at 0 instead of NaN.
    """
    if isinstance(denominator, torch.Tensor):
        return numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)
    return numerator / denominator


def _modality_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    sample_weight: torch.Tensor | None,
    mask_normalization: str = "weighted",
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor | float]:
    """Return ``(mean, weighted sum, valid element count, mean denominator)``.

    The element count stays a plain count for reporting and for deciding which
    modalities are active. The denominator is what the mean actually divides by:
    the element count when unweighted, and the sum of the per-element sample
    weights over the valid elements when weighting is active. Dividing a
    weighted numerator by an unweighted count would rescale the loss by the mean
    sample weight instead of reweighting it, and would disagree with validation's
    :func:`~musubi_tuner.minimax_h3.validation.masked_squared_error_sum`.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"H3 prediction shape {tuple(prediction.shape)} does not match target {tuple(target.shape)}")
    mask_weights = None if mask is None else _broadcast_mask(mask, target)
    if mask_weights is None:
        per_item_valid = None
        elements = target.numel()
    else:
        # One reduction serves both the count and the per-item weight sum, so
        # masked steps keep the single host round-trip they already paid.
        positive = mask_weights > 0
        per_item_valid = mask_weights.to(dtype=torch.float32).sum(dim=tuple(range(1, mask_weights.ndim)))
        elements = int(positive.sum().item())
    if elements == 0:
        zero = prediction.sum() * 0.0
        return zero, zero, 0, 0.0

    squared = (prediction - target).float().square()
    if mask_normalization not in {"weighted", "full"}:
        raise ValueError(f"unsupported H3 loss mask normalization: {mask_normalization}")
    denominator: torch.Tensor | float = (
        float(target.numel())
        if mask_normalization == "full"
        else (float(elements) if mask_weights is None else per_item_valid.sum())
    )
    if sample_weight is not None:
        if sample_weight.shape != (target.shape[0],):
            raise ValueError("H3 sample weighting must contain one value per batch item")
        weights = sample_weight.to(device=squared.device, dtype=torch.float32)
        squared = squared * _expand_batch_values(weights, squared)
        # Kept as a device tensor: every consumer is tensor arithmetic, so no
        # extra synchronization is introduced by the weighted denominator.
        if per_item_valid is None or mask_normalization == "full":
            denominator = weights.sum() * float(target.numel() // target.shape[0])
        else:
            denominator = (per_item_valid.to(dtype=torch.float32) * weights).sum()
    # Avoid ``masked_select`` here: its data-dependent output allocates a second
    # dense loss buffer and introduces an additional synchronization point.
    # Multiplication preserves the exact masked sum while keeping a static shape.
    total = squared.sum() if mask_weights is None else (squared * mask_weights).sum()
    return _safe_divide(total, denominator), total, elements, denominator


def _joint_loss(
    prediction: H3ModelPrediction,
    target: H3ModelPrediction,
    *,
    video_mask: torch.Tensor | None = None,
    audio_mask: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    video_sample_weight: torch.Tensor | None = None,
    audio_sample_weight: torch.Tensor | None = None,
    balance: LossBalance = "token",
    video_weight: float = 1.0,
    audio_weight: float = 1.0,
    mask_normalization: str = "weighted",
) -> H3JointLoss:
    if balance not in {"token", "modality"}:
        raise ValueError(f"unsupported H3 loss balance: {balance}")
    if video_weight < 0 or audio_weight < 0 or video_weight + audio_weight <= 0:
        raise ValueError("H3 video/audio loss weights must be non-negative and not both zero")

    zero_source = prediction.video if prediction.video is not None else prediction.audio
    if zero_source is None:
        raise ValueError("H3 prediction contains no target modality")
    zero = zero_source.sum() * 0.0
    if sample_weight is not None and (video_sample_weight is not None or audio_sample_weight is not None):
        raise ValueError("pass either shared or per-modality H3 sample weights, not both")
    video_sample_weight = sample_weight if video_sample_weight is None else video_sample_weight
    audio_sample_weight = sample_weight if audio_sample_weight is None else audio_sample_weight
    # A zero modality weight contributes nothing to the total, so its squared
    # error is never computed: it is the largest allocation in an observed-modality
    # (v2a/a2v) step, and materializing it only to multiply by zero also reported a
    # meaningless ``loss/video``/``loss/audio`` value for the conditioning modality.
    if prediction.video is None or target.video is None or video_weight == 0:
        if (prediction.video is None) != (target.video is None):
            raise ValueError("H3 video prediction and target presence differ")
        video_mean, video_total, video_elements, video_denominator = zero, zero, 0, 0.0
    else:
        loss_args = (prediction.video, target.video, video_mask, video_sample_weight)
        video_mean, video_total, video_elements, video_denominator = (
            _modality_loss(*loss_args) if mask_normalization == "weighted" else _modality_loss(*loss_args, mask_normalization)
        )
    if prediction.audio is None or target.audio is None or audio_weight == 0:
        if (prediction.audio is None) != (target.audio is None):
            raise ValueError("H3 audio prediction and target presence differ")
        audio_mean, audio_total, audio_elements, audio_denominator = zero, zero, 0, 0.0
    else:
        loss_args = (prediction.audio, target.audio, audio_mask, audio_sample_weight)
        audio_mean, audio_total, audio_elements, audio_denominator = (
            _modality_loss(*loss_args) if mask_normalization == "weighted" else _modality_loss(*loss_args, mask_normalization)
        )

    active_video_weight = video_weight if video_elements else 0.0
    active_audio_weight = audio_weight if audio_elements else 0.0
    if active_video_weight + active_audio_weight == 0:
        return H3JointLoss(zero, video_mean, audio_mean, video_elements, audio_elements)

    if balance == "modality":
        loss = (active_video_weight * video_mean + active_audio_weight * audio_mean) / (active_video_weight + active_audio_weight)
    else:
        # Token balancing pools the two numerators, so it pools the two mean
        # denominators as well; with sample weighting those are weight sums, in
        # the same convention as the validation accumulator.
        weighted_denominator = active_video_weight * video_denominator + active_audio_weight * audio_denominator
        loss = _safe_divide(active_video_weight * video_total + active_audio_weight * audio_total, weighted_denominator)

    return H3JointLoss(loss, video_mean, audio_mean, video_elements, audio_elements)


def joint_velocity_loss(
    prediction: H3ModelPrediction,
    inputs: H3JointNoisyInputs,
    *,
    video_mask: torch.Tensor | None = None,
    audio_mask: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    video_sample_weight: torch.Tensor | None = None,
    audio_sample_weight: torch.Tensor | None = None,
    balance: LossBalance = "token",
    video_weight: float = 1.0,
    audio_weight: float = 1.0,
    mask_normalization: str = "weighted",
) -> H3JointLoss:
    """Reduce video and audio velocity errors with explicit modality balancing."""
    return _joint_loss(
        prediction,
        H3ModelPrediction(inputs.video_target, inputs.audio_target),
        video_mask=video_mask,
        audio_mask=audio_mask,
        sample_weight=sample_weight,
        video_sample_weight=video_sample_weight,
        audio_sample_weight=audio_sample_weight,
        balance=balance,
        video_weight=video_weight,
        audio_weight=audio_weight,
        mask_normalization=mask_normalization,
    )


def joint_prediction_loss(
    prediction: H3ModelPrediction,
    reference: H3ModelPrediction,
    *,
    video_mask: torch.Tensor | None = None,
    audio_mask: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    video_sample_weight: torch.Tensor | None = None,
    audio_sample_weight: torch.Tensor | None = None,
    balance: LossBalance = "token",
    video_weight: float = 1.0,
    audio_weight: float = 1.0,
    mask_normalization: str = "weighted",
) -> H3JointLoss:
    """Measure drift from a detached base-model prediction.

    This is function-space preservation, not CFG distillation: callers must
    evaluate both branches on identical noisy inputs and conditioning.
    """
    detached_reference = H3ModelPrediction(
        reference.video.detach() if reference.video is not None else None,
        reference.audio.detach() if reference.audio is not None else None,
    )
    return _joint_loss(
        prediction,
        detached_reference,
        video_mask=video_mask,
        audio_mask=audio_mask,
        sample_weight=sample_weight,
        video_sample_weight=video_sample_weight,
        audio_sample_weight=audio_sample_weight,
        balance=balance,
        video_weight=video_weight,
        audio_weight=audio_weight,
        mask_normalization=mask_normalization,
    )
