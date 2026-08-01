"""Forward Explorative Modeling utilities for LTX-2 flow-matching training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional

import torch

from musubi_tuner.training.losses import per_element_loss


def validate_ltx2_xm_args(args: Any) -> int:
    """Validate the deliberately bounded first Forward XM integration."""

    raw_k = getattr(args, "ltx2_xm_k", 1)
    k = 1 if raw_k is None else int(raw_k)
    if not 1 <= k <= 32:
        raise ValueError("--ltx2_xm_k must be an integer in [1, 32]")
    args.ltx2_xm_k = k
    if k == 1:
        return k

    incompatible_flags = {
        "self_flow": "--self_flow",
        "hfato": "--hfato",
        "differential_guidance": "--differential_guidance",
        "crepa": "--crepa",
        "tread": "--tread",
        "latent_temporal_weighting": "--latent_temporal_weighting",
        "latent_delta_loss": "--latent_delta_loss",
        "av_attention_loss_weighting": "--av_attention_loss_weighting",
        "blank_preservation": "--blank_preservation",
        "dop": "--dop",
        "prior_divergence": "--prior_divergence",
        "audio_dop": "--audio_dop",
        "ltx2_model_parallel": "--ltx2_model_parallel",
        "ltx2_remote_stage": "--ltx2_remote_stage",
        "ltx2_fsdp": "--ltx2_fsdp",
    }
    enabled = [option for attr, option in incompatible_flags.items() if bool(getattr(args, attr, False))]
    if enabled:
        raise ValueError(f"--ltx2_xm_k > 1 is not yet supported with {', '.join(enabled)}")

    balance_mode = str(getattr(args, "audio_loss_balance_mode", "none") or "none").lower()
    if balance_mode != "none":
        raise ValueError("--ltx2_xm_k > 1 currently requires --audio_loss_balance_mode none")
    if float(getattr(args, "cts_lambda_video_driven", 0.0) or 0.0) != 0.0:
        raise ValueError("--ltx2_xm_k > 1 is not yet supported with --cts_lambda_video_driven")
    if float(getattr(args, "cts_lambda_audio_driven", 0.0) or 0.0) != 0.0:
        raise ValueError("--ltx2_xm_k > 1 is not yet supported with --cts_lambda_audio_driven")
    return k


@dataclass(frozen=True)
class RNGSnapshot:
    """Python and PyTorch RNG state used to replay candidate-independent choices."""

    python_state: object
    cpu_state: torch.Tensor
    cuda_state: Optional[torch.Tensor]
    cuda_device: Optional[torch.device]

    @classmethod
    def capture(cls, device: torch.device) -> "RNGSnapshot":
        cuda_device = device if device.type == "cuda" and torch.cuda.is_available() else None
        cuda_state = torch.cuda.get_rng_state(cuda_device) if cuda_device is not None else None
        return cls(random.getstate(), torch.get_rng_state(), cuda_state, cuda_device)

    def restore(self) -> None:
        random.setstate(self.python_state)
        torch.set_rng_state(self.cpu_state)
        if self.cuda_state is not None and self.cuda_device is not None:
            torch.cuda.set_rng_state(self.cuda_state, self.cuda_device)

    def matches(self, other: "RNGSnapshot") -> bool:
        """Return whether two snapshots describe the exact same RNG state."""

        if self.python_state != other.python_state or self.cuda_device != other.cuda_device:
            return False
        if not torch.equal(self.cpu_state, other.cpu_state):
            return False
        if self.cuda_state is None or other.cuda_state is None:
            return self.cuda_state is None and other.cuda_state is None
        return torch.equal(self.cuda_state, other.cuda_state)


def seeded_randn_like(tensor: torch.Tensor, seed: int) -> torch.Tensor:
    """Draw noise from a private generator without advancing the process RNG."""

    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(int(seed))
    return torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype)


def _broadcast_mask(mask: torch.Tensor, per_elem: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=per_elem.device)
    if per_elem.dim() == 5 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], 1, mask.shape[1], 1, 1)
    elif per_elem.dim() == 5 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1, 1)
    elif per_elem.dim() == 4 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], 1, mask.shape[1], 1)
    elif per_elem.dim() == 4 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1)
    elif per_elem.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    elif per_elem.dim() == 3 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1)
    return mask.to(dtype=per_elem.dtype).expand_as(per_elem)


def per_sample_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    loss_type: str,
    huber_delta: float,
    batch_global: bool = False,
) -> torch.Tensor:
    """Return each sample's FP32 contribution to the configured loss reduction.

    ``batch_global=False`` matches ``apply_loss_mask_per_sample``: every sample
    is normalized by its own active elements. ``batch_global=True`` decomposes
    ``apply_loss_mask`` into per-sample contributions whose sum is the exact
    batch-global scalar. The latter distinction matters for joint AV selection
    when video and audio masks have different active counts.
    """

    per_elem = per_element_loss(pred, target, loss_type, huber_delta).float()
    dims = tuple(range(1, per_elem.dim()))
    if mask is None:
        if batch_global:
            return per_elem.sum(dim=dims) / float(per_elem.numel())
        return per_elem.mean(dim=dims)
    mask_f = _broadcast_mask(mask, per_elem)
    active = mask_f.sum(dim=dims)
    masked_sum = (per_elem * mask_f).sum(dim=dims)
    if batch_global:
        total_active = active.sum()
        if bool(total_active > 0):
            return masked_sum / total_active
        return per_elem.sum(dim=dims) / float(per_elem.numel())
    values = masked_sum / active.clamp(min=1.0)
    return torch.where(active > 0, values, torch.zeros_like(values))


def score_ltx2_candidate(
    output: dict[str, Any],
    *,
    loss_type: str,
    huber_delta: float,
    per_sample_loss: bool = False,
) -> torch.Tensor:
    """Score an LTX video, audio, or joint AV output for Forward XM selection."""

    video_pred = output.get("video_pred")
    video_target = output.get("video_target")
    if not isinstance(video_pred, torch.Tensor) or not isinstance(video_target, torch.Tensor):
        raise TypeError("LTX XM candidate output must contain tensor video_pred and video_target values")
    score = per_sample_reconstruction_loss(
        video_pred,
        video_target,
        output.get("video_loss_mask"),
        loss_type=loss_type,
        huber_delta=huber_delta,
        batch_global=not per_sample_loss,
    ) * float(output.get("video_loss_weight", 1.0))

    audio_pred = output.get("audio_pred")
    audio_target = output.get("audio_target")
    if audio_pred is not None or audio_target is not None:
        if not isinstance(audio_pred, torch.Tensor) or not isinstance(audio_target, torch.Tensor):
            raise TypeError("LTX XM candidate audio prediction and target must either both be tensors or both be absent")
        audio_score = per_sample_reconstruction_loss(
            audio_pred,
            audio_target,
            output.get("audio_loss_mask"),
            loss_type=loss_type,
            huber_delta=huber_delta,
            batch_global=not per_sample_loss,
        )
        score = score + audio_score * float(output.get("audio_loss_weight", 1.0))

    return score.float()


def update_streaming_winner(
    best_score: torch.Tensor,
    candidate_score: torch.Tensor,
    candidate_index: int,
    best_index: torch.Tensor,
    best_video_noise: torch.Tensor,
    candidate_video_noise: torch.Tensor,
    best_audio_noise: Optional[torch.Tensor],
    candidate_audio_noise: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Update per-sample hard-min winners while keeping only one candidate payload."""

    better = candidate_score < best_score
    best_score = torch.where(better, candidate_score, best_score)
    best_index = torch.where(better, torch.full_like(best_index, candidate_index), best_index)
    video_mask = better.to(device=best_video_noise.device).view(better.shape[0], *([1] * (best_video_noise.dim() - 1)))
    best_video_noise = torch.where(video_mask, candidate_video_noise.to(best_video_noise.device), best_video_noise)

    if candidate_audio_noise is not None:
        if best_audio_noise is None:
            best_audio_noise = candidate_audio_noise.detach().clone()
        else:
            audio_mask = better.to(device=best_audio_noise.device).view(better.shape[0], *([1] * (best_audio_noise.dim() - 1)))
            best_audio_noise = torch.where(
                audio_mask,
                candidate_audio_noise.to(best_audio_noise.device),
                best_audio_noise,
            )
    return best_score, best_index, best_video_noise, best_audio_noise


def build_xm_metrics(
    *,
    k: int,
    candidate0_score: torch.Tensor,
    best_score: torch.Tensor,
    best_index: torch.Tensor,
    replay_score: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    finite = torch.isfinite(best_score)
    safe_best = best_score[finite]
    safe_first = candidate0_score[finite]
    metrics = {
        "xm/k": float(k),
        "xm/invalid_samples": float((~finite).sum().item()),
    }
    if safe_best.numel() == 0:
        return metrics
    metrics.update(
        {
            "xm/winner_score": float(safe_best.mean().item()),
            "xm/candidate0_score": float(safe_first.mean().item()),
            "xm/selection_gain": float((safe_first - safe_best).mean().item()),
            "xm/winner_index_mean": float(best_index[finite].float().mean().item()),
        }
    )
    if replay_score is not None:
        replay_error = (replay_score[finite] - safe_best).abs()
        metrics["xm/replay_max_abs_error"] = float(replay_error.max().item())
        metrics["xm/replay_mean_abs_error"] = float(replay_error.mean().item())
    for index in range(k):
        metrics[f"xm/winner_{index}_fraction"] = float((best_index[finite] == index).float().mean().item())
    return metrics
