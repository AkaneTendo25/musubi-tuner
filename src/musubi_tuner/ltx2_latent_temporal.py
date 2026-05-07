"""Latent temporal objectives for LTX-2 video training.

These helpers implement two lightweight training-time additions:

* LTD-style motion weighting: reweight the regular denoising loss by clean
  latent frame-to-frame discrepancy.
* Predicted-latent temporal matching: encourage predicted x0/velocity deltas to
  follow the clean training latent trajectory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LatentTemporalWeightingConfig:
    alpha: float = 0.5
    mode: str = "log"  # "log" | "linear"
    normalize: str = "mean"  # "mean" | "max" | "none"
    clip_min: float = 0.5
    clip_max: float = 2.0
    eps: float = 1e-6


@dataclass
class LatentDeltaLossConfig:
    weight: float = 0.03
    order: str = "1"  # "1" | "2" | "1+2" | "both"
    target: str = "x0"  # "x0" | "velocity"
    loss_type: str = "mse"  # "mse" | "l1" | "huber" | "smooth_l1"
    huber_delta: float = 1.0
    second_order_weight: float = 0.5
    sigma_min: float = 0.0
    sigma_max: float = 1.0
    eps: float = 1e-6


def parse_latent_temporal_args(raw_args: Optional[Sequence[str]]) -> Dict[str, str]:
    """Parse ``key=value`` CLI args into a dict."""
    if not raw_args:
        return {}
    parsed: Dict[str, str] = {}
    for item in raw_args:
        if "=" not in item:
            raise ValueError(f"Latent temporal arg must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _apply_config_overrides(config: Any, raw: Dict[str, str], *, name: str) -> Any:
    if not raw:
        return config
    fields = getattr(config, "__dataclass_fields__", {})
    for key, value in raw.items():
        if key not in fields:
            raise ValueError(f"Unknown {name} arg: {key}")
        current = getattr(config, key)
        if isinstance(current, bool):
            coerced = _to_bool(value)
        elif isinstance(current, int) and not isinstance(current, bool):
            coerced = int(value)
        elif isinstance(current, float):
            coerced = float(value)
        else:
            coerced = str(value)
        setattr(config, key, coerced)
    return config


def build_weighting_config(raw: Optional[Sequence[str]]) -> LatentTemporalWeightingConfig:
    config = _apply_config_overrides(
        LatentTemporalWeightingConfig(),
        parse_latent_temporal_args(raw),
        name="latent_temporal_weighting_args",
    )
    if config.alpha < 0.0:
        raise ValueError("latent temporal weighting alpha must be >= 0")
    if config.mode not in {"log", "linear"}:
        raise ValueError("latent temporal weighting mode must be one of: log, linear")
    if config.normalize not in {"mean", "max", "none"}:
        raise ValueError("latent temporal weighting normalize must be one of: mean, max, none")
    if config.clip_min <= 0.0:
        raise ValueError("latent temporal weighting clip_min must be > 0")
    if config.clip_max < config.clip_min:
        raise ValueError("latent temporal weighting clip_max must be >= clip_min")
    if config.eps <= 0.0:
        raise ValueError("latent temporal weighting eps must be > 0")
    return config


def build_delta_loss_config(raw: Optional[Sequence[str]]) -> LatentDeltaLossConfig:
    config = _apply_config_overrides(
        LatentDeltaLossConfig(),
        parse_latent_temporal_args(raw),
        name="latent_delta_loss_args",
    )
    if config.weight < 0.0:
        raise ValueError("latent delta loss weight must be >= 0")
    order = str(config.order).lower().replace(",", "+")
    if order == "both":
        order = "1+2"
    if order not in {"1", "2", "1+2", "2+1"}:
        raise ValueError("latent delta loss order must be one of: 1, 2, 1+2, both")
    config.order = "1+2" if order == "2+1" else order
    if config.target not in {"x0", "velocity"}:
        raise ValueError("latent delta loss target must be one of: x0, velocity")
    if config.loss_type not in {"mse", "l1", "huber", "smooth_l1"}:
        raise ValueError("latent delta loss loss_type must be one of: mse, l1, huber, smooth_l1")
    if config.huber_delta <= 0.0:
        raise ValueError("latent delta loss huber_delta must be > 0")
    if config.second_order_weight < 0.0:
        raise ValueError("latent delta loss second_order_weight must be >= 0")
    if config.sigma_min < 0.0 or config.sigma_max > 1.0 or config.sigma_max < config.sigma_min:
        raise ValueError("latent delta loss sigma range must satisfy 0 <= sigma_min <= sigma_max <= 1")
    if config.eps <= 0.0:
        raise ValueError("latent delta loss eps must be > 0")
    return config


def _loss_fn(pred: torch.Tensor, target: torch.Tensor, config: LatentDeltaLossConfig) -> torch.Tensor:
    pred_f = pred.float()
    target_f = target.float()
    if config.loss_type == "l1":
        return F.l1_loss(pred_f, target_f, reduction="none")
    if config.loss_type in {"huber", "smooth_l1"}:
        return F.smooth_l1_loss(pred_f, target_f, reduction="none", beta=config.huber_delta)
    return F.mse_loss(pred_f, target_f, reduction="none")


def _broadcast_video_mask(mask: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    mask = mask.to(device=ref.device)
    if ref.dim() == 5 and mask.dim() == 2:
        return mask.view(mask.shape[0], 1, mask.shape[1], 1, 1)
    if ref.dim() == 5 and mask.dim() == 1:
        return mask.view(mask.shape[0], 1, 1, 1, 1)
    if ref.dim() == 5 and mask.dim() == 4:
        return mask.unsqueeze(1)
    if ref.dim() == 5 and mask.dim() == 5:
        return mask
    return mask


def _masked_mean(per_elem: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return per_elem.mean()
    mask = _broadcast_video_mask(mask, per_elem)
    if mask is None:
        return per_elem.mean()
    mask_f = mask.to(device=per_elem.device, dtype=per_elem.dtype)
    denom = mask_f.mean()
    if denom.item() <= 0.0:
        return per_elem.sum() * 0.0
    return (per_elem * mask_f).div(denom).mean()


def _sigma_sample_mask(sigma: torch.Tensor, config: LatentDeltaLossConfig, ref: torch.Tensor) -> torch.Tensor:
    sigma = sigma.to(device=ref.device, dtype=torch.float32)
    if sigma.dim() > 1:
        sigma = sigma.reshape(sigma.shape[0], -1).mean(dim=1)
    sample_mask = (sigma >= config.sigma_min) & (sigma <= config.sigma_max)
    return sample_mask.to(dtype=ref.dtype).view(-1, 1, 1, 1, 1)


def _frame_motion_scores(clean_latents: torch.Tensor, config: LatentTemporalWeightingConfig) -> Optional[torch.Tensor]:
    if clean_latents.dim() != 5 or clean_latents.shape[2] < 2:
        return None

    clean = clean_latents.detach().float()
    gap = clean[:, :, 1:, :, :] - clean[:, :, :-1, :, :]
    gap = gap.pow(2).mean(dim=(1, 3, 4)).clamp_min(config.eps).sqrt()  # [B, F-1]

    bsz, gaps = gap.shape
    frame_scores = torch.zeros((bsz, gaps + 1), device=clean.device, dtype=torch.float32)
    frame_scores[:, 0] = gap[:, 0]
    frame_scores[:, -1] = gap[:, -1]
    if gaps > 1:
        frame_scores[:, 1:-1] = 0.5 * (gap[:, :-1] + gap[:, 1:])

    if config.mode == "log":
        frame_scores = torch.log1p(frame_scores)

    if config.normalize == "mean":
        frame_scores = frame_scores / frame_scores.mean(dim=1, keepdim=True).clamp_min(config.eps)
    elif config.normalize == "max":
        frame_scores = frame_scores / frame_scores.max(dim=1, keepdim=True).values.clamp_min(config.eps)

    return frame_scores


def apply_latent_temporal_weighting(
    per_elem: torch.Tensor,
    context: Optional[Dict[str, torch.Tensor]],
    config: Optional[LatentTemporalWeightingConfig],
) -> tuple[torch.Tensor, Dict[str, float]]:
    if config is None or context is None or per_elem.dim() != 5:
        return per_elem, {}
    clean_latents = context.get("clean_latents")
    if not isinstance(clean_latents, torch.Tensor):
        return per_elem, {}
    scores = _frame_motion_scores(clean_latents, config)
    if scores is None or scores.shape[0] != per_elem.shape[0] or scores.shape[1] != per_elem.shape[2]:
        return per_elem, {}

    weights = 1.0 + float(config.alpha) * scores
    weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(config.eps)
    weights = weights.clamp(min=config.clip_min, max=config.clip_max)
    weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(config.eps)
    weights = weights.to(device=per_elem.device, dtype=per_elem.dtype).view(per_elem.shape[0], 1, per_elem.shape[2], 1, 1)

    metrics = {
        "latent_temporal_weight_mean": float(weights.detach().float().mean().item()),
        "latent_temporal_weight_min": float(weights.detach().float().min().item()),
        "latent_temporal_weight_max": float(weights.detach().float().max().item()),
    }
    return per_elem * weights, metrics


def _select_prediction_target(
    video_pred: torch.Tensor,
    video_target: torch.Tensor,
    context: Dict[str, torch.Tensor],
    config: LatentDeltaLossConfig,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    if config.target == "velocity":
        return video_pred, video_target

    noisy_latents = context.get("noisy_latents")
    clean_latents = context.get("clean_latents")
    sigma = context.get("sigma")
    if not isinstance(noisy_latents, torch.Tensor) or not isinstance(clean_latents, torch.Tensor) or not isinstance(sigma, torch.Tensor):
        return None
    if video_pred.dim() != 5 or noisy_latents.shape != video_pred.shape or clean_latents.shape != video_pred.shape:
        return None

    sigma_5d = sigma.to(device=video_pred.device, dtype=video_pred.dtype)
    if sigma_5d.dim() == 5:
        channel_ok = sigma_5d.shape[1] in (1, video_pred.shape[1])
        if sigma_5d.shape[0] != video_pred.shape[0] or not channel_ok or sigma_5d.shape[2:] != video_pred.shape[2:]:
            return None
    elif sigma_5d.dim() > 1:
        sigma_5d = sigma_5d.reshape(sigma_5d.shape[0], -1).mean(dim=1)
        sigma_5d = sigma_5d.view(-1, 1, 1, 1, 1)
    else:
        sigma_5d = sigma_5d.view(-1, 1, 1, 1, 1)
    x0_pred = noisy_latents.to(device=video_pred.device, dtype=video_pred.dtype) - sigma_5d * video_pred
    return x0_pred, clean_latents.to(device=video_pred.device, dtype=video_pred.dtype)


def compute_latent_delta_loss(
    video_pred: torch.Tensor,
    video_target: torch.Tensor,
    video_loss_mask: Optional[torch.Tensor],
    context: Optional[Dict[str, torch.Tensor]],
    config: Optional[LatentDeltaLossConfig],
) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
    if config is None or context is None or config.weight <= 0.0:
        return None, {}
    if video_pred.dim() != 5 or video_target.dim() != 5 or video_pred.shape != video_target.shape:
        return None, {}
    if video_pred.shape[2] < 2:
        return None, {}

    selected = _select_prediction_target(video_pred, video_target, context, config)
    if selected is None:
        return None, {}
    pred_seq, target_seq = selected
    sigma = context.get("sigma")
    sample_mask = None
    if isinstance(sigma, torch.Tensor):
        sample_mask = _sigma_sample_mask(sigma, config, pred_seq)

    base_mask = _broadcast_video_mask(video_loss_mask, pred_seq)
    order_values = {part.strip() for part in str(config.order).split("+")}
    total: Optional[torch.Tensor] = None
    metrics: Dict[str, float] = {}

    if "1" in order_values:
        pred_delta = pred_seq[:, :, 1:, :, :] - pred_seq[:, :, :-1, :, :]
        target_delta = target_seq[:, :, 1:, :, :] - target_seq[:, :, :-1, :, :]
        pair_mask = None
        if base_mask is not None and base_mask.shape[2] >= 2:
            pair_mask = base_mask[:, :, 1:, :, :] * base_mask[:, :, :-1, :, :]
        if sample_mask is not None:
            pair_mask = sample_mask if pair_mask is None else pair_mask * sample_mask
        delta_loss = _masked_mean(_loss_fn(pred_delta, target_delta, config), pair_mask)
        total = delta_loss if total is None else total + delta_loss
        metrics["loss/latent_delta"] = float((delta_loss * config.weight).detach().item())

    if "2" in order_values and pred_seq.shape[2] >= 3 and config.second_order_weight > 0.0:
        pred_accel = pred_seq[:, :, 2:, :, :] - 2.0 * pred_seq[:, :, 1:-1, :, :] + pred_seq[:, :, :-2, :, :]
        target_accel = target_seq[:, :, 2:, :, :] - 2.0 * target_seq[:, :, 1:-1, :, :] + target_seq[:, :, :-2, :, :]
        triple_mask = None
        if base_mask is not None and base_mask.shape[2] >= 3:
            triple_mask = base_mask[:, :, 2:, :, :] * base_mask[:, :, 1:-1, :, :] * base_mask[:, :, :-2, :, :]
        if sample_mask is not None:
            triple_mask = sample_mask if triple_mask is None else triple_mask * sample_mask
        accel_loss = _masked_mean(_loss_fn(pred_accel, target_accel, config), triple_mask) * config.second_order_weight
        total = accel_loss if total is None else total + accel_loss
        metrics["loss/latent_accel"] = float((accel_loss * config.weight).detach().item())

    if total is None:
        return None, {}

    weighted = total * config.weight
    metrics["loss/latent_temporal_extra"] = float(weighted.detach().item())
    return weighted, metrics
