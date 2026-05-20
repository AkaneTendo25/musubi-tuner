"""Attention-derived loss weights for LTX-2 audio/video training."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


_DIRECTION_TO_MODULE = {
    "a2v": "audio_to_video_attn",
    "v2a": "video_to_audio_attn",
}
_MODALITY_TO_DIRECTION = {
    "video": "a2v",
    "audio": "v2a",
}


@dataclass(frozen=True)
class AVAttentionLossConfig:
    max_weight: float = 1.5
    warmup_steps: int = 400
    max_queries: int = 64
    max_keys: int = 64

    def current_max_weight(self, step: int | None) -> float:
        if self.max_weight <= 1.0:
            return 1.0
        if self.warmup_steps <= 0:
            return float(self.max_weight)
        progress = min(max(float(step or 0) / float(self.warmup_steps), 0.0), 1.0)
        return 1.0 + (float(self.max_weight) - 1.0) * progress


class AVAttentionLossRecorderHandle:
    def __init__(self, previous: list[tuple[torch.nn.Module, dict[str, Any]]]) -> None:
        self._previous = previous

    def remove(self) -> None:
        for module, attrs in self._previous:
            for key, value in attrs.items():
                setattr(module, key, value)
            setattr(module, "_motion_record_attn_map", None)
        self._previous.clear()


def collect_av_attention_loss_modules(transformer: torch.nn.Module) -> list[tuple[str, str, torch.nn.Module]]:
    """Collect A2V/V2A cross-modal attention modules from an LTX-2 transformer."""

    from musubi_tuner.ltx_2.model.transformer.attention import Attention

    out: list[tuple[str, str, torch.nn.Module]] = []
    pattern = re.compile(r"(?:^|\.)(?:model\.)?transformer_blocks\.(\d+)\.(audio_to_video_attn|video_to_audio_attn)$")
    for module_name, module in transformer.named_modules():
        if not isinstance(module, Attention):
            continue
        match = pattern.search(module_name)
        if match is None:
            continue
        attn_name = match.group(2)
        direction = "a2v" if attn_name == "audio_to_video_attn" else "v2a"
        out.append((module_name, direction, module))
    return out


def install_av_attention_loss_recorders(
    modules: list[tuple[str, str, torch.nn.Module]],
    config: AVAttentionLossConfig,
) -> AVAttentionLossRecorderHandle:
    """Enable detached attention-map capture on the selected modules."""

    if not modules:
        raise ValueError("AV attention loss weighting matched no A2V/V2A attention modules")

    previous: list[tuple[torch.nn.Module, dict[str, Any]]] = []
    for _name, _direction, module in modules:
        attrs = {
            "_motion_record_enabled": getattr(module, "_motion_record_enabled", False),
            "_motion_record_max_queries": getattr(module, "_motion_record_max_queries", 32),
            "_motion_record_max_keys": getattr(module, "_motion_record_max_keys", 64),
            "_motion_record_capture_grad": getattr(module, "_motion_record_capture_grad", False),
            "_motion_record_keep_heads": getattr(module, "_motion_record_keep_heads", False),
            "_motion_record_attn_map": getattr(module, "_motion_record_attn_map", None),
        }
        previous.append((module, attrs))
        setattr(module, "_motion_record_enabled", True)
        setattr(module, "_motion_record_max_queries", int(config.max_queries))
        setattr(module, "_motion_record_max_keys", int(config.max_keys))
        setattr(module, "_motion_record_capture_grad", False)
        setattr(module, "_motion_record_keep_heads", False)
        setattr(module, "_motion_record_attn_map", None)

    return AVAttentionLossRecorderHandle(previous)


def apply_av_attention_loss_weighting(
    per_elem: torch.Tensor,
    modules: list[tuple[str, str, torch.nn.Module]],
    config: AVAttentionLossConfig | None,
    *,
    modality: str,
    global_step: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multiply a per-element denoising loss by detached query-importance weights."""

    if config is None or not modules or modality not in _MODALITY_TO_DIRECTION:
        return per_elem, {}

    direction = _MODALITY_TO_DIRECTION[modality]
    query_scores = _collect_query_scores(modules, direction)
    if query_scores is None:
        return per_elem, {}

    loss_weight = _expand_query_scores_to_loss_weight(query_scores, per_elem, config.current_max_weight(global_step))
    if loss_weight is None:
        return per_elem, {}

    weighted = per_elem * loss_weight.to(device=per_elem.device, dtype=per_elem.dtype)
    detached = loss_weight.detach().to(torch.float32)
    return weighted, {
        f"av_attention_loss/{modality}_mean": float(detached.mean().item()),
        f"av_attention_loss/{modality}_max": float(detached.max().item()),
    }


def _collect_query_scores(
    modules: list[tuple[str, str, torch.nn.Module]],
    direction: str,
) -> torch.Tensor | None:
    scores: list[torch.Tensor] = []
    for _name, module_direction, module in modules:
        if module_direction != direction:
            continue
        attn = getattr(module, "_motion_record_attn_map", None)
        if not isinstance(attn, torch.Tensor):
            continue
        attn = attn.detach().to(torch.float32)
        if attn.dim() == 4:
            attn = attn.mean(dim=1)
        if attn.dim() != 3 or attn.shape[-1] <= 0:
            continue
        scores.append(attn.max(dim=-1).values)

    if not scores:
        return None

    min_q = min(int(score.shape[1]) for score in scores)
    if min_q <= 0:
        return None
    scores = [score[:, :min_q] for score in scores]
    return torch.stack(scores, dim=0).mean(dim=0)


def _expand_query_scores_to_loss_weight(
    scores: torch.Tensor,
    per_elem: torch.Tensor,
    current_max_weight: float,
) -> torch.Tensor | None:
    if per_elem.dim() < 2 or scores.dim() != 2:
        return None
    if per_elem.shape[0] != scores.shape[0]:
        if scores.shape[0] == 1:
            scores = scores.expand(per_elem.shape[0], -1)
        else:
            return None

    normalized = _normalize_scores(scores)
    weights = 1.0 + (float(current_max_weight) - 1.0) * normalized

    if per_elem.dim() == 5:
        seq_len = int(per_elem.shape[2] * per_elem.shape[3] * per_elem.shape[4])
        weights = _resize_query_weights(weights, seq_len)
        return weights.view(per_elem.shape[0], 1, per_elem.shape[2], per_elem.shape[3], per_elem.shape[4])
    if per_elem.dim() == 4:
        seq_len = int(per_elem.shape[2] * per_elem.shape[3])
        weights = _resize_query_weights(weights, seq_len)
        return weights.view(per_elem.shape[0], 1, per_elem.shape[2], per_elem.shape[3])
    if per_elem.dim() == 3:
        seq_len = int(per_elem.shape[1])
        weights = _resize_query_weights(weights, seq_len)
        return weights.view(per_elem.shape[0], seq_len, 1)
    if per_elem.dim() == 2:
        seq_len = int(per_elem.shape[1])
        return _resize_query_weights(weights, seq_len)
    return None


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    mins = scores.min(dim=1, keepdim=True).values
    maxs = scores.max(dim=1, keepdim=True).values
    span = (maxs - mins).clamp_min(1e-8)
    normalized = (scores - mins) / span
    flat = (maxs - mins) <= 1e-8
    if bool(flat.any().item()):
        normalized = torch.where(flat.expand_as(normalized), torch.zeros_like(normalized), normalized)
    return normalized.clamp(0.0, 1.0)


def _resize_query_weights(weights: torch.Tensor, seq_len: int) -> torch.Tensor:
    if int(weights.shape[1]) == seq_len:
        return weights
    return F.interpolate(weights.unsqueeze(1), size=seq_len, mode="linear", align_corners=False).squeeze(1)
