from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from typing import Any, Callable, Optional

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _clone_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(v) for v in value)
    return copy.deepcopy(value)


def _safe_abs_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return os.path.abspath(os.path.expanduser(str(path)))


def _path_mtime(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    try:
        return float(os.path.getmtime(path))
    except OSError:
        return None


def _signature_hash(signature: dict[str, Any]) -> str:
    payload = json.dumps(signature, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _torch_load_cpu(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_block_index_spec(spec: Optional[str]) -> set[int]:
    if not spec:
        return set()

    out: set[int] = set()
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if end < start:
                raise ValueError(f"Invalid block range: {token!r}")
            out.update(range(start, end + 1))
        else:
            out.add(int(token))
    return out


def _masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    weighting: Optional[torch.Tensor] = None,
    dtype: torch.dtype,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    pred_f = pred.to(dtype=dtype)
    target_f = target.to(device=pred_f.device, dtype=dtype)
    loss_type = (loss_type or "mse").lower()
    if loss_type in ("mae", "l1"):
        per_elem = torch.nn.functional.l1_loss(pred_f, target_f, reduction="none")
    elif loss_type in ("huber", "smooth_l1"):
        per_elem = torch.nn.functional.smooth_l1_loss(
            pred_f, target_f, reduction="none", beta=float(huber_delta)
        )
    else:
        per_elem = torch.nn.functional.mse_loss(pred_f, target_f, reduction="none")
    if weighting is not None:
        w = weighting.to(device=per_elem.device, dtype=per_elem.dtype)
        while w.dim() < per_elem.dim():
            w = w.view(*w.shape, *([1] * (per_elem.dim() - w.dim())))
        per_elem = per_elem * w
    if mask is None:
        return per_elem.mean()
    mask_f = _expand_mask_like_per_elem(mask, per_elem)
    if mask_f is None:
        return per_elem.mean()
    denom = mask_f.sum().clamp_min(1e-8)
    return (per_elem * mask_f).sum() / denom


def _masked_cosine_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    pred_f = pred.to(dtype=torch.float32)
    tgt_f = tgt.to(device=pred_f.device, dtype=torch.float32)
    channel_dim = 1 if pred_f.dim() >= 3 else -1
    per_elem = 1.0 - torch.nn.functional.cosine_similarity(pred_f, tgt_f, dim=channel_dim)
    per_elem = per_elem.to(dtype=dtype)
    if mask is None:
        return per_elem.mean()
    mask = mask.to(device=per_elem.device)
    if per_elem.dim() == 4 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], mask.shape[1], 1, 1)
    elif per_elem.dim() == 4 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1)
    elif per_elem.dim() == 3 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1)
    mask_f = mask.to(dtype=per_elem.dtype)
    denom = mask_f.mean()
    if denom.item() == 0:
        return per_elem.mean()
    return (per_elem * mask_f).div(denom).mean()


def _masked_kl_softmax_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
    temperature: float = 1.0,
) -> torch.Tensor:
    t = max(float(temperature), 1e-6)
    inv_t = 1.0 / t
    channel_dim = 1 if pred.dim() >= 3 else -1
    pred_f = pred.to(dtype=torch.float32)
    tgt_f = tgt.to(device=pred_f.device, dtype=torch.float32)
    log_p_s = torch.nn.functional.log_softmax(pred_f * inv_t, dim=channel_dim)
    log_p_t = torch.nn.functional.log_softmax(tgt_f * inv_t, dim=channel_dim)
    p_s = log_p_s.exp()
    per_elem = (p_s * (log_p_s - log_p_t)).sum(dim=channel_dim).to(dtype=dtype)
    if mask is None:
        return per_elem.mean()
    mask = mask.to(device=per_elem.device)
    if per_elem.dim() == 4 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], mask.shape[1], 1, 1)
    elif per_elem.dim() == 4 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1)
    elif per_elem.dim() == 3 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1)
    mask_f = mask.to(dtype=per_elem.dtype)
    denom = mask_f.mean()
    if denom.item() == 0:
        return per_elem.mean()
    return (per_elem * mask_f).div(denom).mean()


def _motion_loss_dispatch(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
    loss_type: str,
    kl_temperature: float = 1.0,
) -> torch.Tensor:
    lt = (loss_type or "mse").lower()
    if lt in ("mse", "l2", "mae", "l1", "huber", "smooth_l1"):
        return _masked_mse(pred, tgt, mask, weighting=None, dtype=dtype, loss_type=lt)
    if lt == "cosine":
        return _masked_cosine_loss(pred, tgt, mask, dtype=dtype)
    if lt == "kl_softmax":
        return _masked_kl_softmax_loss(pred, tgt, mask, dtype=dtype, temperature=kl_temperature)
    raise ValueError(f"Unknown motion preservation loss_type: {loss_type}")


def _expand_mask_like_per_elem(mask: Optional[torch.Tensor], per_elem: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
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
    return mask.to(device=per_elem.device, dtype=per_elem.dtype)


def _build_temporal_pair_mask(video_loss_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if video_loss_mask is None or not isinstance(video_loss_mask, torch.Tensor):
        return None
    if video_loss_mask.dim() == 2:
        if video_loss_mask.shape[1] < 2:
            return None
        if video_loss_mask.dtype == torch.bool:
            pair = video_loss_mask[:, 1:] & video_loss_mask[:, :-1]
        else:
            pair = video_loss_mask[:, 1:] * video_loss_mask[:, :-1]
        return pair.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    if video_loss_mask.dim() == 5:
        if video_loss_mask.shape[2] < 2:
            return None
        if video_loss_mask.dtype == torch.bool:
            return video_loss_mask[:, :, 1:, :, :] & video_loss_mask[:, :, :-1, :, :]
        return video_loss_mask[:, :, 1:, :, :] * video_loss_mask[:, :, :-1, :, :]
    return None


def _build_temporal_triplet_mask(video_loss_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if video_loss_mask is None or not isinstance(video_loss_mask, torch.Tensor):
        return None
    if video_loss_mask.dim() == 2:
        if video_loss_mask.shape[1] < 3:
            return None
        if video_loss_mask.dtype == torch.bool:
            triple = video_loss_mask[:, 2:] & video_loss_mask[:, 1:-1] & video_loss_mask[:, :-2]
        else:
            triple = video_loss_mask[:, 2:] * video_loss_mask[:, 1:-1] * video_loss_mask[:, :-2]
        return triple.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    if video_loss_mask.dim() == 5:
        if video_loss_mask.shape[2] < 3:
            return None
        if video_loss_mask.dtype != torch.bool:
            return (
                video_loss_mask[:, :, 2:, :, :]
                * video_loss_mask[:, :, 1:-1, :, :]
                * video_loss_mask[:, :, :-2, :, :]
            )
        return (
            video_loss_mask[:, :, 2:, :, :]
            & video_loss_mask[:, :, 1:-1, :, :]
            & video_loss_mask[:, :, :-2, :, :]
        )
    return None


def _chunked_motion_loss_with_cpu_teacher(
    args: argparse.Namespace,
    student_video_pred: torch.Tensor,
    teacher_video_pred_cpu: torch.Tensor,
    video_loss_mask: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
    chunk_frames: int,
) -> torch.Tensor:
    if chunk_frames <= 0 or student_video_pred.dim() != 5 or teacher_video_pred_cpu.dim() != 5:
        teacher_video_pred = teacher_video_pred_cpu.to(
            device=student_video_pred.device, dtype=dtype, non_blocking=True
        )
        return compute_motion_preservation_loss(
            args, student_video_pred, teacher_video_pred, video_loss_mask, dtype=dtype
        )

    device = student_video_pred.device
    temporal_mode = (
        getattr(args, "motion_preservation_mode", "temporal") == "temporal"
        and student_video_pred.shape[2] > 1
    )
    numerator = torch.zeros((), device=device, dtype=torch.float32)
    denom_mask = torch.zeros((), device=device, dtype=torch.float32)
    unmasked_sum = torch.zeros((), device=device, dtype=torch.float32)
    unmasked_count = 0

    if temporal_mode:
        total_pairs = int(student_video_pred.shape[2] - 1)
        pair_mask = _build_temporal_pair_mask(video_loss_mask)
        for start in range(0, total_pairs, chunk_frames):
            end = min(start + chunk_frames, total_pairs)
            teacher_frames = teacher_video_pred_cpu[:, :, start : (end + 1), :, :].to(
                device=device, dtype=dtype, non_blocking=True
            )
            teacher_delta = teacher_frames[:, :, 1:, :, :] - teacher_frames[:, :, :-1, :, :]
            student_delta = (
                student_video_pred[:, :, (start + 1) : (end + 1), :, :]
                - student_video_pred[:, :, start:end, :, :]
            )
            per_elem = (student_delta.to(torch.float32) - teacher_delta.to(torch.float32)).square()
            unmasked_sum = unmasked_sum + per_elem.sum()
            unmasked_count += int(per_elem.numel())
            if pair_mask is not None:
                mask_chunk = pair_mask[:, :, start:end, :, :]
                mask_f = _expand_mask_like_per_elem(mask_chunk, per_elem)
                if mask_f is not None:
                    numerator = numerator + (per_elem * mask_f).sum()
                    denom_mask = denom_mask + mask_f.sum()
            del teacher_frames, teacher_delta, student_delta, per_elem
    else:
        total_frames = int(student_video_pred.shape[2]) if student_video_pred.dim() == 5 else 0
        for start in range(0, total_frames, chunk_frames):
            end = min(start + chunk_frames, total_frames)
            teacher_chunk = teacher_video_pred_cpu[:, :, start:end, :, :].to(
                device=device, dtype=dtype, non_blocking=True
            )
            student_chunk = student_video_pred[:, :, start:end, :, :]
            per_elem = (student_chunk.to(torch.float32) - teacher_chunk.to(torch.float32)).square()
            unmasked_sum = unmasked_sum + per_elem.sum()
            unmasked_count += int(per_elem.numel())
            if video_loss_mask is not None:
                if video_loss_mask.dim() == 2:
                    mask_chunk = video_loss_mask[:, start:end]
                elif video_loss_mask.dim() == 5:
                    mask_chunk = video_loss_mask[:, :, start:end, :, :]
                else:
                    mask_chunk = video_loss_mask
                mask_f = _expand_mask_like_per_elem(mask_chunk, per_elem)
                if mask_f is not None:
                    numerator = numerator + (per_elem * mask_f).sum()
                    denom_mask = denom_mask + mask_f.sum()
            del teacher_chunk, student_chunk, per_elem

    if denom_mask.item() > 0:
        return numerator / denom_mask
    if unmasked_count == 0:
        return torch.zeros((), device=device, dtype=torch.float32)
    return unmasked_sum / float(unmasked_count)


def compute_motion_preservation_loss(
    args: argparse.Namespace,
    student_video_pred: torch.Tensor,
    teacher_video_pred: torch.Tensor,
    video_loss_mask: Optional[torch.Tensor],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    chunk_frames = int(getattr(args, "motion_preservation_teacher_chunk_frames", 0) or 0)
    second_order_weight = float(getattr(args, "motion_preservation_second_order_weight", 0.0) or 0.0)
    loss_type_initial = str(getattr(args, "motion_preservation_loss_type", "mse") or "mse").lower()
    use_chunked = second_order_weight <= 0.0 and loss_type_initial in ("mse", "l2")
    if (
        isinstance(teacher_video_pred, torch.Tensor)
        and teacher_video_pred.device.type == "cpu"
        and chunk_frames > 0
        and use_chunked
    ):
        return _chunked_motion_loss_with_cpu_teacher(
            args,
            student_video_pred,
            teacher_video_pred,
            video_loss_mask,
            dtype=dtype,
            chunk_frames=chunk_frames,
        )
    if (
        isinstance(teacher_video_pred, torch.Tensor)
        and teacher_video_pred.device.type == "cpu"
        and chunk_frames > 0
        and not use_chunked
        and not bool(getattr(args, "_motion_second_order_chunk_warned", False))
    ):
        logger.info(
            "motion_preservation_second_order_weight > 0 disables chunked CPU teacher loss path. "
            "Teacher targets will be moved to GPU for replay loss."
        )
        setattr(args, "_motion_second_order_chunk_warned", True)
    if isinstance(teacher_video_pred, torch.Tensor) and teacher_video_pred.device != student_video_pred.device:
        teacher_video_pred = teacher_video_pred.to(device=student_video_pred.device, dtype=dtype, non_blocking=True)
    loss_type = str(getattr(args, "motion_preservation_loss_type", "mse") or "mse").lower()
    kl_t = float(getattr(args, "motion_preservation_kl_temperature", 1.0) or 1.0)
    if (
        getattr(args, "motion_preservation_mode", "temporal") == "temporal"
        and student_video_pred.dim() == 5
        and teacher_video_pred.dim() == 5
        and student_video_pred.shape[2] > 1
    ):
        student_delta = student_video_pred[:, :, 1:, :, :] - student_video_pred[:, :, :-1, :, :]
        teacher_delta = teacher_video_pred[:, :, 1:, :, :] - teacher_video_pred[:, :, :-1, :, :]
        pair_mask = _build_temporal_pair_mask(video_loss_mask)
        first_order_loss = _motion_loss_dispatch(
            student_delta, teacher_delta, pair_mask,
            dtype=dtype, loss_type=loss_type, kl_temperature=kl_t,
        )
        if second_order_weight > 0.0 and student_video_pred.shape[2] > 2:
            student_accel = (
                student_video_pred[:, :, 2:, :, :]
                - (2.0 * student_video_pred[:, :, 1:-1, :, :])
                + student_video_pred[:, :, :-2, :, :]
            )
            teacher_accel = (
                teacher_video_pred[:, :, 2:, :, :]
                - (2.0 * teacher_video_pred[:, :, 1:-1, :, :])
                + teacher_video_pred[:, :, :-2, :, :]
            )
            triplet_mask = _build_temporal_triplet_mask(video_loss_mask)
            second_order_loss = _motion_loss_dispatch(
                student_accel, teacher_accel, triplet_mask,
                dtype=dtype, loss_type=loss_type, kl_temperature=kl_t,
            )
            return first_order_loss + float(second_order_weight) * second_order_loss
        return first_order_loss
    return _motion_loss_dispatch(
        student_video_pred, teacher_video_pred, video_loss_mask,
        dtype=dtype, loss_type=loss_type, kl_temperature=kl_t,
    )


def compute_motion_sigma_weights(sigmas: list[float], args: argparse.Namespace) -> Optional[list[float]]:
    if len(sigmas) <= 1:
        return None
    mode = str(getattr(args, "motion_preservation_sigma_sampling", "uniform") or "uniform").lower()
    if mode == "uniform" or mode != "logsnr":
        return None
    sigma_tensor = torch.tensor(sigmas, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
    logsnr = torch.log(((1.0 - sigma_tensor) ** 2) / (sigma_tensor**2))
    weights = 1.0 / (1.0 + logsnr.abs())
    power = float(getattr(args, "motion_preservation_sigma_sampling_power", 1.0) or 1.0)
    weights = weights.clamp_min(1e-8).pow(power).to(torch.float32)
    weights = weights / weights.sum().clamp_min(1e-8)
    return [float(x) for x in weights.tolist()]


def sample_motion_sigma_index(sigmas: list[float], args: argparse.Namespace) -> int:
    if len(sigmas) <= 1:
        return 0
    weights = compute_motion_sigma_weights(sigmas, args)
    if not weights:
        return random.randrange(len(sigmas))
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    return int(torch.multinomial(weight_tensor, num_samples=1, replacement=True).item())


def resolve_motion_replay_sigmas(args: argparse.Namespace) -> list[float]:
    explicit = getattr(args, "motion_preservation_sigma_values", None)
    if explicit:
        tokens = str(explicit).replace(";", ",").split(",")
        sigmas = [float(t.strip()) for t in tokens if t.strip()]
        if not sigmas:
            raise ValueError("motion_preservation_sigma_values is set but no valid sigma values were parsed")
    else:
        num_sigmas = int(getattr(args, "motion_preservation_num_sigmas", 1) or 1)
        if num_sigmas <= 1:
            return []
        sigma_min = float(getattr(args, "motion_preservation_sigma_min", 0.2))
        sigma_max = float(getattr(args, "motion_preservation_sigma_max", 0.8))
        if sigma_max < sigma_min:
            raise ValueError(
                f"motion_preservation_sigma_max must be >= motion_preservation_sigma_min. "
                f"Got min={sigma_min}, max={sigma_max}"
            )
        sigmas = torch.linspace(sigma_min, sigma_max, steps=num_sigmas, dtype=torch.float32).tolist()
    for sigma in sigmas:
        if not (0.0 <= float(sigma) <= 1.0):
            raise ValueError(f"All motion replay sigmas must be in [0,1]. Got: {sigma}")
    return sorted(set(float(sigma) for sigma in sigmas))


def resolve_motion_anchor_cache_size(args: argparse.Namespace, *, num_train_items: int) -> int:
    requested = int(getattr(args, "motion_preservation_anchor_cache_size", 0) or 0)
    auto_enabled = bool(getattr(args, "motion_preservation_anchor_cache_auto", False))
    if not auto_enabled:
        return requested
    ratio = float(getattr(args, "motion_preservation_anchor_cache_auto_ratio", 0.2) or 0.2)
    min_size = int(getattr(args, "motion_preservation_anchor_cache_auto_min", 8) or 8)
    max_size = int(getattr(args, "motion_preservation_anchor_cache_auto_max", 256) or 256)
    derived = int(math.ceil(max(1, int(num_train_items)) * ratio))
    return max(min_size, min(max_size, derived))


def build_noisy_input_for_sigma(
    anchor_latents: torch.Tensor,
    anchor_noise: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma_f = float(sigma)
    sigma_view = torch.full(
        (anchor_latents.shape[0], 1, 1, 1, 1),
        sigma_f,
        device=anchor_latents.device,
        dtype=anchor_latents.dtype,
    )
    noisy_input = (1.0 - sigma_view) * anchor_latents + sigma_view * anchor_noise
    timesteps = torch.full(
        (anchor_latents.shape[0],),
        sigma_f * 1000.0,
        device=anchor_latents.device,
        dtype=torch.float32,
    )
    return noisy_input, timesteps


def build_motion_anchor_cache_signature(
    args: argparse.Namespace,
    *,
    cache_source_name: str,
    cache_dataset_config_path: Optional[str],
    replay_sigmas: list[float],
    use_attn_pres: bool,
    max_queries: int,
    max_keys: int,
) -> dict[str, Any]:
    ckpt_path = _safe_abs_path(getattr(args, "ltx2_checkpoint", None))
    ds_cfg = _safe_abs_path(cache_dataset_config_path if cache_dataset_config_path else getattr(args, "dataset_config", None))
    return {
        "schema": 1,
        "kind": "motion_anchor_cache",
        "checkpoint": ckpt_path,
        "checkpoint_mtime": _path_mtime(ckpt_path),
        "cache_source_name": cache_source_name,
        "dataset_config": ds_cfg,
        "dataset_config_mtime": _path_mtime(ds_cfg),
        "ltx_mode": getattr(args, "ltx_mode", "video"),
        "anchor_source": getattr(args, "motion_preservation_anchor_source", "synthetic"),
        "anchor_cache_size": int(getattr(args, "motion_preservation_anchor_cache_size", 0) or 0),
        "motion_mode": getattr(args, "motion_preservation_mode", "temporal"),
        "num_sigmas": int(getattr(args, "motion_preservation_num_sigmas", 1) or 1),
        "sigma_values": [float(s) for s in replay_sigmas],
        "synthetic_frames": int(getattr(args, "motion_preservation_synthetic_frames", 8) or 8),
        "synthetic_temporal_corr": float(getattr(args, "motion_preservation_synthetic_temporal_corr", 0.92)),
        "synthetic_dataset_mix": float(getattr(args, "motion_preservation_synthetic_dataset_mix", 0.25)),
        "synthetic_content_seeded": bool(getattr(args, "motion_preservation_synthetic_content_seeded", True)),
        "attention_preservation": bool(use_attn_pres),
        "attention_queries": int(max_queries),
        "attention_keys": int(max_keys),
        "attention_per_head": bool(getattr(args, "motion_attention_preservation_per_head", False)),
        "attention_blocks": getattr(args, "motion_attention_preservation_blocks", None),
    }


def load_motion_anchor_cache(path: str, signature: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
    cache_path = _safe_abs_path(path)
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        payload = _torch_load_cpu(cache_path)
        if not isinstance(payload, dict):
            logger.warning("Motion anchor cache has invalid payload type, rebuilding: %s", cache_path)
            return None
        expected_hash = _signature_hash(signature)
        cached_hash = payload.get("signature_hash")
        if cached_hash != expected_hash:
            logger.info("Motion anchor cache signature mismatch; rebuilding: %s", cache_path)
            return None
        entries = payload.get("entries")
        if not isinstance(entries, list):
            logger.warning("Motion anchor cache missing entries list, rebuilding: %s", cache_path)
            return None
        if len(entries) == 0:
            logger.warning("Motion anchor cache is empty, rebuilding: %s", cache_path)
            return None
        if len(entries) > 0 and (not isinstance(entries[0], dict) or "anchor_latents" not in entries[0]):
            logger.warning("Motion anchor cache entry format invalid, rebuilding: %s", cache_path)
            return None
        logger.info("Loaded %d motion anchors from cache: %s", len(entries), cache_path)
        return entries
    except Exception:
        logger.exception("Failed to load motion anchor cache, rebuilding: %s", cache_path)
        return None


def save_motion_anchor_cache(path: str, signature: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    cache_path = _safe_abs_path(path)
    if not cache_path:
        return
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    payload = {
        "version": 1,
        "signature_hash": _signature_hash(signature),
        "signature": signature,
        "created_at": float(time.time()),
        "entries": entries,
    }
    torch.save(payload, cache_path)
    logger.info("Saved motion anchor cache: %s (entries=%d)", cache_path, len(entries))


def _extract_motion_anchor_batch(batch: dict) -> dict:
    keep_keys = (
        "conditions",
        "text",
        "text_mask",
        "frame_rate",
        "ref_latents",
        "reference_latents",
        "audio_latents",
        "audio_lengths",
    )
    return {k: _clone_to_cpu(batch[k]) for k in keep_keys if k in batch}


def build_synthetic_motion_latents(
    base_latents: torch.Tensor,
    *,
    target_frames: int,
    temporal_corr: float,
    content_seeded: bool = True,
) -> torch.Tensor:
    if base_latents.dim() != 5:
        raise ValueError(f"Expected 5D base latents, got shape={tuple(base_latents.shape)}")

    batch_size, channels, _, height, width = base_latents.shape
    frames = max(2, int(target_frames))
    corr = max(0.0, min(0.999, float(temporal_corr)))
    synth = torch.empty(
        (batch_size, channels, frames, height, width),
        device=base_latents.device,
        dtype=base_latents.dtype,
    )

    if content_seeded:
        prev = base_latents[:, :, 0, :, :].clone()
        synth[:, :, 0, :, :] = prev
        if corr >= 0.999:
            for frame_idx in range(1, frames):
                synth[:, :, frame_idx, :, :] = prev
        else:
            noise_scale = math.sqrt(max(1e-6, 1.0 - corr * corr))
            for frame_idx in range(1, frames):
                prev = corr * prev + noise_scale * torch.randn_like(prev)
                synth[:, :, frame_idx, :, :] = prev
    else:
        prev = torch.randn((batch_size, channels, height, width), device=base_latents.device, dtype=base_latents.dtype)
        synth[:, :, 0, :, :] = prev
        if corr >= 0.999:
            for frame_idx in range(1, frames):
                synth[:, :, frame_idx, :, :] = prev
        else:
            noise_scale = math.sqrt(max(1e-6, 1.0 - corr * corr))
            for frame_idx in range(1, frames):
                prev = corr * prev + noise_scale * torch.randn_like(prev)
                synth[:, :, frame_idx, :, :] = prev
        base_f32 = base_latents.to(torch.float32)
        synth_f32 = synth.to(torch.float32)
        base_mean = base_f32.mean()
        base_std = base_f32.std(unbiased=False).clamp_min(1e-6)
        synth_mean = synth_f32.mean()
        synth_std = synth_f32.std(unbiased=False).clamp_min(1e-6)
        synth = (synth - synth_mean.to(dtype=synth.dtype)) * (base_std / synth_std).to(dtype=synth.dtype)
        synth = synth + base_mean.to(dtype=synth.dtype)
    return synth


def build_motion_anchor_cache(
    trainer: Any,
    args: argparse.Namespace,
    accelerator: Any,
    transformer: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    noise_scheduler: Any,
    attention_modules: Optional[list[tuple[str, torch.nn.Module]]] = None,
    *,
    cache_source_name: str = "train_dataset",
    cache_dataset_config_path: Optional[str] = None,
    normalize_batch_fn: Optional[Callable[[dict], dict]] = None,
) -> list[dict[str, Any]]:
    cache_size = int(getattr(args, "motion_preservation_anchor_cache_size", 0) or 0)
    if cache_size <= 0:
        return []

    use_attn_pres = bool(getattr(args, "motion_attention_preservation", False) and attention_modules)
    max_queries = int(getattr(args, "motion_attention_preservation_queries", 32) or 32)
    max_keys = int(getattr(args, "motion_attention_preservation_keys", 64) or 64)
    replay_sigmas = resolve_motion_replay_sigmas(args)
    anchor_cache_path = getattr(args, "motion_preservation_anchor_cache_path", None)
    anchor_cache_rebuild = bool(getattr(args, "motion_preservation_anchor_cache_rebuild", False))
    anchor_cache_signature = build_motion_anchor_cache_signature(
        args,
        cache_source_name=cache_source_name,
        cache_dataset_config_path=cache_dataset_config_path,
        replay_sigmas=replay_sigmas,
        use_attn_pres=use_attn_pres,
        max_queries=max_queries,
        max_keys=max_keys,
    )
    anchor_source = str(getattr(args, "motion_preservation_anchor_source", "synthetic") or "synthetic").lower()
    synthetic_frames = int(getattr(args, "motion_preservation_synthetic_frames", 8) or 8)
    synthetic_temporal_corr = float(getattr(args, "motion_preservation_synthetic_temporal_corr", 0.92))
    synthetic_dataset_mix = float(getattr(args, "motion_preservation_synthetic_dataset_mix", 0.25))
    synthetic_content_seeded = bool(getattr(args, "motion_preservation_synthetic_content_seeded", True))

    entries: list[dict[str, Any]] = []
    max_attempts = max(cache_size * 4, cache_size)
    attempt = 0
    single_frame_dataset_anchor_count = 0
    synthetic_anchor_count = 0
    dataset_anchor_count = 0
    rejected_not_tensor = 0
    rejected_bad_dim = 0
    batches_without_timesteps = 0
    rejected_non_dict_pred = 0
    rejected_skip_step = 0
    rejected_skip_reasons: dict[str, int] = {}
    original_first_frame_p = float(getattr(args, "ltx2_first_frame_conditioning_p", 0.0))
    was_training = transformer.training

    if anchor_cache_path and not anchor_cache_rebuild:
        cached_entries = load_motion_anchor_cache(anchor_cache_path, anchor_cache_signature)
        if cached_entries is not None:
            return cached_entries
    elif anchor_cache_path and anchor_cache_rebuild:
        logger.info("Motion anchor cache rebuild requested; ignoring existing cache: %s", anchor_cache_path)

    logger.info(
        "Building motion anchor cache from base model outputs: anchor_source=%s data_source=%s size=%d",
        anchor_source,
        cache_source_name,
        cache_size,
    )
    if anchor_source in {"synthetic", "hybrid"}:
        logger.info(
            "Motion prior synthetic anchors: frames=%d temporal_corr=%.3f dataset_mix=%.2f content_seeded=%s",
            synthetic_frames,
            synthetic_temporal_corr,
            synthetic_dataset_mix,
            synthetic_content_seeded,
        )
    anchor_start_time = time.time()
    if replay_sigmas:
        logger.info(
            "Motion anchor cache will store multi-sigma teacher targets: sigmas=%s sampling=%s power=%.3f",
            replay_sigmas,
            str(getattr(args, "motion_preservation_sigma_sampling", "uniform")),
            float(getattr(args, "motion_preservation_sigma_sampling_power", 1.0) or 1.0),
        )
    pbar_anchor = tqdm(
        total=cache_size,
        desc="prep: motion anchors",
        leave=False,
        disable=not accelerator.is_local_main_process,
    )

    transformer.eval()
    setattr(args, "ltx2_first_frame_conditioning_p", 0.0)

    try:
        with torch.no_grad():
            for batch in train_dataloader:
                if len(entries) >= cache_size or attempt >= max_attempts:
                    break
                attempt += 1

                if normalize_batch_fn is not None:
                    batch = normalize_batch_fn(batch)
                latents = batch.get("latents")
                if isinstance(latents, dict):
                    latents = latents.get("latents")
                if not isinstance(latents, torch.Tensor):
                    rejected_not_tensor += 1
                    continue
                if latents.dim() != 5:
                    rejected_bad_dim += 1
                    continue

                latents_tensor = trainer.scale_shift_latents(latents)
                use_synthetic_anchor = anchor_source == "synthetic" or (
                    anchor_source == "hybrid" and random.random() > synthetic_dataset_mix
                )
                if use_synthetic_anchor:
                    anchor_latents = build_synthetic_motion_latents(
                        latents_tensor,
                        target_frames=synthetic_frames,
                        temporal_corr=synthetic_temporal_corr,
                        content_seeded=synthetic_content_seeded,
                    )
                    synthetic_anchor_count += 1
                else:
                    anchor_latents = latents_tensor.clone()
                    dataset_anchor_count += 1
                anchor_noise = torch.randn_like(anchor_latents)

                batch_timesteps = batch.get("timesteps")
                if batch_timesteps is None:
                    batches_without_timesteps += 1

                teacher_attn_maps = None
                teacher_attn_maps_multi: list[Optional[dict[str, torch.Tensor]]] = []
                teacher_video_preds_multi: list[Any] = []
                accepted_sigmas: list[float] = []
                anchor_noisy_input = None
                anchor_model_timesteps = None
                multi_sigma_failed = False

                sigma_iter = replay_sigmas if replay_sigmas else [None]
                for sigma_value in sigma_iter:
                    if sigma_value is None:
                        cur_noisy_input, cur_model_timesteps = trainer.get_noisy_model_input_and_timesteps(
                            args,
                            anchor_noise,
                            anchor_latents,
                            batch_timesteps,
                            noise_scheduler,
                            accelerator.device,
                            trainer.dit_dtype,
                        )
                    else:
                        cur_noisy_input, cur_model_timesteps = build_noisy_input_for_sigma(
                            anchor_latents,
                            anchor_noise,
                            sigma_value,
                        )

                    if anchor_noisy_input is None:
                        anchor_noisy_input = cur_noisy_input
                        anchor_model_timesteps = cur_model_timesteps

                    if use_attn_pres:
                        with AttentionMapRecorder(
                            attention_modules or [],
                            max_queries=max_queries,
                            max_keys=max_keys,
                            capture_grad=False,
                            keep_heads=bool(getattr(args, "motion_attention_preservation_per_head", False)),
                        ) as attn_recorder:
                            teacher_pred, _ = trainer.call_dit(
                                args,
                                accelerator,
                                transformer,
                                anchor_latents,
                                batch,
                                anchor_noise,
                                cur_noisy_input,
                                cur_model_timesteps,
                                trainer.dit_dtype,
                            )
                        cur_teacher_attn_maps = None
                        if attn_recorder.maps:
                            cur_teacher_attn_maps = {
                                k: v.detach().to(dtype=torch.float16).cpu()
                                for k, v in attn_recorder.maps.items()
                            }
                        teacher_attn_maps_multi.append(cur_teacher_attn_maps)
                        if teacher_attn_maps is None:
                            teacher_attn_maps = cur_teacher_attn_maps
                    else:
                        teacher_pred, _ = trainer.call_dit(
                            args,
                            accelerator,
                            transformer,
                            anchor_latents,
                            batch,
                            anchor_noise,
                            cur_noisy_input,
                            cur_model_timesteps,
                            trainer.dit_dtype,
                        )

                    if not isinstance(teacher_pred, dict):
                        rejected_non_dict_pred += 1
                        multi_sigma_failed = True
                        break
                    if teacher_pred.get("_skip_step"):
                        rejected_skip_step += 1
                        reason = str(teacher_pred.get("skip_reason", "unknown"))
                        rejected_skip_reasons[reason] = rejected_skip_reasons.get(reason, 0) + 1
                        multi_sigma_failed = True
                        break

                    teacher_video_preds_multi.append(_clone_to_cpu(teacher_pred["video_pred"]))
                    if sigma_value is not None:
                        accepted_sigmas.append(float(sigma_value))

                if multi_sigma_failed:
                    continue
                if anchor_noisy_input is None or anchor_model_timesteps is None:
                    continue

                if not use_synthetic_anchor and int(anchor_latents.shape[2]) <= 1:
                    single_frame_dataset_anchor_count += 1
                entries.append(
                    {
                        "anchor_latents": _clone_to_cpu(anchor_latents),
                        "anchor_noise": _clone_to_cpu(anchor_noise),
                        "anchor_noisy_input": _clone_to_cpu(anchor_noisy_input),
                        "anchor_model_timesteps": _clone_to_cpu(anchor_model_timesteps),
                        "anchor_batch": _extract_motion_anchor_batch(batch),
                        "teacher_video_pred": teacher_video_preds_multi[0],
                        "teacher_video_preds": teacher_video_preds_multi,
                        "anchor_sigmas": accepted_sigmas,
                        "teacher_attention_maps": teacher_attn_maps,
                        "teacher_attention_maps_multi": teacher_attn_maps_multi,
                        "anchor_source": "synthetic" if use_synthetic_anchor else "dataset",
                    }
                )
                pbar_anchor.update(1)
                if accelerator.is_local_main_process:
                    pbar_anchor.set_postfix(built=len(entries), attempts=attempt)
    finally:
        pbar_anchor.close()
        setattr(args, "ltx2_first_frame_conditioning_p", original_first_frame_p)
        if was_training:
            transformer.train()

    if len(entries) < cache_size:
        logger.warning("Built %d/%d motion anchors (max_attempts=%d).", len(entries), cache_size, max_attempts)
        if attempt > 0:
            logger.warning(
                "Anchor build diagnostics: not_tensor=%d bad_dim=%d non_dict_pred=%d skip_step=%d batches_without_timesteps=%d",
                rejected_not_tensor,
                rejected_bad_dim,
                rejected_non_dict_pred,
                rejected_skip_step,
                batches_without_timesteps,
            )
            if rejected_skip_reasons:
                logger.warning("Anchor skip reasons: %s", rejected_skip_reasons)
    else:
        logger.info("Built %d motion anchors in %.1fs.", len(entries), time.time() - anchor_start_time)
    if len(entries) > 0:
        logger.info("Motion anchor source mix: dataset=%d synthetic=%d", dataset_anchor_count, synthetic_anchor_count)
    if getattr(args, "motion_preservation_mode", "temporal") == "temporal" and single_frame_dataset_anchor_count > 0:
        logger.info(
            "Dataset anchor replay: %d/%d anchors are single-frame; temporal mode falls back to full-output replay for those anchors.",
            single_frame_dataset_anchor_count,
            max(1, dataset_anchor_count),
        )

    if anchor_cache_path and entries:
        save_motion_anchor_cache(anchor_cache_path, anchor_cache_signature, entries)

    return entries


def _extract_attn1_block_index(module_name: str) -> Optional[int]:
    match = re.search(r"(?:^|\.)(?:model\.)?transformer_blocks\.(\d+)\.attn1$", module_name)
    if match is None:
        return None
    return int(match.group(1))


def collect_motion_attention_modules(
    transformer: torch.nn.Module,
    block_spec: Optional[str],
) -> list[tuple[str, torch.nn.Module]]:
    from musubi_tuner.ltx_2.model.transformer.attention import Attention

    block_filter = _parse_block_index_spec(block_spec)
    apply_filter = len(block_filter) > 0
    out: list[tuple[str, torch.nn.Module]] = []
    for module_name, module in transformer.named_modules():
        if not isinstance(module, Attention):
            continue
        block_index = _extract_attn1_block_index(module_name)
        if block_index is None:
            continue
        if apply_filter and block_index not in block_filter:
            continue
        out.append((module_name, module))
    return out


def filter_motion_attention_modules_for_swap(
    modules: list[tuple[str, torch.nn.Module]],
    *,
    transformer: torch.nn.Module,
    accelerator: Any,
    blocks_to_swap: int,
) -> list[tuple[str, torch.nn.Module]]:
    if not modules:
        return modules
    swap_count = int(blocks_to_swap or 0)
    if swap_count <= 0:
        return modules

    base_model = transformer.model if hasattr(transformer, "model") else transformer
    transformer_blocks = getattr(base_model, "transformer_blocks", None)
    if transformer_blocks is None:
        logger.warning(
            "motion_attention_preservation: block-swap filtering skipped because transformer_blocks are unavailable."
        )
        return modules

    num_blocks = len(transformer_blocks)
    if num_blocks <= 0:
        return modules

    swap_start = max(0, num_blocks - swap_count)
    if swap_start <= 0:
        logger.info(
            "motion_attention_preservation: all blocks appear swappable (blocks=%d, blocks_to_swap=%d); "
            "keeping all attention modules.",
            num_blocks,
            swap_count,
        )
        return modules

    kept: list[tuple[str, torch.nn.Module]] = []
    dropped = 0
    for module_name, module in modules:
        block_idx = _extract_attn1_block_index(module_name)
        if block_idx is None or block_idx < swap_start:
            kept.append((module_name, module))
        else:
            dropped += 1

    if not kept:
        logger.warning(
            "motion_attention_preservation: block-swap filtering would drop all modules "
            "(blocks=%d, blocks_to_swap=%d, requested=%d). Keeping original module list.",
            num_blocks,
            swap_count,
            len(modules),
        )
        return modules

    logger.info(
        "motion_attention_preservation: filtered modules for block swap on %s: kept=%d dropped=%d "
        "(non-swapped blocks: 0-%d, swapped blocks: %d-%d).",
        str(accelerator.device),
        len(kept),
        dropped,
        max(0, swap_start - 1),
        swap_start,
        max(0, num_blocks - 1),
    )
    return kept


class AttentionMapRecorder:
    def __init__(
        self,
        modules: list[tuple[str, torch.nn.Module]],
        *,
        max_queries: int,
        max_keys: int,
        capture_grad: bool,
        keep_heads: bool,
    ) -> None:
        self.modules = modules
        self.max_queries = max(1, int(max_queries))
        self.max_keys = max(1, int(max_keys))
        self.capture_grad = capture_grad
        self.keep_heads = bool(keep_heads)
        self.maps: dict[str, torch.Tensor] = {}
        self._active_module_names: set[str] = set()

    def collect_maps(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, module in self.modules:
            if name not in self._active_module_names:
                continue
            attn = getattr(module, "_motion_record_attn_map", None)
            if isinstance(attn, torch.Tensor):
                out[name] = attn
        self.maps = out
        return out

    def deactivate(self) -> None:
        for name, module in self.modules:
            if name not in self._active_module_names:
                continue
            setattr(module, "_motion_record_enabled", False)
            setattr(module, "_motion_record_attn_map", None)
        self._active_module_names = set()

    def __enter__(self) -> "AttentionMapRecorder":
        self.maps = {}
        self._active_module_names = set()
        for name, module in self.modules:
            if not hasattr(module, "_motion_record_enabled"):
                continue
            setattr(module, "_motion_record_enabled", True)
            setattr(module, "_motion_record_max_queries", int(self.max_queries))
            setattr(module, "_motion_record_max_keys", int(self.max_keys))
            setattr(module, "_motion_record_capture_grad", bool(self.capture_grad))
            setattr(module, "_motion_record_keep_heads", bool(self.keep_heads))
            setattr(module, "_motion_record_attn_map", None)
            self._active_module_names.add(name)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.collect_maps()
        self.deactivate()
        return False
