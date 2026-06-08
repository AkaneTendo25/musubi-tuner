"""No-reference image-quality reward for video frames via IQA-PyTorch.

This reward wraps off-the-shelf IQA models such as TOPIQ, LIQE, MUSIQ, MANIQA,
or CLIP-IQA from the IQA-PyTorch package:

    pip install pyiqa

The default mode is intended for "more real detail, less smear" RL:

  - score several decoded video frames with a no-reference IQA model;
  - score a deliberately degraded copy of the same frames;
  - reward both absolute perceptual quality and the quality gap to the degraded
    copy.

The gap is useful because it asks whether the IQA model sees meaningful local
information that is lost under blur/down-up degradation, while the absolute
quality term helps reject fake high-frequency noise. Use this reward with
``anti_noise`` and a video quality reward rather than alone.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from ..registry import BaseReward, register_reward

logger = logging.getLogger(__name__)


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def _frame_indices(frames: int, num_frames: int) -> List[int]:
    n = min(max(1, int(num_frames)), frames)
    if n == 1:
        return [0]
    return [round(i * (frames - 1) / (n - 1)) for i in range(n)]


def _video_to_frame_batch(video, *, num_frames: int, max_side: int):
    """Decoded video [C,T,H,W] or [1,C,T,H,W] -> [N,3,H,W] in [0,1]."""
    import torch
    import torch.nn.functional as F

    t = video
    if t.dim() == 5:
        t = t[0]
    if t.dim() != 4:
        raise ValueError(f"iqa_quality: expected decoded video [C,T,H,W], got {tuple(video.shape)}")
    c, frames, h, w = t.shape
    if c < 3:
        raise ValueError(f"iqa_quality: expected at least 3 RGB channels, got {c}")
    t = t[:3].detach().to("cpu", torch.float32).clamp(0.0, 1.0)
    idx = _frame_indices(frames, num_frames)
    batch = t[:, idx].permute(1, 0, 2, 3).contiguous()  # [N,3,H,W]

    max_side = int(max_side)
    if max_side > 0 and max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        new_h = max(16, round(h * scale))
        new_w = max(16, round(w * scale))
        batch = F.interpolate(batch, size=(new_h, new_w), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    return batch


def _odd_kernel(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def _degrade_frames(frames, *, mode: str, blur_kernel: int, downup_scale: float):
    """Create a blur/down-up degraded version of [N,3,H,W] frames in [0,1]."""
    import torch.nn.functional as F

    x = frames
    mode = str(mode).lower()
    if mode in ("blur", "both"):
        k = _odd_kernel(blur_kernel)
        pad = k // 2
        x = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)
    if mode in ("downup", "down_up", "both"):
        scale = min(0.95, max(0.05, float(downup_scale)))
        h, w = x.shape[-2:]
        small_h = max(8, round(h * scale))
        small_w = max(8, round(w * scale))
        x = F.interpolate(x, size=(small_h, small_w), mode="bicubic", align_corners=False)
        x = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    if mode in ("none", "off", "false"):
        return x
    if mode not in ("blur", "downup", "down_up", "both", "none", "off", "false"):
        raise ValueError("iqa_quality: degrade must be one of none|blur|downup|both")
    return x.clamp(0.0, 1.0)


@register_reward("iqa_quality")
class IQAQualityReward(BaseReward):
    """Frame-level perceptual quality/detail reward backed by IQA-PyTorch."""

    kind = "blackbox"
    route = "video"
    needs = frozenset({"video"})

    def __init__(self) -> None:
        self._metric = None
        self._device = None
        self._metric_name = "topiq_nr"
        self._num_frames = 6
        self._batch_size = 8
        self._max_side = 384
        self._degrade = "both"
        self._blur_kernel = 5
        self._downup_scale = 0.5
        self._quality_weight = 0.5
        self._delta_weight = 1.0
        self._invert = False

    def setup(
        self,
        device,
        *,
        metric_name="topiq_nr",
        num_frames=6,
        batch_size=8,
        max_side=384,
        degrade="both",
        blur_kernel=5,
        downup_scale=0.5,
        quality_weight=0.5,
        delta_weight=1.0,
        higher_better="auto",
        **_ignored,
    ) -> None:
        try:
            import pyiqa
        except ImportError as exc:  # pragma: no cover - only hit when optional dep is absent
            raise ImportError(
                "iqa_quality reward requires IQA-PyTorch: pip install pyiqa. "
                "Try --reward_args metric_name=topiq_nr, or another pyiqa NR-IQA metric."
            ) from exc

        dev = "cuda" if device is None else str(device)
        self._device = dev
        self._metric_name = str(metric_name)
        self._num_frames = int(num_frames)
        self._batch_size = int(batch_size)
        self._max_side = int(max_side)
        self._degrade = str(degrade)
        self._blur_kernel = _odd_kernel(int(blur_kernel))
        self._downup_scale = float(downup_scale)
        self._quality_weight = float(quality_weight)
        self._delta_weight = float(delta_weight)

        self._metric = pyiqa.create_metric(self._metric_name, device=dev, as_loss=False)
        if hasattr(self._metric, "eval"):
            self._metric.eval()

        lower_better = bool(getattr(self._metric, "lower_better", False))
        if str(higher_better).lower() == "auto":
            self._invert = lower_better
        else:
            self._invert = not _truthy(higher_better)
        logger.info(
            "iqa_quality: loaded pyiqa metric=%s device=%s invert=%s num_frames=%d max_side=%d degrade=%s",
            self._metric_name,
            dev,
            self._invert,
            self._num_frames,
            self._max_side,
            self._degrade,
        )

    def _predict(self, frames) -> List[float]:
        if self._metric is None:
            raise RuntimeError("iqa_quality: setup() must run before score()")
        import torch

        scores: List[float] = []
        bs = max(1, self._batch_size)
        with torch.inference_mode():
            for start in range(0, frames.shape[0], bs):
                x = frames[start : start + bs].to(self._device, non_blocking=True)
                y = self._metric(x)
                if isinstance(y, (tuple, list)):
                    y = y[0]
                y = torch.as_tensor(y).detach().float().flatten().cpu()
                if self._invert:
                    y = -y
                scores.extend(float(v) for v in y.tolist())
        return scores

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        scores: List[float] = []
        for sample in samples:
            video = sample.get("video")
            if video is None:
                scores.append(0.0)
                continue
            frames = _video_to_frame_batch(video, num_frames=self._num_frames, max_side=self._max_side)
            degraded = _degrade_frames(
                frames,
                mode=self._degrade,
                blur_kernel=self._blur_kernel,
                downup_scale=self._downup_scale,
            )
            raw_scores = self._predict(frames)
            if self._degrade.lower() in ("none", "off", "false"):
                delta_scores = [0.0 for _ in raw_scores]
            else:
                degraded_scores = self._predict(degraded)
                delta_scores = [raw - degraded for raw, degraded in zip(raw_scores, degraded_scores)]
            raw_mean = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0
            delta_mean = sum(delta_scores) / len(delta_scores) if delta_scores else 0.0
            scores.append(self._quality_weight * raw_mean + self._delta_weight * delta_mean)
        return scores, {
            "reward": "iqa_quality",
            "metric_name": self._metric_name,
            "num_frames": self._num_frames,
            "max_side": self._max_side,
            "degrade": self._degrade,
            "quality_weight": self._quality_weight,
            "delta_weight": self._delta_weight,
        }

    def teardown(self) -> None:
        if self._metric is not None:
            try:
                import torch

                del self._metric
                self._metric = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best-effort VRAM release
                self._metric = None
