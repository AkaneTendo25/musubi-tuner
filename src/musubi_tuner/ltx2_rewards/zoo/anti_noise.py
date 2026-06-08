"""Model-free anti-noise video reward.

This is a guardrail for detail/entropy/sharpness rewards. Those rewards can be
hacked by high-frequency speckle: the clip becomes "detailed" numerically while
looking like noise. ``anti_noise`` rewards clean, structured detail instead:

  - it measures high-pass residual energy mainly in low-gradient image regions,
    where true object/detail edges are unlikely;
  - it also penalizes temporal high-pass flicker in those flat regions.

The score is higher-is-better and model-free, so it can be composed with a
positive detail reward, for example:

    --reward_fn "scene_detail:1.0,anti_noise:0.5,hpsv3:0.25"

Use it as a guardrail, not as the only objective: an over-weighted cleanliness
term prefers smooth/flat video.
"""

from __future__ import annotations

from typing import List, Tuple

from ..registry import BaseReward, register_reward


def _odd_kernel(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1


def _frame_indices(frames: int, num_frames: int) -> List[int]:
    n = min(max(1, int(num_frames)), frames)
    if n == 1:
        return [0]
    return [round(i * (frames - 1) / (n - 1)) for i in range(n)]


def _anti_noise_score(
    video,
    *,
    num_frames: int = 8,
    blur_kernel: int = 5,
    flat_quantile: float = 0.35,
    flat_threshold: float = 0.025,
    spatial_scale: float = 80.0,
    temporal_scale: float = 30.0,
) -> float:
    """Score decoded video cleanliness.

    ``video`` is ``[C,T,H,W]`` or ``[1,C,T,H,W]`` in [0,1]. Returns a bounded
    higher-is-better score. The metric is intentionally local and cheap; learned
    visual-quality rewards should still be used for semantic/aesthetic quality.
    """
    import torch
    import torch.nn.functional as F

    t = video
    if t.dim() == 5:
        t = t[0]
    if t.dim() != 4:
        raise ValueError(f"anti_noise: expected decoded video [C,T,H,W], got {tuple(video.shape)}")
    t = t.detach().to("cpu", torch.float32).clamp(0.0, 1.0)
    _c, frames, _h, _w = t.shape
    if frames <= 0:
        return 0.0

    gray = t.mean(dim=0)  # [T,H,W]
    idx = _frame_indices(frames, num_frames)
    x = gray[idx].unsqueeze(1)  # [N,1,H,W]

    k = _odd_kernel(blur_kernel)
    pad = k // 2
    low = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel_size=k, stride=1)
    residual = x - low

    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=x.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=x.dtype).view(1, 1, 3, 3)
    grad_x = F.conv2d(F.pad(low, (1, 1, 1, 1), mode="replicate"), sobel_x)
    grad_y = F.conv2d(F.pad(low, (1, 1, 1, 1), mode="replicate"), sobel_y)
    grad = (grad_x.square() + grad_y.square() + 1e-12).sqrt()

    flat_quantile = min(1.0, max(0.0, float(flat_quantile)))
    q = torch.quantile(grad.flatten(1), flat_quantile, dim=1).view(-1, 1, 1, 1)
    limit = torch.maximum(q, torch.full_like(q, float(flat_threshold)))
    flat = grad <= limit

    flat_count = flat.sum().clamp_min(1)
    spatial_noise = float((residual.abs() * flat).sum() / flat_count)

    temporal_noise = 0.0
    if residual.shape[0] >= 2:
        temporal_mask = flat[1:] & flat[:-1]
        temporal_count = temporal_mask.sum().clamp_min(1)
        temporal_noise = float(((residual[1:] - residual[:-1]).abs() * temporal_mask).sum() / temporal_count)

    penalty = float(spatial_scale) * spatial_noise + float(temporal_scale) * temporal_noise
    return 1.0 / (1.0 + max(0.0, penalty))


@register_reward("anti_noise")
class AntiNoiseReward(BaseReward):
    """High-frequency speckle/flicker guardrail for detail rewards. No model."""

    kind = "blackbox"
    route = "video"
    needs = frozenset({"video"})

    def __init__(self) -> None:
        self._num_frames = 8
        self._blur_kernel = 5
        self._flat_quantile = 0.35
        self._flat_threshold = 0.025
        self._spatial_scale = 80.0
        self._temporal_scale = 30.0

    def setup(
        self,
        device,
        *,
        num_frames=8,
        blur_kernel=5,
        flat_quantile=0.35,
        flat_threshold=0.025,
        spatial_scale=80.0,
        temporal_scale=30.0,
        **_ignored,
    ) -> None:
        self._num_frames = int(num_frames)
        self._blur_kernel = _odd_kernel(int(blur_kernel))
        self._flat_quantile = float(flat_quantile)
        self._flat_threshold = float(flat_threshold)
        self._spatial_scale = float(spatial_scale)
        self._temporal_scale = float(temporal_scale)

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        scores: List[float] = []
        for sample in samples:
            video = sample.get("video")
            if video is None:
                scores.append(0.0)
                continue
            scores.append(
                _anti_noise_score(
                    video,
                    num_frames=self._num_frames,
                    blur_kernel=self._blur_kernel,
                    flat_quantile=self._flat_quantile,
                    flat_threshold=self._flat_threshold,
                    spatial_scale=self._spatial_scale,
                    temporal_scale=self._temporal_scale,
                )
            )
        return scores, {
            "reward": "anti_noise",
            "num_frames": self._num_frames,
            "blur_kernel": self._blur_kernel,
            "flat_quantile": self._flat_quantile,
            "flat_threshold": self._flat_threshold,
            "spatial_scale": self._spatial_scale,
            "temporal_scale": self._temporal_scale,
        }

    def teardown(self) -> None:
        pass
