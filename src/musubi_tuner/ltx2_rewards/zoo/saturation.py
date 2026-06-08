"""Mean color-saturation reward (no model), route=video.

Measures mean HSV saturation over sampled frames: ``S = (max_c - min_c) / (max_c + eps)``, in [0,1];
higher = more vivid color. Model-free and dense (every pixel).
"""

from __future__ import annotations

from typing import List, Tuple

from ..registry import BaseReward, register_reward


def _mean_saturation(video, num_frames: int = 5) -> float:
    """Mean HSV saturation over uniformly sampled frames of a decoded video tensor.

    ``video`` is ``[C,T,H,W]`` or ``[1,C,T,H,W]`` RGB in [0,1] (the do_inference decoded layout).
    """
    import torch

    t = video
    if t.dim() == 5:
        t = t[0]
    if t.dim() != 4:
        raise ValueError(f"saturation: expected decoded video [C,T,H,W], got {tuple(video.shape)}")
    t = t.detach().to("cpu", torch.float32).clamp(0.0, 1.0)
    c, frames, _h, _w = t.shape
    if c < 3:
        return 0.0
    rgb = t[:3]  # [3,T,H,W]
    idx = [round(i * (frames - 1) / max(1, num_frames - 1)) for i in range(min(num_frames, frames))]
    vals = []
    for f in idx:
        fr = rgb[:, f]  # [3,H,W]
        cmax = fr.max(dim=0).values
        cmin = fr.min(dim=0).values
        sat = (cmax - cmin) / (cmax + 1e-6)  # [H,W] in [0,1]
        vals.append(float(sat.mean()))
    return sum(vals) / len(vals) if vals else 0.0


@register_reward("saturation")
class SaturationReward(BaseReward):
    """Mean HSV saturation (color vividness) reward — model-free, demonstrative."""

    kind = "blackbox"
    route = "video"
    needs = frozenset({"video"})

    def __init__(self) -> None:
        self._num_frames = 5

    def setup(self, device, *, num_frames=5, **_ignored) -> None:
        self._num_frames = int(num_frames)

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        scores: List[float] = []
        for sample in samples:
            video = sample.get("video")
            scores.append(_mean_saturation(video, self._num_frames) if video is not None else 0.0)
        return scores, {"reward": "saturation", "num_frames": self._num_frames}

    def teardown(self) -> None:
        pass
