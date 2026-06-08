"""Motion-cadence reward (no model), route=video: match a target temporal motion *rhythm*.

Scores how closely the per-step motion pattern of a clip matches a target cadence, e.g. the
anime "shot on 2s" timing (held frames punctuated by snappy pose-to-pose moves) instead of the
uniform interpolated motion video models default to. Built only on luminance frame differences
``d[t] = mean |f[t+1] - f[t]|`` (cheap, deterministic, like ``dynamic_degree``):

  - ``hold_fraction`` — fraction of steps that are holds (``d[t] < hold_rel * mean(d)``).
  - ``burstiness``    — coefficient of variation ``std(d)/mean(d)``: 0 for uniform motion,
                        ~1 for hold/move alternation.
  - ``motion``        — ``mean(d)``, used only as a floor so a frozen clip cannot win.

Reward = mean of three higher-is-better terms in [0,1]: Gaussian bumps around ``hold_target``
and ``burst_target`` plus a linear ramp ``min(1, motion/motion_floor)``. Defaults target on-2s
cadence: ``hold_target=0.5 hold_scale=0.2 burst_target=1.0 burst_scale=0.5 motion_floor=0.015
hold_rel=0.35`` — all overridable via ``--reward_args``. Clips with fewer than 3 frames score 0.

Example: ``--reward_fn "motion_cadence:1.0" --reward_args hold_target=0.5 burst_target=1.0``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from ..registry import BaseReward, register_reward


def _step_motion(video):
    """Per-step luminance motion ``d[t] = mean |f[t+1]-f[t]|`` as a cpu float32 tensor ``[T-1]``.

    ``video`` is a decoded ``[C,T,H,W]`` / ``[1,C,T,H,W]`` clip (the do_inference layout).
    """
    import torch

    t = video
    if t.dim() == 5:
        t = t[0]
    if t.dim() != 4:
        raise ValueError(f"motion_cadence: expected decoded video [C,T,H,W], got {tuple(video.shape)}")
    g = t.detach().to("cpu", torch.float32).mean(dim=0)  # [T,H,W] luminance proxy
    return (g[1:] - g[:-1]).abs().mean(dim=(1, 2))  # [T-1]


def _cadence_stats(video, hold_rel: float, eps: float = 1e-8) -> Optional[Tuple[float, float, float]]:
    """``(motion, hold_fraction, burstiness)`` of a clip, or None for clips under 3 frames."""
    d = _step_motion(video)
    if d.numel() < 2:
        return None
    mean_d = float(d.mean())
    if mean_d < eps:  # exactly frozen clip
        return 0.0, 1.0, 0.0
    hold_fraction = float((d < hold_rel * mean_d).float().mean())
    burstiness = float(d.std(unbiased=False)) / mean_d
    return mean_d, hold_fraction, burstiness


def _gauss(x: float, target: float, scale: float) -> float:
    return math.exp(-(((x - target) / scale) ** 2))


@register_reward("motion_cadence")
class MotionCadenceReward(BaseReward):
    """Target-cadence reward (hold fraction + burstiness bumps, motion floor). No model."""

    kind = "blackbox"
    route = "video"
    needs = frozenset({"video"})

    def __init__(self) -> None:
        self._hold_target = 0.5
        self._hold_scale = 0.2
        self._burst_target = 1.0
        self._burst_scale = 0.5
        self._motion_floor = 0.015
        self._hold_rel = 0.35

    def setup(
        self,
        device,
        *,
        hold_target=0.5,
        hold_scale=0.2,
        burst_target=1.0,
        burst_scale=0.5,
        motion_floor=0.015,
        hold_rel=0.35,
        **_ignored,
    ) -> None:
        self._hold_target = float(hold_target)
        self._hold_scale = float(hold_scale)
        self._burst_target = float(burst_target)
        self._burst_scale = float(burst_scale)
        self._motion_floor = float(motion_floor)
        self._hold_rel = float(hold_rel)

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        scores: List[float] = []
        for sample in samples:
            video = sample.get("video")
            stats = _cadence_stats(video, self._hold_rel) if video is not None else None
            if stats is None:
                scores.append(0.0)
                continue
            motion, hold_fraction, burstiness = stats
            terms = (
                _gauss(hold_fraction, self._hold_target, self._hold_scale),
                _gauss(burstiness, self._burst_target, self._burst_scale),
                min(1.0, motion / self._motion_floor),
            )
            scores.append(sum(terms) / len(terms))
        return scores, {
            "reward": "motion_cadence",
            "hold_target": self._hold_target,
            "burst_target": self._burst_target,
            "motion_floor": self._motion_floor,
        }

    def teardown(self) -> None:
        pass
