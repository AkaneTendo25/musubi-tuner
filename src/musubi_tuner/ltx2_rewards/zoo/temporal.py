"""Model-free temporal video rewards for physical/temporal plausibility.

Two composable, checkpoint-free video rewards built only on the decoded frames (no learned
model, like ``sharpness``). They are meant to be used **together** as an anti-freeze pair:

  - ``temporal_consistency`` rewards smooth, flicker-free motion (penalizes the temporal
    second derivative: frame-to-frame jitter, morphing, popping).
  - ``dynamic_degree`` rewards the *presence* of motion (the temporal first derivative).

Used alone, ``temporal_consistency`` is maximized by a frozen frame (zero motion = zero
flicker), so optimizing it by itself teaches the policy to stop moving. Composing it with
``dynamic_degree`` removes that degenerate optimum — the policy must move *and* move smoothly,
e.g. ``--reward_fn "temporal_consistency:1.0,dynamic_degree:0.5"``.

These are flow-free proxies computed on pixel luminance (cheap, deterministic). A flow-based
warp-error metric distinguishes smooth fast motion from flicker more precisely, but needs an
optical-flow estimator; the second-derivative proxy here conflates the two somewhat, which is
exactly why it is paired with ``dynamic_degree`` rather than used on its own. All scores are
higher-is-better (the ``temporal_consistency`` distance is inverted as ``1/(1+scale*d)``).
"""

from __future__ import annotations

from typing import List, Tuple

from ..registry import BaseReward, register_reward


def _gray_frames(video):
    """Luminance frames ``[T,H,W]`` (cpu float32) from a decoded video ``[C,T,H,W]``/``[1,C,T,H,W]``."""
    import torch

    t = video
    if t.dim() == 5:
        t = t[0]
    if t.dim() != 4:
        raise ValueError(f"temporal reward: expected decoded video [C,T,H,W], got {tuple(video.shape)}")
    t = t.detach().to("cpu", torch.float32)
    return t.mean(dim=0)  # [T,H,W] luminance proxy


def _temporal_roughness(video) -> float:
    """Mean |second temporal difference| of luminance — high for flicker / jitter / morphing.

    ``d2[t] = f[t+1] - 2 f[t] + f[t-1]``. Zero for a static or constant-velocity intensity ramp;
    large for popping / flicker. Needs >= 3 frames (else 0.0 = perfectly consistent).
    """
    g = _gray_frames(video)
    if g.shape[0] < 3:
        return 0.0
    d2 = g[2:] - 2.0 * g[1:-1] + g[:-2]
    return float(d2.abs().mean())


def _motion_magnitude(video) -> float:
    """Mean |first temporal difference| of luminance — a flow-free proxy for how much motion there is.

    ``d1[t] = f[t+1] - f[t]``. ~0 for a frozen clip, larger the more the frames change. Needs >= 2
    frames (else 0.0 = no motion).
    """
    g = _gray_frames(video)
    if g.shape[0] < 2:
        return 0.0
    d1 = g[1:] - g[:-1]
    return float(d1.abs().mean())


@register_reward("temporal_consistency")
class TemporalConsistencyReward(BaseReward):
    """Smooth-motion / anti-flicker reward (inverse temporal second-derivative). No model.

    Higher = less flicker/jitter. Pair with ``dynamic_degree`` so a frozen frame (its degenerate
    optimum) is not rewarded. ``scale`` (reward_arg, default 100.0) sets the ``1/(1+scale*d)`` knee.
    """

    kind = "blackbox"
    route = "video"
    needs = frozenset({"video"})

    def __init__(self) -> None:
        self._scale = 100.0

    def setup(self, device, *, scale=100.0, **_ignored) -> None:
        self._scale = float(scale)

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        scores: List[float] = []
        for sample in samples:
            video = sample.get("video")
            if video is None:
                scores.append(0.0)
                continue
            roughness = _temporal_roughness(video)
            scores.append(1.0 / (1.0 + self._scale * roughness))
        return scores, {"reward": "temporal_consistency", "scale": self._scale}

    def teardown(self) -> None:
        pass


@register_reward("dynamic_degree")
class DynamicDegreeReward(BaseReward):
    """Motion-presence reward (mean first temporal difference). No model.

    Higher = more motion; ~0 for a frozen clip. Its purpose is to counterbalance
    ``temporal_consistency`` so the policy cannot win by freezing the frame.
    """

    kind = "blackbox"
    route = "video"
    needs = frozenset({"video"})

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        scores: List[float] = []
        for sample in samples:
            video = sample.get("video")
            if video is None:
                scores.append(0.0)
                continue
            scores.append(_motion_magnitude(video))
        return scores, {"reward": "dynamic_degree"}

    def teardown(self) -> None:
        pass
