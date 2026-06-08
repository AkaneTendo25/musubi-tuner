"""Model-free kinematic-physics video reward (optical-flow motion plausibility).

Where ``temporal_consistency`` looks at raw pixel derivatives, this reward analyses the **optical
flow field** — how things actually move — and scores whether that motion is *physically plausible*:

  - **Acceleration boundedness** — real objects have continuous, bounded acceleration. The temporal
    change of the flow field, ``|flow[t] - flow[t-1]|``, is large when objects teleport / pop
    in and out (impulsive, unphysical motion).
  - **Spatial coherence / rigidity** — points on a real object move together, so the flow field is
    spatially smooth within a moving region. Large spatial flow gradients ``|∂flow|`` mean per-pixel
    incoherent motion: morphing, melting, boiling texture.

Both terms are **normalised by the mean flow magnitude (speed)**, so the score judges *how* things
move, not *how much*. A fast but rigid, constant-velocity pan scores as plausible as a slow one;
``dynamic_degree`` is the separate reward that asks whether there is motion at all.

A frozen clip is vacuously plausible, so "stop moving" is a degenerate optimum of this reward.
Two defences: weight ``dynamic_degree`` in the composite (linear counterweight — the policy can
still trade motion for cleanliness), or set the ``motion_gate`` reward_arg (multiplicative — the
score is scaled by ``min(1, mean_flow_speed / motion_gate)``, so a near-static clip earns ~0 and
the freeze direction leaves the reward landscape entirely). Gate units are mean |flow| px/frame at
the analysis scale; ~0.05 is estimator noise on a static clip, ordinary motion is well above 0.1.

This is the cheap, model-free tier of "physics": it catches impossible *motion* (teleporting,
morphing) but NOT semantic physics (gravity direction, collisions, object permanence) — only a
learned judge such as ``videoscore2`` with ``dims=pc`` (Physical/common-sense Consistency) does
that. Compose them, e.g. ``--reward_fn "videoscore2:1.0,motion_physics:0.5,dynamic_degree:0.3"``
with ``--reward_args dims=pc``.

Uses OpenCV Farnebäck dense optical flow (the same estimator the av_align reward uses), imported
lazily so the registry / CPU placeholder path never require it. Higher = more plausible.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..registry import BaseReward, register_reward


def _flow_physics(video, *, max_size: int = 128, motion_floor: float = 0.05) -> Optional[Tuple[float, float, float]]:
    """Return ``(accel_per_speed, incoherence_per_speed, raw_speed)`` for a decoded video, or None if < 2 frames.

    ``video`` is ``[C,T,H,W]`` / ``[1,C,T,H,W]`` in [0,1]. Frames are downscaled to ``max_size`` for
    speed, converted to 8-bit luminance, and run through Farnebäck dense flow. The acceleration and
    spatial-incoherence terms are normalised by ``max(mean speed, motion_floor)``: the floor keeps a
    near-static clip (whose flow is just estimator noise) from looking like wildly incoherent motion.
    """
    import cv2
    import numpy as np
    import torch

    t = video
    if t.dim() == 5:
        t = t[0]
    if t.dim() != 4:
        raise ValueError(f"motion_physics: expected decoded video [C,T,H,W], got {tuple(video.shape)}")
    t = t.detach().to("cpu", torch.float32).clamp(0.0, 1.0)
    _c, frames, h, w = t.shape
    if frames < 2:
        return None

    gray = t.mean(dim=0)  # [T,H,W] luminance
    scale = min(1.0, float(max_size) / float(max(h, w)))
    imgs = []
    for i in range(frames):
        g = (gray[i].numpy() * 255.0).astype(np.uint8)
        if scale < 1.0:
            g = cv2.resize(g, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        imgs.append(g)

    flows = []
    for i in range(frames - 1):
        flow = cv2.calcOpticalFlowFarneback(imgs[i], imgs[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)  # [h,w,2]
    flow_seq = np.stack(flows, axis=0)  # [T-1, h, w, 2]

    raw_speed = float(np.abs(flow_seq).mean())
    speed = max(raw_speed, float(motion_floor))
    # Spatial incoherence: how much the flow field varies across neighbouring pixels (rigid -> ~0).
    grad_x = np.abs(np.diff(flow_seq, axis=2)).mean()
    grad_y = np.abs(np.diff(flow_seq, axis=1)).mean()
    incoherence = float(grad_x + grad_y)
    # Temporal acceleration: how abruptly the flow changes over time (constant velocity -> ~0).
    accel = float(np.abs(np.diff(flow_seq, axis=0)).mean()) if flow_seq.shape[0] >= 2 else 0.0
    return accel / speed, incoherence / speed, raw_speed


@register_reward("motion_physics")
class MotionPhysicsReward(BaseReward):
    """Kinematic-physics reward: bounded acceleration + spatial flow coherence (Farnebäck flow).

    Higher = more physically-plausible motion. A frozen clip is vacuously plausible (score 1.0) —
    pair with ``dynamic_degree``, or set ``motion_gate`` so freezing is not an optimum. ``alpha``
    weights the acceleration (anti-teleport) term, ``beta`` the spatial-incoherence (anti-morph)
    term, ``max_size`` is the flow downscale, and ``motion_gate`` (0 = off) multiplies the score
    by ``min(1, mean_flow_speed / motion_gate)`` so a near-static clip earns ~0 (all reward_args;
    defaults 1.0 / 1.0 / 128 / 0.0).
    """

    kind = "blackbox"
    route = "video"
    needs = frozenset({"video"})

    def __init__(self) -> None:
        self._alpha = 1.0
        self._beta = 1.0
        self._max_size = 128
        self._motion_floor = 0.05
        self._motion_gate = 0.0

    def setup(self, device, *, alpha=1.0, beta=1.0, max_size=128, motion_floor=0.05, motion_gate=0.0, **_ignored) -> None:
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._max_size = int(max_size)
        self._motion_floor = float(motion_floor)
        self._motion_gate = float(motion_gate)

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        scores: List[float] = []
        for sample in samples:
            video = sample.get("video")
            if video is None:
                scores.append(0.0)
                continue
            res = _flow_physics(video, max_size=self._max_size, motion_floor=self._motion_floor)
            if res is None:  # < 2 frames: no motion to judge — plausible, but gated runs need motion
                scores.append(0.0 if self._motion_gate > 0 else 1.0)
                continue
            accel_n, incoh_n, raw_speed = res
            s = 1.0 / (1.0 + self._alpha * accel_n + self._beta * incoh_n)
            if self._motion_gate > 0:
                s *= min(1.0, raw_speed / self._motion_gate)
            scores.append(s)
        return scores, {"reward": "motion_physics", "alpha": self._alpha, "beta": self._beta, "motion_gate": self._motion_gate}

    def teardown(self) -> None:
        pass
