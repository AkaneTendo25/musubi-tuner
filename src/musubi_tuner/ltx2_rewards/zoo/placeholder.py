"""Placeholder reward: deterministic, dependency-free.

Used to wire the RL pipeline (registry -> rollout cache -> NFT loop) end-to-end on
video-only BEFORE porting a real model (e.g. HPSv3). Produces a stable pseudo-random
score per sample so prompt groups have non-zero variance (exercising GRPO) without
needing any model, checkpoint, or decoded media.
"""

from __future__ import annotations

from typing import List, Tuple

from ..registry import BaseReward, register_reward


@register_reward("placeholder")
class PlaceholderReward(BaseReward):
    kind = "blackbox"
    route = "video"
    needs = frozenset({"seed"})

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        scores: List[float] = []
        for i, sample in enumerate(samples):
            key = int(sample.get("seed", sample.get("index", i)))
            # deterministic Knuth multiplicative hash -> [0, 1)
            h = (key * 2654435761) & 0xFFFFFFFF
            scores.append(h / 2**32)
        return scores, {"reward": "placeholder"}
