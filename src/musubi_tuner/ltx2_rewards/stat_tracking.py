"""Per-prompt group-relative advantage (GRPO) for LTX-2 RL.

The standard GRPO formulation always normalizes with ``std + 1e-4`` and never skips,
so a zero-variance group (deterministic collapse / K too small) yields a tiny
denominator and huge spurious advantages. Here such groups are **flagged and excluded**
(advantage 0), because a group with no reward spread carries no usable relative
learning signal.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PerPromptStatTracker:
    """Compute group-relative advantages, one group per unique prompt."""

    def __init__(
        self,
        std_eps: float = 1e-4,
        zero_var_threshold: float = 1e-6,
        global_std: bool = False,
    ) -> None:
        self.std_eps = std_eps
        self.zero_var_threshold = zero_var_threshold
        self.global_std = global_std
        self.flagged_prompts: List[str] = []
        self.num_groups = 0
        self.num_flagged = 0

    def compute(self, prompts: List[str], rewards) -> Tuple[np.ndarray, List[str]]:
        """Return ``(advantages, flagged_prompts)``.

        ``advantages[i]`` is the group-relative advantage of sample ``i``. Zero-variance
        groups get advantage 0 and their prompt is reported in ``flagged_prompts``.
        """
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        prompts = list(prompts)
        if len(prompts) != len(rewards):
            raise ValueError(f"prompts/rewards length mismatch: {len(prompts)} vs {len(rewards)}")

        adv = np.zeros_like(rewards)
        flagged: List[str] = []
        self.num_groups = 0
        global_std = float(rewards.std()) if rewards.size else 0.0

        for prompt in dict.fromkeys(prompts):  # unique prompts, order-preserving
            idx = [i for i, q in enumerate(prompts) if q == prompt]
            group = rewards[idx]
            self.num_groups += 1
            if not np.isfinite(group).all():
                # A NaN/inf reward would make std NaN, dodge the zero-variance test (nan < eps is
                # False), and poison every advantage in the group (and, downstream, the optimizer).
                flagged.append(prompt)
                adv[idx] = 0.0
                continue
            mean = float(group.mean())
            std = float(group.std())
            if std < self.zero_var_threshold:
                flagged.append(prompt)
                adv[idx] = 0.0  # excluded: no relative signal (NOT divided by a 1e-4 floor)
                continue
            denom = (global_std if self.global_std else std) + self.std_eps
            adv[idx] = (group - mean) / denom

        self.flagged_prompts = flagged
        self.num_flagged = len(flagged)
        if flagged:
            logger.warning(
                "GRPO: %d/%d prompt groups had ~zero reward variance (flagged + excluded). "
                "Increase group size K or sampler diversity. First few: %s",
                len(flagged),
                self.num_groups,
                flagged[:5],
            )
        return adv, flagged
