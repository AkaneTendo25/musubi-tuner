"""Composable, reusable reward registry for LTX-2 RL post-training.

Public API: the ``Reward`` protocol + ``BaseReward`` base, the ``@register_reward``
decorator, ``parse_reward_spec``, ``RewardStack`` (VRAM-sequenced scoring), and
``PerPromptStatTracker`` (GRPO advantages). Importing the package registers the
built-in zoo plugins.
"""

from __future__ import annotations

from .registry import (
    VALID_KINDS,
    VALID_ROUTES,
    BaseReward,
    Reward,
    RewardStack,
    get_reward_cls,
    load_reward_plugins,
    parse_reward_spec,
    register_reward,
    registered_rewards,
    unregister,
)
from .stat_tracking import PerPromptStatTracker
from . import zoo  # noqa: F401  (registers built-in plugins)

__all__ = [
    "Reward",
    "BaseReward",
    "RewardStack",
    "register_reward",
    "get_reward_cls",
    "registered_rewards",
    "unregister",
    "parse_reward_spec",
    "load_reward_plugins",
    "PerPromptStatTracker",
    "VALID_KINDS",
    "VALID_ROUTES",
]
