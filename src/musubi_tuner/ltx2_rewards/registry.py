"""Composable reward registry for LTX-2 RL post-training.

Each reward is a plugin that declares its ``kind`` (blackbox|differentiable), its
``route`` (video|audio|sync) and the inputs it ``needs``. The registry plus
``RewardStack`` orchestrate sequential ``setup -> score -> teardown`` so reward
models are never co-resident in VRAM (offline scoring sequences them one at a time).

The module is standalone: usable for offline rollout scoring, online RL, or as
plain eval metrics over decoded media.

Design note: routing is an explicit per-plugin attribute, NOT keyword-inferred from
the reward name. Inferring a route from the name tends to default unknown names to
"sync" (= both branches); here an invalid/missing route raises at registration, so a
diagnostic reward can never be silently routed to both branches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)

VALID_KINDS = ("blackbox", "differentiable")
VALID_ROUTES = ("video", "audio", "sync")


@runtime_checkable
class Reward(Protocol):
    """Structural type a reward plugin must satisfy."""

    name: str
    kind: str
    route: str
    needs: frozenset

    def setup(self, device: Any, **kwargs: Any) -> None: ...
    def teardown(self) -> None: ...
    def score(self, samples: List[dict]) -> Tuple[List[float], dict]: ...
    def score_groups(self, groups: List[dict]) -> Tuple[List[List[float]], dict]: ...


class BaseReward:
    """Convenience base class: blackbox by default, no-op setup/teardown.

    Subclasses set ``kind``/``route``/``needs`` as class attributes and implement
    ``score(samples) -> (per_sample_scores, info)``. All scores must be
    higher-is-better (apply any inversion such as ``1/(1+d)`` inside ``score``).
    """

    name: str = ""
    kind: str = "blackbox"
    route: str = "sync"
    needs: frozenset = frozenset()

    def setup(self, device: Any, **kwargs: Any) -> None:  # noqa: D401 - simple hook
        pass

    def teardown(self) -> None:
        pass

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        raise NotImplementedError

    def score_groups(self, groups: List[dict]) -> Tuple[List[List[float]], dict]:
        """Group-aware scoring hook.

        Ordinary rewards only implement ``score(samples)``; this fallback flattens
        groups, calls ``score``, and slices the result back to one list per group.
        Pairwise/ranking rewards can override this method to compare samples within
        each prompt group directly while still returning scalar scores for GRPO.
        """
        flat = [sample for group in groups for sample in group.get("samples", [])]
        flat_scores, info = self.score(flat)
        out: List[List[float]] = []
        idx = 0
        for group in groups:
            n = len(group.get("samples", []))
            out.append(flat_scores[idx : idx + n])
            idx += n
        return out, info


_REGISTRY: Dict[str, type] = {}


def register_reward(name: str) -> Callable[[type], type]:
    """Class decorator registering a reward plugin under ``name``.

    Validates that ``route`` and ``kind`` are explicit and valid, raising on an
    invalid route (no silent "sync" fallback) or a duplicate name.
    """

    def deco(cls: type) -> type:
        route = getattr(cls, "route", None)
        kind = getattr(cls, "kind", "blackbox")
        if route not in VALID_ROUTES:
            raise ValueError(
                f"reward '{name}' declares invalid route {route!r}; must be one of {VALID_ROUTES}. "
                "Routes are explicit per-plugin (no silent 'sync' default)."
            )
        if kind not in VALID_KINDS:
            raise ValueError(f"reward '{name}' declares invalid kind {kind!r}; must be one of {VALID_KINDS}")
        if name in _REGISTRY:
            raise ValueError(f"reward '{name}' already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_reward_cls(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(f"unknown reward '{name}'. Registered: {registered_rewards()}")
    return _REGISTRY[name]


def registered_rewards() -> List[str]:
    return sorted(_REGISTRY)


def unregister(name: str) -> None:
    """Remove a reward from the registry (primarily for tests)."""
    _REGISTRY.pop(name, None)


def parse_reward_spec(spec: str) -> Dict[str, float]:
    """Parse ``"name:weight,name2:weight2"`` into ``{name: weight}``.

    A bare ``name`` defaults to weight 1.0. Every name must be registered, else
    ``KeyError`` (so a typo fails loudly instead of silently dropping a reward).
    """
    out: Dict[str, float] = {}
    if not spec:
        return out
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" in item:
            name, weight_str = item.split(":", 1)
            name, weight = name.strip(), float(weight_str)
        else:
            name, weight = item, 1.0
        if name not in _REGISTRY:
            raise KeyError(f"reward spec references unknown reward '{name}'. Registered: {registered_rewards()}")
        out[name] = weight
    return out


def load_reward_plugins(paths: List[str]) -> List[str]:
    """Import standalone reward-plugin ``.py`` files so their ``@register_reward`` classes register.

    Lets a custom reward live anywhere on disk (``--reward_plugins my_reward.py``) — no zoo
    edit or package install needed. Returns the reward names newly registered by the files.
    """
    import importlib.util
    from pathlib import Path

    before = set(_REGISTRY)
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"reward plugin file not found: {raw}")
        spec = importlib.util.spec_from_file_location(f"ltx2_reward_plugin_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import reward plugin: {raw}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    new = sorted(set(_REGISTRY) - before)
    logger.info("reward plugins %s registered %s", [str(p) for p in paths], new)
    return new


@dataclass
class RewardStack:
    """Holds the selected rewards + weights and scores samples with VRAM sequencing."""

    weights: Dict[str, float]
    device: Any = None
    reward_args: Dict[str, dict] = field(default_factory=dict)
    _rewards: Dict[str, BaseReward] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rewards = {name: get_reward_cls(name)() for name in self.weights}

    @classmethod
    def from_spec(cls, spec: str, device: Any = None, reward_args: Dict[str, dict] = None) -> "RewardStack":
        return cls(weights=parse_reward_spec(spec), device=device, reward_args=reward_args or {})

    @property
    def routes(self) -> Dict[str, str]:
        return {name: reward.route for name, reward in self._rewards.items()}

    def score_all(self, samples: List[dict]) -> Dict[str, List[float]]:
        """Score every sample with every reward, loading ONE reward model at a time.

        For each reward: ``setup`` -> ``score`` -> ``teardown`` (teardown always runs,
        even on error), guaranteeing reward models are never co-resident.
        Returns ``{reward_name: [score per sample]}``.
        """
        results: Dict[str, List[float]] = {}
        for name, reward in self._rewards.items():
            try:
                reward.setup(self.device, **self.reward_args.get(name, {}))
                scores, _info = reward.score(samples)
                if len(scores) != len(samples):
                    raise ValueError(f"reward '{name}' returned {len(scores)} scores for {len(samples)} samples")
                results[name] = [float(s) for s in scores]
            finally:
                reward.teardown()
        return results

    def score_groups(self, groups: List[dict]) -> Dict[str, List[List[float]]]:
        """Score grouped rollout samples with every selected reward.

        Returns ``{reward_name: [[score per sample in group], ...]}``. Existing
        scalar rewards use ``BaseReward.score_groups`` fallback; pairwise rewards
        override the hook and can compare samples inside each group.
        """
        results: Dict[str, List[List[float]]] = {}
        for name, reward in self._rewards.items():
            try:
                reward.setup(self.device, **self.reward_args.get(name, {}))
                if hasattr(reward, "score_groups"):
                    grouped_scores, _info = reward.score_groups(groups)
                else:
                    flat = [sample for group in groups for sample in group.get("samples", [])]
                    flat_scores, _info = reward.score(flat)
                    grouped_scores = []
                    idx = 0
                    for group in groups:
                        n = len(group.get("samples", []))
                        grouped_scores.append(flat_scores[idx : idx + n])
                        idx += n
                if len(grouped_scores) != len(groups):
                    raise ValueError(f"reward '{name}' returned {len(grouped_scores)} groups for {len(groups)} groups")
                checked: List[List[float]] = []
                for group, scores in zip(groups, grouped_scores):
                    expected = len(group.get("samples", []))
                    if len(scores) != expected:
                        raise ValueError(
                            f"reward '{name}' returned {len(scores)} scores for group {group.get('group_idx')} "
                            f"with {expected} samples"
                        )
                    checked.append([float(s) for s in scores])
                results[name] = checked
            finally:
                reward.teardown()
        return results
