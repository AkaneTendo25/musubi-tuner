from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Optional

import torch


SUPPORTED_GROUP_LR_SCHEDULERS = {
    "constant",
    "constant_with_warmup",
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
}


def parse_group_lr_warmup_args(raw_args: list[str] | None) -> dict[str, int]:
    """Parse ``pattern=warmup_steps`` CLI entries."""
    if not raw_args:
        return {}

    warmups: dict[str, int] = {}
    for entry in raw_args:
        if "=" not in entry:
            raise ValueError(f"Invalid --lr_group_warmup_args entry (expected pattern=steps): {entry}")
        pattern, steps = entry.split("=", 1)
        pattern = pattern.strip()
        if not pattern:
            raise ValueError("--lr_group_warmup_args patterns must not be empty")
        re.compile(pattern)
        warmup_steps = int(steps)
        if warmup_steps < 0:
            raise ValueError(f"--lr_group_warmup_args warmup steps must be >= 0: {entry}")
        warmups[pattern] = warmup_steps
    return warmups


@dataclass(frozen=True)
class GroupLRScheduleRule:
    pattern: str
    scheduler: str
    warmup_steps: Optional[int] = None
    stable_steps: int = 0
    decay_steps: Optional[int] = None
    min_lr_ratio: float = 0.0
    num_cycles: float = 0.5
    power: float = 1.0


def parse_group_lr_scheduler_args(raw_args: list[str] | None) -> list[GroupLRScheduleRule]:
    """Parse ordered ``pattern=scheduler=...,key=value`` group schedule rules."""
    if not raw_args:
        return []

    rules: list[GroupLRScheduleRule] = []
    seen_patterns: set[str] = set()
    allowed_keys = {
        "scheduler",
        "warmup_steps",
        "stable_steps",
        "decay_steps",
        "min_lr_ratio",
        "num_cycles",
        "power",
    }
    for entry in raw_args:
        if "=" not in entry:
            raise ValueError(f"Invalid --lr_group_scheduler_args entry (expected pattern=scheduler=...,key=value): {entry}")
        pattern, raw_config = entry.split("=", 1)
        pattern = pattern.strip()
        if not pattern:
            raise ValueError("--lr_group_scheduler_args patterns must not be empty")
        if pattern in seen_patterns:
            raise ValueError(f"Duplicate --lr_group_scheduler_args pattern: {pattern}")
        re.compile(pattern)
        seen_patterns.add(pattern)

        values: dict[str, str] = {}
        for item in raw_config.split(","):
            if "=" not in item:
                raise ValueError(f"Invalid --lr_group_scheduler_args setting (expected key=value in {entry!r}): {item}")
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key not in allowed_keys:
                raise ValueError(f"Unknown --lr_group_scheduler_args key {key!r}; expected one of {sorted(allowed_keys)}")
            if key in values:
                raise ValueError(f"Duplicate --lr_group_scheduler_args key {key!r} in: {entry}")
            values[key] = value

        scheduler = values.get("scheduler", "").lower()
        if scheduler not in SUPPORTED_GROUP_LR_SCHEDULERS:
            raise ValueError(f"Group scheduler must be one of {sorted(SUPPORTED_GROUP_LR_SCHEDULERS)}; got {scheduler!r}")

        warmup_steps = int(values["warmup_steps"]) if "warmup_steps" in values else None
        stable_steps = int(values.get("stable_steps", 0))
        decay_steps = int(values["decay_steps"]) if "decay_steps" in values else None
        min_lr_ratio = float(values.get("min_lr_ratio", 0.0))
        num_cycles = float(values.get("num_cycles", 0.5))
        power = float(values.get("power", 1.0))
        if warmup_steps is not None and warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0 in: {entry}")
        if stable_steps < 0:
            raise ValueError(f"stable_steps must be >= 0 in: {entry}")
        if decay_steps is not None and decay_steps <= 0:
            raise ValueError(f"decay_steps must be > 0 in: {entry}")
        if not math.isfinite(min_lr_ratio) or not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be finite and within [0, 1] in: {entry}")
        if not math.isfinite(num_cycles) or num_cycles <= 0.0:
            raise ValueError(f"num_cycles must be finite and > 0 in: {entry}")
        if not math.isfinite(power) or power <= 0.0:
            raise ValueError(f"power must be finite and > 0 in: {entry}")

        rules.append(
            GroupLRScheduleRule(
                pattern=pattern,
                scheduler=scheduler,
                warmup_steps=warmup_steps,
                stable_steps=stable_steps,
                decay_steps=decay_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
                power=power,
            )
        )
    return rules


class GroupLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Apply independent schedules to matching optimizer groups.

    Unmatched groups continue to use the wrapped scheduler. Ordered full
    schedule rules use first-match precedence. Legacy warmup-only overrides
    remain available for unmatched groups.
    """

    def __init__(
        self,
        base_scheduler: Any,
        optimizer,
        *,
        num_training_steps: int,
        default_warmup_steps: int = 0,
        warmup_overrides: dict[str, int] | None = None,
        schedule_rules: list[GroupLRScheduleRule] | None = None,
    ) -> None:
        self.scheduler = base_scheduler
        self.optimizer = optimizer
        self.num_training_steps = max(int(num_training_steps), 1)
        self.default_warmup_steps = max(int(default_warmup_steps), 0)
        self._raw_overrides = dict(warmup_overrides or {})
        self._schedule_rules = list(schedule_rules or [])
        base_lrs = list(getattr(base_scheduler, "base_lrs", []))
        if len(base_lrs) != len(optimizer.param_groups):
            base_lrs = [float(group.get("initial_lr", group["lr"])) for group in optimizer.param_groups]
        self._base_lrs = [float(lr) for lr in base_lrs]
        self._last_lr: list[float] = []
        self._compile_rules()
        self._validate_rule_matches()
        self._apply_group_schedules()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.scheduler, name)

    def state_dict(self) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler.state_dict(),
            "num_training_steps": self.num_training_steps,
            "default_warmup_steps": self.default_warmup_steps,
            "warmup_overrides": self._raw_overrides,
            "schedule_rules": [asdict(rule) for rule in self._schedule_rules],
            "base_lrs": self._base_lrs,
            "last_lr": self._last_lr,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.num_training_steps = int(state_dict.get("num_training_steps", self.num_training_steps))
        self.default_warmup_steps = int(state_dict.get("default_warmup_steps", 0))
        self._raw_overrides = dict(state_dict.get("warmup_overrides", {}))
        self._schedule_rules = [GroupLRScheduleRule(**rule) for rule in state_dict.get("schedule_rules", [])]
        self._base_lrs = [float(lr) for lr in state_dict.get("base_lrs", self._base_lrs)]
        if len(self._base_lrs) != len(self.optimizer.param_groups):
            raise ValueError("Cannot resume per-group LR schedules because the optimizer parameter-group count changed")
        self._last_lr = list(state_dict.get("last_lr", []))
        self._compile_rules()
        self._validate_rule_matches()
        self._apply_group_schedules()

    def get_last_lr(self) -> list[float]:
        if self._last_lr:
            return list(self._last_lr)
        if hasattr(self.scheduler, "get_last_lr"):
            return list(self.scheduler.get_last_lr())
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def step(self, *args, **kwargs) -> None:
        self.scheduler.step(*args, **kwargs)
        self._apply_group_schedules()

    def _compile_rules(self) -> None:
        self._compiled_overrides = [(re.compile(pattern), steps) for pattern, steps in self._raw_overrides.items()]
        self._compiled_schedule_rules = [(re.compile(rule.pattern), rule) for rule in self._schedule_rules]

    def _validate_rule_matches(self) -> None:
        group_names = [str(group.get("group_name", "")) for group in self.optimizer.param_groups]
        for pattern, rule in self._compiled_schedule_rules:
            if not any(pattern.search(group_name) for group_name in group_names):
                raise ValueError(
                    f"Per-group LR scheduler pattern {rule.pattern!r} did not match any optimizer group: {group_names}"
                )

    def _resolve_warmup_override(self, group_name: str) -> Optional[int]:
        for pattern, steps in self._compiled_overrides:
            if pattern.search(group_name):
                return steps
        return None

    def _resolve_schedule_rule(self, group_name: str) -> Optional[GroupLRScheduleRule]:
        for pattern, rule in self._compiled_schedule_rules:
            if pattern.search(group_name):
                return rule
        return None

    def _current_step(self) -> int:
        last_epoch = int(getattr(self.scheduler, "last_epoch", 0))
        return max(last_epoch, 0)

    @staticmethod
    def _warmup_factor(step: int, warmup_steps: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(float(step) / float(warmup_steps), 1.0)

    def _schedule_factor(self, step: int, rule: GroupLRScheduleRule) -> float:
        warmup_steps = self.default_warmup_steps if rule.warmup_steps is None else rule.warmup_steps
        if step < warmup_steps:
            return self._warmup_factor(step, warmup_steps)

        elapsed = max(step - warmup_steps, 0)
        if elapsed <= rule.stable_steps:
            return 1.0
        if rule.scheduler in {"constant", "constant_with_warmup"}:
            return 1.0

        available_decay_steps = max(
            self.num_training_steps - warmup_steps - rule.stable_steps,
            1,
        )
        decay_steps = rule.decay_steps or available_decay_steps
        progress = min(max((elapsed - rule.stable_steps) / decay_steps, 0.0), 1.0)

        if rule.scheduler == "linear":
            shape = 1.0 - progress
        elif rule.scheduler == "cosine":
            shape = 0.5 * (1.0 + math.cos(2.0 * math.pi * rule.num_cycles * progress))
        elif rule.scheduler == "cosine_with_restarts":
            if progress >= 1.0:
                shape = 0.0
            else:
                cycle_progress = (rule.num_cycles * progress) % 1.0
                shape = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
        elif rule.scheduler == "polynomial":
            shape = (1.0 - progress) ** rule.power
        else:  # guarded by the parser
            raise ValueError(f"Unsupported group LR scheduler: {rule.scheduler}")
        return rule.min_lr_ratio + (1.0 - rule.min_lr_ratio) * shape

    def _apply_group_schedules(self) -> None:
        step = self._current_step()
        default_factor = max(
            self._warmup_factor(step, self.default_warmup_steps),
            1e-12,
        )

        self._last_lr = []
        for index, group in enumerate(self.optimizer.param_groups):
            group_name = str(group.get("group_name", ""))
            rule = self._resolve_schedule_rule(group_name)
            if rule is not None:
                lr = self._base_lrs[index] * self._schedule_factor(step, rule)
                group["lr"] = lr
                self._last_lr.append(lr)
                continue

            base_lr = float(group["lr"])
            override_steps = self._resolve_warmup_override(group_name)
            if override_steps is not None:
                desired_factor = self._warmup_factor(step, override_steps)
                base_lr *= desired_factor / default_factor
                group["lr"] = base_lr
            self._last_lr.append(base_lr)


class GroupWarmupScheduler(GroupLRScheduler):
    """Backward-compatible warmup-only wrapper."""

    def __init__(
        self,
        base_scheduler: Any,
        optimizer,
        *,
        default_warmup_steps: int = 0,
        warmup_overrides: dict[str, int] | None = None,
    ) -> None:
        super().__init__(
            base_scheduler,
            optimizer,
            num_training_steps=1,
            default_warmup_steps=default_warmup_steps,
            warmup_overrides=warmup_overrides,
        )
