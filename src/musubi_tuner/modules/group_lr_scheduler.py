from __future__ import annotations

import re
from typing import Any, Optional


def parse_group_lr_warmup_args(raw_args: list[str] | None) -> dict[str, int]:
    """Parse ``pattern=warmup_steps`` CLI entries."""
    if not raw_args:
        return {}

    warmups: dict[str, int] = {}
    for entry in raw_args:
        if "=" not in entry:
            raise ValueError(f"Invalid --lr_group_warmup_args entry (expected pattern=steps): {entry}")
        pattern, steps = entry.split("=", 1)
        warmups[pattern.strip()] = int(steps)
    return warmups


class GroupWarmupScheduler:
    """Wrapper that applies per-group warmup overrides on top of a base scheduler.

    The base scheduler still controls the global schedule family and decay shape.
    This wrapper only overrides the warmup ramp for matching parameter groups,
    keeping the default path fully unchanged when no overrides are configured.
    """

    def __init__(
        self,
        base_scheduler: Any,
        optimizer,
        *,
        default_warmup_steps: int = 0,
        warmup_overrides: dict[str, int] | None = None,
    ) -> None:
        self.scheduler = base_scheduler
        self.optimizer = optimizer
        self.default_warmup_steps = max(int(default_warmup_steps), 0)
        self._raw_overrides = dict(warmup_overrides or {})
        self._compiled_overrides = [
            (re.compile(pattern), max(int(steps), 0))
            for pattern, steps in self._raw_overrides.items()
        ]
        self._last_lr: list[float] = []
        self._apply_group_warmup()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.scheduler, name)

    def state_dict(self) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler.state_dict(),
            "default_warmup_steps": self.default_warmup_steps,
            "warmup_overrides": self._raw_overrides,
            "last_lr": self._last_lr,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.default_warmup_steps = int(state_dict.get("default_warmup_steps", 0))
        self._raw_overrides = dict(state_dict.get("warmup_overrides", {}))
        self._compiled_overrides = [
            (re.compile(pattern), max(int(steps), 0))
            for pattern, steps in self._raw_overrides.items()
        ]
        self._last_lr = list(state_dict.get("last_lr", []))
        self._apply_group_warmup()

    def get_last_lr(self) -> list[float]:
        if self._last_lr:
            return list(self._last_lr)
        if hasattr(self.scheduler, "get_last_lr"):
            return list(self.scheduler.get_last_lr())
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, *args, **kwargs) -> None:
        self.scheduler.step(*args, **kwargs)
        self._apply_group_warmup()

    def _resolve_override(self, group_name: str) -> Optional[int]:
        for pattern, steps in self._compiled_overrides:
            if pattern.search(group_name):
                return steps
        return None

    def _current_step(self) -> int:
        last_epoch = int(getattr(self.scheduler, "last_epoch", 0))
        return max(last_epoch, 0)

    def _warmup_factor(self, step: int, warmup_steps: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(float(step) / float(warmup_steps), 1.0)

    def _apply_group_warmup(self) -> None:
        step = self._current_step()
        default_factor = self._warmup_factor(step, self.default_warmup_steps)
        default_factor = max(default_factor, 1e-12)

        self._last_lr = []
        for group in self.optimizer.param_groups:
            base_lr = float(group["lr"])
            group_name = str(group.get("group_name", ""))
            override_steps = self._resolve_override(group_name)
            if override_steps is not None:
                desired_factor = self._warmup_factor(step, override_steps)
                corrected_lr = base_lr * (desired_factor / default_factor)
                group["lr"] = corrected_lr
                self._last_lr.append(corrected_lr)
            else:
                self._last_lr.append(base_lr)
