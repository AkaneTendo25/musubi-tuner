"""Progress reporting for a tiled SR run.

The stages call back once per denoising step; a :class:`ProgressReporter`
turns those increments into a console bar, a host UI bar (ComfyUI), or both.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from tqdm.auto import tqdm


class ProgressReporter(Protocol):
    """Receives the denoising progress of one SR run."""

    def start(self, total_tiles: int, steps_per_tile: int) -> None:
        """Announce the run's total work before the first denoising step."""

    def update(self, n: int = 1) -> None:
        """Advance by ``n`` denoising steps."""

    def close(self) -> None:
        """Finish the run (also called when the run fails)."""


class TqdmProgress:
    """Console progress bar over ``tiles x steps``."""

    def __init__(self) -> None:
        self.bar: Any | None = None

    def start(self, total_tiles: int, steps_per_tile: int) -> None:
        self.bar = tqdm(
            total=total_tiles * steps_per_tile,
            desc=f"SR denoising [{total_tiles} tiles x {steps_per_tile} steps]",
            unit="step",
        )

    def update(self, n: int = 1) -> None:
        if self.bar is not None:
            self.bar.update(n)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None


class CompositeProgress:
    """Fan one run's progress out to several reporters."""

    def __init__(self, reporters: Sequence[ProgressReporter]) -> None:
        self.reporters = list(reporters)

    def start(self, total_tiles: int, steps_per_tile: int) -> None:
        for reporter in self.reporters:
            reporter.start(total_tiles, steps_per_tile)

    def update(self, n: int = 1) -> None:
        for reporter in self.reporters:
            reporter.update(n)

    def close(self) -> None:
        for reporter in self.reporters:
            reporter.close()


__all__ = ["CompositeProgress", "ProgressReporter", "TqdmProgress"]
