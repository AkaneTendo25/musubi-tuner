"""Offload contract consumed by the SR pipeline stages.

The SR package never decides *how* modules are offloaded; the embedding
application (for example the K6 pipeline) builds a concrete handle and injects
it into :func:`musubi_tuner.kandinsky6.runtime.sr.pipeline.factory.load_sr_pipeline`. Stages only
rely on this structural protocol, so any object exposing ``register`` / ``use``
/ ``release`` works. :class:`NoOpOffload` is the default when nothing is injected.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal, Protocol

OffloadStrategy = Literal["none", "module", "block"]


class OffloadHandle(Protocol):
    """Structural interface for module residency management."""

    strategy: OffloadStrategy

    def register(self, name: str, module: Any) -> None:
        """Track ``module`` under ``name`` so later ``use`` calls can move it."""
        ...

    def release(self, *names: str) -> None:
        """Move the named modules off the compute device once a call is finished."""
        ...

    @contextmanager
    def use(self, *names: str, prefetch: str | Sequence[str] | None = None) -> Iterator[None]:
        """Ensure the named modules reside on the compute device for the block."""
        ...


class NoOpOffload:
    """Zero-cost handle so pipeline stages never branch on offload mode."""

    strategy: OffloadStrategy = "none"

    def register(self, name: str, module: Any) -> None:
        """Nothing to track: every module stays where it was loaded."""
        return None

    def release(self, *names: str) -> None:
        """Nothing to release: modules are always resident."""
        return None

    @contextmanager
    def use(self, *names: str, prefetch: str | Sequence[str] | None = None) -> Iterator[None]:
        """Yield immediately: modules are always resident."""
        yield


__all__ = ["NoOpOffload", "OffloadHandle", "OffloadStrategy"]
