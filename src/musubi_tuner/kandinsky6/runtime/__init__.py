"""Bundled Kandinsky 6 inference runtime."""

from __future__ import annotations

__all__ = ("get_pipeline",)


def __getattr__(name: str):
    if name == "get_pipeline":
        from .pipeline.factory import get_pipeline

        return get_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
