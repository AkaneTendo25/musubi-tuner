"""Standalone LU + DiT tiled super-resolution inference."""

from __future__ import annotations

__version__ = "0.3.0"


def load_sr_pipeline(*args, **kwargs):
    """Load SR lazily so the base K6 config does not import the SR model graph."""
    from .factory import load_sr_pipeline as _load_sr_pipeline

    return _load_sr_pipeline(*args, **kwargs)


__all__ = ["load_sr_pipeline"]
