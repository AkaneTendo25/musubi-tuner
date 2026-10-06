"""Result types returned by the packaged SR pipeline."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict
from torch import Tensor


class SRPipelineOutput(BaseModel):
    """Output of :class:`musubi_tuner.kandinsky6.runtime.sr.pipeline.sr_pipeline.Kandinsky6SRPipeline`.

    Attributes:
        frames: ``(batch, 3, frames, height, width)`` uint8 video.
        audio: Per-sample waveforms carried through from the caller, or ``None``.
        path: Output file written when ``save_path`` was passed (one per sample for a batch), else ``None``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    frames: Tensor
    audio: list[np.ndarray] | None = None
    path: str | list[str] | None = None


__all__ = ["SRPipelineOutput"]
