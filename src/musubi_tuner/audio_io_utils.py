from __future__ import annotations

from typing import Optional

import numpy as np


def coerce_decoded_audio_to_channels_first(audio: np.ndarray, channels: Optional[int] = None) -> np.ndarray:
    """Normalize decoded audio arrays to [channels, samples] layout.

    Decoders may return:
    - packed 1D interleaved data: [L0, R0, L1, R1, ...]
    - packed 2D sample-major: [samples, channels]
    - planar 2D channel-major: [channels, samples]
    """
    arr = np.asarray(audio)

    if channels is not None:
        channels = int(channels)
        if channels <= 0:
            channels = None

    if arr.ndim == 1:
        if channels is not None and channels > 1 and arr.size % channels == 0:
            return arr.reshape(-1, channels).T
        return arr.reshape(1, -1)

    if arr.ndim != 2:
        raise ValueError(f"Unexpected audio ndarray shape: {arr.shape}")

    if channels is not None:
        if arr.shape[0] == channels:
            return arr
        if arr.shape[1] == channels:
            return arr.T

    # Fallback heuristic when channel count is unknown:
    # if first axis is larger, treat as [samples, channels].
    if arr.shape[0] > arr.shape[1]:
        return arr.T
    return arr

