"""AV-Align reward — audio/video peak-alignment IoU (algorithmic, no model).

Extra deps (not installed with this package): ``pip install librosa`` for the
reference-exact onset path (without it a torchaudio fallback is used).

AV-Align (Yariv et al.) measures audio<->video synchronization: detect audio onset
peaks and video optical-flow peaks, then compute the Intersection-over-Union of their
times within a 1-frame tolerance window. Higher = better aligned. This is a pure
algorithm — there is no neural model and no checkpoint — so ``setup``/``teardown`` are
no-ops and scoring runs entirely in-process.

The peak-detection primitives are vendored in ``..vendor.av_align``
(deps: numpy / opencv / librosa / torch / torchaudio). The per-sample score detects
audio peaks, extracts frames + fps, detects video peaks, and computes IoU; any
``None``/NaN result maps to ``0.0``, and any per-sample failure is caught and scored
``0.0``.

NOTE: ``detect_audio_peaks`` uses ``librosa`` as its EXACT primary onset path
(``onset_strength`` + ``onset_detect``), bit-identical to the original AV-Align
metric. If ``librosa`` is not installed it falls back to a torch/torchaudio spectral-flux
onset detector whose parity is APPROXIMATE (the absolute IoU may shift), but which still
preserves in-sync vs out-of-sync discrimination. See ``..vendor.av_align`` module
docstring for details.

route = "sync": AV-Align is intrinsically a joint audio+video signal, so the advantage
is applied to BOTH branches. It needs decoded media files on disk (``video_file`` +
``audio_file``); it becomes usable once AV generation writes those out.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from ..registry import BaseReward, register_reward

logger = logging.getLogger(__name__)


def _single_av_align_score(
    video_file: str,
    audio_file: str,
    size: Optional[str] = None,
    max_length_s: Optional[float] = None,
) -> float:
    """IoU of audio onset peaks vs video optical-flow peaks.

    Detect audio peaks, extract frames + fps, detect video peaks, compute IoU;
    ``None``/NaN -> ``0.0``.
    """
    import numpy as np

    from ..vendor.av_align import (
        calc_intersection_over_union,
        detect_audio_peaks,
        detect_video_peaks,
        extract_frames,
    )

    audio_peaks = detect_audio_peaks(audio_file, max_length_s=max_length_s)
    frames, fps = extract_frames(video_file, size, max_length_s=max_length_s)
    _, video_peaks = detect_video_peaks(frames, fps, use_tqdm=False)

    s = calc_intersection_over_union(audio_peaks, video_peaks, fps)
    if s is None or (isinstance(s, float) and np.isnan(s)):
        s = 0.0
    return float(s)


@register_reward("av_align")
class AVAlignReward(BaseReward):
    """Audio/video peak-alignment IoU reward (route=sync, no model)."""

    kind = "blackbox"
    route = "sync"
    needs = frozenset({"video_file", "audio_file"})

    def __init__(self) -> None:
        self._size = None
        self._max_length_s = None

    def setup(self, device, *, size=None, max_length_s=None, **_ignored) -> None:
        # No model to load — purely algorithmic. Just capture optional scoring options.
        self._size = size
        self._max_length_s = None if max_length_s is None else float(max_length_s)

    def score(self, samples: List[dict]) -> Tuple[List[float], dict]:
        scores: List[float] = []
        for sample in samples:
            video_file = sample.get("video_file")
            audio_file = sample.get("audio_file")
            if not video_file or not audio_file:
                logger.warning("av_align: sample missing video_file/audio_file; scoring 0.0")
                scores.append(0.0)
                continue
            try:
                scores.append(
                    _single_av_align_score(
                        video_file=video_file,
                        audio_file=audio_file,
                        size=self._size,
                        max_length_s=self._max_length_s,
                    )
                )
            except Exception as exc:  # any failure -> 0.0
                logger.warning("av_align: scoring failed for %s / %s: %r", video_file, audio_file, exc)
                scores.append(0.0)
        return scores, {"reward": "av_align"}

    def teardown(self) -> None:
        # No model to free.
        pass
