from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import torch


class MediaModality(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class CropMode(str, Enum):
    BEGINNING = "beginning"
    END = "end"
    RANDOM = "random"


class PadMode(str, Enum):
    ERROR = "error"
    ZERO = "zero"
    REPEAT = "repeat"


class MissingMediaPolicy(str, Enum):
    """How to handle an absent media source or stream.

    Decode failures remain errors regardless of this policy.
    """

    ERROR = "error"
    DROP = "drop"
    ZERO = "zero"


_ROLE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class MediaAsset:
    """An H3 input or target asset resolved from a Musubi dataset record."""

    path: Path
    modality: MediaModality
    role: str
    stream_index: int = 0
    start_seconds: float = 0.0
    duration_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not _ROLE_PATTERN.fullmatch(self.role):
            raise ValueError(f"media role must match {_ROLE_PATTERN.pattern}: {self.role!r}")
        if self.stream_index < 0:
            raise ValueError("stream_index must be non-negative")
        if self.start_seconds < 0:
            raise ValueError("start_seconds must be non-negative")
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")


@dataclass(frozen=True)
class AudioProcessingSpec:
    sample_rate: int
    channels: int
    clip_duration_seconds: float | None = None
    crop_mode: CropMode = CropMode.BEGINNING
    pad_mode: PadMode = PadMode.ERROR
    missing: MissingMediaPolicy = MissingMediaPolicy.ERROR
    peak_normalize: bool = False

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("audio sample_rate must be positive")
        if self.channels not in (1, 2):
            raise ValueError("audio channels must be 1 or 2")
        if self.clip_duration_seconds is not None and self.clip_duration_seconds <= 0:
            raise ValueError("clip_duration_seconds must be positive")


@dataclass(frozen=True)
class AudioClip:
    waveform: torch.Tensor
    sample_rate: int
    valid_mask: torch.Tensor
    source_start_seconds: float


def slice_media_asset(asset: MediaAsset, *, start_seconds: float, duration_seconds: float) -> MediaAsset:
    return replace(
        asset,
        start_seconds=asset.start_seconds + start_seconds,
        duration_seconds=duration_seconds,
    )


def fit_audio_length(
    waveform: torch.Tensor,
    target_samples: int,
    *,
    crop_mode: CropMode,
    pad_mode: PadMode,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if waveform.dim() != 2:
        raise ValueError("audio waveform must have shape (channels, samples)")
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")
    samples = waveform.shape[1]
    if samples >= target_samples:
        maximum_start = samples - target_samples
        if crop_mode is CropMode.BEGINNING:
            start = 0
        elif crop_mode is CropMode.END:
            start = maximum_start
        else:
            start = (rng or random).randint(0, maximum_start)
        return waveform[:, start : start + target_samples], torch.ones(target_samples, dtype=torch.bool), start

    if pad_mode is PadMode.ERROR:
        raise ValueError(f"audio has {samples} samples but {target_samples} are required")
    padding = target_samples - samples
    if pad_mode is PadMode.ZERO or samples == 0:
        tail = waveform.new_zeros((waveform.shape[0], padding))
    else:
        repeats = (padding + samples - 1) // samples
        tail = waveform.repeat(1, repeats + 1)[:, :padding]
    valid_mask = torch.zeros(target_samples, dtype=torch.bool)
    valid_mask[:samples] = True
    return torch.cat((waveform, tail), dim=1), valid_mask, 0
