"""Audio file I/O for YuE2 outputs (PyAV): 24-bit FLAC (upstream default) or 32-bit float WAV."""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Optional

import av
import torch

from musubi_tuner.dataset.audio_utils import AudioSource, decode_audio
from musubi_tuner.yue2.yue2_protocol import SAMPLE_RATE

_LAYOUTS = {1: "mono", 2: "stereo"}
# codec, encoder sample format
_FORMATS = {"flac": ("flac", "s32"), "wav": ("pcm_f32le", "flt")}


def write_audio(
    path: str,
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    fmt: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> str:
    """Write ``waveform [C, S]`` (float in [-1, 1], C = 1 or 2) as FLAC (24-bit) or WAV (float). ``fmt`` defaults to
    the file extension. ``metadata`` becomes container tags (Vorbis comments for FLAC)."""
    fmt = (fmt or os.path.splitext(path)[1].lstrip(".") or "flac").lower()
    if fmt not in _FORMATS:
        raise ValueError(f"audio format must be one of {sorted(_FORMATS)}, got {fmt!r}")
    if waveform.ndim != 2 or waveform.shape[0] not in _LAYOUTS or waveform.shape[1] < 1:
        raise ValueError(f"expected a waveform [1|2, S], got {tuple(waveform.shape)}")
    codec, sample_fmt = _FORMATS[fmt]
    layout = _LAYOUTS[waveform.shape[0]]
    samples = waveform.detach().float().cpu().clamp(-1.0, 1.0).contiguous().numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w", format=fmt) as container:
        for key, value in (metadata or {}).items():
            container.metadata[str(key)] = str(value)
        stream = container.add_stream(codec, rate=sample_rate)
        stream.layout = layout
        stream.codec_context.format = sample_fmt
        for start in range(0, samples.shape[1], 4096):
            frame = av.AudioFrame.from_ndarray(samples[:, start : start + 4096], format="fltp", layout=layout)
            frame.sample_rate = sample_rate
            frame.pts = start
            frame.time_base = Fraction(1, sample_rate)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return str(path)


def read_audio(path: str, sample_rate: int = SAMPLE_RATE, channels: int = 2) -> torch.Tensor:
    """Decode a file to ``[channels, S]`` float32 at ``sample_rate`` (``dataset.audio_utils.decode_audio``)."""
    return decode_audio(AudioSource(Path(path), embedded=False), sample_rate=sample_rate, channels=channels)
