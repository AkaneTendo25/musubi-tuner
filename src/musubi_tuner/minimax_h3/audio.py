from __future__ import annotations

import random

import numpy as np
import torch

from musubi_tuner.minimax_h3.architecture import (
    AUDIO_CHANNELS,
    AUDIO_HOP_LENGTH,
    AUDIO_SAMPLE_RATE,
    temporal_shape,
)
from musubi_tuner.minimax_h3.media import (
    AudioClip,
    AudioProcessingSpec,
    CropMode,
    MediaAsset,
    MediaModality,
    MissingMediaPolicy,
    PadMode,
    fit_audio_length,
)


class AudioDecodeError(RuntimeError):
    pass


def audio_valid_mask_to_latent_mask(
    sample_mask: torch.Tensor,
    *,
    hop_length: int = AUDIO_HOP_LENGTH,
) -> torch.Tensor:
    """Conservatively mark an audio latent valid only when its input hop is valid.

    A backend may apply a stricter receptive-field-aware mask when required by
    its audio VAE; it must never mark more latents valid.
    """
    if sample_mask.dtype is not torch.bool or sample_mask.ndim != 1:
        raise ValueError("H3 audio sample mask must be one-dimensional bool")
    if hop_length <= 0 or sample_mask.numel() % hop_length:
        raise ValueError("H3 audio sample mask length must be divisible by the audio VAE hop length")
    return sample_mask.reshape(-1, hop_length).all(dim=1)


def _silent_clip(spec: AudioProcessingSpec) -> AudioClip:
    if spec.clip_duration_seconds is None:
        raise AudioDecodeError("zero substitution requires clip_duration_seconds")
    samples = round(spec.clip_duration_seconds * spec.sample_rate)
    return AudioClip(
        waveform=torch.zeros((spec.channels, samples), dtype=torch.float32),
        sample_rate=spec.sample_rate,
        valid_mask=torch.zeros(samples, dtype=torch.bool),
        source_start_seconds=0.0,
    )


def _handle_missing(spec: AudioProcessingSpec, message: str) -> AudioClip | None:
    if spec.missing is MissingMediaPolicy.DROP:
        return None
    if spec.missing is MissingMediaPolicy.ZERO:
        return _silent_clip(spec)
    raise AudioDecodeError(message)


def load_audio_asset(
    asset: MediaAsset,
    spec: AudioProcessingSpec,
    *,
    rng: random.Random | None = None,
) -> AudioClip | None:
    """Decode an audio stream from an H3 audio file or video container."""
    if asset.modality not in (MediaModality.AUDIO, MediaModality.VIDEO):
        raise ValueError("load_audio_asset requires an audio file or video container")
    if not asset.path.is_file():
        return _handle_missing(spec, f"audio source not found: {asset.path}")

    try:
        import av

        with av.open(str(asset.path)) as container:
            streams = tuple(container.streams.audio)
            if asset.stream_index >= len(streams):
                return _handle_missing(
                    spec,
                    f"audio stream {asset.stream_index} not found in {asset.path}; container has {len(streams)} audio streams",
                )
            stream = streams[asset.stream_index]
            layout = "mono" if spec.channels == 1 else "stereo"
            resampler = av.AudioResampler(format="fltp", layout=layout, rate=spec.sample_rate)
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    chunk = converted.to_ndarray()
                    chunks.append(chunk if chunk.ndim == 2 else chunk[None, :])
            for converted in resampler.resample(None):
                chunk = converted.to_ndarray()
                chunks.append(chunk if chunk.ndim == 2 else chunk[None, :])
    except AudioDecodeError:
        raise
    except Exception as error:
        raise AudioDecodeError(f"cannot decode audio from {asset.path}: {error}") from error

    if not chunks:
        raise AudioDecodeError(f"audio stream in {asset.path} produced no samples")
    waveform = torch.from_numpy(np.concatenate(chunks, axis=1)).to(torch.float32)
    start = round(asset.start_seconds * spec.sample_rate)
    if start >= waveform.shape[1]:
        return _handle_missing(spec, f"audio start time lies beyond the end of {asset.path}")
    end = waveform.shape[1]
    if asset.duration_seconds is not None:
        end = min(end, start + round(asset.duration_seconds * spec.sample_rate))
    waveform = waveform[:, start:end]

    source_offset = 0
    if spec.clip_duration_seconds is not None:
        target_samples = round(spec.clip_duration_seconds * spec.sample_rate)
        waveform, valid_mask, source_offset = fit_audio_length(
            waveform,
            target_samples,
            crop_mode=spec.crop_mode,
            pad_mode=spec.pad_mode,
            rng=rng,
        )
    else:
        valid_mask = torch.ones(waveform.shape[1], dtype=torch.bool)

    if spec.peak_normalize and waveform.numel():
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
    return AudioClip(
        waveform=waveform.contiguous(),
        sample_rate=spec.sample_rate,
        valid_mask=valid_mask,
        source_start_seconds=asset.start_seconds + source_offset / spec.sample_rate,
    )


def target_audio_processing_spec(asset: MediaAsset) -> AudioProcessingSpec:
    """Build the exact target-audio shape paired with an H3 target video crop."""
    if asset.role != "target" or asset.modality not in {MediaModality.VIDEO, MediaModality.AUDIO}:
        raise ValueError("target_audio_processing_spec requires an H3 target video or audio clip")
    frame_count = asset.metadata.get("frame_count")
    if not isinstance(frame_count, int):
        raise TypeError("H3 target media metadata must contain an integer frame_count")
    shape = temporal_shape(frame_count)
    return AudioProcessingSpec(
        sample_rate=AUDIO_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
        clip_duration_seconds=shape.audio_samples / AUDIO_SAMPLE_RATE,
        crop_mode=CropMode.BEGINNING,
        pad_mode=PadMode.ZERO,
        missing=MissingMediaPolicy.ZERO,
    )
