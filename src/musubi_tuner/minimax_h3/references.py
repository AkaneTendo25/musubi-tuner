"""Ref2VA media preparation for MiniMax H3's released reference path."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from musubi_tuner.minimax_h3.architecture import AUDIO_SAMPLE_RATE, CANVAS_MULTIPLE, VIDEO_FPS
from musubi_tuner.minimax_h3.audio import load_audio_asset
from musubi_tuner.minimax_h3.media import AudioProcessingSpec, MediaAsset, MediaModality, MissingMediaPolicy

REFERENCE_IMAGE_SHORT_EDGE = 2048
REFERENCE_VIDEO_SHORT_EDGE = 768
REFERENCE_VIDEO_MAX_PIXELS = 768 * 1344
REFERENCE_VIDEO_SAMPLE_FPS = 2
REFERENCE_VIDEO_TEMPORAL_PATCH = 2
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
MAX_REFERENCES = 12


class H3ReferenceKind(IntEnum):
    IMAGE = 0
    VIDEO = 1
    AUDIO = 2


@dataclass
class H3PreparedReference:
    kind: H3ReferenceKind
    image: Image.Image | None = None
    frames: np.ndarray | None = None
    waveform: torch.Tensor | None = None
    block_timestamps: tuple[float, ...] = ()

    @property
    def has_audio(self) -> bool:
        return self.waveform is not None


def _kind(asset: MediaAsset) -> H3ReferenceKind:
    return {
        MediaModality.IMAGE: H3ReferenceKind.IMAGE,
        MediaModality.VIDEO: H3ReferenceKind.VIDEO,
        MediaModality.AUDIO: H3ReferenceKind.AUDIO,
    }[asset.modality]


def reference_assets(item: Any) -> tuple[MediaAsset, ...]:
    assets = tuple(asset for asset in getattr(item, "h3_media_assets", ()) if asset.role == "reference")
    counts = {
        kind: sum(asset.modality is modality for asset in assets)
        for kind, modality in (
            (H3ReferenceKind.IMAGE, MediaModality.IMAGE),
            (H3ReferenceKind.VIDEO, MediaModality.VIDEO),
            (H3ReferenceKind.AUDIO, MediaModality.AUDIO),
        )
    }
    limits = {
        H3ReferenceKind.IMAGE: MAX_REFERENCE_IMAGES,
        H3ReferenceKind.VIDEO: MAX_REFERENCE_VIDEOS,
        H3ReferenceKind.AUDIO: MAX_REFERENCE_AUDIOS,
    }
    for kind, count in counts.items():
        if count > limits[kind]:
            raise ValueError(f"MiniMax H3 accepts at most {limits[kind]} {kind.name.lower()} references, got {count}")
    if len(assets) > MAX_REFERENCES:
        raise ValueError(f"MiniMax H3 accepts at most {MAX_REFERENCES} references, got {len(assets)}")
    if assets and all(asset.modality is MediaModality.AUDIO for asset in assets):
        raise ValueError("MiniMax H3 audio references require at least one image or video reference")
    return assets


def _multiple_size(width: float, height: float) -> tuple[int, int]:
    return (
        max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def resolve_reference_image_size(width: int, height: int, *, scale: float = 1.0) -> tuple[int, int]:
    if width <= 0 or height <= 0 or width > 4 * height or height > 4 * width:
        raise ValueError(f"H3 reference image must have a positive 1:4 to 4:1 aspect ratio, got {width}x{height}")
    _validate_scale(scale)
    resolved = REFERENCE_IMAGE_SHORT_EDGE * scale / min(width, height)
    return _multiple_size(width * resolved, height * resolved)


def resolve_reference_video_size(width: int, height: int, *, scale: float = 1.0) -> tuple[int, int]:
    if width <= 0 or height <= 0 or width > 4 * height or height > 4 * width:
        raise ValueError(f"H3 reference video must have a positive 1:4 to 4:1 aspect ratio, got {width}x{height}")
    _validate_scale(scale)
    short_edge = REFERENCE_VIDEO_SHORT_EDGE * scale
    ratio = width / height
    if ratio >= 1:
        resolved_width, resolved_height = short_edge * ratio, float(short_edge)
    else:
        resolved_width, resolved_height = float(short_edge), short_edge / ratio
    area = resolved_width * resolved_height
    limit = REFERENCE_VIDEO_MAX_PIXELS * scale * scale
    if area > limit:
        shrink = math.sqrt(limit / area)
        resolved_width *= shrink
        resolved_height *= shrink
    return _multiple_size(resolved_width, resolved_height)


def _validate_scale(scale: float) -> None:
    # H3's spatial rotary grid is area-normalized, so a reference occupies the
    # same coordinate field at any resolution and only its sampling density
    # changes. Scaling below a quarter leaves too few patches to carry identity.
    if not 0.25 <= scale <= 1.0:
        raise ValueError(f"H3 reference scale must lie in [0.25, 1.0], got {scale}")


def _decode_video(path: Path) -> tuple[np.ndarray, float]:
    try:
        import av

        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError(f"H3 reference video has no video stream: {path}")
            stream = container.streams.video[0]
            rate = stream.average_rate or getattr(stream, "guessed_rate", None)
            if rate is None or float(rate) <= 0:
                raise ValueError(f"H3 reference video has no usable frame rate: {path}")
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    except ValueError:
        raise
    except Exception as error:
        raise RuntimeError(f"cannot decode H3 reference video {path}: {error}") from error
    if not frames:
        raise ValueError(f"H3 reference video produced no frames: {path}")
    return np.stack(frames), float(rate)


def resample_reference_frames(frames: np.ndarray, source_fps: float) -> np.ndarray:
    if source_fps <= 0:
        raise ValueError("H3 reference video frame rate must be positive")
    if math.isclose(source_fps, VIDEO_FPS):
        return frames
    scale = VIDEO_FPS / source_fps
    slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
    repeats = np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5))
    return np.repeat(frames, repeats, axis=0)


def _prepare_image(asset: MediaAsset, scale: float = 1.0) -> Image.Image:
    with Image.open(asset.path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        height, width = resolve_reference_image_size(*image.size, scale=scale)
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return image.copy()


def _prepare_video(asset: MediaAsset, target_frames: int, scale: float = 1.0) -> np.ndarray:
    frames, source_fps = _decode_video(asset.path)
    frames = resample_reference_frames(frames, source_fps)
    frames = frames[:target_frames]
    height, width = resolve_reference_video_size(frames.shape[2], frames.shape[1], scale=scale)
    if frames.shape[1:3] != (height, width):
        frames = np.stack(
            [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames]
        )
    return frames


def _prepare_audio(asset: MediaAsset, target_frames: int) -> torch.Tensor | None:
    clip = load_audio_asset(
        asset,
        AudioProcessingSpec(
            sample_rate=AUDIO_SAMPLE_RATE,
            channels=2,
            missing=MissingMediaPolicy.DROP,
        ),
    )
    if clip is None:
        return None
    maximum_samples = round(target_frames / VIDEO_FPS * AUDIO_SAMPLE_RATE)
    return clip.waveform[:, :maximum_samples].contiguous()


def sample_reference_video_frames(frames: np.ndarray) -> tuple[list[np.ndarray], tuple[float, ...]]:
    stride = VIDEO_FPS / REFERENCE_VIDEO_SAMPLE_FPS
    indices: list[int] = []
    cursor = 0.0
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    timestamps = [index / REFERENCE_VIDEO_SAMPLE_FPS for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % REFERENCE_VIDEO_TEMPORAL_PATCH)
    blocks = tuple(
        (timestamps[index] + timestamps[index + REFERENCE_VIDEO_TEMPORAL_PATCH - 1]) / 2
        for index in range(0, len(timestamps), REFERENCE_VIDEO_TEMPORAL_PATCH)
    )
    return [frames[index] for index in indices], blocks


def trim_reference_frames(frame_count: int) -> int:
    if frame_count < 1:
        raise ValueError(f"H3 reference video must provide at least one prepared frame, got {frame_count}")
    return max(1, (frame_count - 5) // 17) * 17 + 5


def prepare_references(item: Any, *, scale: float = 1.0) -> tuple[H3PreparedReference, ...]:
    assets = reference_assets(item)
    target_frames = int(getattr(item, "frame_count", 0) or getattr(item, "content", np.empty((0,))).shape[0])
    if target_frames <= 0:
        targets = [asset for asset in getattr(item, "h3_media_assets", ()) if asset.role == "target"]
        target_frames = int(targets[0].metadata.get("frame_count", 0)) if len(targets) == 1 else 0
    if assets and target_frames <= 0:
        raise ValueError("H3 reference preparation requires the target frame count")

    prepared: list[H3PreparedReference] = []
    for asset in assets:
        kind = _kind(asset)
        if kind is H3ReferenceKind.IMAGE:
            prepared.append(H3PreparedReference(kind=kind, image=_prepare_image(asset, scale)))
        elif kind is H3ReferenceKind.VIDEO:
            prepared.append(
                H3PreparedReference(
                    kind=kind,
                    frames=_prepare_video(asset, target_frames, scale),
                    waveform=_prepare_audio(asset, target_frames),
                )
            )
        else:
            waveform = _prepare_audio(asset, target_frames)
            if waveform is None:
                raise ValueError(f"H3 audio reference produced no waveform: {asset.path}")
            prepared.append(H3PreparedReference(kind=kind, waveform=waveform))
    if prepared and not any(reference.kind is not H3ReferenceKind.AUDIO for reference in prepared):
        raise ValueError("MiniMax H3 audio references require at least one image or video reference")
    return tuple(prepared)
