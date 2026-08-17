"""Ref2VA media preparation for MiniMax H3's released reference path."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from musubi_tuner.minimax_h3.architecture import AUDIO_SAMPLE_RATE, CANVAS_MULTIPLE, VIDEO_FPS, temporal_shape
from musubi_tuner.minimax_h3.audio import load_audio_asset
from musubi_tuner.minimax_h3.image_training import file_identity
from musubi_tuner.minimax_h3.media import (
    AudioProcessingSpec,
    CropMode,
    MediaAsset,
    MediaModality,
    MissingMediaPolicy,
    PadMode,
)

REFERENCE_FINGERPRINT_KEY = "reference_fingerprint"

REFERENCE_IMAGE_SHORT_EDGE = 2048
REFERENCE_IMAGE_SIZE_MODE = "short_edge"
REFERENCE_IMAGE_SIZE_MODES = ("short_edge", "target_area")
REFERENCE_VIDEO_SHORT_EDGE = 768
REFERENCE_VIDEO_MAX_PIXELS = 768 * 1344
REFERENCE_VIDEO_SAMPLE_FPS = 2
REFERENCE_VIDEO_TEMPORAL_PATCH = 2
REFERENCE_VIDEO_FPS = 0.0
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
    # Rate of ``frames`` in source seconds. The truncation path resamples to the
    # released 24 fps grid; temporal subsampling instead keeps its own rate, and
    # the Qwen presentation needs it to label the frames with real source times.
    sample_fps: float = float(VIDEO_FPS)

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


def reference_fingerprint(assets: Sequence[MediaAsset]) -> str | None:
    references = [asset for asset in assets if asset.role == "reference"]
    if not references:
        return None
    entries: list[dict[str, Any]] = []
    for order, asset in enumerate(references):
        entry = {"order": order, "kind": int(_kind(asset)), **file_identity(asset.path)}
        audio_path = asset.metadata.get("audio_path")
        if audio_path:
            entry["audio"] = file_identity(Path(audio_path))
        entries.append(entry)
    descriptor = {"format": 1, "references": entries}
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _round_to_multiple(value: float) -> int:
    # Explicit half-up: ``round`` is banker's, so a dimension landing exactly on
    # a half-multiple would alternate between the two neighbours by parity.
    return max(CANVAS_MULTIPLE, math.floor(value / CANVAS_MULTIPLE + 0.5) * CANVAS_MULTIPLE)


def _multiple_size(width: float, height: float, max_pixels: int | None = None) -> tuple[int, int]:
    """Return ``(height, width)`` snapped to the canvas multiple.

    Rounding happens first so the result is always canvas-aligned. When
    ``max_pixels`` is given the aligned dimensions are then stepped down, one
    canvas multiple at a time off the longer edge, until the area fits: clamping
    before rounding lets the rounding push the area back over the cap, which for
    extreme aspect ratios overshot it severalfold.

    Both edges are floored at ``CANVAS_MULTIPLE``. A cap below
    ``CANVAS_MULTIPLE**2`` (rejected by the validators) therefore cannot be
    satisfied and the single degenerate ``CANVAS_MULTIPLE x CANVAS_MULTIPLE``
    result is returned instead.
    """
    resolved_height = _round_to_multiple(height)
    resolved_width = _round_to_multiple(width)
    if max_pixels is not None:
        while resolved_height * resolved_width > max_pixels and (
            resolved_height > CANVAS_MULTIPLE or resolved_width > CANVAS_MULTIPLE
        ):
            if resolved_height >= resolved_width and resolved_height > CANVAS_MULTIPLE:
                resolved_height -= CANVAS_MULTIPLE
            else:
                resolved_width -= CANVAS_MULTIPLE
    return resolved_height, resolved_width


def validate_reference_image_short_edge(short_edge: int) -> int:
    if short_edge < CANVAS_MULTIPLE:
        raise ValueError(f"H3 reference image short edge must be at least {CANVAS_MULTIPLE}, got {short_edge}")
    return short_edge


def validate_reference_image_sizing(mode: str, max_pixels: int) -> tuple[str, int]:
    if mode not in REFERENCE_IMAGE_SIZE_MODES:
        raise ValueError(f"unsupported H3 reference image sizing mode: {mode}")
    if max_pixels < 0:
        raise ValueError(f"H3 reference image max pixels must be non-negative, got {max_pixels}")
    if mode == "short_edge" and max_pixels:
        raise ValueError("H3 reference image max pixels applies only to target_area sizing")
    return mode, max_pixels


def resolve_reference_image_size(width: int, height: int, short_edge: int = REFERENCE_IMAGE_SHORT_EDGE) -> tuple[int, int]:
    if width <= 0 or height <= 0 or width > 4 * height or height > 4 * width:
        raise ValueError(f"H3 reference image must have a positive 1:4 to 4:1 aspect ratio, got {width}x{height}")
    validate_reference_image_short_edge(short_edge)
    scale = short_edge / min(width, height)
    return _multiple_size(width * scale, height * scale)


def resolve_reference_image_area_size(width: int, height: int, target_pixels: int) -> tuple[int, int]:
    if width <= 0 or height <= 0 or width > 4 * height or height > 4 * width:
        raise ValueError(f"H3 reference image must have a positive 1:4 to 4:1 aspect ratio, got {width}x{height}")
    if target_pixels < CANVAS_MULTIPLE**2:
        raise ValueError(f"H3 reference image target area must be at least {CANVAS_MULTIPLE**2} pixels")
    scale = math.sqrt(target_pixels / (width * height))
    return _multiple_size(width * scale, height * scale)


def validate_reference_video_sizing(short_edge: int, max_pixels: int) -> tuple[int, int]:
    if short_edge < CANVAS_MULTIPLE:
        raise ValueError(f"H3 reference video short edge must be at least {CANVAS_MULTIPLE}, got {short_edge}")
    if max_pixels < CANVAS_MULTIPLE**2:
        raise ValueError(f"H3 reference video max pixels must be at least {CANVAS_MULTIPLE**2}, got {max_pixels}")
    return short_edge, max_pixels


def validate_reference_video_fps(sample_fps: float) -> float:
    """Accept the opt-in reference subsampling rate; 0 keeps the released truncation."""
    value = float(sample_fps)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"H3 reference video fps must be a finite non-negative rate (0 disables), got {sample_fps}")
    return value


def resolve_reference_video_size(
    width: int,
    height: int,
    short_edge: int = REFERENCE_VIDEO_SHORT_EDGE,
    max_pixels: int = REFERENCE_VIDEO_MAX_PIXELS,
) -> tuple[int, int]:
    if width <= 0 or height <= 0 or width > 4 * height or height > 4 * width:
        raise ValueError(f"H3 reference video must have a positive 1:4 to 4:1 aspect ratio, got {width}x{height}")
    validate_reference_video_sizing(short_edge, max_pixels)
    ratio = width / height
    if ratio >= 1:
        resolved_width, resolved_height = short_edge * ratio, float(short_edge)
    else:
        resolved_width, resolved_height = float(short_edge), short_edge / ratio
    area = resolved_width * resolved_height
    if area > max_pixels:
        scale = math.sqrt(max_pixels / area)
        resolved_width *= scale
        resolved_height *= scale
    return _multiple_size(resolved_width, resolved_height, max_pixels)


def _source_frame_limit(target_frames: int, source_fps: float) -> int:
    if target_frames <= 0 or source_fps <= 0:
        raise ValueError("H3 reference frame limits require positive frame counts and frame rates")
    return max(1, math.ceil((target_frames - 0.5) / (VIDEO_FPS / source_fps)))


def _decode_video(path: Path, target_frames: int | None = None) -> tuple[np.ndarray, float]:
    try:
        import av

        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError(f"H3 reference video has no video stream: {path}")
            stream = container.streams.video[0]
            rate = stream.average_rate or getattr(stream, "guessed_rate", None)
            if rate is None or float(rate) <= 0:
                raise ValueError(f"H3 reference video has no usable frame rate: {path}")
            source_fps = float(rate)
            source_limit = _source_frame_limit(target_frames, source_fps) if target_frames is not None else None
            frames = []
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
                if source_limit is not None and len(frames) >= source_limit:
                    break
    except ValueError:
        raise
    except Exception as error:
        raise RuntimeError(f"cannot decode H3 reference video {path}: {error}") from error
    if not frames:
        raise ValueError(f"H3 reference video produced no frames: {path}")
    return np.stack(frames), source_fps


def resample_reference_frames(frames: np.ndarray, source_fps: float) -> np.ndarray:
    if source_fps <= 0:
        raise ValueError("H3 reference video frame rate must be positive")
    if math.isclose(source_fps, VIDEO_FPS):
        return frames
    scale = VIDEO_FPS / source_fps
    slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
    repeats = np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5))
    return np.repeat(frames, repeats, axis=0)


def subsample_reference_frames(frames: np.ndarray, source_fps: float, sample_fps: float) -> np.ndarray:
    """Take ``sample_fps`` frames per source second across the WHOLE reference clip.

    The grid is deterministic: sample ``k`` is source frame
    ``floor(k * source_fps / sample_fps + 0.5)`` -- round-half-up, no RNG -- and
    sampling stops at the end of the clip. A rate at or above the source rate
    degenerates to every frame exactly once, because a repeated index is dropped
    rather than duplicated.
    """
    if source_fps <= 0 or sample_fps <= 0:
        raise ValueError("H3 reference video subsampling requires positive source and sample frame rates")
    stride = source_fps / sample_fps
    total = int(frames.shape[0])
    indices: list[int] = []
    step = 0
    while True:
        index = math.floor(step * stride + 0.5)
        if index >= total:
            break
        if not indices or index > indices[-1]:
            indices.append(index)
        step += 1
    if not indices:
        raise ValueError("H3 reference video subsampling produced no frames")
    return frames[indices]


def land_reference_frame_count(count: int, target_frames: int) -> int:
    """Snap a subsampled length onto the packer's legal grid within the target budget.

    Legal reference lengths are one frame or ``17n + 5``; the ceiling is the same
    one the truncation path enforces -- the largest legal count that still fits
    the target's frame budget. Below that ceiling the nearest legal count wins,
    ties padding upwards, so a subsampled span is completed rather than cut back
    to a much shorter legal length.
    """
    if count < 1:
        raise ValueError(f"H3 reference video must provide at least one subsampled frame, got {count}")
    budget = trim_reference_frames(target_frames)
    if count >= budget:
        return budget
    lower = trim_reference_frames(count)
    upper = 5 if lower == 1 else lower + 17
    if upper > budget:
        return lower
    return upper if upper - count <= count - lower else lower


def _prepare_image(asset: MediaAsset, short_edge: int, size_mode: str, target_pixels: int) -> Image.Image:
    with Image.open(asset.path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if size_mode == "target_area":
            height, width = resolve_reference_image_area_size(*image.size, target_pixels)
        else:
            height, width = resolve_reference_image_size(*image.size, short_edge)
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        return image.copy()


def _prepare_video(
    asset: MediaAsset,
    target_frames: int,
    short_edge: int,
    max_pixels: int,
    sample_fps: float = REFERENCE_VIDEO_FPS,
) -> np.ndarray:
    if validate_reference_video_fps(sample_fps):
        # Subsampling conditions on the whole clip, so the decode cannot stop at
        # the target's worth of source frames the way the truncation path does.
        frames, source_fps = _decode_video(asset.path)
        frames = subsample_reference_frames(frames, source_fps, sample_fps)
        count = land_reference_frame_count(frames.shape[0], target_frames)
        if frames.shape[0] > count:
            frames = frames[:count]
        elif frames.shape[0] < count:
            frames = np.concatenate((frames, np.repeat(frames[-1:], count - frames.shape[0], axis=0)))
    else:
        frames, source_fps = _decode_video(asset.path, target_frames)
        frames = resample_reference_frames(frames, source_fps)
        frames = frames[:target_frames]
    height, width = resolve_reference_video_size(frames.shape[2], frames.shape[1], short_edge, max_pixels)
    if frames.shape[1:3] != (height, width):
        frames = np.stack(
            [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS)) for frame in frames]
        )
    return frames


def _prepare_audio(asset: MediaAsset, target_frames: int) -> torch.Tensor | None:
    # The canonical grid, not frames / 24 * 32000: the latter overshoots by one audio latent row on
    # frame counts 5, 56, 107, ... and would disagree with temporal_shape() everywhere else.
    samples = temporal_shape(target_frames, align=True).audio_samples
    clip = load_audio_asset(
        asset,
        AudioProcessingSpec(
            sample_rate=AUDIO_SAMPLE_RATE,
            channels=2,
            clip_duration_seconds=samples / AUDIO_SAMPLE_RATE,
            crop_mode=CropMode.BEGINNING,
            # A short reference track is zero-padded to the canonical span, matching the target path.
            # Reference rows carry no valid mask in the packed layout, so the padding is invisible to
            # the model: keep reference audio at least as long as the reference video span.
            pad_mode=PadMode.ZERO,
            missing=MissingMediaPolicy.DROP,
        ),
    )
    if clip is None:
        return None
    return clip.waveform.contiguous()


def _reference_audio_asset(asset: MediaAsset) -> MediaAsset:
    audio_path = asset.metadata.get("audio_path")
    if not audio_path:
        return asset
    return MediaAsset(
        Path(audio_path),
        MediaModality.AUDIO,
        asset.role,
        start_seconds=asset.start_seconds,
        duration_seconds=asset.duration_seconds,
    )


def sample_reference_video_frames(
    frames: np.ndarray,
    frames_fps: float = float(VIDEO_FPS),
) -> tuple[list[np.ndarray], tuple[float, ...]]:
    """Present the prepared frames to Qwen at 2 fps, labelled with source time.

    ``frames_fps`` is the rate the prepared frames already carry: the released
    24 fps grid, or the opt-in subsampling rate. Timestamps come from the chosen
    source frame rather than its position, so a clip already below 2 fps keeps
    every frame and is still labelled with the seconds it actually spans.
    """
    if frames_fps <= 0:
        raise ValueError("H3 reference video presentation requires a positive frame rate")
    stride = frames_fps / REFERENCE_VIDEO_SAMPLE_FPS
    indices: list[int] = []
    cursor = 0.0
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    timestamps = [index / frames_fps for index in indices]
    timestamps += [timestamps[-1]] * (-len(timestamps) % REFERENCE_VIDEO_TEMPORAL_PATCH)
    blocks = tuple(
        (timestamps[index] + timestamps[index + REFERENCE_VIDEO_TEMPORAL_PATCH - 1]) / 2
        for index in range(0, len(timestamps), REFERENCE_VIDEO_TEMPORAL_PATCH)
    )
    return [frames[index] for index in indices], blocks


def trim_reference_frames(frame_count: int) -> int:
    if frame_count < 1:
        raise ValueError(f"H3 reference video must provide at least one prepared frame, got {frame_count}")
    if frame_count < 5:
        return 1
    return (frame_count - 5) // 17 * 17 + 5


def prepare_references(
    item: Any,
    image_short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
    image_size_mode: str = REFERENCE_IMAGE_SIZE_MODE,
    image_max_pixels: int = 0,
    video_short_edge: int = REFERENCE_VIDEO_SHORT_EDGE,
    video_max_pixels: int = REFERENCE_VIDEO_MAX_PIXELS,
    video_sample_fps: float = REFERENCE_VIDEO_FPS,
) -> tuple[H3PreparedReference, ...]:
    assets = reference_assets(item)
    validate_reference_image_short_edge(image_short_edge)
    validate_reference_image_sizing(image_size_mode, image_max_pixels)
    validate_reference_video_sizing(video_short_edge, video_max_pixels)
    video_sample_fps = validate_reference_video_fps(video_sample_fps)
    target_size = getattr(item, "bucket_size", None) or getattr(item, "original_size", None)
    if image_size_mode == "target_area":
        if target_size is None or len(target_size) < 2:
            raise ValueError("H3 target-area reference sizing requires the target width and height")
        target_pixels = int(target_size[0]) * int(target_size[1])
        if image_max_pixels:
            target_pixels = min(target_pixels, image_max_pixels)
    else:
        target_pixels = 0
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
            prepared.append(
                H3PreparedReference(kind=kind, image=_prepare_image(asset, image_short_edge, image_size_mode, target_pixels))
            )
        elif kind is H3ReferenceKind.VIDEO:
            include_audio = bool(asset.metadata.get("include_audio", True))
            frames = _prepare_video(asset, target_frames, video_short_edge, video_max_pixels, video_sample_fps)
            # The H3 video VAE accepts only 17n+5 frames (or a single image).
            # Trim once at the shared preparation boundary so Qwen's visual
            # presentation, the DiT latent rows, and any paired soundtrack all
            # describe the same temporal span: the soundtrack is cropped or
            # zero-padded to temporal_shape(frames).audio_samples below.
            frames = frames[: trim_reference_frames(frames.shape[0])]
            prepared.append(
                H3PreparedReference(
                    kind=kind,
                    frames=frames,
                    # A subsampled reference spans the whole clip while its paired
                    # soundtrack still covers the clip's opening span: audio cannot
                    # be subsampled without pitch artefacts, so the crop rule below
                    # is the same one the truncation path uses.
                    waveform=_prepare_audio(_reference_audio_asset(asset), frames.shape[0]) if include_audio else None,
                    sample_fps=video_sample_fps or float(VIDEO_FPS),
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


def reference_modality_variant(references: tuple[H3PreparedReference, ...], modality: str) -> tuple[H3PreparedReference, ...]:
    if modality == "av":
        return references
    selected: list[H3PreparedReference] = []
    for reference in references:
        if reference.kind is H3ReferenceKind.IMAGE:
            selected.append(reference)
        elif modality == "video":
            if reference.kind is H3ReferenceKind.VIDEO:
                selected.append(
                    H3PreparedReference(
                        kind=reference.kind,
                        frames=reference.frames,
                        block_timestamps=reference.block_timestamps,
                        sample_fps=reference.sample_fps,
                    )
                )
        elif reference.kind is H3ReferenceKind.AUDIO:
            selected.append(reference)
        elif reference.waveform is not None:
            selected.append(H3PreparedReference(kind=H3ReferenceKind.AUDIO, waveform=reference.waveform))
    if not any(reference.kind is not H3ReferenceKind.AUDIO for reference in selected):
        raise ValueError(f"H3 stochastic {modality}-reference variant requires at least one visual reference")
    return tuple(selected)
