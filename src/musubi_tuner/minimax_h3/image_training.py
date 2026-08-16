from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from safetensors import safe_open

from musubi_tuner.minimax_h3.architecture import is_valid_frame_count

H3_IMAGE_MODES = ("none", "first", "first_last")
H3_TEXT_VISUAL_MAX_PIXELS = 0


def add_image_training_arguments(parser: argparse.ArgumentParser, *, text_visual: bool = False) -> None:
    parser.add_argument(
        "--h3_image_mode",
        choices=H3_IMAGE_MODES,
        default="none",
        help="condition FL2VA image targets from one first image or separate first/last images",
    )
    parser.add_argument(
        "--h3_image_frame_count",
        type=int,
        help="pixel-frame count for conditioned-image targets; overrides dataset h3_image_frame_count",
    )
    if text_visual:
        parser.add_argument(
            "--h3_text_visual_max_pixels",
            type=int,
            default=H3_TEXT_VISUAL_MAX_PIXELS,
            help="maximum pixels per control image presented to Qwen3-VL; disabled by default, 0 disables the cap",
        )


def validate_image_mode(mode: str, frame_count: int | None) -> None:
    if mode not in H3_IMAGE_MODES:
        raise ValueError(f"MiniMax H3 image mode must be one of {', '.join(H3_IMAGE_MODES)}")
    if mode == "none":
        return
    if frame_count is None:
        raise ValueError("MiniMax H3 conditioned-image training requires an image frame count")
    if not is_valid_frame_count(frame_count):
        raise ValueError("MiniMax H3 image frame count must satisfy frame_count % 17 == 5")


def condition_paths(mode: str, paths: Sequence[Path]) -> tuple[Path, ...]:
    expected = {"first": 1, "first_last": 2}.get(mode, 0)
    if len(paths) != expected:
        raise ValueError(f"MiniMax H3 image mode {mode!r} requires exactly {expected} control image(s), got {len(paths)}")
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"MiniMax H3 control image(s) not found: {', '.join(missing)}")
    return tuple(paths)


def condition_images(mode: str, paths: Sequence[Path]) -> tuple[Image.Image, ...]:
    controls = condition_paths(mode, paths)
    images = []
    for path in controls:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    images = tuple(images)
    return (images[0], images[0]) if mode == "first" else images


def resample_image_targets(content: np.ndarray | list[np.ndarray], frame_count: int) -> np.ndarray:
    frames = [content] if isinstance(content, np.ndarray) and content.ndim == 3 else list(content)
    if not frames:
        raise ValueError("MiniMax H3 image target sequence is empty")
    if any(not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[-1] != 3 for frame in frames):
        raise ValueError("MiniMax H3 image targets must be RGB arrays")
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("MiniMax H3 multi-target images must share one bucket geometry")
    indices = np.rint(np.linspace(0, len(frames) - 1, frame_count)).astype(np.int64)
    return np.stack([frames[int(index)] for index in indices])


def read_text_visual(path: Path, max_pixels: int = H3_TEXT_VISUAL_MAX_PIXELS) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB").copy()
    if max_pixels > 0 and image.width * image.height > max_pixels:
        scale = (max_pixels / (image.width * image.height)) ** 0.5
        width = max(32, int(image.width * scale) // 32 * 32)
        height = max(32, int(image.height * scale) // 32 * 32)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sample_fingerprint(
    *,
    targets: Sequence[Path],
    controls: Sequence[Path],
    mode: str,
    frame_count: int,
    original_size: Sequence[int],
    bucket_size: Sequence[int],
) -> str:
    descriptor = {
        "format": 1,
        "mode": mode,
        "frame_count": int(frame_count),
        "original_size": [int(value) for value in original_size],
        "bucket_size": [int(value) for value in bucket_size[:2]],
        "targets": [file_identity(path) for path in targets],
        "controls": [file_identity(path) for path in controls],
    }
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_matches_fingerprint(path: str | Path, fingerprint: str, key: str = "sample_fingerprint") -> bool:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            return (handle.metadata() or {}).get(key) == fingerprint
    except (OSError, RuntimeError, ValueError):
        return False
