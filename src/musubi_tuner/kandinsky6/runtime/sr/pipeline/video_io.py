"""Video and latent IO for the SR inference CLI.

Reads pixel video, resamples it to the model's training fps, applies VAE-friendly
frame alignment, loads ``.pt`` LR latents, and saves stitched SR output as mp4
(default) or lossless mkv.

The fps handling mirrors the dataset encoding pipeline
(``musubi_tuner.kandinsky6.runtime.sr/datasets/data_encoding/video_download.py``): the SR model was
trained on clips at a fixed ``TARGET_FPS``, so any source is resampled to that
rate before inference. Above-target sources are thinned by fixed-stride index
selection; near-target sources pass through; below-target sources are kept at
their native rate (the motion-compensated ffmpeg upsample used at training time
is intentionally not pulled in here) with a warning.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import torch
from loguru import logger

from musubi_tuner.kandinsky6.runtime.sr.constants import LATENT_NDIM, MAX_NUM_FRAMES, RESAMPLE_FPS_TOLERANCE, TARGET_FPS

try:
    import av
except (ImportError, OSError):
    av = None


def _require_av() -> ModuleType:
    """Return PyAV; only file-based input needs it, so tensor-only hosts import this module without it."""
    if av is None:
        raise RuntimeError("SR file-based video input requires PyAV and FFmpeg. Install with `pip install av`.")
    return av


def align_to_vae_stride(t: int) -> int:
    """Round ``t`` to the nearest valid pixel-frame count ``1 + 8·k`` (ties up).

    SR data uses ``1 + 8·k`` frames so the temporal VAE stride round-trips
    cleanly. Rounding to nearest (not truncation) keeps a requested duration
    close to the target.

    Args:
        t: Raw frame count.

    Returns:
        Nearest valid frame count ``>= 1``.
    """
    if t <= 1:
        return 1
    k_floor = (t - 1) // 8
    t_floor = 1 + 8 * k_floor
    t_ceil = 1 + 8 * (k_floor + 1)
    return t_ceil if (t - t_floor) >= (t_ceil - t) else t_floor


def select_frame_indices(total_frames: int, src_fps: float, target_fps: float) -> list[int]:
    """Fixed-stride frame indices that downsample ``src_fps`` to ``target_fps``.

    Mirrors ``data_encoding.frame_utils.select_frame_indices``: with
    ``step = src_fps / target_fps >= 1``, ``round(i * step)`` is strictly
    non-decreasing, so the selection never duplicates a source frame.

    Args:
        total_frames: Number of frames in the source.
        src_fps: Source frame rate (must be ``>= target_fps``).
        target_fps: Desired frame rate.

    Returns:
        Sorted list of integer source-frame indices.
    """
    step = src_fps / target_fps
    indices = [round(i * step) for i in range(int(total_frames / step))]
    return [i for i in indices if i < total_frames]


def resample_to_target_fps(
    video: torch.Tensor,
    src_fps: float,
    target_fps: int = TARGET_FPS,
) -> tuple[torch.Tensor, int]:
    """Resample a decoded video toward ``target_fps`` (downsample / no-op / keep).

    Three tiers, mirroring the dataset encoding pipeline:

    - ``|src_fps - target_fps| < RESAMPLE_FPS_TOLERANCE``: pass through unchanged.
    - ``src_fps > target_fps``: fixed-stride downsample via
      :func:`select_frame_indices`; the result is at ``target_fps``.
    - ``src_fps < target_fps``: kept at the native rate with a warning (no ffmpeg
      ``minterpolate`` upsample); the clip stays mildly out of distribution.

    Args:
        video: ``[T, C, H, W]`` decoded source video.
        src_fps: Source frame rate.
        target_fps: Training frame rate to resample toward.

    Returns:
        ``(resampled_video, effective_fps)`` where ``effective_fps`` is the rate
        the returned frames play at (``target_fps`` when downsampled, otherwise
        ``round(src_fps)``) and is the correct rate to save the SR result at.
    """
    if abs(src_fps - target_fps) < RESAMPLE_FPS_TOLERANCE:
        return video, round(src_fps)
    if src_fps > target_fps:
        indices = select_frame_indices(video.shape[0], src_fps, target_fps)
        logger.warning(
            "Source fps {:.2f} > target {}fps: downsampling {} frames -> {} (fixed-stride); "
            "source temporal detail beyond {}fps is discarded.",
            src_fps,
            target_fps,
            video.shape[0],
            len(indices),
            target_fps,
        )
        return video[indices], target_fps
    logger.warning(
        "Source fps {:.2f} < target {}fps: keeping native frames (no minterpolate upsample); output is mildly out of distribution.",
        src_fps,
        target_fps,
    )
    return video, round(src_fps)


def clip_to_aligned_frames(video: torch.Tensor, max_num_frames: int = MAX_NUM_FRAMES) -> torch.Tensor:
    """Take the first ``max_num_frames`` and floor-align to ``1 + 8k`` frames.

    Args:
        video: ``[T, C, H, W]`` video.
        max_num_frames: Hard cap applied before alignment (``<= 0`` disables it).

    Returns:
        ``[T', C, H, W]`` with ``T' == 1 + 8k`` and ``T' <= max_num_frames``.

    Raises:
        ValueError: If the video has no frames.
    """
    if 0 < max_num_frames < video.shape[0]:
        logger.warning(
            "Clip has {} frames; only the first {} ({:.1f} s at {} fps, the model's budget) are super-resolved. "
            "Split longer sources into 5 s pieces.",
            video.shape[0],
            max_num_frames,
            max_num_frames / TARGET_FPS,
            TARGET_FPS,
        )
        video = video[:max_num_frames]
    aligned = 1 + 8 * ((video.shape[0] - 1) // 8) if video.shape[0] > 0 else 0
    if aligned == 0:
        msg = "Video has no readable frames."
        raise ValueError(msg)
    return video[:aligned]


def read_video_tchw_uint8(path: Path) -> tuple[torch.Tensor, float]:
    """Decode an mp4/mkv to ``([T, C, H, W] uint8, src_fps)`` (no resample/cap).

    Resampling to the training fps and frame-count alignment are applied by the
    caller via :func:`resample_to_target_fps` and :func:`clip_to_aligned_frames`.

    Args:
        path: Source video file.

    Returns:
        ``([T, C, H, W] uint8 video, native_fps)``.

    Raises:
        ValueError: If the file has no readable frames or reports no fps.
    """
    with _require_av().open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        src_fps = float(stream.average_rate or stream.base_rate or 0.0)
        frames = [torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1) for frame in container.decode(stream)]

    if not frames:
        msg = f"Video {path.name} has no readable frames."
        raise ValueError(msg)
    if src_fps <= 0:
        msg = f"Video {path.name} reports no usable fps (got {src_fps})."
        raise ValueError(msg)
    return torch.stack(frames).contiguous(), src_fps


def load_lr_latent(path: Path) -> torch.Tensor:
    """Load a raw, unscaled LR latent ``.pt`` as ``[T, C, H, W]`` float32.

    The tensor must match what ``encode_lq_video_to_lr_latent`` produces for
    the model's VAE (no scaling factor applied) — the LU multiplies internally.

    Args:
        path: Path to a ``.pt`` file holding a 4D latent tensor.

    Returns:
        ``[T, C, H, W]`` float32 latent on CPU.

    Raises:
        ValueError: If the loaded object is not a rank-4 float tensor.
    """
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, torch.Tensor) or obj.ndim != LATENT_NDIM:
        shape = getattr(obj, "shape", type(obj).__name__)
        msg = f"Expected a rank-4 [T, C, H, W] latent tensor in {path}, got {shape}."
        raise ValueError(msg)
    logger.info("Loaded LR latent {} from {}", tuple(obj.shape), path)
    return obj.float()
