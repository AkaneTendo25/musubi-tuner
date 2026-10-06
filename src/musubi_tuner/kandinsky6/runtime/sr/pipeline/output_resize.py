"""Final-output sizing: bring the SR result down to an exact delivery resolution.

Tiled SR always produces ``source * scale`` pixels — a size dictated by the
tiling invariant (``tile * scale == base``), not by the product. A 432x768
source at x2 yields 864x1536 where the deliverable is 720x1280; a 480x864
source at x2.25 yields 1088x1952 where it is 1080x1920. This module maps that
raw result onto one of the shipped tiers (see ``TARGET_RESOLUTIONS``).

Resize only — no cropping. Where the SR aspect differs slightly from the tier
(a 1.8 source against 16:9, ~1%), the scale is very mildly anisotropic instead
of throwing pixels away; the mismatch is logged so it stays visible.

Downscaling is antialiased and runs in frame chunks — a 2K result is a couple
of GB as uint8 and four times that as float, so a single ``interpolate`` over
the whole clip would spike memory for no reason.
"""

from __future__ import annotations

import torch
from loguru import logger
from torch.nn import functional as f  # noqa: N812

from musubi_tuner.kandinsky6.runtime.sr.constants import RESIZE_FRAME_CHUNK, TARGET_RESOLUTIONS
from musubi_tuner.kandinsky6.runtime.sr.pipeline.config import TargetResizeMode

# Log a warning when the SR and target aspect ratios differ by more than this
# (relative) — a mild anisotropic squeeze is fine, a big one means the wrong
# tier for the route.
ASPECT_MISMATCH_TOLERANCE = 0.02


def fit_within(sr_hw: tuple[int, int], bucket_hw: tuple[int, int]) -> tuple[int, int] | None:
    """Isotropic-fit target: shrink ``sr_hw`` to fit inside ``bucket_hw``.

    One scale factor for both axes, so the aspect is preserved exactly: the
    binding side lands on the bucket, the other comes out at (or under) it,
    snapped to even for the video codec. A result already inside the bucket
    is left alone — fitting never upscales.

    Args:
        sr_hw: The SR result's ``(H, W)``.
        bucket_hw: The resolved tier bucket ``(H, W)``.

    Returns:
        The fitted ``(H, W)``, or ``None`` when no downscale is needed.
    """
    scale = min(bucket_hw[0] / sr_hw[0], bucket_hw[1] / sr_hw[1])
    if scale >= 1.0:
        logger.warning(
            "SR result {}x{} already fits inside the {}x{} bucket — nothing to downscale, keeping it as is.",
            sr_hw[1],
            sr_hw[0],
            bucket_hw[1],
            bucket_hw[0],
        )
        return None
    height, width = (min(bucket, round(side * scale / 2) * 2) for bucket, side in zip(bucket_hw, sr_hw, strict=True))
    return height, width


def resolve_target_hw(
    spec: str | None,
    sr_hw: tuple[int, int],
    mode: TargetResizeMode = "fit",
) -> tuple[int, int] | None:
    """Resolve a ``--target-resolution`` spec into the final ``(H, W)``.

    A tier name (``hd`` / ``fullhd`` / ``2k``) resolves against the SR result's
    aspect ratio, picking the closest entry of that tier — the same
    closest-aspect rule the tiling uses to choose a base resolution. An
    explicit ``WxH`` is taken as the bucket verbatim.

    ``mode`` then decides how the bucket is honoured: ``"fit"`` (the default)
    preserves the aspect exactly — an isotropic downscale into the bucket, so
    an off-tier source is never squeezed; ``"exact"`` returns the bucket
    itself, trading a small anisotropy for exact delivery dimensions.

    Args:
        spec: Tier name, ``"WxH"``, or ``None`` / ``"none"`` to keep the raw
            SR result.
        sr_hw: The SR result's ``(H, W)``, used to pick a tier entry.
        mode: ``"fit"`` (aspect-preserving) or ``"exact"``.

    Returns:
        The target ``(H, W)``, or ``None`` when no resizing is requested
        (or, in ``fit`` mode, needed).

    Raises:
        ValueError: On an unknown tier name or a malformed ``WxH``.
    """
    if spec is None or spec.lower() == "none":
        return None
    key = spec.lower()
    if key in TARGET_RESOLUTIONS:
        source_ratio = sr_hw[1] / sr_hw[0]
        bucket = min(TARGET_RESOLUTIONS[key], key=lambda hw: abs(hw[1] / hw[0] - source_ratio))
    else:
        parts = key.split("x")
        expected_parts = 2
        if len(parts) != expected_parts or not all(part.strip().isdigit() for part in parts):
            msg = (
                f"--target-resolution must be a tier {sorted(TARGET_RESOLUTIONS)} or WxH (e.g. 1280x720), got {spec!r}"
            )
            raise ValueError(msg)
        width, height = (int(part) for part in parts)
        bucket = (height, width)
    if mode == "exact":
        return bucket
    return fit_within(sr_hw, bucket)


def resize_to_target(video: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Antialiased-downscale the SR result to exactly ``target_hw``. No cropping.

    Args:
        video: ``[C, T, H, W]`` uint8 SR result.
        target_hw: Exact output ``(H, W)``.

    Returns:
        ``[C, T, target_h, target_w]`` uint8.

    Raises:
        ValueError: If the target exceeds the SR result on either axis —
            that means the tier does not belong to this route, and silently
            upscaling would hide the mistake.
    """
    _c, _t, height, width = video.shape
    target_h, target_w = target_hw
    if (height, width) == (target_h, target_w):
        return video
    if target_h > height or target_w > width:
        msg = (
            f"target {target_w}x{target_h} exceeds the SR result {width}x{height} — that would "
            f"upscale. Pick a lower tier or a higher --resolution-scale for this source."
        )
        raise ValueError(msg)

    source_ratio, target_ratio = width / height, target_w / target_h
    if abs(source_ratio - target_ratio) / target_ratio > ASPECT_MISMATCH_TOLERANCE:
        logger.warning(
            "Aspect mismatch: SR {}x{} is {:.3f}, target {}x{} is {:.3f} — resizing anisotropically "
            "(no crop), the picture will be squeezed by {:.1f}%.",
            width,
            height,
            source_ratio,
            target_w,
            target_h,
            target_ratio,
            abs(source_ratio / target_ratio - 1) * 100,
        )

    channels, frames = video.shape[0], video.shape[1]
    out = torch.empty((channels, frames, target_h, target_w), dtype=torch.uint8)
    for start in range(0, frames, RESIZE_FRAME_CHUNK):
        chunk = video[:, start : start + RESIZE_FRAME_CHUNK].permute(1, 0, 2, 3).float()
        resized = f.interpolate(chunk, size=(target_h, target_w), mode="bilinear", antialias=True, align_corners=False)
        out[:, start : start + RESIZE_FRAME_CHUNK] = resized.clamp(0, 255).round().to(torch.uint8).permute(1, 0, 2, 3)

    logger.info(
        "Final output: {}x{} -> {}x{} (downscale x{:.2f} / x{:.2f})",
        width,
        height,
        target_w,
        target_h,
        width / target_w,
        height / target_h,
    )
    return out
