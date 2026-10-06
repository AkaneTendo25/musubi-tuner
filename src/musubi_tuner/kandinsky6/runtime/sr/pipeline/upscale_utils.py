"""Fractional-scale resolution and pixel pre-upscaling for tiled SR.

``--resolution-scale 2.25`` cannot drive the tile grid directly — the grid
only works with integer scales that divide the trained base resolutions
(``tile * scale == base``). Fractional totals are therefore decomposed into a
pixel-space pre-upscale times an integer tiling scale (``2.25 = 1.125 x 2``)
and the regular x2 machinery runs unchanged on the slightly enlarged source.
"""

from __future__ import annotations

import torch
from torch.nn import functional


def resolve_scale_request(scale: float) -> tuple[int, float]:
    """Map a requested ``--resolution-scale`` to ``(tiling_scale, pre_upscale)``.

    Args:
        scale: The requested total upscale — ``2``, ``4`` or ``2.25``.

    Returns:
        ``(tiling_scale, pre_upscale)``; ``pre_upscale`` is ``1.0`` for the
        integer scales.
    """
    requested = float(scale)
    if requested == 2.25:  # noqa: PLR2004 — the one supported fractional total
        return 2, 1.125
    if requested in (2.0, 4.0):
        return int(requested), 1.0
    raise ValueError("SR supports total scales 2, 4, and 2.25")


def pre_upscale_video(video: torch.Tensor, factor: float, spatial_multiple: int) -> torch.Tensor:
    """Bilinear-upscale a ``[T, C, H, W]`` uint8 video by ``factor`` in pixel space.

    Target dims are rounded to the nearest multiple of ``spatial_multiple``
    (the VAE spatial factor) so the whole-video encode and the latent tile
    grid stay integer-aligned (``latent_tile_grid_from_pixel_grid`` rejects
    unaligned grids). For sources whose scaled dims already land on the
    factor (512x768 x1.125 -> 576x864) the rounding is a no-op and the total
    scale is exact.

    Args:
        video: ``[T, C, H, W]`` uint8 source video.
        factor: Pixel upscale factor (> 1).
        spatial_multiple: VAE spatial factor to align the target dims to.

    Returns:
        ``[T, C, H', W']`` uint8 video with ``H' ~= H * factor`` aligned.
    """
    if video.ndim != 4:  # noqa: PLR2004
        raise ValueError(f"video must have rank 4 [T,C,H,W], got {tuple(video.shape)}")
    if factor <= 0 or spatial_multiple <= 0:
        raise ValueError("factor and spatial_multiple must be positive")
    height, width = video.shape[-2:]
    target_h = max(spatial_multiple, round(height * factor / spatial_multiple) * spatial_multiple)
    target_w = max(spatial_multiple, round(width * factor / spatial_multiple) * spatial_multiple)
    resized = functional.interpolate(
        video.float(),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized.round_().clamp_(0, 255).to(torch.uint8)
