"""Procedural conditioning masks for MiniMax H3 inpainting and outpainting.

A conditioning mask says which latents the model must generate and which it is
given. It is not a loss mask: the observed part is presented as clean context at
the released conditioning noise level, exactly as reference and keyframe media
already are, and only the generated part is scored.

Masks are drawn per step rather than read from the dataset. Inpainting is
trained on synthetic occlusions, so authoring them offline would fix the
occlusion distribution to whatever was cached and would force a re-cache to
change it.

Every function returns ``True`` where the model must generate.
"""

from __future__ import annotations

from typing import Literal

import torch

VideoMaskMode = Literal["box", "border", "segment"]
AudioMaskMode = Literal["segment"]


def _span(generator: torch.Generator, length: int, minimum: float, maximum: float) -> tuple[int, int]:
    """Draw a sub-span of ``length`` covering a fraction in ``[minimum, maximum]``."""
    if length <= 0:
        return 0, 0
    low = max(1, int(round(length * minimum)))
    high = max(low, min(length, int(round(length * maximum))))
    size = int(torch.randint(low, high + 1, (1,), generator=generator).item())
    start = int(torch.randint(0, length - size + 1, (1,), generator=generator).item())
    return start, start + size


def sample_video_mask(
    *,
    mode: VideoMaskMode,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    generator: torch.Generator,
    minimum: float = 0.25,
    maximum: float = 0.75,
) -> torch.Tensor:
    """Draw a ``[frames, height, width]`` latent mask that is True where generated."""
    if latent_frames <= 0 or latent_height <= 0 or latent_width <= 0:
        raise ValueError("H3 mask geometry must be positive")
    if not 0.0 < minimum <= maximum <= 1.0:
        raise ValueError("H3 mask fractions must satisfy 0 < minimum <= maximum <= 1")

    mask = torch.zeros((latent_frames, latent_height, latent_width), dtype=torch.bool)
    if mode == "box":
        top, bottom = _span(generator, latent_height, minimum, maximum)
        left, right = _span(generator, latent_width, minimum, maximum)
        mask[:, top:bottom, left:right] = True
    elif mode == "border":
        # Outpainting keeps an interior crop and generates everything around it,
        # so the retained region is the complement of the box case.
        top, bottom = _span(generator, latent_height, minimum, maximum)
        left, right = _span(generator, latent_width, minimum, maximum)
        mask[:] = True
        mask[:, top:bottom, left:right] = False
    elif mode == "segment":
        first, last = _span(generator, latent_frames, minimum, maximum)
        mask[first:last] = True
    else:
        raise ValueError(f"unsupported H3 video mask mode: {mode}")
    return mask


def sample_audio_mask(
    *,
    num_audio_latents: int,
    generator: torch.Generator,
    minimum: float = 0.25,
    maximum: float = 0.75,
) -> torch.Tensor:
    """Draw a ``[latents]`` audio mask that is True where generated."""
    if num_audio_latents <= 0:
        raise ValueError("H3 audio mask needs at least one latent")
    if not 0.0 < minimum <= maximum <= 1.0:
        raise ValueError("H3 mask fractions must satisfy 0 < minimum <= maximum <= 1")
    mask = torch.zeros((num_audio_latents,), dtype=torch.bool)
    first, last = _span(generator, num_audio_latents, minimum, maximum)
    mask[first:last] = True
    return mask


def video_mask_to_rows(mask: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """Reduce a latent mask to one flag per packed video row.

    Video rows are frame-major and each covers a ``patch_h x patch_w`` block, so
    a patch is generated when any latent inside it is. Taking ``any`` rather than
    ``all`` keeps the boundary of a region inside the generated set, which is
    where an inpainting seam would otherwise appear.
    """
    if mask.ndim != 3:
        raise ValueError(f"H3 video mask must be [frames, height, width], got {tuple(mask.shape)}")
    _, patch_h, patch_w = patch_size
    frames, height, width = mask.shape
    if height % patch_h or width % patch_w:
        raise ValueError("H3 video mask dimensions must be divisible by the spatial patch")
    blocks = mask.reshape(frames, height // patch_h, patch_h, width // patch_w, patch_w)
    return blocks.any(dim=4).any(dim=2).reshape(-1)


def audio_mask_to_rows(mask: torch.Tensor, *, channels: int) -> torch.Tensor:
    """Repeat an audio latent mask across channel-major packed rows."""
    if mask.ndim != 1:
        raise ValueError(f"H3 audio mask must be one-dimensional, got {tuple(mask.shape)}")
    return mask.repeat(channels)


def rows_to_latent_video_mask(
    rows: torch.Tensor, *, latent_frames: int, latent_height: int, latent_width: int, patch_size
) -> torch.Tensor:
    """Expand a per-row flag back to latent resolution, for loss masking."""
    _, patch_h, patch_w = patch_size
    grid = rows.reshape(latent_frames, latent_height // patch_h, latent_width // patch_w)
    return grid.repeat_interleave(patch_h, dim=1).repeat_interleave(patch_w, dim=2)
