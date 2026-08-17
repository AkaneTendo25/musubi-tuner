"""Procedural conditioning masks for MiniMax H3 inpainting and outpainting.

A conditioning mask says which latents the model must generate and which it is
given. It is not a loss mask: the observed part is presented as clean context at
the released conditioning noise level, exactly as reference and keyframe media
already are, and only the generated part is scored.

The three procedural modes are drawn per step rather than read from the dataset.
Inpainting is trained on synthetic occlusions, so authoring them offline would
fix the occlusion distribution to whatever was cached and would force a re-cache
to change it.

A dataset may nevertheless author real masks -- a segmentation, a matte, a hand
drawn region -- which ``--h3_mask_mode dataset`` reads instead of drawing one.
Those are loaded at train time by :func:`load_conditioning_mask` and reduced by
:func:`conditioning_mask_to_latent`, never cached with the latents: the mask is
applied to the packed rows, so it changes nothing the VAE encoded.

Every procedural sampler returns ``True`` where the model must generate. The
dataset-mask helpers instead return ``True`` where the model is *given* the
content, which is the polarity the authored image carries (white = observed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

VideoMaskMode = Literal["box", "border", "segment"]
AudioMaskMode = Literal["segment"]

# Authored masks are single-channel images: any pixel above this is observed
# context, everything else is a region the model must generate.
CONDITIONING_MASK_THRESHOLD = 127

# Batch key the collator uses for the reduced, latent-resolution dataset mask.
# It ends in ``_mask`` so the batch assembler keeps the name verbatim.
CONDITIONING_MASK_BATCH_KEY = "h3_conditioning_mask"


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


def load_conditioning_mask(path: str | Path) -> torch.Tensor:
    """Read an authored mask image as a ``[height, width]`` bool tensor, True where observed.

    Single channel or colour, alpha ignored: the image is reduced to luma and
    thresholded, so a white region is context the model is given and a black one
    is the region it must generate.
    """
    from PIL import Image  # imported lazily so the samplers stay dependency-free

    with Image.open(str(path)) as image:
        luma = image.convert("L")
        values = torch.frombuffer(bytearray(luma.tobytes()), dtype=torch.uint8)
        return (values.reshape(luma.size[1], luma.size[0]) > CONDITIONING_MASK_THRESHOLD).clone()


def _resize_nearest(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Nearest-neighbour resize of a bool mask, without antialiasing.

    Binary masks have no intermediate values to interpolate: a filtered resize
    would invent grey pixels whose thresholding moves the region boundary.
    """
    if mask.shape == (height, width):
        return mask
    rows = torch.linspace(0, mask.shape[0] - 1, height).round().to(torch.long)
    columns = torch.linspace(0, mask.shape[1] - 1, width).round().to(torch.long)
    return mask.index_select(0, rows).index_select(1, columns)


def conditioning_mask_to_latent(
    observed: torch.Tensor,
    *,
    bucket_size: tuple[int, int],
    latent_frames: int,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    """Reduce an authored pixel mask to a ``[frames, height, width]`` latent mask.

    The mask is first resized to the item's bucket resolution -- authored against
    the target content, so aspect ratio is not preserved -- and then reduced to
    latent cells. A cell counts as observed only when every pixel inside it is,
    which keeps the boundary of an authored region on the generated side exactly
    as :func:`video_mask_to_rows` keeps a patch boundary there.

    The mask is static, so the same plane is repeated across every latent frame.
    """
    if observed.ndim != 2:
        raise ValueError(f"H3 conditioning mask must be [height, width], got {tuple(observed.shape)}")
    if latent_frames <= 0 or latent_height <= 0 or latent_width <= 0:
        raise ValueError("H3 conditioning mask geometry must be positive")
    width, height = bucket_size
    if height <= 0 or width <= 0:
        raise ValueError("H3 conditioning mask bucket size must be positive")
    plane = _resize_nearest(observed.to(torch.bool), height, width)
    if height % latent_height == 0 and width % latent_width == 0:
        plane = plane.reshape(latent_height, height // latent_height, latent_width, width // latent_width)
        plane = plane.all(dim=3).all(dim=1)
    else:
        # A bucket that is not an exact multiple of the latent grid has no whole
        # block to reduce; fall back to sampling the mask at latent resolution.
        plane = _resize_nearest(plane, latent_height, latent_width)
    return plane.unsqueeze(0).expand(latent_frames, latent_height, latent_width).contiguous()


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
