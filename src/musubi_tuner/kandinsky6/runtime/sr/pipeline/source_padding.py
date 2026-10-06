"""Spatial padding of the latent-upscaler path's source.

The LU path encodes the whole clip with the KVAE and cuts the LR latent into
tiles of ``base // scale`` pixels, which puts two constraints on the source:

* ``H`` and ``W`` must be multiples of the VAE spatial stride (16): the
  encoder halves the dims at every stage and fails inside ``PixelUnshuffle``
  otherwise;
* the source must be at least one tile on each axis: a smaller source
  clamps the latent tile, the LU output is then not the base resolution and
  the DiT rejects it.

Real clips satisfy neither (568x320, 426x240, 240-high sources at x2), so the
pipeline pads the source bottom/right onto both constraints before the
encode and crops the stitched SR back to ``source * scale`` afterwards. The
strip mirrors the source (reflect padding keeps every original pixel and
gives the encoder natural content at the border); a source smaller than the
strip itself, which reflection cannot produce, repeats its edge instead.
Callers never see the padding: the pixel path, which upsamples each tile to
the base resolution, takes any size as it is.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from pydantic import BaseModel, ConfigDict
from torch.nn import functional

TileSizeFor = Callable[[int, int], tuple[int, int]]

PIXEL_RANK = 4
STABLE_ITERATIONS = 3


class SourcePadding(BaseModel):
    """Bottom/right padding, in source pixels, that the LU path adds to a clip."""

    model_config = ConfigDict(frozen=True)

    bottom: int = 0
    right: int = 0

    @property
    def any(self) -> bool:
        """Whether the source is padded at all."""
        return self.bottom > 0 or self.right > 0

    def padded_hw(self, source_hw: tuple[int, int]) -> tuple[int, int]:
        """Size of the padded source."""
        return source_hw[0] + self.bottom, source_hw[1] + self.right

    def apply_to_video(self, video: torch.Tensor) -> torch.Tensor:
        """Pad a ``[T, C, H, W]`` pixel video; the input is returned as is when nothing is padded."""
        return _pad_bottom_right(video, self.bottom, self.right)

    def apply_to_latent(self, latent: torch.Tensor, spatial_factor: int) -> torch.Tensor:
        """Pad a ``[T, C, h, w]`` latent by the same amount expressed in latent units."""
        if self.bottom % spatial_factor or self.right % spatial_factor:
            raise ValueError(f"padding {self.bottom}x{self.right} is not a multiple of the VAE stride {spatial_factor}")
        return _pad_bottom_right(latent, self.bottom // spatial_factor, self.right // spatial_factor)

    def crop(self, frames: torch.Tensor, scale: int) -> torch.Tensor:
        """Remove the padding, scaled by the upscale factor, from a ``[C, T, H, W]`` SR result."""
        if not self.any:
            return frames
        height, width = frames.shape[-2:]
        return frames[..., : height - self.bottom * scale, : width - self.right * scale]


def ceil_to_multiple(value: int, multiple: int) -> int:
    """Smallest multiple of ``multiple`` that is ``>= value``."""
    return -(-value // multiple) * multiple


def padded_source_hw(
    height: int,
    width: int,
    spatial_factor: int,
    tile_hw_for: TileSizeFor | None,
) -> tuple[int, int]:
    """Target ``(H, W)`` of an LU-path source: on the VAE stride and at least one tile.

    ``tile_hw_for(h, w)`` is the tile a source of that size is cut with; it
    depends on the aspect ratio (through the closest trained base), so the
    target is re-evaluated on the padded size until it is stable.
    """
    target = (ceil_to_multiple(height, spatial_factor), ceil_to_multiple(width, spatial_factor))
    if tile_hw_for is None:
        return target
    for _ in range(STABLE_ITERATIONS):
        tile_h, tile_w = tile_hw_for(*target)
        candidate = (
            max(target[0], ceil_to_multiple(tile_h, spatial_factor)),
            max(target[1], ceil_to_multiple(tile_w, spatial_factor)),
        )
        if candidate == target:
            break
        target = candidate
    return target


def source_padding(
    height: int,
    width: int,
    spatial_factor: int,
    tile_hw_for: TileSizeFor | None,
) -> SourcePadding:
    """The padding that brings an ``height x width`` source onto the LU path's constraints."""
    target_h, target_w = padded_source_hw(height, width, spatial_factor, tile_hw_for)
    return SourcePadding(bottom=target_h - height, right=target_w - width)


def _pad_bottom_right(tensor: torch.Tensor, bottom: int, right: int) -> torch.Tensor:
    if tensor.ndim != PIXEL_RANK:
        raise ValueError(f"expected a rank-4 [T, C, H, W] tensor, got {tuple(tensor.shape)}")
    if bottom == 0 and right == 0:
        return tensor
    height, width = tensor.shape[-2:]
    # Reflection needs the strip to be shorter than the source on that axis.
    mode = "reflect" if bottom < height and right < width else "replicate"
    padded = functional.pad(tensor.float(), (0, right, 0, bottom), mode=mode)
    return padded.to(tensor.dtype)


__all__ = ["SourcePadding", "ceil_to_multiple", "padded_source_hw", "source_padding"]
