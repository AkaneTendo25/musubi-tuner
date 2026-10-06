"""Even tile-grid layout — the ``TILE_GRID_MODE="even"`` alternative.

The legacy ``axis_positions`` walks a fixed stride and clamps the last tile to
the edge, so all layout slack piles up in the final step: width 976 with tile
384 / stride 288 yields positions ``(0, 288, 576, 592)`` — the 4th column is
shifted 16 px and ~96% duplicates the 3rd. This module distributes the
MINIMAL number of tiles uniformly instead, with the overlap never dropping
below a floor.

The layout is computed in integer units of the VAE spatial factor, so every
position is factor-aligned by construction and the guarantees are exact (no
floating-point edge cases): full coverage, gaps within one unit of each
other, per-gap overlap >= ``min_overlap``, and never more tiles than the
legacy walk. Where the legacy stride divides the span exactly (all current
production shapes) the tile count is unchanged.
"""

from __future__ import annotations

import math

from musubi_tuner.kandinsky6.runtime.sr.core.algo.tiling_utils import TileGrid


def axis_positions_even(length: int, tile: int, min_overlap: float, snap: int) -> tuple[int, ...]:
    """Return evenly distributed, ``snap``-aligned tile start positions.

    Uses the minimal position count whose uniform stride keeps the tile
    overlap at or above ``min_overlap``, then spreads the positions evenly
    over ``[0, length - tile]`` in integer ``snap`` units. When ``length`` or
    ``tile`` is not ``snap``-aligned the same layout is computed at pixel
    precision (``snap=1``) — the latent-grid guard downstream still enforces
    alignment where it actually matters (the LU path).

    Args:
        length: Axis length in pixels.
        tile: Tile size on this axis.
        min_overlap: Overlap floor as a fraction of ``tile`` in ``[0, 1)``.
        snap: Position alignment unit (the VAE spatial factor).

    Returns:
        Strictly increasing positions covering ``[0, length - tile]``.

    Raises:
        ValueError: On non-positive ``tile``/``snap`` or ``min_overlap``
            outside ``[0, 1)``.
    """
    if tile <= 0 or snap <= 0 or not 0 <= min_overlap < 1:
        msg = f"invalid axis spec: tile={tile}, snap={snap}, min_overlap={min_overlap}"
        raise ValueError(msg)
    if tile >= length:
        return (0,)
    unit = snap if length % snap == 0 and tile % snap == 0 else 1
    span_units = (length - tile) // unit
    max_stride_units = max(1, math.floor(tile * (1.0 - min_overlap) / unit))
    count = math.ceil(span_units / max_stride_units) + 1
    return tuple(round(i * span_units / (count - 1)) * unit for i in range(count))


def compute_tile_grid_even(
    h: int,
    w: int,
    tile_hw: tuple[int, int],
    min_overlap: float,
    snap: int,
) -> TileGrid:
    """Build a :class:`TileGrid` with the even per-axis layout.

    Drop-in for ``compute_tile_grid(..., tile_hw=...)``: same ``TileGrid``
    contract (``tops``/``lefts`` are authoritative), only the position layout
    differs — see :func:`axis_positions_even`.

    Args:
        h: Video height in pixels.
        w: Video width in pixels.
        tile_hw: Explicit ``(tile_h, tile_w)``.
        min_overlap: Overlap floor as a fraction of the tile size.
        snap: Position alignment unit (the VAE spatial factor).

    Returns:
        ``TileGrid`` with evenly distributed, aligned tile positions.
    """
    tile_h, tile_w = tile_hw
    return TileGrid(
        tile_h=tile_h,
        tile_w=tile_w,
        tops=axis_positions_even(h, tile_h, min_overlap, snap),
        lefts=axis_positions_even(w, tile_w, min_overlap, snap),
    )
