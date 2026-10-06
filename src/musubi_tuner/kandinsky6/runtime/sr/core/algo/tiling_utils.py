# Shared tiling utilities for spatial video tiling with Hanning-window blending.

from __future__ import annotations

from typing import NamedTuple

import torch


class TileGrid(NamedTuple):
    """Spatial tile grid parameters.

    ``tops`` and ``lefts`` are the authoritative per-axis tile start positions.
    They cover ``[0, length - tile]`` with the nominal stride; the **last**
    position is clamped to ``length - tile`` if a uniform stride would
    overshoot. This means consecutive gaps in ``tops`` / ``lefts`` may be
    shorter than the nominal stride near the right/bottom edge.

    ``n_h``, ``n_w``, ``total_tiles``, ``stride_h``, ``stride_w`` are derived
    properties exposed for backward compatibility with callers that only
    read them. To compute a tile position by index, prefer
    ``grid.tops[row]`` / ``grid.lefts[col]`` over ``row * grid.stride_h`` —
    the latter is wrong for the last clamped tile.
    """

    tile_h: int
    tile_w: int
    tops: tuple[int, ...]
    lefts: tuple[int, ...]

    @property
    def n_h(self) -> int:
        """Number of tile rows."""
        return len(self.tops)

    @property
    def n_w(self) -> int:
        """Number of tile columns."""
        return len(self.lefts)

    @property
    def total_tiles(self) -> int:
        """Total number of tiles (``n_h * n_w``)."""
        return self.n_h * self.n_w

    @property
    def stride_h(self) -> int:
        """Nominal vertical stride (gap between first two rows; ``tile_h`` if only one row)."""
        return self.tops[1] - self.tops[0] if len(self.tops) > 1 else self.tile_h

    @property
    def stride_w(self) -> int:
        """Nominal horizontal stride (gap between first two cols; ``tile_w`` if only one col)."""
        return self.lefts[1] - self.lefts[0] if len(self.lefts) > 1 else self.tile_w


def axis_positions(length: int, tile: int, stride: int) -> tuple[int, ...]:
    """Return tile start positions covering ``[0, length - tile]``.

    Walks at the given ``stride`` from 0 and appends positions while the tile
    still fits inside ``length``. If the last walked position is not exactly
    ``length - tile``, appends one final position clamped to the right edge.
    Single-position result if ``tile >= length``.

    Args:
        length: Axis length in pixels.
        tile: Tile size on this axis.
        stride: Nominal stride between consecutive tiles.

    Returns:
        Strictly increasing tuple of start positions.
    """
    if tile <= 0 or stride <= 0:
        msg = f"tile and stride must be positive (got tile={tile}, stride={stride})"
        raise ValueError(msg)
    if tile >= length:
        return (0,)

    positions: list[int] = []
    p = 0
    while p + tile <= length:
        positions.append(p)
        p += stride
    last = length - tile
    if positions[-1] != last:
        positions.append(last)
    return tuple(positions)


def compute_tile_grid(
    h: int,
    w: int,
    resolution_scale: int,
    overlap: float = 0.5,
    tile_hw: tuple[int, int] | None = None,
) -> TileGrid:
    """Compute tile geometry with configurable overlap.

    By default, tile size is ``(h // resolution_scale, w // resolution_scale)``.
    Pass ``tile_hw`` to override with explicit tile dimensions (e.g. derived
    from a fixed base resolution rather than the source frame size).

    The grid covers the full frame: if the nominal stride does not divide
    evenly into ``(h - tile_h)`` / ``(w - tile_w)``, the last tile in each
    axis is clamped to the right/bottom edge. ``stitch_tiles_hanning``
    handles the resulting non-uniform overlap correctly via Hanning-window
    normalisation.

    Args:
        h: Video height in pixels.
        w: Video width in pixels.
        resolution_scale: Divisor for tile size when ``tile_hw`` is ``None``.
        overlap: Fraction of tile overlap in ``[0, 1)``. Default ``0.5`` (50%).
        tile_hw: Explicit ``(tile_h, tile_w)`` override. When set,
            ``resolution_scale`` is ignored for tile sizing.

    Returns:
        ``TileGrid`` with tile sizes and per-axis tile start positions.
    """
    if tile_hw is None:
        tile_h = h // resolution_scale
        tile_w = w // resolution_scale
    else:
        tile_h, tile_w = tile_hw
    stride_h = max(1, int(tile_h * (1.0 - overlap)))
    stride_w = max(1, int(tile_w * (1.0 - overlap)))

    tops = axis_positions(h, tile_h, stride_h)
    lefts = axis_positions(w, tile_w, stride_w)
    return TileGrid(tile_h=tile_h, tile_w=tile_w, tops=tops, lefts=lefts)


def tile_origin_from_index(grid: TileGrid, tile_index: int) -> tuple[int, int]:
    """Recover the grid-aligned ``(top, left)`` pixel origin of a tile from its index.

    Inverse of the row-major ``tile_index`` assignment in
    ``datasets/data_encoding/video_download.py:load_video_tiled``: tiles are
    numbered column-first, so ``tile_index = row * n_w + col``.

    Args:
        grid: Tile grid parameters from :func:`compute_tile_grid`.
        tile_index: Row-major tile index in ``[0, grid.total_tiles)``.

    Returns:
        ``(top, left)`` pixel offset of the tile within the (unscaled) frame.

    Raises:
        ValueError: If ``tile_index`` is outside ``[0, grid.total_tiles)``.
    """
    if not 0 <= tile_index < grid.total_tiles:
        msg = f"tile_index {tile_index} out of range [0, {grid.total_tiles})"
        raise ValueError(msg)
    row, col = divmod(tile_index, grid.n_w)
    return grid.tops[row], grid.lefts[col]


def extract_all_tiles(
    video: torch.Tensor,
    grid: TileGrid,
) -> list[torch.Tensor]:
    """Extract all spatial tiles from a video tensor.

    Args:
        video: ``[T, C, H, W]`` tensor.
        grid: Tile grid parameters from ``compute_tile_grid``.

    Returns:
        List of ``[T, C, tile_h, tile_w]`` tensors in row-major order
        (``tops`` x ``lefts``).
    """
    tiles: list[torch.Tensor] = []
    for top in grid.tops:
        for left in grid.lefts:
            tile = video[:, :, top : top + grid.tile_h, left : left + grid.tile_w]
            tiles.append(tile)
    return tiles


def hanning_window_2d(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Create a 2D Hanning window with non-zero endpoints.

    Uses ``hann_window(n + 2)[1:-1]`` to avoid exact zeros at boundaries,
    ensuring non-zero weight where only one tile contributes.

    Args:
        h: Window height.
        w: Window width.
        device: Target device.

    Returns:
        ``[h, w]`` float tensor with values in ``(0, 1]``.
    """
    wy = torch.hann_window(h + 2, device=device)[1:-1]
    wx = torch.hann_window(w + 2, device=device)[1:-1]
    return wy[:, None] * wx[None, :]


def stitch_tiles_hanning(
    tiles: list[torch.Tensor],
    grid: TileGrid,
    original_h: int,
    original_w: int,
    scale: int = 1,
) -> torch.Tensor:
    """Stitch tiles into a full frame using Hanning-window weighted blending.

    Each tile is multiplied by a 2D Hanning window and accumulated into
    the output canvas. The final result is normalised by the accumulated
    weights so that overlapping regions blend smoothly. The same scheme
    works for non-uniform overlap at the right/bottom edge: in the clamped
    region both ``pred_acc`` and ``weight_acc`` receive more contributions,
    and the per-pixel division cancels it out.

    When ``scale > 1``, tiles are assumed to be at HR resolution
    (i.e. each tile covers ``tile_h * scale x tile_w * scale`` pixels)
    and the output canvas is ``original_h * scale x original_w * scale``.
    Grid positions are scaled accordingly.

    Args:
        tiles: List of ``[C, T, th, tw]`` float tensors in row-major order
            (same order as ``extract_all_tiles``). When ``scale == 1``,
            ``th == grid.tile_h``; when ``scale > 1``, ``th == grid.tile_h * scale``.
        grid: Tile grid parameters (at LQ / original resolution).
        original_h: LQ frame height.
        original_w: LQ frame width.
        scale: Upscale factor. Output resolution is
            ``(original_h * scale, original_w * scale)``.

    Returns:
        ``[C, T, original_h * scale, original_w * scale]`` float tensor.
    """
    first = tiles[0]
    c, t = first.shape[0], first.shape[1]
    hr_tile_h, hr_tile_w = first.shape[2], first.shape[3]
    device = first.device

    window = hanning_window_2d(hr_tile_h, hr_tile_w, device)
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, hr_tile_h, hr_tile_w]

    out_h = original_h * scale
    out_w = original_w * scale
    pred_acc = torch.zeros(c, t, out_h, out_w, device=device)
    weight_acc = torch.zeros(1, 1, out_h, out_w, device=device)

    tile_idx = 0
    for top in grid.tops:
        for left in grid.lefts:
            y = top * scale
            x = left * scale
            tile = tiles[tile_idx]
            pred_acc[:, :, y : y + hr_tile_h, x : x + hr_tile_w] += tile * window
            weight_acc[:, :, y : y + hr_tile_h, x : x + hr_tile_w] += window
            tile_idx += 1

    return pred_acc / weight_acc.clamp(min=1e-6)
