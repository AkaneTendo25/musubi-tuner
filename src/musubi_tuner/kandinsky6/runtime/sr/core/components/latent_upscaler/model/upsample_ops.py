# Spatial upsampling operations: pixel-shuffle, bilinear, and K-VAE PXS v2.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional

from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.conv_ops import TemporalPadding, UpsamplePaddingMode, make_conv

if TYPE_CHECKING:
    from collections.abc import Callable

DIMS_2 = 2
DIMS_3 = 3
VALID_DIMS = {DIMS_2, DIMS_3}


def icnr_(
    weight: torch.Tensor,
    factor: int,
    init_fn: Callable[[torch.Tensor], torch.Tensor] = nn.init.kaiming_normal_,
) -> None:
    """Initialize sub-pixel-conv weight so initial output equals nearest 2x.

    Implements the ICNR scheme of Aitken et al. (2017): the ``factor**2``
    sub-channel groups consumed by ``PixelShuffle(factor)`` are initialized
    with identical kernels, so on the first forward pass the output is a
    pure nearest-neighbor upsample. This eliminates checkerboard artifacts
    at the start of training.

    Args:
        weight: Conv weight tensor of shape ``(C_out * factor**2, C_in, *kernel)``.
            Modified in-place.
        factor: Upscale factor used by the downstream ``PixelShuffle``.
        init_fn: Base initializer applied to the underlying ``C_out`` kernel
            before replication. Defaults to Kaiming normal.

    Raises:
        ValueError: If ``weight.shape[0]`` is not divisible by ``factor**2``.
    """
    out_c = weight.shape[0] // factor**2
    if out_c * factor**2 != weight.shape[0]:
        msg = f"weight.shape[0] ({weight.shape[0]}) must be divisible by factor**2 ({factor**2})"
        raise ValueError(msg)
    base = torch.empty(out_c, *weight.shape[1:], device=weight.device, dtype=weight.dtype)
    init_fn(base)
    weight.data.copy_(base.repeat_interleave(factor**2, dim=0))


def spatial_nearest_2x(x: torch.Tensor, dims: int, factor: int) -> torch.Tensor:
    """Apply nearest-neighbor upsampling to H, W only, preserving B (and T).

    For 5D ``(B, C, T, H, W)`` tensors, T is merged into batch, interpolated,
    and unmerged. For 4D tensors, interpolation runs directly.

    Args:
        x: Input tensor of shape ``(B, C, H, W)`` or ``(B, C, T, H, W)``.
        dims: Tensor dimensionality — 2 for 4D, 3 for 5D.
        factor: Spatial upscale factor.

    Returns:
        Upsampled tensor with H, W scaled by ``factor``.
    """
    if dims == DIMS_3:
        b, _c, t, _h, _w = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = functional.interpolate(x, scale_factor=factor, mode="nearest")
        return rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
    return functional.interpolate(x, scale_factor=factor, mode="nearest")


class PixelShuffleND(nn.Module):
    """Conv + pixel-shuffle that upscales only H and W.

    Internally applies a convolution to expand channels by ``upscale_factor ** 2``,
    then rearranges channels into spatial dimensions.

    Supports both 4D ``(B, C, H, W)`` and 5D ``(B, C, T, H, W)`` tensors.
    The temporal dimension is always preserved.

    Args:
        channels: Number of input (and output) channels.
        dims: Tensor dimensionality — 2 for 4D input, 3 for 5D (video) input.
        factor: Spatial upscale factor applied to both H and W.
        temporal_mix: Whether the 3D convolution mixes across the temporal dimension.
            When ``False`` and ``dims == 3``, uses ``kernel_size=(1, 3, 3)`` so that
            each frame is convolved independently in H/W. Ignored when ``dims == 2``.
            Defaults to ``True`` (full 3x3x3 kernel).
        icnr: Apply ICNR initialization (Aitken et al., 2017) to the conv weight
            and zero its bias, so the module starts as a nearest-neighbor upsample.
            Removes checkerboard at init. Defaults to ``False``.
        temporal_padding: How the ``temporal_mix=True`` kernel extends T —
            ``"zeros"`` (default) or ``"replicate"``, matching K-VAE's edge repeat.
    """

    def __init__(
        self,
        channels: int,
        dims: int,
        factor: int = 2,
        *,
        temporal_mix: bool = True,
        icnr: bool = False,
        temporal_padding: TemporalPadding = "zeros",
    ) -> None:
        """Initialize PixelShuffleND."""
        super().__init__()
        if dims not in VALID_DIMS:
            msg = f"dims must be one of {VALID_DIMS}, got {dims}"
            raise ValueError(msg)
        self.dims = dims
        self.factor = factor

        if dims == DIMS_2:
            self.conv = nn.Conv2d(channels, channels * factor**2, kernel_size=3, padding=1)
        elif temporal_mix:
            conv3d = make_conv(DIMS_3, temporal_padding)
            self.conv = conv3d(channels, channels * factor**2, kernel_size=3, padding=1)
        else:
            self.conv = nn.Conv3d(channels, channels * factor**2, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        if icnr:
            icnr_(self.conv.weight, factor)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand channels via convolution, then rearrange into spatial dimensions."""
        x = self.conv(x)
        if self.dims == DIMS_3:
            return rearrange(x, "b (c p1 p2) t h w -> b c t (h p1) (w p2)", p1=self.factor, p2=self.factor)
        return rearrange(x, "b (c p1 p2) h w -> b c (h p1) (w p2)", p1=self.factor, p2=self.factor)


class BilinearUpsampleND(nn.Module):
    """Parameter-free bilinear spatial upsampling for 4D and 5D tensors.

    Upscales only H and W dimensions. For 5D ``(B, C, T, H, W)`` tensors,
    the temporal dimension is merged into the batch, interpolated in 2D,
    and unmerged.

    Args:
        dims: Tensor dimensionality — 2 for 4D input, 3 for 5D (video) input.
        factor: Spatial upscale factor applied to both H and W.
    """

    def __init__(self, dims: int, factor: int = 2) -> None:
        """Initialize BilinearUpsampleND."""
        super().__init__()
        if dims not in VALID_DIMS:
            msg = f"dims must be one of {VALID_DIMS}, got {dims}"
            raise ValueError(msg)
        self.dims = dims
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bilinear interpolation to spatial dimensions."""
        if self.dims == DIMS_3:
            b, _c, t, _h, _w = x.shape
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = functional.interpolate(x, scale_factor=self.factor, mode="bilinear", align_corners=False)
            return rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
        return functional.interpolate(x, scale_factor=self.factor, mode="bilinear", align_corners=False)


class PXSv2UpsampleND(nn.Module):
    """K-VAE PXS v2 spatial upsample: ``nearest 2x + Conv_(1,3,3) residual + Conv_1x1``.

    Replicates ``CachedPXSUpsample.spatial_upsample_NEW`` followed by the
    post-``linear`` 1x1x1 conv from K-VAE-3D-2.0. The base path is a
    parameter-free nearest upsample; the residual conv adds learnable detail
    on top, and a final pointwise conv mixes channels.

    Spatial conv is hard-coded to ``kernel=(1, 3, 3)`` for ``dims == 3``,
    matching K-VAE: per-frame, no temporal mixing through the upsample itself.
    Temporal correlation is restored by surrounding ResBlocks.

    Args:
        channels: Number of input (and output) channels.
        dims: Tensor dimensionality — 2 for 4D, 3 for 5D.
        factor: Spatial upscale factor for H and W.
        with_linear: Append a final 1x1 (or 1x1x1) pointwise conv. Disable
            when this module is composed inside a parent that owns its own
            post-mixer (e.g. ``PXSv2HybridUpsampleND``). Defaults to ``True``.
        padding_mode: Edge handling for the residual conv. ``"reflect"`` is the
            historical default; ``"zeros"`` matches the production K-VAE, whose
            ``CachedPXSUpsample`` resolves ``padding_mode or 'reflect'`` against a
            sidecar that passes ``'zeros'``.
    """

    def __init__(
        self,
        channels: int,
        dims: int,
        factor: int = 2,
        *,
        with_linear: bool = True,
        padding_mode: UpsamplePaddingMode = "reflect",
    ) -> None:
        """Initialize PXSv2UpsampleND."""
        super().__init__()
        if dims not in VALID_DIMS:
            msg = f"dims must be one of {VALID_DIMS}, got {dims}"
            raise ValueError(msg)
        self.dims = dims
        self.factor = factor

        if dims == DIMS_2:
            self.spatial_conv: nn.Module = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            )
            self.linear: nn.Module = nn.Conv2d(channels, channels, kernel_size=1) if with_linear else nn.Identity()
        else:
            self.spatial_conv = nn.Conv3d(
                channels,
                channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                padding_mode=padding_mode,
            )
            self.linear = nn.Conv3d(channels, channels, kernel_size=1) if with_linear else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``Linear( NN_2x(x) + Conv·NN_2x(x) )``."""
        up = spatial_nearest_2x(x, self.dims, self.factor)
        return self.linear(up + self.spatial_conv(up))


class PXSv2HybridUpsampleND(nn.Module):
    """Hybrid upsample: ICNR-PXS branch parallel to PXS v2 branch, mixed by 1x1.

    Combines the learnable sub-pixel-conv branch (``PixelShuffleND``, optionally
    ICNR-initialized so it starts as nearest) with the K-VAE PXS v2 branch
    (``nearest + Conv_(1,3,3) residual``), then mixes both via a final 1x1
    pointwise conv.

    Args:
        channels: Number of input (and output) channels.
        dims: Tensor dimensionality — 2 for 4D, 3 for 5D.
        factor: Spatial upscale factor for H and W.
        temporal_mix: Forwarded to the inner ``PixelShuffleND`` (controls 3x3x3
            vs (1,3,3) kernel for ``dims == 3``). The PXS v2 branch always uses
            (1,3,3) regardless. Defaults to ``True``.
        icnr: Apply ICNR initialization to the inner ``PixelShuffleND`` so it
            starts as nearest, matching the PXS v2 branch and removing
            checkerboard at init. Defaults to ``True``.
        temporal_padding: Forwarded to the inner ``PixelShuffleND``.
        padding_mode: Forwarded to the inner ``PXSv2UpsampleND``.
    """

    def __init__(
        self,
        channels: int,
        dims: int,
        factor: int = 2,
        *,
        temporal_mix: bool = True,
        icnr: bool = True,
        temporal_padding: TemporalPadding = "zeros",
        padding_mode: UpsamplePaddingMode = "reflect",
    ) -> None:
        """Initialize PXSv2HybridUpsampleND."""
        super().__init__()
        if dims not in VALID_DIMS:
            msg = f"dims must be one of {VALID_DIMS}, got {dims}"
            raise ValueError(msg)
        self.pxs = PixelShuffleND(
            channels,
            dims=dims,
            factor=factor,
            temporal_mix=temporal_mix,
            icnr=icnr,
            temporal_padding=temporal_padding,
        )
        self.v3 = PXSv2UpsampleND(
            channels,
            dims=dims,
            factor=factor,
            with_linear=False,
            padding_mode=padding_mode,
        )
        if dims == DIMS_2:
            self.linear: nn.Module = nn.Conv2d(channels, channels, kernel_size=1)
        else:
            self.linear = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``Linear( PXS(x) + (NN_2x + Conv·NN_2x)(x) )``."""
        return self.linear(self.pxs(x) + self.v3(x))
