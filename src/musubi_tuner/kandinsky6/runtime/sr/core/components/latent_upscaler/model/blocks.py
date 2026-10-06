# Reusable building blocks for latent upsampler architectures.

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.conv_ops import TemporalPadding, make_conv


class StochasticDepth(nn.Module):
    """Drop residual paths without requiring torchvision."""

    def __init__(self, p: float, mode: str = "row") -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"drop probability must be in [0, 1], got {p}")
        if mode not in ("batch", "row"):
            raise ValueError(f"mode must be 'batch' or 'row', got {mode!r}")
        self.p = p
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        if self.p == 1.0:
            return torch.zeros_like(x)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) if self.mode == "row" else (1,) * x.ndim
        noise = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(1.0 - self.p)
        return x * noise / (1.0 - self.p)


class RMSNorm(nn.Module):
    """Channel-first Root Mean Square normalization with learnable gamma.

    Args:
        dim: Number of channels.
        dims: Convolution dimensionality — 2 for ``(C,1,1)``, 3 for ``(C,1,1,1)``.
    """

    def __init__(self, dim: int, dims: int = 2) -> None:
        """Initialize RMSNorm.

        Args:
            dim: Number of channels.
            dims: Convolution dimensionality — 2 for ``(C,1,1)``, 3 for ``(C,1,1,1)``.
        """
        super().__init__()
        broadcastable_dims = (1, 1) if dims == 2 else (1, 1, 1)
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, *broadcastable_dims))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization along the channel dimension."""
        return functional.normalize(x, dim=1) * self.scale * self.gamma


class ModulatedRMSNorm(nn.Module):
    """RMSNorm followed by spatial FiLM modulation conditioned on a side tensor.

    Computes ``RMSNorm(x) * conv_y(zq) + conv_b(zq)`` where ``conv_y`` and
    ``conv_b`` are 1x1 convolutions of ``zq`` (the conditioning tensor, e.g.
    the LQ latent).  ``zq`` is aligned to ``x`` by nearest-neighbor
    interpolation along the spatial dims, so the same ``zq`` can be reused
    across feature pyramids of different resolutions.

    Both convs keep the stock PyTorch initialization, so modulation is active
    from the first step — the same regime as K-VAE's ``CachedSpatialNorm3D``,
    whose ``conv_y`` / ``conv_b`` are plain 1x1 convs with no custom init.

    Args:
        dim: Number of feature channels.
        zq_dim: Number of channels in the conditioning tensor.
        dims: Convolution dimensionality — 2 for Conv2d, 3 for Conv3d.
    """

    def __init__(self, dim: int, zq_dim: int, dims: int = 2) -> None:
        """Initialize ModulatedRMSNorm.

        Args:
            dim: Number of feature channels.
            zq_dim: Number of channels in the conditioning tensor.
            dims: Convolution dimensionality — 2 for Conv2d, 3 for Conv3d.
        """
        super().__init__()
        conv = nn.Conv2d if dims == 2 else nn.Conv3d
        self.norm = RMSNorm(dim, dims=dims)
        self.conv_y = conv(zq_dim, dim, kernel_size=1)
        self.conv_b = conv(zq_dim, dim, kernel_size=1)

    def forward(self, x: Tensor, zq: Tensor) -> Tensor:
        """Apply RMS norm + FiLM modulation by spatially-aligned ``zq``."""
        if zq.shape[2:] != x.shape[2:]:
            zq = functional.interpolate(zq, size=x.shape[2:], mode="nearest")
        return self.norm(x) * self.conv_y(zq) + self.conv_b(zq)


class LayerScale(nn.Module):
    """Learnable per-channel scaling applied before residual addition.

    Args:
        channels: Number of channels.
        init_value: Initial scale value (small for training stability).
        dims: Convolution dimensionality — 2 for ``(C,1,1)``, 3 for ``(C,1,1,1)``.
    """

    def __init__(self, channels: int, init_value: float = 1e-6, dims: int = 2) -> None:
        """Initialize LayerScale.

        Args:
            channels: Number of channels.
            init_value: Initial scale value.
            dims: Convolution dimensionality — 2 for ``(C,1,1)``, 3 for ``(C,1,1,1)``.
        """
        super().__init__()
        broadcastable_dims = (1, 1) if dims == 2 else (1, 1, 1)
        self.gamma = nn.Parameter(init_value * torch.ones(channels, *broadcastable_dims))

    def forward(self, x: Tensor) -> Tensor:
        """Scale input by learnable per-channel gamma."""
        return x * self.gamma


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXtV2).

    Aggregates global spatial info per channel and normalizes across channels
    to encourage feature diversity and prevent feature collapse.

    Args:
        channels: Number of channels.
        dims: Convolution dimensionality — 2 for spatial dims ``(2,3)``, 3 for ``(2,3,4)``.
    """

    def __init__(self, channels: int, dims: int = 2) -> None:
        """Initialize GRN.

        Args:
            channels: Number of channels.
            dims: Convolution dimensionality — 2 for spatial dims ``(2,3)``, 3 for ``(2,3,4)``.
        """
        super().__init__()
        self.spatial_dims: tuple[int, ...] = (2, 3) if dims == 2 else (2, 3, 4)
        broadcastable = (1, 1) if dims == 2 else (1, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, *broadcastable))
        self.beta = nn.Parameter(torch.zeros(1, channels, *broadcastable))

    def forward(self, x: Tensor) -> Tensor:
        """Apply global response normalization."""
        gx = torch.norm(x, p=2, dim=self.spatial_dims, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ResidualBlock(nn.Module):
    """Pre-activation residual block with optional LayerScale, GRN, and StochasticDepth.

    Standard path: ``RMSNorm → SiLU → Conv3x3 → RMSNorm → SiLU → [GRN →] Conv3x3``.
    Depthwise path: ``DWConv_KxK → RMSNorm → PWConv1x1 → SiLU → [GRN →] PWConv1x1``.
    Skip: identity when ``in_channels == out_channels``, else ``Conv 1x1``.

    When ``zq_dim`` is set, every ``RMSNorm`` is replaced by a
    ``ModulatedRMSNorm`` conditioned on a side tensor ``zq``, and the inner
    layers are stored as named submodules so ``zq`` can be threaded through
    ``forward(x, zq)``.  When ``zq_dim is None``, the legacy ``nn.Sequential``
    layout is preserved bit-identically — the parameter names match older
    checkpoints, so loading them remains backward compatible.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        mid_channels: int | None = None,
        dims: int = 2,
        layer_scale_init: float | None = None,
        *,
        grn: bool = False,
        stochastic_depth_prob: float = 0.0,
        depthwise: bool = False,
        kernel_size: int = 3,
        zq_dim: int | None = None,
        temporal_padding: TemporalPadding = "zeros",
    ) -> None:
        """Initialize residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (defaults to ``in_channels``).
            mid_channels: Internal channel width (defaults to ``out_channels``).
            dims: Convolution dimensionality — 2 for Conv2d, 3 for Conv3d.
            layer_scale_init: Initial LayerScale gamma value. ``None`` disables LayerScale.
            grn: Whether to insert Global Response Normalization on mid_channels.
            stochastic_depth_prob: Drop probability for stochastic depth (0 = off).
            depthwise: Use depthwise separable convolutions instead of standard convs.
            kernel_size: Kernel size for the depthwise convolution (only used when ``depthwise=True``).
            zq_dim: When set, replace every ``RMSNorm`` with ``ModulatedRMSNorm(zq_dim)``;
                the block then accepts a conditioning tensor in ``forward(x, zq)``.
            temporal_padding: How ``dims == 3`` convolutions extend T — ``"zeros"``
                (default) or ``"replicate"``, which repeats the edge frame as K-VAE does.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        mid_channels = mid_channels or out_channels
        self.depthwise = depthwise
        self.has_grn = grn
        self.zq_dim = zq_dim
        conv = make_conv(dims, temporal_padding)

        if zq_dim is None:
            self._build_legacy(in_channels, out_channels, mid_channels, dims, conv, grn=grn, kernel_size=kernel_size)
        else:
            self._build_modulated(
                in_channels,
                out_channels,
                mid_channels,
                dims,
                conv,
                grn=grn,
                kernel_size=kernel_size,
                zq_dim=zq_dim,
            )
        self.shortcut = nn.Identity() if in_channels == out_channels else conv(in_channels, out_channels, kernel_size=1)
        self.layer_scale: nn.Module = (
            LayerScale(out_channels, init_value=layer_scale_init, dims=dims)
            if layer_scale_init is not None
            else nn.Identity()
        )
        self.stochastic_depth: nn.Module = (
            StochasticDepth(stochastic_depth_prob, mode="row") if stochastic_depth_prob > 0 else nn.Identity()
        )

    def _build_legacy(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        dims: int,
        conv: type[nn.Module],
        *,
        grn: bool,
        kernel_size: int,
    ) -> None:
        """Build the original ``nn.Sequential`` layout (parameter names unchanged)."""
        if self.depthwise:
            ks = kernel_size if dims == 2 else (min(kernel_size, 5), kernel_size, kernel_size)
            pad = kernel_size // 2 if dims == 2 else (min(kernel_size, 5) // 2, kernel_size // 2, kernel_size // 2)
            layers: list[nn.Module] = [
                conv(in_channels, in_channels, kernel_size=ks, padding=pad, groups=in_channels),
                RMSNorm(in_channels, dims=dims),
                conv(in_channels, mid_channels, kernel_size=1),
                nn.SiLU(),
            ]
            if grn:
                layers.append(GRN(mid_channels, dims=dims))
            layers.append(conv(mid_channels, out_channels, kernel_size=1))
        else:
            layers = [
                RMSNorm(in_channels, dims=dims),
                nn.SiLU(),
                conv(in_channels, mid_channels, kernel_size=3, padding=1),
                RMSNorm(mid_channels, dims=dims),
                nn.SiLU(),
            ]
            if grn:
                layers.append(GRN(mid_channels, dims=dims))
            layers.append(conv(mid_channels, out_channels, kernel_size=3, padding=1))
        self.block = nn.Sequential(*layers)

    def _build_modulated(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        dims: int,
        conv: type[nn.Module],
        *,
        grn: bool,
        kernel_size: int,
        zq_dim: int,
    ) -> None:
        """Build a path with FiLM-modulated RMSNorms accepting ``zq`` in forward."""
        if self.depthwise:
            ks = kernel_size if dims == 2 else (min(kernel_size, 5), kernel_size, kernel_size)
            pad = kernel_size // 2 if dims == 2 else (min(kernel_size, 5) // 2, kernel_size // 2, kernel_size // 2)
            self.dwconv = conv(in_channels, in_channels, kernel_size=ks, padding=pad, groups=in_channels)
            self.norm1 = ModulatedRMSNorm(in_channels, zq_dim, dims=dims)
            self.pwconv1 = conv(in_channels, mid_channels, kernel_size=1)
            self.grn: nn.Module | None = GRN(mid_channels, dims=dims) if grn else None
            self.pwconv2 = conv(mid_channels, out_channels, kernel_size=1)
        else:
            self.norm1 = ModulatedRMSNorm(in_channels, zq_dim, dims=dims)
            self.conv1 = conv(in_channels, mid_channels, kernel_size=3, padding=1)
            self.norm2 = ModulatedRMSNorm(mid_channels, zq_dim, dims=dims)
            self.grn = GRN(mid_channels, dims=dims) if grn else None
            self.conv2 = conv(mid_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, zq: Tensor | None = None) -> Tensor:
        """Apply residual block with skip connection.

        Args:
            x: Feature tensor.
            zq: Optional conditioning tensor for FiLM-modulated norms.  Required
                when the block was constructed with ``zq_dim`` set; ignored otherwise.
        """
        out = self.block(x) if self.zq_dim is None else self.forward_modulated_path(x, zq)  # type: ignore
        out = self.layer_scale(out)
        return self.shortcut(x) + self.stochastic_depth(out)

    def forward_modulated_path(
        self,
        x: Tensor,
        zq: Tensor,
        norm1: ModulatedRMSNorm | None = None,
        norm2: ModulatedRMSNorm | None = None,
    ) -> Tensor:
        """Evaluate the residual path with optional scale-specific norms."""
        if self.zq_dim is None:
            msg = "forward_modulated_path requires a block with zq_dim"
            raise ValueError(msg)
        active_norm1 = norm1 if norm1 is not None else self.norm1
        if self.depthwise:
            h = self.dwconv(x)
            h = active_norm1(h, zq)
            h = self.pwconv1(h)
            h = functional.silu(h)
            if self.grn is not None:
                h = self.grn(h)
            return self.pwconv2(h)
        active_norm2 = norm2 if norm2 is not None else self.norm2
        h = active_norm1(x, zq)
        h = functional.silu(h)
        h = self.conv1(h)
        h = active_norm2(h, zq)
        h = functional.silu(h)
        if self.grn is not None:
            h = self.grn(h)
        return self.conv2(h)

    def forward_with_norms(
        self,
        x: Tensor,
        zq: Tensor,
        norm1: ModulatedRMSNorm,
        norm2: ModulatedRMSNorm,
    ) -> Tensor:
        """Apply the block with x2-specific modulation and shared transforms."""
        if self.depthwise:
            msg = "scale-specific normalization does not support depthwise residual blocks"
            raise ValueError(msg)
        out = self.forward_modulated_path(x, zq, norm1=norm1, norm2=norm2)
        out = self.layer_scale(out)
        return self.shortcut(x) + self.stochastic_depth(out)

    def zero_init_residual(self) -> None:
        """Initialize the residual path to zero so the block starts as identity."""
        if self.zq_dim is None:
            final_conv = self.block[-1]
        elif self.depthwise:
            final_conv = self.pwconv2
        else:
            final_conv = self.conv2
        nn.init.zeros_(final_conv.weight)  # type: ignore[arg-type]
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)  # type: ignore[arg-type]
