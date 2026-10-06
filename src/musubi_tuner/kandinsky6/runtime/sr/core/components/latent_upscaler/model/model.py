# Convolutional latent-space spatial upsampler with configurable upsampling strategy.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from einops import rearrange
from torch import Tensor, nn

from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.blocks import ResidualBlock, RMSNorm
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.conv_ops import make_conv
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.runtime import forward_with_checkpointing
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.upsample_ops import (
    BilinearUpsampleND,
    PixelShuffleND,
    PXSv2HybridUpsampleND,
    PXSv2UpsampleND,
)

if TYPE_CHECKING:
    from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.config import UpsampleMode, UpsamplePosition
    from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.conv_ops import (
        TemporalPadding,
        UpsamplePaddingMode,
    )


class ConvLatentUpsampler(nn.Module):
    """Lightweight convolutional spatial upsampler for VAE latents.

    Upsampling strategies (controlled by ``upsample_mode``):
        pixel_shuffle: Learned channel-to-space rearrangement (default).
        bilinear: Parameter-free bilinear interpolation.
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 64,
        bottleneck_channels: int | None = None,
        num_pre_residual_blocks: int = 0,
        num_residual_blocks: int = 16,
        upscale_factor: int = 2,
        *,
        input_skip: bool = True,
        upsample_mode: UpsampleMode = "pixel_shuffle",
        upsample_position: UpsamplePosition = "after_projection",
        temporal_mix: bool = True,
        icnr: bool = False,
        gradient_checkpointing: bool = False,
        dims: int = 2,
        expand_ratio: int = 4,
        kernel_size: int = 3,
        layer_scale_init: float | None = None,
        grn: bool = False,
        stochastic_depth_rate: float = 0.0,
        depthwise: bool = False,
        stem_channels: int | None = None,
        zq_dim: int | None = None,
        temporal_padding: TemporalPadding = "zeros",
        upsample_padding_mode: UpsamplePaddingMode = "reflect",
    ) -> None:
        """Initialize the upsampler.

        Args:
            in_channels: Number of VAE latent channels.
            hidden_channels: Internal width of residual blocks.
            bottleneck_channels: Mid-channel bottleneck width for ResidualBlocks (None = hidden_channels).
            num_pre_residual_blocks: Number of residual blocks before upsampling (default 0).
            num_residual_blocks: Number of post-upsample residual refinement blocks.
            upscale_factor: Spatial upscale factor for pixel_shuffle and bilinear modes.
            input_skip: Add ``repeat_interleave`` skip connection from input to ``conv_in`` output.
            upsample_mode: Upsampling strategy — "pixel_shuffle" or "bilinear" (parameter-free interpolation).
            upsample_position: Where to apply upsampling — "before_projection" or "after_projection".
            temporal_mix: Whether pixel-shuffle Conv3d mixes across the temporal dimension.
                When False, uses (1,3,3) kernel. Only affects dims=3 + pixel_shuffle.
            icnr: Apply ICNR initialization to the sub-pixel-conv weight inside the
                ``pixel_shuffle`` and ``pxs_v2_hybrid`` upsample modes. Removes
                checkerboard at init. Ignored by ``bilinear`` and ``pxs_v2``.
            gradient_checkpointing: Enable gradient checkpointing for residual blocks.
            dims: Convolution dimensionality — 2 for Conv2d, 3 for Conv3d.
            expand_ratio: Expansion ratio for inverted bottleneck (used when ``bottleneck_channels`` is None).
            kernel_size: Kernel size for depthwise convolutions.
            layer_scale_init: Initial LayerScale gamma value. ``None`` disables LayerScale.
            grn: Whether to insert Global Response Normalization in residual blocks.
            stochastic_depth_rate: Maximum drop rate for stochastic depth (linearly scheduled).
            depthwise: Use depthwise separable convolutions in residual blocks.
            stem_channels: When set, pre-residual blocks and input projection operate at this
                narrower width, with a ResidualBlock transition to ``hidden_channels`` after upsampling.
            zq_dim: When set, every ``ResidualBlock`` uses ``ModulatedRMSNorm`` conditioned on
                a side tensor (the LQ latent), re-injecting LQ structure at every block via FiLM.
            temporal_padding: How ``dims == 3`` convolutions extend T — ``"zeros"``
                (default) or ``"replicate"``, which repeats the edge frame as K-VAE does.
            upsample_padding_mode: Edge handling for the ``pxs_v2`` residual conv —
                ``"reflect"`` (default) or ``"zeros"``, matching the production K-VAE.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_pre_residual_blocks = num_pre_residual_blocks
        self.num_residual_blocks = num_residual_blocks
        self.upscale_factor = upscale_factor
        self.upsample_mode = upsample_mode
        self.upsample_position = upsample_position
        self.input_skip = input_skip
        self.gradient_checkpointing = gradient_checkpointing
        self.dims = dims
        self.zq_dim = zq_dim

        # Channel widths: stem (pre-upsample) vs main (post-upsample)
        proj_channels = stem_channels if stem_channels is not None else hidden_channels
        post_bottleneck = bottleneck_channels if bottleneck_channels is not None else hidden_channels * expand_ratio
        pre_bottleneck = bottleneck_channels if bottleneck_channels is not None else proj_channels * expand_ratio
        self.bottleneck_channels = post_bottleneck

        # Projection layers
        conv = make_conv(dims, temporal_padding)
        self.input_proj = nn.Sequential(
            conv(in_channels, proj_channels, kernel_size=3, padding=1),
            RMSNorm(proj_channels, dims=dims),
            nn.SiLU(),
        )
        self.output_proj = nn.Sequential(
            RMSNorm(hidden_channels, dims=dims),
            nn.SiLU(),
            conv(hidden_channels, in_channels, kernel_size=3, padding=1),
        )

        # Spatial upsampling
        upsample_channels = in_channels if upsample_position == "before_projection" else proj_channels
        if upsample_mode == "pixel_shuffle":
            self.upsample: nn.Module = PixelShuffleND(
                upsample_channels,
                dims=dims,
                factor=upscale_factor,
                temporal_mix=temporal_mix,
                icnr=icnr,
                temporal_padding=temporal_padding,
            )
        elif upsample_mode == "bilinear":
            self.upsample = BilinearUpsampleND(dims=dims, factor=upscale_factor)
        elif upsample_mode == "pxs_v2":
            self.upsample = PXSv2UpsampleND(
                upsample_channels,
                dims=dims,
                factor=upscale_factor,
                padding_mode=upsample_padding_mode,
            )
        else:
            self.upsample = PXSv2HybridUpsampleND(
                upsample_channels,
                dims=dims,
                factor=upscale_factor,
                temporal_mix=temporal_mix,
                icnr=icnr,
                temporal_padding=temporal_padding,
                padding_mode=upsample_padding_mode,
            )

        # Channel transition: bridges stem width → hidden width after upsampling
        self.channel_transition: ResidualBlock | None = None
        if stem_channels is not None:
            self.channel_transition = ResidualBlock(
                proj_channels,
                out_channels=hidden_channels,
                mid_channels=post_bottleneck,
                dims=dims,
                grn=grn,
                zq_dim=zq_dim,
                temporal_padding=temporal_padding,
            )

        # Residual stacks
        total_blocks = num_pre_residual_blocks + num_residual_blocks
        shared_block_kwargs = {
            "total_blocks": total_blocks,
            "stochastic_depth_rate": stochastic_depth_rate,
            "dims": dims,
            "layer_scale_init": layer_scale_init,
            "grn": grn,
            "depthwise": depthwise,
            "kernel_size": kernel_size,
            "zq_dim": zq_dim,
            "temporal_padding": temporal_padding,
        }
        self.pre_residual_blocks = self._build_residual_stack(
            num_pre_residual_blocks,
            proj_channels,
            pre_bottleneck,
            start_idx=0,
            **shared_block_kwargs,  # type: ignore[arg-type]
        )
        self.residual_blocks = self._build_residual_stack(
            num_residual_blocks,
            hidden_channels,
            post_bottleneck,
            start_idx=num_pre_residual_blocks,
            **shared_block_kwargs,  # type: ignore[arg-type]
        )

    def _build_residual_stack(
        self,
        count: int,
        channels: int,
        mid_channels: int,
        *,
        start_idx: int,
        total_blocks: int,
        stochastic_depth_rate: float,
        dims: int,
        layer_scale_init: float | None,
        grn: bool,
        depthwise: bool,
        kernel_size: int,
        zq_dim: int | None = None,
        temporal_padding: TemporalPadding = "zeros",
    ) -> nn.Sequential:
        """Build a stack of ResidualBlocks with linearly scheduled stochastic depth."""
        blocks: list[nn.Module] = []
        for i in range(count):
            drop_prob = stochastic_depth_rate * (start_idx + i) / max(total_blocks - 1, 1)
            blocks.append(
                ResidualBlock(
                    channels,
                    mid_channels=mid_channels,
                    dims=dims,
                    layer_scale_init=layer_scale_init,
                    grn=grn,
                    stochastic_depth_prob=drop_prob,
                    depthwise=depthwise,
                    kernel_size=kernel_size,
                    zq_dim=zq_dim,
                    temporal_padding=temporal_padding,
                )
            )
        return nn.Sequential(*blocks)

    def _apply_blocks(
        self,
        x: Tensor,
        blocks: nn.Sequential,
        *,
        use_checkpointing: bool,
        zq: Tensor | None = None,
    ) -> Tensor:
        """Apply a residual stack with optional modulation."""
        for block in blocks:
            if zq is not None:
                x = forward_with_checkpointing(block, x, zq, use_checkpointing=use_checkpointing)
            else:
                x = forward_with_checkpointing(block, x, use_checkpointing=use_checkpointing)
        return x

    def _apply_input_projection(self, x: Tensor) -> Tensor:
        z_skip = x
        x = self.input_proj(x)
        if self.input_skip:
            repeats = x.shape[1] // z_skip.shape[1]
            x = x + z_skip.repeat_interleave(repeats, dim=1)
        return x

    @property
    def last_layer_weight(self) -> torch.Tensor:
        """Weight of the final output conv (used by adaptive GAN-weight balancing)."""
        return self.output_proj[-1].weight  # type: ignore

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Upsample latent tensor spatially.

        Args:
            z: Input latent tensor of shape ``(B, C, T, H, W)``.

        Returns:
            Upsampled tensor of shape ``(B, C, T, H', W')``.
        """
        b, _, t, _, _ = z.shape
        x = rearrange(z, "b c t h w -> (b t) c h w") if self.dims == 2 else z
        zq = x if self.zq_dim is not None else None
        if self.upsample_position == "before_projection":
            x = self.upsample(x)
        x = self._apply_input_projection(x)
        use_ckpt = self.training and self.gradient_checkpointing
        x = self._apply_blocks(
            x,
            self.pre_residual_blocks,
            use_checkpointing=use_ckpt,
            zq=zq,
        )
        if self.upsample_position == "after_projection":
            x = self.upsample(x)
        if self.channel_transition is not None:
            if zq is not None:
                x = forward_with_checkpointing(self.channel_transition, x, zq, use_checkpointing=use_ckpt)
            else:
                x = forward_with_checkpointing(self.channel_transition, x, use_checkpointing=use_ckpt)
        x = self._apply_blocks(
            x,
            self.residual_blocks,
            use_checkpointing=use_ckpt,
            zq=zq,
        )
        x = self.output_proj(x)
        return rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t) if self.dims == 2 else x
