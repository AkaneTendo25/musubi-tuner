# Construction and execution helpers for the cascaded multi-scale model.

from __future__ import annotations

from typing import Literal, NamedTuple

from pydantic import BaseModel
from torch import Tensor, nn

from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.blocks import ModulatedRMSNorm, ResidualBlock, RMSNorm
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.conv_ops import TemporalPadding, UpsamplePaddingMode, make_conv
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.motion_correspondence import MotionCorrespondenceBlock, MotionCorrespondenceSpec
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.runtime import forward_with_checkpointing
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.upsample_ops import (
    BilinearUpsampleND,
    PixelShuffleND,
    PXSv2HybridUpsampleND,
    PXSv2UpsampleND,
)

STAGE_FACTOR = 2


class UpsampleSpec(BaseModel):
    """Configuration shared by the two spatial upsample stages."""

    mode: Literal["pixel_shuffle", "bilinear", "pxs_v2", "pxs_v2_hybrid"]
    dims: int
    temporal_mix: bool
    icnr: bool
    temporal_padding: TemporalPadding = "zeros"
    padding_mode: UpsamplePaddingMode = "reflect"


class ResidualStackSpec(BaseModel):
    """Configuration shared by residual stacks at all model resolutions.

    in_channels: input width of the stack's first block; None means use channels.
    """

    channels: int
    mid_channels: int
    total_blocks: int
    stochastic_depth_rate: float
    dims: int
    layer_scale_init: float | None
    grn: bool
    depthwise: bool
    kernel_size: int
    zq_dim: int | None
    in_channels: int | None = None
    temporal_padding: TemporalPadding = "zeros"


class MotionAttentionSpec(BaseModel):
    """Construction and placement contract for motion-correspondence blocks."""

    after_mid_blocks: tuple[int, ...]
    block: MotionCorrespondenceSpec


class BlockSequence(NamedTuple):
    """Non-registering residual and motion-attention execution view."""

    blocks: nn.Sequential
    motion_blocks: nn.ModuleList | None = None
    motion_after_blocks: tuple[int, ...] = ()


def build_stem(
    in_channels: int,
    hidden_channels: int,
    dims: int,
    *,
    bare: bool,
    temporal_padding: TemporalPadding = "zeros",
) -> nn.Sequential:
    """Build an input projection: bare conv, or the legacy conv + RMSNorm + SiLU."""
    conv = make_conv(dims, temporal_padding)
    projection = conv(in_channels, hidden_channels, kernel_size=3, padding=1)
    if bare:
        return nn.Sequential(projection)
    return nn.Sequential(projection, RMSNorm(hidden_channels, dims=dims), nn.SiLU())


def build_output_head(
    hidden_channels: int,
    out_channels: int,
    dims: int,
    *,
    zq_dim: int | None,
    temporal_padding: TemporalPadding = "zeros",
) -> nn.Sequential:
    """Build a norm + SiLU + conv output head, zq-conditioned when ``zq_dim`` is set."""
    conv = make_conv(dims, temporal_padding)
    norm = _output_norm(hidden_channels, dims, zq_dim=zq_dim)
    return nn.Sequential(norm, nn.SiLU(), conv(hidden_channels, out_channels, kernel_size=3, padding=1))


def _output_norm(hidden_channels: int, dims: int, *, zq_dim: int | None) -> nn.Module:
    """Build the norm that opens an output head."""
    if zq_dim is not None:
        return ModulatedRMSNorm(hidden_channels, zq_dim, dims=dims)
    return RMSNorm(hidden_channels, dims=dims)


class OutputHead(nn.Module):
    """Norm, SiLU, and conv whose names match the published Diffusers state dict."""

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        dims: int,
        *,
        zq_dim: int | None,
        temporal_padding: TemporalPadding = "zeros",
    ) -> None:
        """Build ``norm`` + SiLU + ``conv``, zq-conditioned when ``zq_dim`` is set."""
        super().__init__()
        conv = make_conv(dims, temporal_padding)
        self.norm = _output_norm(hidden_channels, dims, zq_dim=zq_dim)
        self.act = nn.SiLU()
        self.conv = conv(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor, zq: Tensor | None = None) -> Tensor:
        """Project features, threading ``zq`` through a modulated norm."""
        norm = self.norm
        x = norm(x, zq) if isinstance(norm, ModulatedRMSNorm) else norm(x)
        return self.conv(self.act(x))


def apply_output_head(head: OutputHead | nn.Sequential, x: Tensor, zq: Tensor | None) -> Tensor:
    """Run an output head, threading ``zq`` through a modulated first norm."""
    if isinstance(head, OutputHead):
        return head(x, zq)
    norm = head[0]
    x = norm(x, zq) if isinstance(norm, ModulatedRMSNorm) else norm(x)
    x = head[1](x)
    return head[2](x)


def build_stage_upsample(channels: int, spec: UpsampleSpec) -> nn.Module:
    """Build one 2x spatial upsample stage."""
    if spec.mode == "pixel_shuffle":
        return PixelShuffleND(
            channels,
            dims=spec.dims,
            factor=STAGE_FACTOR,
            temporal_mix=spec.temporal_mix,
            icnr=spec.icnr,
            temporal_padding=spec.temporal_padding,
        )
    if spec.mode == "bilinear":
        return BilinearUpsampleND(dims=spec.dims, factor=STAGE_FACTOR)
    if spec.mode == "pxs_v2":
        return PXSv2UpsampleND(channels, dims=spec.dims, factor=STAGE_FACTOR, padding_mode=spec.padding_mode)
    return PXSv2HybridUpsampleND(
        channels,
        dims=spec.dims,
        factor=STAGE_FACTOR,
        temporal_mix=spec.temporal_mix,
        icnr=spec.icnr,
        temporal_padding=spec.temporal_padding,
        padding_mode=spec.padding_mode,
    )


def build_residual_stack(count: int, start_index: int, spec: ResidualStackSpec) -> nn.Sequential:
    """Build residual blocks with linearly scheduled stochastic depth."""
    blocks: list[nn.Module] = []
    for index in range(count):
        drop_probability = spec.stochastic_depth_rate * (start_index + index) / max(spec.total_blocks - 1, 1)
        block_in = spec.in_channels if index == 0 and spec.in_channels is not None else spec.channels
        blocks.append(
            ResidualBlock(
                block_in,
                out_channels=spec.channels,
                mid_channels=spec.mid_channels,
                dims=spec.dims,
                layer_scale_init=spec.layer_scale_init,
                grn=spec.grn,
                stochastic_depth_prob=drop_probability,
                depthwise=spec.depthwise,
                kernel_size=spec.kernel_size,
                zq_dim=spec.zq_dim,
                temporal_padding=spec.temporal_padding,
            )
        )
    return nn.Sequential(*blocks)


def build_motion_attention_stack(spec: MotionAttentionSpec) -> nn.ModuleList:
    """Build one motion block for every configured mid-stage placement."""
    return nn.ModuleList([MotionCorrespondenceBlock(spec.block) for _placement in spec.after_mid_blocks])


def apply_block_sequence(
    x: Tensor,
    zq: Tensor | None,
    sequence: BlockSequence,
    *,
    use_checkpointing: bool,
) -> Tensor:
    """Run residual blocks with scheduled motion correspondence."""
    for index, block in enumerate(sequence.blocks):
        inputs = (x, zq) if zq is not None else (x,)
        x = forward_with_checkpointing(block, *inputs, use_checkpointing=use_checkpointing)
        block_number = index + 1
        if sequence.motion_blocks is not None and block_number in sequence.motion_after_blocks:
            motion_index = sequence.motion_after_blocks.index(block_number)
            x = forward_with_checkpointing(
                sequence.motion_blocks[motion_index],
                x,
                # NATTEN merge_attentions custom autograd is incompatible with
                # PyTorch non-reentrant activation checkpointing.
                use_checkpointing=False,
            )
    return x
