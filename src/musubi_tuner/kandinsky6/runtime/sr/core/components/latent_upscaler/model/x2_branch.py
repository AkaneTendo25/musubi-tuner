# Scale-specific x2 adaptation modules for the cascaded latent upscaler.

from __future__ import annotations

import copy
from functools import partial
from typing import TYPE_CHECKING, Literal, NamedTuple

from einops import rearrange
from torch import Tensor, nn

from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.multi_scale_components import (
    OutputHead,
    apply_output_head,
)
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.runtime import forward_with_checkpointing
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.upsample_ops import PXSv2UpsampleND

if TYPE_CHECKING:
    from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.blocks import ResidualBlock, RMSNorm

X2TailMode = Literal["shared", "scale_specific_norm", "private", "private_full"]
X2FinisherMode = Literal["none", "linear", "pxs_residual"]


class SharedSecondStage(NamedTuple):
    """Non-registering view of the x4 second-stage modules."""

    upsample: nn.Module
    blocks: nn.Sequential
    output_proj: OutputHead
    mid_blocks: nn.Sequential | None = None


class ScaleSpecificBlockNorms(nn.Module):
    """Private normalization copies for one shared residual block."""

    def __init__(self, block: ResidualBlock) -> None:
        """Copy the modulated norms from a non-depthwise residual block."""
        super().__init__()
        if block.zq_dim is None or block.depthwise:
            msg = "scale-specific norms require non-depthwise modulated residual blocks"
            raise ValueError(msg)
        self.norm1 = copy.deepcopy(block.norm1)
        self.norm2 = copy.deepcopy(block.norm2)

    def copy_from(self, block: ResidualBlock) -> None:
        """Reset the private norms from their shared x4 counterparts."""
        self.norm1.load_state_dict(block.norm1.state_dict())
        self.norm2.load_state_dict(block.norm2.state_dict())


def require_pxs_upsample(source: nn.Module) -> PXSv2UpsampleND:
    """Require a pxs_v2 upsample stage as the finisher donor."""
    if not isinstance(source, PXSv2UpsampleND):
        msg = f"x2_finisher requires a PXSv2UpsampleND source, got {type(source).__name__}"
        raise TypeError(msg)
    return source


class X2Finisher(nn.Module):
    """The upsample-stage conv pair applied at the adapter grid without the nearest-2x resize."""

    def __init__(self, source: nn.Module, mode: X2FinisherMode) -> None:
        """Clone the finisher convs from a pxs_v2 upsample stage."""
        super().__init__()
        if mode == "none":
            msg = "X2Finisher must not be built with x2_finisher='none'"
            raise ValueError(msg)
        donor = require_pxs_upsample(source)
        self.mode = mode
        self.linear = copy.deepcopy(donor.linear)
        self.spatial_conv = copy.deepcopy(donor.spatial_conv) if mode == "pxs_residual" else None

    def copy_from(self, source: nn.Module) -> None:
        """Reset the finisher convs from their x4 upsample counterparts."""
        donor = require_pxs_upsample(source)
        self.linear.load_state_dict(donor.linear.state_dict())
        if self.spatial_conv is not None:
            self.spatial_conv.load_state_dict(donor.spatial_conv.state_dict())

    def forward(self, x: Tensor) -> Tensor:
        """Apply the cloned convs on the unchanged grid."""
        if self.spatial_conv is not None:
            x = x + self.spatial_conv(x)
        return self.linear(x)


class X2Branch(nn.Module):
    """Modules exclusive to the frozen-e115 x2 adaptation path."""

    def __init__(
        self,
        adapter: nn.Sequential,
        tail_mode: X2TailMode,
        shared_stage: SharedSecondStage,
        *,
        dims: int,
        finisher: X2Finisher | None = None,
    ) -> None:
        """Build the selected scale-specific capacity around a shared stage."""
        super().__init__()
        self.adapter = adapter
        self.tail_mode = tail_mode
        self.dims = dims
        self.finisher = finisher

        self.post_norms: nn.ModuleList | None = None
        self.output_norm: RMSNorm | None = None
        self.mid_blocks: nn.Sequential | None = None
        self.upsample: nn.Module | None = None
        self.blocks: nn.Sequential | None = None
        self.output_proj: OutputHead | None = None

        if tail_mode == "scale_specific_norm":
            self.post_norms = nn.ModuleList([ScaleSpecificBlockNorms(block) for block in shared_stage.blocks])  # type: ignore
            self.output_norm = copy.deepcopy(shared_stage.output_proj.norm)  # type: ignore
        elif tail_mode in ("private", "private_full"):
            self.upsample = copy.deepcopy(shared_stage.upsample)
            self.blocks = copy.deepcopy(shared_stage.blocks)
            self.output_proj = copy.deepcopy(shared_stage.output_proj)
            if tail_mode == "private_full":
                if shared_stage.mid_blocks is None:
                    msg = "x2_tail_mode='private_full' requires the shared stage to expose mid_blocks"
                    raise ValueError(msg)
                self.mid_blocks = copy.deepcopy(shared_stage.mid_blocks)

    def initialize_from_x4(self, shared_stage: SharedSecondStage) -> None:
        """Copy scale-specific tail state from the loaded x4 baseline."""
        if self.tail_mode == "scale_specific_norm":
            if self.post_norms is None or self.output_norm is None:
                msg = "scale-specific normalization modules are not initialized"
                raise RuntimeError(msg)
            for norms, block in zip(self.post_norms, shared_stage.blocks, strict=True):
                norms.copy_from(block)
            self.output_norm.load_state_dict(shared_stage.output_proj.norm.state_dict())
        elif self.tail_mode in ("private", "private_full"):
            if self.upsample is None or self.blocks is None or self.output_proj is None:
                msg = "private x2 tail modules are not initialized"
                raise RuntimeError(msg)
            self.upsample.load_state_dict(shared_stage.upsample.state_dict())
            self.blocks.load_state_dict(shared_stage.blocks.state_dict())
            self.output_proj.load_state_dict(shared_stage.output_proj.state_dict())
            if self.mid_blocks is not None:
                if shared_stage.mid_blocks is None:
                    msg = "private mid blocks require the shared stage to expose mid_blocks"
                    raise RuntimeError(msg)
                self.mid_blocks.load_state_dict(shared_stage.mid_blocks.state_dict())

    def private_stage(self, shared_stage: SharedSecondStage) -> SharedSecondStage:
        """Select the shared stage or the registered private copy."""
        if self.tail_mode not in ("private", "private_full"):
            return shared_stage
        if self.upsample is None or self.blocks is None or self.output_proj is None:
            msg = "private x2 tail modules are not initialized"
            raise RuntimeError(msg)
        return SharedSecondStage(
            upsample=self.upsample,
            blocks=self.blocks,
            output_proj=self.output_proj,
        )

    def apply_post_blocks(
        self,
        x: Tensor,
        zq: Tensor | None,
        stage: SharedSecondStage,
        *,
        use_checkpointing: bool,
    ) -> Tensor:
        """Apply post blocks with shared or x2-specific normalization."""
        for index, block in enumerate(stage.blocks):
            if self.tail_mode == "scale_specific_norm":
                if zq is None or self.post_norms is None:
                    msg = "scale-specific normalization requires zq and private norms"
                    raise RuntimeError(msg)
                norms = self.post_norms[index]
                block_call = partial(block.forward_with_norms, norm1=norms.norm1, norm2=norms.norm2)  # type: ignore
                x = forward_with_checkpointing(block_call, x, zq, use_checkpointing=use_checkpointing)
            elif zq is None:
                x = forward_with_checkpointing(block, x, use_checkpointing=use_checkpointing)
            else:
                x = forward_with_checkpointing(block, x, zq, use_checkpointing=use_checkpointing)
        return x

    def apply_output_projection(self, x: Tensor, zq: Tensor | None, stage: SharedSecondStage) -> Tensor:
        """Project features with a shared or scale-specific final norm."""
        if self.tail_mode != "scale_specific_norm":
            return apply_output_head(stage.output_proj, x, zq)
        if self.output_norm is None:
            msg = "scale-specific output norm is not initialized"
            raise RuntimeError(msg)
        x = self.output_norm(x)
        x = stage.output_proj.act(x)
        return stage.output_proj.conv(x)

    def apply_block_stack(self, blocks: nn.Sequential, x: Tensor, zq: Tensor | None, *, use_checkpointing: bool) -> Tensor:
        """Run a residual stack with the branch's zq-threading convention."""
        for block in blocks:
            if zq is None:
                x = forward_with_checkpointing(block, x, use_checkpointing=use_checkpointing)
            else:
                x = forward_with_checkpointing(block, x, zq, use_checkpointing=use_checkpointing)
        return x

    def apply_adapter(self, x: Tensor, zq: Tensor | None, *, use_checkpointing: bool) -> Tensor:
        """Run the x2-only residual adapter."""
        return self.apply_block_stack(self.adapter, x, zq, use_checkpointing=use_checkpointing)

    def apply_mid_blocks(self, x: Tensor, zq: Tensor | None, *, use_checkpointing: bool) -> Tensor:
        """Run the private mid blocks ahead of the second-stage upsample (private_full only)."""
        if self.mid_blocks is None:
            return x
        return self.apply_block_stack(self.mid_blocks, x, zq, use_checkpointing=use_checkpointing)

    def forward_second_stage(
        self,
        x: Tensor,
        zq: Tensor | None,
        batch_size: int,
        frames: int,
        shared_stage: SharedSecondStage,
        *,
        use_checkpointing: bool,
    ) -> Tensor:
        """Run the selected x2 second stage without registering shared aliases."""
        stage = self.private_stage(shared_stage)
        if self.finisher is not None:
            x = self.finisher(x)
        x = self.apply_mid_blocks(x, zq, use_checkpointing=use_checkpointing)
        x = stage.upsample(x)
        x = self.apply_post_blocks(x, zq, stage, use_checkpointing=use_checkpointing)
        x = self.apply_output_projection(x, zq, stage)
        if self.dims == 2:
            return rearrange(x, "(b t) c h w -> b c t h w", b=batch_size, t=frames)
        return x
