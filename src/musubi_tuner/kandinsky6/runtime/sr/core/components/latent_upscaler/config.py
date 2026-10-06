# Configuration models for the latent upscaler.

from __future__ import annotations

from typing import Annotated, Literal

import pydantic
from pydantic import BaseModel, Field

from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.conv_ops import (  # noqa: TC001 — Pydantic needs runtime access
    TemporalPadding,
    UpsamplePaddingMode,
)

UpsampleMode = Literal["pixel_shuffle", "bilinear", "pxs_v2", "pxs_v2_hybrid"]
UpsamplePosition = Literal["before_projection", "after_projection"]
X2TailMode = Literal["shared", "scale_specific_norm", "private", "private_full"]
X2FinisherMode = Literal["none", "linear", "pxs_residual"]
MotionAttentionBackend = Literal["sdpa", "natten"]
NattenBackend = Literal["cutlass-fna", "hopper-fna", "flex-fna"]
SPATIOTEMPORAL_DIMS = 3
MIN_ROPE_HEAD_DIM = 6


class _BaseModelConfig(BaseModel):
    """Common fields shared by all upsampler architectures."""

    model_config = pydantic.ConfigDict(extra="forbid")

    in_channels: int = 16
    upscale_factor: int = 2
    upsample_mode: UpsampleMode = "pixel_shuffle"
    upsample_position: UpsamplePosition = "after_projection"
    temporal_mix: bool = True
    icnr: bool = False
    dims: int = 2
    input_skip: bool = True
    gradient_checkpointing: bool = False
    modulated_norm: bool = False
    temporal_padding: TemporalPadding = "zeros"  # "replicate" pads T both sides, "causal" pads the past only (K-VAE)
    upsample_padding_mode: UpsamplePaddingMode = "reflect"  # "zeros" matches the production K-VAE upsample


class FlatModelConfig(_BaseModelConfig):
    """Configuration for ConvLatentUpsampler (flat architecture)."""

    architecture: Literal["flat"] = "flat"
    hidden_channels: int = 64
    stem_channels: int | None = None
    num_residual_blocks: int = 16
    num_pre_residual_blocks: int = 0
    bottleneck_channels: int | None = None
    expand_ratio: int = 4
    kernel_size: int = 3
    layer_scale_init: float | None = None
    grn: bool = False
    stochastic_depth_rate: float = 0.0
    depthwise: bool = False


class MotionAttentionConfig(BaseModel):
    """Configuration for cross-frame spatial correspondence blocks."""

    model_config = pydantic.ConfigDict(extra="forbid")

    spatial_kernel_size: int = Field(default=13, ge=3)
    temporal_offsets: tuple[int, ...] = (-3, -2, -1, 1, 2, 3)
    after_mid_blocks: tuple[int, ...] = (4, 8, 12)
    num_heads: int = Field(default=3, gt=0)
    backend: MotionAttentionBackend = "natten"
    natten_backend: NattenBackend = "hopper-fna"
    merge_compile: bool = False

    @pydantic.model_validator(mode="after")
    def validate_attention_geometry(self) -> MotionAttentionConfig:
        """Require centered spatial windows and symmetric temporal offsets."""
        if self.spatial_kernel_size % 2 == 0:
            msg = "motion spatial_kernel_size must be odd"
            raise ValueError(msg)
        if not self.temporal_offsets or 0 in self.temporal_offsets:
            msg = "motion temporal_offsets must be non-empty and exclude 0"
            raise ValueError(msg)
        if tuple(sorted(set(self.temporal_offsets))) != self.temporal_offsets:
            msg = "motion temporal_offsets must be unique and sorted"
            raise ValueError(msg)
        if set(self.temporal_offsets) != {-offset for offset in self.temporal_offsets}:
            msg = "motion temporal_offsets must be symmetric"
            raise ValueError(msg)
        if (
            not self.after_mid_blocks
            or any(block_number <= 0 for block_number in self.after_mid_blocks)
            or tuple(sorted(set(self.after_mid_blocks))) != self.after_mid_blocks
        ):
            msg = "motion after_mid_blocks must contain positive, unique, sorted block numbers"
            raise ValueError(msg)
        return self


class MultiScaleModelConfig(_BaseModelConfig):
    """Configuration for MultiScaleUpsampler (cascaded 2x+2x architecture).

    Decomposes a 4x upscale into two supervised 2x stages: coarse structure
    at 2x, fine detail at 4x.  Each stage gets direct loss supervision.

    Attributes:
        hidden_channels: Internal width of residual blocks.
        num_pre_blocks: Residual blocks at 1x (before first upsample).
        num_mid_blocks: Residual blocks at 2x (between upsamples).
        num_post_blocks: Residual blocks at 4x (after second upsample).
        bottleneck_channels: Mid-channel width for ResidualBlocks (None = hidden * expand_ratio).
        expand_ratio: Inverted bottleneck expansion when bottleneck_channels is None.
        kernel_size: Kernel size for depthwise convolutions.
        layer_scale_init: Initial LayerScale gamma. None disables LayerScale.
        grn: Insert Global Response Normalization in residual blocks.
        stochastic_depth_rate: Max drop rate, linearly scheduled across all stages.
        depthwise: Use depthwise separable convolutions.
        motion_attention: Optional cross-frame spatial correspondence stack.
        bare_stem: Reduce input projections to a single conv (no norm/activation).
        modulated_output_proj: Use a zq-conditioned ModulatedRMSNorm in ``output_proj``.
        global_skip: Add a nearest-upsampled input base to every output and
            zero-init the output heads, so the model starts as a nearest upsampler
            and learns only the residual correction.
        loss_weight_2x: Training loss weight for 2x intermediate output.
        loss_weight_4x: Training loss weight for 4x final output.
        enable_x2_entry: Build the ``mid_input_proj`` submodule and allow
            ``forward(entry="x2")`` — a second entry that injects an already-2x LQ
            latent after ``mid_blocks`` for a single 2x upscale through the shared
            second stage (LAY-462). Off by default so the x4 module set and its
            checkpoints stay unchanged.
        x2_adapter_blocks: Number of x2-only residual blocks before the second stage.
        x2_tail_mode: Amount of scale-specific capacity in the second stage;
            ``private_full`` additionally privatizes ``mid_blocks`` and runs them
            in the x2 path ahead of the second-stage upsample.
        x2_adapter_sources: pre_blocks indices warm-copied into the adapter (one
            per block); None keeps the zero-init identity adapter.
        x2_finisher: upsample_1 conv pair applied at the adapter grid without the
            nearest-2x resize: "linear" clones the 1x1 only, "pxs_residual" adds
            the spatial conv with its residual add.
        stage_channels: Per-grid widths (1x, 2x, 4x) for the K-VAE-shaped pyramid; None keeps constant hidden_channels.
    """

    architecture: Literal["multi_scale"] = "multi_scale"
    upscale_factor: int = 4
    hidden_channels: int = 64
    num_pre_blocks: int = 2
    num_mid_blocks: int = 6
    num_post_blocks: int = 8
    bottleneck_channels: int | None = None
    expand_ratio: int = 4
    kernel_size: int = 3
    layer_scale_init: float | None = None
    grn: bool = False
    stochastic_depth_rate: float = 0.0
    depthwise: bool = False
    motion_attention: MotionAttentionConfig | None = None
    bare_stem: bool = False
    modulated_output_proj: bool = False
    global_skip: bool = False
    loss_weight_2x: float = 0.5
    loss_weight_4x: float = 1.0
    enable_x2_entry: bool = False
    x2_adapter_blocks: int = Field(default=0, ge=0)
    x2_tail_mode: X2TailMode = "shared"
    x2_adapter_sources: tuple[int, ...] | None = None
    x2_finisher: X2FinisherMode = "none"
    stage_channels: tuple[int, int, int] | None = None

    @pydantic.model_validator(mode="after")
    def validate_upscale_factor(self) -> MultiScaleModelConfig:
        """Multi-scale architecture only supports 4x upscale (two 2x stages)."""
        expected = 4
        if self.upscale_factor != expected:
            msg = f"multi_scale architecture requires upscale_factor={expected}, got {self.upscale_factor}"
            raise ValueError(msg)
        return self

    @pydantic.model_validator(mode="after")
    def validate_redesign_flags(self) -> MultiScaleModelConfig:
        """Reject redesign-flag combinations the model cannot execute."""
        if self.modulated_output_proj and not self.modulated_norm:
            msg = "modulated_output_proj requires modulated_norm=true"
            raise ValueError(msg)
        if self.enable_x2_entry and self.global_skip:
            msg = "global_skip is x4-only and incompatible with enable_x2_entry"
            raise ValueError(msg)
        if self.enable_x2_entry and self.modulated_output_proj and self.x2_tail_mode != "private_full":
            msg = "modulated_output_proj with enable_x2_entry requires x2_tail_mode='private_full'"
            raise ValueError(msg)
        return self

    @pydantic.model_validator(mode="after")
    def validate_stage_channels(self) -> MultiScaleModelConfig:
        """Pyramid widths must agree with hidden_channels and exclude width-coupled features."""
        if self.stage_channels is None:
            return self
        if self.hidden_channels != self.stage_channels[0]:
            msg = f"hidden_channels ({self.hidden_channels}) must equal stage_channels[0] ({self.stage_channels[0]})"
            raise ValueError(msg)
        if self.enable_x2_entry and self.x2_tail_mode != "private_full":
            msg = "stage_channels with enable_x2_entry requires x2_tail_mode='private_full'"
            raise ValueError(msg)
        if self.motion_attention is not None:
            msg = "stage_channels is incompatible with motion_attention (motion blocks are built at hidden_channels)"
            raise ValueError(msg)
        if self.input_skip:
            msg = "stage_channels is incompatible with input_skip (channel-repeat skip assumes one width)"
            raise ValueError(msg)
        if self.bottleneck_channels is not None:
            msg = "stage_channels derives per-stage mid widths from expand_ratio; bottleneck_channels is incompatible"
            raise ValueError(msg)
        if self.num_mid_blocks < 1 or self.num_post_blocks < 1:
            msg = "stage_channels needs at least one mid and one post block to host the width transitions"
            raise ValueError(msg)
        return self

    @pydantic.model_validator(mode="after")
    def validate_x2_adaptation(self) -> MultiScaleModelConfig:
        """Reject x2 adaptation settings that cannot be executed."""
        has_adaptation = (
            self.x2_adapter_blocks > 0
            or self.x2_tail_mode != "shared"
            or self.x2_adapter_sources is not None
            or self.x2_finisher != "none"
        )
        if has_adaptation and not self.enable_x2_entry:
            msg = "x2 adaptation requires enable_x2_entry=true"
            raise ValueError(msg)
        if self.x2_tail_mode != "shared" and self.x2_adapter_blocks == 0:
            msg = f"x2_tail_mode={self.x2_tail_mode!r} requires x2_adapter_blocks > 0"
            raise ValueError(msg)
        if self.x2_tail_mode == "scale_specific_norm" and not self.modulated_norm:
            msg = "scale_specific_norm requires modulated_norm=true"
            raise ValueError(msg)
        if self.x2_tail_mode == "scale_specific_norm" and self.depthwise:
            msg = "scale_specific_norm does not support depthwise residual blocks"
            raise ValueError(msg)
        if self.x2_adapter_sources is not None and len(self.x2_adapter_sources) != self.x2_adapter_blocks:
            msg = (
                f"x2_adapter_sources length ({len(self.x2_adapter_sources)}) must equal "
                f"x2_adapter_blocks ({self.x2_adapter_blocks})"
            )
            raise ValueError(msg)
        if self.x2_adapter_sources is not None and any(
            not 0 <= index < self.num_pre_blocks for index in self.x2_adapter_sources
        ):
            msg = f"x2_adapter_sources indices must be within [0, num_pre_blocks={self.num_pre_blocks})"
            raise ValueError(msg)
        if self.x2_finisher != "none" and self.upsample_mode != "pxs_v2":
            msg = "x2_finisher clones the pxs_v2 upsample convs and requires upsample_mode='pxs_v2'"
            raise ValueError(msg)
        return self

    @pydantic.model_validator(mode="after")
    def validate_motion_attention(self) -> MultiScaleModelConfig:
        """Reject motion-attention configurations incompatible with this model."""
        motion = self.motion_attention
        if motion is None:
            return self
        if self.dims != SPATIOTEMPORAL_DIMS:
            msg = "motion_attention requires dims=3"
            raise ValueError(msg)
        if self.enable_x2_entry:
            msg = "motion_attention Round 23 is x4-only and cannot enable the x2 entry"
            raise ValueError(msg)
        if motion.after_mid_blocks[-1] > self.num_mid_blocks:
            msg = "motion after_mid_blocks must not exceed num_mid_blocks"
            raise ValueError(msg)
        if self.hidden_channels % motion.num_heads != 0:
            msg = "hidden_channels must be divisible by motion num_heads"
            raise ValueError(msg)
        head_dim = self.hidden_channels // motion.num_heads
        if head_dim < MIN_ROPE_HEAD_DIM or head_dim % 2 != 0:
            msg = "motion attention head_dim must be even and at least 6 for axial 3D RoPE"
            raise ValueError(msg)
        return self


ModelConfig = Annotated[
    FlatModelConfig | MultiScaleModelConfig,
    Field(discriminator="architecture"),
]
