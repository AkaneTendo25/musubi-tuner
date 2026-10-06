# Factory functions for building upsampler models.

from __future__ import annotations

from typing import TYPE_CHECKING

from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.config import MultiScaleModelConfig
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.model import ConvLatentUpsampler
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.motion_correspondence import MotionCorrespondenceSpec
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.multi_scale_components import MotionAttentionSpec
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.multi_scale_model import MultiScaleUpsampler

if TYPE_CHECKING:
    from torch import nn

    from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.config import FlatModelConfig


def build_upsampler(config: FlatModelConfig | MultiScaleModelConfig) -> nn.Module:
    """Create an upsampler module from config.

    Dispatches on config type: ``FlatModelConfig`` builds a
    ``ConvLatentUpsampler``; ``MultiScaleModelConfig`` builds a
    ``MultiScaleUpsampler``.

    Args:
        config: Model configuration (flat or multi-scale).

    Returns:
        Initialized upsampler module.
    """
    if isinstance(config, MultiScaleModelConfig):
        return _build_multi_scale(config)
    return _build_flat(config)


def _build_flat(config: FlatModelConfig) -> ConvLatentUpsampler:
    """Build a flat (single-stage) ConvLatentUpsampler from config."""
    return ConvLatentUpsampler(
        in_channels=config.in_channels,
        hidden_channels=config.hidden_channels,
        bottleneck_channels=config.bottleneck_channels,
        num_pre_residual_blocks=config.num_pre_residual_blocks,
        num_residual_blocks=config.num_residual_blocks,
        upscale_factor=config.upscale_factor,
        input_skip=config.input_skip,
        upsample_mode=config.upsample_mode,
        upsample_position=config.upsample_position,
        temporal_mix=config.temporal_mix,
        icnr=config.icnr,
        gradient_checkpointing=config.gradient_checkpointing,
        dims=config.dims,
        expand_ratio=config.expand_ratio,
        kernel_size=config.kernel_size,
        layer_scale_init=config.layer_scale_init,
        grn=config.grn,
        stochastic_depth_rate=config.stochastic_depth_rate,
        depthwise=config.depthwise,
        stem_channels=config.stem_channels,
        zq_dim=config.in_channels if config.modulated_norm else None,
        temporal_padding=config.temporal_padding,
        upsample_padding_mode=config.upsample_padding_mode,
    )


def _build_multi_scale(config: MultiScaleModelConfig) -> MultiScaleUpsampler:
    """Build a cascaded 2x+2x MultiScaleUpsampler from config."""
    motion_attention = build_motion_attention_spec(config)
    return MultiScaleUpsampler(
        in_channels=config.in_channels,
        hidden_channels=config.hidden_channels,
        bottleneck_channels=config.bottleneck_channels,
        num_pre_blocks=config.num_pre_blocks,
        num_mid_blocks=config.num_mid_blocks,
        num_post_blocks=config.num_post_blocks,
        input_skip=config.input_skip,
        upsample_mode=config.upsample_mode,
        temporal_mix=config.temporal_mix,
        icnr=config.icnr,
        gradient_checkpointing=config.gradient_checkpointing,
        dims=config.dims,
        expand_ratio=config.expand_ratio,
        kernel_size=config.kernel_size,
        layer_scale_init=config.layer_scale_init,
        grn=config.grn,
        stochastic_depth_rate=config.stochastic_depth_rate,
        depthwise=config.depthwise,
        motion_attention=motion_attention,
        zq_dim=config.in_channels if config.modulated_norm else None,
        bare_stem=config.bare_stem,
        modulated_output_proj=config.modulated_output_proj,
        global_skip=config.global_skip,
        enable_x2_entry=config.enable_x2_entry,
        x2_adapter_blocks=config.x2_adapter_blocks,
        x2_tail_mode=config.x2_tail_mode,
        x2_adapter_sources=config.x2_adapter_sources,
        x2_finisher=config.x2_finisher,
        stage_channels=config.stage_channels,
        temporal_padding=config.temporal_padding,
        upsample_padding_mode=config.upsample_padding_mode,
    )


def build_motion_attention_spec(config: MultiScaleModelConfig) -> MotionAttentionSpec | None:
    """Resolve the nested motion-attention config into a construction spec."""
    motion = config.motion_attention
    if motion is None:
        return None
    return MotionAttentionSpec(
        after_mid_blocks=motion.after_mid_blocks,
        block=MotionCorrespondenceSpec(
            channels=config.hidden_channels,
            spatial_kernel_size=motion.spatial_kernel_size,
            temporal_offsets=motion.temporal_offsets,
            num_heads=motion.num_heads,
            head_dim=config.hidden_channels // motion.num_heads,
            backend=motion.backend,
            natten_backend=motion.natten_backend,
            merge_compile=motion.merge_compile,
        ),
    )
