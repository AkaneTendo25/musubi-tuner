# Multi-scale cascaded upsampler: two 2x stages with intermediate supervision.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from einops import rearrange
from torch import Tensor, nn

from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.multi_scale_components import (
    STAGE_FACTOR,
    BlockSequence,
    MotionAttentionSpec,
    OutputHead,
    ResidualStackSpec,
    UpsampleSpec,
    apply_block_sequence,
    apply_output_head,
    build_motion_attention_stack,
    build_output_head,
    build_residual_stack,
    build_stage_upsample,
    build_stem,
)
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.upsample_ops import DIMS_3, spatial_nearest_2x
from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.x2_branch import (
    SharedSecondStage,
    X2Branch,
    X2Finisher,
    X2FinisherMode,
    X2TailMode,
)

if TYPE_CHECKING:
    from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.config import UpsampleMode
    from musubi_tuner.kandinsky6.runtime.sr.core.components.latent_upscaler.model.conv_ops import TemporalPadding, UpsamplePaddingMode

# The two projections back to latent space. They are scale-specific, so a warm
# start from another network cannot supply them and a warm-started run may want
# to fit them before letting gradients reach the rest.
HEAD_PREFIXES = ("output_proj.", "mid_output_head.")


class MultiScaleUpsampler(nn.Module):
    """Cascaded 2x+2x VAE-latent upsampler with an optional x2 entry.

    The x4 route keeps the original two-stage architecture. The x2 route enters
    after ``mid_blocks`` through a private stem, optional residual adapter, and
    a shared, scale-normalized, or private second stage. ``x2_tail_mode``
    ``'private_full'`` instead routes through the branch's own warm copies of
    ``mid_blocks`` and the second stage, behind an optional finisher cloned
    from ``upsample_1``.
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 64,
        bottleneck_channels: int | None = None,
        num_pre_blocks: int = 2,
        num_mid_blocks: int = 6,
        num_post_blocks: int = 8,
        *,
        input_skip: bool = True,
        upsample_mode: UpsampleMode = "pixel_shuffle",
        temporal_mix: bool = True,
        icnr: bool = False,
        gradient_checkpointing: bool = False,
        dims: int = 3,
        expand_ratio: int = 4,
        kernel_size: int = 3,
        layer_scale_init: float | None = None,
        grn: bool = False,
        stochastic_depth_rate: float = 0.0,
        depthwise: bool = False,
        motion_attention: MotionAttentionSpec | None = None,
        zq_dim: int | None = None,
        bare_stem: bool = False,
        modulated_output_proj: bool = False,
        global_skip: bool = False,
        enable_x2_entry: bool = False,
        x2_adapter_blocks: int = 0,
        x2_tail_mode: X2TailMode = "shared",
        x2_adapter_sources: tuple[int, ...] | None = None,
        x2_finisher: X2FinisherMode = "none",
        stage_channels: tuple[int, int, int] | None = None,
        temporal_padding: TemporalPadding = "zeros",
        upsample_padding_mode: UpsamplePaddingMode = "reflect",
    ) -> None:
        """Initialize MultiScaleUpsampler with the given architecture parameters."""
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.upscale_factor = 4
        self.input_skip = input_skip
        self.gradient_checkpointing = gradient_checkpointing
        self.dims = dims
        self.zq_dim = zq_dim
        self.global_skip = global_skip
        self.enable_x2_entry = enable_x2_entry
        self.x2_tail_mode = x2_tail_mode
        self.x2_adapter_sources = x2_adapter_sources

        if modulated_output_proj and zq_dim is None:
            msg = "modulated_output_proj requires zq_dim (modulated_norm)"
            raise ValueError(msg)
        if enable_x2_entry and global_skip:
            msg = "global_skip does not support the x2 entry"
            raise ValueError(msg)
        if enable_x2_entry and modulated_output_proj and x2_tail_mode != "private_full":
            msg = "modulated_output_proj with the x2 entry requires x2_tail_mode='private_full'"
            raise ValueError(msg)
        if x2_adapter_sources is not None and len(x2_adapter_sources) != x2_adapter_blocks:
            msg = "x2_adapter_sources must list one pre_blocks index per adapter block"
            raise ValueError(msg)
        if x2_adapter_sources is not None and any(not 0 <= index < num_pre_blocks for index in x2_adapter_sources):
            msg = "x2_adapter_sources indices must be within [0, num_pre_blocks)"
            raise ValueError(msg)
        if stage_channels is not None:
            if enable_x2_entry and x2_tail_mode != "private_full":
                msg = "stage_channels with enable_x2_entry requires x2_tail_mode='private_full'"
                raise ValueError(msg)
            if motion_attention is not None:
                msg = "stage_channels is incompatible with motion_attention"
                raise ValueError(msg)
            if input_skip:
                msg = "stage_channels is incompatible with input_skip"
                raise ValueError(msg)
            if bottleneck_channels is not None:
                msg = "stage_channels is incompatible with bottleneck_channels"
                raise ValueError(msg)
            if hidden_channels != stage_channels[0]:
                msg = f"hidden_channels ({hidden_channels}) must equal stage_channels[0] ({stage_channels[0]})"
                raise ValueError(msg)
        self.stage_channels = stage_channels
        w1, w2, w3 = stage_channels if stage_channels is not None else (hidden_channels,) * 3
        self.motion_after_mid_blocks = motion_attention.after_mid_blocks if motion_attention is not None else ()

        if input_skip and hidden_channels % in_channels != 0:
            msg = f"hidden_channels ({hidden_channels}) must be divisible by in_channels ({in_channels}) for input_skip"
            raise ValueError(msg)
        if motion_attention is not None and dims != 3:
            msg = "motion_attention requires dims=3"
            raise ValueError(msg)
        if motion_attention is not None and enable_x2_entry:
            msg = "motion_attention Round 23 is x4-only and cannot enable the x2 entry"
            raise ValueError(msg)
        if motion_attention is not None and not motion_attention.after_mid_blocks:
            msg = "motion_attention requires at least one mid-stage placement"
            raise ValueError(msg)
        if motion_attention is not None and motion_attention.after_mid_blocks[-1] > num_mid_blocks:
            msg = "motion attention placement exceeds the mid residual stack"
            raise ValueError(msg)

        # Projection layers
        self.input_proj = build_stem(in_channels, w1, dims, bare=bare_stem, temporal_padding=temporal_padding)
        self.mid_output_head = build_output_head(w2, in_channels, dims, zq_dim=None, temporal_padding=temporal_padding)
        self.output_proj = OutputHead(
            w3,
            in_channels,
            dims,
            zq_dim=zq_dim if modulated_output_proj else None,
            temporal_padding=temporal_padding,
        )

        # Two 2x upsample stages
        upsample_spec = UpsampleSpec(
            dims=dims,
            mode=upsample_mode,
            temporal_mix=temporal_mix,
            icnr=icnr,
            temporal_padding=temporal_padding,
            padding_mode=upsample_padding_mode,
        )
        self.upsample_1 = build_stage_upsample(w1, upsample_spec)
        self.upsample_2 = build_stage_upsample(w2, upsample_spec)

        # Residual stacks with linear stochastic depth across all stages; the
        # first block of a stage narrows from the previous stage width.
        total_blocks = num_pre_blocks + num_mid_blocks + num_post_blocks

        def stage_spec(channels: int, first_in: int | None) -> ResidualStackSpec:
            return ResidualStackSpec(
                channels=channels,
                mid_channels=bottleneck_channels if bottleneck_channels is not None else channels * expand_ratio,
                total_blocks=total_blocks,
                stochastic_depth_rate=stochastic_depth_rate,
                dims=dims,
                layer_scale_init=layer_scale_init,
                grn=grn,
                depthwise=depthwise,
                kernel_size=kernel_size,
                zq_dim=zq_dim,
                in_channels=first_in,
                temporal_padding=temporal_padding,
            )

        self.pre_blocks = build_residual_stack(num_pre_blocks, 0, stage_spec(w1, None))
        self.mid_blocks = build_residual_stack(num_mid_blocks, num_pre_blocks, stage_spec(w2, w1 if w1 != w2 else None))
        self.post_blocks = build_residual_stack(
            num_post_blocks, num_pre_blocks + num_mid_blocks, stage_spec(w3, w2 if w2 != w3 else None)
        )

        self.mid_motion_blocks: nn.ModuleList | None = None
        if motion_attention is not None:
            self.mid_motion_blocks = build_motion_attention_stack(motion_attention)

        # Build x2-exclusive modules after the complete x4 path so enabling an x2
        # entry does not perturb the base-path RNG initialization.
        self.mid_input_proj: nn.Sequential | None = None
        self.x2_branch: X2Branch | None = None
        if enable_x2_entry:
            self.mid_input_proj = build_stem(
                in_channels, hidden_channels, dims, bare=bare_stem, temporal_padding=temporal_padding
            )
            adapter_spec = stage_spec(w1, None).model_copy(
                update={
                    "total_blocks": max(x2_adapter_blocks, 1),
                    "stochastic_depth_rate": 0.0,
                }
            )
            adapter = build_residual_stack(x2_adapter_blocks, 0, adapter_spec)
            for block in adapter:
                block.zero_init_residual()  # type: ignore[attr-defined]
            finisher = X2Finisher(self.upsample_1, x2_finisher) if x2_finisher != "none" else None
            self.x2_branch = X2Branch(
                adapter,
                x2_tail_mode,
                self.shared_second_stage(),
                dims=dims,
                finisher=finisher,
            )

        if global_skip:
            # Zero heads so step 0 reproduces the nearest base exactly at both scales.
            for head in (self.mid_output_head, self.output_proj):
                final_conv = head.conv if isinstance(head, OutputHead) else head[-1]
                nn.init.zeros_(final_conv.weight)  # type: ignore[arg-type]
                nn.init.zeros_(final_conv.bias)  # type: ignore[arg-type]

    @property
    def last_layer_weight(self) -> torch.Tensor:
        """Weight of the final 4x-output conv (used by adaptive GAN-weight balancing).

        The GAN-loss runs only on the 4x output, so the relevant last layer is
        ``output_proj.conv`` rather than the 2x ``mid_output_head``.
        """
        return self.output_proj.conv.weight

    def shared_second_stage(self) -> SharedSecondStage:
        """Return a non-registering view of the e115-compatible second stage."""
        return SharedSecondStage(
            upsample=self.upsample_2,
            blocks=self.post_blocks,
            output_proj=self.output_proj,
            mid_blocks=self.mid_blocks,
        )

    @staticmethod
    def is_x2_state_key(key: str) -> bool:
        """Return whether a state key belongs exclusively to the x2 path."""
        return key.startswith(("mid_input_proj.", "x2_branch."))

    def initialize_x2_from_x4(self) -> None:
        """Initialize x2-exclusive modules from the currently loaded x4 path."""
        if self.mid_input_proj is None or self.x2_branch is None:
            msg = "x2 initialization requires enable_x2_entry=True"
            raise ValueError(msg)
        self.mid_input_proj.load_state_dict(self.input_proj.state_dict())
        self.x2_branch.initialize_from_x4(self.shared_second_stage())
        if self.x2_adapter_sources is not None:
            for block, source_index in zip(self.x2_branch.adapter, self.x2_adapter_sources, strict=True):
                block.load_state_dict(self.pre_blocks[source_index].state_dict())
        if self.x2_branch.finisher is not None:
            self.x2_branch.finisher.copy_from(self.upsample_1)

    def configure_x2_finetuning(self) -> None:
        """Freeze the e115 path and expose only x2-exclusive parameters."""
        if self.mid_input_proj is None or self.x2_branch is None:
            msg = "x2 fine-tuning requires enable_x2_entry=True"
            raise ValueError(msg)
        self.requires_grad_(requires_grad=False)
        self.mid_input_proj.requires_grad_(requires_grad=True)
        self.x2_branch.requires_grad_(requires_grad=True)

    def _project_with_skip(self, x: Tensor, proj: nn.Module) -> Tensor:
        """Apply a projection and optionally add the channel-repeated input skip."""
        projected = proj(x)
        if self.input_skip:
            repeats = projected.shape[1] // x.shape[1]
            projected = projected + x.repeat_interleave(repeats, dim=1)
        return projected

    def _head_x4(self, x: Tensor, zq: Tensor | None, *, use_ckpt: bool) -> Tensor:
        """x4 entry: ``input_proj -> input_skip -> pre_blocks -> upsample_1`` (-> 2x grid)."""
        x = self._project_with_skip(x, self.input_proj)
        x = apply_block_sequence(
            x,
            zq,
            BlockSequence(self.pre_blocks),
            use_checkpointing=use_ckpt,
        )
        return self.upsample_1(x)

    def _head_x2(self, x: Tensor, zq: Tensor | None, *, use_ckpt: bool) -> Tensor:
        """Project a genuine x2 latent and apply the scale-specific adapter."""
        if self.mid_input_proj is None or self.x2_branch is None:
            msg = "x2 head requires enable_x2_entry=True"
            raise ValueError(msg)
        x = self._project_with_skip(x, self.mid_input_proj)
        return self.x2_branch.apply_adapter(x, zq, use_checkpointing=use_ckpt)

    def _tail(
        self,
        x: Tensor,
        zq: Tensor | None,
        b: int,
        t: int,
        *,
        use_ckpt: bool,
        return_intermediates: bool,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """x4 continuation: ``mid_blocks -> [2x head] -> second stage``."""
        x = apply_block_sequence(
            x,
            zq,
            BlockSequence(self.mid_blocks, self.mid_motion_blocks, self.motion_after_mid_blocks),
            use_checkpointing=use_ckpt,
        )

        # 2x supervision branch (only when the caller will use it)
        z_2x = None
        if return_intermediates:
            z_2x_inner = self.mid_output_head(x)
            z_2x = rearrange(z_2x_inner, "(b t) c h w -> b c t h w", b=b, t=t) if self.dims == 2 else z_2x_inner

        z_4x = self._second_stage(x, zq, b, t, use_ckpt=use_ckpt)
        if return_intermediates:
            return {"2x": z_2x, "4x": z_4x}  # type: ignore[return-value]
        return z_4x

    def _second_stage(self, x: Tensor, zq: Tensor | None, b: int, t: int, *, use_ckpt: bool) -> Tensor:
        """Shared second stage: ``upsample_2 -> post_blocks -> output_proj`` — 2x of the current grid."""
        x = self.upsample_2(x)
        x = apply_block_sequence(x, zq, BlockSequence(self.post_blocks), use_checkpointing=use_ckpt)
        x = apply_output_head(self.output_proj, x, zq)
        return rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t) if self.dims == 2 else x

    def forward(
        self,
        z: torch.Tensor,
        *,
        entry: Literal["x4", "x2"] = "x4",
        return_intermediates: bool | None = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Upsample through the x4 cascade or the configured x2 entry."""
        if entry not in ("x4", "x2"):
            msg = f"entry must be 'x4' or 'x2', got {entry!r}"
            raise ValueError(msg)
        if entry == "x2" and not self.enable_x2_entry:
            msg = "forward(entry='x2') requires the model to be built with enable_x2_entry=True"
            raise ValueError(msg)
        if entry == "x2" and return_intermediates:
            msg = "return_intermediates=True is not supported for entry='x2' — the x2 path has no 2x sub-target"
            raise ValueError(msg)
        if return_intermediates is None:
            return_intermediates = self.training

        b, _, t, _, _ = z.shape
        x = rearrange(z, "b c t h w -> (b t) c h w") if self.dims == 2 else z
        zq = x if self.zq_dim is not None else None
        use_ckpt = self.training and self.gradient_checkpointing

        if entry == "x2":
            if self.x2_branch is None:
                msg = "x2 branch requires enable_x2_entry=True"
                raise ValueError(msg)
            x = self._head_x2(x, zq, use_ckpt=use_ckpt)
            return self.x2_branch.forward_second_stage(
                x,
                zq,
                b,
                t,
                self.shared_second_stage(),
                use_checkpointing=use_ckpt,
            )
        x = self._head_x4(x, zq, use_ckpt=use_ckpt)
        out = self._tail(x, zq, b, t, use_ckpt=use_ckpt, return_intermediates=return_intermediates)
        if not self.global_skip:
            return out
        return self._add_nearest_base(out, z)

    def _add_nearest_base(
        self,
        out: dict[str, torch.Tensor] | torch.Tensor,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """Add the parameter-free nearest base of the input latent to each output."""
        if isinstance(out, dict):
            return {
                "2x": out["2x"] + spatial_nearest_2x(z, DIMS_3, STAGE_FACTOR),
                "4x": out["4x"] + spatial_nearest_2x(z, DIMS_3, self.upscale_factor),
            }
        return out + spatial_nearest_2x(z, DIMS_3, self.upscale_factor)
