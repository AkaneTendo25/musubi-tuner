# Neural network building blocks for the diffusion transformer.

from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch
from loguru import logger
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.attention.flex_attention import flex_attention

try:
    import flash_attn_interface  # pyright: ignore

    FA3 = True
except ImportError:
    FA3 = False

from musubi_tuner.kandinsky6.runtime.sr.core.components.model.kv_cache import assemble_cached_attention_inputs
from musubi_tuner.kandinsky6.runtime.sr.core.components.model.logger import ModelLogger
from musubi_tuner.kandinsky6.runtime.sr.core.components.model.utils import (
    FRACTAL_BLOCK_SIZE,
    framewise_causal_dense,
    get_freqs,
    nablaT_v2_doc,
    nablaT_v2_doc_mfcausal,
)


def flash_attn_funcs() -> tuple[Any, Any]:
    """Import the FA2 varlen entry points lazily, at first flash-path call.

    The nabla + text-free release models never reach a flash path (visual
    self-attention routes through flex_attention, and there is no text
    cross-attention), so flash-attn — the one dependency that needs a from-
    source build with nvcc — must not be required at import time.

    Returns:
        ``(flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func)``.

    Raises:
        ImportError: With an actionable message when a dense-flash or text
            cross-attention path is hit on a machine without flash-attn.
    """
    try:
        from flash_attn import (  # noqa: PLC0415  # pyright: ignore
            flash_attn_varlen_func,
            flash_attn_varlen_qkvpacked_func,
        )
    except ImportError as exc:
        msg = (
            "This attention path (dense flash self-attention or text cross-attention) requires "
            "flash-attn, which is not installed. Nabla text-free models do not need it; for the "
            "flash paths install the flash extra (needs nvcc): pip install 'kandinsky-6-sr[flash]'"
        )
        raise ImportError(msg) from exc
    return flash_attn_varlen_func, flash_attn_varlen_qkvpacked_func



def _ensure_nabla_compatible_flex_bwd_configs() -> None:
    """Guarantee flex_attention's backward keeps a 64-block-compatible config.

    ``nablaT_v2_doc`` builds a ``BLOCK_SIZE=64`` sparse mask. flex_attention's
    backward lowering drops every autotune config whose block sizes don't divide
    the mask block size (``SPARSE_BLOCK_SIZE % conf.block_* == 0``), checking each
    config's OWN block sizes — ``kernel_options`` cannot inject one. On torch
    builds whose head_dim=64 backward configs use ``BLOCK_N=128`` (e.g. the sm90
    default ``FlexBwDConfig(64, 128, 128, 64)``) every choice is filtered out and
    backward compilation aborts with ``NoValidChoicesError``. This is hit only by
    nabla TRAINING — forward-only validation never compiles the backward, which
    is why nabla validation worked but training did not.

    Harmless for non-nabla (flash) runs: ``flex_attention`` is only reached from
    ``attention_flex`` under ``sparse_params is not None``, so a flash run never
    lowers a flex backward and the wrapper installed here is never invoked — it
    only ever fires for a configured nabla run.

    Newer torch fixed this upstream by restricting the backward autotune blocks
    to ``{32, 64}``. This shim back-ports the guarantee: it wraps the inductor
    config provider and prepends an all-64 config whenever none of the returned
    configs divide 64. Fully defensive — any structural mismatch (renamed method,
    different config type) makes it a silent no-op rather than breaking startup,
    and it never removes existing choices.
    """
    try:
        from torch._inductor.choices import InductorChoices  # noqa: PLC0415 — optional, version-dependent internal
    except Exception:
        return

    original = getattr(InductorChoices, "get_flex_attention_bwd_configs", None)
    if original is None or getattr(original, "nabla_patched", False):
        return

    def divides_64(conf: Any) -> bool:
        try:
            return all(64 % b == 0 for b in (conf.block_m1, conf.block_n1, conf.block_m2, conf.block_n2))
        except Exception:
            # Unknown config shape — assume compatible so we don't interfere.
            return True

    def with_64_blocks(conf: Any) -> Any | None:
        fields = {"block_m1": 64, "block_n1": 64, "block_m2": 64, "block_n2": 64}
        try:
            return dataclasses.replace(conf, **fields)
        except Exception:
            try:
                return conf._replace(**fields)  # NamedTuple fallback
            except Exception:
                return None

    def patched(self: Any, head_dim: int, dtype: Any, device_type: str = "cuda") -> Any:
        configs = original(self, head_dim, dtype, device_type)
        try:
            if configs and not any(divides_64(c) for c in configs):
                compat = with_64_blocks(configs[0])
                if compat is not None:
                    configs = [compat, *configs]
        except Exception as exp:
            logger.warning("nabla flex bwd-config shim no-op (config introspection failed): {}", exp)
        return configs

    patched.nabla_patched = True  # type: ignore[attr-defined]
    try:
        InductorChoices.get_flex_attention_bwd_configs = patched  # type: ignore[method-assign]
    except Exception as exp:
        logger.warning("Could not install nabla flex bwd-config shim: {}", exp)


_ensure_nabla_compatible_flex_bwd_configs()

flex = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs", dynamic=True)

# Without FA3 the only varlen entry points available are FA2's
# ``flash_attn_varlen_*_func``, whose C++ binding declares ``max_seqlen_q``/
# ``max_seqlen_k`` as ``SymInt`` but does not support FakeTensors → dynamo
# can't trace through it.  Skip dynamo on the affected methods so the
# surrounding code in ``TransformerDecoderBlock.forward`` (norms, QKV
# projections, modulation, FFN) still gets ``@torch.compile``-fused while
# only the FA call runs in eager.  When FA3 is available the dedicated
# ``flash_attn_interface.flash_attn_varlen_func`` does register a custom op
# and traces cleanly, so this decorator becomes a no-op.
_disable_dynamo_if_fa2 = (lambda fn: fn) if FA3 else torch.compiler.disable


@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_scale_shift_norm(norm: nn.Module, x: Tensor, scale: Tensor, shift: Tensor, idx: Tensor) -> Tensor:
    """Apply adaptive normalization with indexed scale and shift."""
    return norm(x) * (scale.index_select(0, idx) + 1.0) + shift.index_select(0, idx)


@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_gate_sum(x: Tensor, out: Tensor, gate: Tensor, idx: Tensor) -> Tensor:
    """Add gated residual output to input using indexed gate values."""
    return x + gate.index_select(0, idx) * out


@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_scale_shift_norm_spatial(norm: nn.Module, x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """Apply adaptive normalization with per-token scale and shift.

    Unlike ``apply_scale_shift_norm`` which uses ``index_select`` for per-sample
    params, this function takes spatially-varying (per-token) params directly.

    Args:
        norm: Normalization layer (e.g. LayerNorm).
        x: Input tensor of shape ``(tokens, dim)``.
        scale: Per-token scale of shape ``(tokens, dim)``.
        shift: Per-token shift of shape ``(tokens, dim)``.
    """
    return norm(x) * (scale + 1.0) + shift


@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_gate_sum_spatial(x: Tensor, out: Tensor, gate: Tensor) -> Tensor:
    """Add gated residual output to input using per-token gate values.

    Args:
        x: Input tensor of shape ``(tokens, dim)``.
        out: Residual output of shape ``(tokens, dim)``.
        gate: Per-token gate of shape ``(tokens, dim)``.
    """
    return x + gate * out


@torch.autocast(device_type="cuda", enabled=False)
def apply_rotary(x: Tensor, rope: Tensor) -> Tensor:
    """Apply rotary position embeddings to input tensor."""
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
    x_out = rope[..., 0] * x_[..., 0] + rope[..., 1] * x_[..., 1]
    return x_out.reshape(*x.shape)


class _TimestepEmbedder(nn.Module):
    """MLP after the sinusoidal timestep. Names match Diffusers ``TimestepEmbedding``."""

    def __init__(self, model_dim: int, time_dim: int) -> None:
        """Initialize the two-layer timestep MLP.

        Args:
            model_dim: Width of the sinusoidal encoding.
            time_dim: Width of both linear layers.
        """
        super().__init__()
        self.linear_1 = nn.Linear(model_dim, time_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_dim, time_dim, bias=True)

    def forward(self, time_embed: Tensor) -> Tensor:
        """Project a sinusoidal timestep encoding."""
        return self.linear_2(self.act(self.linear_1(time_embed)))


class TimeEmbeddings(nn.Module):
    """Sinusoidal time step embeddings projected through a two-layer MLP."""

    def __init__(self, model_dim: int, time_dim: int, max_period: float = 10000.0) -> None:
        """Initialize time embeddings.

        Args:
            model_dim: Dimension of sinusoidal encoding. Must be even.
            time_dim: Output dimension after MLP projection.
            max_period: Maximum period for frequency computation.
        """
        super().__init__()
        if model_dim % 2 != 0:
            msg = "model_dim must be even"
            raise ValueError(msg)
        self.model_dim = model_dim
        self.max_period = max_period
        self.register_buffer("freqs", get_freqs(model_dim // 2, max_period), persistent=False)
        self.timestep_embedder = _TimestepEmbedder(model_dim, time_dim)

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, time: Tensor) -> tuple[Tensor, Tensor]:
        """Encode time steps into embeddings.

        Args:
            time: Diffusion time steps.

        Returns:
            Tuple of (time embeddings, index tensor mapping tokens to embeddings).
        """
        args = torch.outer(time, self.freqs.to(device=time.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        time_embed = self.timestep_embedder(time_embed)
        time_embed_idx = torch.arange(time_embed.shape[0], device=time_embed.device, dtype=torch.int32)
        return time_embed, time_embed_idx

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        self.timestep_embedder.linear_1.reset_parameters()
        self.timestep_embedder.linear_2.reset_parameters()
        self.freqs = get_freqs(self.model_dim // 2, self.max_period)


class TextEmbeddings(nn.Module):
    """Linear projection with layer normalization for text encoder outputs."""

    def __init__(self, text_dim: int, model_dim: int) -> None:
        """Initialize text embeddings.

        Args:
            text_dim: Input dimension of text encoder hidden states.
            model_dim: Output dimension after projection.
        """
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim, bias=True)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, text_embed: Tensor) -> Tensor:
        """Project and normalize text embeddings.

        Args:
            text_embed: Raw text encoder hidden states.

        Returns:
            Projected and normalized text embeddings.
        """
        text_embed = self.in_layer(text_embed)
        return self.norm(text_embed).type_as(text_embed)

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        self.in_layer.reset_parameters()
        self.norm.reset_parameters()


class VisualEmbeddings(nn.Module):
    """Patchify and project visual tokens into the transformer hidden space."""

    def __init__(  # noqa: PLR0913
        self,
        visual_dim: int,
        model_dim: int,
        patch_size: tuple[int, int, int],
        *,
        use_lq_modulation: bool = False,
        lq_channels: int = 16,
        time_dim: int = 512,
        zero_lq_in_main_path: bool = False,
    ) -> None:
        """Initialize visual embeddings.

        Args:
            visual_dim: Number of input visual channels.
            model_dim: Output dimension after projection.
            patch_size: Patch size as (temporal, height, width).
            use_lq_modulation: Whether to extract and encode LQ patches.
            lq_channels: Number of LQ channels in the input (used only when
                ``use_lq_modulation=True``).
            time_dim: Output dimension for the LQ encoder (used only when
                ``use_lq_modulation=True``).
            zero_lq_in_main_path: Zero out LQ and mask channels in the main
                ``in_layer`` input so LQ reaches the model only via per-block
                ``ModulationLQ``. Only used when ``use_lq_modulation=True``.
        """
        super().__init__()
        self.patch_size = patch_size
        self.use_lq_modulation = use_lq_modulation
        self.zero_lq_in_main_path = zero_lq_in_main_path
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

        if use_lq_modulation:
            self.lq_channels = lq_channels
            # input will be concat [hq(16) | lq(16) | mask(1)]
            self.lq_start = visual_dim - lq_channels - 1
            self.lq_end = visual_dim - 1
            self.lq_encoder = LQEncoder(
                input_dim=math.prod(patch_size) * lq_channels,
                time_dim=time_dim,
            )

    def _patchify(self, x: Tensor, visual_cu_seqlens: Tensor) -> tuple[Tensor, Tensor]:
        """Shared patchification logic for both full input and LQ-only slices.

        Args:
            x: Input tensor of shape ``(duration, height, width, channels)``.
            visual_cu_seqlens: Cumulative sequence lengths.

        Returns:
            Tuple of (patchified tensor, updated cumulative sequence lengths).
        """
        if self.patch_size[0] > 1:
            idxs = torch.ones(x.shape[0], dtype=torch.int32, device=visual_cu_seqlens.device)
            idxs[visual_cu_seqlens[:-1]] += self.patch_size[0] - 1
            x = torch.repeat_interleave(x, idxs, dim=0)
            visual_cu_seqlens = visual_cu_seqlens + torch.arange(
                visual_cu_seqlens.shape[0], device=visual_cu_seqlens.device, dtype=torch.int32
            )

        duration, height, width, dim = x.shape
        # [T, H, W, C] -> [T/pt, H/ph, W/pw, pt*ph*pw*C]
        # Groups of (pt, ph, pw) neighboring pixels are concatenated into one token vector
        x = (
            x.view(
                duration // self.patch_size[0],
                self.patch_size[0],
                height // self.patch_size[1],
                self.patch_size[1],
                width // self.patch_size[2],
                self.patch_size[2],
                dim,
            )
            .permute(0, 2, 4, 1, 3, 5, 6)
            .flatten(3, 6)
        )
        visual_cu_seqlens = visual_cu_seqlens // self.patch_size[0]
        return x, visual_cu_seqlens

    def forward(self, x: Tensor, visual_cu_seqlens: Tensor) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Patchify input and project to model dimension.

        Args:
            x: Visual input of shape ``(duration, height, width, channels)``.
            visual_cu_seqlens: Cumulative sequence lengths for packed sequences.

        Returns:
            When ``use_lq_modulation=False``: ``(projected_patches, cu_seqlens)``.
            When ``use_lq_modulation=True``: ``(projected_patches, cu_seqlens, lq_encoded)``.
        """
        if not self.use_lq_modulation:
            # [T, H, W, 33] -> [T, H/2, W/2, 132]
            # (T x H/2 x W/2) tokens, additive mix of patched HQ/LQ/mask, no cross-terms
            patches, cu = self._patchify(x, visual_cu_seqlens)
            # Linear(132 -> 1792)
            return self.in_layer(patches), cu

        lq_slice = x[..., self.lq_start : self.lq_end]
        if self.zero_lq_in_main_path:
            # Zero out LQ+mask so in_layer sees [HQ | zeros | 0]; LQ only via ModulationLQ
            x = x.clone()
            x[..., self.lq_start :] = 0.0
        # Main path: Linear(132 -> 1792) on all 33 channels (LQ zeroed if zero_lq_in_main_path)
        patches, cu = self._patchify(x, visual_cu_seqlens.clone())
        # LQ path: [T, H, W, 16] -> [T, H/2, W/2, 64]
        # Per-token LQ encodings for bilinear modulation HQ * f(LQ) in decoder blocks
        lq_patches, _ = self._patchify(lq_slice, visual_cu_seqlens.clone())
        # LQEncoder(64 -> 512)
        lq_encoded = self.lq_encoder(lq_patches)
        # Linear(132 -> 1792)
        return self.in_layer(patches), cu, lq_encoded

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        self.in_layer.reset_parameters()
        if self.use_lq_modulation:
            self.lq_encoder.reset_parameters()


class RoPE1D(nn.Module):
    """1D rotary position embeddings for text tokens."""

    def __init__(self, dim: int, max_pos: int = 1024, max_period: float = 10000.0) -> None:
        """Initialize 1D rotary position embeddings.

        Args:
            dim: Embedding dimension. Must be even.
            max_pos: Maximum number of positions to precompute.
            max_period: Maximum period for frequency computation.
        """
        super().__init__()
        self.max_period = max_period
        self.dim = dim
        self.max_pos = max_pos
        freq = get_freqs(dim // 2, max_period)
        pos = torch.arange(max_pos, dtype=freq.dtype)
        self.register_buffer("args", torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, pos: Tensor) -> Tensor:
        """Compute rotary embeddings for given positions.

        Args:
            pos: Position indices.

        Returns:
            Rotary embedding matrix with shape ``(*pos.shape, 1, dim//2, 2, 2)``.
        """
        args = self.args[pos]
        rope = torch.stack([torch.cos(args), -torch.sin(args), torch.sin(args), torch.cos(args)], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        freq = get_freqs(self.dim // 2, self.max_period).to(self.args.device)
        pos = torch.arange(self.max_pos, dtype=freq.dtype, device=freq.device)
        self.args = torch.outer(pos, freq)


class RoPE3D(nn.Module):
    """3D rotary position embeddings for visual tokens along temporal, height, and width axes."""

    def __init__(
        self,
        axes_dims: tuple[int, int, int],
        max_pos: tuple[int, int, int] = (128, 128, 128),
        max_period: float = 10000.0,
    ) -> None:
        """Initialize 3D rotary position embeddings.

        Args:
            axes_dims: Per-axis embedding dimensions (temporal, height, width).
            max_pos: Maximum positions per axis.
            max_period: Maximum period for frequency computation.
        """
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period

        for i, (axes_dim, ax_max_pos) in enumerate(zip(axes_dims, max_pos, strict=False)):
            freq = get_freqs(axes_dim // 2, max_period)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype)
            self.register_buffer(f"args_{i}", torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(
        self,
        shape: tuple[int, int, int],
        pos: tuple[Tensor, Tensor, Tensor],
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tensor:
        """Compute 3D rotary embeddings for given positions.

        Args:
            shape: Spatial dimensions as (duration, height, width).
            pos: Position indices per axis (temporal, height, width).
            scale_factor: Frequency scaling per axis.

        Returns:
            Rotary embedding matrix broadcast to (duration, height, width, 1, dim//2, 2, 2).
        """
        duration, height, width = shape
        args_t = self.args_0[pos[0]] / scale_factor[0]
        args_h = self.args_1[pos[1]] / scale_factor[1]
        args_w = self.args_2[pos[2]] / scale_factor[2]

        args = torch.cat(
            [
                args_t.view(duration, 1, 1, -1).repeat(1, height, width, 1),
                args_h.view(1, height, 1, -1).repeat(duration, 1, width, 1),
                args_w.view(1, 1, width, -1).repeat(duration, height, 1, 1),
            ],
            dim=-1,
        )
        rope = torch.stack([torch.cos(args), -torch.sin(args), torch.sin(args), torch.cos(args)], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        for i, (axes_dim, ax_max_pos) in enumerate(zip(self.axes_dims, self.max_pos, strict=False)):
            freq = get_freqs(axes_dim // 2, self.max_period).to(self.args_0.device)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype, device=freq.device)
            setattr(self, f"args_{i}", torch.outer(pos, freq))


class Modulation(nn.Module):
    """Adaptive modulation layer producing scale, shift, and gate parameters from time embeddings."""

    def __init__(self, time_dim: int, model_dim: int, num_params: int) -> None:
        """Initialize modulation layer.

        Args:
            time_dim: Input dimension of time embeddings.
            model_dim: Per-parameter output dimension.
            num_params: Number of modulation parameters to produce.
        """
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x: Tensor) -> Tensor:
        """Compute modulation parameters from time embeddings.

        Args:
            x: Time embedding tensor.

        Returns:
            Concatenated modulation parameters.
        """
        return self.out_layer(self.activation(x))

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()


class LQEncoder(nn.Module):
    """Lightweight MLP that maps patchified LQ tokens to the time embedding space.

    Architecture: Linear -> SiLU -> Linear. Used to produce per-token conditioning
    vectors from LQ patches for spatially-varying modulation in decoder blocks.

    Args:
        input_dim: Dimension of patchified LQ tokens (patch_volume * lq_channels).
        time_dim: Output dimension matching the time embedding space.
    """

    def __init__(self, input_dim: int, time_dim: int) -> None:
        """Initialize LQ encoder layers.

        Args:
            input_dim: Dimension of patchified LQ tokens (patch_volume * lq_channels).
            time_dim: Output dimension matching the time embedding space.
        """
        super().__init__()
        self.in_layer = nn.Linear(input_dim, time_dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, time_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Encode patchified LQ tokens.

        Args:
            x: Patchified LQ tokens of shape ``(total_tokens, input_dim)``.

        Returns:
            Encoded LQ tokens of shape ``(total_tokens, time_dim)``.
        """
        return self.out_layer(self.activation(self.in_layer(x)))

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        self.in_layer.reset_parameters()
        self.out_layer.reset_parameters()


class ModulationLQ(nn.Module):
    """Per-token modulation layer producing scale, shift, and gate from LQ encodings.

    Identical architecture to ``Modulation`` but designed for per-token (spatially-varying)
    input from ``LQEncoder`` rather than per-sample time embeddings.
    Zero-initialized so the model starts identically to a pretrained checkpoint
    without LQ modulation.

    Args:
        time_dim: Input dimension of LQ encodings (same as time embedding dim).
        model_dim: Per-parameter output dimension.
        num_params: Number of modulation parameters to produce.
    """

    def __init__(self, time_dim: int, model_dim: int, num_params: int) -> None:
        """Initialize modulation layer with zero weights.

        Args:
            time_dim: Input dimension of LQ encodings (same as time embedding dim).
            model_dim: Per-parameter output dimension.
            num_params: Number of modulation parameters to produce.
        """
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        """Compute per-token modulation parameters from LQ encodings.

        Args:
            x: LQ encoding tensor of shape ``(total_tokens, time_dim)``.

        Returns:
            Concatenated modulation parameters ``(total_tokens, num_params * model_dim)``.
        """
        return self.out_layer(self.activation(x))

    def reset_parameters(self) -> None:
        """Re-initialize with zeros for identity behavior."""
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()


@ModelLogger.log_attention
class MultiheadSelfAttention(nn.Module):
    """Multi-head self-attention with flash attention and optional sparse flex attention."""

    def __init__(self, num_channels: int, head_dim: int) -> None:
        """Initialize multi-head self-attention.

        Args:
            num_channels: Total number of channels. Must be divisible by head_dim.
            head_dim: Dimension per attention head.
        """
        super().__init__()
        if num_channels % head_dim != 0:
            msg = "num_channels must be divisible by head_dim"
            raise ValueError(msg)
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

        self.cached_k: Tensor | None = None
        self.cached_v: Tensor | None = None
        self.cached_cu_seqlens: Tensor | None = None
        self.return_kv = False

    def get_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project input into query, key, and value tensors.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (query, key, value) reshaped to ``(seq_len, num_heads, head_dim)``.
        """
        query = self.to_query(x)
        key = self.to_key(x)
        value = self.to_value(x)

        shape = query.shape[:-1]  # for TP compatibility
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*shape, self.num_heads, -1)
        value = value.reshape(*shape, self.num_heads, -1)

        return query, key, value

    def norm_qk(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply RMS normalization to query and key.

        Args:
            q: Query tensor.
            k: Key tensor.

        Returns:
            Tuple of (normalized query, normalized key).
        """
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    @_disable_dynamo_if_fa2
    def scaled_dot_product_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        cu_seqlens: Tensor,
        *,
        return_attn_probs: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, None]:
        """Compute self-attention using flash attention.

        When a KV-cache is present (``cached_k is not None``) the queries are the
        new tokens and attention runs over ``[cache | new]`` keys/values via
        :meth:`attend_with_kv_cache`. When ``return_kv`` is set, the (merged) K/V
        are stored as the cache for the next streaming step.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            cu_seqlens: Cumulative sequence lengths for packed sequences.
            return_attn_probs: Whether to return attention log-sum-exp values.

        Returns:
            Attention output, or tuple of (output, softmax_lse, None) if returning probs.
        """
        if self.cached_k is not None:
            return self.attend_with_kv_cache(query, key, value, cu_seqlens, return_attn_probs=return_attn_probs)

        if self.return_kv:
            self.cached_k, self.cached_v, self.cached_cu_seqlens = key, value, cu_seqlens

        max_seqlen = torch.diff(cu_seqlens).max()
        if FA3:
            out, softmax_lse = flash_attn_interface.flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
            )
        else:
            _, flash_attn_varlen_qkvpacked_func = flash_attn_funcs()
            query_key_value = torch.stack([query, key, value], dim=-3)
            out, softmax_lse, _ = flash_attn_varlen_qkvpacked_func(
                query_key_value, cu_seqlens, max_seqlen, return_attn_probs=True
            )
        out = out.flatten(-2, -1)

        if return_attn_probs:
            return out, softmax_lse, None
        return out

    @torch.compiler.disable
    def attend_with_kv_cache(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        cu_seqlens: Tensor,
        *,
        return_attn_probs: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, None]:
        """Dense flash attention consuming the rolling KV-cache.

        Queries are the new tokens; keys/values are the per-sequence concatenation
        ``[cache | new]`` (with absolute RoPE already applied upstream). When
        ``return_kv`` is set, the merged K/V become the cache for the next
        streaming step. Kept out of ``torch.compile`` because the cache assembly
        indexes ``cu_seqlens`` host-side; it only runs during streaming inference.

        Args:
            query: New-token query tensor.
            key: New-token key tensor (post-RoPE).
            value: New-token value tensor.
            cu_seqlens: Cumulative sequence lengths of the new tokens.
            return_attn_probs: Whether to return attention log-sum-exp values.

        Returns:
            Attention output for the new tokens, or tuple of (output, softmax_lse,
            None) if returning probs.
        """
        key, value, cu_seqlens_q, cu_seqlens_k = assemble_cached_attention_inputs(
            key, value, self.cached_k, self.cached_v, cu_seqlens, self.cached_cu_seqlens
        )
        if self.return_kv:
            self.cached_k, self.cached_v, self.cached_cu_seqlens = key, value, cu_seqlens_k

        max_seqlen_q = torch.diff(cu_seqlens_q).max()
        max_seqlen_k = torch.diff(cu_seqlens_k).max()
        if FA3:
            out, softmax_lse = flash_attn_interface.flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )
        else:
            flash_attn_varlen_func, _ = flash_attn_funcs()
            out, softmax_lse, _ = flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                return_attn_probs=True,
            )
        out = out.flatten(-2, -1)

        if return_attn_probs:
            return out, softmax_lse, None
        return out

    def reset_kv_cache(self) -> None:
        """Clear the rolling KV-cache and stop emitting K/V."""
        self.cached_k = None
        self.cached_v = None
        self.cached_cu_seqlens = None
        self.return_kv = False

    def attention_flex(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        sparse_params: dict[str, Any] | None = None,
        *,
        return_sparsity: bool = False,
    ) -> Tensor | tuple[Tensor, float]:
        """Compute sparse self-attention using flex attention with block masks.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            sparse_params: Sparse attention configuration.
            return_sparsity: Whether to return sparsity percentage.

        Returns:
            Attention output, or tuple of (output, sparsity percentage) if requested.
        """
        if self.cached_k is not None:
            msg = "KV-cache on the flex/NABLA path is out of scope (dense flash only in M0.2; see LAY-429)."
            raise NotImplementedError(msg)

        query = query.unsqueeze(0).transpose(1, 2).contiguous()
        key = key.unsqueeze(0).transpose(1, 2).contiguous()
        value = value.unsqueeze(0).transpose(1, 2).contiguous()

        t, h, w = sparse_params["visual_shape"]
        # Token grid -> fractal 8x8 block grid; the block size matches
        # fractal_flatten and is unrelated to the VAE spatial compression.
        h, w = h // FRACTAL_BLOCK_SIZE, w // FRACTAL_BLOCK_SIZE
        visual_seqlens = sparse_params["visual_seqlens"].to(device=query.device)
        if sparse_params["attention_type"] == "dense_framewise_causal":
            block_mask = framewise_causal_dense(
                visual_seqlens,
                h,
                w,
                mf=sparse_params["mf"],
            )
        elif "mf" not in sparse_params:
            block_mask = nablaT_v2_doc(
                query,
                key,
                visual_seqlens,
                t,
                h,
                w,
                wT=sparse_params["wT"],
                wH=sparse_params["wH"],
                wW=sparse_params["wW"],
                thr=sparse_params["P"],
                add_sta=sparse_params["add_sta"],
                method=sparse_params["method"],
            )
        else:
            block_mask = nablaT_v2_doc_mfcausal(
                query,
                key,
                visual_seqlens,
                t,
                h,
                w,
                wT=sparse_params["wT"],
                wH=sparse_params["wH"],
                wW=sparse_params["wW"],
                thr=sparse_params["P"],
                add_sta=sparse_params["add_sta"],
                mf=sparse_params["mf"],
            )
        out = (
            flex(
                query,
                key,
                value,
                block_mask=block_mask,
                kernel_options={"BLOCK_M": 64, "BLOCK_N": 64},
            )
            .transpose(1, 2)
            .squeeze(0)
            .contiguous()
        )
        out = out.flatten(-2, -1)

        if return_sparsity:
            sparsity = 100.0 * (1 - (1 - block_mask.sparsity() / 100) * (sparse_params["visual_seqlens"].shape[0] - 1))
            return out, sparsity
        return out

    def forward(
        self,
        x: Tensor,
        rope: Tensor,
        cu_seqlens: Tensor,
        sparse_params: dict[str, Any] | None = None,
    ) -> Tensor:
        """Run self-attention with rotary embeddings.

        Args:
            x: Input tensor.
            rope: Rotary position embeddings.
            cu_seqlens: Cumulative sequence lengths for packed sequences.
            sparse_params: Optional sparse attention parameters.

        Returns:
            Self-attention output.
        """
        query, key, value = self.get_qkv(x)
        query, key = self.norm_qk(query, key)
        query = apply_rotary(query, rope).type_as(query)
        key = apply_rotary(key, rope).type_as(key)

        if sparse_params is not None:
            out = self.attention_flex(query, key, value, sparse_params=sparse_params)
        else:
            out = self.scaled_dot_product_attention(query, key, value, cu_seqlens)

        return self.out_layer(out)

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        self.to_query.reset_parameters()
        self.to_key.reset_parameters()
        self.to_value.reset_parameters()

        self.out_layer.reset_parameters()

        self.query_norm.reset_parameters()
        self.key_norm.reset_parameters()


@ModelLogger.log_attention
class MultiheadCrossAttention(nn.Module):
    """Multi-head cross-attention with flash attention."""

    def __init__(self, num_channels: int, head_dim: int) -> None:
        """Initialize multi-head cross-attention.

        Args:
            num_channels: Total number of channels. Must be divisible by head_dim.
            head_dim: Dimension per attention head.
        """
        super().__init__()
        if num_channels % head_dim != 0:
            msg = "num_channels must be divisible by head_dim"
            raise ValueError(msg)
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

    def get_qkv(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project input and condition into query, key, and value tensors.

        Args:
            x: Input tensor for query projection.
            cond: Conditioning tensor for key and value projections.

        Returns:
            Tuple of (query, key, value) reshaped to ``(seq_len, num_heads, head_dim)``.
        """
        query = self.to_query(x)
        key = self.to_key(cond)
        value = self.to_value(cond)

        shape, cond_shape = query.shape[:-1], key.shape[:-1]  # for TP compatibility
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*cond_shape, self.num_heads, -1)
        value = value.reshape(*cond_shape, self.num_heads, -1)

        return query, key, value

    def norm_qk(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply RMS normalization to query and key.

        Args:
            q: Query tensor.
            k: Key tensor.

        Returns:
            Tuple of (normalized query, normalized key).
        """
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    @_disable_dynamo_if_fa2
    def scaled_dot_product_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        cu_seqlens: Tensor,
        cond_cu_seqlens: Tensor,
        *,
        return_attn_probs: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, None]:
        """Compute cross-attention using flash attention.

        Args:
            query: Query tensor.
            key: Key tensor from conditioning input.
            value: Value tensor from conditioning input.
            cu_seqlens: Cumulative sequence lengths for query sequences.
            cond_cu_seqlens: Cumulative sequence lengths for conditioning sequences.
            return_attn_probs: Whether to return attention log-sum-exp values.

        Returns:
            Attention output, or tuple of (output, softmax_lse, None) if returning probs.
        """
        max_seqlen = torch.diff(cu_seqlens).max()
        cond_max_seqlen = torch.diff(cond_cu_seqlens).max()
        flash_attn_varlen_func, _ = flash_attn_funcs()
        out, softmax_lse, _ = flash_attn_varlen_func(
            query, key, value, cu_seqlens, cond_cu_seqlens, max_seqlen, cond_max_seqlen, return_attn_probs=True
        )
        out = out.flatten(-2, -1)

        if return_attn_probs:
            return out, softmax_lse, None
        return out

    def forward(self, x: Tensor, cond: Tensor, cu_seqlens: Tensor, cond_cu_seqlens: Tensor) -> Tensor:
        """Run cross-attention between input and conditioning.

        Args:
            x: Input tensor for query computation.
            cond: Conditioning tensor for key and value computation.
            cu_seqlens: Cumulative sequence lengths for input sequences.
            cond_cu_seqlens: Cumulative sequence lengths for conditioning sequences.

        Returns:
            Cross-attention output.
        """
        query, key, value = self.get_qkv(x, cond)
        query, key = self.norm_qk(query, key)

        out = self.scaled_dot_product_attention(query, key, value, cu_seqlens, cond_cu_seqlens)
        return self.out_layer(out)

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        self.to_query.reset_parameters()
        self.to_key.reset_parameters()
        self.to_value.reset_parameters()

        self.out_layer.reset_parameters()

        self.query_norm.reset_parameters()
        self.key_norm.reset_parameters()


class _GELUProjection(nn.Module):
    """Bias-free linear plus exact GELU. The linear is ``proj``, matching Diffusers ``GELU``."""

    def __init__(self, dim: int, ff_dim: int) -> None:
        """Initialize the projection.

        Args:
            dim: Input dimension.
            ff_dim: Output dimension.
        """
        super().__init__()
        self.proj = nn.Linear(dim, ff_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Project and apply GELU."""
        return functional.gelu(self.proj(x))


class FeedForward(nn.Module):
    """Two-layer feed-forward network with GELU activation."""

    def __init__(self, dim: int, ff_dim: int) -> None:
        """Initialize feed-forward network.

        Args:
            dim: Input and output dimension.
            ff_dim: Hidden layer dimension.
        """
        super().__init__()
        # Dropout keeps the output linear at index 2, matching Diffusers ``FeedForward.net``.
        self.net = nn.ModuleList(
            [
                _GELUProjection(dim, ff_dim),
                nn.Dropout(0.0),
                nn.Linear(ff_dim, dim, bias=False),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply feed-forward transformation.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor.
        """
        for module in self.net:
            x = module(x)
        return x

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        projection = self.net[0]
        output = self.net[2]
        if not isinstance(projection, _GELUProjection) or not isinstance(output, nn.Linear):
            msg = "FeedForward.net layout must be GELU projection, dropout, linear"
            raise TypeError(msg)
        projection.proj.reset_parameters()
        output.reset_parameters()


class OutLayer(nn.Module):
    """Final output layer that unpatchifies and projects visual tokens to pixel space."""

    def __init__(self, model_dim: int, time_dim: int, visual_dim: int, patch_size: tuple[int, int, int]) -> None:
        """Initialize output layer.

        Args:
            model_dim: Input dimension from the transformer.
            time_dim: Dimension of time embeddings for modulation.
            visual_dim: Number of output visual channels.
            patch_size: Patch size as (temporal, height, width).
        """
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(model_dim, math.prod(patch_size) * visual_dim, bias=True)

    def forward(
        self,
        visual_embed: Tensor,
        _text_embed: Tensor,
        time_embed: Tensor,
        visual_cu_seqlens: Tensor,
        time_embed_idx: Tensor,
    ) -> Tensor:
        """Unpatchify and project visual embeddings to output channels.

        Args:
            visual_embed: Visual token embeddings from the transformer.
            _text_embed: Text embeddings (unused, reserved for interface compatibility).
            time_embed: Time step embeddings for adaptive modulation.
            visual_cu_seqlens: Cumulative sequence lengths for visual tokens.
            time_embed_idx: Index mapping visual tokens to their time embeddings.

        Returns:
            Denoised visual output tensor.
        """
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        visual_embed = apply_scale_shift_norm(
            self.norm, visual_embed, scale[:, None, None], shift[:, None, None], time_embed_idx
        ).type_as(visual_embed)
        x = self.out_layer(visual_embed)

        duration, height, width, _dim = x.shape
        x = (
            x.view(
                duration,
                height,
                width,
                -1,
                self.patch_size[0],
                self.patch_size[1],
                self.patch_size[2],
            )
            .permute(0, 4, 1, 5, 2, 6, 3)
            .flatten(0, 1)
            .flatten(1, 2)
            .flatten(2, 3)
        )
        visual_cu_seqlens = visual_cu_seqlens * self.patch_size[0]

        if self.patch_size[0] > 1:
            idxs = torch.ones(duration * self.patch_size[0], dtype=torch.int32, device=visual_cu_seqlens.device)
            idxs[visual_cu_seqlens[:-1]] -= self.patch_size[0] - 1
            x = torch.repeat_interleave(x, idxs, dim=0)
        return x

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        self.norm.reset_parameters()
        self.out_layer.reset_parameters()
        self.modulation.reset_parameters()
