# Copyright 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native MiniMax H3 transformer.

The implementation follows the released Diffusers H3 port at revision
``abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`` while retaining the original
checkpoint module names and fused projections. This keeps H3 independent of
Diffusers model APIs and lets Musubi load the Comfy BF16 repack directly.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch
from diffusers.models.embeddings import get_timestep_embedding
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from musubi_tuner.minimax_h3.activation_offload import ReusableActivationOffloader
from musubi_tuner.minimax_h3.block_sparse_attention import BlockSparseConfig, SequencePlan, block_sparse_attention, build_plan
from musubi_tuner.minimax_h3.int8_attention import HAS_TRITON, int8_attention
from musubi_tuner.minimax_h3.selective_checkpoint import (
    SavedActivations,
    attention_kernel_is_visible,
    checkpoint_context_fn,
    projection_region,
    validate_checkpoint_keep,
)
from musubi_tuner.minimax_h3.triton_kernels import (
    try_fused_indexed_adaln_rmsnorm,
    try_fused_qk_norm_rope,
    try_fused_swiglu,
)
from musubi_tuner.modules.attention import AttentionParams
from musubi_tuner.modules.attention import attention as musubi_attention
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader

MINIMAX_H3_MODALITY_COUNT = 3
_CUDNN_AUTO_WORK_THRESHOLD = 1 << 28
_CUDNN_AUTO_MIN_SEQUENCE = 1024


def h3_profile_scope(name: str) -> AbstractContextManager:
    """A ``torch.profiler.record_function`` label while a profiler is collecting, else a no-op.

    Outside profiling this returns ``nullcontext``, so a block pays a flag read and a null
    context enter per label -- negligible next to its compute. Inside a ``torch.compile``
    region Dynamo treats both as null contexts, and a checkpointed block simply re-enters the
    label on recompute."""
    if getattr(torch.autograd.profiler, "_is_profiler_enabled", False):
        return torch.profiler.record_function(name)
    return nullcontext()


try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _CUDNN_SDPA_ORDER = [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
    _SDPA_HAS_SET_PRIORITY = "set_priority" in inspect.signature(sdpa_kernel).parameters
except (ImportError, AttributeError, TypeError, ValueError):
    SDPBackend = None
    sdpa_kernel = None
    _CUDNN_SDPA_ORDER = None
    _SDPA_HAS_SET_PRIORITY = False


def _cudnn_auto_workload_is_large(query_length: int, key_length: int, head_dim: int) -> bool:
    return (
        min(query_length, key_length) >= _CUDNN_AUTO_MIN_SEQUENCE
        and query_length * key_length * head_dim >= _CUDNN_AUTO_WORK_THRESHOLD
    )


def _use_cudnn_auto_dispatch(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask) -> bool:
    return (
        attention_mask is None
        and SDPBackend is not None
        and _SDPA_HAS_SET_PRIORITY
        and query.device.type == "cuda"
        and query.dtype in {torch.float16, torch.bfloat16}
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and _cudnn_auto_workload_is_large(query.shape[-2], key.shape[-2], query.shape[-1])
    )


SDPA_INT32_EXTENT = 2**31


def view_element_extent(tensor: torch.Tensor) -> int:
    """Indexing bound of a view: the largest ``size * stride`` over its dimensions.

    Not the exact storage span of an arbitrary view (that sums ``(size - 1) * stride``);
    for the fused-QKV value view it is ``batch * rows * 3 * inner_dim``, the address
    range the memory-efficient SDPA backend computes in int32, and it can only err on
    the side of copying slightly early.
    """
    if tensor.numel() == 0:
        return 0
    return max(size * stride for size, stride in zip(tensor.shape, tensor.stride()))


def materialize_wide_view(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a strided view whose storage extent exceeds the int32 range.

    ``value`` reaches attention as a view into the fused QKV buffer (sequence stride
    ``3 * inner_dim``), while Q and K come back fresh from their norms. The
    memory-efficient SDPA backend computes KV block addresses in int32, so once the
    view spans more than 2^31 elements (about 99.9k packed rows for H3) it reads
    wrapped memory and the sample degrades to noise. Ordinary sizes pay nothing;
    the copy only happens past that extent (kohya-ss/musubi-tuner#1089).
    """
    if tensor.is_contiguous() or view_element_extent(tensor) <= SDPA_INT32_EXTENT:
        return tensor
    return tensor.contiguous()


class MiniMaxH3TokenTag(IntEnum):
    VIDEO = 0
    TEXT = 1
    AUDIO = 2


@dataclass(frozen=True)
class MiniMaxH3TransformerConfig:
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    adaln_t_table_size: int | None = None
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    @classmethod
    def from_dict(cls, value: dict) -> MiniMaxH3TransformerConfig:
        fields = cls.__dataclass_fields__
        kwargs = {key: value[key] for key in fields if key in value}
        if "patch_size" in kwargs:
            kwargs["patch_size"] = tuple(kwargs["patch_size"])
        return cls(**kwargs)


@dataclass
class MiniMaxH3TransformerOutput:
    video: torch.Tensor
    audio: torch.Tensor


@dataclass
class MiniMaxH3PackedState:
    """One embedded packed sequence, as the block loop reads and rewrites it.

    Everything here is per-sequence: the row-indexed AdaLN selector, the RoPE
    tables cut for this sequence's positions, and whatever pairwise mask its
    padding produced. Two states therefore share nothing but the weights, which
    is what lets several of them ride one pass over the blocks.
    """

    hidden_states: torch.Tensor
    timestep_embedding: torch.Tensor
    adaln_indices: torch.Tensor
    rotary_emb: tuple[torch.Tensor, torch.Tensor]
    attention_mask: torch.Tensor | None
    timestep_indices: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor


def _apply_rotary_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply the released H3 RoPE formula without changing its operation order."""
    rotary_dim = cos.shape[-1]
    rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((rotary * cos + rotated * sin, passthrough), dim=-1).contiguous()


class MiniMaxH3RotaryPosEmbed(nn.Module):
    def __init__(self, rope_freq_dim: int, rope_theta: float):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim)))
        # The released checkpoint contains this value even though it is fully
        # determined by the config. Keeping it persistent gives strict 535-key loading.
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        time, height, width = frequencies.unbind(dim=1)
        frequencies = torch.cat((time, height, width), dim=-1)
        frequencies = torch.cat((frequencies, frequencies), dim=-1)
        return frequencies.cos(), frequencies.sin()


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(self, freq_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        embedding = get_timestep_embedding(
            timestep,
            self.freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        embedding = embedding.to(self.proj_in.weight.dtype)
        return self.proj_out(F.silu(self.proj_in(embedding)))


class MiniMaxH3AdaLNProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, apply_silu: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.apply_silu = apply_silu

    def forward(self, timestep_embedding: torch.Tensor) -> torch.Tensor:
        # Scaled FP8 stores the frozen weight in E4M3 but dequantizes it to the
        # scale buffer's dtype for the matmul. Casting the activation to the
        # storage dtype would bypass that compute contract and make F.linear see
        # mismatched FP8/BF16 operands.
        compute_dtype = getattr(self.linear, "scale_weight", self.linear.weight).dtype
        activated = F.silu(timestep_embedding) if self.apply_silu else timestep_embedding
        activated = activated.to(compute_dtype)
        return self.linear(activated)


class MiniMaxH3Attention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, head_dim: int, qk_norm_eps: float, attention_mode: str = "torch"):
        super().__init__()
        if attention_mode not in {"torch", "flash", "flash3"}:
            raise ValueError(f"unsupported MiniMax H3 attention mode: {attention_mode}")
        self.heads = heads
        self.head_dim = head_dim
        self.attention_mode = attention_mode
        self.auto_dispatch = False
        self.int8_attention = False
        self.block_sparse_config: BlockSparseConfig | None = None
        self.block_sparse_plan: SequencePlan | None = None
        self.fused_qk_norm_rope = False
        # Set by ``MiniMaxH3Transformer.set_checkpoint_keep("qkv")``; gates the
        # region ``install_projection_region`` wraps around the base projection.
        self.checkpoint_keep_qkv = False
        # Route the attention kernel through ``_attention_core_opaque`` (see there); a no-op
        # outside torch.compile, where both are the same eager code.
        self.opaque_attention = False
        self.inner_dim = heads * head_dim
        self.qkv_proj = nn.Linear(hidden_size, 3 * self.inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps)
        self.out_proj = nn.Linear(self.inner_dim, hidden_size, bias=False)

    def install_projection_region(self) -> None:
        """Wrap the base projection's own ``forward`` in the ``qkv`` checkpoint region, once.

        Wrapping the module's forward rather than the call site is what keeps
        adapters out of the region: a LoRA applied afterwards captures this
        wrapper as its ``org_forward`` and runs its own down/up matmuls around
        it, so only the frozen projection (plain, scaled FP8 or ConvRot INT8)
        is inside. Installed permanently and gated by ``checkpoint_keep_qkv``,
        because an adapter may already hold a reference to it.
        """
        projection = self.qkv_proj
        if getattr(projection, "_h3_projection_region", False):
            return
        base_forward = projection.forward
        attention = self

        def forward(module: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
            if not attention.checkpoint_keep_qkv:
                return base_forward(*args, **kwargs)
            with projection_region(module.in_features, module.out_features):
                return base_forward(*args, **kwargs)

        projection.forward = types.MethodType(forward, projection)
        projection._h3_projection_region = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value = self.project_qkv(hidden_states, rotary_emb)
        core = self._attention_core_opaque if self.opaque_attention else self._attention_core
        hidden_states = core(query, key, value, attention_mask)
        with h3_profile_scope("h3.out"):
            return self.out_proj(hidden_states)

    def project_qkv(
        self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projected, normalised and rotated ``[B, S, heads, head_dim]`` Q/K plus V."""
        with h3_profile_scope("h3.qkv"):
            query, key, value = self.qkv_proj(hidden_states).chunk(3, dim=-1)
        query = query.unflatten(-1, (self.heads, self.head_dim))
        key = key.unflatten(-1, (self.heads, self.head_dim))
        value = materialize_wide_view(value).unflatten(-1, (self.heads, self.head_dim))
        with h3_profile_scope("h3.rope"):
            query, key = self._norm_and_rotate(query, key, rotary_emb)
        return query, key, value

    def _norm_and_rotate(
        self, query: torch.Tensor, key: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused_qk = None
        if self.fused_qk_norm_rope and rotary_emb is not None:
            fused_qk = try_fused_qk_norm_rope(
                query,
                key,
                self.q_norm.weight,
                self.k_norm.weight,
                rotary_emb[0],
                rotary_emb[1],
                self.q_norm.eps,
            )
        if fused_qk is not None:
            return fused_qk
        query = self.q_norm(query)
        key = self.k_norm(key)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)
        return query, key

    def _attention_core(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """``[B, S, heads, head_dim]`` Q/K/V to the ``[B, S, heads * head_dim]`` attention output."""
        if self.block_sparse_config is not None and attention_mask is None:
            # A padding mask is a pairwise condition block selection cannot
            # express, so masked calls fall through to the dense path.
            hidden_states = block_sparse_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), self.block_sparse_config, self.block_sparse_plan
            )
            hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        elif self.int8_attention and attention_mask is None:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            hidden_states = int8_attention(query, key, value).transpose(1, 2).flatten(2, 3)
        elif self.attention_mode in {"flash", "flash3"} and attention_mask is None:
            hidden_states = musubi_attention(
                [query, key, value],
                attn_params=AttentionParams.create_attention_params(self.attention_mode, False),
            )
        else:
            # Padding uses a pairwise mask which FlashAttention cannot express.
            # Keep the exact SDPA path for that rare case instead of changing
            # the packed-sequence semantics.
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if attention_mask is not None:
                attention_mask = attention_mask[None, None, :, :]
            if self.auto_dispatch and _use_cudnn_auto_dispatch(query, key, value, attention_mask):
                with sdpa_kernel(_CUDNN_SDPA_ORDER, set_priority=True):
                    hidden_states = F.scaled_dot_product_attention(
                        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
                    )
            else:
                hidden_states = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
            hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        return hidden_states

    # The same core, invisible to Dynamo. A kernel binding Dynamo cannot trace (a FlashAttention
    # build without ``torch.library`` fake kernels, the Triton INT8 path) breaks the graph *inside*
    # the attention call, and the breaks cascade: the call site is split, the transposes and
    # reshapes around the kernel are stranded in fragments of a few ops, and the norm, modulation,
    # gating and residual elementwise chain of the block is cut in the middle instead of fusing
    # on each side of the kernel. Disabling this one function makes it a single break at the
    # call, with one fused graph before it and one after; it runs eagerly, autograd included,
    # so numerics are the eager ones. SDPA traces natively and stays in the graph.
    _attention_core_opaque = torch.compiler.disable(_attention_core, recursive=True)


class MiniMaxH3FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 2 * ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, hidden_size, bias=False)
        self.fused_swiglu = False
        self.chunk_rows = 0

    def _forward_rows(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.fc1(hidden_states)
        with h3_profile_scope("h3.swiglu"):
            activated = None
            if self.fused_swiglu:
                activated = try_fused_swiglu(projected)
            if activated is None:
                gate, value = projected.chunk(2, dim=-1)
                activated = F.silu(gate) * value
        return self.fc2(activated)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.chunk_rows <= 0 or hidden_states.shape[-2] <= self.chunk_rows:
            return self._forward_rows(hidden_states)
        return torch.cat(
            tuple(self._forward_rows(chunk) for chunk in hidden_states.split(self.chunk_rows, dim=-2)),
            dim=-2,
        )


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig, attention_mode: str = "torch"):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = MiniMaxH3Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_head_dim,
            config.qk_norm_eps,
            attention_mode,
        )
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = MiniMaxH3FeedForward(config.hidden_size, config.ffn_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig, attention_mode: str = "torch"):
        super().__init__()
        self.blocks = nn.ModuleList([MiniMaxH3TokenRefinerBlock(config, attention_mode) for _ in range(config.num_refiner_layers)])
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)

    def forward(self, hidden_states: torch.Tensor, gradient_checkpointing: bool = False) -> torch.Tensor:
        for block in self.blocks:
            if torch.is_grad_enabled() and gradient_checkpointing:
                hidden_states = checkpoint(block, hidden_states, use_reentrant=False)
            else:
                hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class MiniMaxH3TransformerBlock(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig, attention_mode: str = "torch"):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = MiniMaxH3Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_head_dim,
            config.qk_norm_eps,
            attention_mode,
        )
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = MiniMaxH3FeedForward(config.hidden_size, config.ffn_dim)
        self.adaln_proj = MiniMaxH3AdaLNProjection(
            config.time_embed_dim,
            6 * config.hidden_size * MINIMAX_H3_MODALITY_COUNT,
            apply_silu=config.adaln_t_table_size is None,
        )
        self.hidden_size = config.hidden_size
        self.fused_indexed_adaln = False
        self.fused_elementwise = False

    def _norm_and_modulate(
        self,
        norm: nn.RMSNorm,
        hidden_states: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        adaln_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.fused_indexed_adaln:
            fused = try_fused_indexed_adaln_rmsnorm(
                hidden_states,
                norm.weight,
                shift,
                scale,
                adaln_indices,
                norm.eps,
            )
            if fused is not None:
                return fused
        normalized = norm(hidden_states)
        if self.fused_elementwise:
            # shift + normalized * (1 + scale) with the product kept in the fp32 accumulator and
            # rounded once: 2 kernels and 6 tensor passes instead of 3 and 8.
            return torch.addcmul(shift.index_select(0, adaln_indices), normalized, 1.0 + scale.index_select(0, adaln_indices))
        normalized = normalized * (1.0 + scale.index_select(0, adaln_indices))
        return normalized + shift.index_select(0, adaln_indices)

    def _gated_residual(self, hidden_states: torch.Tensor, gate: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        if self.fused_elementwise:
            # One kernel, 4 tensor passes, the product rounded once (the default rounds it twice).
            return torch.addcmul(hidden_states, gate, branch)
        return hidden_states + gate * branch

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Written out in this one frame on purpose: under torch.compile a graph break inside an
        # inlined callee is replayed at every frame level above it (nested graph breaks), so the
        # opaque attention kernel is called from here, not from ``self.attn.forward``, and a block
        # compiles to exactly one graph before the kernel and one after.
        with h3_profile_scope("h3.block"):
            with h3_profile_scope("h3.adaln"):
                modulation = self.adaln_proj(timestep_embedding).view(-1, 6 * self.hidden_size).to(hidden_states.dtype)
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

            with h3_profile_scope("h3.norm"):
                norm_hidden_states = self._norm_and_modulate(self.norm1, hidden_states, shift_attn, scale_attn, adaln_indices)
            with h3_profile_scope("h3.attn"):
                attn = self.attn
                if attn.opaque_attention:
                    query, key, value = attn.project_qkv(norm_hidden_states, rotary_emb)
                    attention = attn._attention_core_opaque(query, key, value, attention_mask)
                    with h3_profile_scope("h3.out"):
                        attention = attn.out_proj(attention)
                else:
                    attention = attn(norm_hidden_states, rotary_emb, attention_mask)
            with h3_profile_scope("h3.gate"):
                hidden_states = self._gated_residual(hidden_states, gate_attn.index_select(0, adaln_indices), attention)

            with h3_profile_scope("h3.norm"):
                norm_hidden_states = self._norm_and_modulate(self.norm2, hidden_states, shift_mlp, scale_mlp, adaln_indices)
            with h3_profile_scope("h3.mlp"):
                feed_forward = self.mlp(norm_hidden_states)
            with h3_profile_scope("h3.gate"):
                return self._gated_residual(hidden_states, gate_mlp.index_select(0, adaln_indices), feed_forward)


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig):
        super().__init__()
        video_patch_dim = config.in_channels * torch.Size(config.patch_size).numel()
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdaLNProjection(
            config.time_embed_dim,
            2 * config.hidden_size,
            apply_silu=config.adaln_t_table_size is None,
        )
        self.video_out = nn.Linear(config.hidden_size, video_patch_dim)
        self.audio_out = nn.Linear(config.hidden_size, config.audio_in_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        timestep_indices: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
    ) -> MiniMaxH3TransformerOutput:
        shift, scale = self.adaln_proj(timestep_embedding).to(hidden_states.dtype).chunk(2, dim=-1)
        media_indices = torch.cat((video_indices, audio_indices))
        media_timestep_indices = timestep_indices.index_select(0, media_indices)
        media = self.norm(hidden_states.index_select(1, media_indices))
        media = media * (1.0 + scale.index_select(0, media_timestep_indices))
        media = media + shift.index_select(0, media_timestep_indices)
        video_hidden, audio_hidden = media.split((video_indices.numel(), audio_indices.numel()), dim=1)
        video_dtype = self.video_out.weight.dtype
        audio_dtype = self.audio_out.weight.dtype
        video = self.video_out(video_hidden.to(video_dtype))
        audio = self.audio_out(audio_hidden.to(audio_dtype))
        return MiniMaxH3TransformerOutput(video=video, audio=audio)


class MiniMaxH3Transformer(nn.Module):
    """Single-stream MiniMax H3 video-audio transformer in the released key layout."""

    def __init__(self, config: MiniMaxH3TransformerConfig | None = None, *, attention_mode: str = "torch"):
        super().__init__()
        self.config = config or MiniMaxH3TransformerConfig()
        config = self.config
        rotary_dim = 2 * 3 * config.rope_freq_dim
        if rotary_dim > config.attention_head_dim:
            raise ValueError(f"H3 rotary width {rotary_dim} exceeds attention head width {config.attention_head_dim}")

        video_patch_dim = config.in_channels * torch.Size(config.patch_size).numel()
        self.video_patch_proj = nn.Linear(video_patch_dim, config.hidden_size)
        self.audio_patch_proj = nn.Linear(config.audio_in_channels, config.hidden_size)
        self.condition_proj = nn.Linear(config.text_dim, config.hidden_size)
        if config.adaln_t_table_size is None:
            self.time_embedder = MiniMaxH3TimeEmbedder(
                config.freq_dim,
                config.time_embed_hidden_dim,
                config.time_embed_dim,
            )
        else:
            if config.adaln_t_table_size < 2:
                raise ValueError("H3 AdaLN timestep table must contain at least two rows")
            self.time_embedder = None
            self.register_buffer(
                "adaln_t_table",
                torch.zeros(config.adaln_t_table_size, config.time_embed_dim),
                persistent=True,
            )
        self.rope = MiniMaxH3RotaryPosEmbed(config.rope_freq_dim, config.rope_theta)
        self.token_refiner = MiniMaxH3TokenRefiner(config, attention_mode)
        self.blocks = nn.ModuleList([MiniMaxH3TransformerBlock(config, attention_mode) for _ in range(config.num_layers)])
        self.final_layer = MiniMaxH3FinalLayer(config)
        self.gradient_checkpointing = False
        self.gradient_checkpointing_blocks: int | None = None
        self.checkpoint_keep = "none"
        self._checkpoint_saved = SavedActivations()
        self._checkpointed_blocks = 0
        self.activation_cpu_offloading = False
        self.activation_cpu_offload_pin_memory = False
        self.reusable_activation_offloader: ReusableActivationOffloader | None = None
        self.blocks_to_swap = 0
        self.offloader = None
        self.layer_streaming = False
        self.int8_attention_mode = "off"

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        self._validate_checkpoint_keep(activation_cpu_offloading=activation_cpu_offloading)
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def set_activation_cpu_offload_pin_memory(self, enabled: bool) -> None:
        self.activation_cpu_offload_pin_memory = bool(enabled)

    def enable_reusable_activation_offload(self) -> None:
        self.reusable_activation_offloader = ReusableActivationOffloader()

    def set_gradient_checkpointing_blocks(self, blocks: int | None) -> None:
        if blocks is not None and not 0 <= blocks <= len(self.blocks):
            raise ValueError(f"H3 gradient checkpoint block count must be in [0, {len(self.blocks)}], got {blocks}")
        self.gradient_checkpointing_blocks = blocks

    def set_checkpoint_keep(self, keep: str) -> None:
        """Choose what block checkpointing keeps instead of recomputing.

        ``none`` is plain checkpointing. ``attention`` keeps each block's fused
        attention output (and logsumexp), so recomputation skips the attention
        forward; ``qkv`` also keeps the QKV projection's matmul output. Weights
        are never kept, so block swap streams them for the recompute exactly as
        before; only activations change.
        """
        keep = validate_checkpoint_keep(keep)
        attention_modes = {
            module.attention_mode for block in self.blocks for module in block.modules() if isinstance(module, MiniMaxH3Attention)
        }
        if keep != "none":
            for attention_mode in sorted(attention_modes):
                if not attention_kernel_is_visible(attention_mode):
                    raise ValueError(
                        f"H3 checkpoint keep={keep!r} cannot see the {attention_mode!r} attention kernel: the installed "
                        "package does not register its torch.library ops, so nothing would be kept; use --sdpa or "
                        "a flash-attn build with registered ops"
                    )
        self._validate_checkpoint_keep(keep=keep)
        self.checkpoint_keep = keep
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, MiniMaxH3Attention):
                    module.checkpoint_keep_qkv = keep == "qkv"
                    if keep == "qkv":
                        module.install_projection_region()

    def _begin_checkpoint_pass(self) -> None:
        self._checkpoint_saved.reset()
        self._checkpointed_blocks = 0

    def _verify_checkpoint_pass(self) -> None:
        """Refuse to run on with a keep mode that kept nothing.

        Whether SDPA takes a fused backend is decided per batch from the mask,
        dtype and head width; the math backend is a composite of ordinary ops
        the policy cannot save. The projection matmul can likewise be hidden by
        a fused adapter kernel. Silently degrading to plain checkpointing would
        be worse than stopping, so a pass whose checkpointed blocks saved fewer
        outputs than there were blocks is an error.
        """
        blocks = self._checkpointed_blocks
        if self.checkpoint_keep == "none" or not blocks:
            return
        saved = self._checkpoint_saved
        if saved.attention < blocks:
            raise RuntimeError(
                f"H3 checkpoint keep={self.checkpoint_keep!r} saved {saved.attention} fused attention outputs for {blocks} "
                "checkpointed block calls: SDPA resolved to the math backend for this batch (mask, dtype or head width), "
                "which the policy cannot keep. Drop --h3_checkpoint_keep or make the batch eligible for a fused backend"
            )
        if self.checkpoint_keep == "qkv" and saved.projection < blocks:
            raise RuntimeError(
                f"H3 checkpoint keep='qkv' saved {saved.projection} QKV projection outputs for {blocks} checkpointed block "
                "calls: the base projection ran outside the dispatcher (a fused adapter kernel or a non-Linear path). "
                "Use --h3_checkpoint_keep attention"
            )

    def _validate_checkpoint_keep(self, *, keep: str | None = None, activation_cpu_offloading: bool | None = None) -> None:
        """Reject a keep mode the other memory options cannot serve; called before each of them changes."""
        keep = self.checkpoint_keep if keep is None else keep
        if keep == "none":
            return
        offloading = self.activation_cpu_offloading if activation_cpu_offloading is None else activation_cpu_offloading
        if offloading:
            raise ValueError(
                "H3 checkpoint keep cannot be combined with CPU offload of checkpoint activations: the kept tensors "
                "live in the selective checkpoint cache, which the offload hooks never see"
            )
        if self.blocks_to_swap and (self.layer_streaming or getattr(self.offloader, "recompute_requires_wait", False)):
            raise ValueError(
                "H3 checkpoint keep requires whole-block H2D-only block swap: layer streaming and the trainable ring "
                "transfer weights inside the checkpointed region, which the selective checkpoint cache cannot replay"
            )

    def enable_attention_auto_dispatch(self) -> None:
        for module in self.modules():
            if isinstance(module, MiniMaxH3Attention):
                if module.attention_mode != "torch":
                    raise ValueError("MiniMax H3 attention auto-dispatch requires SDPA attention")
                module.auto_dispatch = True

    def enable_fused_qk_norm_rope(self) -> None:
        for module in self.modules():
            if isinstance(module, MiniMaxH3Attention):
                module.fused_qk_norm_rope = True

    def enable_fused_indexed_adaln(self) -> None:
        for module in self.blocks:
            module.fused_indexed_adaln = True

    def enable_fused_elementwise(self) -> None:
        """AdaLN modulation and gated residuals as ``addcmul`` (one rounding instead of two)."""
        for block in self.blocks:
            block.fused_elementwise = True

    def set_compile_opaque_attention(self, enabled: bool) -> None:
        """Hide the attention kernel call from Dynamo (see ``MiniMaxH3Attention._attention_core_opaque``)."""
        for module in self.modules():
            if isinstance(module, MiniMaxH3Attention):
                module.opaque_attention = bool(enabled)

    def enable_fused_swiglu(self) -> None:
        for module in self.modules():
            if isinstance(module, MiniMaxH3FeedForward):
                module.fused_swiglu = True

    def set_swiglu_chunk_rows(self, rows: int) -> None:
        if rows < 0:
            raise ValueError("H3 SwiGLU chunk rows must be non-negative")
        for block in self.blocks:
            block.mlp.chunk_rows = rows

    def set_int8_attention_mode(self, mode: str) -> None:
        if mode not in {"off", "aux", "train"}:
            raise ValueError(f"unsupported H3 INT8 attention mode: {mode}")
        if mode != "off" and not HAS_TRITON:
            raise RuntimeError("H3 INT8 attention requires Triton")
        self.int8_attention_mode = mode
        enabled = mode == "train"
        for module in self.modules():
            if isinstance(module, MiniMaxH3Attention):
                module.int8_attention = enabled

    def set_block_sparse_attention(self, config: BlockSparseConfig | None, *, start_block: int = 0) -> None:
        """Enable block-sparse attention from ``start_block`` onwards.

        Every block sees the same packed sequence, so this is a quality knob,
        not a cost one: blocks before ``start_block`` keep exact full attention.
        """
        if config is not None:
            config.validate()
        self.block_sparse_config = config
        self.block_sparse_modules = []
        for index, block in enumerate(self.blocks):
            for module in block.modules():
                if isinstance(module, MiniMaxH3Attention):
                    module.block_sparse_config = config if index >= start_block else None
                    module.block_sparse_plan = None
                    if index >= start_block:
                        self.block_sparse_modules.append(module)

    def _refresh_block_sparse_plan(self, position_ids: torch.Tensor, token_tags: torch.Tensor) -> None:
        """Build the packed-sequence tile order once for each forward."""
        config = getattr(self, "block_sparse_config", None)
        if config is None or config.block_shape is None or not self.block_sparse_modules:
            return
        not_video = torch.nonzero(token_tags != int(MiniMaxH3TokenTag.VIDEO)).flatten()
        target_start = int(not_video[-1]) + 1 if not_video.numel() else 0
        plan = build_plan(position_ids.to(device=token_tags.device), target_start, config)
        for module in self.block_sparse_modules:
            module.block_sparse_plan = plan

    @contextmanager
    def int8_attention_context(self, *, auxiliary: bool):
        enabled = self.int8_attention_mode == "train" or (auxiliary and self.int8_attention_mode == "aux")
        modules = [module for module in self.modules() if isinstance(module, MiniMaxH3Attention)]
        previous = [module.int8_attention for module in modules]
        try:
            for module in modules:
                module.int8_attention = enabled
            yield
        finally:
            for module, was_enabled in zip(modules, previous):
                module.int8_attention = was_enabled

    def enable_block_swap(self, blocks_to_swap: int, config: BlockSwapConfig) -> None:
        num_blocks = len(self.blocks)
        if blocks_to_swap <= 0:
            raise ValueError("MiniMax H3 blocks_to_swap must be positive")
        # Whole-block rings retain two blocks to preserve overlap headroom.
        # Layer streaming owns an independent, smaller ring and can therefore
        # offload the Linear weights from every transformer block.
        layer_streaming = config.h2d_only and config.granularity == "layer"
        max_blocks_to_swap = num_blocks if layer_streaming else num_blocks - 2
        if blocks_to_swap > max_blocks_to_swap:
            raise ValueError(
                f"MiniMax H3 cannot swap more than {max_blocks_to_swap} of {num_blocks} blocks; requested {blocks_to_swap}"
            )
        self.blocks_to_swap = blocks_to_swap
        self.layer_streaming = layer_streaming
        self.offloader = create_offloader(
            "minimax-h3-block",
            self.blocks,
            num_blocks,
            blocks_to_swap,
            config,
        )
        self._validate_checkpoint_keep()

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        if self.blocks_to_swap:
            saved_blocks = self.blocks
            self.blocks = nn.ModuleList()
        self.to(device)
        if self.blocks_to_swap:
            self.blocks = saved_blocks
            if self.layer_streaming:
                self._move_layer_streaming_non_linears(device)

    def _move_layer_streaming_non_linears(self, device: torch.device) -> None:
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    continue
                for parameter in module.parameters(recurse=False):
                    parameter.data = parameter.data.to(device)
                for name, buffer in module.named_buffers(recurse=False):
                    setattr(module, name, buffer.to(device))

    def offload_block_swap_to_cpu(self) -> None:
        """Evacuate the transformer, including the active swap ring, between sequential inference stages."""
        if not self.blocks_to_swap:
            self.to("cpu")
            return
        if self.offloader is None:
            raise RuntimeError("MiniMax H3 block swap is enabled without an offloader")
        self.offloader.offload_to_cpu(self.blocks)
        self.move_to_device_except_swap_blocks(torch.device("cpu"))

    def prepare_block_swap_before_forward(self) -> None:
        if not self.blocks_to_swap:
            return
        if self.offloader is None:
            raise RuntimeError("MiniMax H3 block swap is enabled without an offloader")
        self.offloader.prepare_block_devices_before_forward(self.blocks)
        if self.layer_streaming:
            self._move_layer_streaming_non_linears(self.offloader.device)

    def switch_block_swap_for_inference(self) -> None:
        if self.blocks_to_swap:
            if self.offloader is None:
                raise RuntimeError("MiniMax H3 block swap is enabled without an offloader")
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self) -> None:
        if self.blocks_to_swap:
            if self.offloader is None:
                raise RuntimeError("MiniMax H3 block swap is enabled without an offloader")
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def _checkpointed_block(
        self,
        block: MiniMaxH3TransformerBlock,
        block_index: int,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        def forward(value: torch.Tensor) -> torch.Tensor:
            # The dense trainable ring advances from forward order to reverse
            # order during checkpoint recomputation. Classic block swap uses
            # module backward hooks and does not need this second wait.
            if self.blocks_to_swap and getattr(self.offloader, "recompute_requires_wait", False):
                self.offloader.wait_for_block(block_index)
            return block(value, timestep_embedding, adaln_indices, rotary_emb, attention_mask)

        if self.activation_cpu_offloading:
            # Let checkpoint keep its saved input on CPU while the live block
            # output stays on CUDA for the next block. Wrapping ``forward`` and
            # returning a CPU output would immediately copy that tensor back to
            # CUDA in the following block, adding a full D2H+H2D round trip per
            # layer without reducing the tensors retained for backward.
            if self.reusable_activation_offloader is not None:
                with self.reusable_activation_offloader.context(block_index):
                    return checkpoint(forward, hidden_states, use_reentrant=False)
            with torch.autograd.graph.save_on_cpu(pin_memory=self.activation_cpu_offload_pin_memory):
                return checkpoint(forward, hidden_states, use_reentrant=False)
        if self.checkpoint_keep != "none":
            # The policy keeps the fused attention output (and at ``qkv`` the
            # projection matmul) in the selective checkpoint cache; every other
            # op, weights included, is re-run by the recompute as before.
            self._checkpointed_blocks += 1
            context_fn = checkpoint_context_fn(self.checkpoint_keep, self._checkpoint_saved)
            return checkpoint(forward, hidden_states, use_reentrant=False, context_fn=context_fn)
        return checkpoint(forward, hidden_states, use_reentrant=False)

    def _time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        if self.time_embedder is not None:
            return self.time_embedder(timestep)
        table = self.adaln_t_table.float()
        position = timestep.to(device=table.device, dtype=torch.float32).clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long()
        upper = (lower + 1).clamp(max=table.shape[0] - 1)
        fraction = (position - lower.float()).unsqueeze(1)
        embedding = table.index_select(0, lower) * (1.0 - fraction) + table.index_select(0, upper) * fraction
        return embedding.to(timestep.device)

    @property
    def _checkpoint_start(self) -> int:
        return 0 if self.gradient_checkpointing_blocks is None else len(self.blocks) - self.gradient_checkpointing_blocks

    def _run_block(self, block: MiniMaxH3TransformerBlock, block_index: int, state: MiniMaxH3PackedState) -> torch.Tensor:
        """One block applied to one packed sequence, checkpointed or not.

        The checkpoint decision reads ``torch.is_grad_enabled()``, which is what
        separates a graph-carrying arm from a no-grad one when several arms share
        a single pass over the blocks.
        """
        if torch.is_grad_enabled() and self.gradient_checkpointing and block_index >= self._checkpoint_start:
            return self._checkpointed_block(
                block,
                block_index,
                state.hidden_states,
                state.timestep_embedding,
                state.adaln_indices,
                state.rotary_emb,
                state.attention_mask,
            )
        return block(
            state.hidden_states,
            state.timestep_embedding,
            state.adaln_indices,
            state.rotary_emb,
            state.attention_mask,
        )

    def _finalize(self, state: MiniMaxH3PackedState) -> MiniMaxH3TransformerOutput:
        hidden_states = state.hidden_states
        if self.activation_cpu_offloading:
            hidden_states = hidden_states.to(self.final_layer.norm.weight.device)
        with h3_profile_scope("h3.final"):
            return self.final_layer(
                hidden_states,
                state.timestep_embedding,
                state.timestep_indices,
                state.video_indices,
                state.audio_indices,
            )

    def _begin_backward_arms(self) -> Callable[[int], None] | None:
        """Open this traversal's announcement group, and return the announcer.

        Every graph-carrying pass over the blocks opens a group, fused or not: a
        step may traverse the blocks several times before its single backward,
        autograd unwinds those traversals newest-first, and a pass that announced
        nothing would otherwise have its hook firings consume an older pass's
        announcements and be suppressed -- ring scheduling and all.
        """
        if not self.blocks_to_swap:
            return None
        begin = getattr(self.offloader, "begin_backward_arms", None)
        if begin is not None:
            begin()
        return getattr(self.offloader, "note_backward_arm", None)

    def forward_fused(
        self,
        arms: Sequence[tuple[dict[str, Any], Callable[[], AbstractContextManager] | None]],
    ) -> list[MiniMaxH3TransformerOutput]:
        """Run several packed sequences through ONE pass over the blocks.

        Each arm is ``(forward kwargs, context factory)``: the kwargs are exactly
        what :meth:`forward` accepts, and the context -- re-entered around the
        arm's embedding stage, around every one of its block calls, and around its
        final layer -- is where a caller expresses what makes that arm different
        from the others. A frozen teacher arm passes ``torch.no_grad()`` with its
        adapter disabled; the trainable arm passes nothing.

        The arms do not share a tensor, only the weights: two packed sequences of
        different lengths -- which is what a privileged teacher presentation
        always is -- cannot be stacked on the batch axis, and concatenating them
        into one sequence would need a pairwise mask that costs quadratically and
        forecloses the flash and INT8 kernels. Interleaving instead keeps every
        arm's attention exactly the attention it would have had on its own, and
        isolation between arms is structural: an arm's rows never enter another
        arm's tensor at all.

        What it buys is the streaming: under ``--blocks_to_swap`` each swapped
        block is waited for once and released once per fused call, so N arms cost
        one block-ring traversal instead of N.
        """
        if not arms:
            raise ValueError("MiniMax H3 fused forward needs at least one arm")
        contexts = [context if context is not None else nullcontext for _, context in arms]
        states = []
        for (kwargs, _), context in zip(arms, contexts):
            with context():
                states.append(self._prepare(**kwargs))
        if self.reusable_activation_offloader is not None and torch.is_grad_enabled():
            self.reusable_activation_offloader.begin_forward()
        # An offloader that retires a swapped block from a backward hook counts
        # one firing per block per pass, because that is what a non-fused forward
        # produces. Here each block is invoked once per arm, so every arm that
        # builds a graph adds a firing, and the block must not be retired until
        # the last of them has run -- under gradient checkpointing the others have
        # not even re-run its forward yet. Announced per invocation rather than
        # from an arm count, so an arm whose output carries no graph (a frozen
        # teacher, or a pass with nothing trainable upstream) is not counted:
        # such an arm registers no backward node and fires no hook.
        note_backward_arm = self._begin_backward_arms()
        self._begin_checkpoint_pass()
        for block_index, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(block_index)
            for state, context in zip(states, contexts):
                with context():
                    hidden_states = self._run_block(block, block_index, state)
                    if note_backward_arm is not None and hidden_states.requires_grad:
                        note_backward_arm(block_index)
                    state.hidden_states = hidden_states
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.blocks, block_index)
        self._verify_checkpoint_pass()
        outputs = []
        for state, context in zip(states, contexts):
            with context():
                outputs.append(self._finalize(state))
        return outputs

    def forward(
        self,
        video_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> MiniMaxH3TransformerOutput:
        state = self._prepare(
            video_hidden_states=video_hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            attention_mask=attention_mask,
        )
        if self.reusable_activation_offloader is not None and torch.is_grad_enabled():
            self.reusable_activation_offloader.begin_forward()
        note_backward_arm = self._begin_backward_arms()
        self._begin_checkpoint_pass()
        for block_index, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(block_index)
            state.hidden_states = self._run_block(block, block_index, state)
            if note_backward_arm is not None and state.hidden_states.requires_grad:
                note_backward_arm(block_index)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.blocks, block_index)
        self._verify_checkpoint_pass()
        return self._finalize(state)

    def _prepare(
        self,
        video_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> MiniMaxH3PackedState:
        """Embed one packed sequence into the state the block loop advances."""
        with h3_profile_scope("h3.pack"):
            return self._prepare_packed(
                video_hidden_states,
                audio_hidden_states,
                encoder_hidden_states,
                timestep,
                timestep_indices,
                token_tags,
                position_ids,
                video_indices,
                audio_indices,
                text_indices,
                attention_mask,
            )

    def _prepare_packed(
        self,
        video_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> MiniMaxH3PackedState:
        sequence_length = position_ids.shape[0]
        if position_ids.shape != (sequence_length, 3):
            raise ValueError(f"position_ids must have shape (sequence_length, 3), got {tuple(position_ids.shape)}")
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError("token_tags and timestep_indices must match the packed sequence length")

        self._refresh_block_sparse_plan(position_ids, token_tags)
        rotary_emb = self.rope(position_ids)
        video = self.video_patch_proj(video_hidden_states.to(self.video_patch_proj.weight.dtype))
        audio = self.audio_patch_proj(audio_hidden_states.to(self.audio_patch_proj.weight.dtype))
        text = self.condition_proj(encoder_hidden_states.to(self.condition_proj.weight.dtype))
        with h3_profile_scope("h3.refiner"):
            text = self.token_refiner(text, self.gradient_checkpointing)

        hidden_states = text.new_zeros((text.shape[0], sequence_length, text.shape[-1]))
        hidden_states.index_copy_(1, text_indices, text)
        hidden_states.index_copy_(1, video_indices, video.to(text.dtype))
        hidden_states.index_copy_(1, audio_indices, audio.to(text.dtype))
        # RoPE is shared by all 50 blocks. Cast it once here instead of casting
        # cos/sin independently for Q and K inside every attention call.
        rotary_emb = tuple(component.to(hidden_states.dtype) for component in rotary_emb)

        timestep_embedding = self._time_embedding(timestep)
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_COUNT + token_tags.clamp(min=0)
        is_padding = token_tags < 0
        # A caller-supplied topology narrows attention further; the padding mask
        # is always applied on top so a custom mask cannot re-expose padding.
        # ``True`` means the pair may attend, matching SDPA's boolean contract.
        if attention_mask is not None:
            if attention_mask.dtype is not torch.bool:
                raise ValueError(f"H3 attention mask must be boolean, got {attention_mask.dtype}")
            if attention_mask.shape != (sequence_length, sequence_length):
                raise ValueError(
                    f"H3 attention mask must be [{sequence_length}, {sequence_length}], got {tuple(attention_mask.shape)}"
                )
            attention_mask = attention_mask.to(device=hidden_states.device)
        if bool(is_padding.any()):
            padding_mask = is_padding[None, :] == is_padding[:, None]
            attention_mask = padding_mask if attention_mask is None else attention_mask & padding_mask

        return MiniMaxH3PackedState(
            hidden_states=hidden_states,
            timestep_embedding=timestep_embedding,
            adaln_indices=adaln_indices,
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
            timestep_indices=timestep_indices,
            video_indices=video_indices,
            audio_indices=audio_indices,
        )
