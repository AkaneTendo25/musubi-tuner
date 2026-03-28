# Copied from https://github.com/nvidia-cosmos/cosmos-predict2.5
# cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0
#
# Modified for musubi-tuner: removed transformer_engine, megatron, FSDP, context parallelism,
# NATTEN dependencies. Replaced with PyTorch-native implementations.

import math
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

import logging

logger = logging.getLogger(__name__)


VideoSize = namedtuple("VideoSize", ["T", "H", "W"])


class DataType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    MIX = "mix"

    def __str__(self) -> str:
        return self.value


# ---------------------- RMSNorm -----------------------
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ---------------------- Feed Forward Network -----------------------
class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._hidden_dim)
        if self._layer_id is not None:
            std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


# ---------------------- Attention -----------------------
def torch_attention_op(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    flatten_heads: bool = True,
) -> torch.Tensor:
    """Scaled dot-product attention. Inputs [B, S, H, D]."""
    q = rearrange(q_B_S_H_D, "b s h d -> b h s d")
    k = rearrange(k_B_S_H_D, "b s h d -> b h s d")
    v = rearrange(v_B_S_H_D, "b s h d -> b h s d")
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    if flatten_heads:
        return rearrange(result, "b h s d -> b s (h d)")
    else:
        return rearrange(result, "b h s d -> b s h d")


def apply_rotary_pos_emb_simple(x, freqs):
    """Apply rotary position embedding without transformer_engine dependency."""
    # freqs shape: [seq_len, 1, 1, dim] -> we need to handle this
    # x shape: [B, S, H, D]
    # The cosmos RoPE format: freqs contains interleaved cos/sin values
    # freqs is [L, 1, 1, D] where D = head_dim and contains [cos, sin] concatenated
    half_dim = freqs.shape[-1] // 2
    cos_freqs = freqs[..., :half_dim]
    sin_freqs = freqs[..., half_dim:]

    # x shape: [B, S, H, D]
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    # Broadcast freqs to match x
    # freqs is [S, 1, 1, half_dim], x is [B, S, H, half_dim]
    out1 = x1 * cos_freqs - x2 * sin_freqs
    out2 = x2 * cos_freqs + x1 * sin_freqs
    return torch.cat([out1, out2], dim=-1)


class Attention(nn.Module):
    """Multi-head attention supporting self-attention and cross-attention."""

    def __init__(
        self,
        query_dim: int,
        context_dim=None,
        n_heads=8,
        head_dim=64,
        dropout=0.0,
        qkv_format: str = "bshd",
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.use_wan_fp32_strategy = use_wan_fp32_strategy

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        self.attn_op = torch_attention_op

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._query_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._context_dim)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)

        for layer in [self.q_norm, self.k_norm, self.v_norm]:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(self, x, context=None, rope_emb=None):
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        if self.is_selfattn and rope_emb is not None:
            original_dtype = q.dtype
            if self.use_wan_fp32_strategy:
                q = q.to(torch.float32)
                k = k.to(torch.float32)
            q = apply_rotary_pos_emb_simple(q, rope_emb)
            k = apply_rotary_pos_emb_simple(k, rope_emb)
            if self.use_wan_fp32_strategy:
                q = q.to(original_dtype)
                k = k.to(original_dtype)

        return q, k, v

    def forward(
        self,
        x,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        result = self.attn_op(q, k, v)
        return self.output_dropout(self.output_proj(result))


# ---------------------- Positional Embeddings -----------------------
class VideoRopePosition3DEmb(nn.Module):
    def __init__(
        self,
        *,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.enable_fps_modulation = enable_fps_modulation
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t

        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().to(self.dim_spatial_range.device)
        self.dim_spatial_range = (
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(self.dim_spatial_range.device) / dim_h
        )
        self.dim_temporal_range = (
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(self.dim_spatial_range.device) / dim_t
        )

    def forward(
        self,
        x_B_T_H_W_C: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
    ):
        B_T_H_W_C = x_B_T_H_W_C.shape
        return self.generate_embeddings(B_T_H_W_C, fps=fps)

    def generate_embeddings(
        self,
        B_T_H_W_C,
        fps: Optional[torch.Tensor] = None,
    ):
        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.float())
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.float())
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.float())

        B, T, H, W, _ = B_T_H_W_C
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        if self.enable_fps_modulation:
            if fps is None:
                half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
            else:
                half_emb_t = torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()


class LearnablePosEmbAxis(nn.Module):
    def __init__(
        self,
        *,
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        **kwargs,
    ):
        super().__init__()
        self.interpolation = interpolation
        self.model_channels = model_channels

        self.pos_emb_h = nn.Parameter(torch.zeros(len_h, model_channels))
        self.pos_emb_w = nn.Parameter(torch.zeros(len_w, model_channels))
        self.pos_emb_t = nn.Parameter(torch.zeros(len_t, model_channels))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.model_channels)
        torch.nn.init.trunc_normal_(self.pos_emb_h, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_emb_w, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_emb_t, std=std, a=-3 * std, b=3 * std)

    def forward(self, x_B_T_H_W_C: torch.Tensor, fps=None) -> torch.Tensor:
        B, T, H, W, _ = x_B_T_H_W_C.shape
        emb_h_H = self.pos_emb_h[:H]
        emb_w_W = self.pos_emb_w[:W]
        emb_t_T = self.pos_emb_t[:T]
        emb = (
            repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W)
            + repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W)
            + repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H)
        )
        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(1e-6, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return emb / norm.to(emb.dtype)


# ---------------------- Timestep Embedding -----------------------
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Timesteps(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T):
        assert timesteps_B_T.ndim == 2, f"Expected 2D input, got {timesteps_B_T.ndim}"
        in_dtype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return rearrange(emb.to(dtype=in_dtype), "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.in_dim)
        torch.nn.init.trunc_normal_(self.linear_1.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self.out_dim)
        torch.nn.init.trunc_normal_(self.linear_2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_T_3D = emb
            emb_B_T_D = sample
        else:
            emb_B_T_D = emb
            adaln_lora_B_T_3D = None

        return emb_B_T_D, adaln_lora_B_T_3D


# ---------------------- Patch Embedding -----------------------
class PatchEmbed(nn.Module):
    def __init__(
        self,
        spatial_patch_size,
        temporal_patch_size,
        in_channels=3,
        out_channels=768,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False
            ),
        )
        self.dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return x


# ---------------------- Final Layer -----------------------
class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        spatial_patch_size,
        temporal_patch_size,
        out_channels,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__()
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            torch.nn.init.trunc_normal_(self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)
        self.layer_norm.reset_parameters()

    def forward(
        self,
        x_B_T_H_W_D,
        emb_B_T_D,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ):
        with torch.amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                assert adaln_lora_B_T_3D is not None
                shift_B_T_D, scale_B_T_D = (
                    self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
                ).chunk(2, dim=-1)
            else:
                shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

            shift_B_T_1_1_D = rearrange(shift_B_T_D, "b t d -> b t 1 1 d")
            scale_B_T_1_1_D = rearrange(scale_B_T_D, "b t d -> b t 1 1 d")

            x_B_T_H_W_D = self.layer_norm(x_B_T_H_W_D) * (1 + scale_B_T_1_1_D) + shift_B_T_1_1_D
            x_B_T_H_W_O = self.linear(x_B_T_H_W_D)
        return x_B_T_H_W_O


# ---------------------- Transformer Block -----------------------
class CosmosBlock(nn.Module):
    """Transformer block: self-attention + cross-attention + MLP with AdaLN modulation."""

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            x_dim, None, num_heads, x_dim // num_heads, qkv_format="bshd",
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(
            x_dim, context_dim, num_heads, x_dim // num_heads, qkv_format="bshd",
        )

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

        self.use_wan_fp32_strategy = use_wan_fp32_strategy

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            from torch.utils.checkpoint import checkpoint as torch_checkpoint

            forward_fn = self._forward
            if self.activation_cpu_offloading:
                from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper

                forward_fn = create_cpu_offloading_wrapper(forward_fn, x_B_T_H_W_D.device)
            return torch_checkpoint(
                forward_fn,
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D,
                adaln_lora_B_T_3D,
                extra_per_block_pos_emb,
                use_reentrant=False,
            )
        return self._forward(
            x_B_T_H_W_D, emb_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, extra_per_block_pos_emb
        )

    def _forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        with torch.amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                shift_sa, scale_sa, gate_sa = (
                    self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_ca, scale_ca, gate_ca = (
                    self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_mlp, scale_mlp, gate_mlp = (
                    self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
            else:
                shift_sa, scale_sa, gate_sa = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
                shift_ca, scale_ca, gate_ca = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
                shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # Reshape for broadcasting: (B, T, D) -> (B, T, 1, 1, D)
        def _expand(t):
            return rearrange(t, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        shift_sa, scale_sa, gate_sa = _expand(shift_sa), _expand(scale_sa), _expand(gate_sa)
        shift_ca, scale_ca, gate_ca = _expand(shift_ca), _expand(scale_ca), _expand(gate_ca)
        shift_mlp, scale_mlp, gate_mlp = _expand(shift_mlp), _expand(scale_mlp), _expand(gate_mlp)

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _adaln(x, norm, scale, shift):
            return norm(x) * (1 + scale) + shift

        # Self-attention
        normed = _adaln(x_B_T_H_W_D, self.layer_norm_self_attn, scale_sa, shift_sa)
        sa_out = rearrange(
            self.self_attn(
                rearrange(normed, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d", t=T, h=H, w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_sa * sa_out

        # Cross-attention
        normed = _adaln(x_B_T_H_W_D, self.layer_norm_cross_attn, scale_ca, shift_ca)
        ca_out = rearrange(
            self.cross_attn(
                rearrange(normed, "b t h w d -> b (t h w) d"),
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d", t=T, h=H, w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_ca * ca_out

        # MLP
        normed = _adaln(x_B_T_H_W_D, self.layer_norm_mlp, scale_mlp, shift_mlp)
        mlp_out = self.mlp(normed)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp * mlp_out

        return x_B_T_H_W_D


# ---------------------- Main Model -----------------------
class CosmosDiT(nn.Module):
    """
    Cosmos Predict 2.5 DiT model, ported from MiniTrainDIT.
    Cleaned up for musubi-tuner: no transformer_engine, megatron, FSDP, context parallelism, NATTEN.
    """

    def __init__(
        self,
        max_img_h: int = 960,
        max_img_w: int = 960,
        max_frames: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        concat_padding_mask: bool = True,
        model_channels: int = 5120,
        num_blocks: int = 36,
        num_heads: int = 40,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        pos_emb_cls: str = "rope3d",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_cls = pos_emb_cls
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        self.use_crossattn_projection = use_crossattn_projection

        # Patch embedding
        embed_in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=embed_in_channels,
            out_channels=model_channels,
        )

        # Positional embedding
        kwargs = dict(
            model_channels=model_channels,
            len_h=max_img_h // patch_spatial,
            len_w=max_img_w // patch_spatial,
            len_t=max_frames // patch_temporal,
            max_fps=max_fps,
            min_fps=min_fps,
            head_dim=model_channels // num_heads,
            h_extrapolation_ratio=rope_h_extrapolation_ratio,
            w_extrapolation_ratio=rope_w_extrapolation_ratio,
            t_extrapolation_ratio=rope_t_extrapolation_ratio,
            enable_fps_modulation=rope_enable_fps_modulation,
        )
        self.pos_embedder = VideoRopePosition3DEmb(**kwargs)

        if extra_per_block_abs_pos_emb:
            kwargs["h_extrapolation_ratio"] = extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = extra_t_extrapolation_ratio
            kwargs["interpolation"] = "crop"
            self.extra_pos_embedder = LearnablePosEmbAxis(**kwargs)

        # Timestep embedding
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )

        # Cross-attention projection
        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
                nn.GELU(),
            )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                CosmosBlock(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    use_wan_fp32_strategy=use_wan_fp32_strategy,
                )
                for _ in range(num_blocks)
            ]
        )

        # Final layer
        self.final_layer = FinalLayer(
            hidden_size=model_channels,
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            out_channels=out_channels,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

        # Timestep embedding norm (replaced te.pytorch.RMSNorm with our RMSNorm)
        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)

        # Block swap and gradient checkpointing state
        self.blocks_to_swap = 0
        self.offloader = None
        self.gradient_checkpointing = False

    # region gradient checkpointing

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        for block in self.blocks:
            block.enable_gradient_checkpointing(activation_cpu_offloading)
        logger.info(f"CosmosDiT: Gradient checkpointing enabled. Activation CPU offloading: {activation_cpu_offloading}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        for block in self.blocks:
            block.disable_gradient_checkpointing()
        logger.info("CosmosDiT: Gradient checkpointing disabled.")

    # endregion

    # region block swap

    def enable_block_swap(self, blocks_to_swap: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False):
        from musubi_tuner.modules.custom_offloading_utils import ModelOffloader

        self.blocks_to_swap = blocks_to_swap
        num_blocks = len(self.blocks)

        assert blocks_to_swap <= num_blocks - 1, (
            f"Cannot swap more than {num_blocks - 1} blocks. Requested {blocks_to_swap} blocks to swap."
        )

        self.offloader = ModelOffloader(
            "cosmos_block", self.blocks, num_blocks, blocks_to_swap, supports_backward, device, use_pinned_memory
        )
        logger.info(
            f"CosmosDiT: Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks}. "
            f"Supports backward: {supports_backward}"
        )

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            save_blocks = self.blocks
            self.blocks = None

        self.to(device)

        if self.blocks_to_swap:
            self.blocks = save_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    # endregion

    # region fp8

    def apply_fp8_optimization(self, state_dict, device, move_to_device=True, use_scaled_mm=False):
        from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8

        target_keys = ["blocks"]
        exclude_keys = [
            "layer_norm", "q_norm", "k_norm", "v_norm",
            "x_embedder", "pos_embedder", "extra_pos_embedder",
            "t_embedder", "t_embedding_norm", "final_layer",
            "crossattn_proj", "img_context_proj",
            "adaln_modulation",
        ]

        state_dict = optimize_state_dict_with_fp8(
            state_dict, device, target_keys, exclude_keys, move_to_device=move_to_device
        )
        apply_fp8_monkey_patch(self, state_dict, use_scaled_mm=use_scaled_mm)
        return state_dict

    # endregion

    def unpatchify(self, x_B_T_H_W_M):
        x_B_C_Tt_Hp_Wp = rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
            t=self.patch_temporal,
        )
        return x_B_C_Tt_Hp_Wp

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        if self.concat_padding_mask:
            if padding_mask is not None:
                padding_mask = F.interpolate(
                    padding_mask.unsqueeze(1).float(),
                    size=x_B_C_T_H_W.shape[-2:],
                    mode="nearest",
                )
                x_B_C_T_H_W = torch.cat(
                    [x_B_C_T_H_W, padding_mask.repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
                )
            else:
                # Add ones padding mask channel
                padding_channel = torch.ones(
                    x_B_C_T_H_W.shape[0], 1, *x_B_C_T_H_W.shape[2:],
                    device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype,
                )
                x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_channel], dim=1)

        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb

        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)
        return x_B_T_H_W_D, None, extra_pos_emb

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_B_C_T_H_W: (B, C, T, H, W) latent input
            timesteps_B_T: (B,) or (B, T) timesteps
            crossattn_emb: (B, N, D) text embeddings
            fps: optional fps tensor
            padding_mask: optional padding mask
        """
        x_B_T_H_W_D, rope_emb, extra_pos_emb = self.prepare_embedded_sequence(
            x_B_C_T_H_W, fps=fps, padding_mask=padding_mask,
        )

        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        with torch.amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        if extra_pos_emb is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb.shape

        if self.blocks_to_swap:
            from musubi_tuner.utils.device_utils import clean_memory_on_device

            clean_memory_on_device(x_B_T_H_W_D.device)

        input_device = x_B_T_H_W_D.device
        for block_idx, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(block_idx)

            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D=rope_emb,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb,
            )

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.blocks, block_idx)

        if x_B_T_H_W_D.device != input_device:
            x_B_T_H_W_D = x_B_T_H_W_D.to(input_device)

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)

        return x_B_C_Tt_Hp_Wp


# ---------------------- Model Configs -----------------------
COSMOS_2B_CONFIG = dict(
    max_img_h=240,  # latent space height (pixel 1920 / VAE 8x)
    max_img_w=240,  # latent space width (pixel 1920 / VAE 8x)
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    mlp_ratio=4.0,
    crossattn_emb_channels=1024,
    # Qwen2.5-VL-7B FULL_CONCAT: 28 layers × 3584 hidden_size = 100352
    use_crossattn_projection=True,
    crossattn_proj_in_channels=100352,
    pos_emb_cls="rope3d",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    extra_per_block_abs_pos_emb=False,
    rope_t_extrapolation_ratio=1.0,
    use_wan_fp32_strategy=False,
)

COSMOS_14B_CONFIG = dict(
    max_img_h=240,  # latent space height (pixel 1920 / VAE 8x)
    max_img_w=240,  # latent space width (pixel 1920 / VAE 8x)
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    model_channels=5120,
    num_blocks=36,
    num_heads=40,
    mlp_ratio=4.0,
    crossattn_emb_channels=1024,
    # Qwen2.5-VL-7B FULL_CONCAT: 28 layers × 3584 hidden_size = 100352
    use_crossattn_projection=True,
    crossattn_proj_in_channels=100352,
    pos_emb_cls="rope3d",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    extra_per_block_abs_pos_emb=False,
    rope_t_extrapolation_ratio=1.0,
    use_wan_fp32_strategy=False,
)

COSMOS_CONFIGS = {
    "2B": COSMOS_2B_CONFIG,
    "14B": COSMOS_14B_CONFIG,
}


def load_cosmos_dit(
    config_name: str,
    checkpoint_path: str,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    fp8_scaled: bool = False,
    use_scaled_mm: bool = False,
) -> CosmosDiT:
    """Load a Cosmos DiT model from checkpoint.

    Args:
        config_name: "2B" or "14B"
        checkpoint_path: path to .safetensors or .pt checkpoint
        device: device to load weights to
        dtype: model weight dtype (None for fp8_scaled)
        fp8_scaled: whether to quantize weights to fp8
        use_scaled_mm: whether to use scaled matmul for fp8 (requires SM 8.9+)
    """
    from accelerate import init_empty_weights

    config = COSMOS_CONFIGS[config_name]

    with init_empty_weights():
        model = CosmosDiT(**config)
        if dtype is not None:
            model.to(dtype)

    logger.info(f"Loading Cosmos DiT weights from {checkpoint_path}")

    if fp8_scaled:
        from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8

        target_keys = ["blocks"]
        exclude_keys = [
            "layer_norm", "q_norm", "k_norm", "v_norm",
            "x_embedder", "pos_embedder", "extra_pos_embedder",
            "t_embedder", "t_embedding_norm", "final_layer",
            "crossattn_proj", "adaln_modulation",
        ]

        sd = load_safetensors_with_lora_and_fp8(
            model_files=checkpoint_path,
            fp8_optimization=True,
            calc_device=torch.device(device) if isinstance(device, str) else device,
            move_to_device=True,
            target_keys=target_keys,
            exclude_keys=exclude_keys,
        )
    else:
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd = load_file(checkpoint_path, device=device)
        else:
            sd = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Strip "net." prefix from Cosmos checkpoints (their training wrapper adds this)
    for key in list(sd.keys()):
        if key.startswith("net."):
            sd[key[4:]] = sd.pop(key)

    # Strip "net_ema." keys (EMA weights, not needed for training)
    for key in list(sd.keys()):
        if key.startswith("net_ema."):
            del sd[key]

    # Handle "model." or "state_dict." wrapper keys (some checkpoint formats)
    for prefix in ["model.", "state_dict."]:
        if any(k.startswith(prefix) for k in sd.keys()):
            for key in list(sd.keys()):
                if key.startswith(prefix):
                    sd[key[len(prefix):]] = sd.pop(key)

    # Strip training stat buffers that may be in the checkpoint but not in our model
    for key in list(sd.keys()):
        if key.startswith("accum_"):
            del sd[key]

    if fp8_scaled:
        from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch

        apply_fp8_monkey_patch(model, sd, use_scaled_mm=use_scaled_mm)

    info = model.load_state_dict(sd, strict=False, assign=True)
    if info.missing_keys:
        logger.warning(f"Missing keys when loading Cosmos DiT ({len(info.missing_keys)}): {info.missing_keys[:10]}...")
    if info.unexpected_keys:
        logger.warning(f"Unexpected keys when loading Cosmos DiT ({len(info.unexpected_keys)}): {info.unexpected_keys[:10]}...")

    return model
