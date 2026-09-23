"""YuE2 AR/NAR two-stack model with the split AR-prefill / NAR forward and the musubi mechanism API.

Module tree names are load-bearing for LoRA targets, caching and training code:
``ar.{embed_tokens, blocks.N, lm_head}``, ``nar.{blocks.N, vae2llm, llm2vae, time_embedder.mlp.{0,2},
latent_pos_embed.pe}``, shared ``norm``; each block has ``input_layernorm``, ``self_attn.{qkv_proj, o_proj, q_norm,
k_norm}``, ``post_attention_layernorm``, ``mlp.{gate_up_proj, down_proj}``.

Numerics follow the official YuE2 reference implementation exactly (RMSNorm casts the rsqrt factor to x.dtype before
the multiply; RoPE casts cos/sin to x.dtype; split-half rotary after q/k-norm; SwiGLU ``down(silu(g) * u)``). No module
registers non-persistent buffers or keeps module-level tensor caches, and nothing here enters ``torch.inference_mode``.

The reference evaluates AR and NAR tokens in one sequence with ``torch.where`` routing and a hybrid mask. The AR
positions never attend to NAR positions, so the same result is obtained by a causal AR pass that exports each layer's
post-RoPE K/V, followed by a NAR pass whose queries attend (without a mask) to ``cat(K_ar[:visible], K_nar)``.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from musubi_tuner.modules.convrot_int8_kernels import _build_hadamard, _rotate_activation
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader, weighs_to_device
from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper
from musubi_tuner.yue2.yue2_attention import yue2_attention

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class YuE2Config:
    hidden_size: int = 2048
    num_layers: int = 28
    num_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    vocab_size: int = 184704
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_position_embeddings: int = 24576
    latent_dim: int = 64
    max_latent_frames: int = 24576
    timestep_shift: float = 1.0
    time_freq_dim: int = 256

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @classmethod
    def tiny(cls, **kw) -> "YuE2Config":
        """Test configuration: hidden 256 (keeps the ConvRot group size 256 legal), 3 layers, 4 heads of 64, 2 KV heads."""
        values = dict(
            hidden_size=256,
            num_layers=3,
            num_heads=4,
            num_kv_heads=2,
            head_dim=64,
            intermediate_size=512,
            vocab_size=1024,
            max_latent_frames=128,
        )
        values.update(kw)
        return cls(**values)


def yue2_t_embed_input(t: torch.Tensor, mode: str = "bf16") -> torch.Tensor:
    """Time-embedder input for flow time ``t`` in [0, 1] (training and the ODE sampler share it).

    ``bf16``: ``sigmoid(bf16(clamp(logit(t.double()), -20, 20))).float()`` with the sigmoid evaluated in bf16, equal to
    the reference ``_shift_t_value``; ``fp32``: ``t`` unchanged. Output is fp32.
    """
    t = torch.as_tensor(t)
    if mode == "fp32":
        return t.float()
    if mode != "bf16":
        raise ValueError(f"unknown t_embed mode: {mode}")
    raw = torch.logit(t.double()).clamp(-20, 20)
    return torch.sigmoid(raw.to(torch.bfloat16)).float()


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE tables for integer ``positions [L]``: fp32 ``(cos, sin)``, each ``[L, head_dim // 2]``, on ``device``.

    Built on every call; nothing is cached, so no buffer or module-level tensor state is needed.
    """
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
    inv_freq = 1.0 / (theta**exponents)
    pos = positions.to(device=device).float()
    angle = pos[:, None] * inv_freq[None, :]
    return torch.cos(angle), torch.sin(angle)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Split-half rotary on ``[B, L, H, D]``; cos/sin cast to ``x.dtype``: ``cat[x1*c - x2*s, x2*c + x1*s]``."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.to(x.dtype)[:, None, :]
    sin = sin.to(x.dtype)[:, None, :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class YuE2RMSNorm(nn.Module):
    """``x * rsqrt(mean(x.float()^2) + eps).to(x.dtype) * weight``."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


ATTN_DEFAULTS = {"attn_mode": "torch", "split_attn": False, "query_tile": 256, "sdpa_gqa": "repeat"}


class YuE2Attention(nn.Module):
    """Grouped-query self-attention: fused ``qkv_proj`` (rows q | k | v), per-head RMSNorm on q and k, RoPE, ``o_proj``.

    ``attn_kwargs`` are the backend options forwarded to ``yue2_attention`` (one dict per layer, see
    ``YuE2Model.set_attention``).
    """

    def __init__(self, config: YuE2Config):
        super().__init__()
        self.n_q_heads = config.num_heads
        self.n_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.split_sizes = (config.q_dim, config.kv_dim, config.kv_dim)
        self.qkv_proj = nn.Linear(config.hidden_size, sum(self.split_sizes), bias=False)
        self.o_proj = nn.Linear(config.q_dim, config.hidden_size, bias=False)
        self.q_norm = YuE2RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = YuE2RMSNorm(config.head_dim, config.rms_norm_eps)
        self.attn_kwargs = dict(ATTN_DEFAULTS)

    def _qkv_heads(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        q, k, v = torch.split(self.qkv_proj(h), self.split_sizes, dim=-1)
        q = q.unflatten(-1, (self.n_q_heads, self.head_dim))
        k = k.unflatten(-1, (self.n_kv_heads, self.head_dim))
        v = v.unflatten(-1, (self.n_kv_heads, self.head_dim))
        q = apply_rotary(self.q_norm(q), cos, sin)
        k = apply_rotary(self.k_norm(k), cos, sin)
        return q, k, v

    def forward(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        prefix_k: Optional[torch.Tensor] = None,
        prefix_v: Optional[torch.Tensor] = None,
        causal: bool = True,
        cache: Optional["YuE2KVCache"] = None,
        layer: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``h [B, L, C]`` (already normed) -> ``(out [B, L, C], k, v)``, k/v being this call's own post-RoPE keys and
        raw values ``[B, L, KVH, D]``.

        Key set: the cache contents plus the new rows when ``cache`` is given, ``cat(prefix, own)`` without a mask when
        a prefix is given, otherwise the own rows (masked when ``causal``).
        """
        q, k, v = self._qkv_heads(h, cos, sin)
        if cache is not None:
            keys, values = cache.append(layer, k, v)
            # a single query row may see the whole filled cache, so it needs no mask
            causal = q.shape[1] > 1
        elif prefix_k is not None:
            keys = torch.cat((prefix_k, k), dim=1)
            values = torch.cat((prefix_v, v), dim=1)
            causal = False
        else:
            keys, values = k, v
        out = yue2_attention(q, keys, values, causal=causal, **self.attn_kwargs)
        return self.o_proj(out), k, v


class YuE2MLP(nn.Module):
    """SwiGLU with a fused ``gate_up_proj`` (rows gate | up) and ``down_proj``."""

    def __init__(self, config: YuE2Config):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = torch.chunk(self.gate_up_proj(x), 2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class YuE2Block(nn.Module):
    """Pre-norm decoder layer shared by the AR and NAR stacks (the class name is the LoRA target)."""

    def __init__(self, config: YuE2Config):
        super().__init__()
        self.input_layernorm = YuE2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = YuE2Attention(config)
        self.post_attention_layernorm = YuE2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = YuE2MLP(config)

    def _ffn_residual(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.post_attention_layernorm(x))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        prefix_k: Optional[torch.Tensor] = None,
        prefix_v: Optional[torch.Tensor] = None,
        causal: bool = True,
        return_kv: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns ``(x_out, k, v)``; k/v are the layer's own post-RoPE K and raw V when ``return_kv``, else None.

        A prefix (NAR pass) is attended together with the own rows and disables the causal mask.
        """
        attn_out, k, v = self.self_attn(self.input_layernorm(x), cos, sin, prefix_k, prefix_v, causal)
        x = self._ffn_residual(x + attn_out)
        if not return_kv:
            return x, None, None
        return x, k, v

    def forward_with_cache(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: "YuE2KVCache", layer: int
    ) -> torch.Tensor:
        """Inference step: appends the new K/V to ``cache`` at ``cache.length`` and attends over all filled rows."""
        attn_out, _, _ = self.self_attn(self.input_layernorm(x), cos, sin, cache=cache, layer=layer)
        return self._ffn_residual(x + attn_out)


class YuE2TimestepEmbedder(nn.Module):
    """``mlp = Sequential(Linear(time_freq_dim, H), SiLU, Linear(H, H))`` over ``cat[cos(t*f), sin(t*f)]``."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """``t [B]`` (the ``yue2_t_embed_input`` value) -> ``[B, H]``; frequencies ``exp(-ln(1e4) * arange(128) / 128)``."""
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(dtype))


class YuE2LatentPosEmbed(nn.Module):
    """Persistent ``pe [max_frames, H]`` buffer loaded from the checkpoint (never recomputed; the sinusoid below is
    only the initial value of an unloaded model)."""

    def __init__(self, max_frames: int, hidden_size: int):
        super().__init__()
        pe = torch.zeros(max_frames, hidden_size)
        if pe.device.type != "meta":
            position = torch.arange(0, max_frames, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        self.max_frames = max_frames
        self.register_buffer("pe", pe)

    def forward(self, n: int, dtype: torch.dtype) -> torch.Tensor:
        """``pe[arange(n).clamp(max=max_frames - 1)].to(dtype)`` -> ``[n, H]``."""
        idx = torch.arange(n, device=self.pe.device).clamp(max=self.max_frames - 1)
        return self.pe[idx].to(dtype)


class YuE2Int8Embedding(nn.Module):
    """Pre-quantized ConvRot int8 token embedding: int8 ``weight [V, H]`` + ``scale [V, 1]`` buffers.

    ``forward(ids) = _rotate_activation(weight[ids].float() * scale[ids], h, group_size).to(bf16)`` (row gather, then
    un-rotate the gathered rows with the cached Hadamard), equal to gathering from the full dequantized table.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        self.out_dtype = dtype
        self.register_buffer("weight", torch.zeros(num_embeddings, embedding_dim, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(num_embeddings, 1, dtype=torch.float32))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.to(self.weight.device)
        rows = self.weight[ids].float() * self.scale[ids]
        h = _build_hadamard(self.group_size, device=rows.device, dtype=torch.float32)
        return _rotate_activation(rows, h, self.group_size).to(self.out_dtype)


def _make_blocks(config: YuE2Config) -> nn.ModuleList:
    return nn.ModuleList(YuE2Block(config) for _ in range(config.num_layers))


class YuE2AR(nn.Module):
    """Token stack: ``embed_tokens`` (``nn.Embedding``, may be swapped for ``YuE2Int8Embedding`` by the loader),
    ``blocks`` and ``lm_head``."""

    def __init__(self, config: YuE2Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = _make_blocks(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


class YuE2NAR(nn.Module):
    """Latent stack: ``blocks`` plus the latent I/O projections, the time embedder and the latent position table."""

    def __init__(self, config: YuE2Config):
        super().__init__()
        self.blocks = _make_blocks(config)
        self.vae2llm = nn.Linear(config.latent_dim, config.hidden_size)
        self.llm2vae = nn.Linear(config.hidden_size, config.latent_dim)
        self.time_embedder = YuE2TimestepEmbedder(config.hidden_size, config.time_freq_dim)
        self.latent_pos_embed = YuE2LatentPosEmbed(config.max_latent_frames, config.hidden_size)


@dataclass
class YuE2AROutput:
    hidden: Optional[torch.Tensor]  # norm(x) [B, L, H], or None when return_hidden=False
    kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]]  # per-layer post-RoPE (k, v) [B, n, KVH, D], or None


class YuE2KVCache:
    """Preallocated K/V storage for cached AR inference.

    ``store[layer, 0]`` holds the keys and ``store[layer, 1]`` the values of a layer, each ``[B, capacity, KVH, D]``.
    ``length`` is the number of filled positions (the same for every layer); it is advanced by the model once all
    layers have appended a step.
    """

    def __init__(self, num_layers: int, batch: int, capacity: int, num_kv_heads: int, head_dim: int, device, dtype):
        self.capacity = capacity
        self.length = 0
        self.store = torch.zeros((num_layers, 2, batch, capacity, num_kv_heads, head_dim), device=device, dtype=dtype)

    @property
    def num_layers(self) -> int:
        return self.store.shape[0]

    def ensure_room(self, n: int) -> None:
        if self.length + n > self.capacity:
            raise ValueError(f"YuE2 KV cache capacity {self.capacity} is too small for {self.length} + {n} positions")

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Writes ``k``/``v`` after the filled rows of ``layer``; returns the filled views including them."""
        n = k.shape[1]
        self.ensure_room(n)
        end = self.length + n
        keys, values = self.store[layer].unbind(0)
        keys[:, self.length : end] = k
        values[:, self.length : end] = v
        return keys[:, :end], values[:, :end]


def _visible_kv(
    k: torch.Tensor, v: torch.Tensor, kv_visible: Optional[int], valid_len: Optional[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncates a layer's K/V to the visible prefix; a shortened slice is cloned so its full storage can be freed."""
    n = k.shape[1]
    for limit in (kv_visible, valid_len):
        if limit is not None:
            n = min(n, limit)
    if n == k.shape[1]:
        return k, v
    return k[:, :n].clone(), v[:, :n].clone()


def _get_path(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parent_path, _, attr = path.rpartition(".")
    return (root.get_submodule(parent_path) if parent_path else root), attr


def _hide_modules(root: nn.Module, paths: Sequence[str]) -> list[tuple[nn.Module, str, nn.Module]]:
    """Detach submodules so ``root.to()`` skips them; ``_restore_modules`` puts them back."""
    hidden = []
    for path in paths:
        parent, attr = _get_path(root, path)
        module = parent._modules.get(attr)
        if module is not None:
            hidden.append((parent, attr, module))
            parent._modules[attr] = None
    return hidden


def _restore_modules(hidden: list[tuple[nn.Module, str, nn.Module]]) -> None:
    for parent, attr, module in reversed(hidden):
        parent._modules[attr] = module


class YuE2Model(nn.Module):
    """AR stack, NAR stack and the shared final norm in one module (the trainer's single ``transformer``).

    Attributes: ``config``, ``ar``, ``nar``, ``norm``, ``compute_dtype`` (property: ``norm.weight.dtype``),
    ``is_convrot_int8``, ``blocks_to_swap``, ``gradient_checkpointing``, ``activation_cpu_offloading``, and (set by
    the loader) ``checkpoint_layout`` and ``base_quant``.

    Block swap: two offloaders (``ar`` over ``ar.blocks``, ``nar`` over ``nar.blocks``). ``_run_stack`` switches a
    backward-capable offloader to forward-only for no-grad passes and back for grad passes, allows at most one grad
    pass per swapped stack between ``begin_train_step`` calls, and rejects a no-grad pass after a grad pass (the
    blocks would still be on CPU until backward).
    """

    def __init__(self, config: YuE2Config):
        super().__init__()
        self.config = config
        self.ar = YuE2AR(config)
        self.nar = YuE2NAR(config)
        self.norm = YuE2RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.is_convrot_int8 = False
        self.checkpoint_layout = None
        self.base_quant = "bf16"
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap = 0
        self._offloaders: dict[str, object] = {}
        self._swap_plan: Optional[tuple[int, int, bool, bool]] = None
        self._branches: tuple[str, ...] = ("ar", "nar")
        self._swap_inference = False
        self._grad_passes = {"ar": 0, "nar": 0}
        self._execution_device: Optional[torch.device] = None
        self._resident_in = False
        self._ar_resident = False

    # region basics

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.norm.weight.dtype

    @property
    def device(self) -> torch.device:
        return self._execution_device if self._execution_device is not None else self.norm.weight.device

    def set_attention(self, attn_mode: str, split_attn: bool = False, query_tile: int = 256, sdpa_gqa: str = "repeat") -> None:
        if attn_mode == "sdpa" or attn_mode is None:
            attn_mode = "torch"
        kwargs = {"attn_mode": attn_mode, "split_attn": bool(split_attn), "query_tile": int(query_tile), "sdpa_gqa": sdpa_gqa}
        for blocks in (self.ar.blocks, self.nar.blocks):
            for block in blocks:
                block.self_attn.attn_kwargs = dict(kwargs)

    def _rope_tables(self, start: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        dev = self.device
        return rope_cos_sin(torch.arange(start, start + n, device=dev), self.config.head_dim, self.config.rope_theta, dev)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Token rows ``[B, L, H]`` for ``ids [B, L]`` on the execution device.

        The lookup runs where the table lives (it stays on CPU in NAR-only training); the dtype is the table's output
        dtype.
        """
        table = self.ar.embed_tokens
        rows = table(ids.to(table.weight.device))
        return rows.to(self.device)

    # endregion

    # region forward

    def _run_stack(
        self,
        name: str,
        blocks: nn.ModuleList,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        prefix_kv: Optional[Sequence[tuple[torch.Tensor, torch.Tensor]]] = None,
        causal: bool = True,
        collect_kv: bool = False,
        kv_visible: Optional[int] = None,
        valid_len: Optional[int] = None,
        cache: Optional[YuE2KVCache] = None,
    ) -> tuple[torch.Tensor, Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        off = self._offloaders.get(name)
        if name == "ar" and self._ar_resident:
            off = None
        grad_pass = torch.is_grad_enabled() and not self._swap_inference
        if off is not None and off.supports_backward:
            if off.forward_only == grad_pass:
                off.set_forward_only(not grad_pass)
            if grad_pass:
                self._grad_passes[name] += 1
                if self._grad_passes[name] > 1:
                    raise RuntimeError(
                        f"YuE2 {name} stack: second grad forward before backward under block swap; use batch_size=1"
                        " with --gradient_accumulation_steps"
                    )
            elif self._grad_passes[name] > 0:
                raise RuntimeError(
                    f"YuE2 {name} stack: no-grad pass after a grad pass under block swap (run every no-grad pass first)"
                )
        if collect_kv and grad_pass:
            raise RuntimeError("YuE2 K/V are collected only in no-grad context prefills")
        if prefix_kv is not None and len(prefix_kv) != len(blocks):
            raise ValueError(f"YuE2 {name} stack: {len(prefix_kv)} prefix K/V layers for {len(blocks)} blocks")

        kv = [] if collect_kv else None
        dev = self._execution_device or x.device
        use_gc = self.gradient_checkpointing and self.training and grad_pass and cache is None
        for i, block in enumerate(blocks):
            if off is not None:
                off.wait_for_block(i)
            if cache is not None:
                x = block.forward_with_cache(x, cos, sin, cache, i)
            else:
                pk, pv = (None, None) if prefix_kv is None else prefix_kv[i]
                if use_gc:
                    fn = create_cpu_offloading_wrapper(block, dev) if self.activation_cpu_offloading else block
                    # with activation offload x stays on CPU between blocks, so each checkpoint saves a CPU tensor
                    x, _, _ = torch.utils.checkpoint.checkpoint(fn, x, cos, sin, pk, pv, causal, False, use_reentrant=False)
                else:
                    x, k, v = block(x, cos, sin, pk, pv, causal, collect_kv)
                    if collect_kv:
                        kv.append(_visible_kv(k, v, kv_visible, valid_len))
            if off is not None:
                off.submit_move_blocks_forward(blocks, i)
        if x.device != dev:
            x = x.to(dev)
        if off is not None and off.supports_backward and grad_pass and not x.requires_grad:
            raise RuntimeError(
                f"YuE2 {name} stack ran a training-mode block-swap pass whose output needs no grad; run it under torch.no_grad()"
            )
        return x, kv

    def ar_forward(
        self,
        input_embeds: torch.Tensor,
        *,
        return_kv: bool = False,
        kv_visible: Optional[int] = None,
        return_hidden: bool = True,
        valid_len: Optional[int] = None,
    ) -> YuE2AROutput:
        """Causal AR pass over positions ``0..L-1``.

        ``return_kv`` (no-grad context prefills only) collects per-layer post-RoPE K/V truncated to ``kv_visible``
        (and to ``valid_len`` for right-padded inputs; truncated K/V are cloned so the invisible storage is freed).
        """
        cos, sin = self._rope_tables(0, input_embeds.shape[1])
        x = input_embeds.to(device=self.device, dtype=self.compute_dtype)
        x, kv = self._run_stack(
            "ar", self.ar.blocks, x, cos, sin, causal=True, collect_kv=return_kv, kv_visible=kv_visible, valid_len=valid_len
        )
        return YuE2AROutput(hidden=self.norm(x) if return_hidden else None, kv=kv)

    def lm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.ar.lm_head(hidden)

    def nar_forward(
        self, x_t: torch.Tensor, t_embed: torch.Tensor, kv: Sequence[tuple[torch.Tensor, torch.Tensor]], rope_offset: int
    ) -> torch.Tensor:
        """Flow velocity ``[B, T, 64]`` for the latent window ``x_t [B, T, 64]``.

        ``t_embed`` is the time-embedder input (``[B]`` or ``[1]``), ``kv`` the per-layer context K/V of a no-grad
        ``ar_forward(..., return_kv=True)`` and ``rope_offset`` the first RoPE position of the window (the START slot).
        """
        dev, dt = self.device, self.compute_dtype
        latents = x_t.to(device=dev, dtype=dt)
        # zero START / END slots around the window
        edge = latents.new_zeros(latents.shape[0], 1, latents.shape[2])
        latents = torch.cat((edge, latents, edge), dim=1)
        n = latents.shape[1]

        t = torch.as_tensor(t_embed, device=dev).reshape(-1)
        h = self.nar.vae2llm(latents) + self.nar.time_embedder(t, dt)[:, None, :]
        h = h + self.nar.latent_pos_embed(n, dt)[None]

        cos, sin = self._rope_tables(rope_offset, n)
        h, _ = self._run_stack("nar", self.nar.blocks, h, cos, sin, prefix_kv=kv, causal=False)
        velocity = self.nar.llm2vae(self.norm(h))
        return velocity[:, 1:-1]

    # endregion

    # region inference helpers

    def new_kv_cache(self, capacity: int, batch: int = 1) -> YuE2KVCache:
        cfg = self.config
        return YuE2KVCache(cfg.num_layers, batch, capacity, cfg.num_kv_heads, cfg.head_dim, self.device, self.compute_dtype)

    def _cached_step(self, ids: torch.Tensor, cache: YuE2KVCache) -> torch.Tensor:
        """Runs ``ids [B, L]`` at positions ``cache.length ..`` through the AR stack, appending to ``cache``; returns
        the logits ``[B, V]`` of the last position."""
        n = ids.shape[1]
        cache.ensure_room(n)  # fail before any layer writes
        x = self.embed(ids).to(dtype=self.compute_dtype)
        cos, sin = self._rope_tables(cache.length, n)
        x, _ = self._run_stack("ar", self.ar.blocks, x, cos, sin, cache=cache)
        cache.length += n
        last = self.norm(x[:, -1:])
        return self.ar.lm_head(last)[:, 0]

    def ar_prefill_into_cache(self, ids: torch.Tensor, cache: YuE2KVCache) -> torch.Tensor:
        """Appends ``ids [B, L]`` (any L, may be called repeatedly) to ``cache``; returns last-position logits ``[B, V]``."""
        return self._cached_step(ids, cache)

    def ar_decode_step(self, ids: torch.Tensor, cache: YuE2KVCache) -> torch.Tensor:
        """Appends one token per row ``ids [B, 1]`` to ``cache``; returns its logits ``[B, V]``."""
        if ids.shape[1] != 1:
            raise ValueError(f"ar_decode_step expects ids of shape [B, 1], got {tuple(ids.shape)}")
        return self._cached_step(ids, cache)

    # endregion

    # region musubi mechanism API

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def set_block_swap_plan(
        self,
        ar_blocks: int,
        nar_blocks: int,
        ar_backward: bool,
        nar_backward: bool = True,
        branches: tuple[str, ...] = ("ar", "nar"),
    ) -> None:
        """Per-list swap counts and backward support, plus branch-aware residency (NAR-only: ``ar.lm_head`` and
        ``ar.embed_tokens`` stay on CPU; AR-only: ``nar.*`` stays on CPU)."""
        branches = tuple(branches)
        if not branches or any(b not in ("ar", "nar") for b in branches):
            raise ValueError(f"invalid YuE2 branches: {branches}")
        self._swap_plan = (int(ar_blocks), int(nar_blocks), bool(ar_backward), bool(nar_backward))
        self._branches = branches

    @property
    def cpu_resident_modules(self) -> tuple[str, ...]:
        """Module paths kept on CPU during training."""
        if self._branches == ("nar",):
            return ("ar.lm_head", "ar.embed_tokens")
        if self._branches == ("ar",):
            return ("nar",)
        return ()

    def begin_train_step(self) -> None:
        """Reset the per-stack grad-pass counters (called first in every ``process_batch`` and by validator/sampler)."""
        self._grad_passes = {"ar": 0, "nar": 0}

    def _stack_blocks(self, name: str) -> nn.ModuleList:
        return self.ar.blocks if name == "ar" else self.nar.blocks

    def enable_block_swap(self, num_blocks: int, config: BlockSwapConfig) -> None:
        if self._swap_plan is not None:
            ar_n, nar_n, ar_bwd, nar_bwd = self._swap_plan
        else:
            ar_n = nar_n = num_blocks
            ar_bwd = nar_bwd = config.supports_backward
        self._offloaders = {}
        for name, n, bwd in (("ar", ar_n, ar_bwd), ("nar", nar_n, nar_bwd)):
            if n <= 0:
                continue
            blocks = self._stack_blocks(name)
            if n > len(blocks) - 2:
                raise ValueError(f"YuE2 cannot swap more than {len(blocks) - 2} of {len(blocks)} {name} blocks, got {n}")
            cfg = dataclasses.replace(config, supports_backward=bool(config.supports_backward and bwd))
            self._offloaders[name] = create_offloader(f"yue2-{name}", blocks, len(blocks), n, cfg)
            logger.info(f"YuE2 block swap: {name} {n}/{len(blocks)} blocks, backward={cfg.supports_backward}")
        self.blocks_to_swap = max(ar_n, nar_n, 0)
        self._execution_device = torch.device(config.device)

    def _offloaded_lists(self) -> list[str]:
        return [f"{name}.blocks" for name in self._offloaders]

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        device = torch.device(device)
        resident = [] if self._resident_in else list(self.cpu_resident_modules)
        hidden = _hide_modules(self, self._offloaded_lists() + resident)
        try:
            self.to(device)
        finally:
            _restore_modules(hidden)
        for path in resident:
            self.get_submodule(path).to("cpu")
        self._execution_device = device

    def _stack_is_away(self, name: str) -> bool:
        return not self._resident_in and name in self.cpu_resident_modules

    def prepare_block_swap_before_forward(self) -> None:
        for name, off in self._offloaders.items():
            if self._stack_is_away(name) or (name == "ar" and self._ar_resident):
                continue
            off.prepare_block_devices_before_forward(self._stack_blocks(name))

    def _move_resident(self, to_device: bool) -> None:
        paths = self.cpu_resident_modules
        if not paths or self._execution_device is None or self._resident_in == to_device:
            self._resident_in = to_device and bool(paths)
            return
        target = self._execution_device if to_device else torch.device("cpu")
        for path in paths:
            module = self.get_submodule(path)
            hidden = _hide_modules(module, ["blocks"]) if (to_device and path == "nar" and "nar" in self._offloaders) else []
            try:
                module.to(target)
            finally:
                _restore_modules(hidden)
        self._resident_in = to_device

    def switch_block_swap_for_inference(self) -> None:
        self._swap_inference = True
        for off in self._offloaders.values():
            off.set_forward_only(True)
        self._move_resident(True)
        self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self) -> None:
        self._swap_inference = False
        if self._ar_resident:
            self.set_ar_resident(False)
        for off in self._offloaders.values():
            off.set_forward_only(not off.supports_backward)
        self._move_resident(False)
        self.prepare_block_swap_before_forward()

    def set_ar_resident(self, on: bool) -> None:
        """``--sample_ar_resident``: AR blocks fully on the device and NAR blocks on CPU (on), or restore (off)."""
        dev = self._execution_device
        if dev is None or bool(on) == self._ar_resident:
            return
        ar_off, nar_off = self._offloaders.get("ar"), self._offloaders.get("nar")
        if any(hasattr(o, "cpu_master") for o in (ar_off, nar_off) if o is not None):
            logger.warning("--sample_ar_resident is not supported with --block_swap_h2d_only; keeping the swap layout")
            return
        for off in (ar_off, nar_off):
            if off is not None:
                off.set_forward_only(off.forward_only)  # wait for pending transfers
        if on:
            for block in self.nar.blocks:
                weighs_to_device(block, torch.device("cpu"))
            for block in self.ar.blocks:
                block.to(dev)
                weighs_to_device(block, dev)
            self._ar_resident = True
        else:
            self._ar_resident = False
            if ar_off is not None:
                ar_off.prepare_block_devices_before_forward(self.ar.blocks)
            if nar_off is not None:
                nar_off.prepare_block_devices_before_forward(self.nar.blocks)
            else:
                for block in self.nar.blocks:
                    block.to(dev)
                    weighs_to_device(block, dev)

    # endregion
