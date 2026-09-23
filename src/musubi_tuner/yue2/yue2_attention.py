"""YuE2 attention: causal AR prefill and NAR prefix (K/V-concat) attention over musubi backends.

Causal attention with fewer queries than keys (KV-cache prefill after a cached prefix) is aligned bottom-right: query
``i`` sits at absolute position ``Lk - Lq + i`` and sees keys ``0 .. Lk - Lq + i``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from musubi_tuner.modules.attention import flash_attn_func, flash_attn_interface, sageattn, xops

ATTN_MODES = ("torch", "flash", "flash3", "xformers", "sageattn")


def _normalize_mode(attn_mode: str) -> str:
    if attn_mode in (None, "torch", "sdpa"):
        return "torch"
    if attn_mode not in ATTN_MODES:
        raise ValueError(f"unsupported YuE2 attention mode: {attn_mode}")
    return attn_mode


def _repeat_kv(k: torch.Tensor, v: torch.Tensor, groups: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if groups == 1:
        return k, v
    return k.repeat_interleave(groups, dim=dim), v.repeat_interleave(groups, dim=dim)


def yue2_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    attn_mode: str,
    split_attn: bool = False,
    query_tile: int = 256,
    sdpa_gqa: str = "repeat",
) -> torch.Tensor:
    """Grouped-query attention.

    Args:
        q: ``[B, Lq, H, D]`` post-RoPE queries.
        k, v: ``[B, Lk, KVH, D]`` post-RoPE keys / values (``H % KVH == 0``).
        causal: lower-triangular mask, bottom-right aligned (``Lq <= Lk``; ``Lq == Lk`` for a plain prefill). The NAR
            calls it with ``causal=False``.
        attn_mode: one of ``ATTN_MODES`` (``sdpa`` is an alias of ``torch``). ``sageattn`` is allowed only with grad
            disabled (generation).
        split_attn: tile the queries (``query_tile`` rows per tile, exact causal masks per tile).
        sdpa_gqa: ``repeat`` expands K/V heads with ``repeat_interleave``; ``native`` passes ``enable_gqa=True``.

    Returns:
        ``[B, Lq, H * D]``.
    """
    mode = _normalize_mode(attn_mode)
    b, lq, h, d = q.shape
    lk, kvh = k.shape[1], k.shape[2]
    if h % kvh != 0 or v.shape != k.shape:
        raise ValueError(f"invalid attention shapes q {tuple(q.shape)}, k {tuple(k.shape)}, v {tuple(v.shape)}")
    if causal and lq > lk:
        raise ValueError(f"causal attention needs Lq <= Lk, got {lq} > {lk}")
    offset = lk - lq if causal else 0
    groups = h // kvh
    tile = query_tile if split_attn else None

    if mode == "sageattn":
        if torch.is_grad_enabled():
            raise RuntimeError("sageattn has no backward; YuE2 uses it only for generation under no_grad")
        if sageattn is None:
            raise RuntimeError("sageattn was selected but sageattention is not installed")
        if offset == 0 and not split_attn:
            out = sageattn(q, k, v, is_causal=causal, tensor_layout="NHD")
            return out.reshape(b, lq, h * d)
        mode = "torch"  # bottom-right causal masks and tiling go through SDPA

    if mode == "torch":
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        enable_gqa = False
        if groups > 1:
            if sdpa_gqa == "native":
                enable_gqa = True
            else:
                kt, vt = _repeat_kv(kt, vt, groups, dim=1)
        if tile is not None or offset > 0:
            out = _tiled_sdpa(qt, kt, vt, causal=causal, query_tile=tile or lq, enable_gqa=enable_gqa, q_offset=offset)
        else:
            kwargs = {"enable_gqa": True} if enable_gqa else {}
            out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal, **kwargs)
        return out.transpose(1, 2).reshape(b, lq, h * d)

    if mode == "flash":
        if flash_attn_func is None:
            raise RuntimeError("flash attention was selected but flash_attn is not installed")
        fn = lambda qi, ki, vi, c: flash_attn_func(qi, ki, vi, causal=c)  # noqa: E731
    elif mode == "flash3":
        if flash_attn_interface is None or not hasattr(flash_attn_interface, "flash_attn_func"):
            raise RuntimeError("FlashAttention 3 was selected but flash_attn_interface is not installed")

        def fn(qi, ki, vi, c):
            result = flash_attn_interface.flash_attn_func(qi, ki, vi, causal=c)
            return result[0] if isinstance(result, tuple) else result

    else:  # xformers
        if xops is None:
            raise RuntimeError("xformers was selected but xformers is not installed")
        k, v = _repeat_kv(k, v, groups, dim=2)

        def fn(qi, ki, vi, c):
            bias = None
            if c:
                bias = (
                    xops.fmha.attn_bias.LowerTriangularMask()
                    if qi.shape[1] == ki.shape[1]
                    else xops.fmha.attn_bias.LowerTriangularFromBottomRightMask()
                )
            return xops.memory_efficient_attention(qi, ki, vi, attn_bias=bias)

    if tile is None:
        out = fn(q, k, v, causal)
    else:
        out = _tiled_blhd(fn, q, k, v, causal=causal, query_tile=tile, q_offset=offset)
    return out.reshape(b, lq, h * d)


@torch.compiler.disable
def _tiled_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    query_tile: int,
    enable_gqa: bool,
    q_offset: int = 0,
) -> torch.Tensor:
    """Query-tiled SDPA on ``[B, H, L, D]`` tensors, following the blocked attention of the official YuE2 NAR sampler.

    For a causal tile ``[s, e)`` (absolute positions ``q_offset + s .. q_offset + e - 1``) the keys are
    ``[:q_offset + e]`` and the mask is ``arange(q_offset + e)[None] <= arange(q_offset + s, q_offset + e)[:, None]``;
    a tile starting at absolute position 0 uses ``is_causal=True``. Excluded from torch.compile so compiled blocks do
    not specialise on the tile count.
    """
    lq = q.shape[-2]
    kwargs = {"enable_gqa": True} if enable_gqa else {}
    outputs = []
    for start in range(0, lq, query_tile):
        end = min(start + query_tile, lq)
        qi = q[..., start:end, :]
        if not causal:
            outputs.append(F.scaled_dot_product_attention(qi, k, v, **kwargs))
            continue
        a, e = q_offset + start, q_offset + end
        if a == 0:
            outputs.append(F.scaled_dot_product_attention(qi, k[..., :e, :], v[..., :e, :], is_causal=True, **kwargs))
        else:
            mask = torch.arange(e, device=q.device)[None, :] <= torch.arange(a, e, device=q.device)[:, None]
            outputs.append(F.scaled_dot_product_attention(qi, k[..., :e, :], v[..., :e, :], attn_mask=mask, **kwargs))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-2)


@torch.compiler.disable
def _tiled_blhd(fn, q, k, v, *, causal: bool, query_tile: int, q_offset: int = 0) -> torch.Tensor:
    """Query tiling for ``[B, L, H, D]`` kernels whose causal mask is bottom-right aligned (flash, xformers)."""
    lq = q.shape[1]
    outputs = []
    for start in range(0, lq, query_tile):
        end = min(start + query_tile, lq)
        if causal:
            e = q_offset + end
            outputs.append(fn(q[:, start:end], k[:, :e], v[:, :e], True))
        else:
            outputs.append(fn(q[:, start:end], k, v, False))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
