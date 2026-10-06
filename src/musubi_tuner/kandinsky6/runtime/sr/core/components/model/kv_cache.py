# Pure-torch helpers for the streaming KV-cache in self-attention.
#
# Kept free of ``flash_attn`` imports so the cache bookkeeping is importable and
# testable on CPU. The actual attention kernel call lives in
# ``musubi_tuner.kandinsky6.runtime.sr.model.nn``; this module only assembles its packed inputs.

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import nn


def assemble_cached_attention_inputs(
    new_key: Tensor,
    new_value: Tensor,
    cached_key: Tensor,
    cached_value: Tensor,
    cu_seqlens_new: Tensor,
    cached_cu_seqlens: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Merge cached and new K/V into packed varlen attention inputs.

    For each packed sequence ``i`` the keys/values become ``[cache_i | new_i]``;
    the query packing stays the new-token packing (queries are the freshly
    arrived tokens that attend back over the cache plus themselves).

    Args:
        new_key: New keys, packed ``(sum(new_lens), num_heads, head_dim)``.
        new_value: New values, same packing as ``new_key``.
        cached_key: Cached keys, packed ``(sum(cache_lens), num_heads, head_dim)``.
        cached_value: Cached values, same packing as ``cached_key``.
        cu_seqlens_new: Cumulative sequence lengths of the new tokens, ``(num_seqs + 1,)``.
        cached_cu_seqlens: Cumulative sequence lengths of the cache, ``(num_seqs + 1,)``.

    Returns:
        Tuple ``(key, value, cu_seqlens_q, cu_seqlens_k)`` where ``key``/``value``
        are the merged packed tensors, ``cu_seqlens_q`` is the query packing
        (``= cu_seqlens_new``) and ``cu_seqlens_k`` the merged key packing.
    """
    num_seqs = cu_seqlens_new.numel() - 1
    key_parts: list[Tensor] = []
    value_parts: list[Tensor] = []
    for i in range(num_seqs):
        cache_start, cache_end = int(cached_cu_seqlens[i]), int(cached_cu_seqlens[i + 1])
        new_start, new_end = int(cu_seqlens_new[i]), int(cu_seqlens_new[i + 1])
        key_parts.append(cached_key[cache_start:cache_end])
        key_parts.append(new_key[new_start:new_end])
        value_parts.append(cached_value[cache_start:cache_end])
        value_parts.append(new_value[new_start:new_end])

    key = torch.cat(key_parts, dim=0)
    value = torch.cat(value_parts, dim=0)

    merged_lens = torch.diff(cached_cu_seqlens) + torch.diff(cu_seqlens_new)
    cu_seqlens_k = torch.cat([cu_seqlens_new.new_zeros(1), torch.cumsum(merged_lens, dim=0)]).to(cu_seqlens_new.dtype)
    cu_seqlens_q = cu_seqlens_new
    return key, value, cu_seqlens_q, cu_seqlens_k


def evict_cached_kv(
    cached_key: Tensor,
    cached_value: Tensor,
    cached_cu_seqlens: Tensor,
    max_tokens_per_seq: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Trim a packed KV-cache to its last ``max_tokens_per_seq`` tokens per sequence.

    Implements the fixed-length rolling window: for each packed sequence the
    oldest tokens are dropped so at most ``max_tokens_per_seq`` most-recent ones
    remain. Sequences already within the limit are kept whole. ``cu_seqlens`` is
    rebuilt over the trimmed packing. Pure torch (no ``flash_attn``) so it stays
    CPU-testable.

    Args:
        cached_key: Cached keys, packed ``(sum(cache_lens), num_heads, head_dim)``.
        cached_value: Cached values, same packing as ``cached_key``.
        cached_cu_seqlens: Cumulative sequence lengths of the cache, ``(num_seqs + 1,)``.
        max_tokens_per_seq: Maximum tokens to keep per packed sequence (the rolling
            window length in tokens, i.e. ``cache_frames * H * W``).

    Returns:
        Tuple ``(key, value, cu_seqlens)`` of the trimmed packed tensors and the
        rebuilt cumulative sequence lengths.
    """
    num_seqs = cached_cu_seqlens.numel() - 1
    key_parts: list[Tensor] = []
    value_parts: list[Tensor] = []
    kept_lens: list[int] = []
    for i in range(num_seqs):
        start, end = int(cached_cu_seqlens[i]), int(cached_cu_seqlens[i + 1])
        keep_start = max(start, end - max_tokens_per_seq)
        key_parts.append(cached_key[keep_start:end])
        value_parts.append(cached_value[keep_start:end])
        kept_lens.append(end - keep_start)

    key = torch.cat(key_parts, dim=0)
    value = torch.cat(value_parts, dim=0)
    lens = torch.tensor(kept_lens, device=cached_cu_seqlens.device)
    cu_seqlens = torch.cat([cached_cu_seqlens.new_zeros(1), torch.cumsum(lens, dim=0)]).to(cached_cu_seqlens.dtype)
    return key, value, cu_seqlens


def iter_self_attentions(model: nn.Module) -> Iterator[nn.Module]:
    """Yield every submodule exposing the streaming KV-cache API.

    Duck-typed on the presence of ``reset_kv_cache`` so this module stays free of
    ``flash_attn`` (which importing ``MultiheadSelfAttention`` would pull in) and
    avoids a circular import with ``musubi_tuner.kandinsky6.runtime.sr.model.nn``.

    Args:
        model: Module to scan recursively.

    Yields:
        Each self-attention submodule with a rolling KV-cache.
    """
    for module in model.modules():
        if callable(getattr(module, "reset_kv_cache", None)):
            yield module


def set_return_kv(model: nn.Module, *, flag: bool) -> None:
    """Toggle ``return_kv`` on every self-attention submodule.

    Args:
        model: Model whose self-attention layers to update.
        flag: When ``True`` the next forward writes its merged K/V into the cache.
    """
    for attn in iter_self_attentions(model):
        attn.return_kv = flag


def reset_kv_caches(model: nn.Module) -> None:
    """Clear the rolling KV-cache on every self-attention submodule.

    Args:
        model: Model whose self-attention caches to reset.
    """
    for attn in iter_self_attentions(model):
        attn.reset_kv_cache()


@contextmanager
def kv_cache_session(model: nn.Module) -> Iterator[None]:
    """Scope a streaming generation: caches are guaranteed clean before and after.

    Entry reset protects against state leaked by a previously crashed session;
    the ``finally`` reset guarantees that even if generation dies mid-clip (e.g.
    OOM) no layer keeps stale ``cached_k`` or a dangling ``return_kv=True``, so
    subsequent non-streaming calls on the same model stay correct.

    Args:
        model: Model whose self-attention caches to scope.

    Yields:
        Nothing; run the chunked generation loop inside the block.
    """
    reset_kv_caches(model)
    try:
        yield
    finally:
        reset_kv_caches(model)


@contextmanager
def kv_writer(model: nn.Module) -> Iterator[None]:
    """Mark the enclosed forward as the cache writer (``return_kv=True``).

    ``return_kv`` is switched off in a ``finally`` so a crash inside the writer
    forward cannot leave the model silently appending K/V on later calls.

    Args:
        model: Model whose self-attention layers write K/V inside the block.

    Yields:
        Nothing; run the writer forward (and its state capture) inside the block.
    """
    set_return_kv(model, flag=True)
    try:
        yield
    finally:
        set_return_kv(model, flag=False)


# A captured per-step cache "slot": one ``(key, value, cu_seqlens)`` triple per
# self-attention layer (aligned with ``iter_self_attentions`` order), or ``None``
# for an empty slot (no past frames cached yet at that denoising step).
LayerCache = tuple["Tensor | None", "Tensor | None", "Tensor | None"]
KVSlot = "list[LayerCache] | None"


def capture_kv_state(model: nn.Module) -> list[LayerCache]:
    """Snapshot every self-attention's rolling KV-cache.

    Stores the current ``(cached_k, cached_v, cached_cu_seqlens)`` references for
    each self-attention layer. The references are safe to hold: the attention
    forward reassigns these attributes to freshly concatenated tensors rather
    than mutating them in place, so a snapshot never aliases future writes.

    Args:
        model: Model whose self-attention caches to snapshot.

    Returns:
        Per-layer ``(key, value, cu_seqlens)`` triples in ``iter_self_attentions`` order.
    """
    return [(attn.cached_k, attn.cached_v, attn.cached_cu_seqlens) for attn in iter_self_attentions(model)]


def restore_kv_state(model: nn.Module, slot: KVSlot) -> None:
    """Load a captured slot (or clear) into every self-attention's cache.

    Args:
        model: Model whose self-attention caches to overwrite.
        slot: Per-layer ``(key, value, cu_seqlens)`` triples from
            :func:`capture_kv_state`, or ``None`` to clear all caches (an empty
            slot — no past frames yet).
    """
    attns = list(iter_self_attentions(model))
    if slot is None:
        for attn in attns:
            attn.cached_k, attn.cached_v, attn.cached_cu_seqlens = None, None, None
        return
    for attn, (key, value, cu_seqlens) in zip(attns, slot, strict=True):
        attn.cached_k, attn.cached_v, attn.cached_cu_seqlens = key, value, cu_seqlens


def evict_kv_slot(slot: KVSlot, max_tokens_per_seq: int) -> KVSlot:
    """Apply fixed-length rolling eviction to a captured slot, per layer.

    Args:
        slot: A captured per-step slot, or ``None`` (empty slot passes through).
        max_tokens_per_seq: Rolling window length in tokens (``cache_frames * H * W``).

    Returns:
        The evicted slot, or ``None`` if the input was ``None``.
    """
    if slot is None:
        return None
    evicted: list[LayerCache] = []
    for key, value, cu_seqlens in slot:
        if key is None:
            evicted.append((None, None, None))
        else:
            evicted.append(evict_cached_kv(key, value, cu_seqlens, max_tokens_per_seq))
    return evicted
