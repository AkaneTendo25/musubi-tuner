"""Utility functions for tensor manipulation, patching, and sparse attention masks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

if TYPE_CHECKING:
    from torch.nn import Module

# Side of the local 8x8 token block used by fractal (NABLA) attention.
# Independent of the VAE spatial compression — do not swap for VAE_SPATIAL_FACTOR.
FRACTAL_BLOCK_SIZE = 8


def exist(item: object) -> bool:
    """Check if an item is not None."""
    return item is not None


def freeze(model: Module) -> Module:
    """Freeze all model parameters by disabling gradient computation.

    Args:
        model: PyTorch module to freeze.

    Returns:
        The same model with all ``requires_grad`` set to False.
    """
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.autocast(device_type="cuda", enabled=False)
def get_freqs(dim: int, max_period: float = 10000.0) -> Tensor:
    """Compute sinusoidal frequency schedule for rotary embeddings.

    Args:
        dim: Embedding dimension.
        max_period: Maximum period of the sinusoidal frequencies.

    Returns:
        Frequency tensor of shape ``(dim,)``.
    """
    return torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim, dtype=torch.float32) / dim)


def fractal_flatten(
    x: Tensor,
    rope: Tensor,
    cu_seqlens: Tensor,
    shape: tuple[int, int, int],
    *,
    fractal: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Flatten spatial dimensions with optional fractal (local block) patching.

    Args:
        x: Input tensor with spatial dimensions.
        rope: Rotary position embedding tensor matching ``x`` layout.
        cu_seqlens: Cumulative sequence lengths (per-frame counts).
        shape: Spatial shape as ``(length, height, width)``.
        fractal: If True, apply local 8x8 block patching before flattening.

    Returns:
        Tuple of (flattened ``x``, flattened ``rope``, scaled ``cu_seqlens``).
    """
    _, height, width = shape
    if fractal:
        block = FRACTAL_BLOCK_SIZE
        x = local_patching(x, shape, (1, block, block), dim=0)
        rope = local_patching(rope, shape, (1, block, block), dim=0)
        x = x.flatten(0, 1)
        rope = rope.flatten(0, 1)
    else:
        x = x.flatten(0, 2)
        rope = rope.flatten(0, 2)
    cu_seqlens = cu_seqlens * (height * width)
    return x, rope, cu_seqlens


def fractal_unflatten(
    x: Tensor,
    cu_seqlens: Tensor,
    shape: tuple[int, int, int],
    *,
    fractal: bool = False,
) -> tuple[Tensor, Tensor]:
    """Unflatten spatial dimensions, inverse of ``fractal_flatten``.

    Args:
        x: Flattened input tensor.
        cu_seqlens: Scaled cumulative sequence lengths.
        shape: Target spatial shape as ``(length, height, width)``.
        fractal: If True, reverse local 8x8 block patching.

    Returns:
        Tuple of (unflattened ``x``, rescaled ``cu_seqlens``).
    """
    _, height, width = shape
    if fractal:
        block = FRACTAL_BLOCK_SIZE
        x = x.reshape(-1, block**2, *x.shape[1:])
        x = local_merge(x, shape, (1, block, block), dim=0)
    else:
        x = x.reshape(*shape, *x.shape[1:])
    cu_seqlens = cu_seqlens // (height * width)
    return x, cu_seqlens


def mean_var_len(x: Tensor, cu_seqlens: Tensor, dim: int = 0) -> Tensor:
    """Compute per-sequence mean over variable-length sequences.

    Args:
        x: Input tensor.
        cu_seqlens: Cumulative sequence lengths defining sequence boundaries.
        dim: Dimension to reduce over.

    Returns:
        Concatenated per-sequence means.
    """
    return torch.cat(
        [local_x.mean(dim=dim) for local_x in torch.split(x, torch.diff(cu_seqlens).tolist(), dim=0)],
        dim=0,
    )


def local_patching(x: Tensor, shape: tuple[int, int, int], group_size: tuple[int, int, int], dim: int = 0) -> Tensor:
    """Rearrange tensor into local spatial patches.

    Groups neighboring elements along each spatial axis into patches,
    producing a tensor with patch-count and intra-patch dimensions.

    Args:
        x: Input tensor with spatial dimensions starting at ``dim``.
        shape: Spatial shape as ``(duration, height, width)``.
        group_size: Patch size per axis ``(g1, g2, g3)``.
        dim: First spatial dimension index.

    Returns:
        Patched tensor with shape ``(..., num_patches, patch_elems, ...)``.
    """
    duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], duration // g1, g1, height // g2, g2, width // g3, g3, *x.shape[dim + 3 :])
    x = x.permute(
        *range(len(x.shape[:dim])), dim, dim + 2, dim + 4, dim + 1, dim + 3, dim + 5, *range(dim + 6, len(x.shape))
    )
    return x.flatten(dim, dim + 2).flatten(dim + 1, dim + 3)


def local_merge(x: Tensor, shape: tuple[int, int, int], group_size: tuple[int, int, int], dim: int = 0) -> Tensor:
    """Reverse local patching, restoring the original spatial layout.

    Args:
        x: Patched tensor with ``(num_patches, patch_elems)`` at ``dim``.
        shape: Original spatial shape as ``(duration, height, width)``.
        group_size: Patch size per axis ``(g1, g2, g3)``.
        dim: First spatial dimension index.

    Returns:
        Tensor with restored spatial dimensions.
    """
    duration, height, width = shape
    g1, g2, g3 = group_size
    x = x.reshape(*x.shape[:dim], duration // g1, height // g2, width // g3, g1, g2, g3, *x.shape[dim + 2 :])
    x = x.permute(
        *range(len(x.shape[:dim])), dim, dim + 3, dim + 1, dim + 4, dim + 2, dim + 5, *range(dim + 6, len(x.shape))
    )
    return x.flatten(dim, dim + 1).flatten(dim + 1, dim + 2).flatten(dim + 2, dim + 3)


def global_patching(x: Tensor, shape: tuple[int, int, int], group_size: tuple[int, int, int], dim: int = 0) -> Tensor:
    """Rearrange tensor into global patches by strided sampling.

    Unlike ``local_patching``, groups spatially distant elements
    that are evenly spaced across the volume.

    Args:
        x: Input tensor with spatial dimensions starting at ``dim``.
        shape: Spatial shape as ``(duration, height, width)``.
        group_size: Number of output patches per axis.
        dim: First spatial dimension index.

    Returns:
        Globally patched tensor.
    """
    latent_group_size = [axis // axis_group_size for axis, axis_group_size in zip(shape, group_size, strict=False)]
    x = local_patching(x, shape, latent_group_size, dim)
    return x.transpose(dim, dim + 1)


def global_merge(x: Tensor, shape: tuple[int, int, int], group_size: tuple[int, int, int], dim: int = 0) -> Tensor:
    """Reverse global patching, restoring the original spatial layout.

    Args:
        x: Globally patched tensor.
        shape: Original spatial shape as ``(duration, height, width)``.
        group_size: Number of patches per axis.
        dim: First spatial dimension index.

    Returns:
        Tensor with restored spatial dimensions.
    """
    latent_group_size = [axis // axis_group_size for axis, axis_group_size in zip(shape, group_size, strict=False)]
    x = x.transpose(dim, dim + 1)
    return local_merge(x, shape, latent_group_size, dim)


@torch.compile()
@torch.no_grad()
def fast_sta_nabla(T: int, H: int, W: int, wT: int = 3, wH: int = 3, wW: int = 3, device: str = "cuda") -> Tensor:
    """Build a static spatiotemporal neighborhood attention mask.

    Each token attends to spatial and temporal neighbors within
    the given window sizes.

    Args:
        T: Temporal dimension (number of frames).
        H: Height dimension.
        W: Width dimension.
        wT: Temporal window size (odd).
        wH: Height window size (odd).
        wW: Width window size (odd).
        device: Device for the output tensor.

    Returns:
        Boolean mask of shape ``(T*H*W, T*H*W)``.
    """
    max_dim = max(T, H, W)
    r = torch.arange(0, max_dim, 1, dtype=torch.int16, device=device)
    mat = (r.unsqueeze(1) - r.unsqueeze(0)).abs()
    sta_t, sta_h, sta_w = mat[:T, :T].flatten(), mat[:H, :H].flatten(), mat[:W, :W].flatten()
    sta_t = sta_t <= wT // 2
    sta_h = sta_h <= wH // 2
    sta_w = sta_w <= wW // 2
    sta_hw = (sta_h.unsqueeze(1) * sta_w.unsqueeze(0)).reshape(H, H, W, W).transpose(1, 2).flatten()
    sta = (sta_t.unsqueeze(1) * sta_hw.unsqueeze(0)).reshape(T, T, H * W, H * W).transpose(1, 2)
    return sta.reshape(T * H * W, T * H * W)


@torch.compile(dynamic=True)
@torch.no_grad()
def nablaT_v2_doc(
    q: Tensor,
    k: Tensor,
    seq: Tensor,
    T: int,
    H: int,
    W: int,
    *,
    wT: int = 3,
    wH: int = 3,
    wW: int = 3,
    thr: float = 0.9,
    add_sta: bool = True,
    method: str = "topcdf",
) -> BlockMask:
    """Build a dynamic sparse attention BlockMask with document boundaries.

    Estimates an approximate attention map from block-averaged queries and
    keys, then binarizes it to select the most relevant blocks per query.

    Args:
        q: Query tensor of shape ``(B, heads, seq_len, dim)``.
        k: Key tensor of shape ``(B, heads, seq_len, dim)``.
        seq: Cumulative document lengths (e.g. [0, 31, 51, 66, 97] - video boundaries in seq_len).
        T: Temporal dimension.
        H: Height dimension.
        W: Width dimension.
        wT: Temporal window for static attention.
        wH: Height window for static attention.
        wW: Width window for static attention.
        thr: Threshold for CDF cutoff (``topcdf``) or token fraction (``topk``).
        add_sta: Whether to add static local attention.
        method: Binarization method, ``"topcdf"`` or ``"topk"``.

    Returns:
        Sparse ``BlockMask`` with block size 64.
    """
    if method not in {"topcdf", "topk"}:
        msg = f"nabla method should be topcdf or topk, got {method}"
        raise ValueError(msg)
    # Q/K are the authoritative execution tensors when this function is
    # reached through attention. Keep every mask intermediate on their device.
    device = q.device
    seq = seq.to(device=device)

    # Map estimation
    B, h, S, D = q.shape
    qa = q.reshape(B, h, S // 64, 64, D).mean(-2)
    ka = k.reshape(B, h, S // 64, 64, D).mean(-2).transpose(-2, -1)
    attn_map = qa @ ka

    d = torch.diff(seq)
    doc = (
        torch.eye(d.numel(), dtype=torch.bool, device=device)
        .repeat_interleave(d * H * W, dim=0)
        .repeat_interleave(d * H * W, dim=1)
    )
    attn_map += doc.log()
    attn_map = torch.softmax(attn_map / math.sqrt(D), dim=-1)
    if method == "topcdf":
        # Map binarization
        vals, inds = attn_map.sort(-1)
        cvals = vals.cumsum_(-1)
        mask = (cvals >= 1 - thr).int()
        mask = mask.gather(-1, inds.argsort(-1))
    else:
        attn_map = attn_map.reshape(B * h * S // 64, S // 64)
        dl = d.tolist()
        start_row = 0
        mask = torch.zeros_like(attn_map)
        for di in dl:
            d_full = di * W * H * h * B
            end_row = start_row + d_full
            k = max(1, int(thr * di * W * H))
            group = attn_map[start_row:end_row, :]
            _, topk_indices = torch.topk(group, k, dim=-1)
            row_indices = torch.arange(start_row, end_row, device=mask.device).view(-1, 1)
            mask[row_indices, topk_indices] = 1
            start_row = end_row
        mask = mask.reshape(B, h, S // 64, S // 64)

    if add_sta:
        sta = fast_sta_nabla(T, H, W, wT, wH, wW, device=device).unsqueeze_(0).unsqueeze_(0)
        mask = torch.logical_or(mask, sta)
    mask = torch.logical_and(mask, doc)

    # BlockMask creation
    kv_nb = mask.sum(-1).to(torch.int32)
    kv_inds = mask.argsort(dim=-1, descending=True).to(torch.int32)
    return BlockMask.from_kv_blocks(torch.zeros_like(kv_nb), kv_inds, kv_nb, kv_inds, BLOCK_SIZE=64, mask_mod=None)


def block_mask_from_bool(mask: Tensor) -> BlockMask:
    """Convert a boolean block-attention matrix into a flex-attention ``BlockMask``.

    Args:
        mask: Boolean tensor broadcastable to ``(B, heads, S // 64, S // 64)``;
            ``True`` at ``(..., i, j)`` lets query block ``i`` attend key block ``j``.

    Returns:
        Sparse ``BlockMask`` with block size 64 and no intra-block ``mask_mod``.
    """
    kv_nb = mask.sum(-1).to(torch.int32)
    kv_inds = mask.argsort(dim=-1, descending=True).to(torch.int32)
    return BlockMask.from_kv_blocks(torch.zeros_like(kv_nb), kv_inds, kv_nb, kv_inds, BLOCK_SIZE=64, mask_mod=None)


def build_framewise_causal_block_doc(
    seq: Tensor,
    H: int,
    W: int,
    *,
    mf: int = 2,
) -> Tensor:
    """Build the frame-wise causal block-attention structure (no content sparsity).

    Causal across frames (a frame attends only to current and past frames), fully
    bidirectional within each ``mf``-frame group and within a frame, and isolated
    across packed documents. Operates at 64-block granularity, so ``H * W`` is the
    number of 64-token blocks per frame. Weight-independent: depends only on the
    document layout, never on Q/K content.

    Args:
        seq: Cumulative per-document frame counts, ``(num_docs + 1,)``.
        H: Block-grid height per frame.
        W: Block-grid width per frame (``H * W`` = 64-token blocks per frame).
        mf: Multi-frame group size; frames in a group attend bidirectionally.

    Returns:
        Boolean ``(S // 64, S // 64)`` block-attention matrix.
    """
    device = seq.device
    d = torch.diff(seq)
    doc1 = (
        torch.eye(d.numel(), dtype=torch.bool, device=device)
        .repeat_interleave(d, dim=0)
        .repeat_interleave(d, dim=1)
        .tril()
    )
    group_sizes = [[c.sum().item() for c in torch.ones((dd,)).split(mf)] for dd in d]
    cl = torch.tensor([x for xs in group_sizes for x in xs], dtype=torch.int32, device=device)
    doc2 = (
        torch.eye(cl.numel(), dtype=torch.bool, device=device).repeat_interleave(cl, dim=0).repeat_interleave(cl, dim=1)
    )
    return torch.logical_or(doc1, doc2).repeat_interleave(H * W, dim=0).repeat_interleave(H * W, dim=1)


@torch.compile(dynamic=True)
@torch.no_grad()
def nablaT_v2_doc_mfcausal(
    q: Tensor,
    k: Tensor,
    seq: Tensor,
    T: int,
    H: int,
    W: int,
    *,
    wT: int = 3,
    wH: int = 3,
    wW: int = 3,
    thr: float = 0.9,
    add_sta: bool = True,
    mf: int = 2,
) -> BlockMask:
    """Build a dynamic sparse attention BlockMask with multi-frame causal masking.

    Similar to ``nablaT_v2_doc`` but adds causal constraints across
    multi-frame groups within each document.

    Args:
        q: Query tensor of shape ``(B, heads, seq_len, dim)``.
        k: Key tensor of shape ``(B, heads, seq_len, dim)``.
        seq: Cumulative document lengths.
        T: Temporal dimension.
        H: Height dimension.
        W: Width dimension.
        wT: Temporal window for static attention.
        wH: Height window for static attention.
        wW: Width window for static attention.
        thr: CDF threshold for binarization.
        add_sta: Whether to add static local attention.
        mf: Multi-frame group size for causal masking.

    Returns:
        Sparse ``BlockMask`` with block size 64.
    """
    # Q/K are the authoritative execution tensors.
    device = q.device
    seq = seq.to(device=device)

    # Map estimation
    B, h, S, D = q.shape
    qa = q.reshape(B, h, S // 64, 64, D).mean(-2)
    ka = k.reshape(B, h, S // 64, 64, D).mean(-2).transpose(-2, -1)
    attn_map = qa @ ka

    doc = build_framewise_causal_block_doc(seq, H, W, mf=mf)
    attn_map += doc.log()
    attn_map = torch.softmax(attn_map / math.sqrt(D), dim=-1)

    # Map binarization
    vals, inds = attn_map.sort(-1)
    cvals = vals.cumsum_(-1)
    mask = (cvals >= 1 - thr).int()
    mask = mask.gather(-1, inds.argsort(-1))

    if add_sta:
        sta = fast_sta_nabla(T, H, W, wT, wH, wW, device=device).unsqueeze_(0).unsqueeze_(0)
        mask = torch.logical_or(mask, sta)
    mask = torch.logical_and(mask, doc)

    return block_mask_from_bool(mask)


@torch.no_grad()
def framewise_causal_dense(
    seq: Tensor,
    H: int,
    W: int,
    *,
    mf: int = 2,
) -> BlockMask:
    """Build the dense (non-sparse) frame-wise causal ``BlockMask``.

    The exact, weight-independent causal reference: the frame-wise causal block
    structure (causal across frames, bidirectional within an ``mf``-group and within
    a frame) with no content-based sparsity. Runs through the same ``flex_attention``
    kernel as the NABLA path; only the ``BlockMask`` content differs. Used as the
    streaming ground-truth mask and to validate that the NABLA causal selection stays
    within this causal envelope.

    Args:
        seq: Cumulative per-document frame counts, ``(num_docs + 1,)``.
        H: Block-grid height per frame.
        W: Block-grid width per frame (``H * W`` = 64-token blocks per frame).
        mf: Multi-frame group size; frames in a group attend bidirectionally.

    Returns:
        Dense causal ``BlockMask`` with block size 64.
    """
    doc = build_framewise_causal_block_doc(seq, H, W, mf=mf)
    return block_mask_from_bool(doc[None, None])
