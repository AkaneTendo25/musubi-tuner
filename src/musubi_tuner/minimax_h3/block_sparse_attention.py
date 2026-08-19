"""Block-sparse attention over the packed H3 sequence.

Keys are grouped into equal blocks. Each query block scores every key block by
the dot product of their means and attends to the top-k blocks plus its own.
Selection is discrete and carries no gradient; gradients flow through the
retained blocks. Runs on ``flex_attention`` with a ``BlockMask``, which supplies
the backward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

_COMPILED: dict[str, object] = {}

#: Block edge in rows.
DEFAULT_BLOCK = 128


def _compiled_flex():
    """Compiled kernel, built once per process.

    Uncompiled ``flex_attention`` materializes the full score matrix; the
    compiled path fuses the mask and skips excluded blocks.
    """
    if "fn" not in _COMPILED:
        _COMPILED["fn"] = torch.compile(flex_attention, dynamic=False)
    return _COMPILED["fn"]


@dataclass(frozen=True)
class BlockSparseConfig:
    """Block selection settings.

    Args:
        block: rows per block, query and key side alike.
        kv_fraction: share of key blocks kept per query block, before its own
            block is added. ``1.0`` reproduces dense attention.
        share_heads: score once for all heads instead of per head.
        min_blocks: lower bound on retained blocks.
    """

    block: int = DEFAULT_BLOCK
    kv_fraction: float = 0.25
    share_heads: bool = False
    min_blocks: int = 1

    def validate(self) -> None:
        if self.block <= 0 or self.block % 8:
            raise ValueError("block must be a positive multiple of 8")
        if not 0.0 < self.kv_fraction <= 1.0:
            raise ValueError("kv_fraction must lie in (0, 1]")
        if self.min_blocks < 1:
            raise ValueError("min_blocks must be at least 1")


def _block_means(tensor: torch.Tensor, block: int) -> torch.Tensor:
    """(B, H, S, D) -> (B, H, S // block, D)."""
    batch, heads, rows, dim = tensor.shape
    return tensor.view(batch, heads, rows // block, block, dim).mean(dim=3)


def select_blocks(query: torch.Tensor, key: torch.Tensor, cfg: BlockSparseConfig) -> torch.Tensor:
    """Boolean (B, H, Nq, Nk) map of the key blocks each query block attends to."""
    cfg.validate()
    q_mean = _block_means(query, cfg.block)
    k_mean = _block_means(key, cfg.block)
    if cfg.share_heads:
        q_mean = q_mean.mean(dim=1, keepdim=True)
        k_mean = k_mean.mean(dim=1, keepdim=True)

    scores = torch.einsum("bhqd,bhkd->bhqk", q_mean, k_mean)
    n_keys = scores.shape[-1]
    keep = max(cfg.min_blocks, min(n_keys, int(round(n_keys * cfg.kv_fraction))))

    chosen = scores.topk(keep, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, chosen, True)

    # Own block is never dropped: a mean-based score can rank it low.
    diagonal = torch.arange(mask.shape[-2], device=mask.device)
    mask[..., diagonal, diagonal.clamp(max=n_keys - 1)] = True

    if cfg.share_heads:
        mask = mask.expand(-1, query.shape[1], -1, -1)
    return mask


def block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cfg: BlockSparseConfig,
) -> torch.Tensor:
    """Attention over the selected blocks. Shapes are (B, H, S, D).

    Rows are padded to a whole number of blocks and trimmed afterwards.
    """
    cfg.validate()
    rows = query.shape[-2]
    pad = (-rows) % cfg.block
    if pad:
        query, key, value = (torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in (query, key, value))

    keep = select_blocks(query, key, cfg)
    padded = query.shape[-2]

    # A zero score is a full weight after the softmax, so padded rows must be
    # excluded. Drop fully padded blocks here; the straddling block holds real
    # rows and is cut per row by the mask below.
    if pad:
        keep = keep.clone()
        keep[..., -(-rows // cfg.block) :] = False

    # Built from the selected indices: evaluating a mask function over every
    # row pair would materialize a grid the size of the attention matrix.
    counts = keep.sum(dim=-1).to(torch.int32)
    indices = torch.argsort(keep.to(torch.int8), dim=-1, descending=True, stable=True).to(torch.int32)

    def mask_mod(batch, head, q_row, k_row):
        return (k_row < rows) & keep[batch, head, q_row // cfg.block, k_row // cfg.block]

    block_mask = BlockMask.from_kv_blocks(
        counts,
        indices,
        BLOCK_SIZE=cfg.block,
        mask_mod=mask_mod,
        seq_lengths=(padded, padded),
    )
    out = _compiled_flex()(query, key, value, block_mask=block_mask)
    return out[..., :rows, :] if pad else out
