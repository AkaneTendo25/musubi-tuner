"""Block-sparse attention over the packed H3 sequence.

Keys are grouped into equal blocks. Each query block scores every key block by
the dot product of their means and attends to the top-k blocks plus its own.
Selection is discrete and carries no gradient; gradients flow through the
retained blocks. Runs on ``flex_attention`` with a ``BlockMask``, which supplies
the backward pass.

Two properties of the packed sequence shape the layout:

* the target video is a trailing run of rows, preceded by text, audio and
  reference context. Context is a small share of the sequence and carries the
  conditioning, so it stays dense; only target-video rows are thinned.
* target-video rows follow the lattice in raster order, where rows adjacent in
  the sequence can be far apart in the frame. A block cut from raster order
  therefore averages unrelated rows, and the mean it is selected by describes
  no real region. ``block_shape`` reorders rows so each block is a 3D tile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from musubi_tuner.minimax_h3.miowtion_predictor import (
    TILE_SIZE,
    MiowtionBundle,
    MiowtionHeadGroup,
    MiowtionPlan,
    pool_miowtion_tiles,
)

_COMPILED: dict[tuple[object, ...], object] = {}

#: Block edge in rows.
DEFAULT_BLOCK = 128


def _compiled_flex(cache_key: tuple[object, ...] = ("generic",)):
    """Compiled kernel, built once per process.

    Uncompiled ``flex_attention`` materializes the full score matrix; the
    compiled path fuses the mask and skips excluded blocks.
    """
    if cache_key not in _COMPILED:
        # Dynamic shapes on purpose: sequence length varies per sample, and a
        # static compile would rebuild the kernel for every new length until
        # the recompile limit is hit and the call silently falls back to the
        # dense math path.
        _COMPILED[cache_key] = torch.compile(flex_attention, dynamic=True)
    return _COMPILED[cache_key]


def prepare_flex_attention() -> None:
    """Create the compiled FlexAttention wrapper before entering disabled model code."""
    # Released Miowtion plans use up to 37 distinct head-group extents in one
    # forward. FlexAttention specializes on that extent even with dynamic
    # shapes, so Dynamo's default limit of eight would drop later groups to
    # eager dense attention. This is process-wide because Dynamo counts
    # recompiles per Python code object across compiled wrappers.
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
    _compiled_flex()


@dataclass(frozen=True)
class BlockSparseConfig:
    """Block selection settings.

    Args:
        block: rows per block, query and key side alike.
        kv_fraction: share of key blocks kept per query block, before its own
            block is added. ``1.0`` reproduces dense attention. Ignored when
            ``threshold`` is set.
        threshold: keep the highest scoring blocks until their share of the
            score mass reaches this value, instead of a fixed count. A flat
            region then costs a few blocks and a busy one takes as many as it
            needs, where a fixed share misjudges both.
        share_heads: score once for all heads instead of per head.
        min_blocks: lower bound on retained blocks.
        block_shape: lattice tile ``(frames, height, width)`` whose product is
            ``block``. ``None`` keeps raster order, where a block spans whatever
            rows happen to be adjacent.
    """

    block: int = DEFAULT_BLOCK
    kv_fraction: float = 0.25
    threshold: float | None = None
    share_heads: bool = False
    min_blocks: int = 1
    block_shape: tuple[int, int, int] | None = None

    def validate(self) -> None:
        if self.block <= 0 or self.block % 8:
            raise ValueError("block must be a positive multiple of 8")
        if not 0.0 < self.kv_fraction <= 1.0:
            raise ValueError("kv_fraction must lie in (0, 1]")
        if self.threshold is not None and not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must lie in (0, 1]")
        if self.min_blocks < 1:
            raise ValueError("min_blocks must be at least 1")
        if self.block_shape is not None:
            if len(self.block_shape) != 3 or any(extent < 1 for extent in self.block_shape):
                raise ValueError("block_shape must be three positive extents")
            frames, height, width = self.block_shape
            if frames * height * width != self.block:
                raise ValueError(f"block_shape must multiply to block ({self.block}), got {self.block_shape}")


@dataclass(frozen=True)
class SequencePlan:
    """Row order that groups target-video rows into lattice tiles."""

    order: torch.Tensor
    inverse: torch.Tensor
    context_rows: int


def _lattice_index(column: torch.Tensor) -> torch.Tensor:
    """Map ordered rotary coordinates to consecutive lattice positions."""
    return torch.unique(column, sorted=True, return_inverse=True)[1]


def build_plan(position_ids: torch.Tensor, target_start: int, cfg: BlockSparseConfig) -> SequencePlan | None:
    """Order target-video rows into lattice tiles while leaving context first."""
    if cfg.block_shape is None:
        return None
    rows = position_ids.shape[0]
    if target_start >= rows:
        return None

    tile_frames, tile_height, tile_width = cfg.block_shape
    target = position_ids[target_start:]
    frame = _lattice_index(target[:, 0]) // tile_frames
    height = _lattice_index(target[:, 1]) // tile_height
    width = _lattice_index(target[:, 2]) // tile_width
    span_height = int(height.max()) + 1
    span_width = int(width.max()) + 1
    tile = (frame * span_height + height) * span_width + width

    device = position_ids.device
    order = torch.cat((torch.arange(target_start, device=device), torch.argsort(tile, stable=True) + target_start))
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(rows, device=device)
    return SequencePlan(order=order, inverse=inverse, context_rows=target_start)


def _block_means(tensor: torch.Tensor, block: int) -> torch.Tensor:
    """(B, H, S, D) -> (B, H, S // block, D)."""
    batch, heads, rows, dim = tensor.shape
    return tensor.view(batch, heads, rows // block, block, dim).mean(dim=3)


def select_blocks(query: torch.Tensor, key: torch.Tensor, cfg: BlockSparseConfig) -> torch.Tensor:
    """Boolean (B, H, Nq, Nk) map of the key blocks each query block attends to."""
    cfg.validate()
    # Selection feeds only topk/argsort/scatter -- none differentiable -- so it
    # runs detached: keeping the scoring chain in the graph would retain every
    # block's mean/einsum/softmax intermediates as saved-for-backward without
    # ever producing a gradient from them.
    with torch.no_grad():
        q_mean = _block_means(query.detach(), cfg.block)
        k_mean = _block_means(key.detach(), cfg.block)
        if cfg.share_heads:
            q_mean = q_mean.mean(dim=1, keepdim=True)
            k_mean = k_mean.mean(dim=1, keepdim=True)

        scores = torch.einsum("bhqd,bhkd->bhqk", q_mean, k_mean)
        n_keys = scores.shape[-1]

        if cfg.threshold is None:
            keep = max(cfg.min_blocks, min(n_keys, int(round(n_keys * cfg.kv_fraction))))
            chosen = scores.topk(keep, dim=-1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(-1, chosen, True)
        else:
            order = scores.argsort(dim=-1, descending=True)
            soft = torch.softmax(scores.gather(-1, order).float(), dim=-1)
            mass = soft.cumsum(dim=-1)
            # Keep the block that crosses the threshold, not just those below it,
            # so a single dominant block is never dropped.
            take = (mass - soft) < cfg.threshold
            take[..., : cfg.min_blocks] = True
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(-1, order, take)

        # Own block is never dropped: a mean-based score can rank it low.
        diagonal = torch.arange(mask.shape[-2], device=mask.device)
        mask[..., diagonal, diagonal.clamp(max=n_keys - 1)] = True

        if cfg.share_heads:
            mask = mask.expand(-1, query.shape[1], -1, -1)
    return mask


def _counts_and_indices(keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return kept-block counts and indices with retained blocks first."""
    counts = keep.sum(dim=-1).to(torch.int32)
    indices = torch.argsort(keep.to(torch.int8), dim=-1, descending=True, stable=True).to(torch.int32)
    return counts, indices


def block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cfg: BlockSparseConfig,
    plan: SequencePlan | None = None,
) -> torch.Tensor:
    """Attention over the selected blocks. Shapes are (B, H, S, D).

    Rows are padded to a whole number of blocks and trimmed afterwards. Given a
    plan, target rows are reordered into lattice tiles and restored afterwards.
    """
    cfg.validate()
    rows = query.shape[-2]
    if plan is not None:
        query, key, value = (tensor.index_select(-2, plan.order) for tensor in (query, key, value))
    pad = (-rows) % cfg.block
    if pad:
        query, key, value = (torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in (query, key, value))

    keep = select_blocks(query, key, cfg)
    padded = query.shape[-2]

    if plan is not None and plan.context_rows:
        context_blocks = -(-plan.context_rows // cfg.block)
        keep = keep.clone()
        keep[..., :context_blocks] = True
        keep[..., :context_blocks, :] = True

    # A zero score is a full weight after the softmax, so padded rows must be
    # excluded. The straddling block holds real rows, so the cut is per row and
    # comes from a tensor rather than a captured length: a captured int is a
    # guard on its value, so every new sequence length would recompile the
    # kernel until the limit is reached and the call falls back to dense math.
    valid = torch.arange(padded, device=query.device) < rows

    # Built from the selected indices: evaluating a mask function over every
    # row pair would materialize a grid the size of the attention matrix.
    def mask_mod(batch, head, q_row, k_row):
        return valid[k_row]

    whole = keep.clone()
    whole[..., rows // cfg.block :] = False
    partial = keep & ~whole
    partial_counts, partial_indices = _counts_and_indices(partial)
    whole_counts, whole_indices = _counts_and_indices(whole)

    block_mask = BlockMask.from_kv_blocks(
        partial_counts,
        partial_indices,
        whole_counts,
        whole_indices,
        BLOCK_SIZE=cfg.block,
        mask_mod=mask_mod,
        seq_lengths=(padded, padded),
    )
    out = _compiled_flex()(query, key, value, block_mask=block_mask)
    if pad:
        out = out[..., :rows, :]
    return out.index_select(-2, plan.inverse) if plan is not None else out


def _miowtion_keep(
    scores: torch.Tensor, n_video: int, video_tokens: int, valid_tiles: torch.Tensor, keep_fraction: float
) -> torch.Tensor:
    """Miowtion top-k with equal-cost budgeting and Bresenham rounding."""
    if not 0.0 < keep_fraction <= 1.0 or not math.isfinite(keep_fraction):
        raise ValueError(f"keep_fraction must lie in (0, 1], got {keep_fraction}")
    batch, heads, _, n_tiles = scores.shape
    keep = torch.zeros((batch, heads, n_tiles, n_tiles), dtype=torch.bool, device=scores.device)
    keep[..., :, n_video:] = valid_tiles[n_video:]
    keep[..., n_video:, :] = valid_tiles
    video_valid = valid_tiles[:n_video]
    if keep_fraction == 1.0:
        keep[..., :n_video, :n_video] = video_valid
        return keep
    ideal = math.ceil(video_tokens / TILE_SIZE)
    per_row = keep_fraction * ideal * ideal / n_video
    low = max(1, min(n_video, math.floor(per_row)))
    high = max(1, min(n_video, math.ceil(per_row)))
    fraction = round(per_row - math.floor(per_row), 12)
    ranked_scores = scores[..., :n_video, :n_video].float().masked_fill(~video_valid, -torch.inf)
    diagonal = torch.arange(n_video, device=scores.device)
    ranked_scores[..., diagonal, diagonal] = torch.inf
    chosen = ranked_scores.topk(high, dim=-1).indices
    row = torch.arange(n_video, device=scores.device)
    extra = torch.floor((row + 1) * fraction) > torch.floor(row * fraction)
    allowed = low + extra.to(torch.long)
    take = torch.arange(high, device=scores.device).view(1, 1, 1, -1) < allowed.view(1, 1, -1, 1)
    keep[..., :n_video, :n_video].scatter_(-1, chosen, take.expand(batch, heads, -1, -1))
    return keep


def _valid_key_mask(valid_rows: torch.Tensor):
    def mask_mod(batch_index, head_index, query_row, key_row):
        return valid_rows[key_row]

    return mask_mod


def miowtion_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    bundle: MiowtionBundle,
    plan: MiowtionPlan,
    layer_index: int,
    keep_fraction: float | None = None,
) -> torch.Tensor:
    """Learned Miowtion block sparse attention over ``(B,H,S,D)`` Q/K/V."""
    if query.shape != key.shape or query.shape != value.shape or query.ndim != 4:
        raise ValueError("Miowtion query, key, and value must have the same (B,H,S,D) shape")
    _, heads, rows, dim = query.shape
    if rows != plan.rows or heads != bundle.num_heads or dim != bundle.head_dim:
        raise ValueError(
            f"Miowtion QKV geometry {(heads, rows, dim)} disagrees with bundle/plan "
            f"{(bundle.num_heads, plan.rows, bundle.head_dim)}"
        )
    if len(plan.head_shape) != bundle.num_layers or not 0 <= layer_index < bundle.num_layers:
        raise ValueError(f"invalid Miowtion layer_index {layer_index}")
    fraction = bundle.keep_fraction if keep_fraction is None else keep_fraction
    head_outputs: list[torch.Tensor | None] = [None] * heads
    groups = plan.head_groups(layer_index, query.device)
    gather_budget = 128 << 20
    bytes_per_head = rows * dim * query.element_size()
    max_group_heads = max(1, gather_budget // bytes_per_head)
    groups = tuple(MiowtionHeadGroup(group.shape, heads) for group in groups for heads in group.heads.split(max_group_heads))
    for group in groups:
        layout = plan.layout(group.shape, query.device)
        group_q, group_k, group_v = (
            tensor.index_select(1, group.heads).index_select(2, layout.gather_index) for tensor in (query, key, value)
        )
        invalid = ~layout.slot_valid
        if invalid.any():
            padding = invalid.view(1, 1, -1, 1)
            group_q.masked_fill_(padding, 0)
            group_k.masked_fill_(padding, 0)
            group_v.masked_fill_(padding, 0)
        with torch.no_grad():
            q_features = pool_miowtion_tiles(group_q, layout)
            k_features = pool_miowtion_tiles(group_k, layout)
            scores = bundle.predictor.layers[layer_index](q_features, k_features, group.heads)
            keep = _miowtion_keep(scores, layout.n_video_tiles, rows - plan.target_start, layout.valid_count > 0, fraction)
        whole = keep & layout.full_tile.view(1, 1, 1, -1)
        partial = keep & ~whole
        partial_counts, partial_indices = _counts_and_indices(partial)
        whole_counts, whole_indices = _counts_and_indices(whole)
        block_mask = BlockMask.from_kv_blocks(
            partial_counts,
            partial_indices,
            whole_counts,
            whole_indices,
            BLOCK_SIZE=TILE_SIZE,
            mask_mod=_valid_key_mask(layout.slot_valid),
            seq_lengths=(layout.n_tiles * TILE_SIZE, layout.n_tiles * TILE_SIZE),
        )
        # Dynamo specializes FlexAttention on the head extent despite
        # ``dynamic=True``. Keep one wrapper per extent so 50 layers with
        # different head-group sizes do not exhaust its recompile limit and
        # silently fall back to dense mathematical attention.
        kernel_key = ("miowtion", group_q.shape[1], layout.n_tiles)
        tiled_output = _compiled_flex(kernel_key)(group_q, group_k, group_v, block_mask=block_mask)
        restored = tiled_output.index_select(2, layout.inverse_slots)
        for local, global_head in enumerate(group.heads.tolist()):
            head_outputs[global_head] = restored[:, local]
    if any(output is None for output in head_outputs):
        raise RuntimeError("Miowtion plan did not assign every attention head")
    return torch.stack(head_outputs, dim=1)  # type: ignore[arg-type]
