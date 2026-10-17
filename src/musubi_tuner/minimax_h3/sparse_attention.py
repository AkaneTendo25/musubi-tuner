from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from musubi_tuner.minimax_h3.triton_kernels import reference_block_sparse_forward


@dataclass(frozen=True)
class H3ReferenceSparseLayout:
    reference_start: int
    reference_rows: int
    rows_per_frame: int
    permutation: torch.Tensor | None = None
    inverse_permutation: torch.Tensor | None = None
    block_first_frames: torch.Tensor | None = None
    block_last_frames: torch.Tensor | None = None

    def validate(self, sequence_rows: int) -> None:
        if self.reference_start < 0 or self.reference_rows <= 0 or self.rows_per_frame <= 0:
            raise ValueError("H3 sparse reference layout values must be positive")
        if self.reference_start + self.reference_rows > sequence_rows:
            raise ValueError("H3 sparse reference layout exceeds the packed sequence")
        if self.block_first_frames is None and self.reference_rows % self.rows_per_frame:
            raise ValueError("H3 sparse reference rows must contain complete latent frames")
        if (self.permutation is None) != (self.inverse_permutation is None):
            raise ValueError("H3 sparse reference layout requires both forward and inverse permutations")
        if self.permutation is not None and (
            self.permutation.shape != (sequence_rows,) or self.inverse_permutation.shape != (sequence_rows,)
        ):
            raise ValueError("H3 sparse reference permutations must match the packed sequence")
        if (self.block_first_frames is None) != (self.block_last_frames is None):
            raise ValueError("H3 sparse reference layout requires both block frame-span tensors")
        router_blocks = math.ceil(self.reference_rows / 128)
        if self.block_first_frames is not None and (
            self.block_first_frames.shape != (router_blocks,) or self.block_last_frames.shape != (router_blocks,)
        ):
            raise ValueError("H3 sparse reference frame spans must match its router blocks")


@dataclass(frozen=True)
class H3ReferenceSparseConfig:
    router_block_size: int = 128
    topk_blocks: int = 32
    temporal_radius: int = 2
    layer_ranges: tuple[tuple[int, int], ...] = ((20, 30), (40, 50))
    data_path: str = "packed"

    def applies_to_layer(self, layer_index: int) -> bool:
        return any(start <= layer_index < stop for start, stop in self.layer_ranges)

    def validate(self) -> None:
        if self.router_block_size != 128:
            raise ValueError("H3 sparse reference attention currently requires 128-row blocks")
        if self.topk_blocks <= 0 or self.temporal_radius < 0:
            raise ValueError("H3 sparse reference attention requires positive top-k and a non-negative radius")
        if not self.layer_ranges or any(start < 0 or stop <= start for start, stop in self.layer_ranges):
            raise ValueError("H3 sparse reference attention layer ranges are invalid")
        if self.data_path not in {"packed", "mapped_int64"}:
            raise ValueError("H3 sparse reference attention data path must be packed or mapped_int64")


class H3UniversalSparseRouter(nn.Module):
    """Small layer-conditioned index branch shared by every H3 attention block.

    The router consumes block summaries of H3's normalized, rotary Q/K tensors. It is
    trained ahead of concept training against dense block-attention mass, then frozen
    while the concept LoRA is trained. Its weights are not part of the resulting LoRA.
    """

    def __init__(
        self,
        *,
        num_layers: int = 50,
        num_heads: int = 56,
        head_dim: int = 128,
        index_dim: int = 32,
        block_size: int = 128,
        layer_specific_projections: bool = False,
        summary_subblocks: int = 1,
        projection_groups: int = 1,
        positive_kernel: bool = False,
    ) -> None:
        super().__init__()
        if min(num_layers, num_heads, head_dim, index_dim, block_size, summary_subblocks, projection_groups) <= 0:
            raise ValueError("H3 sparse router dimensions must be positive")
        if block_size % summary_subblocks:
            raise ValueError("H3 sparse router subblocks must divide its block size")
        if num_heads % projection_groups:
            raise ValueError("H3 sparse router projection groups must divide its head count")
        if projection_groups > 1 and not layer_specific_projections:
            raise ValueError("H3 sparse router grouped projections require layer-specific projections")
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.index_dim = index_dim
        self.block_size = block_size
        self.summary_subblocks = summary_subblocks
        self.summary_dim = head_dim * summary_subblocks
        self.layer_specific_projections = layer_specific_projections
        self.projection_groups = projection_groups
        self.positive_kernel = positive_kernel
        if layer_specific_projections:
            projection_shape = (
                (num_layers, self.summary_dim, index_dim)
                if projection_groups == 1
                else (num_layers, projection_groups, self.summary_dim, index_dim)
            )
            self.query_layer_projection = nn.Parameter(torch.empty(projection_shape))
            self.key_layer_projection = nn.Parameter(torch.empty(projection_shape))
            nn.init.xavier_uniform_(self.query_layer_projection)
            nn.init.xavier_uniform_(self.key_layer_projection)
        else:
            self.query_projection = nn.Linear(self.summary_dim, index_dim, bias=False)
            self.key_projection = nn.Linear(self.summary_dim, index_dim, bias=False)
        self.query_layer_scale = nn.Parameter(torch.ones(num_layers, num_heads, index_dim))
        self.key_layer_scale = nn.Parameter(torch.ones(num_layers, num_heads, index_dim))
        self.logit_scale = nn.Parameter(torch.zeros(num_layers, num_heads, 1, 1))

    def summarize(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise ValueError("H3 sparse router expects [batch, rows, heads, head_dim]")
        if tensor.shape[2:] != (self.num_heads, self.head_dim):
            raise ValueError("H3 sparse router input does not match its head geometry")
        blocks = math.ceil(tensor.shape[1] / self.block_size)
        padded_rows = blocks * self.block_size
        if padded_rows != tensor.shape[1]:
            tensor = F.pad(tensor, (0, 0, 0, 0, 0, padded_rows - tensor.shape[1]))
        rows_per_summary = self.block_size // self.summary_subblocks
        summaries = tensor.unflatten(1, (blocks, self.summary_subblocks, rows_per_summary)).mean(dim=3)
        return summaries.permute(0, 1, 3, 2, 4).flatten(3, 4)

    def score_summaries(
        self,
        query_summary: torch.Tensor,
        key_summary: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        if not 0 <= layer_index < self.num_layers:
            raise ValueError(f"H3 sparse router layer index {layer_index} is out of range")
        expected = (self.num_heads, self.summary_dim)
        if query_summary.ndim != 4 or query_summary.shape[2:] != expected:
            raise ValueError("H3 sparse router query summary has incompatible geometry")
        if key_summary.ndim != 4 or key_summary.shape[2:] != expected:
            raise ValueError("H3 sparse router key summary has incompatible geometry")
        if self.layer_specific_projections:
            if self.projection_groups == 1:
                query_projection = self.query_layer_projection[layer_index]
                key_projection = self.key_layer_projection[layer_index]
                query_index = torch.einsum("bqhd,di->bqhi", query_summary.detach().float(), query_projection)
                key_index = torch.einsum("bkhd,di->bkhi", key_summary.detach().float(), key_projection)
            else:
                heads_per_group = self.num_heads // self.projection_groups
                head_groups = torch.arange(self.num_heads, device=query_summary.device) // heads_per_group
                query_projection = self.query_layer_projection[layer_index, head_groups]
                key_projection = self.key_layer_projection[layer_index, head_groups]
                query_index = torch.einsum("bqhd,hdi->bqhi", query_summary.detach().float(), query_projection)
                key_index = torch.einsum("bkhd,hdi->bkhi", key_summary.detach().float(), key_projection)
        else:
            query_index = self.query_projection(query_summary.detach().float())
            key_index = self.key_projection(key_summary.detach().float())
        query_index = query_index * self.query_layer_scale[layer_index]
        key_index = key_index * self.key_layer_scale[layer_index]
        if self.positive_kernel:
            query_features = F.softplus(query_index)
            key_features = F.softplus(key_index)
            kernel = torch.einsum("bqhd,bkhd->bhqk", query_features, key_features)
            return kernel.clamp_min(1e-12).log() + self.logit_scale[layer_index]
        query_index = F.normalize(query_index, dim=-1)
        key_index = F.normalize(key_index, dim=-1)
        scores = torch.einsum("bqhd,bkhd->bhqk", query_index, key_index)
        return scores * self.logit_scale[layer_index].exp()

    def forward(self, query: torch.Tensor, key: torch.Tensor, layer_index: int) -> torch.Tensor:
        return self.score_summaries(self.summarize(query), self.summarize(key), layer_index)

    @staticmethod
    def teacher_kl_loss(scores: torch.Tensor, teacher_mass: torch.Tensor) -> torch.Tensor:
        """Match dense per-block attention mass without backpropagating into H3."""
        if scores.shape != teacher_mass.shape:
            raise ValueError("H3 sparse router scores and teacher mass must have the same shape")
        target = teacher_mass.detach().float().clamp_min(0)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        per_row = F.kl_div(scores.float().log_softmax(dim=-1), target, reduction="none").sum(dim=-1)
        return per_row.mean()


def _flex_attention_api():
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except ImportError as error:
        raise RuntimeError("H3 sparse reference attention requires PyTorch FlexAttention") from error
    return create_block_mask, flex_attention


_compiled_flex_attention = None


def _compiled_flex():
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        _, flex_attention = _flex_attention_api()
        _compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
    return _compiled_flex_attention


def _select_router_blocks(scores: torch.Tensor, maximum_blocks: int) -> torch.Tensor:
    """Return the deterministic top-k reference blocks for each query and head."""
    maximum_blocks = min(maximum_blocks, scores.shape[-1])
    selected = torch.zeros_like(scores, dtype=torch.bool)
    selected.scatter_(dim=-1, index=scores.topk(maximum_blocks, dim=-1).indices, value=True)
    return selected


def _triton_hybrid_reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: H3ReferenceSparseLayout,
    selected_reference_blocks: torch.Tensor,
    reference_weights: torch.Tensor | None,
    block_size: int,
    data_path: str,
) -> torch.Tensor:
    """Run dense non-reference queries and routed reference queries separately."""
    reference_stop = layout.reference_start + layout.reference_rows
    before_rows = layout.reference_start
    after_rows = query.shape[1] - reference_stop

    nonreference_query = torch.cat((query[:, :before_rows], query[:, reference_stop:]), dim=1)
    dense_nonreference = F.scaled_dot_product_attention(
        nonreference_query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
    )

    def pad_rows(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = tensor.shape[1]
        padded_rows = math.ceil(rows / block_size) * block_size if rows else 0
        padded = F.pad(tensor, (0, 0, 0, 0, 0, padded_rows - rows)) if padded_rows != rows else tensor
        valid = torch.arange(padded_rows, device=tensor.device) < rows
        return padded, valid

    segment_rows = (before_rows, layout.reference_rows, after_rows)
    padded_segment_rows = tuple(math.ceil(rows / block_size) * block_size if rows else 0 for rows in segment_rows)
    if data_path == "packed":
        key_before, valid_before = pad_rows(key[:, :before_rows])
        key_reference, valid_reference = pad_rows(key[:, before_rows:reference_stop])
        key_after, valid_after = pad_rows(key[:, reference_stop:])
        value_before, _ = pad_rows(value[:, :before_rows])
        value_reference, _ = pad_rows(value[:, before_rows:reference_stop])
        value_after, _ = pad_rows(value[:, reference_stop:])
        query_reference, _ = pad_rows(query[:, before_rows:reference_stop])

        # Materialize head-major operands for contiguous kernel access.
        sparse_key = torch.cat((key_before, key_reference, key_after), dim=1).transpose(1, 2)[0].contiguous()
        sparse_value = torch.cat((value_before, value_reference, value_after), dim=1).transpose(1, 2)[0].contiguous()
        kv_valid = torch.cat((valid_before, valid_reference, valid_after))
        sparse_query = query_reference.transpose(1, 2)[0].contiguous()
        kv_row_indices = None
    elif data_path == "mapped_int64":
        # Keep token-major Q/K/V and map padded logical blocks to physical rows.
        # Triton widens physical pointer arithmetic to int64, which is required
        # for fused QKV around 100k rows.
        physical_segments = (
            torch.arange(0, before_rows, device=query.device, dtype=torch.int32),
            torch.arange(before_rows, reference_stop, device=query.device, dtype=torch.int32),
            torch.arange(reference_stop, query.shape[1], device=query.device, dtype=torch.int32),
        )
        logical_segments = [
            F.pad(rows, (0, padded - rows.numel()), value=-1) if padded != rows.numel() else rows
            for rows, padded in zip(physical_segments, padded_segment_rows, strict=True)
        ]
        kv_row_indices = torch.cat(logical_segments)
        kv_valid = kv_row_indices >= 0
        sparse_query = query[0, before_rows:reference_stop].transpose(0, 1)
        sparse_key = key[0].transpose(0, 1)
        sparse_value = value[0].transpose(0, 1)
    else:
        raise ValueError(f"unsupported H3 sparse reference data path: {data_path}")

    heads, query_blocks, reference_blocks = selected_reference_blocks.shape
    before_blocks = padded_segment_rows[0] // block_size
    after_blocks = padded_segment_rows[2] // block_size
    total_blocks = before_blocks + reference_blocks + after_blocks
    selected = torch.zeros((heads, query_blocks, total_blocks), device=query.device, dtype=torch.bool)
    weights = torch.ones((heads, query_blocks, total_blocks), device=query.device, dtype=torch.float32)
    if before_blocks:
        selected[..., :before_blocks] = True
    selected[..., before_blocks : before_blocks + reference_blocks] = selected_reference_blocks
    if reference_weights is not None:
        weights[..., before_blocks : before_blocks + reference_blocks] = reference_weights
    if after_blocks:
        selected[..., before_blocks + reference_blocks :] = True

    route_counts = selected.sum(dim=-1)
    # Triton unrolls the route loop, so quantize its constexpr width to avoid
    # compiling a new kernel for every small layer-to-layer count variation.
    route_slots = math.ceil(int(route_counts.max().item()) / 16) * 16
    sorted_blocks = torch.argsort(selected.to(torch.int8), dim=-1, descending=True)
    if route_slots > total_blocks:
        sorted_blocks = F.pad(sorted_blocks, (0, route_slots - total_blocks))
    slot_valid = torch.arange(route_slots, device=query.device) < route_counts[..., None]
    block_indices = sorted_blocks[..., :route_slots].to(torch.int32).masked_fill(~slot_valid, -1)
    block_weights = weights.gather(-1, sorted_blocks[..., :route_slots]).masked_fill(~slot_valid, 0)
    sparse_reference = reference_block_sparse_forward(
        sparse_query,
        sparse_key,
        sparse_value,
        block_indices,
        block_weights,
        block_rows=block_size,
        kv_valid_mask=kv_valid,
        _kv_row_indices=kv_row_indices,
    )[:, : layout.reference_rows]
    sparse_reference = sparse_reference.unsqueeze(0)

    dense_before = dense_nonreference[:, :, :before_rows]
    dense_after = dense_nonreference[:, :, before_rows : before_rows + after_rows]
    output = torch.cat((dense_before, sparse_reference, dense_after), dim=2)
    return output.transpose(1, 2).flatten(2, 3).to(query.dtype)


def reference_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: H3ReferenceSparseLayout,
    config: H3ReferenceSparseConfig,
    routing_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply head-specific routed attention to a packed single-reference sequence.

    Query, key, and value use H3's ``[batch, rows, heads, head_dim]`` layout.
    Non-reference queries and all interactions with non-reference keys remain dense.
    """
    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("H3 sparse reference attention requires matching rank-four Q/K/V tensors")
    if query.shape[0] != 1:
        raise ValueError("H3 sparse reference attention currently requires batch size one")
    config.validate()
    sequence_rows = query.shape[1]
    layout.validate(sequence_rows)
    reference_stop = layout.reference_start + layout.reference_rows
    router_blocks = (layout.reference_rows + config.router_block_size - 1) // config.router_block_size
    topk_blocks = min(config.topk_blocks, router_blocks)

    query_reference = key_reference = None
    if routing_scores is None:
        query_reference = query[:, layout.reference_start : reference_stop]
        key_reference = key[:, layout.reference_start : reference_stop]
        padded_rows = router_blocks * config.router_block_size
        if padded_rows != layout.reference_rows:
            pad = padded_rows - layout.reference_rows
            query_reference = F.pad(query_reference, (0, 0, 0, 0, 0, pad))
            key_reference = F.pad(key_reference, (0, 0, 0, 0, 0, pad))
    if routing_scores is None:
        query_blocks = query_reference.unflatten(1, (router_blocks, config.router_block_size)).mean(dim=2)
        key_blocks = key_reference.unflatten(1, (router_blocks, config.router_block_size)).mean(dim=2)
        per_head_scores = torch.einsum("bqhd,bkhd->bhqk", query_blocks, key_blocks)[0]
    else:
        if routing_scores.ndim == 4:
            if routing_scores.shape[0] != 1:
                raise ValueError("H3 sparse reference routing scores currently require batch size one")
            routing_scores = routing_scores[0]
        expected_scores = (query.shape[2], router_blocks, router_blocks)
        if routing_scores.shape != expected_scores:
            raise ValueError(
                f"H3 sparse reference routing scores have shape {tuple(routing_scores.shape)}, expected {expected_scores}"
            )
        per_head_scores = routing_scores.detach()
    selected_by_head = _select_router_blocks(per_head_scores, topk_blocks)

    if layout.block_first_frames is None:
        router_offsets = torch.arange(router_blocks, device=query.device) * config.router_block_size
        router_first_frames = router_offsets // layout.rows_per_frame
        router_last_frames = (router_offsets + config.router_block_size - 1) // layout.rows_per_frame
    else:
        router_first_frames = layout.block_first_frames.to(query.device)
        router_last_frames = layout.block_last_frames.to(query.device)
    router_local = (router_first_frames[:, None] - router_last_frames[None, :] <= config.temporal_radius) & (
        router_first_frames[None, :] - router_last_frames[:, None] <= config.temporal_radius
    )
    selected_by_head |= router_local.unsqueeze(0)
    if query.is_cuda and query.shape[-1] == 128:
        return _triton_hybrid_reference_attention(
            query,
            key,
            value,
            layout,
            selected_by_head,
            None,
            config.router_block_size,
            config.data_path,
        )

    def reference_mask(_batch, head, query_index, key_index):
        query_is_reference = (query_index >= layout.reference_start) & (query_index < reference_stop)
        key_is_reference = (key_index >= layout.reference_start) & (key_index < reference_stop)
        query_block = ((query_index - layout.reference_start) // config.router_block_size).clamp(0, router_blocks - 1)
        key_block = ((key_index - layout.reference_start) // config.router_block_size).clamp(0, router_blocks - 1)
        return (~query_is_reference) | (~key_is_reference) | selected_by_head[head, query_block, key_block]

    create_block_mask, _ = _flex_attention_api()
    block_mask = create_block_mask(
        reference_mask,
        1,
        query.shape[2],
        sequence_rows,
        sequence_rows,
        device=query.device,
        BLOCK_SIZE=config.router_block_size,
    )
    flex_query = query.transpose(1, 2)
    flex_key = key.transpose(1, 2)
    flex_value = value.transpose(1, 2)
    output = _compiled_flex()(
        flex_query,
        flex_key,
        flex_value,
        block_mask=block_mask,
        score_mod=None,
    )
    return output.transpose(1, 2).flatten(2, 3).to(query.dtype)
