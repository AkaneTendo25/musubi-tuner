"""Optional Triton kernels for MiniMax H3's repeated transformer primitives."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - depends on the runtime
    triton = None
    tl = None
    HAS_TRITON = False

_LOGGED = False


if HAS_TRITON:

    @triton.jit
    def _swiglu_fwd(projected, output, elements, width: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < elements
        rows = offsets // width
        columns = offsets - rows * width
        base = rows * (2 * width) + columns
        gate = tl.load(projected + base, mask=mask, other=0.0).to(tl.float32)
        value = tl.load(projected + base + width, mask=mask, other=0.0).to(tl.float32)
        sigmoid = tl.sigmoid(gate)
        tl.store(output + offsets, gate * sigmoid * value, mask=mask)

    @triton.jit
    def _swiglu_bwd(grad_output, projected, grad_projected, elements, width: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < elements
        rows = offsets // width
        columns = offsets - rows * width
        base = rows * (2 * width) + columns
        grad = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(projected + base, mask=mask, other=0.0).to(tl.float32)
        value = tl.load(projected + base + width, mask=mask, other=0.0).to(tl.float32)
        sigmoid = tl.sigmoid(gate)
        silu = gate * sigmoid
        silu_grad = sigmoid * (1.0 + gate * (1.0 - sigmoid))
        tl.store(grad_projected + base, grad * value * silu_grad, mask=mask)
        tl.store(grad_projected + base + width, grad * silu, mask=mask)

    @triton.jit(do_not_specialize=["sequence"])
    def _indexed_adaln_rmsnorm_fwd(
        x,
        weight,
        shift,
        scale,
        indices,
        output,
        inv_rms,
        shift_stride,
        scale_stride,
        sequence,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        token = row % sequence
        offsets = tl.arange(0, block)
        mask = offsets < hidden
        values = tl.load(x + row * hidden + offsets, mask=mask, other=0.0).to(tl.float32)
        modulation_row = tl.load(indices + token)
        weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        shifts = tl.load(shift + modulation_row * shift_stride + offsets, mask=mask, other=0.0).to(tl.float32)
        scales = tl.load(scale + modulation_row * scale_stride + offsets, mask=mask, other=0.0).to(tl.float32)
        inverse = tl.rsqrt(tl.sum(values * values, axis=0) / hidden + eps)
        result = values * inverse * weights * (1.0 + scales) + shifts
        tl.store(output + row * hidden + offsets, result, mask=mask)
        tl.store(inv_rms + row, inverse)

    @triton.jit(do_not_specialize=["sequence"])
    def _indexed_adaln_rmsnorm_bwd(
        grad_output,
        x,
        weight,
        scale,
        indices,
        inv_rms,
        grad_x,
        scale_stride,
        sequence,
        hidden: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        token = row % sequence
        offsets = tl.arange(0, block)
        mask = offsets < hidden
        modulation_row = tl.load(indices + token)
        grad = tl.load(grad_output + row * hidden + offsets, mask=mask, other=0.0).to(tl.float32)
        values = tl.load(x + row * hidden + offsets, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        scales = tl.load(scale + modulation_row * scale_stride + offsets, mask=mask, other=0.0).to(tl.float32)
        inverse = tl.load(inv_rms + row)
        grad_normalized = grad * weights * (1.0 + scales)
        dot = tl.sum(grad_normalized * values, axis=0)
        dx = inverse * grad_normalized - values * (inverse * inverse * inverse / hidden) * dot
        tl.store(grad_x + row * hidden + offsets, dx, mask=mask)

    @triton.jit
    def _reference_block_sparse_fwd(
        query,
        key,
        value,
        block_indices,
        block_weights,
        active_heads,
        active_query_blocks,
        kv_valid_mask,
        kv_row_indices,
        output,
        logsumexp_log2,
        stride_qh,
        stride_qs,
        stride_kh,
        stride_ks,
        stride_vh,
        stride_vs,
        stride_oh,
        stride_os,
        query_sequence: tl.constexpr,
        kv_sequence: tl.constexpr,
        heads: tl.constexpr,
        selected_blocks: tl.constexpr,
        head_dim: tl.constexpr,
        block_rows: tl.constexpr,
        query_tile_rows: tl.constexpr,
        scale_log2: tl.constexpr,
        mapped_query_blocks: tl.constexpr,
        mapped_kv_rows: tl.constexpr,
    ):
        """Flash-style reference attention with runtime block indices/weights."""
        tiles_per_block: tl.constexpr = block_rows // query_tile_rows
        active_tile = tl.program_id(0)
        active = active_tile // tiles_per_block
        tile_in_block = active_tile % tiles_per_block
        if mapped_query_blocks:
            query_block = tl.load(active_query_blocks + active)
            head = tl.load(active_heads + active)
            route_base = active * selected_blocks
        else:
            query_block = active
            head = tl.program_id(1)
            route_base = (head * tl.cdiv(query_sequence, block_rows) + query_block) * selected_blocks
        row_offsets = query_block * block_rows + tile_in_block * query_tile_rows + tl.arange(0, query_tile_rows)
        row_offsets_64 = row_offsets.to(tl.int64)
        dim_offsets = tl.arange(0, head_dim)
        row_mask = row_offsets < query_sequence
        q = tl.load(
            query + head * stride_qh + row_offsets_64[:, None] * stride_qs + dim_offsets[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        running_max = tl.full((query_tile_rows,), -float("inf"), tl.float32)
        running_sum = tl.zeros((query_tile_rows,), tl.float32)
        accumulator = tl.zeros((query_tile_rows, head_dim), tl.float32)
        for slot in range(selected_blocks):
            key_block = tl.load(block_indices + route_base + slot)
            importance = tl.load(block_weights + route_base + slot).to(tl.float32)
            logical_key_rows = key_block * block_rows + tl.arange(0, block_rows)
            key_mask = (key_block >= 0) & (logical_key_rows < kv_sequence)
            key_mask &= tl.load(kv_valid_mask + logical_key_rows, mask=key_mask, other=0).to(tl.int1)
            if mapped_kv_rows:
                key_rows = tl.load(kv_row_indices + logical_key_rows, mask=key_mask, other=-1)
                key_mask &= key_rows >= 0
            else:
                key_rows = logical_key_rows
            key_rows_64 = key_rows.to(tl.int64)
            k = tl.load(
                key + head * stride_kh + key_rows_64[:, None] * stride_ks + dim_offsets[None, :],
                mask=key_mask[:, None],
                other=0.0,
            )
            logits = tl.dot(q, tl.trans(k)).to(tl.float32) * scale_log2
            logits += tl.log2(tl.maximum(importance, 1.0e-20))
            logits = tl.where(row_mask[:, None] & key_mask[None, :], logits, -float("inf"))
            block_max = tl.max(logits, axis=1)
            next_max = tl.maximum(running_max, block_max)
            # A padded route slot (key_block == -1) masks the whole block, so on
            # the leading slot next_max is still -inf and both exponents would be
            # -inf - -inf = NaN. Pin them: the rescale is a no-op (1.0) and a
            # fully masked block contributes nothing (0.0).
            empty = next_max == float("-inf")
            old_scale = tl.where(empty, 1.0, tl.exp2(running_max - next_max))
            probabilities = tl.where(empty[:, None], 0.0, tl.exp2(logits - next_max[:, None]))
            block_sum = tl.sum(probabilities, axis=1)
            v = tl.load(
                value + head * stride_vh + key_rows_64[:, None] * stride_vs + dim_offsets[None, :],
                mask=key_mask[:, None],
                other=0.0,
            )
            accumulator = accumulator * old_scale[:, None] + tl.dot(probabilities.to(v.dtype), v)
            running_sum = running_sum * old_scale + block_sum
            running_max = next_max

        normalized = accumulator / running_sum[:, None]
        tl.store(
            output + head * stride_oh + row_offsets_64[:, None] * stride_os + dim_offsets[None, :],
            normalized,
            mask=row_mask[:, None],
        )
        tl.store(
            logsumexp_log2 + head * query_sequence + row_offsets,
            running_max + tl.log2(running_sum),
            mask=row_mask,
        )

    @triton.jit
    def _reference_block_sparse_dq(
        query,
        key,
        value,
        block_indices,
        block_weights,
        active_heads,
        active_query_blocks,
        kv_valid_mask,
        kv_row_indices,
        output,
        logsumexp_log2,
        grad_output,
        grad_query,
        stride_qh,
        stride_qs,
        stride_kh,
        stride_ks,
        stride_vh,
        stride_vs,
        stride_oh,
        stride_os,
        stride_doh,
        stride_dos,
        stride_dqh,
        stride_dqs,
        query_sequence: tl.constexpr,
        kv_sequence: tl.constexpr,
        heads: tl.constexpr,
        selected_blocks: tl.constexpr,
        head_dim: tl.constexpr,
        block_rows: tl.constexpr,
        query_tile_rows: tl.constexpr,
        scale: tl.constexpr,
        scale_log2: tl.constexpr,
        mapped_query_blocks: tl.constexpr,
        mapped_kv_rows: tl.constexpr,
    ):
        query_tile = tl.program_id(0)
        tiles_per_block: tl.constexpr = block_rows // query_tile_rows
        active = query_tile // tiles_per_block
        tile_in_block = query_tile % tiles_per_block
        if mapped_query_blocks:
            query_block = tl.load(active_query_blocks + active)
            head = tl.load(active_heads + active)
            route_base = active * selected_blocks
        else:
            query_block = active
            head = tl.program_id(1)
            route_base = (head * tl.cdiv(query_sequence, block_rows) + query_block) * selected_blocks
        rows = query_block * block_rows + tile_in_block * query_tile_rows + tl.arange(0, query_tile_rows)
        rows_64 = rows.to(tl.int64)
        dims = tl.arange(0, head_dim)
        row_mask = rows < query_sequence
        q = tl.load(
            query + head * stride_qh + rows_64[:, None] * stride_qs + dims[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        out = tl.load(
            output + head * stride_oh + rows_64[:, None] * stride_os + dims[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        dout = tl.load(
            grad_output + head * stride_doh + rows_64[:, None] * stride_dos + dims[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        lse = tl.load(logsumexp_log2 + head * query_sequence + rows, mask=row_mask, other=0.0)
        delta = tl.sum(out.to(tl.float32) * dout.to(tl.float32), axis=1)
        dq = tl.zeros((query_tile_rows, head_dim), tl.float32)
        for slot in range(selected_blocks):
            key_block = tl.load(block_indices + route_base + slot)
            importance = tl.load(block_weights + route_base + slot).to(tl.float32)
            logical_key_rows = key_block * block_rows + tl.arange(0, block_rows)
            key_mask = (key_block >= 0) & (logical_key_rows < kv_sequence)
            key_mask &= tl.load(kv_valid_mask + logical_key_rows, mask=key_mask, other=0).to(tl.int1)
            if mapped_kv_rows:
                key_rows = tl.load(kv_row_indices + logical_key_rows, mask=key_mask, other=-1)
                key_mask &= key_rows >= 0
            else:
                key_rows = logical_key_rows
            key_rows_64 = key_rows.to(tl.int64)
            k = tl.load(
                key + head * stride_kh + key_rows_64[:, None] * stride_ks + dims[None, :],
                mask=key_mask[:, None],
                other=0.0,
            )
            v = tl.load(
                value + head * stride_vh + key_rows_64[:, None] * stride_vs + dims[None, :],
                mask=key_mask[:, None],
                other=0.0,
            )
            logits = tl.dot(q, tl.trans(k)).to(tl.float32) * scale_log2
            logits += tl.log2(tl.maximum(importance, 1.0e-20))
            p = tl.exp2(logits - lse[:, None])
            p = tl.where(row_mask[:, None] & key_mask[None, :], p, 0.0)
            dp = tl.dot(dout, tl.trans(v)).to(tl.float32)
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k).to(tl.float32) * scale
        tl.store(
            grad_query + head * stride_dqh + rows_64[:, None] * stride_dqs + dims[None, :],
            dq,
            mask=row_mask[:, None],
        )

    @triton.jit
    def _reference_block_sparse_dkv(
        query,
        key,
        value,
        active_heads,
        active_key_blocks,
        kv_valid_mask,
        kv_row_indices,
        incoming_query_blocks,
        incoming_weights,
        incoming_counts,
        output,
        logsumexp_log2,
        grad_output,
        grad_key,
        grad_value,
        stride_qh,
        stride_qs,
        stride_kh,
        stride_ks,
        stride_vh,
        stride_vs,
        stride_oh,
        stride_os,
        stride_doh,
        stride_dos,
        stride_dkh,
        stride_dks,
        stride_dvh,
        stride_dvs,
        query_sequence: tl.constexpr,
        kv_sequence: tl.constexpr,
        max_incoming: tl.constexpr,
        head_dim: tl.constexpr,
        block_rows: tl.constexpr,
        query_tile_rows: tl.constexpr,
        key_tile_rows: tl.constexpr,
        scale: tl.constexpr,
        scale_log2: tl.constexpr,
        mapped_kv_rows: tl.constexpr,
    ):
        active_tile = tl.program_id(0)
        tiles_per_block: tl.constexpr = block_rows // key_tile_rows
        active = active_tile // tiles_per_block
        tile_in_block = active_tile % tiles_per_block
        head = tl.load(active_heads + active)
        key_block = tl.load(active_key_blocks + active)
        count = tl.load(incoming_counts + active)
        logical_key_rows = key_block * block_rows + tile_in_block * key_tile_rows + tl.arange(0, key_tile_rows)
        dims = tl.arange(0, head_dim)
        key_mask = logical_key_rows < kv_sequence
        key_mask &= tl.load(kv_valid_mask + logical_key_rows, mask=key_mask, other=0).to(tl.int1)
        if mapped_kv_rows:
            key_rows = tl.load(kv_row_indices + logical_key_rows, mask=key_mask, other=-1)
            key_mask &= key_rows >= 0
        else:
            key_rows = logical_key_rows
        key_rows_64 = key_rows.to(tl.int64)
        k = tl.load(
            key + head * stride_kh + key_rows_64[:, None] * stride_ks + dims[None, :],
            mask=key_mask[:, None],
            other=0.0,
        )
        v = tl.load(
            value + head * stride_vh + key_rows_64[:, None] * stride_vs + dims[None, :],
            mask=key_mask[:, None],
            other=0.0,
        )
        dk = tl.zeros((key_tile_rows, head_dim), tl.float32)
        dv = tl.zeros((key_tile_rows, head_dim), tl.float32)
        incoming_base = active * max_incoming
        for slot in range(max_incoming):
            query_block = tl.load(incoming_query_blocks + incoming_base + slot)
            importance = tl.load(incoming_weights + incoming_base + slot).to(tl.float32)
            for query_tile in range(0, block_rows, query_tile_rows):
                query_rows = query_block * block_rows + query_tile + tl.arange(0, query_tile_rows)
                query_rows_64 = query_rows.to(tl.int64)
                query_mask = (slot < count) & (query_rows < query_sequence)
                q = tl.load(
                    query + head * stride_qh + query_rows_64[:, None] * stride_qs + dims[None, :],
                    mask=query_mask[:, None],
                    other=0.0,
                )
                out = tl.load(
                    output + head * stride_oh + query_rows_64[:, None] * stride_os + dims[None, :],
                    mask=query_mask[:, None],
                    other=0.0,
                )
                dout = tl.load(
                    grad_output + head * stride_doh + query_rows_64[:, None] * stride_dos + dims[None, :],
                    mask=query_mask[:, None],
                    other=0.0,
                )
                lse = tl.load(logsumexp_log2 + head * query_sequence + query_rows, mask=query_mask, other=0.0)
                delta = tl.sum(out.to(tl.float32) * dout.to(tl.float32), axis=1)
                logits = tl.dot(q, tl.trans(k)).to(tl.float32) * scale_log2
                logits += tl.log2(tl.maximum(importance, 1.0e-20))
                p = tl.exp2(logits - lse[:, None])
                p = tl.where(query_mask[:, None] & key_mask[None, :], p, 0.0)
                dp = tl.dot(dout, tl.trans(v)).to(tl.float32)
                ds = p * (dp - delta[:, None])
                dk += tl.dot(tl.trans(ds.to(q.dtype)), q).to(tl.float32) * scale
                dv += tl.dot(tl.trans(p.to(dout.dtype)), dout).to(tl.float32)
        tl.store(
            grad_key + head * stride_dkh + key_rows_64[:, None] * stride_dks + dims[None, :],
            dk,
            mask=key_mask[:, None],
        )
        tl.store(
            grad_value + head * stride_dvh + key_rows_64[:, None] * stride_dvs + dims[None, :],
            dv,
            mask=key_mask[:, None],
        )

    @triton.jit(do_not_specialize=["sequence"])
    def _rmsnorm_split_rope_fwd(
        x,
        weight,
        cos,
        sin,
        output,
        inv_rms,
        stride_xb,
        stride_xs,
        stride_xh,
        stride_xd,
        stride_cs,
        stride_cd,
        sequence,
        heads: tl.constexpr,
        dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = row % heads
        s = (row // heads) % sequence
        b = row // (heads * sequence)
        offsets = tl.arange(0, block)
        mask = offsets < dim
        x_base = x + b * stride_xb + s * stride_xs + h * stride_xh
        values = tl.load(x_base + offsets * stride_xd, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        inverse = tl.rsqrt(tl.sum(values * values, axis=0) / dim + eps)
        normalized = values * inverse * weights

        half = rotary_dim // 2
        partner_offsets = tl.where(offsets < half, offsets + half, offsets - half)
        partner_values = tl.load(x_base + partner_offsets * stride_xd, mask=offsets < rotary_dim, other=0.0).to(tl.float32)
        partner_weights = tl.load(weight + partner_offsets, mask=offsets < rotary_dim, other=0.0).to(tl.float32)
        partner = partner_values * inverse * partner_weights
        cos_values = tl.load(cos + s * stride_cs + offsets * stride_cd, mask=offsets < rotary_dim, other=1.0)
        sin_values = tl.load(sin + s * stride_cs + offsets * stride_cd, mask=offsets < rotary_dim, other=0.0)
        rotated = tl.where(offsets < half, -partner, partner)
        result = tl.where(offsets < rotary_dim, normalized * cos_values + rotated * sin_values, normalized)
        tl.store(output + row * dim + offsets, result, mask=mask)
        tl.store(inv_rms + row, inverse)

    @triton.jit(do_not_specialize=["sequence"])
    def _rmsnorm_split_rope_bwd(
        grad_output,
        x,
        weight,
        cos,
        sin,
        inv_rms,
        grad_x,
        stride_xb,
        stride_xs,
        stride_xh,
        stride_xd,
        stride_cs,
        stride_cd,
        sequence,
        heads: tl.constexpr,
        dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = row % heads
        s = (row // heads) % sequence
        b = row // (heads * sequence)
        offsets = tl.arange(0, block)
        mask = offsets < dim
        half = rotary_dim // 2
        partner_offsets = tl.where(offsets < half, offsets + half, offsets - half)

        grad = tl.load(grad_output + row * dim + offsets, mask=mask, other=0.0).to(tl.float32)
        partner_grad = tl.load(
            grad_output + row * dim + partner_offsets,
            mask=offsets < rotary_dim,
            other=0.0,
        ).to(tl.float32)
        cos_values = tl.load(cos + s * stride_cs + offsets * stride_cd, mask=offsets < rotary_dim, other=1.0)
        partner_sin = tl.load(
            sin + s * stride_cs + partner_offsets * stride_cd,
            mask=offsets < rotary_dim,
            other=0.0,
        )
        rope_grad = tl.where(
            offsets < half,
            grad * cos_values + partner_grad * partner_sin,
            grad * cos_values - partner_grad * partner_sin,
        )
        rope_grad = tl.where(offsets < rotary_dim, rope_grad, grad)

        x_base = x + b * stride_xb + s * stride_xs + h * stride_xh
        values = tl.load(x_base + offsets * stride_xd, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        inverse = tl.load(inv_rms + row)
        grad_normalized = rope_grad * weights
        dot = tl.sum(grad_normalized * values, axis=0)
        dx = inverse * grad_normalized - values * (inverse * inverse * inverse / dim) * dot
        tl.store(grad_x + row * dim + offsets, dx, mask=mask)

    @triton.jit(do_not_specialize=["sequence"])
    def _combined_qk_fwd(
        q,
        k,
        qw,
        kw,
        cos,
        sin,
        qo,
        ko,
        qinv,
        kinv,
        stride_b,
        stride_s,
        stride_h,
        stride_d,
        stride_cs,
        stride_cd,
        sequence,
        heads: tl.constexpr,
        dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = row % heads
        s = (row // heads) % sequence
        b = row // (heads * sequence)
        offsets = tl.arange(0, block)
        mask = offsets < dim
        base = b * stride_b + s * stride_s + h * stride_h
        qv = tl.load(q + base + offsets * stride_d, mask=mask, other=0.0).to(tl.float32)
        kv = tl.load(k + base + offsets * stride_d, mask=mask, other=0.0).to(tl.float32)
        qweight = tl.load(qw + offsets, mask=mask, other=0.0).to(tl.float32)
        kweight = tl.load(kw + offsets, mask=mask, other=0.0).to(tl.float32)
        qi = tl.rsqrt(tl.sum(qv * qv, axis=0) / dim + eps)
        ki = tl.rsqrt(tl.sum(kv * kv, axis=0) / dim + eps)
        qn = qv * qi * qweight
        kn = kv * ki * kweight
        half = rotary_dim // 2
        partner_offsets = tl.where(offsets < half, offsets + half, offsets - half)
        qp = tl.load(q + base + partner_offsets * stride_d, mask=offsets < rotary_dim, other=0.0).to(tl.float32)
        kp = tl.load(k + base + partner_offsets * stride_d, mask=offsets < rotary_dim, other=0.0).to(tl.float32)
        qpw = tl.load(qw + partner_offsets, mask=offsets < rotary_dim, other=0.0).to(tl.float32)
        kpw = tl.load(kw + partner_offsets, mask=offsets < rotary_dim, other=0.0).to(tl.float32)
        qp = qp * qi * qpw
        kp = kp * ki * kpw
        cv = tl.load(cos + s * stride_cs + offsets * stride_cd, mask=offsets < rotary_dim, other=1.0)
        sv = tl.load(sin + s * stride_cs + offsets * stride_cd, mask=offsets < rotary_dim, other=0.0)
        qr = tl.where(offsets < half, -qp, qp)
        kr = tl.where(offsets < half, -kp, kp)
        tl.store(qo + row * dim + offsets, tl.where(offsets < rotary_dim, qn * cv + qr * sv, qn), mask=mask)
        tl.store(ko + row * dim + offsets, tl.where(offsets < rotary_dim, kn * cv + kr * sv, kn), mask=mask)
        tl.store(qinv + row, qi)
        tl.store(kinv + row, ki)

    @triton.jit(do_not_specialize=["sequence"])
    def _combined_qk_bwd(
        qgo,
        kgo,
        q,
        k,
        qw,
        kw,
        cos,
        sin,
        qinv,
        kinv,
        qgx,
        kgx,
        stride_b,
        stride_s,
        stride_h,
        stride_d,
        stride_cs,
        stride_cd,
        sequence,
        heads: tl.constexpr,
        dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = row % heads
        s = (row // heads) % sequence
        b = row // (heads * sequence)
        offsets = tl.arange(0, block)
        mask = offsets < dim
        half = rotary_dim // 2
        partner_offsets = tl.where(offsets < half, offsets + half, offsets - half)
        qg = tl.load(qgo + row * dim + offsets, mask=mask, other=0.0).to(tl.float32)
        kg = tl.load(kgo + row * dim + offsets, mask=mask, other=0.0).to(tl.float32)
        qpg = tl.load(qgo + row * dim + partner_offsets, mask=offsets < rotary_dim, other=0.0).to(tl.float32)
        kpg = tl.load(kgo + row * dim + partner_offsets, mask=offsets < rotary_dim, other=0.0).to(tl.float32)
        cv = tl.load(cos + s * stride_cs + offsets * stride_cd, mask=offsets < rotary_dim, other=1.0)
        psv = tl.load(sin + s * stride_cs + partner_offsets * stride_cd, mask=offsets < rotary_dim, other=0.0)
        qrg = tl.where(offsets < half, qg * cv + qpg * psv, qg * cv - qpg * psv)
        krg = tl.where(offsets < half, kg * cv + kpg * psv, kg * cv - kpg * psv)
        qrg = tl.where(offsets < rotary_dim, qrg, qg)
        krg = tl.where(offsets < rotary_dim, krg, kg)
        base = b * stride_b + s * stride_s + h * stride_h
        qv = tl.load(q + base + offsets * stride_d, mask=mask, other=0.0).to(tl.float32)
        kv = tl.load(k + base + offsets * stride_d, mask=mask, other=0.0).to(tl.float32)
        qweight = tl.load(qw + offsets, mask=mask, other=0.0).to(tl.float32)
        kweight = tl.load(kw + offsets, mask=mask, other=0.0).to(tl.float32)
        qi = tl.load(qinv + row)
        ki = tl.load(kinv + row)
        qgn = qrg * qweight
        kgn = krg * kweight
        qdot = tl.sum(qgn * qv, axis=0)
        kdot = tl.sum(kgn * kv, axis=0)
        tl.store(qgx + row * dim + offsets, qi * qgn - qv * (qi * qi * qi / dim) * qdot, mask=mask)
        tl.store(kgx + row * dim + offsets, ki * kgn - kv * (ki * ki * ki / dim) * kdot, mask=mask)


class _FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected):
        width = projected.shape[-1] // 2
        output_shape = (*projected.shape[:-1], width)
        output = torch.empty(output_shape, device=projected.device, dtype=projected.dtype)
        elements = output.numel()
        block = 256
        _swiglu_fwd[(triton.cdiv(elements, block),)](projected, output, elements, width=width, block=block, num_warps=4)
        ctx.save_for_backward(projected)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (projected,) = ctx.saved_tensors
        width = projected.shape[-1] // 2
        elements = grad_output.numel()
        grad_projected = torch.empty_like(projected)
        block = 256
        _swiglu_bwd[(triton.cdiv(elements, block),)](
            grad_output.contiguous(),
            projected,
            grad_projected,
            elements,
            width=width,
            block=block,
            num_warps=4,
        )
        return grad_projected


class _FusedIndexedAdaLNRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, shift, scale, indices, eps):
        batch, sequence, hidden = x.shape
        rows = batch * sequence
        output = torch.empty_like(x)
        inv_rms = torch.empty(rows, device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(hidden)
        _indexed_adaln_rmsnorm_fwd[(rows,)](
            x,
            weight,
            shift,
            scale,
            indices,
            output,
            inv_rms,
            shift.stride(0),
            scale.stride(0),
            sequence=sequence,
            hidden=hidden,
            eps=float(eps),
            block=block,
            num_warps=8,
        )
        ctx.save_for_backward(x, weight, scale, indices, inv_rms)
        ctx.block = block
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, scale, indices, inv_rms = ctx.saved_tensors
        batch, sequence, hidden = x.shape
        grad_x = torch.empty_like(x)
        _indexed_adaln_rmsnorm_bwd[(batch * sequence,)](
            grad_output.contiguous(),
            x,
            weight,
            scale,
            indices,
            inv_rms,
            grad_x,
            scale.stride(0),
            sequence=sequence,
            hidden=hidden,
            block=ctx.block,
            num_warps=4,
        )
        return grad_x, None, None, None, None, None


class _FusedRMSNormSplitRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, cos, sin, eps):
        batch, sequence, heads, dim = x.shape
        rotary_dim = cos.shape[-1]
        output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        inv_rms = torch.empty(batch * sequence * heads, device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(dim)
        _rmsnorm_split_rope_fwd[(batch * sequence * heads,)](
            x,
            weight,
            cos,
            sin,
            output,
            inv_rms,
            *x.stride(),
            *cos.stride(),
            sequence=sequence,
            heads=heads,
            dim=dim,
            rotary_dim=rotary_dim,
            eps=float(eps),
            block=block,
        )
        ctx.save_for_backward(x, weight, cos, sin, inv_rms)
        ctx.block = block
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, cos, sin, inv_rms = ctx.saved_tensors
        batch, sequence, heads, dim = x.shape
        grad_x = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        _rmsnorm_split_rope_bwd[(batch * sequence * heads,)](
            grad_output.contiguous(),
            x,
            weight,
            cos,
            sin,
            inv_rms,
            grad_x,
            *x.stride(),
            *cos.stride(),
            sequence=sequence,
            heads=heads,
            dim=dim,
            rotary_dim=cos.shape[-1],
            block=ctx.block,
        )
        return grad_x, None, None, None, None


class _CombinedFusedQKRMSNormSplitRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, q_weight, k_weight, cos, sin, eps):
        batch, sequence, heads, dim = query.shape
        rows = batch * sequence * heads
        query_output = torch.empty_like(query)
        key_output = torch.empty_like(key)
        query_inv_rms = torch.empty(rows, device=query.device, dtype=torch.float32)
        key_inv_rms = torch.empty_like(query_inv_rms)
        block = triton.next_power_of_2(dim)
        _combined_qk_fwd[(rows,)](
            query,
            key,
            q_weight,
            k_weight,
            cos,
            sin,
            query_output,
            key_output,
            query_inv_rms,
            key_inv_rms,
            *query.stride(),
            *cos.stride(),
            sequence=sequence,
            heads=heads,
            dim=dim,
            rotary_dim=cos.shape[-1],
            eps=float(eps),
            block=block,
        )
        ctx.save_for_backward(query, key, q_weight, k_weight, cos, sin, query_inv_rms, key_inv_rms)
        ctx.block = block
        return query_output, key_output

    @staticmethod
    def backward(ctx, query_grad_output, key_grad_output):
        query, key, q_weight, k_weight, cos, sin, query_inv_rms, key_inv_rms = ctx.saved_tensors
        batch, sequence, heads, dim = query.shape
        query_grad = torch.empty_like(query)
        key_grad = torch.empty_like(key)
        _combined_qk_bwd[(batch * sequence * heads,)](
            query_grad_output.contiguous(),
            key_grad_output.contiguous(),
            query,
            key,
            q_weight,
            k_weight,
            cos,
            sin,
            query_inv_rms,
            key_inv_rms,
            query_grad,
            key_grad,
            *query.stride(),
            *cos.stride(),
            sequence=sequence,
            heads=heads,
            dim=dim,
            rotary_dim=cos.shape[-1],
            block=ctx.block,
        )
        return query_grad, key_grad, None, None, None, None, None


def try_fused_qk_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return fused Q/K results, or ``None`` when the safe eager path is required."""
    global _LOGGED
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", lambda: False)
    if not HAS_TRITON or is_compiling():
        return None
    if query.device.type != "cuda" or query.dtype not in (torch.float16, torch.bfloat16):
        return None
    if key.shape != query.shape or key.dtype != query.dtype:
        return None
    if query.ndim != 4 or query.shape[-1] > 256 or query.shape[-1] & (query.shape[-1] - 1):
        return None
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != query.shape[1]:
        return None
    if cos.shape[-1] <= 0 or cos.shape[-1] > query.shape[-1] or cos.shape[-1] % 2:
        return None
    if q_weight.requires_grad or k_weight.requires_grad:
        return None
    if not (query.stride(-1) == key.stride(-1) == cos.stride(-1) == 1):
        return None
    cos = cos.to(query.dtype)
    sin = sin.to(query.dtype)
    if not _LOGGED:
        logger.info("MiniMax H3: using fused Triton Q/K RMSNorm + split RoPE")
        _LOGGED = True
    if query.stride() == key.stride():
        return _CombinedFusedQKRMSNormSplitRoPE.apply(query, key, q_weight, k_weight, cos, sin, eps)
    return (
        _FusedRMSNormSplitRoPE.apply(query, q_weight, cos, sin, eps),
        _FusedRMSNormSplitRoPE.apply(key, k_weight, cos, sin, eps),
    )


def try_fused_indexed_adaln_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    """Fuse RMSNorm with token-indexed AdaLN modulation for frozen H3 bases."""
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", lambda: False)
    if not HAS_TRITON or is_compiling() or x.device.type != "cuda":
        return None
    if x.ndim != 3 or x.dtype not in (torch.float16, torch.bfloat16) or not x.is_contiguous():
        return None
    hidden = x.shape[-1]
    if hidden <= 0 or hidden > 8192:
        return None
    if weight.shape != (hidden,) or shift.ndim != 2 or scale.shape != shift.shape or shift.shape[1] != hidden:
        return None
    if indices.ndim != 1 or indices.shape[0] != x.shape[1] or indices.dtype not in (torch.int32, torch.int64):
        return None
    if any(t.device != x.device for t in (weight, shift, scale, indices)):
        return None
    if any(t.dtype != x.dtype for t in (weight, shift, scale)):
        return None
    if any(t.requires_grad for t in (weight, shift, scale)):
        return None
    if not weight.is_contiguous() or not indices.is_contiguous() or shift.stride(1) != 1 or scale.stride(1) != 1:
        return None
    return _FusedIndexedAdaLNRMSNorm.apply(x, weight, shift, scale, indices, eps)


def try_fused_swiglu(projected: torch.Tensor) -> torch.Tensor | None:
    """Apply H3's fused-gate SwiGLU without materializing gate/value outputs."""
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", lambda: False)
    if not HAS_TRITON or is_compiling() or projected.device.type != "cuda":
        return None
    if projected.ndim < 2 or projected.shape[-1] % 2 or projected.shape[-1] == 0:
        return None
    if projected.dtype not in (torch.float16, torch.bfloat16) or not projected.is_contiguous():
        return None
    return _FusedSwiGLU.apply(projected)


def _invert_block_routes(block_indices: torch.Tensor, block_weights: torch.Tensor, kv_blocks: int):
    """Build a compact key-major route table for conflict-free dK/dV."""
    heads, query_blocks, _ = block_indices.shape
    head_ids = torch.arange(heads, device=block_indices.device)[:, None, None].expand_as(block_indices)
    query_ids = torch.arange(query_blocks, device=block_indices.device)[None, :, None].expand_as(block_indices)
    valid = block_indices >= 0
    keys = block_indices[valid].long()
    linear_keys = head_ids[valid].long() * kv_blocks + keys
    order = linear_keys.argsort(stable=True)
    linear_keys = linear_keys[order]
    sorted_queries = query_ids[valid][order].to(torch.int32)
    sorted_weights = block_weights[valid][order]
    active, counts = torch.unique_consecutive(linear_keys, return_counts=True)
    max_incoming = int(counts.max().item())
    starts = counts.cumsum(0) - counts
    ranks = torch.arange(linear_keys.numel(), device=block_indices.device) - torch.repeat_interleave(starts, counts)
    rows = torch.repeat_interleave(torch.arange(active.numel(), device=block_indices.device), counts)
    incoming_queries = torch.full((active.numel(), max_incoming), -1, device=block_indices.device, dtype=torch.int32)
    incoming_weights = torch.zeros((active.numel(), max_incoming), device=block_weights.device, dtype=block_weights.dtype)
    incoming_queries[rows, ranks] = sorted_queries
    incoming_weights[rows, ranks] = sorted_weights
    return (
        (active // kv_blocks).to(torch.int32),
        (active % kv_blocks).to(torch.int32),
        incoming_queries,
        incoming_weights,
        counts.to(torch.int32),
    )


def _launch_reference_block_sparse_dkv(
    query,
    key,
    value,
    active_heads,
    active_key_blocks,
    kv_valid_mask,
    kv_row_indices,
    incoming_query_blocks,
    incoming_weights,
    incoming_counts,
    output,
    logsumexp_log2,
    grad_output,
    grad_key,
    grad_value,
    query_sequence,
    kv_sequence,
    max_incoming,
    head_dim,
    block_rows,
    query_tile_rows,
    key_tile_rows,
    num_warps,
    num_stages,
):
    mapped_kv_rows = kv_row_indices is not None
    if kv_row_indices is None:
        kv_row_indices = kv_valid_mask
    _reference_block_sparse_dkv[(active_heads.numel() * (block_rows // key_tile_rows),)](
        query,
        key,
        value,
        active_heads,
        active_key_blocks,
        kv_valid_mask,
        kv_row_indices,
        incoming_query_blocks,
        incoming_weights,
        incoming_counts,
        output,
        logsumexp_log2,
        grad_output,
        grad_key,
        grad_value,
        *query.stride()[:2],
        *key.stride()[:2],
        *value.stride()[:2],
        *output.stride()[:2],
        *grad_output.stride()[:2],
        *grad_key.stride()[:2],
        *grad_value.stride()[:2],
        query_sequence=query_sequence,
        kv_sequence=kv_sequence,
        max_incoming=max_incoming,
        head_dim=head_dim,
        block_rows=block_rows,
        query_tile_rows=query_tile_rows,
        key_tile_rows=key_tile_rows,
        scale=head_dim**-0.5,
        scale_log2=(head_dim**-0.5) * 1.4426950408889634,
        mapped_kv_rows=mapped_kv_rows,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _route_count_buckets(block_indices: torch.Tensor):
    """Yield compact head/query route tables grouped by rounded loop width."""
    _, query_blocks, _ = block_indices.shape
    counts = (block_indices >= 0).sum(dim=-1).flatten()
    maximum = int(counts.max().item())
    flattened = block_indices.flatten(0, 1)
    limit = 16
    lower = 0
    while lower < maximum:
        upper = min(limit, maximum)
        in_bucket = (counts > lower) & (counts <= upper)
        if bool(in_bucket.any()):
            rows = in_bucket.nonzero(as_tuple=False).flatten()
            yield (
                (rows // query_blocks).to(torch.int32),
                (rows % query_blocks).to(torch.int32),
                flattened[rows, :upper].contiguous(),
                upper,
            )
        lower = upper
        limit *= 2


def _launch_reference_block_sparse_fwd(
    query,
    key,
    value,
    block_indices,
    block_weights,
    kv_valid_mask,
    kv_row_indices,
    output,
    logsumexp_log2,
    block_rows,
    query_tile_rows,
    bucket_queries,
    num_warps,
    num_stages,
):
    heads, query_sequence, head_dim = query.shape
    kv_sequence = key.shape[1]
    mapped_kv_rows = kv_row_indices is not None
    if kv_row_indices is not None:
        kv_sequence = kv_row_indices.numel()
    else:
        kv_row_indices = kv_valid_mask
    query_blocks = (query_sequence + block_rows - 1) // block_rows
    tiles_per_block = block_rows // query_tile_rows
    if bucket_queries:
        flattened_weights = block_weights.flatten(0, 1)
        for active_heads, active_queries, bucket_indices, loop_width in _route_count_buckets(block_indices):
            route_rows = active_heads.long() * query_blocks + active_queries.long()
            bucket_weights = flattened_weights[route_rows, :loop_width].contiguous()
            _reference_block_sparse_fwd[(active_heads.numel() * tiles_per_block,)](
                query,
                key,
                value,
                bucket_indices,
                bucket_weights,
                active_heads,
                active_queries,
                kv_valid_mask,
                kv_row_indices,
                output,
                logsumexp_log2,
                *query.stride()[:2],
                *key.stride()[:2],
                *value.stride()[:2],
                *output.stride()[:2],
                query_sequence=query_sequence,
                kv_sequence=kv_sequence,
                heads=heads,
                selected_blocks=loop_width,
                head_dim=head_dim,
                block_rows=block_rows,
                query_tile_rows=query_tile_rows,
                scale_log2=(head_dim**-0.5) * 1.4426950408889634,
                mapped_query_blocks=True,
                mapped_kv_rows=mapped_kv_rows,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    else:
        _reference_block_sparse_fwd[(query_blocks * tiles_per_block, heads)](
            query,
            key,
            value,
            block_indices,
            block_weights,
            block_indices,
            block_indices,
            kv_valid_mask,
            kv_row_indices,
            output,
            logsumexp_log2,
            *query.stride()[:2],
            *key.stride()[:2],
            *value.stride()[:2],
            *output.stride()[:2],
            query_sequence=query_sequence,
            kv_sequence=kv_sequence,
            heads=heads,
            selected_blocks=block_indices.shape[-1],
            head_dim=head_dim,
            block_rows=block_rows,
            query_tile_rows=query_tile_rows,
            scale_log2=(head_dim**-0.5) * 1.4426950408889634,
            mapped_query_blocks=False,
            mapped_kv_rows=mapped_kv_rows,
            num_warps=num_warps,
            num_stages=num_stages,
        )


def _launch_reference_block_sparse_dq(
    query,
    key,
    value,
    block_indices,
    block_weights,
    kv_valid_mask,
    kv_row_indices,
    output,
    logsumexp_log2,
    grad_output,
    grad_query,
    block_rows,
    query_tile_rows,
    bucket_queries,
    num_warps,
):
    heads, query_sequence, head_dim = query.shape
    kv_sequence = key.shape[1]
    mapped_kv_rows = kv_row_indices is not None
    if kv_row_indices is not None:
        kv_sequence = kv_row_indices.numel()
    else:
        kv_row_indices = kv_valid_mask
    query_blocks = (query_sequence + block_rows - 1) // block_rows
    tiles_per_block = block_rows // query_tile_rows
    if bucket_queries:
        flattened_weights = block_weights.flatten(0, 1)
        for active_heads, active_queries, bucket_indices, loop_width in _route_count_buckets(block_indices):
            route_rows = active_heads.long() * query_blocks + active_queries.long()
            bucket_weights = flattened_weights[route_rows, :loop_width].contiguous()
            _reference_block_sparse_dq[(active_heads.numel() * tiles_per_block,)](
                query,
                key,
                value,
                bucket_indices,
                bucket_weights,
                active_heads,
                active_queries,
                kv_valid_mask,
                kv_row_indices,
                output,
                logsumexp_log2,
                grad_output,
                grad_query,
                *query.stride()[:2],
                *key.stride()[:2],
                *value.stride()[:2],
                *output.stride()[:2],
                *grad_output.stride()[:2],
                *grad_query.stride()[:2],
                query_sequence=query_sequence,
                kv_sequence=kv_sequence,
                heads=heads,
                selected_blocks=loop_width,
                head_dim=head_dim,
                block_rows=block_rows,
                query_tile_rows=query_tile_rows,
                scale=head_dim**-0.5,
                scale_log2=(head_dim**-0.5) * 1.4426950408889634,
                mapped_query_blocks=True,
                mapped_kv_rows=mapped_kv_rows,
                num_warps=num_warps,
                num_stages=1,
            )
    else:
        _reference_block_sparse_dq[(query_blocks * tiles_per_block, heads)](
            query,
            key,
            value,
            block_indices,
            block_weights,
            block_indices,
            block_indices,
            kv_valid_mask,
            kv_row_indices,
            output,
            logsumexp_log2,
            grad_output,
            grad_query,
            *query.stride()[:2],
            *key.stride()[:2],
            *value.stride()[:2],
            *output.stride()[:2],
            *grad_output.stride()[:2],
            *grad_query.stride()[:2],
            query_sequence=query_sequence,
            kv_sequence=kv_sequence,
            heads=heads,
            selected_blocks=block_indices.shape[-1],
            head_dim=head_dim,
            block_rows=block_rows,
            query_tile_rows=query_tile_rows,
            scale=head_dim**-0.5,
            scale_log2=(head_dim**-0.5) * 1.4426950408889634,
            mapped_query_blocks=False,
            mapped_kv_rows=mapped_kv_rows,
            num_warps=num_warps,
            num_stages=1,
        )


class _ReferenceBlockSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        block_indices,
        block_weights,
        kv_valid_mask,
        kv_row_indices,
        block_rows,
        bucket_dkv,
        bucket_queries,
        query_tile_rows,
        key_tile_rows,
        forward_num_warps,
        forward_num_stages,
        dq_num_warps,
        dkv_num_warps,
        dkv_num_stages,
    ):
        heads, query_sequence, _ = query.shape
        mapped_kv_rows = kv_row_indices if kv_row_indices.numel() else None
        kv_sequence = kv_row_indices.numel() if mapped_kv_rows is not None else key.shape[1]
        kv_blocks = (kv_sequence + block_rows - 1) // block_rows
        output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
        logsumexp_log2 = torch.empty((heads, query_sequence), device=query.device, dtype=torch.float32)
        _launch_reference_block_sparse_fwd(
            query,
            key,
            value,
            block_indices,
            block_weights,
            kv_valid_mask,
            mapped_kv_rows,
            output,
            logsumexp_log2,
            block_rows,
            query_tile_rows,
            bucket_queries,
            forward_num_warps,
            forward_num_stages,
        )
        ctx.save_for_backward(
            query,
            key,
            value,
            block_indices,
            block_weights,
            kv_valid_mask,
            kv_row_indices,
            output,
            logsumexp_log2,
        )
        ctx.kv_blocks = kv_blocks
        ctx.block_rows = block_rows
        ctx.bucket_dkv = bucket_dkv
        ctx.bucket_queries = bucket_queries
        ctx.query_tile_rows = query_tile_rows
        ctx.key_tile_rows = key_tile_rows
        ctx.dq_num_warps = dq_num_warps
        ctx.dkv_num_warps = dkv_num_warps
        ctx.dkv_num_stages = dkv_num_stages
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            query,
            key,
            value,
            block_indices,
            block_weights,
            kv_valid_mask,
            kv_row_indices,
            output,
            logsumexp_log2,
        ) = ctx.saved_tensors
        _, query_sequence, head_dim = query.shape
        mapped_kv_rows = kv_row_indices if kv_row_indices.numel() else None
        kv_sequence = kv_row_indices.numel() if mapped_kv_rows is not None else key.shape[1]
        block_rows = ctx.block_rows
        grad_query = torch.empty(query.shape, device=query.device, dtype=query.dtype)
        grad_key = torch.zeros_like(key, memory_format=torch.preserve_format)
        grad_value = torch.zeros_like(value, memory_format=torch.preserve_format)
        query_tile_rows = ctx.query_tile_rows
        key_tile_rows = ctx.key_tile_rows
        active_heads, active_key_blocks, incoming_query_blocks, incoming_weights, incoming_counts = _invert_block_routes(
            block_indices, block_weights, ctx.kv_blocks
        )
        _launch_reference_block_sparse_dq(
            query,
            key,
            value,
            block_indices,
            block_weights,
            kv_valid_mask,
            mapped_kv_rows,
            output,
            logsumexp_log2,
            grad_output,
            grad_query,
            block_rows,
            query_tile_rows,
            ctx.bucket_queries,
            ctx.dq_num_warps,
        )
        maximum_incoming = incoming_query_blocks.shape[1]
        if ctx.bucket_dkv:
            # Group key blocks by rounded incoming-route count to bound each
            # kernel's compile-time loop width.
            bucket_limit = 16
            lower_limit = 0
            while lower_limit < maximum_incoming:
                upper_limit = min(bucket_limit, maximum_incoming)
                in_bucket = (incoming_counts > lower_limit) & (incoming_counts <= upper_limit)
                if bool(in_bucket.any()):
                    bucket_heads = active_heads[in_bucket].contiguous()
                    bucket_key_blocks = active_key_blocks[in_bucket].contiguous()
                    bucket_counts = incoming_counts[in_bucket].contiguous()
                    bucket_queries = incoming_query_blocks[in_bucket, :upper_limit].contiguous()
                    bucket_weights = incoming_weights[in_bucket, :upper_limit].contiguous()
                    _launch_reference_block_sparse_dkv(
                        query,
                        key,
                        value,
                        bucket_heads,
                        bucket_key_blocks,
                        kv_valid_mask,
                        mapped_kv_rows,
                        bucket_queries,
                        bucket_weights,
                        bucket_counts,
                        output,
                        logsumexp_log2,
                        grad_output,
                        grad_key,
                        grad_value,
                        query_sequence,
                        kv_sequence,
                        upper_limit,
                        head_dim,
                        block_rows,
                        query_tile_rows,
                        key_tile_rows,
                        ctx.dkv_num_warps,
                        ctx.dkv_num_stages,
                    )
                lower_limit = upper_limit
                bucket_limit *= 2
        else:
            _launch_reference_block_sparse_dkv(
                query,
                key,
                value,
                active_heads,
                active_key_blocks,
                kv_valid_mask,
                mapped_kv_rows,
                incoming_query_blocks,
                incoming_weights,
                incoming_counts,
                output,
                logsumexp_log2,
                grad_output,
                grad_key,
                grad_value,
                query_sequence,
                kv_sequence,
                maximum_incoming,
                head_dim,
                block_rows,
                query_tile_rows,
                key_tile_rows,
                ctx.dkv_num_warps,
                ctx.dkv_num_stages,
            )
        return (grad_query, grad_key, grad_value) + (None,) * 14


def reference_block_sparse_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_indices: torch.Tensor,
    block_weights: torch.Tensor,
    *,
    block_rows: int = 128,
    kv_valid_mask: torch.Tensor | None = None,
    _kv_row_indices: torch.Tensor | None = None,
    _bucket_dkv: bool = True,
    _bucket_queries: bool = True,
    _query_tile_rows: int = 32,
    _key_tile_rows: int = 16,
    _forward_num_warps: int = 4,
    _forward_num_stages: int = 1,
    _dq_num_warps: int = 4,
    _dkv_num_warps: int = 4,
    _dkv_num_stages: int = 1,
) -> torch.Tensor:
    """Prototype forward kernel for regular H3 reference block sparsity.

    Inputs use query ``[heads, query_rows, head_dim]`` and key/value
    ``[heads, kv_rows, head_dim]``. Routes use
    ``[heads, ceil(query_rows / block_rows), selected_blocks]``. This deliberately
    exposes routes as tensors so changing them never creates a new compiled graph.
    Routes are ordinary runtime tensors, so changing them does not create a new
    compiled graph. The autograd path computes exact gradients for the selected
    weighted softmax; packed non-reference composition remains separate.
    """
    if not HAS_TRITON or query.device.type != "cuda":
        raise RuntimeError("H3 reference block-sparse kernel requires Triton CUDA")
    if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape:
        raise ValueError("H3 block-sparse query and matching key/value must be rank-3 tensors")
    heads, query_sequence, head_dim = query.shape
    if key.shape[0] != heads or key.shape[2] != head_dim:
        raise ValueError("H3 block-sparse query and key/value must have matching heads and head_dim")
    query_blocks = (query_sequence + block_rows - 1) // block_rows
    logical_kv_sequence = _kv_row_indices.numel() if _kv_row_indices is not None else key.shape[1]
    kv_blocks = (logical_kv_sequence + block_rows - 1) // block_rows
    if head_dim != 128 or block_rows != 128:
        raise ValueError("H3 block-sparse attention requires head_dim=block_rows=128")
    if block_rows % _query_tile_rows or block_rows % _key_tile_rows:
        raise ValueError("H3 block-sparse backward tile rows must divide the 128-row block")
    if _query_tile_rows not in (32, 64) or _key_tile_rows not in (16, 32, 64):
        raise ValueError("H3 block-sparse backward tile rows use supported powers of two")
    if any(warps not in (4, 8) for warps in (_forward_num_warps, _dq_num_warps, _dkv_num_warps)):
        raise ValueError("H3 block-sparse kernels support four or eight warps")
    if _forward_num_stages not in (1, 2, 3, 4) or _dkv_num_stages not in (1, 2, 3):
        raise ValueError("H3 block-sparse kernel pipeline stages are outside the supported range")
    if block_indices.shape != block_weights.shape or block_indices.shape[:2] != (heads, query_blocks):
        raise ValueError("H3 block-sparse route tensors have incompatible geometry")
    if block_indices.dtype != torch.int32:
        raise ValueError("H3 block-sparse indices must be int32")
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("H3 block-sparse Q/K/V head dimensions must be contiguous")
    if _kv_row_indices is None:
        kv_row_indices = torch.empty(0, device=key.device, dtype=torch.int32)
    else:
        if _kv_row_indices.shape != (logical_kv_sequence,) or _kv_row_indices.dtype != torch.int32:
            raise ValueError("H3 block-sparse KV row map must be int32 [logical_kv_rows]")
        mapped_rows = _kv_row_indices[_kv_row_indices >= 0]
        if mapped_rows.numel() and int(mapped_rows.max()) >= key.shape[1]:
            raise ValueError("H3 block-sparse KV row map points outside the physical KV sequence")
        kv_row_indices = _kv_row_indices.contiguous()
    if kv_valid_mask is None:
        kv_valid_mask = torch.ones(logical_kv_sequence, device=key.device, dtype=torch.bool)
    elif kv_valid_mask.shape != (logical_kv_sequence,) or kv_valid_mask.dtype != torch.bool:
        raise ValueError("H3 block-sparse KV validity mask must be boolean [kv_rows]")
    valid_indices = block_indices[block_indices >= 0]
    if valid_indices.numel() and int(valid_indices.max()) >= kv_blocks:
        raise ValueError("H3 block-sparse route contains a key block outside the KV sequence")
    return _ReferenceBlockSparseAttention.apply(
        query,
        key,
        value,
        block_indices.contiguous(),
        block_weights.contiguous(),
        kv_valid_mask.contiguous(),
        kv_row_indices,
        block_rows,
        _bucket_dkv,
        _bucket_queries,
        _query_tile_rows,
        _key_tile_rows,
        _forward_num_warps,
        _forward_num_stages,
        _dq_num_warps,
        _dkv_num_warps,
        _dkv_num_stages,
    )
