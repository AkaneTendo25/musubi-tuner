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

    @triton.jit
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
        sequence: tl.constexpr,
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

    @triton.jit
    def _indexed_adaln_rmsnorm_bwd(
        grad_output,
        x,
        weight,
        scale,
        indices,
        inv_rms,
        grad_x,
        scale_stride,
        sequence: tl.constexpr,
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
        output,
        sequence: tl.constexpr,
        selected_blocks: tl.constexpr,
        head_dim: tl.constexpr,
        block_rows: tl.constexpr,
        scale_log2: tl.constexpr,
    ):
        """Flash-style reference attention with runtime block indices/weights.

        Unreachable from the trained paths and therefore untested on CPU: the
        online-softmax algebra below can only be exercised on a GPU.
        """
        query_block = tl.program_id(0)
        head = tl.program_id(1)
        row_offsets = query_block * block_rows + tl.arange(0, block_rows)
        dim_offsets = tl.arange(0, head_dim)
        row_mask = row_offsets < sequence
        head_base = head * sequence * head_dim
        q = tl.load(
            query + head_base + row_offsets[:, None] * head_dim + dim_offsets[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        running_max = tl.full((block_rows,), -float("inf"), tl.float32)
        running_sum = tl.zeros((block_rows,), tl.float32)
        accumulator = tl.zeros((block_rows, head_dim), tl.float32)
        route_base = (head * tl.cdiv(sequence, block_rows) + query_block) * selected_blocks

        for slot in range(selected_blocks):
            key_block = tl.load(block_indices + route_base + slot)
            importance = tl.load(block_weights + route_base + slot).to(tl.float32)
            key_rows = key_block * block_rows + tl.arange(0, block_rows)
            key_mask = (key_block >= 0) & (key_rows < sequence)
            k = tl.load(
                key + head_base + key_rows[:, None] * head_dim + dim_offsets[None, :],
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
                value + head_base + key_rows[:, None] * head_dim + dim_offsets[None, :],
                mask=key_mask[:, None],
                other=0.0,
            )
            accumulator = accumulator * old_scale[:, None] + tl.dot(probabilities.to(v.dtype), v)
            running_sum = running_sum * old_scale + block_sum
            running_max = next_max

        normalized = accumulator / running_sum[:, None]
        tl.store(
            output + head_base + row_offsets[:, None] * head_dim + dim_offsets[None, :],
            normalized,
            mask=row_mask[:, None],
        )

    @triton.jit
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
        sequence: tl.constexpr,
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

    @triton.jit
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
        sequence: tl.constexpr,
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

    @triton.jit
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
        sequence: tl.constexpr,
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

    @triton.jit
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
        sequence: tl.constexpr,
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


def reference_block_sparse_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_indices: torch.Tensor,
    block_weights: torch.Tensor,
    *,
    block_rows: int = 128,
) -> torch.Tensor:
    """Prototype forward kernel for regular H3 reference block sparsity.

    Inputs use ``[heads, sequence, head_dim]`` and routes use
    ``[heads, ceil(sequence / block_rows), selected_blocks]``. This deliberately
    exposes routes as tensors so changing them never creates a new compiled graph.
    Backward and packed non-reference composition are implemented only after the
    forward ceiling demonstrates that the target speedup is attainable.
    """
    if not HAS_TRITON or query.device.type != "cuda":
        raise RuntimeError("H3 reference block-sparse kernel requires Triton CUDA")
    if query.shape != key.shape or query.shape != value.shape or query.ndim != 3:
        raise ValueError("H3 block-sparse Q/K/V must have matching [heads, rows, dim] shapes")
    heads, sequence, head_dim = query.shape
    query_blocks = (sequence + block_rows - 1) // block_rows
    if head_dim != 128 or block_rows != 128:
        raise ValueError("H3 block-sparse prototype currently requires head_dim=block_rows=128")
    if block_indices.shape != block_weights.shape or block_indices.shape[:2] != (heads, query_blocks):
        raise ValueError("H3 block-sparse route tensors have incompatible geometry")
    if block_indices.dtype != torch.int32:
        raise ValueError("H3 block-sparse indices must be int32")
    output = torch.empty_like(query)
    selected_blocks = block_indices.shape[-1]
    _reference_block_sparse_fwd[(query_blocks, heads)](
        query,
        key,
        value,
        block_indices.contiguous(),
        block_weights.contiguous(),
        output,
        sequence=sequence,
        selected_blocks=selected_blocks,
        head_dim=head_dim,
        block_rows=block_rows,
        scale_log2=(head_dim**-0.5) * 1.4426950408889634,
        num_warps=8,
        num_stages=3,
    )
    return output
