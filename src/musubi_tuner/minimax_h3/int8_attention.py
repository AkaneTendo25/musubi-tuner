"""H3-owned experimental INT8-QK attention with a training backward.

The forward kernel quantizes Q/K per row for the score tensor-core product,
keeps P*V and V in the model dtype, and retains online-softmax state in FP32.
On Ampere and newer GPUs, training reuses the optimized FlashAttention backward
without recomputing its forward. A native tiled backward remains the fallback.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - runtime dependent
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _quantize_rows_kernel(x, quantized, scales, rows: tl.constexpr, dim: tl.constexpr, block: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < dim
        values = tl.load(x + row * dim + offsets, mask=mask, other=0.0).to(tl.float32)
        maximum = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-8)
        scale = maximum / 127.0
        rounded = tl.extra.cuda.libdevice.rint(values / scale)
        rounded = tl.maximum(-127.0, tl.minimum(127.0, rounded))
        tl.store(quantized + row * dim + offsets, rounded.to(tl.int8), mask=mask)
        tl.store(scales + row, scale)

    @triton.jit
    def _int8_attention_fwd_kernel(
        query,
        key,
        value,
        key_scale,
        output,
        logsumexp,
        sequence,
        stride_bh,
        scale_log2: tl.constexpr,
        head_dim: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        store_logsumexp: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        rows_m = query_block * block_m + tl.arange(0, block_m)
        rows_n = tl.arange(0, block_n)
        dims = tl.arange(0, head_dim)
        mask_m = rows_m < sequence
        base = batch_head * stride_bh

        q_values = tl.load(query + base + rows_m[:, None] * head_dim + dims[None, :], mask=mask_m[:, None], other=0.0).to(
            tl.float32
        )
        q_maximum = tl.maximum(tl.max(tl.abs(q_values), axis=1), 1.0e-8)
        q_scale = q_maximum / 127.0
        q = tl.extra.cuda.libdevice.rint(q_values / q_scale[:, None])
        q = tl.maximum(-127.0, tl.minimum(127.0, q)).to(tl.int8)
        running_max = tl.full((block_m,), -float("inf"), tl.float32)
        running_sum = tl.zeros((block_m,), tl.float32)
        accumulator = tl.zeros((block_m, head_dim), tl.float32)

        for start_n in tl.range(0, sequence, block_n):
            key_rows = start_n + rows_n
            mask_n = key_rows < sequence
            k = tl.load(key + base + key_rows[:, None] * head_dim + dims[None, :], mask=mask_n[:, None], other=0)
            k_scale = tl.load(key_scale + batch_head * sequence + key_rows, mask=mask_n, other=0.0)
            scores = tl.dot(q, tl.trans(k)).to(tl.float32)
            scores *= q_scale[:, None] * k_scale[None, :] * scale_log2
            scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, -float("inf"))

            block_maximum = tl.max(scores, axis=1)
            next_maximum = tl.maximum(running_max, block_maximum)
            previous_correction = tl.exp2(running_max - next_maximum)
            probabilities = tl.exp2(scores - next_maximum[:, None])
            v = tl.load(value + base + key_rows[:, None] * head_dim + dims[None, :], mask=mask_n[:, None], other=0.0)
            accumulator = accumulator * previous_correction[:, None] + tl.dot(probabilities.to(v.dtype), v)
            running_sum = running_sum * previous_correction + tl.sum(probabilities, axis=1)
            running_max = next_maximum

        result = accumulator / running_sum[:, None]
        tl.store(output + base + rows_m[:, None] * head_dim + dims[None, :], result, mask=mask_m[:, None])
        if store_logsumexp:
            natural_lse = (running_max + tl.log2(running_sum)) * 0.6931471805599453
            tl.store(logsumexp + batch_head * sequence + rows_m, natural_lse, mask=mask_m)

    @triton.jit
    def _attention_backward_delta_kernel(output, grad_output, delta, head_dim: tl.constexpr, block: tl.constexpr):
        row = tl.program_id(0)
        dims = tl.arange(0, block)
        mask = dims < head_dim
        values = tl.load(output + row * head_dim + dims, mask=mask, other=0.0).to(tl.float32)
        gradients = tl.load(grad_output + row * head_dim + dims, mask=mask, other=0.0).to(tl.float32)
        tl.store(delta + row, tl.sum(values * gradients, axis=0))

    @triton.jit
    def _int8_attention_dq_kernel(
        query,
        key,
        value,
        query_scale,
        key_scale,
        grad_output,
        logsumexp,
        delta,
        grad_query,
        sequence,
        stride_bh,
        scale_log2: tl.constexpr,
        scale: tl.constexpr,
        head_dim: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        rows_m = query_block * block_m + tl.arange(0, block_m)
        rows_n = tl.arange(0, block_n)
        dims = tl.arange(0, head_dim)
        mask_m = rows_m < sequence
        base = batch_head * stride_bh

        q = tl.load(query + base + rows_m[:, None] * head_dim + dims[None, :], mask=mask_m[:, None], other=0)
        q_scale = tl.load(query_scale + batch_head * sequence + rows_m, mask=mask_m, other=0.0)
        grad_o = tl.load(grad_output + base + rows_m[:, None] * head_dim + dims[None, :], mask=mask_m[:, None], other=0.0)
        lse = tl.load(logsumexp + batch_head * sequence + rows_m, mask=mask_m, other=0.0) * 1.4426950408889634
        row_delta = tl.load(delta + batch_head * sequence + rows_m, mask=mask_m, other=0.0)
        accumulator = tl.zeros((block_m, head_dim), tl.float32)
        for start_n in tl.range(0, sequence, block_n):
            key_rows = start_n + rows_n
            mask_n = key_rows < sequence
            k = tl.load(key + base + key_rows[:, None] * head_dim + dims[None, :], mask=mask_n[:, None], other=0)
            k_scale = tl.load(key_scale + batch_head * sequence + key_rows, mask=mask_n, other=0.0)
            scores = tl.dot(q, tl.trans(k)).to(tl.float32)
            scores *= q_scale[:, None] * k_scale[None, :] * scale_log2
            probabilities = tl.exp2(scores - lse[:, None])
            probabilities = tl.where(mask_m[:, None] & mask_n[None, :], probabilities, 0.0)
            v = tl.load(value + base + key_rows[:, None] * head_dim + dims[None, :], mask=mask_n[:, None], other=0.0)
            grad_probability = tl.dot(grad_o, tl.trans(v)).to(tl.float32)
            grad_score = probabilities * (grad_probability - row_delta[:, None]) * scale
            k_dequantized = (k.to(tl.float32) * k_scale[:, None]).to(grad_o.dtype)
            accumulator += tl.dot(grad_score.to(grad_o.dtype), k_dequantized)

        tl.store(grad_query + base + rows_m[:, None] * head_dim + dims[None, :], accumulator, mask=mask_m[:, None])

    @triton.jit
    def _int8_attention_dkdv_kernel(
        query,
        key,
        value,
        query_scale,
        key_scale,
        grad_output,
        logsumexp,
        delta,
        grad_key,
        grad_value,
        sequence,
        stride_bh,
        scale_log2: tl.constexpr,
        scale: tl.constexpr,
        head_dim: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        key_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        rows_m = tl.arange(0, block_m)
        rows_n = key_block * block_n + tl.arange(0, block_n)
        dims = tl.arange(0, head_dim)
        mask_n = rows_n < sequence
        base = batch_head * stride_bh

        k = tl.load(key + base + rows_n[:, None] * head_dim + dims[None, :], mask=mask_n[:, None], other=0)
        k_scale = tl.load(key_scale + batch_head * sequence + rows_n, mask=mask_n, other=0.0)
        v = tl.load(value + base + rows_n[:, None] * head_dim + dims[None, :], mask=mask_n[:, None], other=0.0)
        grad_key_accumulator = tl.zeros((block_n, head_dim), tl.float32)
        grad_value_accumulator = tl.zeros((block_n, head_dim), tl.float32)

        for start_m in tl.range(0, sequence, block_m):
            query_rows = start_m + rows_m
            mask_m = query_rows < sequence
            q = tl.load(query + base + query_rows[:, None] * head_dim + dims[None, :], mask=mask_m[:, None], other=0)
            q_scale = tl.load(query_scale + batch_head * sequence + query_rows, mask=mask_m, other=0.0)
            scores = tl.dot(q, tl.trans(k)).to(tl.float32)
            scores *= q_scale[:, None] * k_scale[None, :] * scale_log2
            lse = tl.load(logsumexp + batch_head * sequence + query_rows, mask=mask_m, other=0.0) * 1.4426950408889634
            probabilities = tl.exp2(scores - lse[:, None])
            probabilities = tl.where(mask_m[:, None] & mask_n[None, :], probabilities, 0.0)
            grad_o = tl.load(grad_output + base + query_rows[:, None] * head_dim + dims[None, :], mask=mask_m[:, None], other=0.0)
            grad_probability = tl.dot(grad_o, tl.trans(v)).to(tl.float32)
            row_delta = tl.load(delta + batch_head * sequence + query_rows, mask=mask_m, other=0.0)
            grad_score = probabilities * (grad_probability - row_delta[:, None]) * scale
            q_dequantized = (q.to(tl.float32) * q_scale[:, None]).to(grad_o.dtype)
            grad_key_accumulator += tl.dot(tl.trans(grad_score.to(grad_o.dtype)), q_dequantized)
            grad_value_accumulator += tl.dot(tl.trans(probabilities.to(grad_o.dtype)), grad_o)

        tl.store(grad_key + base + rows_n[:, None] * head_dim + dims[None, :], grad_key_accumulator, mask=mask_n[:, None])
        tl.store(
            grad_value + base + rows_n[:, None] * head_dim + dims[None, :],
            grad_value_accumulator,
            mask=mask_n[:, None],
        )


def _check_inputs(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if not HAS_TRITON:
        raise RuntimeError("H3 INT8 attention requires Triton")
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise RuntimeError("H3 INT8 attention requires CUDA tensors")
    if query.shape != key.shape or query.shape != value.shape or query.ndim != 4:
        raise ValueError("H3 INT8 attention expects equal [batch, heads, sequence, head_dim] tensors")
    if query.shape[-1] != 128:
        raise ValueError(f"H3 INT8 attention supports head_dim=128, got {query.shape[-1]}")
    if query.dtype not in (torch.float16, torch.bfloat16) or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("H3 INT8 attention requires matching FP16 or BF16 query, key, and value tensors")


def _quantize_rows(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.contiguous()
    rows = value.numel() // value.shape[-1]
    quantized = torch.empty_like(value, dtype=torch.int8)
    scales = torch.empty(value.shape[:-1], device=value.device, dtype=torch.float32)
    _quantize_rows_kernel[(rows,)](value, quantized, scales, rows, value.shape[-1], triton.next_power_of_2(value.shape[-1]))
    return quantized, scales


def _int8_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    store_logsumexp: bool = False,
    save_native_state: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
]:
    _check_inputs(query, key, value)
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if save_native_state:
        query_q, query_scale = _quantize_rows(query)
    else:
        query_q, query_scale = None, None
    key_q, key_scale = _quantize_rows(key)
    output = torch.empty_like(value)
    batch, heads, sequence, head_dim = query.shape
    logsumexp = torch.empty((batch, heads, sequence), device=query.device, dtype=torch.float32) if store_logsumexp else None
    block_m = 64
    block_n = 64
    _int8_attention_fwd_kernel[(triton.cdiv(sequence, block_m), batch * heads)](
        query,
        key_q,
        value,
        key_scale,
        output,
        logsumexp if logsumexp is not None else output,
        sequence,
        sequence * head_dim,
        1.0 / math.sqrt(head_dim) * math.log2(math.e),
        head_dim,
        block_m,
        block_n,
        store_logsumexp,
        num_warps=4,
        num_stages=3,
    )
    return output, query_q, key_q, value, query_scale, key_scale, logsumexp


def _flash_backward_supported(query: torch.Tensor) -> bool:
    return (
        query.is_cuda
        and hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention_backward")
        and torch.cuda.get_device_capability(query.device)[0] >= 8
    )


class _Int8Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        ctx.flash_backward = _flash_backward_supported(query)
        output, query_q, key_q, value, query_scale, key_scale, logsumexp = _int8_forward(
            query, key, value, store_logsumexp=True, save_native_state=not ctx.flash_backward
        )
        assert logsumexp is not None
        if ctx.flash_backward:
            ctx.save_for_backward(query, key, value, logsumexp, output)
        else:
            assert query_q is not None and query_scale is not None
            ctx.save_for_backward(query_q, key_q, value, query_scale, key_scale, logsumexp, output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.contiguous()
        if ctx.flash_backward:
            query, key, value, logsumexp, output = ctx.saved_tensors
            _, _, sequence, head_dim = query.shape
            philox_seed = torch.empty((), device=grad_output.device, dtype=torch.int64)
            philox_offset = torch.empty((), device=grad_output.device, dtype=torch.int64)
            return torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
                grad_output,
                query,
                key,
                value,
                output,
                logsumexp,
                None,
                None,
                sequence,
                sequence,
                0.0,
                False,
                philox_seed,
                philox_offset,
                scale=1.0 / math.sqrt(head_dim),
            )
        query_q, key_q, value, query_scale, key_scale, logsumexp, output = ctx.saved_tensors
        batch, heads, sequence, head_dim = query_q.shape
        grad_query = torch.empty_like(grad_output)
        grad_key = torch.empty_like(grad_output)
        grad_value = torch.empty_like(grad_output)
        delta = torch.empty((batch, heads, sequence), device=grad_output.device, dtype=torch.float32)
        rows = batch * heads * sequence
        _attention_backward_delta_kernel[(rows,)](output, grad_output, delta, head_dim, triton.next_power_of_2(head_dim))
        block_m = 64
        block_n = 64
        common = (
            query_q,
            key_q,
            value,
            query_scale,
            key_scale,
            grad_output,
            logsumexp,
            delta,
        )
        launch = (
            sequence,
            sequence * head_dim,
            1.0 / math.sqrt(head_dim) * math.log2(math.e),
            1.0 / math.sqrt(head_dim),
            head_dim,
            block_m,
            block_n,
        )
        _int8_attention_dq_kernel[(triton.cdiv(sequence, block_m), batch * heads)](
            *common, grad_query, *launch, num_warps=4, num_stages=2
        )
        _int8_attention_dkdv_kernel[(triton.cdiv(sequence, block_n), batch * heads)](
            *common, grad_key, grad_value, *launch, num_warps=4, num_stages=2
        )
        return grad_query, grad_key, grad_value


def int8_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Run H3 INT8-QK attention with an optimized training backward."""

    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (query, key, value)):
        return _Int8Attention.apply(query, key, value)
    return _int8_forward(query, key, value)[0]
