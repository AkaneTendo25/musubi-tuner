"""Eager CUDA fusion for the H3 attention residual and following AdaLN RMSNorm."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _residual_forward(
        hidden,
        branch,
        gate,
        indices,
        output,
        gate_stride,
        hidden_size: tl.constexpr,
        block: tl.constexpr,
        shared_indices: tl.constexpr,
        index_sequence: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block)
        valid = offsets < hidden_size
        base = row * hidden_size + offsets
        index_offset = row % index_sequence if shared_indices else row
        table_row = tl.load(indices + index_offset)
        hidden_value = tl.load(hidden + base, mask=valid, other=0.0).to(tl.float32)
        branch_value = tl.load(branch + base, mask=valid, other=0.0).to(tl.float32)
        gate_value = tl.load(gate + table_row * gate_stride + offsets, mask=valid, other=0.0).to(tl.float32)
        tl.store(output + base, (hidden_value + gate_value * branch_value).to(tl.bfloat16), mask=valid)


def _native_rms_ops():
    forward = getattr(torch.ops.aten, "_fused_rms_norm", None)
    backward = getattr(torch.ops.aten, "_fused_rms_norm_backward", None)
    if forward is None or backward is None:
        return None
    return forward.default, backward.default


def _flat_indices(indices: torch.Tensor, rows: int) -> torch.Tensor:
    if indices.numel() == rows:
        return indices.reshape(-1)
    return indices.repeat(rows // indices.numel())


def _index_backward(table: torch.Tensor, indices: torch.Tensor, selected_grad: torch.Tensor) -> torch.Tensor:
    if indices.ndim == 1 and selected_grad.shape[0] != indices.numel():
        selected_grad = selected_grad.reshape(-1, indices.numel(), selected_grad.shape[-1]).sum(dim=0)
    return torch.ops.aten._index_put_impl_.default(torch.zeros_like(table), [indices.reshape(-1)], selected_grad, True, True)


class _ResidualNormAdaLNNative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, branch, gate, weight, shift, scale, indices, eps):
        rows, width = hidden.shape
        native_forward, _ = _native_rms_ops()
        residual = torch.empty_like(hidden)
        _residual_forward[(rows,)](
            hidden,
            branch,
            gate,
            indices,
            residual,
            gate.stride(0),
            hidden_size=width,
            block=triton.next_power_of_2(width),
            num_warps=8,
            shared_indices=indices.numel() != rows,
            index_sequence=indices.numel(),
        )
        normalized, rstd = native_forward(residual, [width], weight, float(eps))
        flat_indices = _flat_indices(indices, rows)
        modulated = torch.addcmul(
            shift.index_select(0, flat_indices),
            normalized,
            1.0 + scale.index_select(0, flat_indices),
        )

        needs = tuple(bool(value) for value in ctx.needs_input_grad[:6])
        need_hidden, need_branch, need_gate, need_weight, need_shift, need_scale = needs
        need_residual_vjp = need_hidden or need_branch or need_gate
        ctx.save_for_backward(
            residual,
            normalized if need_scale else None,
            rstd if (need_residual_vjp or need_weight) else None,
            branch if need_gate else None,
            gate if (need_branch or need_gate) else None,
            weight if (need_residual_vjp or need_weight) else None,
            scale if (need_residual_vjp or need_weight or need_shift or need_scale) else None,
            indices,
        )
        ctx.width = width
        ctx.needs = needs
        return residual, modulated

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_residual, grad_modulated):
        residual, normalized, rstd, branch, gate, weight, scale, indices = ctx.saved_tensors
        _, native_backward = _native_rms_ops()
        need_hidden, need_branch, need_gate, need_weight, need_shift, need_scale = ctx.needs
        need_residual_vjp = need_hidden or need_branch or need_gate
        rows = residual.shape[0]
        flat_indices = _flat_indices(indices, rows)
        if grad_residual is None:
            grad_residual = torch.zeros_like(residual)
        if grad_modulated is None:
            grad_modulated = torch.zeros_like(residual)
        grad_modulated = grad_modulated.contiguous()

        grad_r = grad_weight = None
        if need_residual_vjp or need_weight:
            selected_scale = scale.index_select(0, flat_indices)
            grad_normalized = grad_modulated * (1.0 + selected_scale)
            grad_from_norm, grad_weight = native_backward(
                grad_normalized,
                residual,
                [ctx.width],
                rstd,
                weight,
                [need_residual_vjp, need_weight],
            )
            if need_residual_vjp:
                grad_r = grad_residual.contiguous() + grad_from_norm

        grad_hidden = grad_r if need_hidden else None
        grad_branch = grad_r * gate.index_select(0, flat_indices) if need_branch else None
        grad_gate = _index_backward(gate, indices, grad_r * branch) if need_gate else None
        # Shift values are not needed in backward; the scale table has the same shape,
        # dtype and device and supplies the correctly shaped zero destination.
        grad_shift = _index_backward(scale, indices, grad_modulated) if need_shift else None
        grad_scale = _index_backward(scale, indices, grad_modulated * normalized) if need_scale else None
        return grad_hidden, grad_branch, grad_gate, grad_weight, grad_shift, grad_scale, None, None


def _eligible(hidden, branch, gate, weight, shift, scale, indices) -> bool:
    if not HAS_TRITON or torch.compiler.is_compiling() or _native_rms_ops() is None:
        return False
    if hidden.device.type != "cuda" or hidden.ndim != 3 or hidden.shape != branch.shape:
        return False
    batch, sequence, width = hidden.shape
    if gate.ndim != 2 or gate.shape != shift.shape or gate.shape != scale.shape or gate.shape[1] != width:
        return False
    if weight.shape != (width,):
        return False
    floating = (hidden, branch, gate, weight, shift, scale)
    if any(value.dtype is not torch.bfloat16 or value.device != hidden.device for value in floating):
        return False
    if not hidden.is_contiguous() or not branch.is_contiguous() or not weight.is_contiguous():
        return False
    if any(value.stride(1) != 1 for value in (gate, shift, scale)):
        return False
    if indices.device != hidden.device or indices.dtype not in (torch.int32, torch.int64) or not indices.is_contiguous():
        return False
    shared = indices.ndim == 1 and indices.shape[0] == sequence
    batched = indices.ndim == 2 and tuple(indices.shape) == (batch, sequence)
    return shared or batched


def try_residual_norm_adaln_gated(hidden, branch, gate, weight, shift, scale, indices, eps):
    """Return the fused boundary when eligible, otherwise let the caller use eager PyTorch."""
    if not _eligible(hidden, branch, gate, weight, shift, scale, indices):
        return None
    batch, sequence, width = hidden.shape
    residual, modulated = _ResidualNormAdaLNNative.apply(
        hidden.reshape(batch * sequence, width),
        branch.reshape(batch * sequence, width),
        gate,
        weight,
        shift,
        scale,
        indices.reshape(-1) if indices.ndim == 2 else indices,
        eps,
    )
    return residual.reshape_as(hidden), modulated.reshape_as(hidden)
