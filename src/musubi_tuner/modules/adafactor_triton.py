"""Triton fast path for fused-backward Adafactor on dense BF16 matrices."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from transformers import Adafactor

from musubi_tuner.modules.adafactor_fused import adafactor_step_param


@triton.jit
def _row_col_sum_sq(
    grad_ptr,
    row_ptr,
    col_ptr,
    rows,
    cols,
    stride_row,
    stride_col,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    row_offsets = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    row_mask = row_offsets < rows
    col_mask = col_offsets < cols
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = row_offsets[:, None] * stride_row + col_offsets[None, :] * stride_col
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_sq = grad * grad
    tl.atomic_add(row_ptr + row_offsets, tl.sum(grad_sq, axis=1), mask=row_mask)
    tl.atomic_add(col_ptr + col_offsets, tl.sum(grad_sq, axis=0), mask=col_mask)


@triton.jit
def _update_sum_sq(
    grad_ptr,
    row_factor_ptr,
    col_factor_ptr,
    update_sq_ptr,
    rows,
    cols,
    stride_row,
    stride_col,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    row_offsets = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    row_mask = row_offsets < rows
    col_mask = col_offsets < cols
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = row_offsets[:, None] * stride_row + col_offsets[None, :] * stride_col
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_factor = tl.load(row_factor_ptr + row_offsets, mask=row_mask, other=0.0)
    col_factor = tl.load(col_factor_ptr + col_offsets, mask=col_mask, other=0.0)
    update = grad * row_factor[:, None] * col_factor[None, :]
    tl.atomic_add(update_sq_ptr, tl.sum(tl.where(mask, update * update, 0.0)))


@triton.jit
def _apply_update(
    grad_ptr,
    param_ptr,
    row_factor_ptr,
    col_factor_ptr,
    clip_ptr,
    learning_rate,
    weight_decay_scale,
    seed,
    rows,
    cols,
    stride_row,
    stride_col,
    STOCHASTIC: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    row_offsets = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    row_mask = row_offsets < rows
    col_mask = col_offsets < cols
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = row_offsets[:, None] * stride_row + col_offsets[None, :] * stride_col
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_factor = tl.load(row_factor_ptr + row_offsets, mask=row_mask, other=0.0)
    col_factor = tl.load(col_factor_ptr + col_offsets, mask=col_mask, other=0.0)
    clip = tl.load(clip_ptr)
    update = grad * row_factor[:, None] * col_factor[None, :] / clip
    result = param - update * learning_rate - param * weight_decay_scale
    if STOCHASTIC:
        bits = result.to(tl.int32, bitcast=True)
        noise = (tl.randint(seed, offsets) & 0xFFFF).to(tl.int32)
        result = ((bits + noise) & -65536).to(tl.float32, bitcast=True)
    tl.store(param_ptr + offsets, result.to(param_ptr.dtype.element_ty), mask=mask)


@torch.no_grad()
def _triton_matrix_step(
    param: Tensor,
    grad: Tensor,
    row_state: Tensor,
    col_state: Tensor,
    *,
    beta2: float,
    eps: float,
    clip_threshold: float,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> None:
    rows, cols = param.shape
    stride_row, stride_col = param.stride()
    block_rows, block_cols = 32, 128
    grid = (triton.cdiv(rows, block_rows), triton.cdiv(cols, block_cols))
    row_sum = torch.zeros_like(row_state)
    col_sum = torch.zeros_like(col_state)
    _row_col_sum_sq[grid](
        grad,
        row_sum,
        col_sum,
        rows,
        cols,
        stride_row,
        stride_col,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
    )
    row_state.mul_(beta2).add_(row_sum / cols + eps, alpha=1.0 - beta2)
    col_state.mul_(beta2).add_(col_sum / rows + eps, alpha=1.0 - beta2)
    row_factor = (row_state / row_state.mean()).rsqrt()
    col_factor = col_state.rsqrt()
    update_sq = torch.zeros((), device=param.device, dtype=torch.float32)
    _update_sum_sq[grid](
        grad,
        row_factor,
        col_factor,
        update_sq,
        rows,
        cols,
        stride_row,
        stride_col,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
    )
    clip = torch.clamp((update_sq / param.numel()).sqrt() / clip_threshold, min=1.0)
    _apply_update[grid](
        grad,
        param,
        row_factor,
        col_factor,
        clip,
        learning_rate,
        learning_rate * weight_decay,
        seed,
        rows,
        cols,
        stride_row,
        stride_col,
        STOCHASTIC=param.dtype == torch.bfloat16,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
    )


def _supports_triton_step(param: Tensor, group: dict) -> bool:
    if param.grad is None or param.grad.is_sparse:
        return False
    factored, use_first_moment = Adafactor._get_options(group, param.grad.shape)
    return bool(
        factored
        and not use_first_moment
        and param.ndim == 2
        and param.dtype == torch.bfloat16
        and param.is_cuda
        and param.is_contiguous()
        and param.grad.is_contiguous()
        and param.grad.stride() == param.stride()
        and not group.get("scale_parameter", True)
        and not group.get("relative_step", True)
        and not hasattr(param, "int_data")
    )


@torch.no_grad()
def adafactor_step_param_triton(self: Adafactor, param: Tensor, group: dict) -> None:
    if not _supports_triton_step(param, group):
        adafactor_step_param(self, param, group)
        return
    state = self.state[param]
    if len(state) == 0:
        state["step"] = 0
        state["exp_avg_sq_row"] = torch.zeros(param.shape[0], device=param.device, dtype=torch.float32)
        state["exp_avg_sq_col"] = torch.zeros(param.shape[1], device=param.device, dtype=torch.float32)
        state["RMS"] = 0
    if "triton_sr_seed" not in state:
        parameter_index = self._adafactor_triton_parameter_indices[id(param)]
        state["triton_sr_seed"] = ((parameter_index + 1) * 2654435761) & 0x7FFFFFFF
    state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(device=param.device, dtype=torch.float32)
    state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(device=param.device, dtype=torch.float32)
    state["step"] += 1
    state["RMS"] = 0
    state["triton_sr_seed"] = (int(state["triton_sr_seed"]) + 1) & 0x7FFFFFFF
    _triton_matrix_step(
        param,
        param.grad,
        state["exp_avg_sq_row"],
        state["exp_avg_sq_col"],
        beta2=1.0 - state["step"] ** group["decay_rate"],
        eps=float(group["eps"][0]),
        clip_threshold=float(group["clip_threshold"]),
        learning_rate=float(group["lr"]),
        weight_decay=float(group.get("weight_decay") or 0.0),
        seed=state["triton_sr_seed"],
    )


@torch.no_grad()
def adafactor_step_triton(self: Adafactor, closure=None):
    loss = closure() if closure is not None else None
    for group in self.param_groups:
        for param in group["params"]:
            adafactor_step_param_triton(self, param, group)
    return loss


def patch_adafactor_triton(optimizer: Adafactor) -> None:
    parameter_indices = {}
    for group in optimizer.param_groups:
        for param in group["params"]:
            parameter_indices[id(param)] = len(parameter_indices)
    optimizer._adafactor_triton_parameter_indices = parameter_indices
    optimizer.step_param = adafactor_step_param_triton.__get__(optimizer)
    optimizer.step = adafactor_step_triton.__get__(optimizer)
