"""SMMF — Square-Matricized Momentum Factorization (Park et al., 2024).

Reference: https://github.com/eai-lab/SMMF (paper: "SMMF: Square-Matricized
Momentum Factorization for Memory-Efficient Optimization").

A standalone, memory-efficient optimizer in the Adafactor/CAME family. It keeps
Adam-style first- and second-moment statistics but stores each as a rank-1
non-negative factorization of a near-square reshape of the parameter, plus a
1-bit sign mask for the first moment. Persistent optimizer state is therefore
O(rows + cols) + numel/8 bytes instead of AdamW's 2*numel*4 bytes.

    effective_shape = most-square (rows, cols) with rows*cols == numel
    m, v stored as (row, col) rank-1 factors; m also stores a packed sign mask
    each step: decompress -> EMA -> recompress (rank-1, like Adafactor's v)
    update = m / (sqrt(v) + eps)            # no Adam bias correction

The moment EMAs use the paper's Adafactor-style time-varying schedule (not
constant Adam betas):

    beta_m = beta * growth_rate ** (step - 1)
    beta_v = 1 - step ** decay_rate

``decay_rate`` defaults to -0.8 (the official repo's recommendation for
Transformer models; the paper's class default is -0.5).

The sign mask is bit-packed to uint8 (1 bit/param) rather than the reference's
bool tensor (1 byte/param), an 8x reduction in that buffer.

``step_param`` is provided for ``--fused_backward_pass``. Each step reconstructs
the full dense moment matrices from their factors, so per-step working memory
scales with the parameter size even though the persistent state stays small.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch.optim.optimizer import Optimizer


def _effective_shape(numel: int) -> tuple[int, int]:
    """Most-square ``(rows, cols)`` with ``rows * cols == numel`` (square-matricization)."""
    root = int(numel**0.5)
    if root * root == numel:
        return root, root
    for width in range(root, 0, -1):
        if numel % width == 0:
            return numel // width, width
    return numel, 1


def _factor_dtype(grad: torch.Tensor) -> torch.dtype:
    return torch.float64 if grad.dtype == torch.float64 else torch.float32


def _pack_sign(mask: torch.Tensor) -> torch.Tensor:
    """Pack a boolean mask into uint8, 8 bits per byte."""
    flat = mask.reshape(-1).to(torch.uint8)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat((flat, flat.new_zeros(pad)))
    flat = flat.view(-1, 8)
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=flat.device, dtype=torch.uint8)
    return (flat * weights).sum(dim=1).to(torch.uint8)


def _unpack_sign(packed: torch.Tensor, numel: int, shape: tuple[int, int]) -> torch.Tensor:
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=packed.device, dtype=torch.uint8)
    bits = (packed.view(-1, 1).bitwise_and(weights) != 0).reshape(-1)
    return bits[:numel].view(shape)


def _decompress(state: dict[str, Any], prefix: str) -> torch.Tensor:
    """Reconstruct the dense moment matrix from its rank-1 factors (+ sign for m)."""
    update = torch.outer(state[f"{prefix}_row"], state[f"{prefix}_col"])
    if prefix == "m":
        shape = tuple(state["effective_shape"])
        sign = _unpack_sign(state["m_sign"], int(state["m_sign_numel"]), shape)
        update = torch.where(sign, update, -update)
    return update


def _compress(matrix: torch.Tensor, state: dict[str, Any], prefix: str) -> None:
    """Rank-1 NNMF (Adafactor-style row/col sums) of a dense moment into its factors.

    For the first moment the sign is stored separately and ``|m|`` is factorized;
    for the second moment the matrix is already non-negative.
    """
    if prefix == "m":
        state["m_sign"] = _pack_sign(matrix > 0)
        matrix = matrix.abs()
    else:
        matrix = matrix.clamp_min(0)

    row = state[f"{prefix}_row"]
    col = state[f"{prefix}_col"]
    torch.sum(matrix, dim=1, out=row)
    torch.sum(matrix, dim=0, out=col)

    # Normalize one factor by the total sum so outer(row, col) reproduces the
    # matrix (rank-1 reconstruction that preserves row/col marginals).
    if matrix.shape[0] < matrix.shape[1]:
        row.div_(row.sum().clamp_min(torch.finfo(row.dtype).tiny))
    else:
        col.div_(col.sum().clamp_min(torch.finfo(col.dtype).tiny))


class SMMF(Optimizer):
    """Square-Matricized Momentum Factorization optimizer.

    Args:
        params: iterable of parameters or parameter-group dicts.
        lr: learning rate.
        beta: first-moment coefficient (the ``beta_m`` base, default 0.9).
        eps: denominator epsilon.
        weight_decay: weight decay coefficient.
        decay_rate: second-moment schedule exponent in ``beta_v = 1 - step**decay_rate``
            (default -0.8, the official Transformer recommendation; paper default -0.5).
        growth_rate: first-moment schedule factor in ``beta_m = beta * growth_rate**(step-1)``.
        vector_reshape: if False, 1-D parameters use plain (unfactored) Adam state
            instead of being reshaped to a square matrix.
        weight_decay_mode: ``'adamw'`` (decoupled, default) or ``'adam'`` (coupled into grad).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        beta: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decay_rate: float = -0.8,
        growth_rate: float = 0.999,
        vector_reshape: bool = True,
        weight_decay_mode: str = "adamw",
    ):
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if not (-1.0 <= decay_rate <= 0.0):
            raise ValueError(f"decay_rate must be in [-1, 0], got {decay_rate}")
        if not (0.0 <= growth_rate <= 1.0):
            raise ValueError(f"growth_rate must be in [0, 1], got {growth_rate}")
        if weight_decay_mode not in ("adamw", "adam"):
            raise ValueError(f"weight_decay_mode must be 'adamw' or 'adam', got {weight_decay_mode!r}")
        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            weight_decay=weight_decay,
            decay_rate=decay_rate,
            growth_rate=growth_rate,
            vector_reshape=vector_reshape,
        )
        super().__init__(params, defaults)
        self.weight_decay_mode = weight_decay_mode

    def _betas(self, group: dict, step: int) -> tuple[float, float]:
        beta_m = float(group["beta"]) * float(group["growth_rate"]) ** (float(step) - 1.0)
        beta_v = 1.0 - float(step) ** float(group["decay_rate"])
        return beta_m, beta_v

    @torch.no_grad()
    def _factorized_update(self, state: dict[str, Any], group: dict, grad: torch.Tensor) -> torch.Tensor:
        original_shape = grad.shape
        if len(state) == 0:
            effective_shape = _effective_shape(int(grad.numel()))
            dtype = _factor_dtype(grad)
            rows, cols = effective_shape
            state["step"] = 0
            state["effective_shape"] = effective_shape
            state["m_row"] = torch.zeros(rows, device=grad.device, dtype=dtype)
            state["m_col"] = torch.zeros(cols, device=grad.device, dtype=dtype)
            state["m_sign"] = torch.zeros((int(grad.numel()) + 7) // 8, device=grad.device, dtype=torch.uint8)
            state["m_sign_numel"] = int(grad.numel())
            state["v_row"] = torch.zeros(rows, device=grad.device, dtype=dtype)
            state["v_col"] = torch.zeros(cols, device=grad.device, dtype=dtype)

        effective_shape = tuple(state["effective_shape"])
        grad_matrix = grad.contiguous().view(effective_shape).to(dtype=state["m_row"].dtype)

        update_m = _decompress(state, "m")
        update_v = _decompress(state, "v")

        state["step"] = int(state["step"]) + 1
        beta_m, beta_v = self._betas(group, int(state["step"]))
        update_m.mul_(beta_m).add_(grad_matrix, alpha=1.0 - beta_m)
        update_v.mul_(beta_v).addcmul_(grad_matrix, grad_matrix, value=1.0 - beta_v)

        _compress(update_m, state, "m")
        _compress(update_v, state, "v")

        update = update_m / update_v.sqrt().add_(float(group["eps"]))
        return update.contiguous().view(original_shape)

    @torch.no_grad()
    def _adam_update(self, state: dict[str, Any], group: dict, grad: torch.Tensor) -> torch.Tensor:
        if len(state) == 0:
            state["step"] = 0
            state["m"] = torch.zeros_like(grad, dtype=_factor_dtype(grad))
            state["v"] = torch.zeros_like(grad, dtype=_factor_dtype(grad))
        g = grad.to(dtype=state["m"].dtype)
        state["step"] = int(state["step"]) + 1
        beta_m, beta_v = self._betas(group, int(state["step"]))
        state["m"].mul_(beta_m).add_(g, alpha=1.0 - beta_m)
        state["v"].mul_(beta_v).addcmul_(g, g, value=1.0 - beta_v)
        update = state["m"] / state["v"].sqrt().add_(float(group["eps"]))
        return update.view(grad.shape)

    @torch.no_grad()
    def _step_p(self, p: torch.Tensor, group: dict) -> None:
        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("SMMF does not support sparse gradients")
        lr = float(group["lr"])
        wd = float(group["weight_decay"])

        if wd != 0.0:
            if self.weight_decay_mode == "adam":
                grad = grad.add(p, alpha=wd)
            else:  # decoupled (adamw)
                p.mul_(1.0 - lr * wd)

        state = self.state[p]
        dimension = len(grad.squeeze().shape)
        factorize = not (dimension <= 1 and not bool(group["vector_reshape"]))
        if factorize:
            update = self._factorized_update(state, group, grad)
        else:
            update = self._adam_update(state, group, grad)
        p.add_(update.to(dtype=p.dtype), alpha=-lr)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self._step_p(p, group)
        return loss

    @torch.no_grad()
    def step_param(self, p: torch.Tensor, group: dict) -> None:
        """Per-param fused-backward entry point (see ``--fused_backward_pass``)."""
        if p.grad is None:
            return
        self._step_p(p, group)
        p.grad = None
