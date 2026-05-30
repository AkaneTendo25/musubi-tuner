"""Lion sign-momentum optimizer (Chen et al., arXiv:2302.06675), with optional
8-bit momentum storage and in-place updates for Int8QTWeight parameters.

    update = sign(b1*m + (1-b1)*g)
    p     -= lr * update           # decoupled weight decay: p *= 1 - lr*wd
    m      = b2*m + (1-b2)*g

One momentum buffer, no second moment (half the optimizer state of AdamW).

- Lion          m in fp32.
- Lion8bit      m stored block-wise in fp8 e4m3 (1 byte/param + fp32 per-block
                scales); the update runs in fp32, only the state is 8-bit.
- Lion8bitInt8  Lion8bit that keeps Int8QTWeight parameters in int8 storage:
                reads p.dequantize(), updates in fp32, writes back through
                p.requantize_(stochastic_rounding=True). Plain tensors fall back
                to Lion8bit, so one instance can drive mixed param groups.

All three expose step_param() for --fused_backward_pass.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim.optimizer import Optimizer


_TINY = 1e-12
_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0  # max finite magnitude of e4m3fn


def _quantize_blockwise(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block absmax fp8 e4m3 quantization. Returns ``(q, scale)``.

    ``q`` is ``float8_e4m3fn`` of shape ``(num_blocks, block_size)``; ``scale`` is
    ``(num_blocks,)`` fp32. A zero block yields scale==_TINY and codes 0, so
    dequantization recovers exactly 0. Padding (when numel % block_size != 0) is
    zero-filled inside the last block; the caller passes the original numel/shape to
    :func:`_dequantize_blockwise` to strip it.
    """
    flat = x.reshape(-1).to(torch.float32)
    n = flat.numel()
    pad = (-n) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block_size)
    absmax = blocks.abs().amax(dim=1)
    scale = (absmax / _FP8_MAX).clamp_min(_TINY)
    q = (blocks / scale.unsqueeze(1)).to(_FP8)
    return q, scale


def _dequantize_blockwise(
    q: torch.Tensor,
    scale: torch.Tensor,
    n: int,
    shape: torch.Size,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    blocks = q.to(torch.float32) * scale.unsqueeze(1)
    return blocks.reshape(-1)[:n].view(shape).to(dtype)


def _lion_core(
    w: torch.Tensor,
    g: torch.Tensor,
    m_prev: torch.Tensor,
    lr: float,
    b1: float,
    b2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Lion update step (no weight decay — caller applies it before this).

    Inputs and outputs are all in working precision (fp32 in the 8-bit variants).
    Returns ``(w_new, m_new)``: the post-update weight and the next-step momentum.
    """
    # sign(b1*m + (1-b1)*g); .mul allocates, so m_prev is not mutated
    update_dir = (m_prev.mul(b1).add_(g, alpha=1.0 - b1)).sign_()
    w_new = w.add(update_dir, alpha=-lr)
    m_new = m_prev.mul(b2).add_(g, alpha=1.0 - b2)
    return w_new, m_new


class Lion(Optimizer):
    """Sign-momentum optimizer with full-precision m buffer. Decoupled weight decay."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        b1, b2 = betas
        if not (0.0 < b1 < 1.0 and 0.0 < b2 < 1.0):
            raise ValueError(f"invalid betas {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _step_p(self, p: torch.Tensor, group: dict) -> None:
        """One-parameter step shared by ``step`` and ``step_param``."""
        g = p.grad
        if g.is_sparse:
            raise RuntimeError("Lion does not support sparse gradients")
        lr = group["lr"]
        b1, b2 = group["betas"]
        wd = group["weight_decay"]
        state = self.state[p]
        if len(state) == 0:
            state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        if wd != 0.0:
            p.mul_(1.0 - lr * wd)
        m_prev = state["m"]
        w_new, m_new = _lion_core(p, g, m_prev, lr, b1, b2)
        p.copy_(w_new)
        state["m"] = m_new

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


class Lion8bit(Optimizer):
    """Lion with the momentum buffer stored block-wise in fp8 e4m3.

    1 byte/param for m plus fp32 per-block scales. The update runs in fp32 (m is
    dequantized on entry, re-quantized on exit); only the stored state is 8-bit.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        block_size: int = 256,
    ):
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        b1, b2 = betas
        if not (0.0 < b1 < 1.0 and 0.0 < b2 < 1.0):
            raise ValueError(f"invalid betas {betas}")
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._block_size = block_size

    @torch.no_grad()
    def _step_p(self, p: torch.Tensor, group: dict) -> None:
        g = p.grad
        if g.is_sparse:
            raise RuntimeError("Lion8bit does not support sparse gradients")
        g = g.to(torch.float32)
        bs = self._block_size
        lr = group["lr"]
        b1, b2 = group["betas"]
        wd = group["weight_decay"]
        state = self.state[p]
        if len(state) == 0:
            state["numel"] = p.numel()
            state["shape"] = p.shape
            z = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
            state["m_q"], state["m_scale"] = _quantize_blockwise(z, bs)
        if wd != 0.0:
            p.mul_(1.0 - lr * wd)
        n, shape = state["numel"], state["shape"]
        m_prev = _dequantize_blockwise(state["m_q"], state["m_scale"], n, shape)
        w = p.to(torch.float32)
        w_new, m_new = _lion_core(w, g, m_prev, lr, b1, b2)
        p.copy_(w_new.to(p.dtype))
        state["m_q"], state["m_scale"] = _quantize_blockwise(m_new, bs)

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
        if p.grad is None:
            return
        self._step_p(p, group)
        p.grad = None


class Lion8bitInt8(Lion8bit):
    """Lion8bit that writes back into Int8QTWeight parameters in place.

    When ``p`` exposes ``dequantize``/``requantize_`` (Int8QTWeight), the step reads
    ``p.dequantize()``, computes the Lion update in fp32, and writes back via
    ``p.requantize_(w_new, stochastic_rounding=...)``, keeping the weight in int8.
    Plain-tensor parameters behave like :class:`Lion8bit`, so one instance can drive
    mixed param groups. ``stochastic_rounding`` defaults to True (matches the
    int8_weights training default); False writes deterministic round-to-nearest.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        block_size: int = 256,
        stochastic_rounding: bool = True,
    ):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, block_size=block_size)
        self._stochastic_rounding = bool(stochastic_rounding)

    @torch.no_grad()
    def _step_p(self, p: torch.Tensor, group: dict) -> None:
        g = p.grad
        if g.is_sparse:
            raise RuntimeError("Lion8bitInt8 does not support sparse gradients")
        g = g.to(torch.float32)
        bs = self._block_size
        sr = self._stochastic_rounding
        lr = group["lr"]
        b1, b2 = group["betas"]
        wd = group["weight_decay"]
        state = self.state[p]

        is_quant = hasattr(p, "dequantize") and hasattr(p, "requantize_")
        w = p.dequantize().to(torch.float32) if is_quant else p.to(torch.float32)

        if len(state) == 0:
            state["numel"] = w.numel()
            state["shape"] = w.shape
            z = torch.zeros(w.shape, dtype=torch.float32, device=w.device)
            state["m_q"], state["m_scale"] = _quantize_blockwise(z, bs)

        if wd != 0.0:
            if is_quant:
                w = w.mul(1.0 - lr * wd)
            else:
                p.mul_(1.0 - lr * wd)
                w = p.to(torch.float32)

        n, shape = state["numel"], state["shape"]
        m_prev = _dequantize_blockwise(state["m_q"], state["m_scale"], n, shape)
        w_new, m_new = _lion_core(w, g, m_prev, lr, b1, b2)

        if is_quant:
            # keep int8 storage; SR absorbs the quantization residual
            p.requantize_(w_new, stochastic_rounding=sr)
        else:
            p.copy_(w_new.to(p.dtype))
        state["m_q"], state["m_scale"] = _quantize_blockwise(m_new, bs)

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
        if p.grad is None:
            return
        self._step_p(p, group)
        p.grad = None
