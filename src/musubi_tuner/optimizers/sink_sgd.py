"""SinkSGD optimizer.

This is an opt-in optimizer for LoRA-style training. It combines SGD momentum
with Sinkhorn-style update normalization and optional spectral scaling for
matrix parameters.
"""

from __future__ import annotations

import math
from typing import Any

import torch


_VALID_STATE_PRECISIONS = {"auto", "fp32", "bf16_sr"}


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _copy_stochastic_bf16_(target: torch.Tensor, source: torch.Tensor) -> None:
    if target.dtype != torch.bfloat16:
        target.copy_(source)
        return
    random_bits = torch.randint_like(source.float(), dtype=torch.int32, low=0, high=1 << 16)
    rounded = source.float().view(torch.int32).add_(random_bits).bitwise_and_(-65536).view(torch.float32)
    target.copy_(rounded)


def _add_stochastic_(target: torch.Tensor, update: torch.Tensor, alpha: float) -> None:
    if target.dtype == torch.bfloat16:
        _copy_stochastic_bf16_(target, target.float().add(update.float(), alpha=alpha))
    else:
        target.add_(update.to(target.dtype), alpha=alpha)


def _project_orthogonal_to_param(update: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    original_shape = update.shape
    update_flat = update.reshape(-1).float()
    param_flat = param.detach().reshape(-1).float()
    denom = torch.dot(param_flat, param_flat).clamp_min(1e-30)
    projected = update_flat - param_flat * (torch.dot(param_flat, update_flat) / denom)
    old_norm = update_flat.norm().clamp_min(1e-12)
    new_norm = projected.norm().clamp_min(1e-12)
    return projected.mul_(old_norm / new_norm).view(original_shape).to(update.dtype)


def _sinkhorn_normalize(update: torch.Tensor, param: torch.Tensor, iterations: int, orthogonal: bool) -> torch.Tensor:
    original_shape = update.shape
    original_dtype = update.dtype
    update = update.float()

    if update.ndim == 1:
        if orthogonal:
            update = _project_orthogonal_to_param(update, param).float()
        norm = update.norm(p=2).clamp_min(1e-12)
        return update.mul_(math.sqrt(update.numel()) / norm).view(original_shape).to(original_dtype)

    update_2d = update.view(update.shape[0], -1)
    param_2d = param.detach().float().view(param.shape[0], -1)
    rows, cols = update_2d.shape
    row_target = math.sqrt(cols)
    col_target = math.sqrt(rows)

    for _ in range(iterations):
        row_norm = update_2d.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        update_2d = update_2d * (row_target / row_norm)
        if orthogonal:
            denom = (param_2d * param_2d).sum(dim=1, keepdim=True).clamp_min(1e-30)
            dot = (param_2d * update_2d).sum(dim=1, keepdim=True)
            update_2d = update_2d - param_2d * (dot / denom)

        col_norm = update_2d.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
        update_2d = update_2d * (col_target / col_norm)
        if orthogonal:
            denom = (param_2d * param_2d).sum(dim=0, keepdim=True).clamp_min(1e-30)
            dot = (param_2d * update_2d).sum(dim=0, keepdim=True)
            update_2d = update_2d - param_2d * (dot / denom)

    return update_2d.view(original_shape).to(original_dtype)


def _init_spectral_state(state: dict[str, Any], param: torch.Tensor) -> None:
    rows = param.shape[0]
    cols = param.numel() // rows
    device = param.device
    dtype = torch.float32
    state["spectral_u"] = torch.randn(rows, device=device, dtype=dtype)
    state["spectral_v"] = torch.randn(cols, device=device, dtype=dtype)
    state["spectral_u"].div_(state["spectral_u"].norm().clamp_min(1e-12))
    state["spectral_v"].div_(state["spectral_v"].norm().clamp_min(1e-12))


def _spectral_scale(update: torch.Tensor, state: dict[str, Any], lr: float) -> torch.Tensor:
    rows = update.shape[0]
    cols = update.numel() // rows
    matrix = update.float().view(rows, cols)
    u = state["spectral_u"]
    v = state["spectral_v"]

    v_next = matrix.t().mv(u)
    v_norm = v_next.norm()
    if v_norm > 1e-6:
        v.copy_(v_next / v_norm.clamp_min(1e-12))

    u_next = matrix.mv(v)
    u_norm = u_next.norm()
    if u_norm > 1e-6:
        u.copy_(u_next / u_norm.clamp_min(1e-12))

    sigma = torch.dot(u, matrix.mv(v)).abs().clamp_min(1.0 / (math.sqrt(rows) + math.sqrt(cols)))
    target = math.sqrt(rows / cols)
    return update.mul(float(lr) * target / sigma)


class SinkSGD(torch.optim.Optimizer):
    """SGD with Sinkhorn-normalized updates and optional spectral scaling.

    Supported aliases in the trainer are ``SinkSGD``, ``SinkSGD_adv``,
    ``sinksgd``, ``sink_sgd``, and ``sinksgdadv``.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.995,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        nesterov_coef: float = 0.8,
        normed_momentum: bool = True,
        sinkhorn_iterations: int = 3,
        orthogonal_sinkhorn: bool = True,
        orthogonal_gradient: bool = False,
        spectral_normalization: bool = False,
        state_precision: str = "auto",
        **kwargs,
    ) -> None:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"SinkSGD got unsupported optimizer args: {unknown}")
        if lr <= 0:
            raise ValueError("SinkSGD requires lr > 0")
        if not 0 <= momentum < 1:
            raise ValueError("SinkSGD momentum must be in [0, 1)")
        if not 0 <= nesterov_coef <= 1:
            raise ValueError("SinkSGD nesterov_coef must be in [0, 1]")
        if sinkhorn_iterations < 1:
            raise ValueError("SinkSGD sinkhorn_iterations must be >= 1")
        state_precision = str(state_precision).lower()
        if state_precision not in _VALID_STATE_PRECISIONS:
            raise ValueError(f"SinkSGD state_precision must be one of {sorted(_VALID_STATE_PRECISIONS)}")

        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
            "nesterov": _as_bool(nesterov),
            "nesterov_coef": float(nesterov_coef),
            "normed_momentum": _as_bool(normed_momentum),
            "sinkhorn_iterations": int(sinkhorn_iterations),
            "orthogonal_sinkhorn": _as_bool(orthogonal_sinkhorn),
            "orthogonal_gradient": _as_bool(orthogonal_gradient),
            "spectral_normalization": _as_bool(spectral_normalization),
            "state_precision": state_precision,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _actual_state_precision(param: torch.Tensor, group: dict[str, Any]) -> str:
        requested = group["state_precision"]
        if requested == "auto":
            return "bf16_sr" if param.dtype == torch.bfloat16 else "fp32"
        return requested

    def _init_state(self, param: torch.Tensor, group: dict[str, Any]) -> dict[str, Any]:
        state = self.state[param]
        if state:
            return state

        precision = self._actual_state_precision(param, group)
        state["step"] = 0
        state["actual_state_precision"] = precision
        buffer_dtype = torch.bfloat16 if precision == "bf16_sr" else torch.float32
        state["momentum_buffer"] = torch.zeros_like(param, dtype=buffer_dtype, memory_format=torch.preserve_format)
        if group["spectral_normalization"] and param.ndim >= 2:
            _init_spectral_state(state, param)
        return state

    @staticmethod
    def _read_momentum(state: dict[str, Any]) -> torch.Tensor:
        return state["momentum_buffer"].float()

    @staticmethod
    def _write_momentum(state: dict[str, Any], value: torch.Tensor) -> None:
        buffer = state["momentum_buffer"]
        if state["actual_state_precision"] == "bf16_sr":
            _copy_stochastic_bf16_(buffer, value)
        else:
            buffer.copy_(value)

    def step_param(self, param: torch.Tensor, group: dict[str, Any]) -> None:
        self.step_parameter(param, group)

    @torch.no_grad()
    def step_parameter(self, param: torch.Tensor, group: dict[str, Any], i: int | None = None) -> None:
        if param.grad is None:
            return
        if param.grad.is_sparse:
            raise RuntimeError("SinkSGD does not support sparse gradients")

        state = self._init_state(param, group)
        state["step"] += 1

        grad = param.grad.detach().float()
        param_float = param.detach().float()
        if group["weight_decay"] != 0:
            grad = grad.add(param_float, alpha=group["weight_decay"])
        if group["orthogonal_gradient"]:
            grad = _project_orthogonal_to_param(grad, param).float()

        momentum = group["momentum"]
        if momentum > 0:
            momentum_input = grad
            if group["normed_momentum"]:
                momentum_input = _sinkhorn_normalize(
                    momentum_input,
                    param,
                    group["sinkhorn_iterations"],
                    group["orthogonal_sinkhorn"],
                ).float()
            buffer = self._read_momentum(state).mul_(momentum).add_(momentum_input)
            self._write_momentum(state, buffer)
            if group["nesterov"]:
                update = grad.add(buffer, alpha=momentum * group["nesterov_coef"])
            else:
                update = buffer
        else:
            update = grad

        update = _sinkhorn_normalize(
            update,
            param,
            group["sinkhorn_iterations"],
            group["orthogonal_sinkhorn"],
        )
        if group["spectral_normalization"] and param.ndim >= 2:
            update = _spectral_scale(update, state, group["lr"])
            _add_stochastic_(param, update, alpha=-1.0)
        else:
            _add_stochastic_(param, update, alpha=-group["lr"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                self.step_parameter(param, group)
        return loss
