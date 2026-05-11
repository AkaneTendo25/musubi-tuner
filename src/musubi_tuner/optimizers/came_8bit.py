"""CAME optimizers.

Implements:
  - "CAME: Confidence-guided Adaptive Memory Efficient Optimization"
    (Luo et al., 2023, https://arxiv.org/abs/2307.02047)
  - Memory-efficient block-wise 8-bit quantization of optimizer state, per
    "SANA 1.5" (https://arxiv.org/pdf/2501.18427v3), section 2.3
    "Block-wise Quantization Strategy".

Features:
  * step_parameter / step_param — per-parameter step entry point used by
    fused-backward-pass training loops that require the optimizer to expose
    a single-param step.
  * stochastic_rounding — uses local stochastic-rounding ``add`` for bf16
    targets, eliminating "lost updates" from precision floor when
    LR * gradient is below ~1e-3.
  * use_cautious — Cautious Optimizers masking
    (https://arxiv.org/abs/2411.16085): drops update components whose sign
    disagrees with the gradient sign, scaled by the kept fraction.
"""

from __future__ import annotations

import gc
import sys
from typing import Any

import torch
import torch.optim


def _copy_stochastic_(target: torch.Tensor, source: torch.Tensor) -> None:
    """Copy ``source`` (fp32) into ``target`` (bf16) using stochastic rounding."""
    result = torch.randint_like(
        source,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
    )
    result.add_(source.view(dtype=torch.int32))
    result.bitwise_and_(-65536)
    target.copy_(result.view(dtype=torch.float32))


def _add_stochastic_scaled_(target: torch.Tensor, source: torch.Tensor, alpha: float = 1.0) -> None:
    """Stochastic ``target += source * alpha`` for bf16 targets."""
    target32 = target.float()
    source32 = source.float()
    if alpha == 1.0:
        result32 = target32 + source32
    else:
        result32 = target32 + source32 * alpha
    _copy_stochastic_(target, result32)


class CAME8bit(torch.optim.Optimizer):
    """CAME with block-wise 8-bit quantization of the first moment and
    (non-factored) second moment.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr: external learning rate (required, > 0).
        eps: regularization constants (square gradient, instability), default
            ``(1e-30, 1e-16)``.
        clip_threshold: RMS clip threshold for the update, default ``1.0``.
        betas: ``(beta1, beta2, beta3)`` for first moment, second moment, and
            instability moment respectively. Default ``(0.9, 0.999, 0.9999)``.
        weight_decay: L2 weight decay coefficient. Default ``0.0``.
        stochastic_rounding: when ``True`` and a parameter is bf16, weight
            updates and weight decay use stochastic rounding instead of plain
            ``data.add_``. Default ``False``.
        use_cautious: when ``True``, masks update components whose sign disagrees
            with the gradient sign and rescales by the kept fraction (Cautious
            Optimizers). Default ``False``.
        min_8bit_size: minimum tensor element count to be eligible for 8-bit
            quantization (only Linear / 1x1 conv shapes). Default ``16384``.
        quant_block_size: number of values per quantization block. Larger blocks
            save more memory at the cost of precision. Default ``2048``.
    """

    def __init__(
        self,
        params,
        lr: float | None = None,
        eps: tuple[float, float] = (1e-30, 1e-16),
        clip_threshold: float = 1.0,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        stochastic_rounding: bool = False,
        use_cautious: bool = False,
        min_8bit_size: int = 16384,
        quant_block_size: int = 2048,
    ) -> None:
        if lr is None or lr <= 0.0:
            raise ValueError(f"{self.__class__.__name__} requires lr > 0")
        if not all(0.0 <= beta <= 1.0 for beta in betas):
            raise ValueError(f"{self.__class__.__name__} betas must be in [0, 1]")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "betas": betas,
            "weight_decay": weight_decay,
            "stochastic_rounding": stochastic_rounding,
            "use_cautious": use_cautious,
            "min_8bit_size": min_8bit_size,
            "quant_block_size": quant_block_size,
        }
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return True

    @property
    def supports_flat_params(self) -> bool:
        return False

    @staticmethod
    def _should_use_matrix_factorization(grad_shape: torch.Size) -> bool:
        d = len(grad_shape)
        return d == 2 or (d == 4 and grad_shape[2] == 1 and grad_shape[3] == 1)

    @staticmethod
    def _should_quantize_param(grad_shape: torch.Size, min_8bit_size: int) -> bool:
        if CAME8bit._should_use_matrix_factorization(grad_shape):
            return grad_shape.numel() > min_8bit_size
        return False

    @staticmethod
    def _quantize_param(params: torch.Tensor, quant_block_size: int):
        if params.numel() <= 1:
            return params
        chunks = params.split(quant_block_size)
        out: list = [None] * len(chunks)
        for i, chunk in enumerate(chunks):
            chunk_max = chunk.max()
            chunk_min = chunk.min()
            scale = (chunk_max - chunk_min) / 255.0
            values = ((chunk - chunk_min) / scale).round().byte()
            out[i] = {"value": values, "scale": scale, "min": chunk_min}
        return out

    @staticmethod
    def _dequantize_param(quantized) -> torch.Tensor:
        if not isinstance(quantized, list):
            return quantized
        chunks: list = [None] * len(quantized)
        for i, q in enumerate(quantized):
            chunks[i] = (q["value"].float() * q["scale"]) + q["min"]
        return torch.cat(chunks)

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(
        exp_avg_sq_row: torch.Tensor,
        exp_avg_sq_col: torch.Tensor,
    ) -> torch.Tensor:
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @torch.no_grad()
    def step_parameter(self, p: torch.nn.Parameter, group: dict[str, Any], i: int = 0) -> None:
        """Run a single optimization step for one parameter.

        Per-parameter entry point used by fused-backward-pass trainers and
        block-coordinate gradient-release hooks.
        """
        if p.grad is None:
            return
        grad = p.grad.data
        if grad.dtype in {torch.float16, torch.bfloat16}:
            grad = grad.float()
        if grad.is_sparse:
            raise RuntimeError(f"{self.__class__.__name__} does not support sparse gradients.")

        grad_shape = grad.shape
        use_factor = self._should_use_matrix_factorization(grad_shape)
        should_quantize = self._should_quantize_param(grad_shape, group["min_8bit_size"])
        state = self.state[p]

        if len(state) == 0:
            state["step"] = 0
            state["RMS"] = 0
            state["exp_avg"] = (
                self._quantize_param(torch.zeros_like(grad), group["quant_block_size"])
                if should_quantize
                else torch.zeros_like(grad)
            )
            if use_factor:
                state["exp_avg_sq_row"] = torch.zeros(grad_shape[0]).type_as(grad)
                state["exp_avg_sq_col"] = torch.zeros(grad_shape[1]).type_as(grad)
                state["exp_avg_res_row"] = torch.zeros(grad_shape[0]).type_as(grad)
                state["exp_avg_res_col"] = torch.zeros(grad_shape[1]).type_as(grad)
            else:
                state["exp_avg_sq"] = (
                    self._quantize_param(torch.zeros_like(grad), group["quant_block_size"])
                    if should_quantize
                    else torch.zeros_like(grad)
                )

        state["step"] += 1
        state["RMS"] = self._rms(p.data)

        update: torch.Tensor = (grad ** 2) + group["eps"][0]
        if use_factor:
            exp_avg_sq_row = state["exp_avg_sq_row"]
            exp_avg_sq_col = state["exp_avg_sq_col"]
            sq_update = update if len(grad_shape) == 2 else update.squeeze()
            exp_avg_sq_row.mul_(group["betas"][1]).add_(
                sq_update.mean(dim=-1), alpha=1.0 - group["betas"][1]
            )
            exp_avg_sq_col.mul_(group["betas"][1]).add_(
                sq_update.mean(dim=-2), alpha=1.0 - group["betas"][1]
            )
            update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
            if update.shape != grad_shape:
                update = update.view(grad_shape)
        else:
            exp_avg_sq = (
                self._dequantize_param(state["exp_avg_sq"])
                if should_quantize
                else state["exp_avg_sq"]
            )
            exp_avg_sq.mul_(group["betas"][1]).add_(update, alpha=1.0 - group["betas"][1])
            update = exp_avg_sq.rsqrt()
            state["exp_avg_sq"] = (
                self._quantize_param(exp_avg_sq, group["quant_block_size"])
                if should_quantize
                else exp_avg_sq
            )

        update.mul_(grad)
        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

        exp_avg = (
            self._dequantize_param(state["exp_avg"])
            if should_quantize
            else state["exp_avg"]
        )
        exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])
        state["exp_avg"] = (
            self._quantize_param(exp_avg, group["quant_block_size"])
            if should_quantize
            else exp_avg
        )

        if use_factor:
            res = (update - exp_avg) ** 2 + group["eps"][1]
            exp_avg_res_row = state["exp_avg_res_row"]
            exp_avg_res_col = state["exp_avg_res_col"]
            re_update = res if len(grad_shape) == 2 else res.squeeze()
            exp_avg_res_row.mul_(group["betas"][2]).add_(
                re_update.mean(dim=-1), alpha=1.0 - group["betas"][2]
            )
            exp_avg_res_col.mul_(group["betas"][2]).add_(
                re_update.mean(dim=-2), alpha=1.0 - group["betas"][2]
            )
            res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
            if res_approx.shape != grad.shape:
                res_approx = res_approx.view(grad.shape)
            update = res_approx.mul_(exp_avg)
        else:
            update = exp_avg.clone()

        if group["use_cautious"]:
            mask = (update * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            update.mul_(mask)

        if group["weight_decay"] > 0:
            if p.dtype == torch.bfloat16 and group["stochastic_rounding"]:
                _add_stochastic_scaled_(p.data, p.data, alpha=-group["weight_decay"] * group["lr"])
            else:
                p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

        update.mul_(group["lr"])
        if p.dtype == torch.bfloat16 and group["stochastic_rounding"]:
            _add_stochastic_scaled_(p.data, -update)
        else:
            p.data.add_(-update)

    @torch.no_grad()
    def step_param(self, p: torch.nn.Parameter, group: dict[str, Any]) -> None:
        self.step_parameter(p, group, 0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                self.step_parameter(p, group, i)
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        quantizable_keys = (
            "exp_avg",
            "exp_avg_sq",
            "exp_avg_sq_col",
            "exp_avg_sq_row",
            "exp_avg_res_col",
            "exp_avg_res_row",
        )
        for state in self.state.values():
            for key in quantizable_keys:
                if key not in state:
                    continue
                value = state[key]
                if isinstance(value, list):
                    for entry in value:
                        if isinstance(entry, dict) and "value" in entry:
                            entry["value"] = entry["value"].byte()
                elif isinstance(value, torch.Tensor):
                    state[key] = value.float()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class CAME(CAME8bit):
    """Plain CAME optimizer with fp32 optimizer states.

    This shares CAME8bit's per-parameter step path and BF16 stochastic
    rounding, but never stores optimizer states in 8-bit quantized chunks.
    """

    def __init__(
        self,
        params,
        lr: float | None = None,
        eps: tuple[float, float] = (1e-30, 1e-16),
        clip_threshold: float = 1.0,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        stochastic_rounding: bool = False,
        use_cautious: bool = False,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
            stochastic_rounding=stochastic_rounding,
            use_cautious=use_cautious,
            min_8bit_size=sys.maxsize,
        )

    @staticmethod
    def _should_quantize_param(grad_shape: torch.Size, min_8bit_size: int) -> bool:
        return False
