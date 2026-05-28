"""Int8 weight-only quantized full fine-tuning for LTX-2.

Trainable weights are stored as int8 (1 byte/param, no bf16 master); the optimizer
updates them in place with stochastic rounding, since a small bf16 update would round
to zero under nearest rounding. Forward/backward dequantize to bf16 for the GEMM, so
this needs no FP8 tensor cores. Full bf16 gradients (no low-rank projection).

Granularity is selectable: row-wise (group_size=0, one scale per output channel) or
group-wise (group_size>0, one scale per G input elements — finer scales, lower
per-step quantization error).

Based on the int8 quantized-training recipe from PyTorch torchao (BSD-3-Clause).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils._python_dispatch import return_and_correct_aliasing

aten = torch.ops.aten


@torch.no_grad()
def quantize_int8(
    tensor: Tensor,
    group_size: int = 0,
    stochastic_rounding: bool = False,
    eps: float = 1e-12,
    outlier_clip_quantile: float = 1.0,
):
    """Symmetric int8 quantization. Returns (int8 data [out,in], scale).

    group_size<=0 -> row-wise: scale shape (out,). group_size>0 -> group-wise along the
    input dim: scale shape (out, in//group_size). Stochastic rounding rounds up with
    probability x-floor(x) so that small updates that would round to zero under nearest
    rounding still make stochastic progress.

    outlier_clip_quantile<1.0 sets the scale from a per-row/group quantile of |w| rather
    than the absmax: the top (1 - q) of weights gets clipped to ±127·scale, the rest gets
    a tighter grid. Default 1.0 uses the absmax.
    """
    out_features, in_features = tensor.shape
    use_group = bool(group_size) and group_size > 0 and in_features % group_size == 0 and group_size < in_features
    if use_group:
        t = tensor.reshape(out_features, in_features // group_size, group_size)
        if outlier_clip_quantile >= 1.0:
            scale = t.abs().amax(-1) / 127
        else:
            scale = torch.quantile(t.abs().float(), outlier_clip_quantile, dim=-1).to(t.dtype) / 127
        inv = (1.0 / scale.float().clamp(eps)).unsqueeze(-1)
        q = t.float() * inv
    else:
        if outlier_clip_quantile >= 1.0:
            scale = tensor.abs().amax(1) / 127
        else:
            scale = torch.quantile(tensor.abs().float(), outlier_clip_quantile, dim=1).to(tensor.dtype) / 127
        inv = (1.0 / scale.float().clamp(eps)).view(-1, 1)
        q = tensor.float() * inv
    if stochastic_rounding:
        q = (q + torch.rand_like(q)).floor()
    else:
        q = q.round()
    q = q.clip(-128, 127).reshape(out_features, in_features).to(torch.int8)
    return q, scale


def _dequantize(int_data: Tensor, scale: Tensor) -> Tensor:
    if scale.ndim == 1:  # row-wise
        return int_data * scale.view(-1, 1)
    out_features, in_features = int_data.shape
    n_groups = scale.shape[-1]
    return (int_data.reshape(out_features, n_groups, in_features // n_groups) * scale.unsqueeze(-1)).reshape(
        out_features, in_features
    )


class _Int8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: "Int8QTWeight", bias: Tensor | None = None):
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        w = weight.dequantize()  # bf16 weight; group-wise scale can't factor out of the GEMM
        out = input @ w.T
        return out + bias if bias is not None else out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input, weight = ctx.saved_tensors
        w = weight.dequantize()
        grad_input = grad_output @ w
        grad_weight = grad_output.reshape(-1, weight.shape[0]).T @ input.reshape(-1, weight.shape[1])
        grad_bias = grad_output.reshape(-1, weight.shape[0]).sum(0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias


_OPS: dict = {}


def _implements(ops):
    ops = ops if isinstance(ops, (list, tuple)) else [ops]

    def deco(fn):
        for op in ops:
            _OPS[op] = fn
        return fn

    return deco


class Int8QTWeight(Tensor):
    """Int8 weight that updates in place with stochastic rounding (row- or group-wise).

    outlier_clip_quantile (default 1.0 = absmax) is preserved on the weight and applied
    on every in-place requantize so the scale stays tight against the bulk distribution
    during training instead of being inflated by a few outliers.
    """

    @staticmethod
    def __new__(cls, int_data: Tensor, scale: Tensor, group_size: int = 0, outlier_clip_quantile: float = 1.0):
        return Tensor._make_wrapper_subclass(cls, int_data.shape, dtype=scale.dtype, device=int_data.device, requires_grad=False)

    def __init__(self, int_data: Tensor, scale: Tensor, group_size: int = 0, outlier_clip_quantile: float = 1.0):
        assert int_data.dtype is torch.int8 and int_data.ndim == 2
        self.int_data = int_data
        self.scale = scale
        self.group_size = int(group_size)
        self.outlier_clip_quantile = float(outlier_clip_quantile)

    def __tensor_flatten__(self):
        return ["int_data", "scale"], [self.group_size, self.outlier_clip_quantile]

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, meta, outer_size, outer_stride):
        return cls(inner_tensors["int_data"], inner_tensors["scale"], meta[0], meta[1] if len(meta) > 1 else 1.0)

    def dequantize(self) -> Tensor:
        return _dequantize(self.int_data, self.scale)

    @classmethod
    @torch.no_grad()
    def from_float(cls, weight: Tensor, group_size: int = 0, outlier_clip_quantile: float = 1.0) -> "Int8QTWeight":
        int_data, scale = quantize_int8(
            weight.detach(), group_size, stochastic_rounding=False, outlier_clip_quantile=outlier_clip_quantile
        )
        return cls(int_data, scale, group_size, outlier_clip_quantile)

    def __repr__(self):
        return f"Int8QTWeight(shape={tuple(self.shape)}, group_size={self.group_size}, dtype={self.dtype})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.linear:
            return _Int8Linear.apply(args[0], args[1], args[2] if len(args) > 2 else None)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        handler = _OPS.get(func)
        if handler is None:
            raise NotImplementedError(f"Int8QTWeight: unimplemented op {func}")
        return handler(func, types, args, kwargs)


@_implements([aten.detach.default, aten.clone.default])
def _(func, types, args, kwargs):
    out = Int8QTWeight(
        func(args[0].int_data),
        func(args[0].scale),
        args[0].group_size,
        getattr(args[0], "outlier_clip_quantile", 1.0),
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@_implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", None)
    out = Int8QTWeight(
        args[0].int_data.to(device=device),
        args[0].scale.to(device=device, dtype=dtype),
        args[0].group_size,
        getattr(args[0], "outlier_clip_quantile", 1.0),
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@_implements(aten.zeros_like.default)
def _(func, types, args, kwargs):
    dtype = kwargs.get("dtype", args[0].dtype)
    device = kwargs.get("device", args[0].device)
    return torch.zeros(args[0].shape, dtype=dtype, device=device)  # plain tensor for optimizer state


@_implements([aten.sub.Tensor, aten.mul.Tensor, aten.linalg_vector_norm.default])
def _(func, types, args, kwargs):
    args = [x.dequantize() if isinstance(x, Int8QTWeight) else x for x in args]
    return func(*args, **kwargs)


@_implements(aten.copy_.default)
def _(func, types, args, kwargs):
    dst, src = args[0], args[1]
    if isinstance(dst, Int8QTWeight) and isinstance(src, Int8QTWeight):
        dst.int_data.copy_(src.int_data)
        dst.scale.copy_(src.scale)
    elif isinstance(dst, Int8QTWeight):
        int_data, scale = quantize_int8(
            src,
            dst.group_size,
            stochastic_rounding=True,  # SR on update
            outlier_clip_quantile=getattr(dst, "outlier_clip_quantile", 1.0),
        )
        dst.int_data.copy_(int_data)
        dst.scale.copy_(scale)
    else:
        dst.copy_(src.dequantize())
    return dst


@_implements([aten.add_.Tensor, aten.addcdiv_.default, aten.mul_.Tensor])
def _(func, types, args, kwargs):
    original = args[0]
    out = func(args[0].dequantize(), *args[1:], **kwargs)
    return original.copy_(out)


def convert_to_int8_training(model: nn.Module, *, filter_fn=None, group_size: int = 0, outlier_clip_quantile: float = 1.0) -> int:
    """Swap eligible nn.Linear weights to Int8QTWeight in place. Returns count."""
    replaced = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module.weight, Int8QTWeight):
            continue
        if filter_fn is not None and not filter_fn(module, name):
            continue
        w = module.weight
        module.weight = nn.Parameter(
            Int8QTWeight.from_float(w.data, group_size, outlier_clip_quantile=outlier_clip_quantile),
            requires_grad=w.requires_grad,
        )
        replaced += 1
    return replaced
