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
    sparse_ratio: float = 0.0,
):
    """Symmetric int8 quantization.

    Returns (int8 data [out,in], scale) when sparse_ratio<=0, or
    (int8 data, scale, sparse_idx, sparse_val) when sparse_ratio>0.

    group_size<=0 -> row-wise: scale shape (out,). group_size>0 -> group-wise along the
    input dim: scale shape (out, in//group_size). Stochastic rounding rounds up with
    probability x-floor(x) so that small updates that would round to zero under nearest
    rounding still make stochastic progress.

    outlier_clip_quantile<1.0 sets the scale from a per-row/group quantile of |w| rather
    than the absmax: the top (1 - q) of weights gets clipped to ±127·scale, the rest gets
    a tighter grid. Default 1.0 uses the absmax.

    sparse_ratio>0 (dense-and-sparse) holds the top
    ``sparse_ratio`` fraction of |w| as an exact fp32 side-vector (sparse_idx flat
    indices, sparse_val values) and zeroes those positions BEFORE computing the int8
    scale, so a few heavy outliers no longer inflate the grid for the bulk. Outliers are
    scattered back on dequantize. Returns the extra (sparse_idx int64, sparse_val fp32).
    Orthogonal to outlier_clip_quantile (clip discards the tail; sparse preserves it
    exactly) — use one or the other.
    """
    out_features, in_features = tensor.shape
    sparse_idx = None
    sparse_val = None
    work = tensor
    if sparse_ratio and sparse_ratio > 0.0:
        flat = tensor.reshape(-1)
        k = int(flat.numel() * float(sparse_ratio))
        if k > 0:
            sparse_idx = flat.abs().topk(k, sorted=False).indices
            sparse_val = flat[sparse_idx].to(torch.float32).clone()
            work = flat.clone()
            work[sparse_idx] = 0.0  # exclude outliers from the dense int8 grid
            work = work.reshape(out_features, in_features)
    use_group = bool(group_size) and group_size > 0 and in_features % group_size == 0 and group_size < in_features
    if use_group:
        t = work.reshape(out_features, in_features // group_size, group_size)
        if outlier_clip_quantile >= 1.0:
            scale = t.abs().amax(-1) / 127
        else:
            scale = torch.quantile(t.abs().float(), outlier_clip_quantile, dim=-1).to(t.dtype) / 127
        inv = (1.0 / scale.float().clamp(eps)).unsqueeze(-1)
        q = t.float() * inv
    else:
        if outlier_clip_quantile >= 1.0:
            scale = work.abs().amax(1) / 127
        else:
            scale = torch.quantile(work.abs().float(), outlier_clip_quantile, dim=1).to(work.dtype) / 127
        inv = (1.0 / scale.float().clamp(eps)).view(-1, 1)
        q = work.float() * inv
    if stochastic_rounding:
        q = (q + torch.rand_like(q)).floor()
    else:
        q = q.round()
    q = q.clip(-128, 127).reshape(out_features, in_features).to(torch.int8)
    if sparse_ratio and sparse_ratio > 0.0:
        if sparse_idx is None:  # k==0 (tiny tensor): empty side-vectors
            sparse_idx = tensor.new_empty(0, dtype=torch.int64)
            sparse_val = tensor.new_empty(0, dtype=torch.float32)
        return q, scale, sparse_idx, sparse_val
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
    def __new__(
        cls,
        int_data: Tensor,
        scale: Tensor,
        group_size: int = 0,
        outlier_clip_quantile: float = 1.0,
        sparse_idx: Tensor | None = None,
        sparse_val: Tensor | None = None,
        sparse_ratio: float = 0.0,
    ):
        return Tensor._make_wrapper_subclass(cls, int_data.shape, dtype=scale.dtype, device=int_data.device, requires_grad=False)

    def __init__(
        self,
        int_data: Tensor,
        scale: Tensor,
        group_size: int = 0,
        outlier_clip_quantile: float = 1.0,
        sparse_idx: Tensor | None = None,
        sparse_val: Tensor | None = None,
        sparse_ratio: float = 0.0,
    ):
        assert int_data.dtype is torch.int8 and int_data.ndim == 2
        self.int_data = int_data
        self.scale = scale
        self.group_size = int(group_size)
        self.outlier_clip_quantile = float(outlier_clip_quantile)
        self.sparse_ratio = float(sparse_ratio)
        # dense-and-sparse outlier side-vectors (None when sparse_ratio<=0).
        self.sparse_idx = sparse_idx
        self.sparse_val = sparse_val

    def __tensor_flatten__(self):
        inner = ["int_data", "scale"]
        if self.sparse_val is not None:
            inner = inner + ["sparse_idx", "sparse_val"]
        return inner, [self.group_size, self.outlier_clip_quantile, self.sparse_ratio]

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, meta, outer_size, outer_stride):
        ocq = meta[1] if len(meta) > 1 else 1.0
        sr = meta[2] if len(meta) > 2 else 0.0
        return cls(
            inner_tensors["int_data"],
            inner_tensors["scale"],
            meta[0],
            ocq,
            inner_tensors.get("sparse_idx"),
            inner_tensors.get("sparse_val"),
            sr,
        )

    def dequantize(self) -> Tensor:
        w = _dequantize(self.int_data, self.scale)
        if self.sparse_val is not None and self.sparse_val.numel() > 0:
            # w is fresh (int_data*scale); scatter outliers in place, no clone
            w = w.contiguous()
            w.view(-1)[self.sparse_idx] = self.sparse_val.to(w.dtype)
        return w

    @torch.no_grad()
    def requantize_(self, t: Tensor, stochastic_rounding: bool = True) -> "Int8QTWeight":
        """In-place write of ``t`` into the int8 storage.

        Replaces int_data/scale (and the dense-sparse outlier side-vectors when
        sparse_ratio>0) with a fresh quantization of ``t`` using this weight's
        group_size / outlier_clip_quantile / sparse_ratio. stochastic_rounding=True
        (default) is the per-step SR update path; False writes deterministic
        round-to-nearest, for optimizers that handle the rounding residual themselves.
        """
        if self.sparse_ratio > 0.0:
            int_data, scale, sidx, sval = quantize_int8(
                t,
                self.group_size,
                stochastic_rounding=stochastic_rounding,
                outlier_clip_quantile=self.outlier_clip_quantile,
                sparse_ratio=self.sparse_ratio,
            )
            self.int_data.copy_(int_data)
            self.scale.copy_(scale)
            # k = floor(numel*ratio) is constant for a fixed weight, so the side-vector
            # sizes are stable across updates and copy_ is in place.
            self.sparse_idx.copy_(sidx)
            self.sparse_val.copy_(sval)
            return self
        int_data, scale = quantize_int8(
            t,
            self.group_size,
            stochastic_rounding=stochastic_rounding,
            outlier_clip_quantile=self.outlier_clip_quantile,
        )
        self.int_data.copy_(int_data)
        self.scale.copy_(scale)
        return self

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        weight: Tensor,
        group_size: int = 0,
        outlier_clip_quantile: float = 1.0,
        sparse_ratio: float = 0.0,
    ) -> "Int8QTWeight":
        if sparse_ratio and sparse_ratio > 0.0:
            int_data, scale, sidx, sval = quantize_int8(
                weight.detach(),
                group_size,
                stochastic_rounding=False,
                outlier_clip_quantile=outlier_clip_quantile,
                sparse_ratio=sparse_ratio,
            )
            return cls(int_data, scale, group_size, outlier_clip_quantile, sidx, sval, sparse_ratio)
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
    a = args[0]
    sidx = getattr(a, "sparse_idx", None)
    sval = getattr(a, "sparse_val", None)
    out = Int8QTWeight(
        func(a.int_data),
        func(a.scale),
        a.group_size,
        getattr(a, "outlier_clip_quantile", 1.0),
        func(sidx) if sidx is not None else None,
        func(sval) if sval is not None else None,
        getattr(a, "sparse_ratio", 0.0),
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@_implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", None)
    a = args[0]
    sidx = getattr(a, "sparse_idx", None)
    sval = getattr(a, "sparse_val", None)
    out = Int8QTWeight(
        a.int_data.to(device=device),
        a.scale.to(device=device, dtype=dtype),
        a.group_size,
        getattr(a, "outlier_clip_quantile", 1.0),
        sidx.to(device=device) if sidx is not None else None,  # idx stays int64
        sval.to(device=device) if sval is not None else None,  # val stays fp32
        getattr(a, "sparse_ratio", 0.0),
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
        if getattr(dst, "sparse_val", None) is not None and getattr(src, "sparse_val", None) is not None:
            dst.sparse_idx.copy_(src.sparse_idx)
            dst.sparse_val.copy_(src.sparse_val)
    elif isinstance(dst, Int8QTWeight):
        if getattr(dst, "sparse_ratio", 0.0) > 0.0:
            int_data, scale, sidx, sval = quantize_int8(
                src,
                dst.group_size,
                stochastic_rounding=True,  # SR on update
                outlier_clip_quantile=getattr(dst, "outlier_clip_quantile", 1.0),
                sparse_ratio=dst.sparse_ratio,
            )
            dst.int_data.copy_(int_data)
            dst.scale.copy_(scale)
            dst.sparse_idx.copy_(sidx)
            dst.sparse_val.copy_(sval)
        else:
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


@_implements(
    [
        aten.add_.Tensor,
        aten.sub_.Tensor,
        aten.addcdiv_.default,
        aten.addcmul_.default,
        aten.mul_.Tensor,
        aten.div_.Tensor,
        aten.lerp_.Scalar,
        aten.lerp_.Tensor,
    ]
)
def _(func, types, args, kwargs):
    # Dequantize every Int8QTWeight in args, not just args[0]: ops like
    # grad.sub_(param) put the weight at args[1], which would otherwise recurse
    # through __torch_dispatch__ until the recursion limit.
    original = args[0]
    deq_args = tuple(a.dequantize() if isinstance(a, Int8QTWeight) else a for a in args)
    out = func(*deq_args, **kwargs)
    return original.copy_(out)


def convert_to_int8_training(
    model: nn.Module, *, filter_fn=None, group_size: int = 0, outlier_clip_quantile: float = 1.0, sparse_ratio: float = 0.0
) -> int:
    """Swap eligible nn.Linear weights to Int8QTWeight in place. Returns count."""
    replaced = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module.weight, Int8QTWeight):
            continue
        if filter_fn is not None and not filter_fn(module, name):
            continue
        w = module.weight
        module.weight = nn.Parameter(
            Int8QTWeight.from_float(w.data, group_size, outlier_clip_quantile=outlier_clip_quantile, sparse_ratio=sparse_ratio),
            requires_grad=w.requires_grad,
        )
        replaced += 1
    return replaced
