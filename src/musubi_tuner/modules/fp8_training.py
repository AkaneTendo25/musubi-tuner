"""FP8 full fine-tuning for LTX-2 via torch._scaled_mm (per-tensor dynamic scaling).

Based on the FP8 training recipe from NVIDIA TransformerEngine and PyTorch torchao
(both BSD-3-Clause); independent re-implementation, no torchao runtime dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2

_FP8_DTYPES = {"e4m3": E4M3, "e5m2": E5M2}


def resolve_fp8_dtype(name: str) -> torch.dtype:
    key = str(name).lower().replace("float8_", "").replace("fn", "")
    if key not in _FP8_DTYPES:
        raise ValueError(f"Unknown fp8 dtype {name!r}; expected one of {sorted(_FP8_DTYPES)}")
    return _FP8_DTYPES[key]


def is_fp8_training_supported(device: torch.device | int | None = None) -> bool:
    """True if the CUDA device has FP8 tensor cores (sm_89+)."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) >= (8, 9)


def assert_fp8_training_supported(device: torch.device | int | None = None) -> None:
    if not is_fp8_training_supported(device):
        cap = torch.cuda.get_device_capability(device) if torch.cuda.is_available() else None
        raise RuntimeError(
            f"--fp8_gemm requires FP8 tensor cores (compute capability >= 8.9 / Ada or Hopper); "
            f"got {cap}. torch._scaled_mm cannot run FP8 GEMMs on this GPU. "
            f"Use int8 full-FT (--qgalore_full_ft) on pre-Ada hardware instead."
        )


def _to_fp8(t: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor dynamic scaling. Returns (fp8 tensor, dequant scale fp32)."""
    fp8_max = torch.finfo(fp8_dtype).max
    amax = t.detach().abs().amax().clamp(min=1e-12)
    scale = fp8_max / amax
    q = (t.float() * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    inv = (1.0 / scale).to(torch.float32)  # _scaled_mm multiplies operands by these
    return q, inv


def _scaled_matmul(a_hp: torch.Tensor, b_hp: torch.Tensor, a_dtype: torch.dtype, b_dtype: torch.dtype) -> torch.Tensor:
    """Compute a_hp @ b_hp in bf16 via an FP8 GEMM. a_hp:(M,K) b_hp:(K,N).

    _scaled_mm needs the contraction dim K to be a multiple of 16. In the grad_weight
    GEMM K is the token count (data-dependent, often not %16), so zero-pad it —
    padded zeros don't change the matmul, so the result stays exact.
    """
    k = a_hp.shape[-1]
    if k % 16 != 0:
        pad = 16 - (k % 16)
        a_hp = torch.nn.functional.pad(a_hp, (0, pad))  # (M,K) -> (M,K+pad)
        b_hp = torch.nn.functional.pad(b_hp, (0, 0, 0, pad))  # (K,N) -> (K+pad,N)
    aq, sa = _to_fp8(a_hp.contiguous(), a_dtype)  # (M,K) row-major fp8
    bq, sb = _to_fp8(b_hp.t().contiguous(), b_dtype)  # quantize (N,K) row-major ...
    bq = bq.t()  # ... -> (K,N) column-major
    return torch._scaled_mm(aq, bq, scale_a=sa.to(aq.device), scale_b=sb.to(aq.device), out_dtype=torch.bfloat16)


_scaled_matmul_compiled = None
_compile_disabled = False


def _fp8_gemm_compiled(*args) -> torch.Tensor:
    """Call the compiled GEMM; on any compile/runtime failure, warn once and fall back
    to eager for the rest of the run."""
    global _compile_disabled
    try:
        return _scaled_matmul_compiled(*args)
    except Exception as e:  # noqa: BLE001
        if not _compile_disabled:
            logger.warning("--fp8_gemm_compile: compiled FP8 GEMM failed (%s); using eager FP8 GEMM.", e)
            _compile_disabled = True
        return _scaled_matmul(*args)


def _get_matmul(compile_gemm: bool) -> Callable[..., torch.Tensor]:
    """Return the GEMM helper, optionally torch.compiled.

    Region-compiling only this helper (not the surrounding block) fuses the scaling
    (amax/cast/clamp) into the GEMM and survives the graph breaks block-level compile
    hits on LTX-2 (gradient checkpointing, flash attention, Python control flow).
    """
    global _scaled_matmul_compiled, _compile_disabled
    if not compile_gemm or _compile_disabled:
        return _scaled_matmul
    if _scaled_matmul_compiled is None:
        try:
            # One compiled helper serves every (attn/FFN) x (fwd/grad_x/grad_w) GEMM
            # shape; raise the recompile cap above dynamo's default of 8 so all of them
            # compile instead of thrashing back to eager.
            import torch._dynamo as _dynamo

            _dynamo.config.recompile_limit = max(int(getattr(_dynamo.config, "recompile_limit", 8)), 64)
            _scaled_matmul_compiled = torch.compile(_scaled_matmul, dynamic=False)
        except Exception as e:  # noqa: BLE001
            _compile_disabled = True
            logger.warning("--fp8_gemm_compile: torch.compile unavailable (%s); using eager FP8 GEMM.", e)
            return _scaled_matmul
    return _fp8_gemm_compiled


class _Float8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, grad_dtype: torch.dtype, mm: Callable) -> torch.Tensor:
        ctx.save_for_backward(x, w)
        ctx.grad_dtype = grad_dtype
        ctx.mm = mm
        return mm(x, w.t(), E4M3, E4M3)  # y = x @ w.T  (M,N)

    @staticmethod
    def backward(ctx, gy: torch.Tensor):
        x, w = ctx.saved_tensors
        gd, mm = ctx.grad_dtype, ctx.mm
        grad_x = mm(gy, w, gd, E4M3)  # gy @ w     (M,K)
        grad_w = mm(gy.t().contiguous(), x, gd, E4M3)  # gy.T @ x   (N,K)
        return grad_x, grad_w, None, None


class Float8TrainingLinear(nn.Linear):
    """Drop-in nn.Linear with FP8 fwd/bwd GEMMs; keeps a bf16 master weight + bias.

    The optimizer only sees the bf16 weight/grad, so it stays optimizer-agnostic.
    """

    grad_dtype: torch.dtype = E4M3
    compile_gemm: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        mm = _get_matmul(self.compile_gemm)
        y = _Float8LinearFn.apply(x.reshape(-1, shp[-1]), self.weight, self.grad_dtype, mm)
        y = y.reshape(*shp[:-1], self.out_features)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y


def _fp8_dims_ok(linear: nn.Linear) -> bool:
    # FP8 GEMMs (_scaled_mm) require both matmul dims to be multiples of 16.
    return (linear.in_features % 16 == 0) and (linear.out_features % 16 == 0)


@dataclass
class Fp8FftSummary:
    replaced: int = 0
    replaced_numel: int = 0
    skipped_not_target: int = 0
    skipped_dims: int = 0
    skipped_small: int = 0
    replaced_names: list[str] = field(default_factory=list)


def ltx2_fp8_filter(targets: str | Iterable[str] = "video", min_weight_numel: int = 16384) -> Callable[[nn.Linear, str], bool]:
    """Filter for the big LTX-2 attention/FFN GEMMs; excludes gates, norms, AdaLN.

    Reuses Q-GaLore's LTX-2 target matching so --fp8_gemm_targets behaves like
    --qgalore_targets (video / audio / attn / ff / blocks / all).
    """
    from musubi_tuner.optimizers.q_galore import _matches_ltx2_target, _parse_target_tokens

    tokens = _parse_target_tokens(targets)

    def _keep(mod: nn.Linear, fqn: str) -> bool:
        name = fqn.lower()
        if "gate" in name or "norm" in name:  # gated-attention logits / RMSNorm-adjacent
            return False
        return _matches_ltx2_target(fqn, tokens)

    _keep._tokens = tokens  # type: ignore[attr-defined]
    _keep._min_numel = int(min_weight_numel)  # type: ignore[attr-defined]
    return _keep


def convert_ltx2_to_fp8_training(
    model: nn.Module,
    *,
    targets: str | Iterable[str] = "video",
    grad_dtype: torch.dtype = E4M3,
    min_weight_numel: int = 16384,
    compile_gemm: bool = False,
) -> Fp8FftSummary:
    """Swap eligible nn.Linear -> Float8TrainingLinear in place. Keeps weights."""
    keep = ltx2_fp8_filter(targets, min_weight_numel)
    summary = Fp8FftSummary()
    modules = dict(model.named_modules())
    for parent_name, parent in list(modules.items()):
        for cname, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear) or isinstance(child, Float8TrainingLinear):
                continue
            fqn = f"{parent_name}.{cname}" if parent_name else cname
            if not keep(child, fqn):
                summary.skipped_not_target += 1
                continue
            if child.weight.numel() < int(min_weight_numel):
                summary.skipped_small += 1
                continue
            if not _fp8_dims_ok(child):
                summary.skipped_dims += 1
                continue
            # In-place class reassignment preserves the exact weight/bias tensors and
            # consumes no RNG. (A fresh Float8TrainingLinear would run nn.Linear's random
            # init, desyncing the global RNG — hence data/noise sampling — vs a bf16 run.)
            child.__class__ = Float8TrainingLinear
            child.grad_dtype = grad_dtype
            child.compile_gemm = compile_gemm
            summary.replaced += 1
            summary.replaced_numel += int(child.weight.numel())
            summary.replaced_names.append(fqn)
    return summary
