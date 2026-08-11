"""Benchmark H3 indexed AdaLN RMSNorm forward and backward on CUDA."""

from __future__ import annotations

import argparse

import torch
from torch.nn import functional as F

from musubi_tuner.minimax_h3.triton_kernels import try_fused_indexed_adaln_rmsnorm


def _timed(operation, iterations: int) -> float:
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    torch.manual_seed(123)
    dtype = torch.bfloat16
    hidden = 2688
    x = torch.randn(1, args.sequence, hidden, device="cuda", dtype=dtype)
    weight = torch.randn(hidden, device="cuda", dtype=dtype)
    shift = torch.randn(6, hidden, device="cuda", dtype=dtype)
    scale = torch.randn(6, hidden, device="cuda", dtype=dtype) * 0.1
    indices = torch.randint(0, 6, (args.sequence,), device="cuda")
    upstream = torch.randn_like(x)

    def eager(value: torch.Tensor) -> torch.Tensor:
        normalized = F.rms_norm(value, (hidden,), weight, 1e-5)
        return normalized * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)

    def fused(value: torch.Tensor) -> torch.Tensor:
        output = try_fused_indexed_adaln_rmsnorm(value, weight, shift, scale, indices, 1e-5)
        if output is None:
            raise RuntimeError("fused indexed AdaLN kernel was not selected")
        return output

    eager_x = x.detach().clone().requires_grad_(True)
    fused_x = x.detach().clone().requires_grad_(True)
    eager_output = eager(eager_x)
    fused_output = fused(fused_x)
    torch.autograd.backward(eager_output, upstream)
    torch.autograd.backward(fused_output, upstream)
    forward_error = float(((fused_output.float() - eager_output.float()).norm() / eager_output.float().norm()).detach())
    gradient_error = float(((fused_x.grad.float() - eager_x.grad.float()).norm() / eager_x.grad.float().norm()).detach())

    with torch.no_grad():
        eager_ms = _timed(lambda: eager(x), args.iterations)
        fused_ms = _timed(lambda: fused(x), args.iterations)

    def eager_fb() -> None:
        value = x.detach().requires_grad_(True)
        torch.autograd.backward(eager(value), upstream)

    def fused_fb() -> None:
        value = x.detach().requires_grad_(True)
        torch.autograd.backward(fused(value), upstream)

    eager_fb_ms = _timed(eager_fb, args.iterations)
    fused_fb_ms = _timed(fused_fb, args.iterations)
    print(f"sequence={args.sequence} forward_rel_l2={forward_error:.8f} gradient_rel_l2={gradient_error:.8f}")
    print(f"forward eager_ms={eager_ms:.4f} fused_ms={fused_ms:.4f} speedup={eager_ms / fused_ms:.3f}x")
    print(f"fwd_bwd eager_ms={eager_fb_ms:.4f} fused_ms={fused_fb_ms:.4f} speedup={eager_fb_ms / fused_fb_ms:.3f}x")


if __name__ == "__main__":
    main()
