"""Benchmark H3 SwiGLU activation forward and backward on CUDA."""

from __future__ import annotations

import argparse

import torch
from torch.nn import functional as F

from musubi_tuner.minimax_h3.triton_kernels import try_fused_swiglu


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
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(123)
    width = 14336
    projected = torch.randn(1, args.sequence, 2 * width, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn(1, args.sequence, width, device="cuda", dtype=torch.bfloat16)

    def eager(value: torch.Tensor) -> torch.Tensor:
        gate, linear = value.chunk(2, dim=-1)
        return F.silu(gate) * linear

    def fused(value: torch.Tensor) -> torch.Tensor:
        output = try_fused_swiglu(value)
        if output is None:
            raise RuntimeError("fused SwiGLU kernel was not selected")
        return output

    eager_x = projected.detach().clone().requires_grad_(True)
    fused_x = projected.detach().clone().requires_grad_(True)
    eager_output = eager(eager_x)
    fused_output = fused(fused_x)
    torch.autograd.backward(eager_output, upstream)
    torch.autograd.backward(fused_output, upstream)
    forward_error = float(((fused_output.float() - eager_output.float()).norm() / eager_output.float().norm()).detach())
    gradient_error = float(((fused_x.grad.float() - eager_x.grad.float()).norm() / eager_x.grad.float().norm()).detach())

    with torch.no_grad():
        eager_ms = _timed(lambda: eager(projected), args.iterations)
        fused_ms = _timed(lambda: fused(projected), args.iterations)

    def eager_fb() -> None:
        value = projected.detach().requires_grad_(True)
        torch.autograd.backward(eager(value), upstream)

    def fused_fb() -> None:
        value = projected.detach().requires_grad_(True)
        torch.autograd.backward(fused(value), upstream)

    eager_fb_ms = _timed(eager_fb, args.iterations)
    fused_fb_ms = _timed(fused_fb, args.iterations)
    print(f"sequence={args.sequence} forward_rel_l2={forward_error:.8f} gradient_rel_l2={gradient_error:.8f}")
    print(f"forward eager_ms={eager_ms:.4f} fused_ms={fused_ms:.4f} speedup={eager_ms / fused_ms:.3f}x")
    print(f"fwd_bwd eager_ms={eager_fb_ms:.4f} fused_ms={fused_fb_ms:.4f} speedup={eager_fb_ms / fused_fb_ms:.3f}x")


if __name__ == "__main__":
    main()
