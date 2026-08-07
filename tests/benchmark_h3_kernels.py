"""Focused GPU benchmark for H3's fused Q/K norm+RoPE primitive."""

from __future__ import annotations

import argparse
import torch
from torch.nn import functional as F

from musubi_tuner.minimax_h3.model import _apply_rotary_emb
from musubi_tuner.minimax_h3.triton_kernels import try_fused_qk_norm_rope


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
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(123)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    heads, dim, rotary_dim, eps = 56, 128, 96, 1e-5
    q = torch.randn(1, args.sequence, heads, dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    q_weight = torch.randn(dim, device=device, dtype=dtype)
    k_weight = torch.randn(dim, device=device, dtype=dtype)
    angles = torch.randn(args.sequence, rotary_dim // 2, device=device, dtype=torch.float32)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(dtype)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(dtype)

    def eager():
        return (
            _apply_rotary_emb(F.rms_norm(q, (dim,), q_weight, eps), cos, sin),
            _apply_rotary_emb(F.rms_norm(k, (dim,), k_weight, eps), cos, sin),
        )

    fused = try_fused_qk_norm_rope(q, k, q_weight, k_weight, cos, sin, eps)
    if fused is None:
        raise RuntimeError("fused H3 kernel was not selected")
    expected = eager()
    forward_error = max(
        float((actual.float() - reference.float()).norm() / reference.float().norm())
        for actual, reference in zip(fused, expected, strict=True)
    )

    upstream_q = torch.randn_like(q)
    upstream_k = torch.randn_like(k)
    q_eager = q.detach().clone().requires_grad_(True)
    k_eager = k.detach().clone().requires_grad_(True)
    q_fused = q.detach().clone().requires_grad_(True)
    k_fused = k.detach().clone().requires_grad_(True)
    eager_out = (
        _apply_rotary_emb(F.rms_norm(q_eager, (dim,), q_weight, eps), cos, sin),
        _apply_rotary_emb(F.rms_norm(k_eager, (dim,), k_weight, eps), cos, sin),
    )
    fused_out = try_fused_qk_norm_rope(q_fused, k_fused, q_weight, k_weight, cos, sin, eps)
    assert fused_out is not None
    torch.autograd.backward(eager_out, (upstream_q, upstream_k))
    torch.autograd.backward(fused_out, (upstream_q, upstream_k))
    gradient_error = max(
        float((actual.float() - reference.float()).norm() / reference.float().norm())
        for actual, reference in ((q_fused.grad, q_eager.grad), (k_fused.grad, k_eager.grad))
    )

    with torch.no_grad():
        eager_ms = _timed(eager, args.iterations)
        fused_ms = _timed(lambda: try_fused_qk_norm_rope(q, k, q_weight, k_weight, cos, sin, eps), args.iterations)
    print(f"sequence={args.sequence} forward_rel_l2={forward_error:.8f} gradient_rel_l2={gradient_error:.8f}")
    print(f"eager_ms={eager_ms:.4f} fused_ms={fused_ms:.4f} speedup={eager_ms / fused_ms:.3f}x")


if __name__ == "__main__":
    main()
