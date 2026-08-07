"""Compile smoke/benchmark for one production-shape H3 transformer block."""

from __future__ import annotations

import argparse

import torch

from musubi_tuner.minimax_h3.model import MiniMaxH3TransformerBlock, MiniMaxH3TransformerConfig


def _step(block, source, timestep, indices, rotary):
    hidden = source.detach().clone().requires_grad_(True)
    output = block(hidden, timestep, indices, rotary, None)
    (gradient,) = torch.autograd.grad(output.float().square().mean(), hidden)
    return output.detach(), gradient.detach()


def _timed(block, source, timestep, indices, rotary, iterations):
    for _ in range(2):
        _step(block, source, timestep, indices, rotary)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _step(block, source, timestep, indices, rotary)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    torch.manual_seed(321)
    device, dtype = torch.device("cuda"), torch.bfloat16
    config = MiniMaxH3TransformerConfig(num_layers=1, num_refiner_layers=0)
    block = MiniMaxH3TransformerBlock(config).to(device=device, dtype=dtype).requires_grad_(False)
    source = torch.randn(1, args.sequence, config.hidden_size, device=device, dtype=dtype)
    timestep = torch.randn(2, config.time_embed_dim, device=device, dtype=dtype)
    indices = torch.arange(args.sequence, device=device) % 6
    angles = torch.randn(args.sequence, 3 * config.rope_freq_dim, device=device)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(dtype)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(dtype)
    rotary = (cos, sin)

    expected, expected_grad = _step(block, source, timestep, indices, rotary)
    eager_ms = _timed(block, source, timestep, indices, rotary, args.iterations)
    block.attn.fused_qk_norm_rope = True
    fused, fused_grad = _step(block, source, timestep, indices, rotary)
    fused_ms = _timed(block, source, timestep, indices, rotary, args.iterations)
    fused_output_error = float((fused.float() - expected.float()).norm() / expected.float().norm())
    fused_gradient_error = float((fused_grad.float() - expected_grad.float()).norm() / expected_grad.float().norm())
    compiled = torch.compile(block, backend="inductor", mode="default", dynamic=False, fullgraph=False)
    actual, actual_grad = _step(compiled, source, timestep, indices, rotary)
    compiled_ms = _timed(compiled, source, timestep, indices, rotary, args.iterations)
    output_error = float((actual.float() - expected.float()).norm() / expected.float().norm())
    gradient_error = float((actual_grad.float() - expected_grad.float()).norm() / expected_grad.float().norm())
    print(
        f"fused_output_rel_l2={fused_output_error:.8f} fused_gradient_rel_l2={fused_gradient_error:.8f} "
        f"fused_ms={fused_ms:.3f} fused_speedup={eager_ms / fused_ms:.3f}x"
    )
    print(f"sequence={args.sequence} output_rel_l2={output_error:.8f} gradient_rel_l2={gradient_error:.8f}")
    print(f"eager_ms={eager_ms:.3f} compiled_ms={compiled_ms:.3f} speedup={eager_ms / compiled_ms:.3f}x")
    print(f"peak_allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}")


if __name__ == "__main__":
    main()
