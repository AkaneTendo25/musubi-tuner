"""Manual CUDA benchmark for H3 INT8 attention.

Run with ``PYTHONPATH=src python tests/benchmark_h3_int8_attention.py [sequence]``.
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn.functional as F

from musubi_tuner.minimax_h3.int8_attention import _int8_forward, _quantize_rows, int8_attention


def measure_cuda(name, function, iterations=20):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    print(f"{name}: {start.elapsed_time(end) / iterations:.3f} ms")


def measure(name, function, tensors, gradient, iterations=10):
    for _ in range(3):
        output = function(*tensors)
        output.backward(gradient)
        for tensor in tensors:
            tensor.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(iterations):
        output = function(*tensors)
        output.backward(gradient)
        for tensor in tensors:
            tensor.grad = None
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000 / iterations
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"{name}: {elapsed:.2f} ms, {peak:.1f} MiB peak")


def measure_forward(name, function, tensors, iterations=20):
    with torch.no_grad():
        for _ in range(3):
            function(*tensors)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(iterations):
            function(*tensors)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000 / iterations
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"{name} forward: {elapsed:.2f} ms, {peak:.1f} MiB peak")


def main():
    sequence = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    torch.manual_seed(123)
    shape = (1, 56, sequence, 128)
    tensors = tuple(torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    gradient = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    sdpa = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    print(f"H3 shape={shape}")
    measure_cuda("quantize Q", lambda: _quantize_rows(tensors[0]))
    measure_cuda("quantize K", lambda: _quantize_rows(tensors[1]))
    measure_cuda("complete INT8 forward internals", lambda: _int8_forward(*tensors))
    measure_forward("SDPA", sdpa, tensors)
    measure_forward("INT8", int8_attention, tensors)
    measure("SDPA", sdpa, tensors, gradient)
    measure("INT8", int8_attention, tensors, gradient)


if __name__ == "__main__":
    main()
