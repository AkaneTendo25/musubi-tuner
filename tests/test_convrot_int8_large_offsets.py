"""Opt-in CUDA regressions for row offsets exceeding signed int32.

Run with H3_RUN_LARGE_GPU_TESTS=1; each case uses about 7 GiB of VRAM.
"""

import os

import pytest
import torch

from musubi_tuner.modules import convrot_int8_kernels as kernels

pytestmark = pytest.mark.skipif(
    os.environ.get("H3_RUN_LARGE_GPU_TESTS") != "1" or not torch.cuda.is_available() or not kernels.HAS_TRITON,
    reason="requires CUDA/Triton and H3_RUN_LARGE_GPU_TESTS=1 (large allocations)",
)


@pytest.mark.parametrize("mode", ["tensorwise", "per_channel", "lora"])
def test_large_gemm_matches_row_split(mode):
    m, n, k = 80000, 28672, 32
    assert (m - 1) * n > 2**31
    generator = torch.Generator(device="cuda").manual_seed(123)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=generator)
    weight = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8, generator=generator)
    scales = torch.linspace(0.01, 0.02, n if mode != "tensorwise" else 1, device="cuda")
    if mode == "lora":
        down = torch.randn(m, 16, device="cuda", dtype=torch.bfloat16, generator=generator)
        up = torch.randn(16, n, device="cuda", dtype=torch.bfloat16, generator=generator)

    def run(start, stop):
        if mode == "lora":
            return kernels.int8_lora_linear(x[start:stop], weight, scales, down[start:stop], up)
        return kernels.int8_linear(x[start:stop], weight, scales)

    actual = run(0, m)
    for start in range(0, m, 4096):
        stop = min(start + 4096, m)
        torch.testing.assert_close(actual[start:stop], run(start, stop), rtol=0, atol=0)


def test_large_row_quantization_matches_row_split():
    # The row pointer itself, not just the GEMM output, crosses 2**31.
    m, n = 75008, 28672
    assert (m - 1) * n > 2**31
    x = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    pattern = torch.linspace(-2, 3, n, device="cuda", dtype=torch.bfloat16)
    x.copy_(pattern)
    actual, scales = kernels.triton_quantize_rowwise(x)
    for start in range(0, m, 4096):
        stop = min(start + 4096, m)
        expected, expected_scales = kernels.triton_quantize_rowwise(x[start:stop])
        torch.testing.assert_close(actual[start:stop], expected, rtol=0, atol=0)
        torch.testing.assert_close(scales[start:stop], expected_scales, rtol=0, atol=0)
