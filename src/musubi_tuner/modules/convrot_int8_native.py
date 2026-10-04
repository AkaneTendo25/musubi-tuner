"""Eager CUDA kernels used by the ConvRot INT8 training path."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    libdevice = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"block": 32}, num_warps=4, num_stages=2),
            triton.Config({"block": 64}, num_warps=4, num_stages=2),
            triton.Config({"block": 64}, num_warps=8, num_stages=2),
            triton.Config({"block": 128}, num_warps=8, num_stages=2),
        ],
        key=["rows", "cols"],
    )
    @triton.jit
    def _transpose_int8_kernel(source, output, rows, cols, block: tl.constexpr):
        row = tl.program_id(0) * block + tl.arange(0, block)
        col = tl.program_id(1) * block + tl.arange(0, block)
        mask = (row[:, None] < rows) & (col[None, :] < cols)
        tile = tl.load(source + row[:, None] * cols + col[None, :], mask=mask)
        tl.store(output + col[None, :] * rows + row[:, None], tile, mask=mask)

    @triton.jit
    def _scaled_quantize_rowwise_kernel(
        x_ptr, column_scale_ptr, q_ptr, row_scale_ptr, rows, cols, stride_xm, stride_xn, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < cols
        x = tl.load(x_ptr + row * stride_xm + offsets * stride_xn, mask=mask, other=0.0)
        column_scale = tl.load(column_scale_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)
        scaled = (x.to(tl.bfloat16) * column_scale).to(tl.bfloat16)
        maximum = tl.max(tl.abs(scaled), axis=0)
        quant_scale = tl.maximum(maximum / 127.0, 1e-30)
        normalized = (scaled / quant_scale.to(tl.bfloat16)).to(tl.bfloat16)
        code = tl.clamp(libdevice.rint(normalized.to(tl.float32)), -128.0, 127.0).to(tl.int32)
        tl.store(q_ptr + row * cols + offsets, code.to(tl.int8), mask=mask)
        tl.store(row_scale_ptr + row, quant_scale.to(tl.float32))


def transpose_int8_contiguous_or_fallback(source: torch.Tensor) -> torch.Tensor:
    """Return a contiguous transpose, using the measured Triton path in eager CUDA."""
    if (
        not HAS_TRITON
        or torch.compiler.is_compiling()
        or source.device.type != "cuda"
        or source.dtype is not torch.int8
        or source.ndim != 2
        or not source.is_contiguous()
    ):
        return source.t().contiguous()
    rows, cols = source.shape
    output = torch.empty((cols, rows), device=source.device, dtype=source.dtype)
    grid = lambda meta: (triton.cdiv(rows, meta["block"]), triton.cdiv(cols, meta["block"]))
    _transpose_int8_kernel[grid](source, output, rows, cols)
    return output


def scaled_quantize_rowwise_bf16(x: torch.Tensor, column_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = x.shape
    scale = column_scale.reshape(-1)
    if not scale.is_contiguous():
        scale = scale.contiguous()
    quantized = torch.empty((rows, cols), dtype=torch.int8, device=x.device)
    row_scale = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    block = max(128, triton.next_power_of_2(cols))
    _scaled_quantize_rowwise_kernel[(rows,)](x, scale, quantized, row_scale, rows, cols, x.stride(0), x.stride(1), BLOCK=block)
    return quantized, row_scale


def scaled_int8_lora_linear_or_fallback(
    x: torch.Tensor,
    column_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    down_output: torch.Tensor,
    up_transposed: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Fuse BF16 column scaling with row quantization when its tested contract holds."""
    from musubi_tuner.modules.convrot_int8_kernels import int8_lora_linear

    eligible = (
        HAS_TRITON
        and not torch.compiler.is_compiling()
        and x.device.type == "cuda"
        and x.dtype is torch.bfloat16
        and x.ndim == 2
        and x.stride(1) == 1
        and column_scale.device == x.device
        and column_scale.numel() == x.shape[1]
    )
    if not eligible:
        materialized = x * column_scale.reshape(1, -1).to(x.dtype)
        return int8_lora_linear(materialized, weight, weight_scale, down_output, up_transposed, out_dtype)

    quantized, input_scale = scaled_quantize_rowwise_bf16(x, column_scale)
    from musubi_tuner.modules import convrot_int8_kernels as kernels

    rows, inner = quantized.shape
    columns, rank = weight.shape[0], down_output.shape[-1]
    scales = weight_scale.reshape(-1).to(torch.float32).contiguous()
    if scales.numel() == 1:
        scales = scales.expand(columns).contiguous()
    output = torch.empty((rows, columns), device=x.device, dtype=out_dtype)
    down2d = down_output.reshape(rows, rank)

    def grid(meta):
        return (triton.cdiv(rows, meta["block_m"]) * triton.cdiv(columns, meta["block_n"]),)

    kernels._int8_matmul_dequant_lora_kernel[grid](
        quantized,
        weight,
        output,
        input_scale,
        scales,
        down2d,
        up_transposed,
        rows,
        columns,
        inner,
        rank=rank,
        stride_am=quantized.stride(0),
        stride_ak=quantized.stride(1),
        stride_bk=weight.stride(1),
        stride_bn=weight.stride(0),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        stride_dm=down2d.stride(0),
        stride_dr=down2d.stride(1),
        stride_ur=up_transposed.stride(0),
        stride_un=up_transposed.stride(1),
        block_r=triton.next_power_of_2(rank),
    )
    return output
