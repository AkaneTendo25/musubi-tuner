"""
W8A8 activation quantization for reducing VRAM during LoRA training.

Uses custom autograd.Functions that save only references to quantized weight
buffers instead of full-size dequantized weights, eliminating ~32MB per linear
layer from the autograd graph.

Two modes:
  - int8 (default): Converts FP8 weights to int8 per-channel, uses torch._int_mm.
    Requires SM 7.5+ (Turing: RTX 2080+, T4, A100, etc.)
  - fp8: Keeps FP8 weights as-is, uses transient dequantization.
    Works on any GPU that supports FP8 (same as --fp8_scaled).
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SMALL_BATCH_THRESHOLD = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dequantize_weight(weight, scale_weight):
    """Dequantize quantized weight to float32, handling all scale shapes."""
    if scale_weight.ndim < 3:
        # per-tensor [1] or per-channel [out, 1]
        return weight.float() * scale_weight.float()
    else:
        # block-wise [out, num_blocks, 1]
        out_features, num_blocks, _ = scale_weight.shape
        w = weight.float().contiguous().view(out_features, num_blocks, -1)
        w = w * scale_weight.float()
        return w.view(weight.shape)


def _quantize_int8_per_token(x):
    """Per-token int8 quantization.

    Args:
        x: float tensor [M, K]
    Returns:
        (int8 tensor [M, K], float32 scale [M, 1])
    """
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max / 127.0).clamp(min=1e-30).float()
    quantized = (x.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale


# ---------------------------------------------------------------------------
# Custom autograd Functions
# ---------------------------------------------------------------------------

class _W8A8Int8Function(torch.autograd.Function):
    """int8 W8A8: per-token input quantization + torch._int_mm.

    Saves only int8 weight and float32 per-channel scale in the autograd graph
    (both are existing module buffers, so 0 extra bytes).
    """

    @staticmethod
    def forward(ctx, x, weight_int8, weight_scale, bias):
        # weight_int8: [out, in] int8
        # weight_scale: [out, 1] float32
        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        x_int8, x_scale = _quantize_int8_per_token(x_2d)

        # [M, K] @ [K, N] -> [M, N] int32   (N = out_features)
        # _int_mm handles transposed second operand (column-major) natively via cuBLAS;
        # no .contiguous() needed on weight_int8.t(), which avoids a 16MB transient copy.
        out_int32 = torch._int_mm(x_int8.contiguous(), weight_int8.t())

        # Rescale: x_scale [M,1] * weight_scale^T [1,out] broadcasts to [M, out]
        output = out_int32.float() * x_scale * weight_scale.t()

        if bias is not None:
            output = output + bias.float()

        output = output.to(x.dtype).reshape(*original_shape[:-1], -1)

        # Save ONLY references to existing module buffers — 0 extra bytes
        ctx.save_for_backward(weight_int8, weight_scale)
        ctx.input_dtype = x.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Only activation gradients are expected (LoRA-only training).
        # needs_input_grad = (x=True, weight=False, scale=False, bias=False)
        if ctx.needs_input_grad[1]:
            raise RuntimeError(
                "W8A8 int8 does not support weight gradients (full fine-tuning). "
                "Use --network_module for LoRA training."
            )

        weight_int8, weight_scale = ctx.saved_tensors
        # Transient dequantization — freed after this function returns
        weight_f = weight_int8.float() * weight_scale  # [out, in]

        go_2d = grad_output.reshape(-1, grad_output.shape[-1]).float()
        grad_input = go_2d @ weight_f  # [M, out] @ [out, in] -> [M, in]
        grad_input = grad_input.to(ctx.input_dtype).reshape(
            *grad_output.shape[:-1], weight_f.shape[1]
        )
        return grad_input, None, None, None


class _W8A8TransientDequantFunction(torch.autograd.Function):
    """Generic W8A8: dequantize transiently, don't save dequantized weight.

    Works with any quantized format (int8 per-channel, FP8 per-tensor/channel/block).
    Uses standard F.linear for the forward matmul.
    """

    @staticmethod
    def forward(ctx, x, weight, scale_weight, bias):
        weight_deq = _dequantize_weight(weight, scale_weight).to(x.dtype)
        output = F.linear(x, weight_deq, bias)
        # weight_deq is NOT saved — freed when this scope ends
        ctx.save_for_backward(weight, scale_weight)
        ctx.input_dtype = x.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_input_grad[1]:
            raise RuntimeError(
                "W8A8 does not support weight gradients (full fine-tuning). "
                "Use --network_module for LoRA training."
            )

        weight, scale_weight = ctx.saved_tensors
        weight_deq = _dequantize_weight(weight, scale_weight)  # transient

        go_2d = grad_output.reshape(-1, grad_output.shape[-1]).float()
        grad_input = go_2d @ weight_deq  # [M, out] @ [out, in]
        grad_input = grad_input.to(ctx.input_dtype).reshape(
            *grad_output.shape[:-1], weight_deq.shape[1]
        )
        return grad_input, None, None, None


# ---------------------------------------------------------------------------
# Weight conversion
# ---------------------------------------------------------------------------

def _convert_fp8_to_int8_weights(module):
    """One-time conversion: FP8 weight + scale -> int8 weight + per-channel scale.

    After conversion:
      module.weight.data  = int8 [out, in]
      module.scale_weight = float32 [out, 1]
    """
    with torch.no_grad():
        weight_float = _dequantize_weight(module.weight.data, module.scale_weight)
        abs_max = weight_float.abs().amax(dim=1, keepdim=True)  # [out, 1]
        scale = (abs_max / 127.0).clamp(min=1e-30).float()
        weight_int8 = (weight_float / scale).round().clamp(-127, 127).to(torch.int8)

        module.weight.requires_grad_(False)
        module.weight.data = weight_int8
        module.scale_weight.data = scale  # [out, 1] float32


# ---------------------------------------------------------------------------
# Monkey-patched forward functions
# ---------------------------------------------------------------------------

def _make_w8a8_int8_forward(use_int_mm):
    """Create W8A8 int8 forward with small-batch fallback."""
    if use_int_mm:
        def forward(self, x):
            num_tokens = x.reshape(-1, x.shape[-1]).shape[0]
            if num_tokens <= SMALL_BATCH_THRESHOLD:
                w = (self.weight.detach().float() * self.scale_weight.float()).to(x.dtype)
                return F.linear(x, w, self.bias)
            return _W8A8Int8Function.apply(x, self.weight, self.scale_weight, self.bias)
    else:
        # _int_mm unavailable: still save VRAM via custom autograd
        def forward(self, x):
            num_tokens = x.reshape(-1, x.shape[-1]).shape[0]
            if num_tokens <= SMALL_BATCH_THRESHOLD:
                w = (self.weight.detach().float() * self.scale_weight.float()).to(x.dtype)
                return F.linear(x, w, self.bias)
            return _W8A8TransientDequantFunction.apply(
                x, self.weight, self.scale_weight, self.bias
            )
    return forward


def _make_w8a8_fp8_forward():
    """Create W8A8 fp8 forward with small-batch fallback."""
    def forward(self, x):
        num_tokens = x.reshape(-1, x.shape[-1]).shape[0]
        if num_tokens <= SMALL_BATCH_THRESHOLD:
            w = _dequantize_weight(self.weight.detach(), self.scale_weight).to(x.dtype)
            return F.linear(x, w, self.bias)
        return _W8A8TransientDequantFunction.apply(
            x, self.weight, self.scale_weight, self.bias
        )
    return forward


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_w8a8_monkey_patch(model, w8a8_mode="int8"):
    """Replace forward methods on FP8-patched linears with W8A8 versions.

    Must be called AFTER apply_fp8_monkey_patch + load_state_dict
    (needs scale_weight buffers and loaded FP8 weights).

    For int8 mode: also converts FP8 weights to int8 per-channel.

    Args:
        model: nn.Module with FP8-patched linear layers
        w8a8_mode: "int8" (default, SM 7.5+) or "fp8" (any GPU)
    """
    use_int_mm = hasattr(torch, "_int_mm")
    if w8a8_mode == "int8" and not use_int_mm:
        logger.warning(
            "torch._int_mm not available (requires PyTorch 2.1+); "
            "W8A8 int8 will use dequantize+matmul fallback (still saves VRAM)."
        )

    # Build the forward function once (shared by all patched layers)
    if w8a8_mode == "int8":
        new_forward = _make_w8a8_int8_forward(use_int_mm)
    else:
        new_forward = _make_w8a8_fp8_forward()

    patched_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not hasattr(module, "scale_weight"):
            continue

        if w8a8_mode == "int8":
            _convert_fp8_to_int8_weights(module)

        module.forward = new_forward.__get__(module, type(module))
        patched_count += 1

    logger.info("W8A8 %s: patched %d linear layers", w8a8_mode, patched_count)
    return model
