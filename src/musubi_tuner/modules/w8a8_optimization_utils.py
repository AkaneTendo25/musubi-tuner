"""
W8A8 activation quantization for reducing VRAM during LoRA training.

Uses custom autograd.Functions that save only references to quantized weight
buffers instead of full-size dequantized weights, eliminating ~32MB per linear
layer from the autograd graph.

Two modes:
  - int8 (default): Converts FP8 weights to int8 tensorwise, uses torch._int_mm.
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
_INT8_CONVERSION_CHUNK_ELEMENTS = 4 * 1024 * 1024
_INT_MM_MIN_CUDA_CAPABILITY = (7, 5)


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


def _iter_weight_row_slices(weight, *, max_chunk_elements: int | None = None):
    """Yield row slices capped by element count to bound transient fp32 memory."""
    if max_chunk_elements is None:
        max_chunk_elements = _INT8_CONVERSION_CHUNK_ELEMENTS
    if weight.ndim != 2:
        yield slice(0, weight.shape[0])
        return

    out_features, in_features = weight.shape
    rows_per_chunk = max(1, max_chunk_elements // max(int(in_features), 1))
    for start in range(0, out_features, rows_per_chunk):
        yield slice(start, min(start + rows_per_chunk, out_features))


def _slice_scale_weight_for_rows(scale_weight, row_slice):
    if scale_weight.ndim == 0:
        return scale_weight
    if scale_weight.ndim == 1:
        return scale_weight
    if scale_weight.shape[0] == 1:
        return scale_weight
    return scale_weight[row_slice]


def _dequantize_weight_chunk(weight_chunk, scale_weight_chunk):
    """Dequantize one row chunk to float32 without materializing the whole weight."""
    chunk = weight_chunk.to(dtype=torch.float32, copy=True)
    if scale_weight_chunk.ndim < 3:
        chunk.mul_(scale_weight_chunk.float())
        return chunk

    out_features, num_blocks, _ = scale_weight_chunk.shape
    if not chunk.is_contiguous():
        chunk = chunk.contiguous()
    chunk = chunk.view(out_features, num_blocks, -1)
    chunk.mul_(scale_weight_chunk.float())
    return chunk.view(weight_chunk.shape)


def _dequantized_weight_abs_max_chunked(weight, scale_weight):
    abs_max = torch.zeros((), device=weight.device, dtype=torch.float32)
    for row_slice in _iter_weight_row_slices(weight):
        scale_chunk = _slice_scale_weight_for_rows(scale_weight, row_slice)
        chunk = _dequantize_weight_chunk(weight[row_slice], scale_chunk)
        if chunk.numel() == 0:
            continue
        abs_max = torch.maximum(abs_max, chunk.abs_().max())
    return abs_max


def _quantize_dequantized_weight_to_int8_chunked(weight, scale_weight, int8_scale):
    weight_int8 = torch.empty(weight.shape, device=weight.device, dtype=torch.int8)
    for row_slice in _iter_weight_row_slices(weight):
        scale_chunk = _slice_scale_weight_for_rows(scale_weight, row_slice)
        chunk = _dequantize_weight_chunk(weight[row_slice], scale_chunk)
        chunk.div_(int8_scale)
        chunk.round_()
        chunk.clamp_(min=-127, max=127)
        weight_int8[row_slice].copy_(chunk.to(torch.int8))
    return weight_int8


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


def _int_mm_allow_small_m(a, b):
    """torch._int_mm requires M > 16 on CUDA; pad tiny batches instead of dequantizing weights."""
    m = a.shape[0]
    if m > SMALL_BATCH_THRESHOLD:
        return torch._int_mm(a, b)

    padded_m = SMALL_BATCH_THRESHOLD + 1
    padding = torch.zeros((padded_m - m, a.shape[1]), device=a.device, dtype=a.dtype)
    padded = torch.cat((a, padding), dim=0)
    return torch._int_mm(padded, b)[:m]


def _supports_int_mm(device):
    """Return whether torch._int_mm can be used for tensors on *device*."""
    if not hasattr(torch, "_int_mm"):
        return False
    device = torch.device(device)
    if device.type == "cpu":
        return True
    if device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) >= _INT_MM_MIN_CUDA_CAPABILITY


def _get_triton_mm_8bit():
    """Import the Triton row-major int8 matmul lazily."""
    try:
        from musubi_tuner.modules.triton_mm_8bit import mm_8bit
    except ImportError:
        return None
    return mm_8bit


def _supports_triton_mm(device):
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Custom autograd Functions
# ---------------------------------------------------------------------------

class _W8A8Int8Function(torch.autograd.Function):
    """int8 W8A8: per-token input quantization + torch._int_mm.

    Saves only int8 weight and float32 tensor scale in the autograd graph
    (both are existing module buffers, so 0 extra bytes).
    """

    @staticmethod
    def forward(ctx, x, weight_int8, weight_int8_t, weight_scale, bias, triton_mm):
        # weight_int8: [out, in] int8
        # weight_int8_t: optional [in, out] contiguous int8 fallback for fast backward RHS layout
        # weight_scale: [1] float32
        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        x_int8, x_scale = _quantize_int8_per_token(x_2d)

        # [M, K] @ [K, N] -> [M, N] int32   (N = out_features)
        # _int_mm handles transposed second operand (column-major) natively via cuBLAS;
        # no .contiguous() needed on weight_int8.t(), which avoids a 16MB transient copy.
        out_int32 = _int_mm_allow_small_m(x_int8.contiguous(), weight_int8.t())

        # Rescale: x_scale [M,1] * scalar weight_scale broadcasts to [M, out]
        output = out_int32.float() * x_scale * weight_scale.float().view(1, 1)

        if bias is not None:
            output = output + bias.float()

        output = output.to(x.dtype).reshape(*original_shape[:-1], -1)

        # Save ONLY references to existing module buffers - 0 extra bytes
        if triton_mm is None:
            ctx.save_for_backward(weight_int8_t, weight_scale)
        else:
            ctx.save_for_backward(weight_int8, weight_scale)
        ctx.input_dtype = x.dtype
        ctx.triton_mm = triton_mm
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Only activation gradients are expected (LoRA-only training).
        # needs_input_grad = (x=True, weight=False, weight_t=False, scale=False, bias=False, triton_mm=False)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            raise RuntimeError(
                "W8A8 int8 does not support weight gradients (full fine-tuning). "
                "Use --network_module for LoRA training."
            )

        weight_saved, weight_scale = ctx.saved_tensors
        go_2d = grad_output.reshape(-1, grad_output.shape[-1]).float()

        grad_int8, grad_scale = _quantize_int8_per_token(go_2d)
        if ctx.triton_mm is None:
            grad_int32 = _int_mm_allow_small_m(grad_int8.contiguous(), weight_saved.t())
            in_features = weight_saved.shape[0]
        else:
            try:
                grad_int32 = ctx.triton_mm(grad_int8.contiguous(), weight_saved)
            except Exception:
                logger.exception("W8A8 Triton backward failed; falling back to torch._int_mm.")
                grad_int32 = _int_mm_allow_small_m(grad_int8.contiguous(), weight_saved)
            in_features = weight_saved.shape[1]
        grad_input = grad_int32.float() * grad_scale * weight_scale.float().view(1, 1)
        grad_input = grad_input.to(ctx.input_dtype).reshape(
            *grad_output.shape[:-1], in_features
        )
        return grad_input, None, None, None, None, None


class _W8A8TransientDequantFunction(torch.autograd.Function):
    """Generic W8A8: dequantize transiently, don't save dequantized weight.

    Works with any quantized format (int8 per-channel, FP8 per-tensor/channel/block).
    Uses standard F.linear for the forward matmul.
    """

    @staticmethod
    def forward(ctx, x, weight, scale_weight, bias):
        weight_deq = _dequantize_weight(weight, scale_weight).to(x.dtype)
        output = F.linear(x, weight_deq, bias)
        # weight_deq is NOT saved - freed when this scope ends
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

def _convert_fp8_to_int8_weights(module, *, keep_transpose_buffer: bool):
    """One-time conversion: FP8 weight + scale -> int8 weight + tensor scale.

    After conversion:
      module.weight.data  = int8 [out, in]
      module.scale_weight = float32 [1]
    """
    with torch.no_grad():
        weight = module.weight.detach()
        scale_weight = module.scale_weight.detach()
        abs_max = _dequantized_weight_abs_max_chunked(weight, scale_weight)
        scale = (abs_max / 127.0).clamp(min=1e-30).float()
        weight_int8 = _quantize_dequantized_weight_to_int8_chunked(weight, scale_weight, scale)

        module.weight.requires_grad_(False)
        module.weight = torch.nn.Parameter(weight_int8, requires_grad=False)
        if module.bias is not None:
            module.bias.requires_grad_(False)
        module.scale_weight = scale.reshape(1).to(device=weight_int8.device, dtype=torch.float32)
        if keep_transpose_buffer:
            weight_int8_t = weight_int8.t().contiguous()
            if "weight_int8_t" in module._buffers:
                module.weight_int8_t = weight_int8_t
            else:
                module.register_buffer("weight_int8_t", weight_int8_t, persistent=False)
        elif "weight_int8_t" in module._buffers:
            del module._buffers["weight_int8_t"]


# ---------------------------------------------------------------------------
# Monkey-patched forward functions
# ---------------------------------------------------------------------------

def _make_w8a8_int8_forward(use_int_mm, triton_mm):
    """Create W8A8 int8 forward with small-batch fallback."""
    if use_int_mm:
        def forward(self, x):
            if not _supports_int_mm(x.device):
                return _W8A8TransientDequantFunction.apply(
                    x, self.weight, self.scale_weight, self.bias
                )
            if triton_mm is not None and _supports_triton_mm(x.device):
                weight_int8_t = None
                triton_mm_for_device = triton_mm
            elif hasattr(self, "weight_int8_t"):
                weight_int8_t = self.weight_int8_t
                triton_mm_for_device = None
            else:
                return _W8A8TransientDequantFunction.apply(
                    x, self.weight, self.scale_weight, self.bias
                )
            return _W8A8Int8Function.apply(
                x, self.weight, weight_int8_t, self.scale_weight, self.bias, triton_mm_for_device
            )
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

def apply_w8a8_monkey_patch(model, w8a8_mode="int8", state_dict: dict[str, torch.Tensor] | None = None):
    """Replace forward methods on FP8-patched linears with W8A8 versions.

    Must be called AFTER apply_fp8_monkey_patch + load_state_dict
    (needs scale_weight buffers and loaded FP8 weights).

    For int8 mode: also converts FP8 weights to int8 tensorwise.

    Args:
        model: nn.Module with FP8-patched linear layers
        w8a8_mode: "int8" (default, SM 7.5+) or "fp8" (any GPU)
        state_dict: Optional loaded state dict to update in-place after int8
            conversion. With load_state_dict(assign=True), this releases the
            old FP8 weight entries instead of keeping both FP8 and int8 copies.
    """
    if w8a8_mode not in {"int8", "fp8"}:
        raise ValueError(f"Unsupported W8A8 mode: {w8a8_mode!r}. Expected 'int8' or 'fp8'.")

    use_int_mm = hasattr(torch, "_int_mm")
    triton_mm = _get_triton_mm_8bit() if w8a8_mode == "int8" and use_int_mm else None
    if w8a8_mode == "int8" and not use_int_mm:
        logger.warning(
            "torch._int_mm not available (requires PyTorch 2.1+); "
            "W8A8 int8 will use dequantize+matmul fallback (still saves VRAM)."
        )
    if w8a8_mode == "int8" and use_int_mm and triton_mm is None:
        logger.info("W8A8 int8: Triton row-major matmul unavailable; keeping transpose buffers for fast backward.")

    # Build the forward function once (shared by all patched layers)
    if w8a8_mode == "int8":
        new_forward = _make_w8a8_int8_forward(use_int_mm, triton_mm)
    else:
        new_forward = _make_w8a8_fp8_forward()

    patched_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not hasattr(module, "scale_weight"):
            continue

        if w8a8_mode == "int8":
            _convert_fp8_to_int8_weights(module, keep_transpose_buffer=triton_mm is None)
            if state_dict is not None:
                weight_key = f"{name}.weight"
                scale_key = f"{name}.scale_weight"
                if weight_key in state_dict:
                    state_dict[weight_key] = module.weight.detach()
                if scale_key in state_dict:
                    state_dict[scale_key] = module.scale_weight.detach()

        module.forward = new_forward.__get__(module, type(module))
        patched_count += 1

    backend = "triton" if triton_mm is not None else "torch"
    logger.info("W8A8 %s (%s): patched %d linear layers", w8a8_mode, backend, patched_count)
    return model
