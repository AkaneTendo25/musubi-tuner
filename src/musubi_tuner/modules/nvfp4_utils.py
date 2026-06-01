"""NVIDIA FP4 (E2M1) checkpoint loading utilities for pre-quantized LTX-2 models.

Handles the Lightricks nvfp4 checkpoint format where transformer block weights
(typically blocks 10-47) are stored as packed uint8 with two-level scaling:
  - ``weight``: uint8 packed (two FP4 E2M1 nibbles per byte), shape ``[out, in/2]``
  - ``weight_scale``: float8_e4m3fn per-block scales
  - ``weight_scale_2``: float32 per-tensor scalar scale

Early blocks (0-9), VAE, vocoder, connectors, biases, and norms are stored in bf16
and passed through unchanged.

The on-the-fly forward patch keeps packed weights in GPU memory (~10-11 GB for the
transformer) and dequantizes to bf16 only during each ``F.linear`` call, mirroring
the approach used by ``nf4_optimization_utils.py``.
"""

import json
import os
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from tqdm import tqdm

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FP4 E2M1 lookup table
# ---------------------------------------------------------------------------
# Nibble values 0x0-0x7 are positive, 0x8-0xF are negative mirrors.
FP4_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.bfloat16,
)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_nvfp4_checkpoint(model_path: str) -> bool:
    """Check whether a safetensors file is in Lightricks nvfp4 format.

    Detection is based on the ``_quantization_metadata`` key in the file
    metadata containing ``"nvfp4"`` format entries, or the presence of
    ``weight_scale_2`` keys alongside ``weight_scale`` keys.
    """
    try:
        from safetensors import safe_open
        check_path = model_path if isinstance(model_path, str) else model_path[0]
        with safe_open(check_path, framework="pt") as f:
            meta = f.metadata()
            if meta is not None:
                qm = meta.get("_quantization_metadata", "")
                if '"nvfp4"' in qm:
                    return True
            # Fallback: check for weight_scale_2 keys
            keys = f.keys()
            has_ws2 = any(k.endswith(".weight_scale_2") for k in keys)
            has_ws = any(k.endswith(".weight_scale") for k in keys)
            has_uint8 = False
            if has_ws2 and has_ws:
                # Confirm at least one uint8 weight
                for k in keys:
                    if k.endswith(".weight"):
                        t = f.get_tensor(k)
                        if t.dtype == torch.uint8:
                            has_uint8 = True
                            break
            return has_ws2 and has_ws and has_uint8
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Dequantize (for on-the-fly forward or one-shot conversion)
# ---------------------------------------------------------------------------

def dequantize_nvfp4_weight(
    w_uint8: torch.Tensor,
    block_scale: torch.Tensor,
    tensor_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an NVFP4-packed weight tensor.

    Args:
        w_uint8: Packed weight ``[out_features, in_features // 2]`` as uint8.
        block_scale: Per-block scales (float8_e4m3fn or similar), flattened or shaped.
        tensor_scale: Per-tensor scalar scale (float32).
        dtype: Output dtype.

    Returns:
        ``[out_features, in_features]`` tensor in *dtype*.
    """
    lut = FP4_E2M1_LUT.to(w_uint8.device)

    # Unpack two 4-bit values from each byte
    lo = (w_uint8 & 0x0F).to(torch.int64)
    hi = (w_uint8 >> 4).to(torch.int64)
    # Interleave: low nibble first, then high nibble (NVFP4 byte order)
    unpacked = torch.stack([lo, hi], dim=-1).reshape(-1)

    # Map 4-bit indices to FP4 E2M1 float values
    values = lut[unpacked]

    # Apply block scales: each scale covers a fixed group of consecutive values
    # Cast scales to bfloat16 first — PyTorch cannot promote float8 * bfloat16 directly
    total_values = values.shape[0]
    total_scales = block_scale.numel()
    block_size = total_values // total_scales
    bs_flat = block_scale.to(torch.bfloat16).to(w_uint8.device).reshape(-1).repeat_interleave(block_size)

    result = values * bs_flat * tensor_scale.to(torch.bfloat16).to(w_uint8.device)

    # Reshape to real weight dimensions (2x columns since 2 values per packed byte)
    real_shape = (w_uint8.shape[0], w_uint8.shape[1] * 2)
    return result.reshape(real_shape).to(dtype)


# ---------------------------------------------------------------------------
# Monkey-patched forward
# ---------------------------------------------------------------------------

def nvfp4_linear_forward_patch(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement forward for NVFP4-quantized Linear layers.

    Dequantizes packed uint8 weights on-the-fly using the stored block and
    tensor scales, then runs a standard ``F.linear``.
    """
    w = dequantize_nvfp4_weight(
        self.weight,               # packed uint8 [out, in//2]
        self.nvfp4_block_scale,    # float8_e4m3fn block scales
        self.nvfp4_tensor_scale,   # float32 per-tensor scale
        dtype=torch.bfloat16,
    )
    # Cast to input dtype for the matmul (e.g. if input is fp32 or fp16)
    if w.dtype != x.dtype:
        w = w.to(x.dtype)
    return F.linear(x, w, self.bias)


# ---------------------------------------------------------------------------
# Monkey-patch application (mirrors apply_nf4_monkey_patch)
# ---------------------------------------------------------------------------

def apply_nvfp4_monkey_patch(
    model: nn.Module,
    state_dict: dict,
) -> nn.Module:
    """Register NVFP4 buffers and replace forward on quantized Linears.

    Identifies quantized layers by looking for ``weight_scale`` and
    ``weight_scale_2`` keys in the state dict alongside uint8 ``.weight``
    tensors.  Non-quantized layers (bf16 weights) are left untouched.
    """
    # Collect quantized module paths from state dict.
    # Keys have been renamed by load_nvfp4_state_dict:
    #   .weight_scale   -> .nvfp4_block_scale
    #   .weight_scale_2 -> .nvfp4_tensor_scale
    nvfp4_modules: dict = {}  # module_path -> (packed_shape, real_out, real_in)
    for key in state_dict:
        if key.endswith(".nvfp4_tensor_scale"):
            module_path = key.rsplit(".nvfp4_tensor_scale", 1)[0]
            weight_key = module_path + ".weight"
            bs_key = module_path + ".nvfp4_block_scale"
            if weight_key in state_dict and bs_key in state_dict:
                wt = state_dict[weight_key]
                if wt.dtype == torch.uint8:
                    out_f = wt.shape[0]
                    in_f = wt.shape[1] * 2  # real in_features (2 values per packed byte)
                    nvfp4_modules[module_path] = (wt.shape, out_f, in_f)

    patched_count = 0
    for name, module in model.named_modules():
        if name not in nvfp4_modules:
            continue
        if not isinstance(module, nn.Linear):
            continue

        packed_shape, out_f, in_f = nvfp4_modules[name]
        bs_key = name + ".nvfp4_block_scale"
        ts_key = name + ".nvfp4_tensor_scale"

        # Replace the weight parameter with a buffer of packed shape
        # so load_state_dict(assign=True) doesn't complain about shape mismatch.
        del module.weight
        module.register_buffer("weight", torch.zeros(packed_shape, dtype=torch.uint8))

        # Register scale buffers
        bs_tensor = state_dict[bs_key]
        ts_tensor = state_dict[ts_key]
        module.register_buffer("nvfp4_block_scale", torch.zeros_like(bs_tensor))
        module.register_buffer("nvfp4_tensor_scale", torch.zeros_like(ts_tensor))

        # Store metadata
        module.nvfp4_out_features = out_f
        module.nvfp4_in_features = in_f
        module._nvfp4_quantized = True

        # Replace forward
        def new_forward(self, x):
            return nvfp4_linear_forward_patch(self, x)
        module.forward = new_forward.__get__(module, type(module))
        patched_count += 1

    logger.info(f"Number of NVFP4 monkey-patched Linear layers: {patched_count}")
    return model


# ---------------------------------------------------------------------------
# State dict loading for nvfp4 checkpoints
# ---------------------------------------------------------------------------

def load_nvfp4_state_dict(
    model_files: Union[str, List[str]],
    state_dict_key_filter: Optional[Callable[[str], bool]] = None,
    move_to_device: bool = False,
    target_device: Optional[Union[str, torch.device]] = None,
) -> dict:
    """Load an nvfp4 checkpoint into a state dict, preserving packed format.

    Quantized layers keep their uint8 packed weights, weight_scale (fp8), and
    weight_scale_2 (fp32) as separate state dict entries.  Non-quantized layers
    (bf16) are loaded as-is.

    The ``weight_scale`` / ``weight_scale_2`` keys are renamed to
    ``nvfp4_block_scale`` / ``nvfp4_tensor_scale`` in the output so they
    match the buffer names registered by ``apply_nvfp4_monkey_patch``.
    """
    if isinstance(model_files, str):
        model_files = [model_files]

    state_dict = {}
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file) as f:
            all_keys = list(f.keys())

            # Identify quantized layers
            ws2_keys = set(k for k in all_keys if k.endswith(".weight_scale_2"))
            ws_keys = set(k for k in all_keys if k.endswith(".weight_scale"))
            skip_suffixes = (".weight_scale", ".weight_scale_2", ".pre_quant_scale", ".input_scale", ".comfy_quant")

            for key in tqdm(all_keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                if state_dict_key_filter is not None and not state_dict_key_filter(key):
                    continue

                # Skip auxiliary scale keys — they are loaded alongside their .weight
                if key.endswith(skip_suffixes):
                    continue

                value = f.get_tensor(key)

                # For uint8 .weight tensors with nvfp4 scales, also load the scales
                if value.dtype == torch.uint8 and key.endswith(".weight"):
                    prefix = key[:-len(".weight")]
                    ws_key = prefix + ".weight_scale"
                    ws2_key = prefix + ".weight_scale_2"
                    if ws_key in ws_keys and ws2_key in ws2_keys:
                        # Load scales and store with buffer-compatible names
                        block_scale = f.get_tensor(ws_key)
                        tensor_scale = f.get_tensor(ws2_key)

                        if move_to_device and target_device is not None:
                            value = value.to(target_device)
                            block_scale = block_scale.to(target_device)
                            tensor_scale = tensor_scale.to(target_device)

                        state_dict[key] = value
                        state_dict[prefix + ".nvfp4_block_scale"] = block_scale
                        state_dict[prefix + ".nvfp4_tensor_scale"] = tensor_scale
                        continue

                # Non-quantized tensor — load normally
                if move_to_device and target_device is not None:
                    value = value.to(target_device)
                state_dict[key] = value

    return state_dict


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------

def is_nvfp4_module(module: nn.Module) -> bool:
    """Check whether *module* has been NVFP4-quantized."""
    return getattr(module, "_nvfp4_quantized", False)
