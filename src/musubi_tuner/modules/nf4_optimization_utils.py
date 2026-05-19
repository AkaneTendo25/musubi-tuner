"""NF4 (4-bit NormalFloat) quantization utilities for base model compression.

Parallel to fp8_optimization_utils.py — provides quantize/dequantize, monkey-patching,
and safetensors loading with NF4 quantization.

The NF4 codebook is the set of 16 values that are optimal for normally-distributed
weights (as derived in the QLoRA paper).  Weights are stored as packed uint8 (two
4-bit indices per byte) with per-block absmax scales.
"""

import os
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from tqdm import tqdm

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NF4 codebook (16 values, symmetric around 0, optimal for N(0,1) weights)
# ---------------------------------------------------------------------------
NF4_CODEBOOK = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)

DEFAULT_NF4_BLOCK_SIZE = 32


# ---------------------------------------------------------------------------
# Pack / unpack helpers
# ---------------------------------------------------------------------------

def pack_uint4(indices: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit indices into uint8.  ``indices`` must have even length."""
    assert indices.numel() % 2 == 0, "Number of elements must be even for uint4 packing"
    flat = indices.view(-1)
    high = flat[0::2].to(torch.uint8)
    low = flat[1::2].to(torch.uint8)
    return (high << 4) | low


def unpack_uint4(packed: torch.Tensor, num_elements: int) -> torch.Tensor:
    """Unpack uint8 → pairs of 4-bit indices.  Returns ``[num_elements]`` uint8."""
    flat = packed.view(-1)
    high = (flat >> 4) & 0x0F
    low = flat & 0x0F
    # interleave: [h0, l0, h1, l1, ...]
    out = torch.stack([high, low], dim=1).view(-1)
    return out[:num_elements].to(torch.uint8)


# ---------------------------------------------------------------------------
# Quantize / dequantize
# ---------------------------------------------------------------------------

def quantize_nf4_block(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_NF4_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight tensor to NF4 with per-block absmax scaling.

    Args:
        tensor: ``[out_features, in_features]`` float weight.
        block_size: number of elements per quantization block.

    Returns:
        packed_weight: ``[out_features, in_features // 2]`` uint8 (two indices per byte).
        scale: ``[out_features, num_blocks, 1]`` float32 absmax per block.
    """
    assert tensor.ndim == 2, f"Expected 2-D tensor, got {tensor.ndim}-D"
    out_features, in_features = tensor.shape
    if in_features % block_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by block_size ({block_size})"
        )
    num_blocks = in_features // block_size

    # Reshape to [out, num_blocks, block_size]
    t = tensor.float().contiguous().view(out_features, num_blocks, block_size)

    # Per-block absmax scale
    scale = t.abs().amax(dim=2, keepdim=True).clamp_(min=1e-8)  # [out, num_blocks, 1]

    # Normalize to [-1, 1]
    normalized = t / scale  # [out, num_blocks, block_size]

    # Find nearest codebook entry using bucketize (O(N) memory, not O(N*16)).
    # The codebook is sorted, so we compute midpoints between consecutive values
    # and use bucketize to find which bin each normalized value falls into.
    codebook = NF4_CODEBOOK.to(normalized.device)  # [16]
    # Midpoints between consecutive codebook values → 15 boundaries
    midpoints = (codebook[:-1] + codebook[1:]) / 2.0  # [15]
    flat_norm = normalized.reshape(-1)  # [N]
    indices = torch.bucketize(flat_norm, midpoints)  # [N] int64, values in [0, 15]

    # Reshape indices back and pack
    indices = indices.view(out_features, in_features)  # [out, in]
    packed = pack_uint4(indices).view(out_features, in_features // 2)

    return packed, scale.to(torch.float32)


def dequantize_nf4_block(
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    out_features: int,
    in_features: int,
    block_size: int = DEFAULT_NF4_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize NF4-packed weight back to a dense float tensor.

    Args:
        packed_weight: ``[out_features, in_features // 2]`` uint8.
        scale: ``[out_features, num_blocks, 1]`` float32.
        out_features, in_features: original weight shape.
        block_size: must match quantization block size.
        dtype: output dtype (e.g. bfloat16).

    Returns:
        ``[out_features, in_features]`` tensor in *dtype*.
    """
    num_blocks = in_features // block_size
    codebook = NF4_CODEBOOK.to(scale.device)  # [16]

    indices = unpack_uint4(packed_weight, out_features * in_features)
    indices = indices.to(scale.device).long()
    values = codebook[indices]  # float32

    values = values.view(out_features, num_blocks, block_size)
    values = values * scale  # broadcast [out, num_blocks, 1]
    values = values.view(out_features, in_features)
    return values.to(dtype)


# ---------------------------------------------------------------------------
# State-dict-level quantization (mirrors optimize_state_dict_with_fp8)
# ---------------------------------------------------------------------------

def optimize_state_dict_with_nf4(
    state_dict: dict,
    calc_device: Union[str, torch.device],
    target_layer_keys: Optional[List[str]] = None,
    exclude_layer_keys: Optional[List[str]] = None,
    block_size: int = DEFAULT_NF4_BLOCK_SIZE,
    move_to_device: bool = False,
) -> dict:
    """Quantize target Linear weights in *state_dict* to NF4 in-place.

    For each target ``.weight`` key the dict is updated with:
    * original key → packed uint8 weight ``[out, in//2]``
    * ``.scale_weight`` → per-block scale ``[out, num_blocks, 1]``
    * ``.nf4_shape`` → ``torch.tensor([out, in])`` (original shape)
    """
    optimized_count = 0

    target_keys_list = []
    for key in list(state_dict.keys()):
        is_target = (
            target_layer_keys is None or any(p in key for p in target_layer_keys)
        ) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(
            p in key for p in exclude_layer_keys
        )
        if is_target and not is_excluded and isinstance(state_dict[key], torch.Tensor):
            target_keys_list.append(key)

    for key in tqdm(target_keys_list, desc="NF4 quantizing"):
        value = state_dict[key]
        if value.ndim != 2:
            continue  # skip non-2D (e.g. bias, norm)
        out_features, in_features = value.shape
        if in_features % block_size != 0:
            logger.warning(
                "Skipping NF4 for %s: in_features=%d not divisible by block_size=%d",
                key,
                in_features,
                block_size,
            )
            continue

        original_device = value.device
        original_dtype = value.dtype
        if calc_device is not None:
            value = value.to(calc_device)

        packed, scale = quantize_nf4_block(value, block_size)

        target_device = calc_device if (calc_device is not None and move_to_device) else original_device
        state_dict[key] = packed.to(target_device)
        state_dict[key.replace(".weight", ".scale_weight")] = scale.to(
            dtype=original_dtype, device=target_device
        )
        state_dict[key.replace(".weight", ".nf4_shape")] = torch.tensor(
            [out_features, in_features], dtype=torch.int64
        )

        optimized_count += 1
        if calc_device is not None and optimized_count % 10 == 0:
            clean_memory_on_device(calc_device)

    logger.info(f"Number of NF4-quantized Linear layers: {optimized_count}")
    return state_dict


def load_safetensors_with_nf4_optimization(
    model_files: List[str],
    calc_device: Union[str, torch.device],
    target_layer_keys: Optional[List[str]] = None,
    exclude_layer_keys: Optional[List[str]] = None,
    block_size: int = DEFAULT_NF4_BLOCK_SIZE,
    move_to_device: bool = False,
    weight_hook=None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
    key_filter: Optional[Callable[[str], bool]] = None,
) -> dict:
    """Load safetensors and quantize target layers to NF4.

    Mirrors ``load_safetensors_with_fp8_optimization``.
    """

    def is_target_key(key):
        is_target = (
            target_layer_keys is None or any(p in key for p in target_layer_keys)
        ) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(
            p in key for p in exclude_layer_keys
        )
        return is_target and not is_excluded

    optimized_count = 0
    state_dict: dict = {}

    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
            f = (
                TensorWeightAdapter(weight_transform_hooks, original_f)
                if weight_transform_hooks is not None
                else original_f
            )
            keys = [key for key in f.keys() if key_filter is None or key_filter(key)]
            for key in tqdm(keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                value = f.get_tensor(key)
                original_device = value.device

                if weight_hook is not None:
                    value = weight_hook(key, value, keep_on_calc_device=(calc_device is not None))

                if not is_target_key(key) or value.ndim != 2:
                    target_device = calc_device if (calc_device is not None and move_to_device) else original_device
                    state_dict[key] = value.to(target_device)
                    continue

                out_features, in_features = value.shape
                if in_features % block_size != 0:
                    logger.warning(
                        "Skipping NF4 for %s: in_features=%d not divisible by block_size=%d",
                        key,
                        in_features,
                        block_size,
                    )
                    target_device = calc_device if (calc_device is not None and move_to_device) else original_device
                    state_dict[key] = value.to(target_device)
                    continue

                if calc_device is not None:
                    value = value.to(calc_device)

                original_dtype = value.dtype
                packed, scale = quantize_nf4_block(value, block_size)

                target_device = calc_device if (calc_device is not None and move_to_device) else original_device
                state_dict[key] = packed.to(target_device)
                state_dict[key.replace(".weight", ".scale_weight")] = scale.to(
                    dtype=original_dtype, device=target_device
                )
                state_dict[key.replace(".weight", ".nf4_shape")] = torch.tensor(
                    [out_features, in_features], dtype=torch.int64
                )

                optimized_count += 1
                if calc_device is not None and optimized_count % 10 == 0:
                    clean_memory_on_device(calc_device)

    logger.info(f"Number of NF4-quantized Linear layers: {optimized_count}")
    return state_dict


# ---------------------------------------------------------------------------
# Monkey-patched forward
# ---------------------------------------------------------------------------

def nf4_linear_forward_patch(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement forward for NF4-quantized Linear layers."""
    w = dequantize_nf4_block(
        self.weight,
        self.scale_weight,
        self.nf4_out_features,
        self.nf4_in_features,
        self.nf4_block_size,
        self.scale_weight.dtype,
    )
    if hasattr(self, "awq_scales"):
        # Un-scale columns: divide by the AWQ scales that were multiplied before quantization
        w = w / self.awq_scales.unsqueeze(0)  # [1, in] broadcast over [out, in]
        w = w.to(self.scale_weight.dtype)
    return F.linear(x, w, self.bias)


def apply_nf4_monkey_patch(
    model: nn.Module,
    optimized_state_dict: dict,
    block_size: int = DEFAULT_NF4_BLOCK_SIZE,
    awq_scales: Optional[dict] = None,
) -> nn.Module:
    """Register NF4 buffers and replace forward on quantized Linears.

    Mirrors ``apply_fp8_monkey_patch``.

    Args:
        awq_scales: Optional dict mapping module_name -> scale tensor [in_features].
            If provided, each patched module gets an ``awq_scales`` buffer for
            un-scaling during the forward pass.
    """
    # Identify NF4-quantized modules from the state dict
    shape_keys = [k for k in optimized_state_dict if k.endswith(".nf4_shape")]
    patched_module_paths: set = set()
    shape_info: dict = {}
    scale_shape_info: dict = {}
    for shape_key in shape_keys:
        module_path = shape_key.rsplit(".nf4_shape", 1)[0]
        patched_module_paths.add(module_path)
        shape_tensor = optimized_state_dict[shape_key]
        shape_info[module_path] = (int(shape_tensor[0].item()), int(shape_tensor[1].item()))
        scale_key = module_path + ".scale_weight"
        scale_shape_info[module_path] = optimized_state_dict[scale_key].shape

    patched_count = 0
    for name, module in model.named_modules():
        if name not in patched_module_paths:
            continue
        if not isinstance(module, nn.Linear):
            continue

        out_f, in_f = shape_info[name]
        scale_shape = scale_shape_info[name]
        packed_shape = (out_f, in_f // 2)

        # Replace weight parameter with a buffer of packed shape so
        # load_state_dict(assign=True) doesn't complain about shape mismatch.
        weight_dtype = module.weight.dtype if module.weight.dtype != torch.float32 else torch.uint8
        del module.weight
        module.register_buffer("weight", torch.zeros(packed_shape, dtype=torch.uint8))

        # Register buffers so load_state_dict can assign them
        module.register_buffer("scale_weight", torch.ones(scale_shape, dtype=weight_dtype))
        module.register_buffer("nf4_shape", torch.zeros(2, dtype=torch.int64))

        # Store metadata as plain attributes
        module.nf4_out_features = out_f
        module.nf4_in_features = in_f
        module.nf4_block_size = block_size
        module._nf4_quantized = True

        # Register AWQ scales buffer if available
        if awq_scales is not None:
            # AWQ scales are keyed by weight key (e.g. "layer.weight")
            weight_key = name + ".weight"
            if weight_key in awq_scales:
                module.register_buffer(
                    "awq_scales",
                    awq_scales[weight_key].to(dtype=torch.float32),
                )

        # Replace forward
        def new_forward(self, x):
            return nf4_linear_forward_patch(self, x)

        module.forward = new_forward.__get__(module, type(module))
        patched_count += 1

    logger.info(f"Number of NF4 monkey-patched Linear layers: {patched_count}")
    return model


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------

def is_nf4_module(module: nn.Module) -> bool:
    """Check whether *module* has been NF4-quantized (attribute-based, not dtype)."""
    return getattr(module, "_nf4_quantized", False)
