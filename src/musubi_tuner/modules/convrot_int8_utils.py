"""ConvRot INT8 optimization for frozen base weights during LoRA training.

Mirrors the structure of fp8_optimization_utils.py: quantize target Linear weights at
load time, store the INT8 weight (in the rotated basis) under the original ``.weight``
key and the per-channel scale under a sibling ``.scale_weight`` key, then monkey-patch
the matching ``nn.Linear`` modules' ``forward``.

Unlike the fp8 path, the forward goes through a custom ``torch.autograd.Function``:
the fused Triton kernel (rotation + dynamic row-wise INT8 quantization + INT8 GEMM
with dequantization epilogue) has no autograd support, and the base weight is frozen
so only grad_x is needed in backward: grad_x = rotate(g @ W_rot).

Keeping the module an ``nn.Linear`` (patched forward, INT8 ``.weight``) is load-bearing:
LoRA targets modules by class name "Linear", block swap streams ``module.weight.data``
of Linear-named modules, and compile exclusion also keys on the class name.
"""

import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from tqdm import tqdm

from musubi_tuner.modules.convrot_int8_kernels import (
    HAS_TRITON,
    _build_hadamard,
    _rotate_activation,
    int8_linear,
    int8_lora_linear,
    quantize_int8_convrot_weight,
)
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONVROT_GROUPSIZE = 256


def quantize_weight_convrot(key: str, tensor: torch.Tensor, groupsize: int = CONVROT_GROUPSIZE):
    """Quantize a single weight tensor with ConvRot INT8, or return None if not applicable.

    The quantization is always done with the eager implementation: it is deterministic
    and independent of triton availability, so the resulting state dict is identical
    across environments.

    Returns:
        (quantized int8 [N, K] in rotated basis, scale float32 [N, 1]), or None if the
        tensor is not a 2D weight or its in_features is not divisible by groupsize.
    """
    if tensor.ndim != 2:
        logger.info(f"Skipping ConvRot INT8 for {key}: not a 2D weight (ndim={tensor.ndim})")
        return None
    if tensor.shape[1] % groupsize != 0:
        logger.info(f"Skipping ConvRot INT8 for {key}: in_features {tensor.shape[1]} not divisible by {groupsize}")
        return None
    return quantize_int8_convrot_weight(tensor, groupsize)


class ConvRotInt8Quantizer:
    """Strategy object that streams safetensors files and quantizes target weights.

    Passed as ``quantizer`` to ``load_safetensors_with_lora_and_fp8``; carries its own
    streaming loader so the fp8 path stays untouched.
    """

    def __init__(
        self,
        target_layer_keys: Optional[List[str]] = None,
        exclude_layer_keys: Optional[List[str]] = None,
        groupsize: int = CONVROT_GROUPSIZE,
    ):
        self.target_layer_keys = target_layer_keys
        self.exclude_layer_keys = exclude_layer_keys
        self.groupsize = groupsize

    def is_target_key(self, key: str) -> bool:
        is_target = (self.target_layer_keys is None or any(pattern in key for pattern in self.target_layer_keys)) and key.endswith(
            ".weight"
        )
        is_excluded = self.exclude_layer_keys is not None and any(pattern in key for pattern in self.exclude_layer_keys)
        return is_target and not is_excluded

    def load_and_quantize(
        self,
        model_files: List[str],
        calc_device: Union[str, torch.device, None],
        move_to_device: bool = False,
        weight_hook: Optional[callable] = None,
        disable_numpy_memmap: bool = False,
        weight_transform_hooks: Optional[WeightTransformHooks] = None,
        placement_fn=None,
    ) -> dict:
        """Load state dict from safetensors files, quantizing target weights to ConvRot INT8.

        Same streaming contract as load_safetensors_with_fp8_optimization: the LoRA merge
        weight_hook runs on the raw (bf16) weight before quantization.
        """
        optimized_count = 0
        state_dict = {}
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
                f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f

                keys = f.keys()
                for key in tqdm(keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                    value = f.get_tensor(key)
                    original_device = value.device  # usually cpu

                    if weight_hook is not None:
                        value = weight_hook(key, value, keep_on_calc_device=(calc_device is not None))

                    if not self.is_target_key(key):
                        target_device = calc_device if (calc_device is not None and move_to_device) else original_device
                        if placement_fn is not None:
                            target_device = placement_fn(key, target_device)
                        state_dict[key] = value.to(target_device)
                        continue

                    if calc_device is not None:
                        value = value.to(calc_device)

                    if value.dtype.itemsize == 1:
                        raise ValueError(
                            f"Layer {key} is already in {value.dtype} format. Loading pre-quantized weights with"
                            " --convrot_int8 is not supported yet. Please use fp16/bf16/float32 model weights."
                            + f" / レイヤー {key} は既に{value.dtype}形式です。事前量子化済み重みの --convrot_int8 での"
                            "読み込みは未対応です。FP16/BF16/Float32のモデル重みを使用してください。"
                        )

                    result = quantize_weight_convrot(key, value, self.groupsize)
                    if result is None:
                        # leave the layer unquantized (bf16)
                        if not move_to_device:
                            value = value.to(original_device)
                        if placement_fn is not None:
                            value = value.to(placement_fn(key, value.device))
                        state_dict[key] = value
                        continue
                    quantized_weight, scale_tensor = result

                    scale_key = key.replace(".weight", ".scale_weight")
                    assert key != scale_key, "weight key and scale key must be different"

                    if not move_to_device:
                        quantized_weight = quantized_weight.to(original_device)
                    if placement_fn is not None:
                        quantized_weight = quantized_weight.to(placement_fn(key, quantized_weight.device))

                    # scale stays float32 [N, 1]: the Triton epilogue and the backward want fp32,
                    # and the shape maps 1:1 to ComfyUI's `weight_scale`
                    state_dict[key] = quantized_weight
                    state_dict[scale_key] = scale_tensor.to(device=quantized_weight.device)

                    optimized_count += 1

                    if calc_device is not None and optimized_count % 10 == 0:
                        clean_memory_on_device(calc_device)

        logger.info(f"Number of ConvRot INT8 optimized Linear layers: {optimized_count}")
        return state_dict


def _unrotated_weight(wq: torch.Tensor, w_scale: torch.Tensor, groupsize: int, dtype: torch.dtype) -> torch.Tensor:
    """Recover the ordinary weight from its rotated INT8 form.

    The rotation is orthogonal, so ``W_rot @ H`` is ``W`` again, and the
    quantization error the rotation bought was already fixed when the weight was
    stored. Undoing it costs a matmul against the weight rather than against the
    activations, and leaves an ordinary BF16 matrix for the vendor GEMM.
    """
    h = _build_hadamard(groupsize, device=wq.device, dtype=dtype)
    dequantized = wq.to(dtype) * w_scale.reshape(-1, 1).to(dtype)
    return _rotate_activation(dequantized, h, groupsize)


class ConvRotInt8LinearFn(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, wq, w_scale, bias, groupsize, bwd_mode, fwd_mode="int8"):
        # x: [..., K] bf16/fp16, wq: [N, K] int8 (rotated basis), w_scale: [N, 1] fp32
        # F.linear casts its inputs to the autocast dtype under autocast; the fused kernel
        # bypasses F.linear, so replicate that here. In K2 the fp32 modulation adds promote
        # the activations to fp32, and downstream flash-attn only accepts fp16/bf16.
        if torch.is_autocast_enabled(x.device.type):
            cast_dtype = torch.get_autocast_dtype(x.device.type)
            x = x.to(cast_dtype)
            if bias is not None:
                bias = bias.to(cast_dtype)
        if fwd_mode == "bf16":
            out = F.linear(x, _unrotated_weight(wq, w_scale, groupsize, x.dtype), bias)
        elif HAS_TRITON and x.is_cuda:
            out = int8_linear(x, wq, w_scale.reshape(-1), bias, x.dtype, True, groupsize)
        else:
            # eager fallback: rotation + transient dequantized matmul, no activation quantization
            h = _build_hadamard(groupsize, device=x.device, dtype=x.dtype)
            x_rot = _rotate_activation(x, h, groupsize)
            w_rot = wq.to(x.dtype) * w_scale.reshape(-1, 1).to(x.dtype)
            out = F.linear(x_rot, w_rot, bias)
        # wq/w_scale are live buffers, so saving them adds no activation memory; x is not
        # saved (base is frozen, no grad_weight needed)
        ctx.save_for_backward(wq, w_scale)
        ctx.groupsize = groupsize
        ctx.bwd_mode = bwd_mode
        ctx.fwd_mode = fwd_mode
        ctx.bias_needs_grad = bias is not None and bias.requires_grad
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        wq, w_scale = ctx.saved_tensors
        gs = ctx.groupsize
        g2d = grad_out.reshape(-1, grad_out.shape[-1])  # [M, N]

        grad_x = None
        if ctx.needs_input_grad[0]:
            if ctx.fwd_mode == "bf16":
                # The un-rotated weight is the ordinary W, so the gradient needs no
                # rotation of its own: the saving applies to both directions.
                weight = _unrotated_weight(wq, w_scale, gs, grad_out.dtype)
                grad_x = (g2d @ weight).reshape(*grad_out.shape[:-1], wq.shape[1])
            else:
                # grad_x = g @ W = g @ (W_rot R) = rotate(g @ W_rot), R = block-diag Hadamard
                if ctx.bwd_mode == "int8":
                    # fold per-channel weight scale into g, then reuse the fused Triton GEMM
                    # (row-wise quant of g + int8 GEMM + dequant epilogue in one pipeline).
                    # transient int8 transpose of wq: [K, N], ~1 byte/param, freed after mm
                    g_scaled = g2d * w_scale.reshape(1, -1).to(g2d.dtype)
                    one = torch.ones(1, device=g2d.device, dtype=torch.float32)
                    gx_rot = int8_linear(g_scaled, wq.t().contiguous(), one, None, grad_out.dtype, False, gs)
                else:
                    # transient bf16 dequant of the rotated weight (stays in rotated basis)
                    w_rot = wq.to(grad_out.dtype) * w_scale.reshape(-1, 1).to(grad_out.dtype)
                    gx_rot = g2d @ w_rot  # [M, K]
                h = _build_hadamard(gs, device=gx_rot.device, dtype=gx_rot.dtype)
                grad_x = _rotate_activation(gx_rot, h, gs).reshape(*grad_out.shape[:-1], wq.shape[1])

        grad_bias = g2d.sum(dim=0) if ctx.bias_needs_grad else None
        return grad_x, None, None, grad_bias, None, None, None


class ConvRotInt8LoRAFn(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, wq, w_scale, bias, groupsize, down, up, lora_scale):
        if torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))
        h = _build_hadamard(int(groupsize), device=x.device, dtype=x.dtype)
        rotated = _rotate_activation(x, h, int(groupsize))
        down_compute = down.to(x.dtype)
        up_compute = up.to(x.dtype)
        down_output = F.linear(x, down_compute)
        scaled_up = up_compute.t() * lora_scale
        output = int8_lora_linear(rotated, wq, w_scale, down_output, scaled_up, x.dtype)
        if bias is not None:
            output = output + bias.to(output.dtype)
        ctx.save_for_backward(x, wq, w_scale, down, up, down_output)
        ctx.groupsize = int(groupsize)
        ctx.lora_scale = lora_scale
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, wq, w_scale, down, up, down_output = ctx.saved_tensors
        grad2d = grad_output.reshape(-1, grad_output.shape[-1])
        scale = ctx.lora_scale
        grad_down_output = F.linear(grad2d, up.to(grad2d.dtype).t()) * scale
        h = _build_hadamard(ctx.groupsize, device=grad_output.device, dtype=grad_output.dtype)
        rotated_down = _rotate_activation(down.to(grad_output.dtype), h, ctx.groupsize)
        base_input = grad2d * w_scale.reshape(1, -1).to(grad2d.dtype)
        combined_rotated = int8_lora_linear(
            base_input,
            wq.t().contiguous(),
            torch.ones(wq.shape[1], device=wq.device, dtype=torch.float32),
            grad_down_output,
            rotated_down,
            grad_output.dtype,
        )
        grad_x = _rotate_activation(combined_rotated, h, ctx.groupsize).reshape_as(x)
        x2d = x.reshape(-1, x.shape[-1]).to(grad_down_output.dtype)
        grad_down = (grad_down_output.t() @ x2d).to(down.dtype)
        grad_up = ((grad2d.t() @ down_output.to(grad2d.dtype)) * scale).to(up.dtype)
        grad_bias = grad2d.sum(0) if ctx.needs_input_grad[3] else None
        return grad_x, None, None, grad_bias, None, grad_down, grad_up, None


def convrot_int8_lora_forward(module, x, down, up, scale):
    return ConvRotInt8LoRAFn.apply(x, module.weight, module.scale_weight, module.bias, module._convrot_groupsize, down, up, scale)


def convrot_int8_linear_forward_patch(self: nn.Linear, x):
    return ConvRotInt8LinearFn.apply(
        x,
        self.weight,
        self.scale_weight,
        self.bias,
        self._convrot_groupsize,
        self._convrot_bwd_mode,
        getattr(self, "_convrot_fwd_mode", "int8"),
    )


def apply_convrot_int8_monkey_patch(
    model,
    optimized_state_dict,
    bwd_mode: str = "bf16",
    groupsize: int = CONVROT_GROUPSIZE,
    fwd_mode: str = "int8",
):
    """
    Apply monkey patching to a model using a ConvRot INT8 optimized state dict.

    Args:
        model (nn.Module): Model instance to patch
        optimized_state_dict (dict): state dict produced by ConvRotInt8Quantizer
        bwd_mode (str): "bf16" (transient dequant, safe default) or "int8" (quantizes
            gradients, faster, requires triton)
        groupsize (int): ConvRot group size
        fwd_mode (str): "int8" (rotate the activations and run the fused INT8 kernel,
            the default) or "bf16" (undo the rotation on the weight instead and hand
            the vendor GEMM an ordinary matrix; same stored weights, same arithmetic
            result, so it trades quantized compute for a better-tuned kernel)

    Returns:
        nn.Module: The patched model (same instance, modified in-place)
    """
    if bwd_mode not in ("bf16", "int8"):
        raise ValueError(f"Unsupported ConvRot INT8 backward mode: {bwd_mode}")
    if fwd_mode not in ("bf16", "int8"):
        raise ValueError(f"Unsupported ConvRot INT8 forward mode: {fwd_mode}")
    if fwd_mode == "bf16" and bwd_mode == "int8":
        raise ValueError("ConvRot INT8 forward mode 'bf16' has no rotated activations for an INT8 backward")
    if bwd_mode == "int8" and not HAS_TRITON:
        raise ValueError("ConvRot INT8 backward mode 'int8' requires triton. Install triton (triton-windows on Windows).")
    if not HAS_TRITON and fwd_mode != "bf16":
        logger.warning(
            "triton is not available: ConvRot INT8 falls back to transient dequantization in forward."
            " Weight VRAM is still reduced, but there is no speedup. Install triton (triton-windows on Windows)"
            " for the fused INT8 kernels."
        )

    scale_keys = [k for k in optimized_state_dict.keys() if k.endswith(".scale_weight")]

    patched_module_paths = set()
    scale_shape_info = {}
    for scale_key in scale_keys:
        module_path = scale_key.rsplit(".scale_weight", 1)[0]
        patched_module_paths.add(module_path)
        scale_shape_info[module_path] = optimized_state_dict[scale_key].shape

    patched_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in patched_module_paths:
            # register the scale_weight as a buffer to load the state_dict
            module.register_buffer("scale_weight", torch.ones(scale_shape_info[name], dtype=torch.float32))
            module._convrot_groupsize = groupsize
            module._convrot_bwd_mode = bwd_mode
            module._convrot_fwd_mode = fwd_mode
            module._convrot_lora_fused = False

            def new_forward(self, x):
                return convrot_int8_linear_forward_patch(self, x)

            module.forward = new_forward.__get__(module, type(module))

            patched_count += 1

    logger.info(f"Number of ConvRot INT8 monkey-patched Linear layers: {patched_count}")
    return model


def enable_convrot_int8_lora_fusion(model) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Linear) and hasattr(module, "_convrot_groupsize"):
            module._convrot_lora_fused = True
            count += 1
    logger.info("Enabled fused ConvRot INT8 LoRA epilogues on %d Linear layers", count)
    return count
