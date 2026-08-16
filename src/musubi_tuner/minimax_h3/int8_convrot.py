from __future__ import annotations

import json
import logging
import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

logger = logging.getLogger(__name__)

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def parse_comfy_quant_marker(marker: torch.Tensor) -> dict:
    try:
        payload = bytes(marker.detach().cpu().flatten().tolist()).decode("utf-8")
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid Comfy quantization marker") from error
    if not isinstance(value, dict):
        raise ValueError("Comfy quantization marker must contain a JSON object")
    return value


def _is_power_of_four(value: int) -> bool:
    if value < 4:
        return False
    while value > 1 and value % 4 == 0:
        value //= 4
    return value == 1


def _hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not _is_power_of_four(size):
        raise ValueError(f"INT8 ConvRot group size must be a power of four, got {size}")
    key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    base = torch.tensor(
        ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1)),
        device=device,
        dtype=dtype,
    )
    matrix = base
    current = 4
    while current < size:
        matrix = torch.kron(matrix, base)
        current *= 4
    matrix = matrix / math.sqrt(size)
    _HADAMARD_CACHE[key] = matrix
    return matrix


def rotate_activation(value: torch.Tensor, group_size: int) -> torch.Tensor:
    if value.shape[-1] % group_size:
        raise ValueError(f"INT8 ConvRot group size {group_size} does not divide {value.shape[-1]} input features")
    matrix = _hadamard(group_size, value.device, value.dtype)
    grouped = value.reshape(-1, value.shape[-1] // group_size, group_size)
    return torch.matmul(grouped, matrix).reshape(value.shape)


def _quantize_rows(value: torch.Tensor, *, owns_buffer: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    # The rescale is done in place to avoid a second full-size temporary, so the buffer
    # must never alias a tensor the caller still owns: ``.float()`` is an identity view
    # when the input is already float32, which would silently corrupt it.
    float_value = value.float() if owns_buffer else value.to(torch.float32, copy=True)
    scale = (float_value.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-30)
    quantized = float_value.div_(scale).round_().clamp_(-127, 127).to(torch.int8)
    return quantized, scale


def _int_mm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "_int_mm"):
        raise RuntimeError("INT8 ConvRot requires torch._int_mm on CUDA")
    rows = left.shape[0]
    # torch._int_mm requires strictly more than 16 rows, so 16 itself must be
    # padded as well; zero rows contribute nothing and are sliced back off.
    if rows > 16:
        return torch._int_mm(left.contiguous(), right)
    padded = F.pad(left, (0, 0, 0, 17 - rows))
    return torch._int_mm(padded.contiguous(), right)[:rows]


def _unrotated_weight(weight: torch.Tensor, scale: torch.Tensor, group_size: int, dtype: torch.dtype) -> torch.Tensor:
    """Recover the ordinary weight from its stored rotated INT8 form.

    The block-diagonal Hadamard is symmetric and orthogonal, so rotating the
    dequantized rotated weight once more undoes the offline rotation. This is the same
    identity the online ConvRot path uses for its BF16 forward.
    """
    dequantized = weight.to(dtype) * scale.reshape(-1, 1).to(dtype)
    return rotate_activation(dequantized, group_size)


def _int8_available(tensor: torch.Tensor) -> bool:
    return bool(tensor.is_cuda) and hasattr(torch, "_int_mm")


class _Int8ConvRotFunction(torch.autograd.Function):
    """Pre-quantized ConvRot INT8 linear with the same forward/backward modes as the online path.

    ``fwd_mode``:
      - ``int8``: rotate the activations, quantize them row-wise and run ``torch._int_mm``
        (transient BF16 dequantization when the INT8 matmul is unavailable, e.g. on CPU).
      - ``bf16``: undo the rotation on the weight instead and hand ``F.linear`` an
        ordinary matrix. Same stored weights, same arithmetic, unquantized activations.

    ``bwd_mode`` (only meaningful with ``fwd_mode='int8'``):
      - ``bf16``: transient dequantization of the rotated weight, ``grad_x = rotate(g @ W_rot)``.
      - ``int8``: fold the per-channel weight scale into ``g``, quantize its rows and run
        the INT8 matmul before rotating.
    """

    @staticmethod
    def forward(ctx, inputs, weight, scale, bias, group_size, fwd_mode, bwd_mode):
        group_size = int(group_size)
        if fwd_mode == "bf16":
            dense_weight = _unrotated_weight(weight, scale, group_size, inputs.dtype)
            output = F.linear(inputs, dense_weight, bias.to(inputs.dtype) if bias is not None else None)
        elif _int8_available(inputs):
            rotated = rotate_activation(inputs, group_size)
            flat = rotated.reshape(-1, rotated.shape[-1])
            quantized, input_scale = _quantize_rows(flat, owns_buffer=True)
            accumulated = _int_mm(quantized, weight.t()).float()
            accumulated.mul_(input_scale).mul_(scale.reshape(1, -1))
            if bias is not None:
                accumulated.add_(bias.float())
            output = accumulated.to(inputs.dtype).reshape(*inputs.shape[:-1], weight.shape[0])
        else:
            rotated = rotate_activation(inputs, group_size)
            dense_weight = (weight.float() * scale.float()).to(inputs.dtype)
            output = F.linear(rotated, dense_weight, bias.to(inputs.dtype) if bias is not None else None)
        ctx.save_for_backward(weight, scale)
        ctx.input_dtype = inputs.dtype
        ctx.input_shape = inputs.shape
        ctx.group_size = group_size
        ctx.fwd_mode = fwd_mode
        ctx.bwd_mode = bwd_mode
        return output

    @staticmethod
    def backward(ctx, grad_output):
        weight, scale = ctx.saved_tensors
        folded = grad_output.reshape(-1, grad_output.shape[-1])
        if ctx.fwd_mode == "bf16":
            # The un-rotated weight is the ordinary W, so the gradient needs no rotation
            # of its own: the saving applies to both directions.
            dense_weight = _unrotated_weight(weight, scale, ctx.group_size, torch.float32)
            grad_input = folded.float() @ dense_weight
            return grad_input.to(ctx.input_dtype).reshape(ctx.input_shape), None, None, None, None, None, None
        if ctx.bwd_mode == "int8" and _int8_available(folded):
            # Copy before the in-place fold: ``folded`` is a view of the caller's gradient,
            # and .float() on an fp32 gradient would hand back that very tensor.
            scaled = folded.to(torch.float32, copy=True)
            scaled.mul_(scale.reshape(1, -1))
            quantized, grad_scale = _quantize_rows(scaled, owns_buffer=True)
            grad_input = _int_mm(quantized, weight).float()
            grad_input.mul_(grad_scale)
        else:
            dense_weight = weight.float() * scale.float()
            grad_input = folded.float() @ dense_weight
        grad_input = rotate_activation(grad_input.to(ctx.input_dtype), ctx.group_size)
        return grad_input.reshape(ctx.input_shape), None, None, None, None, None, None


def _int8_linear_forward(module: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
    # ``_convrot_groupsize`` is a plain Python int stashed at enable time; reading the
    # registered buffer instead costs a device sync on every patched Linear per forward.
    group_size = getattr(module, "_convrot_groupsize", None)
    if group_size is None:
        group_size = int(module.int8_convrot_groupsize.item())
    return _Int8ConvRotFunction.apply(
        inputs,
        module.weight,
        module.scale_weight,
        module.bias,
        group_size,
        getattr(module, "_convrot_fwd_mode", "int8"),
        getattr(module, "_convrot_bwd_mode", "bf16"),
    )


def load_comfy_int8_convrot_state_dict(
    checkpoint_path,
    *,
    device: torch.device,
    placement_fn=None,
) -> tuple[dict[str, torch.Tensor], int]:
    state_dict: dict[str, torch.Tensor] = {}
    marker_config: dict[str, dict] = {}
    with MemoryEfficientSafeOpen(str(checkpoint_path)) as handle:
        keys = list(handle.keys())
        for key in keys:
            if key.endswith(".comfy_quant"):
                base = key[: -len(".comfy_quant")]
                config = parse_comfy_quant_marker(handle.get_tensor(key))
                if config.get("format") != "int8_tensorwise" or config.get("convrot") is not True:
                    raise ValueError(f"{key} is not an INT8 ConvRot marker")
                group_size = int(config.get("convrot_groupsize", 0))
                if not _is_power_of_four(group_size):
                    raise ValueError(f"{key} contains invalid ConvRot group size {group_size}")
                marker_config[base] = config

        scale_bases = {key[: -len(".weight_scale")] for key in keys if key.endswith(".weight_scale")}
        weight_bases = {
            key[: -len(".weight")] for key in keys if key.endswith(".weight") and handle.tensor_dtype(key) is torch.int8
        }
        if not marker_config or set(marker_config) != scale_bases or set(marker_config) != weight_bases:
            raise ValueError(
                "INT8 ConvRot checkpoint has inconsistent marker, scale, and quantized-weight sets: "
                f"markers={len(marker_config)}, scales={len(scale_bases)}, weights={len(weight_bases)}"
            )

        for key in keys:
            if key.endswith(".comfy_quant"):
                continue
            value = handle.get_tensor(key)
            if key.endswith(".weight_scale"):
                base = key[: -len(".weight_scale")]
                scale = value.float().reshape(-1, 1)
                scale_key = f"{base}.scale_weight"
                target_device = placement_fn(scale_key, device) if placement_fn is not None else device
                state_dict[scale_key] = scale.to(target_device)
                state_dict[f"{base}.int8_convrot_groupsize"] = torch.tensor(
                    int(marker_config[base]["convrot_groupsize"]), device=target_device, dtype=torch.int32
                )
            else:
                target_device = placement_fn(key, device) if placement_fn is not None else device
                state_dict[key] = value.to(target_device)
    return state_dict, len(marker_config)


def prepare_int8_convrot_modules(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> int:
    scale_shapes = {
        key[: -len(".scale_weight")]: tuple(value.shape) for key, value in state_dict.items() if key.endswith(".scale_weight")
    }
    registered = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name not in scale_shapes:
            continue
        group_key = f"{name}.int8_convrot_groupsize"
        if group_key not in state_dict:
            raise ValueError(f"INT8 ConvRot state is missing {group_key}")
        module.weight.requires_grad_(False)
        if module.bias is not None:
            module.bias.requires_grad_(False)
        module.register_buffer("scale_weight", torch.ones(scale_shapes[name], dtype=torch.float32))
        module.register_buffer("int8_convrot_groupsize", torch.zeros((), dtype=torch.int32))
        registered += 1
    if registered != len(scale_shapes):
        raise ValueError(f"INT8 ConvRot checkpoint targets {len(scale_shapes)} linears, but the model exposes {registered}")
    return registered


def enable_int8_convrot(model: nn.Module, fwd_mode: str = "int8", bwd_mode: str = "bf16") -> int:
    if fwd_mode not in ("bf16", "int8"):
        raise ValueError(f"Unsupported ConvRot INT8 forward mode: {fwd_mode}")
    if bwd_mode not in ("bf16", "int8"):
        raise ValueError(f"Unsupported ConvRot INT8 backward mode: {bwd_mode}")
    if fwd_mode == "bf16" and bwd_mode == "int8":
        raise ValueError("ConvRot INT8 forward mode 'bf16' has no rotated activations for an INT8 backward")
    patched = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear) or not hasattr(module, "scale_weight"):
            continue
        if module.weight.dtype is not torch.int8:
            raise TypeError("INT8 ConvRot scale is attached to a non-INT8 Linear weight")
        group_size = int(module.int8_convrot_groupsize.item())
        if module.weight.shape[1] % group_size:
            raise ValueError(f"INT8 ConvRot group size {group_size} does not divide {module.weight.shape[1]} input features")
        # Same attribute names as the online monkey patch: the LoRA fusion selector, the
        # fused LoRA epilogue and the block-swap guard then see one kind of ConvRot module.
        module._convrot_groupsize = group_size
        module._convrot_fwd_mode = fwd_mode
        module._convrot_bwd_mode = bwd_mode
        module._convrot_lora_fused = False
        module.forward = MethodType(_int8_linear_forward, module)
        patched += 1
    logger.info(
        "Enabled H3 INT8 ConvRot runtime on %d Linear layers (forward %s, backward %s)",
        patched,
        fwd_mode,
        bwd_mode,
    )
    return patched
