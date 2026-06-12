# Shared tensor helpers for LoRA-family network modules.

import math
from typing import Any, Dict, List, Optional, Union

import torch


def _metadata_tensor_to_int(value: Optional[torch.Tensor], default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, torch.Tensor):
        return int(round(float(value.detach().cpu().item())))
    return int(round(float(value)))


def _metadata_tensor_to_bool(value: Optional[torch.Tensor], default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, torch.Tensor):
        return bool(int(round(float(value.detach().cpu().item()))))
    return bool(value)


def _metadata_tensor_to_float(value: Optional[torch.Tensor], default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _solve_oft_block_size(n_elements: int) -> int:
    return int(round((1.0 + math.sqrt(1.0 + 8.0 * float(n_elements))) / 2.0))


def _parse_bool_network_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_optional_float_network_arg(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    return float(value)


def _effective_weight_dtype(
    weight: torch.Tensor,
    scale_weight: Optional[torch.Tensor],
    dtype: Optional[torch.dtype],
) -> torch.dtype:
    if dtype is not None:
        return dtype
    if isinstance(scale_weight, torch.Tensor) and scale_weight.is_floating_point() and scale_weight.dtype.itemsize >= 2:
        return scale_weight.dtype
    if weight.is_floating_point() and weight.dtype.itemsize >= 2:
        return weight.dtype
    return torch.float32


def _get_scaled_weight(
    weight: torch.Tensor,
    scale_weight: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    detach: bool = False,
    non_blocking: bool = False,
) -> torch.Tensor:
    if detach:
        weight = weight.detach()
        scale_weight = scale_weight.detach()
    target_dtype = _effective_weight_dtype(weight, scale_weight, dtype)
    target_device = device if device is not None else weight.device
    dense_weight = weight.to(device=target_device, dtype=target_dtype, non_blocking=non_blocking)
    scale = scale_weight.to(device=target_device, dtype=target_dtype, non_blocking=non_blocking)
    if scale.ndim < 3:
        return dense_weight * scale

    out_features, num_blocks, _ = scale.shape
    dense_weight = dense_weight.contiguous().view(out_features, num_blocks, -1)
    dense_weight = dense_weight * scale
    return dense_weight.view(weight.shape)


def _get_effective_module_weight(
    module: torch.nn.Module,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    detach: bool = False,
    non_blocking: bool = False,
) -> torch.Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise AttributeError(f"{module.__class__.__name__} has no tensor weight")

    scale_weight = getattr(module, "scale_weight", None)
    if isinstance(scale_weight, torch.Tensor):
        if all(hasattr(module, attr) for attr in ("nf4_out_features", "nf4_in_features", "nf4_block_size")):
            from musubi_tuner.modules.nf4_optimization_utils import dequantize_nf4_block

            if detach:
                weight = weight.detach()
                scale_weight = scale_weight.detach()
            target_dtype = _effective_weight_dtype(weight, scale_weight, dtype)
            target_device = device if device is not None else weight.device
            dense_weight = dequantize_nf4_block(
                weight.to(device=target_device, non_blocking=non_blocking),
                scale_weight.to(device=target_device, non_blocking=non_blocking),
                int(getattr(module, "nf4_out_features")),
                int(getattr(module, "nf4_in_features")),
                int(getattr(module, "nf4_block_size")),
                scale_weight.dtype,
            )
            awq_scales = getattr(module, "awq_scales", None)
            if isinstance(awq_scales, torch.Tensor):
                awq_scales = awq_scales.to(device=dense_weight.device, dtype=dense_weight.dtype, non_blocking=non_blocking)
                dense_weight = dense_weight / awq_scales.unsqueeze(0)
            return dense_weight.to(dtype=target_dtype)

        return _get_scaled_weight(
            weight,
            scale_weight,
            dtype=dtype,
            device=device,
            detach=detach,
            non_blocking=non_blocking,
        )

    if detach:
        weight = weight.detach()
    target_dtype = dtype
    if target_dtype is None and (not weight.is_floating_point() or weight.dtype.itemsize < 2):
        target_dtype = torch.float32
    if target_dtype is None and device is None:
        return weight
    to_kwargs: Dict[str, Any] = {"non_blocking": non_blocking}
    if target_dtype is not None:
        to_kwargs["dtype"] = target_dtype
    if device is not None:
        to_kwargs["device"] = device
    return weight.to(**to_kwargs)


def _get_dora_compute_dtype(weight: torch.Tensor) -> torch.dtype:
    if not weight.is_floating_point() or weight.dtype.itemsize < 2:
        return torch.float32
    if weight.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return weight.dtype


def _get_lora_weight_from_tensors(down_weight: torch.Tensor, up_weight: torch.Tensor) -> torch.Tensor:
    if len(down_weight.size()) == 2:
        return up_weight @ down_weight
    if down_weight.size()[2:4] == (1, 1):
        return (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
    return torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)


def _get_split_lora_weight(split_dims: List[int], down_weights: List[torch.Tensor], up_weights: List[torch.Tensor]) -> torch.Tensor:
    total_dims = sum(split_dims)
    in_dim = down_weights[0].size(1)
    device = down_weights[0].device
    dtype = down_weights[0].dtype
    weight = torch.zeros((total_dims, in_dim), device=device, dtype=dtype)

    offset = 0
    for split_dim, down_weight, up_weight in zip(split_dims, down_weights, up_weights):
        weight[offset : offset + split_dim] = up_weight @ down_weight
        offset += split_dim

    return weight


def _get_linear_weight_norm_factored(
    base_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    compute_dtype = _get_dora_compute_dtype(base_weight)
    base_weight = base_weight.to(dtype=compute_dtype)
    down_weight = down_weight.to(device=base_weight.device, dtype=compute_dtype)
    up_weight = up_weight.to(device=base_weight.device, dtype=compute_dtype)

    w_norm_sq = (base_weight * base_weight).sum(dim=1)
    if float(scaling) == 0.0:
        return torch.sqrt(w_norm_sq.clamp_min_(0))

    u_term = base_weight @ down_weight.transpose(0, 1)
    gram = down_weight @ down_weight.transpose(0, 1)
    cross_term = (up_weight * u_term).sum(dim=1)
    ba_norm_sq = ((up_weight @ gram) * up_weight).sum(dim=1)
    norm_sq = w_norm_sq + (2.0 * float(scaling)) * cross_term + (float(scaling) * float(scaling)) * ba_norm_sq
    return torch.sqrt(norm_sq.clamp_min_(0))


def _get_conv_weight_norm_factored(
    base_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    out_channels = base_weight.shape[0]
    flat_weight = base_weight.reshape(out_channels, -1)
    flat_down = down_weight.reshape(down_weight.shape[0], -1)
    flat_up = up_weight.reshape(out_channels, -1)
    norms = _get_linear_weight_norm_factored(flat_weight, flat_down, flat_up, scaling)
    return norms.view((1, out_channels) + (1,) * max(0, base_weight.dim() - 2))


def _get_dense_weight_norm(base_weight: torch.Tensor, lora_weight: torch.Tensor, scaling: float) -> torch.Tensor:
    compute_dtype = _get_dora_compute_dtype(base_weight)
    total = base_weight.to(dtype=compute_dtype) + float(scaling) * lora_weight.to(device=base_weight.device, dtype=compute_dtype)
    if total.dim() == 2:
        return torch.linalg.vector_norm(total, dim=1)
    dims = tuple(range(1, total.dim()))
    return total.norm(p=2, dim=dims, keepdim=True).transpose(1, 0)


def _get_dora_weight_norm(
    base_weight: torch.Tensor,
    down_weight: Union[torch.Tensor, List[torch.Tensor]],
    up_weight: Union[torch.Tensor, List[torch.Tensor]],
    scaling: float,
    split_dims: Optional[List[int]] = None,
) -> torch.Tensor:
    if split_dims is not None:
        lora_weight = _get_split_lora_weight(split_dims, down_weight, up_weight)
        return _get_dense_weight_norm(base_weight, lora_weight, scaling)

    if base_weight.dim() == 2:
        return _get_linear_weight_norm_factored(base_weight, down_weight, up_weight, scaling)
    return _get_conv_weight_norm_factored(base_weight, down_weight, up_weight, scaling)


def _reshape_dora_factor_for_weight(factor: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
    return factor.reshape(-1).view(base_weight.shape[0], *([1] * (base_weight.dim() - 1)))


def _reshape_dora_factor_for_output(factor: torch.Tensor, output: torch.Tensor, is_conv2d: bool) -> torch.Tensor:
    flat_factor = factor.reshape(-1)
    if is_conv2d:
        return flat_factor.view(1, -1, *([1] * (output.dim() - 2)))
    return flat_factor.view(*([1] * (output.dim() - 1)), -1)


def _remove_module_bias(base_output: torch.Tensor, bias: Optional[torch.Tensor], is_conv2d: bool) -> torch.Tensor:
    if bias is None:
        return base_output
    if bias.device != base_output.device or bias.dtype != base_output.dtype:
        bias = bias.to(device=base_output.device, dtype=base_output.dtype)
    if is_conv2d:
        return base_output - bias.view(1, -1, *([1] * (base_output.dim() - 2)))
    return base_output - bias.view(*([1] * (base_output.dim() - 1)), -1)


def _apply_dora_weight_merge(
    base_weight: torch.Tensor, delta_weight: torch.Tensor, magnitude: torch.Tensor, weight_norm: torch.Tensor
) -> torch.Tensor:
    if weight_norm.is_floating_point():
        eps = 1e-12 if weight_norm.dtype in (torch.float32, torch.float64) else 1e-6
        weight_norm = weight_norm.clamp_min(eps)
    factor = magnitude.to(device=weight_norm.device, dtype=weight_norm.dtype) / weight_norm
    return _reshape_dora_factor_for_weight(factor, base_weight) * (base_weight + delta_weight)


def _convert_absolute_dora_oft_scales_to_ratios(state_dict: Dict[str, torch.Tensor]) -> None:
    for key in list(state_dict.keys()):
        if not key.endswith(".dora_scale"):
            continue
        module_name = key[: -len(".dora_scale")]
        initial_norm_key = f"{module_name}.initial_norm"
        if initial_norm_key not in state_dict:
            continue
        scale = state_dict[key]
        initial_norm = state_dict[initial_norm_key].to(dtype=scale.dtype)
        eps = 1e-12 if scale.dtype in (torch.float32, torch.float64) else 1e-6
        state_dict[key] = scale / initial_norm.clamp_min(eps)
