# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor
from torch.nn import Parameter
from torch.utils.checkpoint import checkpoint

from ...common import Modality, VarlenHandler, is_hopper_arch
from ...infra.parallelism import ulysses_scheduler
from musubi_tuner.modules.custom_offloading_utils import ModelOffloader, weighs_to_device
from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper

try:
    from magi_compiler import magi_compile
    from magi_compiler.api import magi_register_custom_op
    from magi_compiler.config import CompileConfig
except ImportError:
    class _FallbackOffloadConfig:
        gpu_resident_weight_ratio = 1.0

    class CompileConfig:  # type: ignore[override]
        def __init__(self) -> None:
            self.offload_config = _FallbackOffloadConfig()

    def magi_compile(config_patch=None):
        del config_patch

        def decorator(obj):
            return obj

        return decorator

    def magi_register_custom_op(*args, **kwargs):
        del args, kwargs

        def decorator(func):
            return func

        return decorator


MAGIHUMAN_ATTN_MODE = "flash"
MAGIHUMAN_SPLIT_ATTN = False
MAGIHUMAN_ATTN_COMPUTE_DTYPE = torch.bfloat16
SPLIT_ATTN_QUERY_CHUNK = 2048


def configure_attention_backend(attn_mode: str, split_attn: bool = False, compute_dtype: torch.dtype = torch.bfloat16):
    global MAGIHUMAN_ATTN_MODE, MAGIHUMAN_SPLIT_ATTN, MAGIHUMAN_ATTN_COMPUTE_DTYPE
    MAGIHUMAN_ATTN_MODE = attn_mode
    MAGIHUMAN_SPLIT_ATTN = split_attn
    MAGIHUMAN_ATTN_COMPUTE_DTYPE = compute_dtype


def _resolve_compute_dtype(params_dtype: torch.dtype, compute_dtype: Optional[torch.dtype] = None) -> torch.dtype:
    if compute_dtype is not None:
        return compute_dtype
    if params_dtype in (torch.float16, torch.bfloat16, torch.float32):
        return params_dtype
    return torch.bfloat16


def _get_attention_backend_compute_dtype() -> torch.dtype:
    if MAGIHUMAN_ATTN_MODE in ("flash", "flash3", "xformers") and MAGIHUMAN_ATTN_COMPUTE_DTYPE not in (
        torch.float16,
        torch.bfloat16,
    ):
        return torch.bfloat16
    return MAGIHUMAN_ATTN_COMPUTE_DTYPE


def _run_sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    if query.shape[1] != key.shape[1]:
        if query.shape[1] % key.shape[1] != 0:
            raise RuntimeError(
                f"MagiHuman SDPA backend requires query heads divisible by key/value heads, got "
                f"{query.shape[1]} query heads and {key.shape[1]} key/value heads."
            )
        repeat_factor = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)
    out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    return out.permute(0, 2, 1, 3).contiguous()


def _run_attention_backend_once(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    mode = MAGIHUMAN_ATTN_MODE
    if mode == "torch":
        return _run_sdpa(query, key, value)
    if mode == "xformers":
        try:
            import xformers.ops as xops
        except ImportError as exc:
            raise ImportError("xformers attention was requested for MagiHuman, but xformers is not installed.") from exc
        return xops.memory_efficient_attention(query, key, value)
    if mode == "flash3" and HAS_FA3 and is_hopper_arch():
        from flash_attn_interface import flash_attn_func as fa3_flash_attn_func

        return fa3_flash_attn_func(query, key, value)

    if mode in ("flash", "flash3"):
        if HAS_FA3 and is_hopper_arch():
            from flash_attn_interface import flash_attn_func as fa3_flash_attn_func

            return fa3_flash_attn_func(query, key, value)

        from flash_attn.flash_attn_interface import flash_attn_func as fa2_flash_attn_func

        return fa2_flash_attn_func(query, key, value)

    raise ValueError(f"Unsupported MagiHuman attention backend: {mode}")


def _run_attention_backend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    backend_compute_dtype = _get_attention_backend_compute_dtype()
    query = query.to(backend_compute_dtype).contiguous()
    key = key.to(backend_compute_dtype).contiguous()
    value = value.to(backend_compute_dtype).contiguous()

    if MAGIHUMAN_SPLIT_ATTN and query.shape[1] > SPLIT_ATTN_QUERY_CHUNK:
        outputs = []
        for start in range(0, query.shape[1], SPLIT_ATTN_QUERY_CHUNK):
            stop = start + SPLIT_ATTN_QUERY_CHUNK
            outputs.append(_run_attention_backend_once(query[:, start:stop], key, value))
        return torch.cat(outputs, dim=1)

    return _run_attention_backend_once(query, key, value)


@dataclass
class FFAHandler:
    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    attn_type_map: torch.Tensor
    softmax_scale: float


# Define the MLP activation type
class MLPActivationType(Enum):
    """Enumeration of supported activation functions for MLP"""

    SWIGLU7 = "swiglu7"
    GELU7 = "gelu7"


def swiglu7(x, alpha: float = 1.702, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer (from GPT-OSS)
    return (out_glu * (x_linear + 1)).to(out_dtype)


def gelu7(x, alpha: float = 1.702, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x_glu = x
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu.to(out_dtype)


def create_activation_func(activation_type: MLPActivationType) -> Callable:
    match activation_type:
        case MLPActivationType.SWIGLU7:
            return swiglu7
        case MLPActivationType.GELU7:
            return gelu7
        case _:
            raise ValueError(f"Unknown activation type: {activation_type}")


class ModalityDispatcher:
    permuted_modality_mapping: torch.Tensor
    group_size: torch.Tensor
    group_size_cpu: list[int]
    num_modalities: int

    def __init__(self, modality_mapping: torch.Tensor, num_modalities: int):
        """
        Initialize dispatcher.
        This runs once during object construction and precomputes all mappings.
        """
        self.modality_mapping = modality_mapping
        self.num_modalities = num_modalities

        self.permuted_modality_mapping = self._precompute_permute_mapping(modality_mapping)

        self.group_size = torch.bincount(self.permuted_modality_mapping, minlength=num_modalities).to(torch.int32)
        self.group_size_cpu: list[int] = [int(x) for x in self.group_size.to("cpu").tolist()]

    def _precompute_permute_mapping(self, modality_mapping):
        # 1. Compute forward and inverse permutation mappings.
        # argsort is an efficient O(N log N) operation.
        self.permute_mapping = torch.argsort(modality_mapping)
        self.inv_permute_mapping = torch.argsort(self.permute_mapping)

        # 2. Compute group size for each modality.
        # bincount is highly efficient for counting.
        permuted_modality_mapping = modality_mapping[self.permute_mapping]

        return permuted_modality_mapping

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        grouped_tensors = torch.split(x, self.group_size_cpu, dim=0)
        return list(grouped_tensors)

    def undispatch(self, *processed_groups: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(processed_groups, dim=0)

    @staticmethod
    def permute(x: torch.Tensor, permute_mapping: torch.Tensor) -> torch.Tensor:
        """Apply forward permutation to tensor."""
        return x[permute_mapping]

    @staticmethod
    def inv_permute(x: torch.Tensor, inv_permute_mapping: torch.Tensor) -> torch.Tensor:
        """Apply inverse permutation to tensor."""
        return x[inv_permute_mapping]


def freq_bands(
    num_bands: int, temperature: float = 10000.0, step: int = 2, device: Optional[torch.device] = None
) -> torch.Tensor:
    exp = torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(torch.float32) / num_bands
    bands = 1.0 / (temperature**exp)
    return bands


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat([x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]], dim=-1)


class ElementWiseFourierEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        max_res: int = 224,
        temperature: float = 10000.0,
        in_pixels: bool = True,
        linear_bands: bool = False,
        learnable: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            dim: Output feature dimension, total channels, must be divisible by 6
            max_res: Max pixel-frequency resolution for pixel-domain bands
            temperature: Temperature in inverse-frequency mode
            in_pixels: True -> pixel-frequency bands, False -> inverse-frequency bands
            linear_bands: Whether pixel-frequency bands are linearly spaced
            learnable: Whether frequency bands are trainable
        """
        super().__init__()
        self.dim = dim
        self.in_pixels = in_pixels
        self.learnable = learnable
        self.temperature = temperature
        self.max_res = max_res
        self.linear_bands = linear_bands
        self.device = device
        self.dtype = dtype
        # Make frequency bands trainable or register as buffer
        bands = self.get_default_bands()
        if self.learnable:
            self.bands = nn.Parameter(bands)
        else:
            self.register_buffer("bands", bands)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [L,9], column order (time, row, col, T, H, W, ref_T, ref_H, ref_W)
        Returns:
            emb: [L, dim] element-wise Fourier embedding
        """
        # Use slicing instead of unbind + stack to reduce intermediates
        coords_xyz = coords[:, :3]  # [L,3] -> (t, h, w)
        sizes = coords[:, 3:6]  # [L,3] -> (T, H, W)
        refs = coords[:, 6:9]  # [L,3] -> (ref_T, ref_H, ref_W)

        # Compute scale factors
        scales = (refs - 1) / (sizes - 1)  # [L,3]

        # NOTE: if both ref and size are 1, scale is fixed to 1; otherwise invalid
        scales[(refs == 1) & (sizes == 1)] = 1
        assert not scales.isnan().any(), "scales has nan"
        assert not scales.isinf().any(), "scales has inf"

        # Center alignment: apply to h,w only (not time)
        centers = (sizes - 1) / 2  # [L,3]
        centers[:, 0] = 0  # Do not center the time dimension
        coords_xyz = coords_xyz - centers  # [L,3]

        # Project to frequency bands in one shot: [L,3,B]
        proj = coords_xyz.unsqueeze(-1) * scales.unsqueeze(-1) * self.bands

        # Compute sin & cos and concatenate
        sin_proj = proj.sin()  # [L,3,B]
        cos_proj = proj.cos()

        return torch.cat((sin_proj, cos_proj), dim=1).flatten(1)

    def reset_parameters(self):
        bands = self.get_default_bands()
        self.bands.copy_(bands)

    def get_default_bands(self):
        if self.in_pixels:
            raise NotImplementedError("in_pixels are not implemented yet")
        else:
            bands = freq_bands(self.dim // 8, temperature=self.temperature, step=1, device=self.device).to(self.dtype)
        return bands


class MultiModalityRMSNorm(nn.Module):
    __constants__ = ["dim", "eps", "num_modality"]
    dim: int
    eps: float
    num_modality: int

    def __init__(self, dim: int, eps: float = 1e-6, device: torch.device | None = None, num_modality: int = 1):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.num_modality = num_modality

        self.weight = torch.nn.Parameter(torch.zeros(dim * num_modality, device=device, dtype=torch.float32))
        if num_modality > 1:
            self.forward = self.forward_multi_experts
        else:
            self.forward = self.forward_single_expert

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def rms(self, x: torch.Tensor) -> torch.Tensor:
        t, original_dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return t

    def forward_multi_experts(self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher) -> torch.Tensor:
        original_dtype = x.dtype
        t = self.rms(x)

        weight_chunked = self.weight.to(device=t.device).chunk(self.num_modality, dim=0)
        t_list = modality_dispatcher.dispatch(t)
        for i in range(self.num_modality):
            t_list[i] = t_list[i] * (weight_chunked[i] + 1)
        t = modality_dispatcher.undispatch(*t_list)

        return t.to(original_dtype)

    def forward_single_expert(self, x: torch.Tensor, modality_dispatcher: Optional[ModalityDispatcher] = None) -> torch.Tensor:
        t, original_dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * (self.weight.to(device=t.device) + 1)).to(original_dtype)


class _BF16ComputeLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        output_dtype: Optional[torch.dtype],
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        ctx.compute_dtype = compute_dtype
        ctx.input_dtype = input.dtype
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.has_bias = bias is not None
        ctx.save_for_backward(input, weight, bias if bias is not None else torch.tensor([], device=input.device, dtype=weight.dtype))

        compute_device = input.device
        # Convert input to specified input data type
        input_cast = input.to(device=compute_device, dtype=compute_dtype)
        # Convert weight to computation data type
        weight_cast = weight.to(device=compute_device, dtype=compute_dtype)
        # Perform linear operation
        output = torch.matmul(input_cast, weight_cast.t())

        # Add bias if present
        if bias is not None:
            bias_cast = bias.to(device=compute_device, dtype=compute_dtype)
            output = output + bias_cast
        else:
            bias_cast = None

        # Convert output to specified output data type
        return output.to(output_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, bias = ctx.saved_tensors
        del bias

        compute_dtype = ctx.compute_dtype
        compute_device = grad_output.device
        grad_output_cast = grad_output.to(device=compute_device, dtype=compute_dtype)
        input_cast = input.to(device=compute_device, dtype=compute_dtype)
        weight_cast = weight.to(device=compute_device, dtype=compute_dtype)

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output_cast, weight_cast).to(ctx.input_dtype)

        if ctx.needs_input_grad[1]:
            grad_output_2d = grad_output_cast.reshape(-1, grad_output_cast.shape[-1])
            input_2d = input_cast.reshape(-1, input_cast.shape[-1])
            grad_weight = torch.matmul(grad_output_2d.transpose(0, 1), input_2d).to(ctx.weight_dtype)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            reduce_dims = tuple(range(grad_output_cast.ndim - 1))
            grad_bias = grad_output_cast.sum(dim=reduce_dims).to(ctx.bias_dtype)

        return grad_input, grad_weight, grad_bias, None, None


class BaseLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "num_layers", "num_experts"]
    _FP8_DEQUANT_MAX_BYTES = 96 * 1024 * 1024
    in_features: int
    out_features: int
    num_layers_for_initialization: int
    num_experts: int
    weight: Tensor

    def __init__(
        self,
        in_features,
        out_features,
        num_layers_for_initialization,
        num_experts,
        bias=True,
        device=None,
        dtype=None,
        compute_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        params_dtype = torch.bfloat16 if dtype is None else dtype
        factory_kwargs = {"device": device, "dtype": params_dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers_for_initialization = num_layers_for_initialization
        self.num_experts = num_experts
        self.use_bias = bias
        self.compute_dtype = _resolve_compute_dtype(params_dtype, compute_dtype)
        self.weight = Parameter(torch.empty((out_features * num_experts, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features * num_experts, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def _ensure_parameters_on_input_device(self, input: torch.Tensor) -> None:
        target_device = self.weight.device if self.weight is not None else input.device
        if self.bias is not None and self.bias.device != target_device:
            self.bias.data = self.bias.data.to(target_device, non_blocking=target_device.type != "cpu")
        if hasattr(self, "scale_weight") and isinstance(self.scale_weight, torch.Tensor) and self.scale_weight.device != target_device:
            self.scale_weight = self.scale_weight.to(target_device, non_blocking=target_device.type != "cpu")

    @classmethod
    def _get_fp8_chunk_rows(cls, weight: torch.Tensor, compute_dtype: torch.dtype, target_device: torch.device) -> int:
        if target_device.type != "cuda" or weight.ndim != 2:
            return weight.shape[0]

        out_features, in_features = weight.shape
        bytes_per_element = torch.tensor([], dtype=compute_dtype).element_size()
        max_rows = cls._FP8_DEQUANT_MAX_BYTES // max(1, in_features * bytes_per_element)
        if max_rows <= 0 or max_rows >= out_features:
            return out_features

        if max_rows >= 256:
            max_rows = max(256, (max_rows // 256) * 256)
        return max(1, min(out_features, max_rows))

    @staticmethod
    def _dequantize_fp8_weight_chunk(
        weight_chunk: torch.Tensor, scale_chunk: torch.Tensor, original_dtype: torch.dtype
    ) -> torch.Tensor:
        if scale_chunk.ndim < 3:
            return weight_chunk.to(original_dtype) * scale_chunk

        out_features, num_blocks, _ = scale_chunk.shape
        dequantized_weight = weight_chunk.to(original_dtype).contiguous().view(out_features, num_blocks, -1)
        dequantized_weight = dequantized_weight * scale_chunk
        return dequantized_weight.view(weight_chunk.shape)

    @classmethod
    def _fp8_linear_forward_impl(
        cls,
        input: torch.Tensor,
        weight: torch.Tensor,
        scale_weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        original_dtype: torch.dtype,
        target_device: torch.device,
    ) -> torch.Tensor:
        input_cast = input.to(device=target_device, dtype=original_dtype)
        chunk_rows = cls._get_fp8_chunk_rows(weight, original_dtype, target_device)
        out_features = weight.shape[0]
        if chunk_rows >= out_features:
            dequantized_weight = cls._dequantize_fp8_weight_chunk(weight.to(device=target_device), scale_weight, original_dtype)
            return F.linear(input_cast, dequantized_weight, bias)

        outputs = []
        for start in range(0, out_features, chunk_rows):
            end = min(start + chunk_rows, out_features)
            weight_chunk = weight[start:end].to(device=target_device)
            scale_chunk = scale_weight[start:end]
            bias_chunk = bias[start:end] if bias is not None else None
            dequantized_weight = cls._dequantize_fp8_weight_chunk(weight_chunk, scale_chunk, original_dtype)
            outputs.append(F.linear(input_cast, dequantized_weight, bias_chunk))
            del weight_chunk, scale_chunk, bias_chunk, dequantized_weight

        return torch.cat(outputs, dim=-1)

    def _fp8_forward(self, input: torch.Tensor, output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        output_dtype = input.dtype if output_dtype is None else output_dtype
        original_dtype = self.scale_weight.dtype
        target_device = input.device
        weight = self.weight
        scale_weight = self.scale_weight.to(device=target_device, dtype=original_dtype)
        bias = self.bias.to(device=target_device, dtype=original_dtype) if self.bias is not None else None
        output = self._fp8_linear_forward_impl(input, weight, scale_weight, bias, original_dtype, target_device)
        return output.to(output_dtype)

    def forward(
        self,
        input: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
        modality_dispatcher: Optional[ModalityDispatcher] = None,
    ) -> torch.Tensor:
        self._ensure_parameters_on_input_device(input)
        if hasattr(self, "scale_weight"):
            return self._fp8_forward(input, output_dtype)
        output_dtype = input.dtype if output_dtype is None else output_dtype
        return _BF16ComputeLinear.apply(input, self.weight, self.bias, output_dtype, self.compute_dtype)


class NativeMoELinear(BaseLinear):
    def forward(
        self,
        input: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
        modality_dispatcher: Optional[ModalityDispatcher] = None,
    ) -> torch.Tensor:
        self._ensure_parameters_on_input_device(input)
        output_dtype = input.dtype if output_dtype is None else output_dtype

        input_list = modality_dispatcher.dispatch(input)  # type: ignore
        weight_chunked = self.weight.chunk(self.num_experts, dim=0)
        scale_chunked = None
        if hasattr(self, "scale_weight"):
            if self.scale_weight.ndim == 0 or self.scale_weight.numel() == 1:
                scale_chunked = [self.scale_weight] * self.num_experts
            else:
                scale_chunked = self.scale_weight.chunk(self.num_experts, dim=0)

        if self.bias is not None:
            bias_chunked = self.bias.chunk(self.num_experts, dim=0)

        for i in range(self.num_experts):
            if scale_chunked is not None:
                original_dtype = scale_chunked[i].dtype
                bias_chunk = bias_chunked[i].to(original_dtype) if self.bias is not None else None
                input_list[i] = self._fp8_linear_forward_impl(
                    input_list[i],
                    weight_chunked[i],
                    scale_chunked[i],
                    bias_chunk,
                    original_dtype,
                    input.device,
                ).to(output_dtype)
            else:
                input_list[i] = _BF16ComputeLinear.apply(
                    input_list[i],
                    weight_chunked[i],
                    bias_chunked[i] if self.bias is not None else None,
                    output_dtype,
                    self.compute_dtype,
                )
        return modality_dispatcher.undispatch(*input_list)  # type: ignore


def create_linear(
    in_features, out_features, num_layers=1, num_experts=1, bias=True, device=None, dtype=None, compute_dtype=None
) -> BaseLinear | NativeMoELinear:
    if num_experts == 1:
        return BaseLinear(in_features, out_features, num_layers, num_experts, bias, device, dtype, compute_dtype)
    else:
        return NativeMoELinear(in_features, out_features, num_layers, num_experts, bias, device, dtype, compute_dtype)


HAS_MAGI_ATTENTION = importlib.util.find_spec("magi_attention") is not None
HAS_FA3 = importlib.util.find_spec("flash_attn_interface") is not None


@magi_register_custom_op(name="infra::flash_attn_func", is_subgraph_boundary=True)
def flash_attn_func(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return _run_attention_backend(query, key, value)


def _split_q_range_with_no_overlap(
    q_ranges: torch.Tensor, k_ranges: torch.Tensor
) -> Tuple[List[List[int]], List[List[List[int]]]]:
    range_boundary = torch.unique(q_ranges, sorted=True).tolist()
    candidates = [[start, end, []] for start, end in zip(range_boundary[:-1], range_boundary[1:])]
    q_ranges = q_ranges.tolist()
    k_ranges = k_ranges.tolist()
    for q_range, k_range in zip(q_ranges, k_ranges):
        q_start, q_end = q_range
        for q_range_cand in candidates:
            if q_start <= q_range_cand[0] and q_range_cand[1] <= q_end:
                q_range_cand[2].append(k_range)
    q_ranges_out = []
    k_ranges_out = []
    for q_range_cand in candidates:
        if len(q_range_cand[2]) > 0:
            q_ranges_out.append(q_range_cand[0:2])
            k_ranges_out.append(q_range_cand[2])
    return q_ranges_out, k_ranges_out


def _flash_attn_with_correction(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_ranges: List[List[int]], k_range_list: List[List[List[int]]]
):
    output = torch.zeros_like(query)
    output_lse = torch.zeros((query.shape[0], query.shape[1]), dtype=torch.float32, device=query.device)

    for q_range, k_ranges in zip(q_ranges, k_range_list):
        q_start, q_end = q_range
        k_segments = [key[k_start:k_end] for k_start, k_end in k_ranges]
        v_segments = [value[k_start:k_end] for k_start, k_end in k_ranges]
        cat_k = torch.cat(k_segments, dim=0).unsqueeze(0)
        cat_v = torch.cat(v_segments, dim=0).unsqueeze(0)
        qo_out = _run_attention_backend(query[q_start:q_end].unsqueeze(0), cat_k, cat_v).squeeze(0)
        output[q_start:q_end] = qo_out
    return output, output_lse


def _custom_flex_flash_attn_func(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_ranges: torch.Tensor, k_ranges: torch.Tensor, **kwargs
):
    q_ranges, k_range_list = _split_q_range_with_no_overlap(q_ranges, k_ranges)
    return _flash_attn_with_correction(query, key, value, q_ranges, k_range_list)


def _flex_flash_attn_func_infer_output_meta(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_ranges: torch.Tensor, k_ranges: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(query)
    output_lse = torch.empty((query.shape[0], query.shape[1]), dtype=torch.float32, device=query.device)
    return output, output_lse


@magi_register_custom_op(
    name="infra::flex_flash_attn_func",
    mutates_args=(),
    infer_output_meta_fn=_flex_flash_attn_func_infer_output_meta,
    is_subgraph_boundary=True,
)
def flex_flash_attn_func(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_ranges: torch.Tensor, k_ranges: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if MAGIHUMAN_ATTN_MODE in ("flash", "flash3") and HAS_MAGI_ATTENTION and is_hopper_arch():
        from magi_attention.api import flex_flash_attn_func as magi_flex_flash_attn_func

        return magi_flex_flash_attn_func(query, key, value, q_ranges, k_ranges)
    else:
        return _custom_flex_flash_attn_func(query, key, value, q_ranges, k_ranges)


def _attention_with_cp_infer_output_meta(q: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return torch.empty_like(q, dtype=_get_attention_backend_compute_dtype()).squeeze(0)


@magi_register_custom_op(
    name="infra::flash_attn_with_cp",
    mutates_args=(),
    infer_output_meta_fn=_attention_with_cp_infer_output_meta,
    is_subgraph_boundary=True,
)
def flash_attn_with_cp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cp_split_sizes: List[int]) -> torch.Tensor:
    backend_compute_dtype = _get_attention_backend_compute_dtype()
    q, k, v = q.to(backend_compute_dtype), k.to(backend_compute_dtype), v.to(backend_compute_dtype)

    from ...infra.distributed import get_cp_group, get_cp_world_size
    from ...infra.parallelism.all_to_all_primitive import batch_scatter_head_gather_seqlen, scatter_seqlen_gather_head

    if get_cp_world_size() > 1:
        q, k, v = batch_scatter_head_gather_seqlen([q.squeeze(0), k.squeeze(0), v.squeeze(0)], cp_split_sizes, get_cp_group())
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    self_attn_out = flash_attn_func(q, k, v).squeeze(0)

    if get_cp_world_size() > 1:
        self_attn_out = scatter_seqlen_gather_head(self_attn_out, cp_split_sizes, get_cp_group(), async_op=False)
        self_attn_out = rearrange(self_attn_out, "(cp sq) hn hd -> sq (cp hn) hd", cp=get_cp_world_size())

    return self_attn_out


@magi_register_custom_op(
    name="infra::flex_flash_attn_with_cp",
    mutates_args=(),
    infer_output_meta_fn=_attention_with_cp_infer_output_meta,
    is_subgraph_boundary=True,
)
def flex_flash_attn_with_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    cp_split_sizes: List[int],
) -> torch.Tensor:
    backend_compute_dtype = _get_attention_backend_compute_dtype()
    q = q.to(backend_compute_dtype).squeeze(0)
    k = k.to(backend_compute_dtype).squeeze(0)
    v = v.to(backend_compute_dtype).squeeze(0)

    from ...infra.distributed import get_cp_group, get_cp_world_size
    from ...infra.parallelism.all_to_all_primitive import batch_scatter_head_gather_seqlen, scatter_seqlen_gather_head

    if get_cp_world_size() > 1:
        q, k, v = batch_scatter_head_gather_seqlen([q, k, v], cp_split_sizes, get_cp_group())

    out, _ = flex_flash_attn_func(q, k, v, q_ranges=q_ranges, k_ranges=k_ranges)

    if get_cp_world_size() > 1:
        out = scatter_seqlen_gather_head(out, cp_split_sizes, get_cp_group(), async_op=False)
        out = rearrange(out, "(cp sq) hn hd -> sq (cp hn) hd", cp=get_cp_world_size())

    return out


@dataclass
class AttentionConfig:
    hidden_size: int
    num_heads_q: int
    num_heads_kv: int
    head_dim: int
    params_dtype: torch.dtype
    compute_dtype: torch.dtype
    checkpoint_qk_layernorm_rope: bool
    num_modality: int
    num_layers: int
    use_local_attn: bool = False
    enable_attn_gating: bool = False


class Attention(torch.nn.Module):
    config: AttentionConfig

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config

        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, eps=1e-6, num_modality=config.num_modality)
        self.gating_size = config.num_heads_q if config.enable_attn_gating else 0

        self.linear_qkv = create_linear(
            config.hidden_size,
            config.num_heads_q * config.head_dim + config.num_heads_kv * config.head_dim * 2 + self.gating_size,
            num_experts=config.num_modality,
            bias=False,
            dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            num_layers=config.num_layers,
        )
        self.linear_proj = create_linear(
            config.num_heads_q * config.head_dim,
            config.hidden_size,
            bias=False,
            num_experts=config.num_modality,
            dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            num_layers=config.num_layers,
        )
        self.q_norm = MultiModalityRMSNorm(config.head_dim, num_modality=config.num_modality)
        self.k_norm = MultiModalityRMSNorm(config.head_dim, num_modality=config.num_modality)

        self.q_size = config.num_heads_q * config.head_dim
        self.kv_size = config.num_heads_kv * config.head_dim

    def reset_parameters(self):
        if hasattr(self.linear_proj, "reset_parameters_output_layer"):
            self.linear_proj.reset_parameters_output_layer()

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: List[int],
    ) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states, modality_dispatcher=modality_dispatcher).to(self.config.compute_dtype)
        qkv: torch.Tensor = self.linear_qkv(hidden_states, modality_dispatcher=modality_dispatcher).to(torch.float32)

        q, k, v, g = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size, self.gating_size], dim=1)
        q = q.view(-1, self.config.num_heads_q, self.config.head_dim)
        k = k.view(-1, self.config.num_heads_kv, self.config.head_dim)
        v = v.view(-1, self.config.num_heads_kv, self.config.head_dim)
        g = g.view(k.shape[0], self.config.num_heads_q, -1)

        q = self.q_norm(q, modality_dispatcher=modality_dispatcher)
        k = self.k_norm(k, modality_dispatcher=modality_dispatcher)

        q = ModalityDispatcher.inv_permute(q, inv_permute_mapping).unsqueeze(0)
        k = ModalityDispatcher.inv_permute(k, inv_permute_mapping).unsqueeze(0)
        v = ModalityDispatcher.inv_permute(v, inv_permute_mapping).unsqueeze(0)

        sin_emb, cos_emb = rope.tensor_split(2, -1)
        q = apply_rotary_emb_torch(q, cos_emb, sin_emb)
        k = apply_rotary_emb_torch(k, cos_emb, sin_emb)

        if self.config.use_local_attn:
            self_attn_out = flex_flash_attn_with_cp(
                q, k, v, local_attn_handler.q_ranges, local_attn_handler.k_ranges, cp_split_sizes
            )
        else:
            self_attn_out = flash_attn_with_cp(q, k, v, cp_split_sizes)
        self_attn_out = ModalityDispatcher.permute(self_attn_out, permute_mapping)

        if self.config.enable_attn_gating:
            self_attn_out = self_attn_out * torch.sigmoid(g)

        self_attn_out = self_attn_out.view(-1, self.config.num_heads_q * self.config.head_dim).to(self.config.compute_dtype)
        out = self.linear_proj(self_attn_out, modality_dispatcher=modality_dispatcher)
        return out


@dataclass
class MLPConfig:
    hidden_size: int
    intermediate_size: int
    activation_type: MLPActivationType
    params_dtype: torch.dtype
    compute_dtype: torch.dtype
    num_modality: int = 1
    num_layers: int = 1
    gated_act: bool = False


class MLP(torch.nn.Module):
    config: MLPConfig

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        num_experts = config.num_modality
        self.pre_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=config.num_modality)
        intermediate_size_up = config.intermediate_size * 2 if config.gated_act else config.intermediate_size

        self.up_gate_proj = create_linear(
            config.hidden_size,
            intermediate_size_up,
            bias=False,
            dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            num_layers=config.num_layers,
            num_experts=num_experts,
        )
        self.down_proj = create_linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            num_layers=config.num_layers,
            num_experts=num_experts,
        )
        self.activation_func = create_activation_func(config.activation_type)

    def forward(self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher) -> torch.Tensor:
        x = self.pre_norm(x, modality_dispatcher=modality_dispatcher).to(self.config.compute_dtype)
        x = self.up_gate_proj(x, modality_dispatcher=modality_dispatcher).to(torch.float32)
        x = self.activation_func(x).to(self.config.compute_dtype)
        x = self.down_proj(x, modality_dispatcher=modality_dispatcher).to(torch.float32)
        return x

    def extra_repr(self) -> str:
        return f"{self.up_gate_proj.weight.shape=}, {self.down_proj.weight.shape=}"


@dataclass
class AdapterConfig:
    hidden_size: int
    num_attention_heads: int
    text_in_channels: int
    video_in_channels: int
    audio_in_channels: int
    compute_dtype: torch.dtype


class Adapter(torch.nn.Module):
    config: AdapterConfig

    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.video_embedder = nn.Linear(config.video_in_channels, config.hidden_size, bias=True, dtype=config.compute_dtype)
        self.text_embedder = nn.Linear(config.text_in_channels, config.hidden_size, bias=True, dtype=config.compute_dtype)
        self.audio_embedder = nn.Linear(config.audio_in_channels, config.hidden_size, bias=True, dtype=config.compute_dtype)
        self.rope = ElementWiseFourierEmbed(config.hidden_size // config.num_attention_heads, in_pixels=False, learnable=False)

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        video_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ):
        rope = self.rope(coords_mapping)
        output_x = torch.zeros(x.shape[0], self.config.hidden_size, device=x.device, dtype=self.config.compute_dtype)
        output_x[text_mask] = self.text_embedder(x[text_mask, : self.config.text_in_channels].to(self.config.compute_dtype))
        output_x[audio_mask] = self.audio_embedder(x[audio_mask, : self.config.audio_in_channels].to(self.config.compute_dtype))
        output_x[video_mask] = self.video_embedder(x[video_mask, : self.config.video_in_channels].to(self.config.compute_dtype))
        return output_x, rope


class TransFormerLayer(torch.nn.Module):
    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        num_modality = 3 if layer_idx in config.mm_layers else 1
        use_local_attn = layer_idx in config.local_attn_layers
        self.post_norm = layer_idx in config.post_norm_layers
        attention_config = AttentionConfig(
            hidden_size=config.hidden_size,
            num_heads_q=config.num_heads_q,
            num_heads_kv=config.num_heads_kv,
            head_dim=config.head_dim,
            params_dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            checkpoint_qk_layernorm_rope=config.checkpoint_qk_layernorm_rope,
            num_modality=num_modality,
            num_layers=config.num_layers,
            use_local_attn=use_local_attn,
            enable_attn_gating=config.enable_attn_gating,
        )
        self.attention: Attention = Attention(attention_config)

        activation_type = MLPActivationType.GELU7 if layer_idx in config.gelu7_layers else MLPActivationType.SWIGLU7
        if activation_type == MLPActivationType.SWIGLU7:
            gated_act = True
            intermediate_size = int(config.hidden_size * 4 * 2 / 3) // 4 * 4
        else:
            gated_act = False
            intermediate_size = config.hidden_size * 4
        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            activation_type=activation_type,
            params_dtype=config.params_dtype,
            compute_dtype=config.compute_dtype,
            num_modality=num_modality,
            num_layers=config.num_layers,
            gated_act=gated_act,
        )
        self.mlp: MLP = MLP(mlp_config)
        if self.post_norm:
            self.attn_post_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=num_modality)
            self.mlp_post_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=num_modality)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: List[int],
    ) -> torch.Tensor:
        attn_out = self.attention(
            hidden_states,
            rope,
            permute_mapping,
            inv_permute_mapping,
            varlen_handler,
            local_attn_handler,
            modality_dispatcher,
            cp_split_sizes,
        )
        if self.post_norm:
            attn_out = self.attn_post_norm(attn_out, modality_dispatcher=modality_dispatcher)
        hidden_states = hidden_states + attn_out

        mlp_out = self.mlp(hidden_states, modality_dispatcher)
        if self.post_norm:
            mlp_out = self.mlp_post_norm(mlp_out, modality_dispatcher=modality_dispatcher)
        hidden_states = hidden_states + mlp_out
        return hidden_states


is_base_model = True


def config_patch(compile_config: CompileConfig) -> CompileConfig:
    global is_base_model
    if is_base_model:
        is_base_model = False
    else:
        # Fully offload SR model for memory-constrained GPU
        compile_config.offload_config.gpu_resident_weight_ratio = 0.0
    return compile_config


@magi_compile(config_patch=config_patch)
class TransformerBlock(torch.nn.Module):
    def __init__(self, model_config: Any):
        super().__init__()
        self.layers: list[TransFormerLayer] = nn.ModuleList()
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        for layer_idx in range(model_config.num_layers):
            self.layers.append(TransFormerLayer(model_config, layer_idx))

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: List[int],
    ) -> torch.Tensor:
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                layer_forward = layer
                if self.activation_cpu_offloading:
                    layer_forward = create_cpu_offloading_wrapper(layer_forward, x.device)
                x = checkpoint(
                    layer_forward,
                    x,
                    rope,
                    permute_mapping,
                    inv_permute_mapping,
                    varlen_handler,
                    local_attn_handler,
                    modality_dispatcher,
                    cp_split_sizes,
                    use_reentrant=False,
                )
            else:
                x = layer(
                    x,
                    rope,
                    permute_mapping,
                    inv_permute_mapping,
                    varlen_handler,
                    local_attn_handler,
                    modality_dispatcher,
                    cp_split_sizes,
                )
        return x


@dataclass
class TransformerConfig:
    hidden_size: int
    video_in_channels: int
    audio_in_channels: int
    text_in_channels: int
    params_dtype: torch.dtype
    compute_dtype: torch.dtype
    post_process_dtype: torch.dtype


class DiTModel(torch.nn.Module):
    config: TransformerConfig

    def __init__(self, model_config: Any):
        super().__init__()
        self.config = TransformerConfig(
            hidden_size=model_config.hidden_size,
            video_in_channels=model_config.video_in_channels,
            audio_in_channels=model_config.audio_in_channels,
            text_in_channels=model_config.text_in_channels,
            params_dtype=model_config.params_dtype,
            compute_dtype=model_config.compute_dtype,
            post_process_dtype=torch.float32,
        )
        adapter_config = AdapterConfig(
            hidden_size=model_config.hidden_size,
            num_attention_heads=model_config.num_heads_q,
            text_in_channels=model_config.text_in_channels,
            video_in_channels=model_config.video_in_channels,
            audio_in_channels=model_config.audio_in_channels,
            compute_dtype=model_config.compute_dtype,
        )
        self.adapter: Adapter = Adapter(adapter_config)
        self.block: TransformerBlock = TransformerBlock(model_config=model_config)
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap: Optional[int] = None
        self.offloader: Optional[ModelOffloader] = None
        self.final_norm_video = MultiModalityRMSNorm(self.config.hidden_size)
        self.final_norm_audio = MultiModalityRMSNorm(self.config.hidden_size)
        self.final_linear_video = nn.Linear(
            self.config.hidden_size, self.config.video_in_channels, bias=False, dtype=self.config.compute_dtype
        )
        self.final_linear_audio = nn.Linear(
            self.config.hidden_size, self.config.audio_in_channels, bias=False, dtype=self.config.compute_dtype
        )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading
        self.block.enable_gradient_checkpointing(activation_cpu_offloading)

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.block.disable_gradient_checkpointing()

    def enable_block_swap(self, num_blocks: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False):
        self.blocks_to_swap = num_blocks
        if num_blocks <= 0:
            self.offloader = None
            return

        num_layers = len(self.block.layers)
        assert num_blocks <= num_layers - 2, (
            f"Cannot swap more than {num_layers - 2} MagiHuman transformer layers. Requested {num_blocks}."
        )
        self.offloader = ModelOffloader(
            "magihuman",
            self.block.layers,
            num_layers,
            num_blocks,
            supports_backward,
            device,
            use_pinned_memory,
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap and self.offloader is not None:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap and self.offloader is not None:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            saved_layers = self.block.layers
            self.block.layers = None

        self.to(device)

        if self.blocks_to_swap:
            self.block.layers = saved_layers

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap and self.offloader is not None:
            self.offloader.prepare_block_devices_before_forward(self.block.layers)

    def _forward_layers(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler,
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: List[int],
    ) -> torch.Tensor:
        cpu_device = torch.device("cpu")
        for layer_idx, layer in enumerate(self.block.layers):
            if self.offloader is not None:
                self.offloader.wait_for_block(layer_idx)
                self._ensure_layer_weights_on_device(layer, x.device)

            if self.training and self.gradient_checkpointing:
                layer_forward = layer
                if self.activation_cpu_offloading:
                    layer_forward = create_cpu_offloading_wrapper(layer_forward, x.device)
                x = checkpoint(
                    layer_forward,
                    x,
                    rope,
                    permute_mapping,
                    inv_permute_mapping,
                    varlen_handler,
                    local_attn_handler,
                    modality_dispatcher,
                    cp_split_sizes,
                    use_reentrant=False,
                )
            else:
                x = layer(
                    x,
                    rope,
                    permute_mapping,
                    inv_permute_mapping,
                    varlen_handler,
                    local_attn_handler,
                    modality_dispatcher,
                    cp_split_sizes,
                )

            if self.offloader is not None:
                self.offloader.submit_move_blocks_forward(self.block.layers, layer_idx)
                if self._should_offload_current_block_after_forward(layer_idx):
                    weighs_to_device(layer, cpu_device)

        return x

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        modality_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler,
    ):
        x = ulysses_scheduler().dispatch(x)
        coords_mapping = ulysses_scheduler().dispatch(coords_mapping)
        modality_mapping = ulysses_scheduler().dispatch(modality_mapping)
        cp_split_sizes = ulysses_scheduler().cp_split_sizes

        modality_dispatcher = ModalityDispatcher(modality_mapping, 3)
        permute_mapping, inv_permute_mapping = modality_dispatcher.permute_mapping, modality_dispatcher.inv_permute_mapping
        video_mask = modality_mapping == Modality.VIDEO
        audio_mask = modality_mapping == Modality.AUDIO
        text_mask = modality_mapping == Modality.TEXT

        x, rope = self.adapter(x, coords_mapping, video_mask, audio_mask, text_mask)
        x = x.to(self.config.compute_dtype)
        x = ModalityDispatcher.permute(x, permute_mapping)
        x = self._forward_layers(
            x,
            rope,
            permute_mapping=permute_mapping,
            inv_permute_mapping=inv_permute_mapping,
            varlen_handler=varlen_handler,
            local_attn_handler=local_attn_handler,
            modality_dispatcher=modality_dispatcher,
            cp_split_sizes=cp_split_sizes,
        )
        x = ModalityDispatcher.inv_permute(x, inv_permute_mapping)

        x_video = x[video_mask].to(self.final_norm_video.weight.dtype)
        x_video = self.final_norm_video(x_video)
        x_video = self.final_linear_video(x_video)

        x_audio = x[audio_mask].to(self.final_norm_audio.weight.dtype)
        x_audio = self.final_norm_audio(x_audio)
        x_audio = self.final_linear_audio(x_audio)

        output_dtype = x_video.dtype if x_video.numel() > 0 else x_audio.dtype if x_audio.numel() > 0 else self.config.compute_dtype
        x_out = torch.zeros(
            x.shape[0],
            max(self.config.video_in_channels, self.config.audio_in_channels),
            device=x.device,
            dtype=output_dtype,
        )
        x_out[video_mask, : self.config.video_in_channels] = x_video.to(output_dtype)
        x_out[audio_mask, : self.config.audio_in_channels] = x_audio.to(output_dtype)
        x_out = ulysses_scheduler().undispatch(x_out)
        return x_out

    def _should_offload_current_block_after_forward(self, block_idx: int) -> bool:
        if self.blocks_to_swap is None or self.blocks_to_swap == 0 or self.offloader is None:
            return False

        if not self.offloader.forward_only:
            return block_idx < self.blocks_to_swap

        num_blocks = len(self.block.layers)
        if self.blocks_to_swap < (num_blocks // 2):
            return not (self.blocks_to_swap <= block_idx < num_blocks - self.blocks_to_swap)
        return True

    @staticmethod
    def _ensure_layer_weights_on_device(layer: torch.nn.Module, device: torch.device) -> None:
        for module in layer.modules():
            if hasattr(module, "weight") and module.weight is not None and module.weight.device != device:
                weighs_to_device(layer, device)
                return
