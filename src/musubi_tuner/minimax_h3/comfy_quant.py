"""Runtime support for Comfy-Org NVFP4/AWQ conditioner checkpoints."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

logger = logging.getLogger(__name__)

COMFY_QUANT_SUFFIX = ".comfy_quant"
NVFP4_BLOCK_SIZE = 16
NVFP4_VALUE_MAX = 6.0
NVFP4_SCALE_MAX = 448.0
_DEQUANT_CHUNK_ELEMENTS = 16 * 1024 * 1024


def has_comfy_quantized_layers(path: Path) -> bool:
    """Return whether a safetensors checkpoint contains Comfy quantization markers."""
    path = Path(path)
    if not path.is_file():
        return False
    with safe_open(path, framework="pt", device="cpu") as handle:
        source_keys = list(handle.keys())
        return any(key.endswith(COMFY_QUANT_SUFFIX) for key in source_keys)


def _parse_quant_config(blob: torch.Tensor) -> dict:
    try:
        value = json.loads(bytes(blob.cpu().tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Comfy quantization marker") from exc
    if not isinstance(value, dict):
        raise TypeError("Comfy quantization marker must contain a JSON object")
    return value


def unswizzle_nvfp4_scales(scales: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    """Convert the cuBLAS 128x4 blocked scale layout to row-major order."""
    row_blocks = (rows + 127) // 128
    column_blocks = (columns + 3) // 4
    padded_rows = row_blocks * 128
    padded_columns = column_blocks * 4
    expected_elements = padded_rows * padded_columns
    if scales.numel() != expected_elements:
        raise ValueError(
            f"invalid NVFP4 scale tensor: expected {expected_elements} elements for {rows}x{columns}, got {scales.numel()}"
        )
    values = scales.reshape(-1, 32, 16)
    values = values.reshape(-1, 32, 4, 4).transpose(1, 2)
    values = values.reshape(row_blocks, column_blocks, 4, 32, 4)
    values = values.reshape(row_blocks, column_blocks, 128, 4)
    values = values.permute(0, 2, 1, 3).reshape(padded_rows, padded_columns)
    return values[:rows, :columns].contiguous()


def swizzle_nvfp4_scales(scales: torch.Tensor) -> torch.Tensor:
    """Pad and convert row-major per-16 scales to the cuBLAS 128x4 layout."""
    rows, columns = scales.shape
    padded_rows = ((rows + 127) // 128) * 128
    padded_columns = ((columns + 3) // 4) * 4
    padded = torch.zeros((padded_rows, padded_columns), device=scales.device, dtype=scales.dtype)
    padded[:rows, :columns] = scales
    row_blocks = padded_rows // 128
    column_blocks = padded_columns // 4
    values = padded.view(row_blocks, 128, column_blocks, 4).permute(0, 2, 1, 3)
    return values.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(padded_rows, padded_columns).contiguous()


def _encode_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Encode float32 values as unpacked E2M1 nibbles with round-to-nearest-even."""
    if values.dtype is not torch.float32:
        raise TypeError("E2M1 encoding requires float32 input")
    bits = values.view(torch.int32)
    sign = bits & -0x80000000
    magnitude = (bits ^ sign).view(torch.float32)
    saturated = magnitude >= NVFP4_VALUE_MAX
    denormal = (~saturated) & (magnitude < 1.0)
    normal = ~(saturated | denormal)

    # Adding 2**22 aligns the FP32 mantissa so subtracting the bias leaves the
    # single E2M1 denormal bit, with the hardware FP32 add providing RNE.
    denormal_bias_bits = 149 << 23
    denormal_bias = torch.tensor(denormal_bias_bits, dtype=torch.int32, device=values.device).view(torch.float32)
    denormal_code = ((magnitude + denormal_bias).view(torch.int32) - denormal_bias_bits).to(torch.uint8)

    normal_bits = magnitude.view(torch.int32)
    odd_mantissa = (normal_bits >> 22) & 1
    normal_code = (normal_bits + ((1 - 127) << 23) + ((1 << 21) - 1) + odd_mantissa) >> 22
    codes = torch.full_like(normal_code, 7, dtype=torch.uint8)
    codes = torch.where(denormal, denormal_code, codes)
    codes = torch.where(normal, normal_code.to(torch.uint8), codes)
    return codes | ((sign >> 28).to(torch.uint8) & 8)


def quantize_nvfp4_activations(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Quantize a 2D activation into packed E2M1 data and two-level scales."""
    if inputs.ndim != 2:
        raise ValueError("NVFP4 activation quantization requires a rank-2 tensor")
    rows, columns = inputs.shape
    if columns % NVFP4_BLOCK_SIZE:
        raise ValueError(f"NVFP4 activation width must be divisible by {NVFP4_BLOCK_SIZE}")
    padded_rows = ((rows + 15) // 16) * 16
    if padded_rows != rows:
        inputs = F.pad(inputs, (0, 0, 0, padded_rows - rows))

    blocks = inputs.reshape(padded_rows, -1, NVFP4_BLOCK_SIZE).float()
    tensor_scale = (blocks.abs().amax() / (NVFP4_SCALE_MAX * NVFP4_VALUE_MAX)).reshape(())
    safe_tensor_scale = tensor_scale.clamp_min(torch.finfo(torch.float32).tiny)
    row_scales = (blocks.abs().amax(dim=-1) / NVFP4_VALUE_MAX / safe_tensor_scale).clamp_max(NVFP4_SCALE_MAX)
    row_scales = row_scales.to(torch.float8_e4m3fn)
    decoded_scales = tensor_scale * row_scales.float()
    safe_scales = torch.where(decoded_scales == 0, torch.ones_like(decoded_scales), decoded_scales)
    normalized = (blocks / safe_scales.unsqueeze(-1)).clamp(-NVFP4_VALUE_MAX, NVFP4_VALUE_MAX)
    normalized = torch.where(decoded_scales.unsqueeze(-1) == 0, 0, normalized).reshape(padded_rows, columns)
    codes = _encode_e2m1(normalized)
    packed = ((codes[:, 0::2] << 4) | codes[:, 1::2]).contiguous()
    return packed, swizzle_nvfp4_scales(row_scales), tensor_scale, rows


def nvfp4_scaled_mm_available(device: torch.device | None = None) -> bool:
    if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(F, "scaled_mm"):
        return False
    if device is None:
        return True
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 10


def _nvfp4_scaled_mm(
    inputs: torch.Tensor,
    packed_weight: torch.Tensor,
    blocked_weight_scales: torch.Tensor,
    weight_tensor_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    from torch.nn.functional import ScalingType, SwizzleType

    packed_inputs, input_scales, input_tensor_scale, original_rows = quantize_nvfp4_activations(inputs)
    output = F.scaled_mm(
        packed_inputs.view(torch.float4_e2m1fn_x2),
        packed_weight.view(torch.float4_e2m1fn_x2).t(),
        scale_a=[input_scales.reshape(-1), input_tensor_scale],
        scale_recipe_a=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
        scale_b=[blocked_weight_scales.reshape(-1), weight_tensor_scale],
        scale_recipe_b=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
        swizzle_a=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        swizzle_b=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        bias=bias,
        output_dtype=inputs.dtype,
    )
    return output[:original_rows]


class ComfyInt8Embedding(nn.Module):
    """Per-row INT8 embedding used by the quantized H3 Qwen3-VL checkpoint."""

    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor, output_dtype: torch.dtype) -> None:
        super().__init__()
        if qweight.ndim != 2 or qweight.dtype is not torch.int8:
            raise ValueError("Comfy INT8 embedding weight must be a rank-2 int8 tensor")
        if scales.numel() != qweight.shape[0]:
            raise ValueError("Comfy INT8 embedding must have one scale per row")
        self.num_embeddings, self.embedding_dim = qweight.shape
        self.output_dtype = output_dtype
        self.register_buffer("qweight", qweight.contiguous(), persistent=False)
        self.register_buffer(
            "scales_u8",
            scales.detach().float().reshape(-1).contiguous().view(torch.uint8),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        lookup_ids = input_ids.reshape(-1).to(self.qweight.device)
        rows = self.qweight.index_select(0, lookup_ids).float()
        scales = self.scales_u8.view(torch.float32).index_select(0, lookup_ids)
        output = (rows * scales.unsqueeze(1)).to(self.output_dtype)
        return output.to(input_ids.device).reshape(*input_ids.shape, self.embedding_dim)


class ComfyNvfp4Linear(nn.Module):
    """Frozen NVFP4/AWQ Linear with weight-only and optional W4A4 execution."""

    def __init__(
        self,
        source: nn.Linear,
        packed_weight: torch.Tensor,
        blocked_scales: torch.Tensor,
        per_tensor_scale: torch.Tensor,
        pre_quant_scale: torch.Tensor | None,
        output_dtype: torch.dtype,
        scaled_mm: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.output_dtype = output_dtype
        self.scaled_mm = scaled_mm
        if packed_weight.dtype is not torch.uint8 or packed_weight.shape != (
            self.out_features,
            self.in_features // 2,
        ):
            raise ValueError(
                f"invalid NVFP4 packed weight shape {tuple(packed_weight.shape)} for "
                f"Linear({self.in_features}, {self.out_features})"
            )
        if self.in_features % NVFP4_BLOCK_SIZE != 0:
            raise ValueError("NVFP4 Linear input size must be divisible by 16")
        if per_tensor_scale.numel() != 1:
            raise ValueError("NVFP4 weight_scale_2 must be a scalar")
        if pre_quant_scale is not None and pre_quant_scale.numel() != self.in_features:
            raise ValueError("NVFP4 pre_quant_scale must have one value per input column")

        scales = unswizzle_nvfp4_scales(
            blocked_scales.view(torch.float8_e4m3fn),
            self.out_features,
            self.in_features // NVFP4_BLOCK_SIZE,
        )
        self.register_buffer("packed_weight", packed_weight.contiguous(), persistent=False)
        self.register_buffer("scales_u8", scales.view(torch.uint8), persistent=False)
        if scaled_mm:
            self.register_buffer("blocked_scales_u8", blocked_scales.contiguous().view(torch.uint8), persistent=False)
        self.register_buffer(
            "per_tensor_scale_u8",
            per_tensor_scale.detach().float().reshape(1).contiguous().view(torch.uint8),
            persistent=False,
        )
        if pre_quant_scale is not None:
            if pre_quant_scale.dtype is not torch.bfloat16:
                raise ValueError(f"NVFP4 pre_quant_scale must be bfloat16, got {pre_quant_scale.dtype}")
            self.register_buffer(
                "pre_quant_scale_u8",
                pre_quant_scale.detach().reshape(-1).contiguous().view(torch.uint8),
                persistent=False,
            )
        self.register_parameter("bias", source.bias)

    def _pre_quant_scale(self) -> torch.Tensor | None:
        value = getattr(self, "pre_quant_scale_u8", None)
        return None if value is None else value.view(torch.bfloat16)

    @torch.no_grad()
    def dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        rows = self.out_features
        columns = self.in_features
        result = torch.empty((rows, columns), device=self.packed_weight.device, dtype=dtype)
        e2m1_values = torch.tensor(
            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
            device=self.packed_weight.device,
            dtype=torch.float32,
        )
        block_scales = self.scales_u8.view(torch.float8_e4m3fn).float()
        block_scales = block_scales * self.per_tensor_scale_u8.view(torch.float32).reshape(())
        chunk_rows = max(1, _DEQUANT_CHUNK_ELEMENTS // columns)
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            packed = self.packed_weight[start:end]
            codes = torch.stack((packed >> 4, packed & 15), dim=-1).reshape(end - start, columns)
            magnitudes = torch.index_select(e2m1_values, 0, (codes & 7).flatten().to(torch.int32)).view_as(codes)
            decoded = magnitudes * torch.where((codes & 8) > 0, -1.0, 1.0)
            decoded = decoded.view(end - start, columns // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
            decoded = decoded * block_scales[start:end].unsqueeze(-1)
            result[start:end].copy_(decoded.reshape(end - start, columns))
        return result

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.packed_weight.device != inputs.device:
            self.to(inputs.device)
        pre_quant_scale = self._pre_quant_scale()
        if pre_quant_scale is not None:
            inputs = inputs * pre_quant_scale.to(dtype=inputs.dtype)
        if self.scaled_mm:
            original_shape = inputs.shape
            inputs_2d = inputs.reshape(-1, original_shape[-1])
            output = _nvfp4_scaled_mm(
                inputs_2d,
                self.packed_weight,
                self.blocked_scales_u8.view(torch.float8_e4m3fn),
                self.per_tensor_scale_u8.view(torch.float32).reshape(()),
                self.bias,
            )
            return output.reshape(*original_shape[:-1], self.out_features)
        weight = self.dequantize_weight(inputs.dtype)
        return F.linear(inputs, weight, self.bias)


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, attribute, replacement)


@torch.no_grad()
def load_comfy_quantized_state_dict(
    model: nn.Module,
    checkpoint: Path,
    *,
    key_map: Callable[[str], str],
    output_dtype: torch.dtype,
    nvfp4_scaled_mm: bool = False,
) -> int:
    """Load a Comfy NVFP4/INT8 safetensors checkpoint into a meta-initialized model."""
    checkpoint = Path(checkpoint)
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        source_keys = list(handle.keys())
        marker_keys = [key for key in source_keys if key.endswith(COMFY_QUANT_SUFFIX)]
        if not marker_keys:
            raise ValueError(f"{checkpoint.name} does not contain Comfy quantized layers")

        consumed: set[str] = set()
        format_counts: dict[str, int] = {}
        for marker_key in marker_keys:
            prefix = marker_key[: -len(COMFY_QUANT_SUFFIX)]
            config = _parse_quant_config(handle.get_tensor(marker_key))
            quant_format = config.get("format")
            if not isinstance(quant_format, str):
                raise TypeError(f"{prefix} has no quantization format")
            format_counts[quant_format] = format_counts.get(quant_format, 0) + 1
            module_path = key_map(prefix)
            module = model.get_submodule(module_path)
            weight_key = f"{prefix}.weight"
            scale_key = f"{prefix}.weight_scale"
            if weight_key not in source_keys or scale_key not in source_keys:
                raise ValueError(f"{prefix} is missing its quantized weight or scale")
            weight = handle.get_tensor(weight_key)
            weight_scale = handle.get_tensor(scale_key)

            consumed_keys = {marker_key, weight_key, scale_key}
            if isinstance(module, nn.Embedding):
                if quant_format != "int8_tensorwise":
                    raise ValueError(f"unsupported Comfy embedding quantization {quant_format!r} on {prefix}")
                replacement = ComfyInt8Embedding(weight, weight_scale, output_dtype)
            elif isinstance(module, nn.Linear):
                if quant_format != "nvfp4":
                    raise ValueError(f"unsupported Comfy Linear quantization {quant_format!r} on {prefix}")
                scale_2_key = f"{prefix}.weight_scale_2"
                if scale_2_key not in source_keys:
                    raise ValueError(f"{prefix} is missing weight_scale_2")
                pre_scale_key = f"{prefix}.pre_quant_scale"
                replacement = ComfyNvfp4Linear(
                    module,
                    weight,
                    weight_scale,
                    handle.get_tensor(scale_2_key),
                    handle.get_tensor(pre_scale_key) if pre_scale_key in source_keys else None,
                    output_dtype,
                    nvfp4_scaled_mm,
                )
                consumed_keys.add(scale_2_key)
                if pre_scale_key in source_keys:
                    consumed_keys.add(pre_scale_key)
            else:
                raise TypeError(f"Comfy quant marker {prefix} maps to {type(module).__name__}; expected Linear or Embedding")
            _replace_submodule(model, module_path, replacement)
            consumed.update(consumed_keys)

        expected = set(model.state_dict())
        state_dict: dict[str, torch.Tensor] = {}
        for source_key in source_keys:
            if source_key in consumed:
                continue
            source_prefix, separator, suffix = source_key.rpartition(".")
            if not separator:
                raise ValueError(f"{checkpoint.name} contains unexpected conditioner key {source_key!r}")
            target_key = f"{key_map(source_prefix)}.{suffix}"
            if target_key not in expected:
                raise ValueError(f"{checkpoint.name} contains unexpected conditioner key {source_key!r}")
            state_dict[target_key] = handle.get_tensor(source_key)

    missing = sorted(expected - set(state_dict))
    if missing:
        raise ValueError(f"{checkpoint.name} is missing {len(missing)} text tensor(s), examples: {missing[:5]}")
    info = model.load_state_dict(state_dict, strict=True, assign=True)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(f"strict NVFP4/AWQ Qwen3-VL load failed: {info}")
    logger.info("Attached %d Comfy-quantized H3 layers: %s", len(marker_keys), format_counts)
    return len(marker_keys)
