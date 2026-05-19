"""Opt-in N-GPU model-parallel placement for LTX-2 training."""

from __future__ import annotations

import argparse
import logging
import os
import re
import threading
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ENV_ENABLED = "LTX2_MODEL_PARALLEL"
ENV_DEVICES = "LTX2_MODEL_PARALLEL_DEVICES"
ENV_SPLITS = "LTX2_MODEL_PARALLEL_SPLITS"
MP_CODEC_NONE = "none"
MP_CODEC_INT8 = "int8"
MP_CODEC_INT4 = "int4"
MP_CODECS = (MP_CODEC_NONE, MP_CODEC_INT8, MP_CODEC_INT4)

_TRANSFER_COUNTER_LOCK = threading.Lock()
_TRANSFER_COUNTERS: dict[str, int] = {}


def _parser_has_option(parser: argparse.ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


def add_ltx2_model_parallel_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    if _parser_has_option(parser, "--ltx2_model_parallel"):
        return parser

    parser.add_argument(
        "--ltx2_model_parallel",
        action="store_true",
        help=(
            "Opt-in LTX-2 single-process model parallelism. Splits transformer blocks across multiple CUDA devices "
            "instead of using DDP model replication."
        ),
    )
    parser.add_argument(
        "--ltx2_model_parallel_devices",
        type=str,
        default=None,
        help=(
            "Comma-separated CUDA device ids for --ltx2_model_parallel. Default: all visible CUDA devices. "
            "The first id must match accelerator.device, usually cuda:0."
        ),
    )
    parser.add_argument(
        "--ltx2_model_parallel_splits",
        type=str,
        default=None,
        help=(
            "Comma-separated transformer block boundary indices for --ltx2_model_parallel. "
            "For N devices, provide N-1 boundaries. Default: even block split."
        ),
    )
    parser.add_argument(
        "--ltx2_mp_profile_transfers",
        action="store_true",
        help=(
            "Log forward and backward activation-transfer timing at LTX-2 model-parallel device boundaries. "
            "Only active with --ltx2_model_parallel."
        ),
    )
    parser.add_argument(
        "--ltx2_mp_profile_log_every",
        type=int,
        default=20,
        help="Log every N model-parallel activation transfers when --ltx2_mp_profile_transfers is enabled. Default: 20.",
    )
    parser.add_argument(
        "--ltx2_mp_activation_codec",
        type=str,
        default=MP_CODEC_NONE,
        choices=MP_CODECS,
        help=(
            "Optional activation codec at LTX-2 model-parallel device boundaries. "
            "'none' preserves existing raw tensor transfers; 'int8'/'int4' use blockwise quantization. Default: none."
        ),
    )
    parser.add_argument(
        "--ltx2_mp_grad_codec",
        type=str,
        default=MP_CODEC_NONE,
        choices=MP_CODECS,
        help=(
            "Optional backward activation-gradient codec at LTX-2 model-parallel device boundaries. "
            "'none' preserves raw gradient transfers; 'int8'/'int4' use blockwise quantization. Default: none."
        ),
    )
    parser.add_argument(
        "--ltx2_mp_int8_block_size",
        type=int,
        default=256,
        help="Block size for --ltx2_mp_activation_codec and --ltx2_mp_grad_codec low-bit codecs. Default: 256.",
    )
    return parser


def _env_flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def is_ltx2_model_parallel_enabled(args: argparse.Namespace | None = None) -> bool:
    return bool(getattr(args, "ltx2_model_parallel", False)) or _env_flag(ENV_ENABLED)


def _split_csv_ints(spec: str | None, *, field_name: str) -> list[int]:
    if spec is None or str(spec).strip() == "":
        return []
    values: list[int] = []
    for raw in str(spec).split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            values.append(int(raw))
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a comma-separated integer list, got {spec!r}") from exc
    return values


def resolve_device_ids(
    args: argparse.Namespace | None = None,
    *,
    cuda_device_count: int | None = None,
) -> list[int]:
    if cuda_device_count is None:
        cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    spec = getattr(args, "ltx2_model_parallel_devices", None) if args is not None else None
    if spec is None:
        spec = os.getenv(ENV_DEVICES)

    device_ids = _split_csv_ints(spec, field_name="ltx2_model_parallel_devices")
    if not device_ids:
        device_ids = list(range(int(cuda_device_count)))

    if len(device_ids) < 2:
        raise ValueError("LTX2 model parallelism requires at least two CUDA devices")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError(f"LTX2 model-parallel device ids must be unique, got {device_ids}")
    invalid = [device_id for device_id in device_ids if device_id < 0 or device_id >= int(cuda_device_count)]
    if invalid:
        raise ValueError(
            f"LTX2 model-parallel device ids {invalid} are outside available CUDA devices 0..{int(cuda_device_count) - 1}"
        )
    return device_ids


def even_split_points(num_blocks: int, num_devices: int) -> list[int]:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if num_devices < 2:
        raise ValueError("num_devices must be at least 2")
    if num_devices > num_blocks:
        raise ValueError(f"Cannot split {num_blocks} blocks across {num_devices} devices")

    base = num_blocks // num_devices
    rem = num_blocks % num_devices
    counts = [base + (1 if idx < rem else 0) for idx in range(num_devices)]
    points: list[int] = []
    cursor = 0
    for count in counts[:-1]:
        cursor += count
        points.append(cursor)
    return points


def resolve_split_points(
    num_blocks: int,
    num_devices: int,
    args: argparse.Namespace | None = None,
) -> list[int]:
    spec = getattr(args, "ltx2_model_parallel_splits", None) if args is not None else None
    if spec is None:
        spec = os.getenv(ENV_SPLITS)

    points = _split_csv_ints(spec, field_name="ltx2_model_parallel_splits")
    if not points:
        return even_split_points(num_blocks, num_devices)

    expected = num_devices - 1
    if len(points) != expected:
        raise ValueError(
            f"ltx2_model_parallel_splits must contain {expected} boundary value(s) for {num_devices} devices, got {points}"
        )
    if points != sorted(points) or len(set(points)) != len(points):
        raise ValueError(f"ltx2_model_parallel_splits must be strictly increasing, got {points}")
    if points[0] <= 0 or points[-1] >= num_blocks:
        raise ValueError(f"ltx2_model_parallel_splits must be inside 1..{num_blocks - 1}, got {points}")
    return points


@dataclass(frozen=True)
class ModelParallelTransferConfig:
    profile_transfers: bool = False
    profile_log_every: int = 20
    activation_codec: str = MP_CODEC_NONE
    grad_codec: str = MP_CODEC_NONE
    int8_block_size: int = 256

    @classmethod
    def from_args(cls, args: argparse.Namespace | None = None) -> "ModelParallelTransferConfig":
        if args is None:
            return cls()
        profile_log_every = getattr(args, "ltx2_mp_profile_log_every", 20)
        int8_block_size = getattr(args, "ltx2_mp_int8_block_size", 256)
        return cls(
            profile_transfers=bool(getattr(args, "ltx2_mp_profile_transfers", False)),
            profile_log_every=int(20 if profile_log_every is None else profile_log_every),
            activation_codec=str(getattr(args, "ltx2_mp_activation_codec", MP_CODEC_NONE) or MP_CODEC_NONE).lower(),
            grad_codec=str(getattr(args, "ltx2_mp_grad_codec", MP_CODEC_NONE) or MP_CODEC_NONE).lower(),
            int8_block_size=int(256 if int8_block_size is None else int8_block_size),
        )

    @property
    def enabled(self) -> bool:
        return (
            self.profile_transfers
            or self.activation_codec != MP_CODEC_NONE
            or self.grad_codec != MP_CODEC_NONE
        )

    def validate(self) -> None:
        if self.activation_codec not in MP_CODECS:
            raise ValueError(f"ltx2_mp_activation_codec must be one of {MP_CODECS}, got {self.activation_codec!r}")
        if self.grad_codec not in MP_CODECS:
            raise ValueError(f"ltx2_mp_grad_codec must be one of {MP_CODECS}, got {self.grad_codec!r}")
        if self.profile_log_every <= 0:
            raise ValueError("ltx2_mp_profile_log_every must be > 0")
        if self.int8_block_size <= 0:
            raise ValueError("ltx2_mp_int8_block_size must be > 0")


@dataclass(frozen=True)
class ModelParallelPlan:
    device_ids: tuple[int, ...]
    split_points: tuple[int, ...]
    block_device_ids: tuple[int, ...]

    @property
    def devices(self) -> tuple[torch.device, ...]:
        return tuple(torch.device(f"cuda:{device_id}") for device_id in self.device_ids)

    @property
    def block_devices(self) -> tuple[torch.device, ...]:
        return tuple(torch.device(f"cuda:{device_id}") for device_id in self.block_device_ids)

    @property
    def input_device(self) -> torch.device:
        return torch.device(f"cuda:{self.device_ids[0]}")

    def ranges(self) -> list[tuple[int, int, int]]:
        starts = [0, *self.split_points]
        ends = [*self.split_points, len(self.block_device_ids)]
        return [(device_id, start, end) for device_id, start, end in zip(self.device_ids, starts, ends)]


def build_model_parallel_plan(
    num_blocks: int,
    args: argparse.Namespace | None = None,
    *,
    cuda_device_count: int | None = None,
) -> ModelParallelPlan:
    device_ids = resolve_device_ids(args, cuda_device_count=cuda_device_count)
    split_points = resolve_split_points(num_blocks, len(device_ids), args)

    starts = [0, *split_points]
    ends = [*split_points, num_blocks]
    block_device_ids: list[int] = [0] * num_blocks
    for device_id, start, end in zip(device_ids, starts, ends):
        for block_idx in range(start, end):
            block_device_ids[block_idx] = device_id

    return ModelParallelPlan(
        device_ids=tuple(device_ids),
        split_points=tuple(split_points),
        block_device_ids=tuple(block_device_ids),
    )


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _next_transfer_count(key: str) -> int:
    with _TRANSFER_COUNTER_LOCK:
        count = _TRANSFER_COUNTERS.get(key, 0) + 1
        _TRANSFER_COUNTERS[key] = count
        return count


def _log_transfer_profile(
    *,
    label: str,
    direction: str,
    codec: str,
    source_device: torch.device,
    target_device: torch.device,
    raw_bytes: int,
    wire_bytes: int,
    elapsed_ms: float,
    config: ModelParallelTransferConfig,
) -> None:
    if not config.profile_transfers:
        return

    count_key = f"{label}:{direction}:{codec}:{source_device}->{target_device}"
    count = _next_transfer_count(count_key)
    if count != 1 and count % config.profile_log_every != 0:
        return

    raw_mb = raw_bytes / (1024.0 * 1024.0)
    wire_mb = wire_bytes / (1024.0 * 1024.0)
    ratio = raw_bytes / max(1, wire_bytes)
    logger.info(
        "LTX2 MP transfer %s %s #%d: %s -> %s codec=%s raw=%.2fMB wire=%.2fMB ratio=%.2fx time=%.2fms",
        label,
        direction,
        count,
        source_device,
        target_device,
        codec,
        raw_mb,
        wire_mb,
        ratio,
        elapsed_ms,
    )


def quantize_int8_blocks_for_ltx2_mp(
    tensor: torch.Tensor,
    block_size: int,
    *,
    stochastic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int, tuple[int, ...], torch.dtype]:
    """Blockwise symmetric int8 quantization for activation transport experiments."""

    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if not torch.is_floating_point(tensor):
        raise TypeError("int8 model-parallel transport only supports floating tensors")

    orig_shape = tuple(tensor.shape)
    orig_numel = int(tensor.numel())
    if orig_numel == 0:
        return (
            torch.empty(0, dtype=torch.int8, device=tensor.device),
            torch.empty(0, dtype=torch.float32, device=tensor.device),
            orig_numel,
            orig_shape,
            tensor.dtype,
        )

    flat = tensor.contiguous().view(-1)
    pad = (-orig_numel) % int(block_size)
    if pad:
        flat = F.pad(flat, (0, pad))

    blocks = flat.view(-1, int(block_size))
    blocks_f32 = blocks.to(torch.float32)
    scale = (blocks_f32.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
    q_f32 = (blocks_f32 / scale[:, None]).clamp(-127, 127)
    if stochastic:
        floor = torch.floor(q_f32)
        q_f32 = floor + (torch.rand_like(q_f32) < (q_f32 - floor)).to(q_f32.dtype)
    else:
        q_f32 = torch.round(q_f32)
    q = q_f32.clamp(-127, 127).to(torch.int8)
    return q, scale, orig_numel, orig_shape, tensor.dtype


def dequantize_int8_blocks_for_ltx2_mp(
    q: torch.Tensor,
    scale: torch.Tensor,
    orig_numel: int,
    orig_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if orig_numel == 0:
        return torch.empty(orig_shape, dtype=dtype, device=q.device)

    x = q.to(torch.float32) * scale.to(torch.float32)[:, None]
    x = x.reshape(-1)[:orig_numel]
    return x.view(orig_shape).to(dtype=dtype)


def quantize_int4_blocks_for_ltx2_mp(
    tensor: torch.Tensor,
    block_size: int,
    *,
    stochastic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int, tuple[int, ...], torch.dtype]:
    """Blockwise symmetric int4 quantization packed two values per byte."""

    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if not torch.is_floating_point(tensor):
        raise TypeError("int4 model-parallel transport only supports floating tensors")

    orig_shape = tuple(tensor.shape)
    orig_numel = int(tensor.numel())
    if orig_numel == 0:
        return (
            torch.empty(0, dtype=torch.uint8, device=tensor.device),
            torch.empty(0, dtype=torch.float32, device=tensor.device),
            orig_numel,
            orig_shape,
            tensor.dtype,
        )

    flat = tensor.contiguous().view(-1)
    pad = (-orig_numel) % int(block_size)
    if pad:
        flat = F.pad(flat, (0, pad))

    blocks = flat.view(-1, int(block_size))
    blocks_f32 = blocks.to(torch.float32)
    scale = (blocks_f32.abs().amax(dim=1) / 7.0).clamp_min(1e-12)
    q_f32 = (blocks_f32 / scale[:, None]).clamp(-7, 7)
    if stochastic:
        floor = torch.floor(q_f32)
        q_f32 = floor + (torch.rand_like(q_f32) < (q_f32 - floor)).to(q_f32.dtype)
    else:
        q_f32 = torch.round(q_f32)
    q = q_f32.clamp(-7, 7).to(torch.int8)
    nibbles = (q.reshape(-1).to(torch.int16) + 8).to(torch.uint8)
    if int(nibbles.numel()) % 2:
        nibbles = F.pad(nibbles, (0, 1), value=8)

    low = nibbles[0::2]
    high = nibbles[1::2]
    packed = torch.bitwise_or(low, torch.bitwise_left_shift(high, 4))
    return packed.contiguous(), scale, orig_numel, orig_shape, tensor.dtype


def dequantize_int4_blocks_for_ltx2_mp(
    packed: torch.Tensor,
    scale: torch.Tensor,
    orig_numel: int,
    orig_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if orig_numel == 0:
        return torch.empty(orig_shape, dtype=dtype, device=packed.device)

    total_padded = int(scale.numel()) * int((int(packed.numel()) * 2) // max(1, int(scale.numel())))
    low = torch.bitwise_and(packed, 0x0F)
    high = torch.bitwise_and(torch.bitwise_right_shift(packed, 4), 0x0F)
    nibbles = torch.empty(int(packed.numel()) * 2, dtype=torch.uint8, device=packed.device)
    nibbles[0::2] = low
    nibbles[1::2] = high
    nibbles = nibbles[:total_padded]

    block_size = total_padded // int(scale.numel())
    q = nibbles.to(torch.int16).sub_(8).to(torch.float32).view(-1, block_size)
    x = q * scale.to(torch.float32)[:, None]
    x = x.reshape(-1)[:orig_numel]
    return x.view(orig_shape).to(dtype=dtype)


def _transfer_tensor_raw(tensor: torch.Tensor, target_device: torch.device) -> tuple[torch.Tensor, int, int]:
    nbytes = _tensor_nbytes(tensor)
    return tensor.to(target_device), nbytes, nbytes


def _transfer_tensor_int8(
    tensor: torch.Tensor,
    target_device: torch.device,
    block_size: int,
) -> tuple[torch.Tensor, int, int]:
    if not torch.is_floating_point(tensor):
        return _transfer_tensor_raw(tensor, target_device)

    q, scale, orig_numel, orig_shape, dtype = quantize_int8_blocks_for_ltx2_mp(tensor, block_size)
    raw_bytes = _tensor_nbytes(tensor)
    wire_bytes = _tensor_nbytes(q) + _tensor_nbytes(scale)
    q = q.to(target_device)
    scale = scale.to(target_device)
    return dequantize_int8_blocks_for_ltx2_mp(q, scale, orig_numel, orig_shape, dtype), raw_bytes, wire_bytes


def _transfer_tensor_int4(
    tensor: torch.Tensor,
    target_device: torch.device,
    block_size: int,
) -> tuple[torch.Tensor, int, int]:
    if not torch.is_floating_point(tensor):
        return _transfer_tensor_raw(tensor, target_device)

    packed, scale, orig_numel, orig_shape, dtype = quantize_int4_blocks_for_ltx2_mp(tensor, block_size)
    raw_bytes = _tensor_nbytes(tensor)
    wire_bytes = _tensor_nbytes(packed) + _tensor_nbytes(scale)
    packed = packed.to(target_device)
    scale = scale.to(target_device)
    return dequantize_int4_blocks_for_ltx2_mp(packed, scale, orig_numel, orig_shape, dtype), raw_bytes, wire_bytes


def _profiled_transfer(
    tensor: torch.Tensor,
    target_device: torch.device,
    codec: str,
    block_size: int,
    *,
    config: ModelParallelTransferConfig,
    label: str,
    direction: str,
) -> torch.Tensor:
    source_device = tensor.device
    target_device = torch.device(target_device)

    if config.profile_transfers:
        _sync_if_cuda(source_device)
        _sync_if_cuda(target_device)
        start = time.perf_counter()

    if codec == MP_CODEC_INT8:
        out, raw_bytes, wire_bytes = _transfer_tensor_int8(tensor, target_device, block_size)
    elif codec == MP_CODEC_INT4:
        out, raw_bytes, wire_bytes = _transfer_tensor_int4(tensor, target_device, block_size)
    else:
        out, raw_bytes, wire_bytes = _transfer_tensor_raw(tensor, target_device)
        codec = MP_CODEC_NONE

    if config.profile_transfers:
        _sync_if_cuda(source_device)
        _sync_if_cuda(target_device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _log_transfer_profile(
            label=label,
            direction=direction,
            codec=codec,
            source_device=source_device,
            target_device=target_device,
            raw_bytes=raw_bytes,
            wire_bytes=wire_bytes,
            elapsed_ms=elapsed_ms,
            config=config,
        )

    return out


class _ModelParallelActivationTransfer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        target_device: torch.device,
        activation_codec: str,
        grad_codec: str,
        block_size: int,
        profile_transfers: bool,
        profile_log_every: int,
        label: str,
    ) -> torch.Tensor:
        config = ModelParallelTransferConfig(
            profile_transfers=bool(profile_transfers),
            profile_log_every=int(profile_log_every),
            activation_codec=str(activation_codec),
            grad_codec=str(grad_codec),
            int8_block_size=int(block_size),
        )
        config.validate()
        ctx.source_device = tensor.device
        ctx.grad_codec = config.grad_codec
        ctx.block_size = config.int8_block_size
        ctx.profile_transfers = config.profile_transfers
        ctx.profile_log_every = config.profile_log_every
        ctx.label = str(label)
        return _profiled_transfer(
            tensor,
            torch.device(target_device),
            config.activation_codec,
            config.int8_block_size,
            config=config,
            label=ctx.label,
            direction="forward",
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        config = ModelParallelTransferConfig(
            profile_transfers=bool(ctx.profile_transfers),
            profile_log_every=int(ctx.profile_log_every),
            activation_codec=MP_CODEC_NONE,
            grad_codec=str(ctx.grad_codec),
            int8_block_size=int(ctx.block_size),
        )
        grad_input = _profiled_transfer(
            grad_output,
            torch.device(ctx.source_device),
            config.grad_codec,
            config.int8_block_size,
            config=config,
            label=str(ctx.label),
            direction="backward",
        )
        return grad_input, None, None, None, None, None, None, None


def move_ltx2_model_parallel_activation(
    tensor: torch.Tensor,
    target_device: torch.device,
    config: ModelParallelTransferConfig | None,
    *,
    label: str,
) -> torch.Tensor:
    """Move an activation across a model-parallel boundary with optional compression."""

    if config is None or not config.enabled:
        return tensor.to(target_device)

    config.validate()
    return _ModelParallelActivationTransfer.apply(
        tensor,
        torch.device(target_device),
        config.activation_codec,
        config.grad_codec,
        config.int8_block_size,
        config.profile_transfers,
        config.profile_log_every,
        label,
    )


def _unwrap_transformer(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if hasattr(model, "model") else model


def is_ltx2_model_parallel_active(model: torch.nn.Module) -> bool:
    base_model = _unwrap_transformer(model)
    return bool(getattr(base_model, "_ltx2_model_parallel_enabled", False))


def clip_grad_norm_model_parallel(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    """Clip gradients for parameters spread across multiple devices.

    PyTorch's fused/foreach clipping paths can assume a single device in some
    wrappers. This computes the global norm from per-gradient norms, transfers
    only scalar norms to the first gradient device, then scales each gradient on
    its owning device.
    """

    if isinstance(parameters, torch.Tensor):
        parameter_list = [parameters]
    else:
        parameter_list = list(parameters)

    grads: list[torch.Tensor] = []
    for parameter in parameter_list:
        if parameter is None or parameter.grad is None:
            continue
        grads.append(parameter.grad.detach())

    if not grads:
        return torch.tensor(0.0)

    max_norm = float(max_norm)
    norm_type = float(norm_type)
    first_device = grads[0].device

    if norm_type == float("inf"):
        local_norms = [
            grad.coalesce()._values().abs().max() if grad.is_sparse else grad.abs().max()
            for grad in grads
        ]
        total_norm = torch.max(
            torch.stack([norm.to(device=first_device, dtype=torch.float32) for norm in local_norms])
        )
    else:
        local_norms = []
        for grad in grads:
            values = grad.coalesce()._values() if grad.is_sparse else grad
            local_norms.append(torch.linalg.vector_norm(values.float(), norm_type))
        total_norm = torch.linalg.vector_norm(
            torch.stack([norm.to(device=first_device, dtype=torch.float32) for norm in local_norms]),
            norm_type,
        )

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of gradients from ltx2 model-parallel parameters is non-finite: {total_norm.item()}"
        )

    clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for grad in grads:
        grad.mul_(clip_coef.to(device=grad.device, non_blocking=True))

    return total_norm


def get_ltx2_model_parallel_plan(model: torch.nn.Module) -> ModelParallelPlan | None:
    base_model = _unwrap_transformer(model)
    plan = getattr(base_model, "_ltx2_model_parallel_plan", None)
    return plan if isinstance(plan, ModelParallelPlan) else None


def resolve_ltx2_module_device_for_model_parallel(
    plan: ModelParallelPlan,
    module_path: str,
) -> torch.device:
    match = re.search(r"(?:^|[._])transformer_blocks[._](\d+)(?:[._]|$)", module_path)
    if match is None:
        return plan.input_device

    block_idx = int(match.group(1))
    if block_idx < 0 or block_idx >= len(plan.block_device_ids):
        raise ValueError(f"LoRA module path references invalid transformer block {block_idx}: {module_path}")
    return torch.device(f"cuda:{plan.block_device_ids[block_idx]}")


def place_ltx2_lora_network_for_model_parallel(
    network: torch.nn.Module,
    transformer: torch.nn.Module,
) -> dict[str, int]:
    plan = get_ltx2_model_parallel_plan(transformer)
    if plan is None:
        return {}

    lora_modules = list(getattr(network, "text_encoder_loras", []) or []) + list(
        getattr(network, "unet_loras", []) or []
    )
    counts: dict[str, int] = {}
    for lora_module in lora_modules:
        module_path = str(getattr(lora_module, "module_path", "") or getattr(lora_module, "lora_name", ""))
        device = resolve_ltx2_module_device_for_model_parallel(plan, module_path)
        lora_module.to(device)
        counts[str(device)] = counts.get(str(device), 0) + 1

    network._ltx2_model_parallel_lora_placed = True
    network._ltx2_model_parallel_lora_device_counts = counts
    if counts:
        logger.info(
            "LTX2 model-parallel LoRA placement: %s",
            ", ".join(f"{device}: {count}" for device, count in sorted(counts.items())),
        )
    return counts


def validate_ltx2_model_parallel_setup(args: argparse.Namespace, accelerator) -> None:
    if not is_ltx2_model_parallel_enabled(args):
        return
    ModelParallelTransferConfig.from_args(args).validate()
    if not torch.cuda.is_available():
        raise RuntimeError("LTX2 model parallelism requires CUDA")
    if int(getattr(accelerator, "num_processes", 1)) != 1:
        raise RuntimeError("LTX2 model parallelism is single-process model parallelism; use accelerate --num_processes 1")
    if int(getattr(args, "blocks_to_swap", 0) or 0) > 0:
        raise RuntimeError("LTX2 model parallelism is incompatible with --blocks_to_swap")
    if bool(getattr(args, "blockwise_checkpointing", False)):
        raise RuntimeError("LTX2 model parallelism is not compatible with --blockwise_checkpointing yet")
    if bool(getattr(args, "compile", False)):
        raise RuntimeError("LTX2 model parallelism is not compatible with --compile yet")

    device_ids = resolve_device_ids(args)
    accelerator_device = torch.device(getattr(accelerator, "device", torch.device("cuda:0")))
    if accelerator_device.type == "cuda":
        accelerator_index = 0 if accelerator_device.index is None else int(accelerator_device.index)
        if device_ids[0] != accelerator_index:
            raise RuntimeError(
                "The first LTX2 model-parallel device must match accelerator.device "
                f"({accelerator_device}); got cuda:{device_ids[0]}. Use CUDA_VISIBLE_DEVICES to remap GPUs."
            )


def enable_ltx2_model_parallel(model: torch.nn.Module, args: argparse.Namespace) -> ModelParallelPlan:
    base_model = _unwrap_transformer(model)
    blocks = getattr(base_model, "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError("LTX2 model parallelism requires transformer_blocks on the base model")

    num_blocks = len(blocks)
    plan = build_model_parallel_plan(num_blocks, args)
    input_device = plan.input_device

    saved_blocks = base_model.transformer_blocks
    base_model.transformer_blocks = torch.nn.ModuleList()
    base_model.to(input_device)
    base_model.transformer_blocks = saved_blocks

    for block_idx, block in enumerate(base_model.transformer_blocks):
        block.to(plan.block_devices[block_idx])

    base_model._ltx2_model_parallel_enabled = True
    base_model._ltx2_model_parallel_plan = plan
    base_model._ltx2_model_parallel_block_devices = plan.block_devices
    base_model._ltx2_model_parallel_input_device = input_device
    transfer_config = ModelParallelTransferConfig.from_args(args)
    transfer_config.validate()
    base_model._ltx2_model_parallel_transfer_config = transfer_config

    ranges = ", ".join(
        f"cuda:{device_id}: blocks {start}-{end - 1}" for device_id, start, end in plan.ranges() if start < end
    )
    logger.info("LTX2 model parallel enabled: %s", ranges)
    if transfer_config.enabled:
        logger.info(
            "LTX2 model-parallel transfer experiment: profile=%s log_every=%d activation_codec=%s "
            "grad_codec=%s int8_block_size=%d",
            transfer_config.profile_transfers,
            transfer_config.profile_log_every,
            transfer_config.activation_codec,
            transfer_config.grad_codec,
            transfer_config.int8_block_size,
        )
    return plan
