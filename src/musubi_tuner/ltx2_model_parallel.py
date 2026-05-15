"""Opt-in N-GPU model-parallel placement for LTX-2 training."""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass
from typing import Sequence

import torch

logger = logging.getLogger(__name__)

ENV_ENABLED = "LTX2_MODEL_PARALLEL"
ENV_DEVICES = "LTX2_MODEL_PARALLEL_DEVICES"
ENV_SPLITS = "LTX2_MODEL_PARALLEL_SPLITS"


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

    ranges = ", ".join(
        f"cuda:{device_id}: blocks {start}-{end - 1}" for device_id, start, end in plan.ranges() if start < end
    )
    logger.info("LTX2 model parallel enabled: %s", ranges)
    return plan
