"""TensorBoard helpers for timestep distribution logging."""

from __future__ import annotations

import argparse
import logging
from collections import deque
from typing import Dict, Optional

import torch
from accelerate import Accelerator

logger = logging.getLogger(__name__)


def get_tensorboard_writer(self, accelerator: Accelerator):
    for tracker in getattr(accelerator, "trackers", []):
        if getattr(tracker, "name", None) == "tensorboard" and hasattr(tracker, "writer"):
            return tracker.writer
    return None


def should_log_timestep_distribution_to_tensorboard(self, args: argparse.Namespace, accelerator: Accelerator) -> bool:
    if not accelerator.is_main_process:
        return False
    if not bool(getattr(args, "log_timestep_distribution_tensorboard", False)):
        return False
    return self._get_tensorboard_writer(accelerator) is not None


def get_timestep_distribution_logging_payload(
    self,
    args: argparse.Namespace,
    timesteps: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    del args
    return {"main": timesteps}


def prepare_timestep_distribution_values(
    self,
    timesteps: torch.Tensor,
    accelerator: Accelerator,
) -> Optional[torch.Tensor]:
    try:
        flat = timesteps.detach().reshape(-1).to(dtype=torch.float32)
        if accelerator.num_processes > 1:
            try:
                flat = accelerator.gather(flat)
            except Exception:
                flat = flat.cpu()
        else:
            flat = flat.cpu()

        if flat.numel() == 0:
            return None
        flat = flat[torch.isfinite(flat)]
        if flat.numel() == 0:
            return None
        return flat.contiguous()
    except Exception as e:
        logger.warning("Failed to accumulate timestep distribution for TensorBoard logging: %s", e)
        return None


def accumulate_timestep_distribution(
    self,
    timestep_buffers: Dict[str, deque],
    name: str,
    timesteps: torch.Tensor,
    accelerator: Accelerator,
) -> None:
    try:
        values = self._prepare_timestep_distribution_values(timesteps, accelerator)
        if values is None or values.numel() == 0:
            return
        if name not in timestep_buffers:
            timestep_buffers[name] = deque()
        timestep_buffers[name].append(values)
    except Exception as e:
        logger.warning("Failed to accumulate timestep distribution for TensorBoard logging: %s", e)


def log_timestep_distribution_histogram(
    self,
    accelerator: Accelerator,
    global_step: int,
    tag: str,
    values: torch.Tensor,
) -> None:
    try:
        writer = self._get_tensorboard_writer(accelerator)
        if writer is None or values.numel() == 0:
            return
        try:
            writer.add_histogram(tag, values, global_step=global_step, bins=100)
        except TypeError:
            writer.add_histogram(tag, values, global_step=global_step)

        writer.add_scalar(f"{tag}_mean", float(values.mean().item()), global_step)
        writer.add_scalar(f"{tag}_std", float(values.std(unbiased=False).item()), global_step)
        writer.add_scalar(f"{tag}_min", float(values.min().item()), global_step)
        writer.add_scalar(f"{tag}_max", float(values.max().item()), global_step)
        writer.add_scalar(f"{tag}_count", int(values.numel()), global_step)
    except Exception as e:
        logger.warning("Failed to log timestep distribution histogram to TensorBoard: %s", e)
