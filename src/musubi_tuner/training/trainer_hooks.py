"""Default NetworkTrainer extension hooks."""

from __future__ import annotations

import argparse
import math

import torch


def is_model_parallel_enabled(self, args) -> bool:
    return False


def validate_model_parallel_setup(self, args, accelerator) -> None:
    pass


def enable_model_parallel_transformer(self, args, accelerator, transformer) -> None:
    pass


def place_network_for_model_parallel(self, args, accelerator, transformer, network) -> None:
    pass


def clip_grad_norm_for_model_parallel(self, args, accelerator, params, optimizer):
    return accelerator.clip_grad_norm_(params, args.max_grad_norm)


def pre_train_hook(self, args, accelerator, transformer=None, network=None):
    pass


def compute_prior_divergence_addition(self, args, accelerator, transformer, network, video_pred, network_dtype):
    return None


def preservation_backward(self, args, accelerator, transformer, network, network_dtype):
    return {}


def compute_validation_extra_loss(
    self,
    args,
    accelerator,
    transformer,
    network,
    batch,
    global_step: int,
    network_dtype,
):
    return None, {}


def modify_video_loss_per_element(self, args, per_elem, out, network_dtype):
    return per_elem, {}


def modify_audio_loss_per_element(self, args, per_elem, out, network_dtype):
    return per_elem, {}


def compute_video_extra_loss(self, args, out, network_dtype):
    return None, {}


def apply_differential_guidance_target(
    self,
    args: argparse.Namespace,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if not bool(getattr(args, "differential_guidance", False)):
        return target
    scale = float(getattr(args, "differential_guidance_scale", 3.0))
    if not math.isfinite(scale):
        raise ValueError("--differential_guidance_scale must be finite.")
    if scale == 1.0:
        return target
    detached_pred = pred.detach().to(device=target.device, dtype=target.dtype)
    return detached_pred + scale * (target - detached_pred)
