#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import math
import os
import random
import re
import time
from multiprocessing import Value
from typing import Any, Callable, Optional

import toml
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from safetensors.torch import save_file

from musubi_tuner.dataset.accumulation_group_sampler import build_accumulation_group_sampler
from musubi_tuner.dataset.audio_quota_sampler import (
    build_audio_sampler,
    split_concat_indices_by_audio,
    sync_dataset_group_epoch_without_loading,
)
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LTX2
from musubi_tuner.training.accelerator_setup import (
    clean_memory_on_device,
    collator_class,
    prepare_accelerator,
)
from musubi_tuner.training.metadata import (
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_MINIMUM_KEYS,
)
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.runtime_utils import (
    offload_optimizer_state_during_validation,
)
from musubi_tuner.training.sampling_prompts import should_sample_images
from musubi_tuner.training.timesteps import (
    compute_loss_weighting_for_sd3,
)
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.ogm_ge import compute_ogm_ge_coefficients
from musubi_tuner.audio_loss_balance import (
    compute_ema_magnitude_audio_weight,
    compute_inverse_frequency_audio_weight,
    compute_uncertainty_weighted_loss,
    update_audio_presence_ema,
    update_loss_ema,
)
from musubi_tuner.utils import huggingface_utils, model_utils, sai_model_spec, train_utils
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, mem_eff_save_file
from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer, ltx2_setup_parser
from musubi_tuner.ltx2_model_parallel import (
    add_ltx2_model_parallel_args,
    clip_grad_norm_model_parallel,
    enable_ltx2_model_parallel,
    is_ltx2_model_parallel_enabled,
    validate_ltx2_model_parallel_setup,
)
from musubi_tuner.ltx2_remote_stage import (
    enable_ltx2_remote_stage,
    is_ltx2_remote_stage_enabled,
    optimizer_step_ltx2_remote_stage,
    prune_ltx2_remote_stage_local_blocks,
    save_ltx2_remote_stage_state,
    validate_ltx2_remote_stage_setup,
    zero_grad_ltx2_remote_stage,
)
from musubi_tuner.ltx2_motion_preservation import (
    AttentionMapRecorder as _AttentionMapRecorder,
    build_motion_anchor_cache as _build_motion_anchor_cache,
    build_noisy_input_for_sigma as _build_noisy_input_for_sigma,
    collect_motion_attention_modules as _collect_motion_attention_modules,
    compute_motion_preservation_loss as _compute_motion_preservation_loss,
    filter_motion_attention_modules_for_swap as _filter_motion_attention_modules_for_swap,
    resolve_motion_anchor_cache_size as _resolve_motion_anchor_cache_size,
    sample_motion_sigma_index as _sample_motion_sigma_index,
)

import copy
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _all_declared_datasets_are_audio(user_config: dict) -> bool:
    declared_datasets: list[dict] = []
    for section_name in ("datasets", "validation_datasets"):
        section = user_config.get(section_name, [])
        if isinstance(section, list):
            declared_datasets.extend(ds for ds in section if isinstance(ds, dict))

    if not declared_datasets:
        return False

    return all(("audio_directory" in ds or "audio_jsonl_file" in ds) for ds in declared_datasets)


def _all_manifest_datasets_are_audio(manifest: dict) -> bool:
    declared_datasets: list[dict] = []
    for section_name in ("datasets", "validation_datasets"):
        section = manifest.get(section_name, [])
        if isinstance(section, list):
            declared_datasets.extend(entry for entry in section if isinstance(entry, dict))

    if not declared_datasets:
        return False

    return all(entry.get("dataset_type") == "audio" for entry in declared_datasets)


def _is_attention_geometry_param(param_name: str) -> bool:
    # Attention geometry parameters most tied to motion priors.
    return (
        re.search(
            r"(?:^|\.)(?:attn\d+|audio_attn\d+|audio_to_video_attn|video_to_audio_attn)\.(?:to_q|to_k|q_norm|k_norm)\.",
            param_name,
        )
        is not None
    )


def _is_self_attention_geometry_param(param_name: str) -> bool:
    return (
        re.search(
            r"(?:^|\.)(?:attn1|audio_attn1)\.(?:to_q|to_k|q_norm|k_norm)\.",
            param_name,
        )
        is not None
    )


class EMAModel:
    """Exponential Moving Average of model weights.

    Maintains shadow weights that are updated as:
        ema_weights = decay * ema_weights + (1 - decay) * model_weights

    Reference: https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    Similar to diffusers EMAModel but simplified for our use case.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: Model to track
            decay: EMA decay rate (higher = slower update, more smoothing)
            update_after_step: Start EMA decay after this many steps (warmup period
                              where shadow = current weights)
            update_every: Update EMA every N steps
            device: Device to store EMA weights (None = same as model)
        """
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.step = 0
        self._decay_started = False

        # Create shadow parameters (initialized from model)
        self.shadow_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                if device is not None:
                    self.shadow_params[name] = param.data.clone().to(device)
                else:
                    self.shadow_params[name] = param.data.clone()

    def _copy_to_shadow(self, model: torch.nn.Module) -> None:
        """Copy current model weights to shadow (used during warmup)."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow_params and param.requires_grad:
                    self.shadow_params[name].copy_(param.data.to(self.shadow_params[name].device))

    def update(self, model: torch.nn.Module) -> None:
        """Update EMA weights.

        During warmup (step < update_after_step): shadow = current weights
        After warmup: shadow = decay * shadow + (1-decay) * current
        """
        self.step += 1

        # During warmup period, just copy current weights to shadow
        # This ensures EMA starts from a reasonable state
        if self.step <= self.update_after_step:
            self._copy_to_shadow(model)
            return

        # Check if this is an update step
        if (self.step - self.update_after_step) % self.update_every != 0:
            return

        # Perform EMA update: shadow = decay * shadow + (1 - decay) * current
        # Using lerp_: shadow.lerp_(current, 1-decay) = shadow*(1-(1-decay)) + current*(1-decay)
        #            = shadow*decay + current*(1-decay)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow_params and param.requires_grad:
                    self.shadow_params[name].lerp_(param.data.to(self.shadow_params[name].device), 1.0 - self.decay)

    def apply_to(self, model: torch.nn.Module) -> dict:
        """Apply EMA weights to model, returning original weights for restoration."""
        original_params = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow_params:
                    original_params[name] = param.data.clone()
                    param.data.copy_(self.shadow_params[name].to(param.device))
        return original_params

    def restore(self, model: torch.nn.Module, original_params: dict) -> None:
        """Restore original weights to model."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])

    def state_dict(self) -> dict:
        """Get EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "step": self.step,
            "shadow_params": {k: v.cpu() for k, v in self.shadow_params.items()},
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from checkpoint."""
        self.decay = state_dict["decay"]
        self.step = state_dict["step"]
        for name, param in state_dict["shadow_params"].items():
            if name in self.shadow_params:
                self.shadow_params[name].copy_(param.to(self.shadow_params[name].device))


def _masked_mse(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: Optional[torch.Tensor],
    *,
    weighting: Optional[torch.Tensor],
    dtype: torch.dtype,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    per_elem_modifier: Optional[Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, float]]]] = None,
) -> torch.Tensor:
    if isinstance(tgt, torch.Tensor):
        pred = pred.to(device=tgt.device, dtype=dtype)
    else:
        pred = pred.to(dtype=dtype)
    if loss_type in ("mae", "l1"):
        per_elem = torch.nn.functional.l1_loss(pred, tgt, reduction="none")
    elif loss_type in ("huber", "smooth_l1"):
        per_elem = torch.nn.functional.smooth_l1_loss(pred, tgt, reduction="none", beta=huber_delta)
    else:
        per_elem = torch.nn.functional.mse_loss(pred, tgt, reduction="none")
    if weighting is not None:
        w = weighting
        if isinstance(w, torch.Tensor) and w.dim() != per_elem.dim():
            while w.dim() > per_elem.dim() and w.shape[-1] == 1:
                w = w.squeeze(-1)
        per_elem = per_elem * w
    if per_elem_modifier is not None:
        per_elem, _ = per_elem_modifier(per_elem)
    if mask is None:
        return per_elem.mean()

    mask = mask.to(device=per_elem.device)
    if per_elem.dim() == 5 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], 1, mask.shape[1], 1, 1)
    elif per_elem.dim() == 5 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1, 1)
    elif per_elem.dim() == 4 and mask.dim() == 2:
        mask = mask.view(mask.shape[0], 1, mask.shape[1], 1)
    elif per_elem.dim() == 4 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1, 1)
    elif per_elem.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    elif per_elem.dim() == 3 and mask.dim() == 1:
        mask = mask.view(mask.shape[0], 1, 1)

    mask_f = mask.to(dtype=per_elem.dtype)
    denom = mask_f.mean()
    if denom.item() == 0:
        return per_elem.mean()
    return (per_elem * mask_f).div(denom).mean()


def _normalize_ltx2_batch_for_call_dit(batch: dict) -> dict:
    """Normalize legacy text-embedding batch keys for LTX-2 call_dit compatibility."""
    if not isinstance(batch, dict):
        return batch

    conditions = batch.get("conditions")
    conditions_dict = conditions if isinstance(conditions, dict) else None

    def _first_tensor(keys: tuple[str, ...]) -> Optional[torch.Tensor]:
        if conditions_dict is not None:
            for key in keys:
                value = conditions_dict.get(key)
                if isinstance(value, torch.Tensor):
                    return value
        for key in keys:
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                return value
        return None

    video_prompt_embeds = _first_tensor(("video_prompt_embeds", "video_prompt"))
    audio_prompt_embeds = _first_tensor(("audio_prompt_embeds", "audio_prompt"))
    prompt_embeds = _first_tensor(("prompt_embeds", "text", "prompt"))
    prompt_attention_mask = _first_tensor(("prompt_attention_mask", "text_mask", "prompt_mask", "attention_mask"))

    if video_prompt_embeds is None:
        video_prompt_embeds = prompt_embeds

    if video_prompt_embeds is None and audio_prompt_embeds is None and prompt_embeds is None:
        if not isinstance(conditions, dict):
            return batch
        return batch

    normalized = batch
    if conditions_dict is None:
        normalized = dict(normalized)
        conditions_dict = {}
    else:
        conditions_dict = dict(conditions_dict)
        if normalized is batch:
            normalized = dict(normalized)

    if isinstance(video_prompt_embeds, torch.Tensor):
        conditions_dict.setdefault("video_prompt_embeds", video_prompt_embeds)
    if isinstance(audio_prompt_embeds, torch.Tensor):
        conditions_dict.setdefault("audio_prompt_embeds", audio_prompt_embeds)
    if isinstance(prompt_embeds, torch.Tensor):
        conditions_dict.setdefault("prompt_embeds", prompt_embeds)
        normalized.setdefault("text", prompt_embeds)
    if isinstance(prompt_attention_mask, torch.Tensor):
        conditions_dict.setdefault("prompt_attention_mask", prompt_attention_mask)
        normalized.setdefault("text_mask", prompt_attention_mask)

    normalized["conditions"] = conditions_dict
    return normalized


def _resolve_text_captions(batch: dict) -> list[str] | None:
    """Return normalized caption list from a LTX-2 batch, if available."""
    captions = batch.get("captions")
    if captions is None:
        return None
    if isinstance(captions, str):
        return [captions]
    if isinstance(captions, (list, tuple)):
        if len(captions) == 0:
            return []
        result: list[str] = []
        for caption in captions:
            if not isinstance(caption, str):
                return None
            result.append(caption)
        return result
    return None


def _add_full_ft_text_encoder_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--full_ft_train_text_encoder",
        action="store_true",
        default=False,
        help=(
            "Enable full-FT training for the Gemma text encoder in addition to the denoising transformer. "
            "The text encoder is not trained by default in full-FT mode."
        ),
    )
    parser.add_argument(
        "--full_ft_text_encoder_lr",
        type=float,
        default=None,
        help=(
            "Learning rate for text-encoder parameters when --full_ft_train_text_encoder is enabled. Defaults to --learning_rate."
        ),
    )
    parser.add_argument(
        "--full_ft_text_encoder_fallback",
        action="store_true",
        default=False,
        help=(
            "When --full_ft_train_text_encoder is enabled and a batch has no cached text embeddings, "
            "encode captions through Gemma at runtime."
        ),
    )
    return parser


def _get_ltx2_batch_tensor(
    batch: dict,
    key: str,
    *,
    nested_key: str = "latents",
) -> Optional[torch.Tensor]:
    value = batch.get(key)
    if isinstance(value, dict):
        value = value.get(nested_key)
    return value if isinstance(value, torch.Tensor) else None


def _validate_full_ft_ic_batch(
    args: argparse.Namespace,
    batch: dict,
    *,
    split_name: str,
    batch_index: int,
) -> None:
    ic_lora_strategy = str(getattr(args, "ic_lora_strategy", "none") or "none").lower()
    if ic_lora_strategy in {"none", "auto"}:
        return

    batch = _normalize_ltx2_batch_for_call_dit(batch)
    latents = _get_ltx2_batch_tensor(batch, "latents")
    if latents is None:
        raise ValueError(f"{split_name} batch {batch_index} is missing latents")

    ltx_mode = str(getattr(args, "ltx_mode", "video") or "video").lower()
    ref_latents = _get_ltx2_batch_tensor(batch, "ref_latents")
    audio_latents = _get_ltx2_batch_tensor(batch, "audio_latents")
    ref_audio_latents = _get_ltx2_batch_tensor(batch, "ref_audio_latents")

    def _normalize_stacked_refs(ref_value: Optional[torch.Tensor], *, single_ndim: int) -> Optional[torch.Tensor]:
        if ref_value is None:
            return None
        if ref_value.dim() == single_ndim + 1:
            if int(ref_value.shape[1]) <= 0:
                raise ValueError(f"{split_name} batch {batch_index} has an empty stacked reference tensor.")
            return ref_value[:, 0, ...]
        return ref_value

    ref_latents = _normalize_stacked_refs(ref_latents, single_ndim=5)
    ref_audio_latents = _normalize_stacked_refs(ref_audio_latents, single_ndim=4)

    if ic_lora_strategy == "v2v":
        if ltx_mode != "video":
            raise ValueError("--ic_lora_strategy v2v requires --ltx_mode=video")
        if ref_latents is None:
            raise ValueError(
                f"{split_name} batch {batch_index} is missing ref_latents required for --ic_lora_strategy v2v. "
                "Set reference_directory/reference_cache_directory and cache reference latents."
            )
        if ref_latents.dim() != 5:
            raise ValueError(
                f"{split_name} batch {batch_index} ref_latents must be 5D [B, C, F, H, W], got {tuple(ref_latents.shape)}"
            )
        if ref_latents.shape[0] != latents.shape[0]:
            raise ValueError(
                f"{split_name} batch {batch_index} latents/ref_latents batch mismatch: {latents.shape[0]} vs {ref_latents.shape[0]}"
            )
        if ref_latents.shape[1] != latents.shape[1]:
            raise ValueError(
                f"{split_name} batch {batch_index} latents/ref_latents channel mismatch: "
                f"{latents.shape[1]} vs {ref_latents.shape[1]}"
            )
        ref_h, ref_w = int(ref_latents.shape[3]), int(ref_latents.shape[4])
        tgt_h, tgt_w = int(latents.shape[3]), int(latents.shape[4])
        if ref_h != tgt_h or ref_w != tgt_w:
            h_ratio = tgt_h / ref_h
            w_ratio = tgt_w / ref_w
            if abs(h_ratio - w_ratio) > 0.01 or abs(h_ratio - round(h_ratio)) > 0.01:
                raise ValueError(
                    f"{split_name} batch {batch_index} has invalid V2V spatial mismatch: "
                    f"latents={tgt_h}x{tgt_w}, ref_latents={ref_h}x{ref_w}, "
                    f"h_ratio={h_ratio:.2f}, w_ratio={w_ratio:.2f}"
                )
        return

    if ic_lora_strategy == "audio_ref_ic":
        if ltx_mode not in {"av", "audio"}:
            raise ValueError("--ic_lora_strategy audio_ref_ic requires --ltx_mode=av or audio")
        if audio_latents is None:
            raise ValueError(
                f"{split_name} batch {batch_index} is missing audio_latents required for --ic_lora_strategy audio_ref_ic."
            )
        if ref_audio_latents is None:
            raise ValueError(
                f"{split_name} batch {batch_index} is missing ref_audio_latents required for --ic_lora_strategy audio_ref_ic. "
                "Set reference_audio_directory/reference_audio_cache_directory and cache reference audio latents."
            )
        return

    if ic_lora_strategy == "av_ic":
        if ltx_mode != "av":
            raise ValueError(f"--ic_lora_strategy {ic_lora_strategy} requires --ltx_mode=av")
        if ref_latents is None:
            raise ValueError(
                f"{split_name} batch {batch_index} is missing ref_latents required for --ic_lora_strategy {ic_lora_strategy}."
            )
        if ref_latents.dim() != 5:
            raise ValueError(
                f"{split_name} batch {batch_index} ref_latents must be 5D [B, C, F, H, W], got {tuple(ref_latents.shape)}"
            )
        if ref_latents.shape[0] != latents.shape[0]:
            raise ValueError(
                f"{split_name} batch {batch_index} latents/ref_latents batch mismatch: {latents.shape[0]} vs {ref_latents.shape[0]}"
            )
        if ref_latents.shape[1] != latents.shape[1]:
            raise ValueError(
                f"{split_name} batch {batch_index} latents/ref_latents channel mismatch: "
                f"{latents.shape[1]} vs {ref_latents.shape[1]}"
            )
        ref_h, ref_w = int(ref_latents.shape[3]), int(ref_latents.shape[4])
        tgt_h, tgt_w = int(latents.shape[3]), int(latents.shape[4])
        if ref_h != tgt_h or ref_w != tgt_w:
            h_ratio = tgt_h / ref_h
            w_ratio = tgt_w / ref_w
            if abs(h_ratio - w_ratio) > 0.01 or abs(h_ratio - round(h_ratio)) > 0.01:
                raise ValueError(
                    f"{split_name} batch {batch_index} {ic_lora_strategy} has invalid spatial mismatch: "
                    f"latents={tgt_h}x{tgt_w}, ref_latents={ref_h}x{ref_w}, "
                    f"h_ratio={h_ratio:.2f}, w_ratio={w_ratio:.2f}"
                )
        if audio_latents is None:
            raise ValueError(
                f"{split_name} batch {batch_index} is missing audio_latents required for --ic_lora_strategy {ic_lora_strategy}."
            )
        if ref_audio_latents is None:
            raise ValueError(
                f"{split_name} batch {batch_index} is missing ref_audio_latents required for --ic_lora_strategy {ic_lora_strategy}."
            )
        return

    if ic_lora_strategy == "video_ref_only_av":
        if ltx_mode != "av":
            raise ValueError("--ic_lora_strategy video_ref_only_av requires --ltx_mode=av")
        if ref_latents is None:
            raise ValueError(
                f"{split_name} batch {batch_index} is missing ref_latents required for --ic_lora_strategy video_ref_only_av."
            )
        if ref_latents.dim() != 5:
            raise ValueError(
                f"{split_name} batch {batch_index} ref_latents must be 5D [B, C, F, H, W], got {tuple(ref_latents.shape)}"
            )
        if ref_latents.shape[0] != latents.shape[0]:
            raise ValueError(
                f"{split_name} batch {batch_index} latents/ref_latents batch mismatch: {latents.shape[0]} vs {ref_latents.shape[0]}"
            )
        if ref_latents.shape[1] != latents.shape[1]:
            raise ValueError(
                f"{split_name} batch {batch_index} latents/ref_latents channel mismatch: "
                f"{latents.shape[1]} vs {ref_latents.shape[1]}"
            )
        ref_h, ref_w = int(ref_latents.shape[3]), int(ref_latents.shape[4])
        tgt_h, tgt_w = int(latents.shape[3]), int(latents.shape[4])
        if ref_h != tgt_h or ref_w != tgt_w:
            h_ratio = tgt_h / ref_h
            w_ratio = tgt_w / ref_w
            if abs(h_ratio - w_ratio) > 0.01 or abs(h_ratio - round(h_ratio)) > 0.01:
                raise ValueError(
                    f"{split_name} batch {batch_index} video_ref_only_av has invalid spatial mismatch: "
                    f"latents={tgt_h}x{tgt_w}, ref_latents={ref_h}x{ref_w}, "
                    f"h_ratio={h_ratio:.2f}, w_ratio={w_ratio:.2f}"
                )
        if audio_latents is None:
            raise ValueError(
                f"{split_name} batch {batch_index} is missing audio_latents required for --ic_lora_strategy video_ref_only_av."
            )


def _run_full_ft_ic_preflight(
    args: argparse.Namespace,
    dataset_group,
    *,
    split_name: str,
    max_batches: int = 8,
) -> None:
    ic_lora_strategy = str(getattr(args, "ic_lora_strategy", "none") or "none").lower()
    if ic_lora_strategy in {"none", "auto"} or dataset_group is None:
        return

    num_batches = min(len(dataset_group), max_batches)
    for batch_index in range(num_batches):
        try:
            batch = dataset_group[batch_index]
        except Exception as exc:
            raise ValueError(
                f"{split_name} IC preflight failed while loading batch {batch_index} for "
                f"--ic_lora_strategy {ic_lora_strategy}: {exc}"
            ) from exc
        _validate_full_ft_ic_batch(
            args,
            batch,
            split_name=split_name,
            batch_index=batch_index,
        )


def _run_image_prior_ft_preflight(
    args: argparse.Namespace,
    dataset_group,
    *,
    split_name: str,
    max_batches: int = 16,
) -> None:
    if not bool(getattr(args, "image_prior_ft", False)):
        return
    if not bool(getattr(args, "image_prior_ft_strict", True)) or dataset_group is None:
        return

    num_batches = min(len(dataset_group), max_batches)
    checked = 0
    for batch_index in range(num_batches):
        try:
            batch = dataset_group[batch_index]
        except Exception as exc:
            raise ValueError(f"{split_name} Image-prior full-FT preflight failed while loading batch {batch_index}: {exc}") from exc

        batch = _normalize_ltx2_batch_for_call_dit(batch)
        latents = _get_ltx2_batch_tensor(batch, "latents")
        if latents is None:
            raise ValueError(f"{split_name} batch {batch_index} is missing latents")
        if latents.dim() != 5:
            raise ValueError(f"{split_name} batch {batch_index} latents must be 5D [B, C, F, H, W], got {tuple(latents.shape)}")
        if int(latents.shape[2]) != 1:
            raise ValueError(
                f"--image_prior_ft expects image-only cached latents with F=1. "
                f"{split_name} batch {batch_index} has F={int(latents.shape[2])}. "
                "Use an image dataset/cache or pass --no-image_prior_ft_strict for mixed experiments."
            )
        if _get_ltx2_batch_tensor(batch, "audio_latents") is not None:
            raise ValueError(
                f"--image_prior_ft expects video/image batches without audio_latents; "
                f"{split_name} batch {batch_index} includes audio latents."
            )
        checked += 1

    logger.info("Image-prior full-FT preflight passed for %s split (%d batch(es) checked).", split_name, checked)


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _clone_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(v) for v in value)
    return copy.deepcopy(value)


def _move_to_device(value: Any, device: torch.device, *, dtype: Optional[torch.dtype] = None) -> Any:
    if isinstance(value, torch.Tensor):
        out = value.to(device=device)
        if dtype is not None and out.dtype.is_floating_point:
            out = out.to(dtype=dtype)
        return out
    if isinstance(value, dict):
        return {k: _move_to_device(v, device, dtype=dtype) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device, dtype=dtype) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device, dtype=dtype) for v in value)
    return value


def _device_matches(a: torch.device, b: Optional[torch.device]) -> bool:
    if b is None:
        return True
    if a.type != b.type:
        return False
    # Treat index-less CUDA device ("cuda") as a wildcard for any CUDA index.
    if a.type == "cuda":
        if a.index is None or b.index is None:
            return True
        return int(a.index) == int(b.index)
    return a == b


def _safe_abs_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return os.path.abspath(os.path.expanduser(str(path)))


def _path_mtime(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    try:
        return float(os.path.getmtime(path))
    except OSError:
        return None


def _signature_hash(signature: dict[str, Any]) -> str:
    payload = json.dumps(signature, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _torch_load_cpu(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_ewc_cache_signature(args: argparse.Namespace) -> dict[str, Any]:
    ckpt_path = _safe_abs_path(getattr(args, "ltx2_checkpoint", None))
    ds_cfg = _safe_abs_path(getattr(args, "dataset_config", None))
    return {
        "schema": 1,
        "kind": "ewc_cache",
        "checkpoint": ckpt_path,
        "checkpoint_mtime": _path_mtime(ckpt_path),
        "dataset_config": ds_cfg,
        "dataset_config_mtime": _path_mtime(ds_cfg),
        "ewc_target": getattr(args, "ewc_target", "attn_norm_bias"),
        "ewc_num_batches": int(getattr(args, "ewc_num_batches", 8) or 0),
        "ewc_max_param_tensors": int(getattr(args, "ewc_max_param_tensors", 256) or 0),
        "blocks_to_swap": int(getattr(args, "blocks_to_swap", 0) or 0),
        "freeze_early_blocks": int(getattr(args, "freeze_early_blocks", 0) or 0),
        "freeze_block_indices": getattr(args, "freeze_block_indices", None),
        "block_lr_scales": getattr(args, "block_lr_scales", None),
        "non_block_lr_scale": float(getattr(args, "non_block_lr_scale", 1.0) or 0.0),
        "attn_geometry_lr_scale": float(getattr(args, "attn_geometry_lr_scale", 1.0) or 0.0),
        "freeze_attn_geometry": bool(getattr(args, "freeze_attn_geometry", False)),
    }


def _load_ewc_cache(
    path: str,
    signature: dict[str, Any],
    transformer: torch.nn.Module,
    target_device: Optional[torch.device] = None,
) -> Optional[dict[str, Any]]:
    cache_path = _safe_abs_path(path)
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        payload = _torch_load_cpu(cache_path)
        if not isinstance(payload, dict):
            logger.warning("EWC cache has invalid payload type, rebuilding: %s", cache_path)
            return None
        expected_hash = _signature_hash(signature)
        cached_hash = payload.get("signature_hash")
        if cached_hash != expected_hash:
            logger.info("EWC cache signature mismatch; rebuilding: %s", cache_path)
            return None

        param_names = payload.get("param_names")
        theta_ref = payload.get("theta_ref")
        fisher = payload.get("fisher")
        if not isinstance(param_names, list) or not isinstance(theta_ref, dict) or not isinstance(fisher, dict):
            logger.warning("EWC cache payload is malformed, rebuilding: %s", cache_path)
            return None

        named_params = dict(transformer.named_parameters())
        restored_params: list[tuple[str, torch.nn.Parameter]] = []
        restored_theta: dict[str, torch.Tensor] = {}
        restored_fisher: dict[str, torch.Tensor] = {}
        all_params: list[tuple[str, torch.nn.Parameter]] = []
        all_theta: dict[str, torch.Tensor] = {}
        all_fisher: dict[str, torch.Tensor] = {}
        dropped_device = 0
        for name in param_names:
            if not isinstance(name, str):
                continue
            param = named_params.get(name)
            theta = theta_ref.get(name)
            fish = fisher.get(name)
            if param is None or not param.requires_grad:
                continue
            if isinstance(theta, torch.Tensor) and isinstance(fish, torch.Tensor):
                all_params.append((name, param))
                all_theta[name] = theta.detach().cpu().float()
                all_fisher[name] = fish.detach().cpu().float().clamp_min(1e-12)
            if target_device is not None and not _device_matches(param.device, target_device):
                dropped_device += 1
                continue
            if not isinstance(theta, torch.Tensor) or not isinstance(fish, torch.Tensor):
                continue
            restored_params.append((name, param))
            restored_theta[name] = theta.detach().cpu().float()
            restored_fisher[name] = fish.detach().cpu().float().clamp_min(1e-12)

        if not restored_params:
            if target_device is not None and all_params:
                logger.warning(
                    "EWC cache device filter kept 0 tensors on %s; falling back to unfiltered cache tensors=%d.",
                    str(target_device),
                    len(all_params),
                )
                restored_params = all_params
                restored_theta = all_theta
                restored_fisher = all_fisher
            else:
                logger.warning("EWC cache has no usable parameters for current run, rebuilding: %s", cache_path)
                return None

        if dropped_device > 0:
            logger.info(
                "Loaded EWC cache: %s (tensors=%d, dropped_device=%d target_device=%s)",
                cache_path,
                len(restored_params),
                dropped_device,
                str(target_device),
            )
        else:
            logger.info("Loaded EWC cache: %s (tensors=%d)", cache_path, len(restored_params))
        return {
            "params": restored_params,
            "theta_ref": restored_theta,
            "fisher": restored_fisher,
        }
    except Exception:
        logger.exception("Failed to load EWC cache, rebuilding: %s", cache_path)
        return None


def _save_ewc_cache(path: str, signature: dict[str, Any], ewc_state: dict[str, Any]) -> None:
    cache_path = _safe_abs_path(path)
    if not cache_path:
        return
    params = ewc_state.get("params")
    theta_ref = ewc_state.get("theta_ref")
    fisher = ewc_state.get("fisher")
    if not isinstance(params, list) or not isinstance(theta_ref, dict) or not isinstance(fisher, dict):
        return
    param_names = [name for name, _ in params if isinstance(name, str)]
    payload = {
        "version": 1,
        "signature_hash": _signature_hash(signature),
        "signature": signature,
        "created_at": float(time.time()),
        "param_names": param_names,
        "theta_ref": {k: v.detach().cpu().float() for k, v in theta_ref.items() if isinstance(v, torch.Tensor)},
        "fisher": {k: v.detach().cpu().float() for k, v in fisher.items() if isinstance(v, torch.Tensor)},
    }
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    torch.save(payload, cache_path)
    logger.info("Saved EWC cache: %s (tensors=%d)", cache_path, len(param_names))


def _is_ewc_target_param(name: str, target: str) -> bool:
    if target == "all_trainable":
        return True
    if target == "attn_norm_bias":
        return (
            re.search(
                r"(?:^|\.)(?:attn\d+|audio_attn\d+|audio_to_video_attn|video_to_audio_attn)\.(?:q_norm|k_norm)\.weight$",
                name,
            )
            is not None
            or re.search(
                r"(?:^|\.)(?:attn\d+|audio_attn\d+|audio_to_video_attn|video_to_audio_attn)\.(?:to_q|to_k)\.bias$",
                name,
            )
            is not None
        )
    if target == "attn_geometry":
        return _is_attention_geometry_param(name)
    return False


def _ewc_param_priority(name: str) -> tuple[int, int, str]:
    block_idx = _extract_transformer_block_index(name)
    block_sort = int(block_idx) if block_idx is not None else 10**9

    # Prioritize motion-relevant attention structure first.
    if re.search(r"(?:^|\.)(?:attn1|audio_attn1|audio_to_video_attn|video_to_audio_attn)\.(?:to_q|to_k|q_norm|k_norm)\.", name):
        tier = 0
    elif _is_attention_geometry_param(name):
        tier = 1
    elif re.search(r"(?:^|\.)(?:attn1|audio_attn1|audio_to_video_attn|video_to_audio_attn)\.(?:to_v|to_out)\.", name):
        tier = 2
    elif ".ff." in name:
        tier = 3
    elif ".norm" in name:
        tier = 4
    elif block_idx is not None:
        tier = 5
    else:
        tier = 6
    return tier, block_sort, name


def _cap_ewc_selected_params(
    selected: list[tuple[str, torch.nn.Parameter]],
    *,
    target: str,
    max_tensors: int,
) -> list[tuple[str, torch.nn.Parameter]]:
    if max_tensors <= 0 or len(selected) <= max_tensors:
        return selected

    ranked = sorted(selected, key=lambda item: _ewc_param_priority(item[0]))
    if target != "all_trainable":
        return ranked[:max_tensors]

    # For all_trainable: keep motion-relevant priority while spreading coverage across blocks.
    by_block: dict[int, list[tuple[str, torch.nn.Parameter]]] = {}
    non_block: list[tuple[str, torch.nn.Parameter]] = []
    for item in ranked:
        block_idx = _extract_transformer_block_index(item[0])
        if block_idx is None:
            non_block.append(item)
        else:
            by_block.setdefault(int(block_idx), []).append(item)

    picked: list[tuple[str, torch.nn.Parameter]] = []
    active_blocks = sorted(by_block.keys())
    while active_blocks and len(picked) < max_tensors:
        next_active: list[int] = []
        for block_idx in active_blocks:
            bucket = by_block[block_idx]
            if bucket:
                picked.append(bucket.pop(0))
                if len(picked) >= max_tensors:
                    break
            if bucket:
                next_active.append(block_idx)
        active_blocks = next_active

    if len(picked) < max_tensors and non_block:
        remaining = max_tensors - len(picked)
        picked.extend(non_block[:remaining])

    if len(picked) < max_tensors:
        leftovers: list[tuple[str, torch.nn.Parameter]] = []
        for block_idx in sorted(by_block.keys()):
            leftovers.extend(by_block[block_idx])
        if leftovers:
            remaining = max_tensors - len(picked)
            picked.extend(leftovers[:remaining])

    return picked[:max_tensors]


def _filter_ewc_selected_params_for_device(
    selected: list[tuple[str, torch.nn.Parameter]],
    *,
    target_device: Optional[torch.device],
    stage: str,
) -> list[tuple[str, torch.nn.Parameter]]:
    if target_device is None:
        return selected
    kept: list[tuple[str, torch.nn.Parameter]] = []
    dropped = 0
    for name, param in selected:
        if not _device_matches(param.device, target_device):
            dropped += 1
            continue
        kept.append((name, param))
    if dropped > 0:
        logger.info(
            "EWC %s device filter: kept=%d dropped=%d target_device=%s",
            stage,
            len(kept),
            dropped,
            str(target_device),
        )
    if len(kept) == 0 and dropped > 0:
        logger.warning(
            "EWC %s device filter kept 0 tensors on %s; falling back to unfiltered selection.",
            stage,
            str(target_device),
        )
        return selected
    return kept


def _build_fisher_ewc_stats(
    trainer: LTX2NetworkTrainer,
    args: argparse.Namespace,
    accelerator: Accelerator,
    transformer: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    noise_scheduler: Any,
    optimizer: Any,
    target_device: Optional[torch.device] = None,
) -> Optional[dict[str, Any]]:
    ewc_lambda = float(getattr(args, "ewc_lambda", 0.0) or 0.0)
    if ewc_lambda <= 0.0:
        return None

    ewc_num_batches = int(getattr(args, "ewc_num_batches", 8) or 0)
    if ewc_num_batches <= 0:
        logger.warning("ewc_lambda > 0 but ewc_num_batches <= 0. Skipping EWC.")
        return None

    ewc_target = str(getattr(args, "ewc_target", "attn_norm_bias") or "attn_norm_bias")
    if ewc_target not in {"attn_norm_bias", "attn_geometry", "all_trainable"}:
        raise ValueError(f"Invalid ewc_target: {ewc_target}")

    ewc_max_param_tensors = int(getattr(args, "ewc_max_param_tensors", 256) or 0)

    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name, param in transformer.named_parameters():
        if not param.requires_grad:
            continue
        if _is_ewc_target_param(name, ewc_target):
            selected.append((name, param))

    selected = _filter_ewc_selected_params_for_device(
        selected,
        target_device=target_device,
        stage="selection",
    )

    selected = _cap_ewc_selected_params(
        selected,
        target=ewc_target,
        max_tensors=ewc_max_param_tensors,
    )

    if not selected:
        logger.warning("EWC requested but no matching trainable parameters were selected (target=%s).", ewc_target)
        return None
    selected_block_count = len({idx for idx in (_extract_transformer_block_index(name) for name, _ in selected) if idx is not None})
    selected_geom_count = sum(1 for name, _ in selected if _is_attention_geometry_param(name))
    logger.info(
        "EWC selected tensors=%d blocks=%d attn_geometry=%d",
        len(selected),
        selected_block_count,
        selected_geom_count,
    )

    logger.info(
        "Building Fisher/EWC stats on %d parameter tensors (target=%s, batches=%d).",
        len(selected),
        ewc_target,
        ewc_num_batches,
    )
    ewc_start_time = time.time()
    pbar_ewc = tqdm(
        total=ewc_num_batches,
        desc="prep: fisher/ewc",
        leave=False,
        disable=not accelerator.is_local_main_process,
    )

    theta_ref = {name: param.detach().cpu().float().clone() for name, param in selected}
    fisher_acc = {name: torch.zeros_like(theta_ref[name]) for name, _ in selected}
    valid_batches = 0
    skipped_batches = 0

    was_training = transformer.training
    original_caption_dropout = float(getattr(args, "caption_dropout_rate", 0.0))
    transformer.eval()
    setattr(args, "caption_dropout_rate", 0.0)

    try:
        for batch in train_dataloader:
            if valid_batches >= ewc_num_batches:
                break
            batch = _normalize_ltx2_batch_for_call_dit(batch)
            latents = batch.get("latents")
            if isinstance(latents, dict):
                latents = latents.get("latents")
            if not isinstance(latents, torch.Tensor) or latents.dim() != 5:
                skipped_batches += 1
                continue

            latents_tensor = trainer.scale_shift_latents(latents)
            noise = torch.randn_like(latents_tensor)
            noisy_model_input, timesteps = trainer.get_noisy_model_input_and_timesteps(
                args,
                noise,
                latents_tensor,
                batch.get("timesteps"),
                noise_scheduler,
                accelerator.device,
                trainer.dit_dtype,
            )
            weighting = compute_loss_weighting_for_sd3(
                args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, trainer.dit_dtype
            )
            model_pred, target = trainer.call_dit(
                args,
                accelerator,
                transformer,
                latents_tensor,
                batch,
                noise,
                noisy_model_input,
                timesteps,
                trainer.dit_dtype,
            )
            if isinstance(model_pred, dict):
                out = model_pred
                if out.get("_skip_step"):
                    skipped_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                _ewc_loss_type = getattr(args, "loss_type", "mse")
                _ewc_huber_delta = float(getattr(args, "huber_delta", 1.0))
                loss = _masked_mse(
                    out["video_pred"],
                    out["video_target"],
                    out.get("video_loss_mask"),
                    weighting=weighting,
                    dtype=trainer.dit_dtype,
                    loss_type=_ewc_loss_type,
                    huber_delta=_ewc_huber_delta,
                ) * float(out.get("video_loss_weight", 1.0))
                audio_pred = out.get("audio_pred")
                audio_target = out.get("audio_target")
                if audio_pred is not None and audio_target is not None:
                    loss = loss + _masked_mse(
                        audio_pred,
                        audio_target,
                        out.get("audio_loss_mask"),
                        weighting=weighting,
                        dtype=trainer.dit_dtype,
                        loss_type=_ewc_loss_type,
                        huber_delta=_ewc_huber_delta,
                    ) * float(out.get("audio_loss_weight", 1.0))
            else:
                _ewc_loss_type = getattr(args, "loss_type", "mse")
                _ewc_huber_delta = float(getattr(args, "huber_delta", 1.0))
                pred = model_pred.to(device=target.device, dtype=trainer.dit_dtype)
                if _ewc_loss_type in ("mae", "l1"):
                    loss = torch.nn.functional.l1_loss(pred, target, reduction="none")
                elif _ewc_loss_type in ("huber", "smooth_l1"):
                    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=_ewc_huber_delta)
                else:
                    loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
                if weighting is not None:
                    w = weighting
                    if isinstance(w, torch.Tensor) and w.dim() != loss.dim():
                        while w.dim() > loss.dim() and w.shape[-1] == 1:
                            w = w.squeeze(-1)
                    loss = loss * w
                loss = loss.mean()

            accelerator.backward(loss)
            for name, param in selected:
                if param.grad is None:
                    continue
                fisher_acc[name].add_(param.grad.detach().float().pow(2).cpu())
            optimizer.zero_grad(set_to_none=True)
            valid_batches += 1
            pbar_ewc.update(1)
            if accelerator.is_local_main_process:
                pbar_ewc.set_postfix(valid=valid_batches, skipped=skipped_batches)
    finally:
        pbar_ewc.close()
        setattr(args, "caption_dropout_rate", original_caption_dropout)
        if was_training:
            transformer.train()
        optimizer.zero_grad(set_to_none=True)

    if valid_batches == 0:
        logger.warning("EWC statistics build produced 0 valid batches; skipping EWC regularization.")
        return None

    for name in list(fisher_acc.keys()):
        fisher_acc[name] = fisher_acc[name].div(float(valid_batches)).clamp_min(1e-12)

    logger.info(
        "EWC stats built with %d valid batches (%d skipped) in %.1fs.",
        valid_batches,
        skipped_batches,
        time.time() - ewc_start_time,
    )
    return {
        "params": [(name, param) for name, param in selected],
        "theta_ref": theta_ref,
        "fisher": fisher_acc,
    }


def _compute_ewc_penalty(
    ewc_state: Optional[dict[str, Any]],
    *,
    dtype: torch.dtype,
    target_device: Optional[torch.device] = None,
) -> tuple[Optional[torch.Tensor], int, int]:
    if not ewc_state:
        return None, 0, 0
    theta_ref = ewc_state["theta_ref"]
    fisher = ewc_state["fisher"]
    if dtype in (torch.float16, torch.bfloat16):
        compute_dtype = dtype
    else:
        compute_dtype = torch.float32
    penalty: Optional[torch.Tensor] = None
    skipped_mismatched_device = 0
    used_params = 0
    for name, param in ewc_state["params"]:
        if target_device is not None and not _device_matches(param.device, target_device):
            skipped_mismatched_device += 1
            continue
        theta = theta_ref[name].to(device=param.device, dtype=compute_dtype, non_blocking=True)
        fisher_w = fisher[name].to(device=param.device, dtype=compute_dtype, non_blocking=True)
        diff = param.to(compute_dtype) - theta
        term = (fisher_w * diff.square()).mean()
        penalty = term if penalty is None else (penalty + term)
        used_params += 1

    if skipped_mismatched_device > 0 and not bool(ewc_state.get("_warned_mixed_device_skip", False)):
        logger.info(
            "EWC runtime: skipped %d/%d tensors not on target device %s (used=%d).",
            skipped_mismatched_device,
            len(ewc_state.get("params", [])),
            str(target_device),
            used_params,
        )
        ewc_state["_warned_mixed_device_skip"] = True

    if penalty is None:
        return None, used_params, skipped_mismatched_device
    if target_device is not None:
        return penalty.to(device=target_device, dtype=torch.float32), used_params, skipped_mismatched_device
    return penalty.to(dtype=torch.float32), used_params, skipped_mismatched_device


def _short_drift_label(name: str) -> str:
    s = re.sub(r"^transformer_blocks\.", "b", name)
    s = s.replace(".weight", "").replace(".bias", "_bias")
    return s


def _build_weight_drift_state(
    args: argparse.Namespace,
    transformer: torch.nn.Module,
) -> Optional[dict[str, Any]]:
    interval = int(getattr(args, "log_weight_drift_every", 0) or 0)
    if interval <= 0:
        return None
    target = str(getattr(args, "weight_drift_target", "attn_geometry") or "attn_geometry")
    if target not in {"attn_norm_bias", "attn_geometry", "all_trainable"}:
        raise ValueError(f"Invalid weight_drift_target: {target}")

    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name, param in transformer.named_parameters():
        if not param.requires_grad:
            continue
        if _is_ewc_target_param(name, target):
            selected.append((name, param))

    if not selected:
        logger.warning("weight_drift: no matching parameters (target=%s); disabled.", target)
        return None

    theta_ref = {name: p.detach().cpu().float().clone() for name, p in selected}
    logger.info(
        "weight_drift: tracking %d tensors (target=%s, every=%d steps, top_k=%d).",
        len(selected),
        target,
        interval,
        int(getattr(args, "weight_drift_top_k", 20) or 20),
    )
    return {
        "theta_ref": theta_ref,
        "params": selected,
        "target": target,
        "interval": interval,
        "top_k": int(getattr(args, "weight_drift_top_k", 20) or 20),
    }


def _register_grad_norm_hooks(
    args: argparse.Namespace,
    transformer: torch.nn.Module,
) -> Optional[dict[str, Any]]:
    interval = int(getattr(args, "log_grad_norm_every", 0) or 0)
    if interval <= 0:
        return None
    target = str(getattr(args, "grad_norm_target", "attn_geometry") or "attn_geometry")
    if target not in {"attn_norm_bias", "attn_geometry", "all_trainable"}:
        raise ValueError(f"Invalid grad_norm_target: {target}")
    top_k = int(getattr(args, "grad_norm_top_k", 20) or 20)

    state: dict[str, Any] = {
        "interval": interval,
        "target": target,
        "top_k": top_k,
        "norms": {},
        "hooks": [],
    }

    n = 0
    for name, param in transformer.named_parameters():
        if not param.requires_grad:
            continue
        if not _is_ewc_target_param(name, target):
            continue

        def _make_hook(pname: str):
            def _hook(p: torch.nn.Parameter) -> None:
                if p.grad is None:
                    return
                try:
                    state["norms"][pname] = float(p.grad.detach().norm(2).item())
                except Exception:
                    pass

            return _hook

        try:
            handle = param.register_post_accumulate_grad_hook(_make_hook(name))
            state["hooks"].append(handle)
            n += 1
        except Exception as exc:
            logger.warning("grad_norm: failed to hook %s: %s", name, exc)

    if n == 0:
        logger.warning("grad_norm: no matching parameters (target=%s); disabled.", target)
        return None

    logger.info(
        "grad_norm: hooked %d tensors (target=%s, every=%d steps, top_k=%d).",
        n,
        target,
        interval,
        top_k,
    )
    return state


def _compute_grad_norm_logs(state: Optional[dict[str, Any]]) -> dict[str, float]:
    if not state:
        return {}
    norms = state.get("norms") or {}
    if not norms:
        return {}
    top_k = int(state.get("top_k", 20))
    items = [(k, v) for k, v in norms.items() if v == v]
    if not items:
        return {}
    values = [v for _, v in items]
    logs: dict[str, float] = {
        "grad_norm/mean": float(sum(values) / len(values)),
        "grad_norm/max": float(max(values)),
        "grad_norm/n_tensors": float(len(values)),
    }
    top_items = sorted(items, key=lambda x: x[1], reverse=True)[:top_k]
    for name, v in top_items:
        logs[f"grad_norm/layer/{_short_drift_label(name)}"] = float(v)
    return logs


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device=device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        seq = [_move_batch_to_device(v, device) for v in batch]
        return type(batch)(seq) if isinstance(batch, tuple) else seq
    return batch


def _move_batch_to_cpu(batch: Any) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.detach().cpu().clone()
    if isinstance(batch, dict):
        return {k: _move_batch_to_cpu(v) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        seq = [_move_batch_to_cpu(v) for v in batch]
        return type(batch)(seq) if isinstance(batch, tuple) else seq
    return batch


def _build_output_drift_state(
    args: argparse.Namespace,
    trainer: Any,
    transformer: torch.nn.Module,
    val_dataloader: Any,
    noise_scheduler: Any,
    accelerator: Any,
) -> Optional[dict[str, Any]]:
    interval = int(getattr(args, "log_output_drift_every", 0) or 0)
    if interval <= 0:
        return None
    if val_dataloader is None:
        logger.warning("output_drift: requires --validation_dataset_config; disabled.")
        return None
    k_batches = int(getattr(args, "output_drift_batches", 1) or 1)
    fixed_t_value = float(getattr(args, "output_drift_timestep", 500.0) or 500.0)

    probes: list[dict[str, Any]] = []
    was_training = transformer.training
    transformer.eval()
    try:
        with torch.no_grad():
            for batch in val_dataloader:
                if len(probes) >= k_batches:
                    break
                batch = _normalize_ltx2_batch_for_call_dit(batch)
                latents = batch.get("latents")
                latents_tensor = latents["latents"] if isinstance(latents, dict) else latents
                if not isinstance(latents_tensor, torch.Tensor):
                    continue
                latents_tensor = trainer.scale_shift_latents(latents_tensor)
                noise = torch.randn_like(latents_tensor)
                bsz = int(noise.shape[0])
                fixed_t = [fixed_t_value] * bsz
                try:
                    noisy_model_input, timesteps = trainer.get_noisy_model_input_and_timesteps(
                        args,
                        noise,
                        latents_tensor,
                        fixed_t,
                        noise_scheduler,
                        accelerator.device,
                        trainer.dit_dtype,
                    )
                    model_pred, _ = trainer.call_dit(
                        args,
                        accelerator,
                        transformer,
                        latents_tensor,
                        batch,
                        noise,
                        noisy_model_input,
                        timesteps,
                        trainer.dit_dtype,
                    )
                except Exception as exc:
                    logger.warning("output_drift: probe build failed for a batch: %s", exc)
                    continue

                ref_pred_cpu: Optional[torch.Tensor] = None
                if isinstance(model_pred, dict):
                    v_pred = model_pred.get("video_pred")
                    if isinstance(v_pred, torch.Tensor):
                        ref_pred_cpu = v_pred.detach().cpu().float().clone()
                elif isinstance(model_pred, torch.Tensor):
                    ref_pred_cpu = model_pred.detach().cpu().float().clone()
                if ref_pred_cpu is None:
                    continue

                probes.append(
                    {
                        "batch_cpu": _move_batch_to_cpu(batch),
                        "latents_cpu": latents_tensor.detach().cpu().clone(),
                        "noise_cpu": noise.detach().cpu().clone(),
                        "timesteps_override": list(fixed_t),
                        "ref_pred_cpu": ref_pred_cpu,
                    }
                )
    finally:
        if was_training:
            transformer.train()

    if not probes:
        logger.warning("output_drift: failed to capture any probe batches; disabled.")
        return None
    logger.info(
        "output_drift: captured %d probe batches at t=%.1f, every=%d steps.",
        len(probes),
        fixed_t_value,
        interval,
    )
    return {"probes": probes, "interval": interval, "fixed_t": fixed_t_value}


def _compute_output_drift_logs(
    state: Optional[dict[str, Any]],
    trainer: Any,
    transformer: torch.nn.Module,
    args: argparse.Namespace,
    noise_scheduler: Any,
    accelerator: Any,
) -> dict[str, float]:
    if not state:
        return {}
    probes = state.get("probes") or []
    if not probes:
        return {}
    device = accelerator.device

    was_training = transformer.training
    transformer.eval()
    mse_vals: list[float] = []
    cos_vals: list[float] = []
    try:
        with torch.no_grad():
            for probe in probes:
                batch = _move_batch_to_device(probe["batch_cpu"], device)
                latents_tensor = probe["latents_cpu"].to(device=device)
                noise = probe["noise_cpu"].to(device=device)
                try:
                    noisy_model_input, timesteps = trainer.get_noisy_model_input_and_timesteps(
                        args,
                        noise,
                        latents_tensor,
                        probe["timesteps_override"],
                        noise_scheduler,
                        device,
                        trainer.dit_dtype,
                    )
                    model_pred, _ = trainer.call_dit(
                        args,
                        accelerator,
                        transformer,
                        latents_tensor,
                        batch,
                        noise,
                        noisy_model_input,
                        timesteps,
                        trainer.dit_dtype,
                    )
                except Exception as exc:
                    logger.warning("output_drift: forward failed: %s", exc)
                    continue

                cur_pred: Optional[torch.Tensor] = None
                if isinstance(model_pred, dict):
                    v_pred = model_pred.get("video_pred")
                    if isinstance(v_pred, torch.Tensor):
                        cur_pred = v_pred.detach()
                elif isinstance(model_pred, torch.Tensor):
                    cur_pred = model_pred.detach()
                if cur_pred is None:
                    continue
                cur_f = cur_pred.to(torch.float32)
                ref = probe["ref_pred_cpu"].to(device=cur_f.device, dtype=torch.float32)
                if cur_f.shape != ref.shape:
                    continue
                mse_vals.append(float((cur_f - ref).square().mean().item()))
                cos = torch.nn.functional.cosine_similarity(
                    cur_f.flatten().unsqueeze(0),
                    ref.flatten().unsqueeze(0),
                    dim=1,
                    eps=1e-8,
                ).item()
                cos_vals.append(float(cos))
    finally:
        if was_training:
            transformer.train()

    if not mse_vals:
        return {}
    return {
        "output_drift/mse": float(sum(mse_vals) / len(mse_vals)),
        "output_drift/cosine": float(sum(cos_vals) / len(cos_vals)),
        "output_drift/n_probes": float(len(mse_vals)),
    }


def _compute_weight_drift_logs(state: Optional[dict[str, Any]]) -> dict[str, float]:
    if not state:
        return {}
    theta_ref = state["theta_ref"]
    top_k = int(state.get("top_k", 20))
    eps = 1e-9

    per_layer: list[tuple[str, float, float]] = []
    for name, param in state["params"]:
        ref = theta_ref.get(name)
        if ref is None:
            continue
        cur = param.detach().cpu().float()
        if cur.shape != ref.shape:
            continue
        ref_norm = ref.norm(2).item()
        cur_norm = cur.norm(2).item()
        delta_norm = (cur - ref).norm(2).item()
        frob_rel = delta_norm / max(ref_norm, eps)
        dot = (cur.flatten() * ref.flatten()).sum().item()
        cos_sim = dot / max(cur_norm * ref_norm, eps)
        per_layer.append((name, frob_rel, cos_sim))

    if not per_layer:
        return {}

    frob_values = [x[1] for x in per_layer]
    cos_values = [x[2] for x in per_layer]
    logs: dict[str, float] = {
        "drift/frob_mean": float(sum(frob_values) / len(frob_values)),
        "drift/frob_max": float(max(frob_values)),
        "drift/cosine_mean": float(sum(cos_values) / len(cos_values)),
        "drift/cosine_min": float(min(cos_values)),
        "drift/n_tensors": float(len(per_layer)),
    }
    top_frob = sorted(per_layer, key=lambda x: x[1], reverse=True)[:top_k]
    top_cos = sorted(per_layer, key=lambda x: x[2])[:top_k]
    for name, frob_rel, _ in top_frob:
        logs[f"drift/frob/{_short_drift_label(name)}"] = float(frob_rel)
    for name, _, cos_sim in top_cos:
        logs[f"drift/cosine/{_short_drift_label(name)}"] = float(cos_sim)
    return logs


def _fused_step_pending_grads(
    optimizer: Any,
    accelerator: Accelerator,
    max_grad_norm: float,
    *,
    ltx2_model_parallel: bool = False,
) -> int:
    """Run one fused-style parameter step for any pending grads.

    Used when fused backward hooks defer stepping on the first backward pass
    and no second backward pass happens.
    """
    params_to_step: list[tuple[torch.Tensor, Any]] = []
    for param_group in optimizer.param_groups:
        for parameter in param_group.get("params", []):
            has_float_grad = getattr(parameter, "float_grad", None) is not None
            if parameter is None or (parameter.grad is None and not has_float_grad):
                continue
            params_to_step.append((parameter, param_group))

    if not params_to_step:
        return 0

    if accelerator.sync_gradients and max_grad_norm != 0.0:
        _clip_grad_norm_ltx2(
            [p for p, _ in params_to_step],
            accelerator,
            max_grad_norm,
            ltx2_model_parallel=ltx2_model_parallel,
            optimizer=optimizer,
        )

    for parameter, param_group in params_to_step:
        optimizer.step_param(parameter, param_group)
        parameter.grad = None
        if getattr(parameter, "float_grad", None) is not None:
            parameter.float_grad = None

    return len(params_to_step)


def _unscale_gradients_if_needed(accelerator: Accelerator, optimizer: Any | None = None) -> None:
    unscale_gradients = getattr(accelerator, "unscale_gradients", None)
    if not callable(unscale_gradients):
        return
    try:
        unscale_gradients(optimizer)
    except TypeError:
        unscale_gradients()


def _clip_grad_norm_ltx2(
    parameters,
    accelerator: Accelerator,
    max_grad_norm: float,
    *,
    ltx2_model_parallel: bool,
    optimizer: Any | None = None,
) -> torch.Tensor | None:
    if max_grad_norm == 0.0:
        return None

    parameter_list = list(parameters)
    if not parameter_list:
        return None

    if ltx2_model_parallel:
        _unscale_gradients_if_needed(accelerator, optimizer)
        return clip_grad_norm_model_parallel(parameter_list, max_grad_norm)

    return accelerator.clip_grad_norm_(parameter_list, max_grad_norm)


def _attach_fused_step_param(optimizer: Any, base_optimizer: Any) -> bool:
    """Expose a base optimizer's per-parameter step on an accelerator wrapper."""
    if callable(getattr(optimizer, "step_param", None)):
        return True

    step_param = getattr(base_optimizer, "step_param", None)
    if not callable(step_param):
        return False

    setattr(optimizer, "step_param", step_param)
    return True


def _parse_block_index_spec(spec: Optional[str]) -> set[int]:
    if not spec:
        return set()

    out: set[int] = set()
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if end < start:
                raise ValueError(f"Invalid block range in freeze_block_indices: {token!r}")
            out.update(range(start, end + 1))
        else:
            out.add(int(token))
    return out


def _parse_block_lr_rules(specs: Optional[list[str]]) -> list[tuple[int, Optional[int], float]]:
    if not specs:
        return []

    rules: list[tuple[int, Optional[int], float]] = []
    for raw in specs:
        text = raw.strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"Invalid block_lr_scales entry {text!r}. Expected format like 0-11:0.1")

        range_part, scale_part = text.split(":", 1)
        scale = float(scale_part.strip())
        if scale < 0.0:
            raise ValueError(f"block_lr_scales must use non-negative scales, got {scale} in {text!r}")

        range_part = range_part.strip()
        if "-" in range_part:
            start_s, end_s = range_part.split("-", 1)
            start = int(start_s.strip())
            end_raw = end_s.strip()
            end: Optional[int] = None if end_raw == "" else int(end_raw)
            if end is not None and end < start:
                raise ValueError(f"Invalid block_lr_scales range {range_part!r} in {text!r}")
            rules.append((start, end, scale))
        else:
            idx = int(range_part)
            rules.append((idx, idx, scale))
    return rules


def _extract_transformer_block_index(param_name: str) -> Optional[int]:
    # Works for both "transformer_blocks.X.*" and "model.transformer_blocks.X.*".
    match = re.search(r"(?:^|\.)(?:model\.)?transformer_blocks\.(\d+)\.", param_name)
    if match is None:
        return None
    return int(match.group(1))


def _format_int_ranges(values: list[int]) -> str:
    if not values:
        return "none"

    ranges: list[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = value
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def _summarize_qgalore_replacement_coverage(replaced_names: list[str] | None) -> dict[str, Any]:
    by_block: dict[int, int] = {}
    non_block = 0
    for name in replaced_names or []:
        block_index = _extract_transformer_block_index(name)
        if block_index is None:
            non_block += 1
            continue
        by_block[block_index] = by_block.get(block_index, 0) + 1

    touched_blocks = sorted(by_block)
    per_block_counts = list(by_block.values())
    return {
        "touched_blocks": touched_blocks,
        "touched_block_count": len(touched_blocks),
        "touched_block_ranges": _format_int_ranges(touched_blocks),
        "per_block_min": min(per_block_counts) if per_block_counts else 0,
        "per_block_max": max(per_block_counts) if per_block_counts else 0,
        "non_block": non_block,
    }


def _resolve_block_lr_scale(block_index: int, rules: list[tuple[int, Optional[int], float]]) -> Optional[float]:
    for start, end, scale in rules:
        if end is None:
            if block_index >= start:
                return scale
        elif start <= block_index <= end:
            return scale
    return None


def _build_full_ft_param_groups(
    transformer: torch.nn.Module,
    base_lr: float,
    *,
    freeze_early_blocks: int,
    freeze_block_indices_spec: Optional[str],
    block_lr_scales_spec: Optional[list[str]],
    non_block_lr_scale: float,
    attn_geometry_lr_scale: float,
    freeze_attn_geometry: bool,
    exclude_param_prefixes: Optional[list[str]] = None,
    qgalore_group_kwargs: Optional[dict[str, Any]] = None,
    apollo_group_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, Any]], list[list[str]], dict[str, Any]]:
    if base_lr <= 0:
        raise ValueError(f"learning_rate must be > 0 for full fine-tune, got {base_lr}")
    if freeze_early_blocks < 0:
        raise ValueError("freeze_early_blocks must be >= 0")
    if non_block_lr_scale < 0.0:
        raise ValueError("non_block_lr_scale must be >= 0")
    if attn_geometry_lr_scale < 0.0:
        raise ValueError("attn_geometry_lr_scale must be >= 0")

    block_lr_rules = _parse_block_lr_rules(block_lr_scales_spec)
    frozen_blocks = set(range(freeze_early_blocks))
    frozen_blocks.update(_parse_block_index_spec(freeze_block_indices_spec))

    grouped_params: dict[float, list[torch.nn.Parameter]] = {}
    grouped_names: dict[float, list[str]] = {}
    qgalore_grouped_params: dict[float, list[torch.nn.Parameter]] = {}
    qgalore_grouped_names: dict[float, list[str]] = {}
    apollo_grouped_params: dict[float, list[torch.nn.Parameter]] = {}
    apollo_grouped_names: dict[float, list[str]] = {}
    frozen_param_count = 0
    trainable_param_count = 0
    qgalore_param_count = 0
    qgalore_param_numel = 0
    apollo_param_count = 0
    apollo_param_numel = 0
    frozen_attn_geometry_count = 0
    trainable_attn_geometry_count = 0
    trainable_by_block: dict[str, int] = {}
    excluded_prefixes = tuple(exclude_param_prefixes or [])

    for name, param in transformer.named_parameters():
        if any(name.startswith(prefix) for prefix in excluded_prefixes):
            continue
        is_qgalore_param = bool(getattr(param, "_qgalore_weight", False))
        if not param.requires_grad and not is_qgalore_param:
            continue

        block_index = _extract_transformer_block_index(name)
        if block_index is not None and block_index in frozen_blocks:
            if param.requires_grad:
                param.requires_grad_(False)
            frozen_param_count += 1
            if _is_attention_geometry_param(name):
                frozen_attn_geometry_count += 1
            continue

        if block_index is None:
            scale = float(non_block_lr_scale)
        else:
            scale = _resolve_block_lr_scale(block_index, block_lr_rules)
            if scale is None:
                scale = 1.0
            trainable_by_block[str(block_index)] = trainable_by_block.get(str(block_index), 0) + 1

        is_attn_geometry = _is_attention_geometry_param(name)
        if is_attn_geometry:
            if freeze_attn_geometry:
                if param.requires_grad:
                    param.requires_grad_(False)
                frozen_param_count += 1
                frozen_attn_geometry_count += 1
                continue
            scale *= float(attn_geometry_lr_scale)

        if scale <= 0.0:
            if param.requires_grad:
                param.requires_grad_(False)
            frozen_param_count += 1
            if is_attn_geometry:
                frozen_attn_geometry_count += 1
            continue

        if is_qgalore_param:
            qgalore_grouped_params.setdefault(scale, []).append(param)
            qgalore_grouped_names.setdefault(scale, []).append(name)
            qgalore_param_count += 1
            qgalore_param_numel += int(param.numel())
        elif apollo_group_kwargs is not None and param.dim() == 2:
            apollo_grouped_params.setdefault(scale, []).append(param)
            apollo_grouped_names.setdefault(scale, []).append(name)
            apollo_param_count += 1
            apollo_param_numel += int(param.numel())
        else:
            grouped_params.setdefault(scale, []).append(param)
            grouped_names.setdefault(scale, []).append(name)
        trainable_param_count += 1
        if is_attn_geometry:
            trainable_attn_geometry_count += 1

    if trainable_param_count == 0:
        raise ValueError("No trainable parameters remain after freeze/lr-scale settings.")

    # Keep order deterministic by scale value.
    scales = sorted(grouped_params.keys())
    apollo_scales = sorted(apollo_grouped_params.keys())
    qgalore_scales = sorted(qgalore_grouped_params.keys())
    param_groups = [{"params": grouped_params[s], "lr": base_lr * s} for s in scales]
    param_name_groups = [grouped_names[s] for s in scales]
    if apollo_grouped_params:
        if not apollo_group_kwargs:
            raise ValueError("APOLLO parameters were found, but apollo_group_kwargs was not provided.")
        for scale in apollo_scales:
            group = {"params": apollo_grouped_params[scale], "lr": base_lr * scale}
            group.update(apollo_group_kwargs)
            param_groups.append(group)
            param_name_groups.append(apollo_grouped_names[scale])
    if qgalore_grouped_params:
        if qgalore_group_kwargs is None:
            raise ValueError("Q-GaLore parameters were found, but qgalore_group_kwargs was not provided.")
        for scale in qgalore_scales:
            group = {"params": qgalore_grouped_params[scale], "lr": base_lr * scale}
            group.update(qgalore_group_kwargs)
            param_groups.append(group)
            param_name_groups.append(qgalore_grouped_names[scale])

    stats = {
        "frozen_param_count": frozen_param_count,
        "trainable_param_count": trainable_param_count,
        "qgalore_param_count": qgalore_param_count,
        "qgalore_param_numel": qgalore_param_numel,
        "apollo_param_count": apollo_param_count,
        "apollo_param_numel": apollo_param_numel,
        "num_lr_groups": len(scales) + len(apollo_scales) + len(qgalore_scales),
        "apollo_lr_scales": apollo_scales,
        "qgalore_lr_scales": qgalore_scales,
        "num_apollo_lr_groups": len(apollo_scales),
        "num_qgalore_lr_groups": len(qgalore_scales),
        "lr_scales": scales + apollo_scales + qgalore_scales,
        "frozen_blocks": sorted(frozen_blocks),
        "block_lr_rules": block_lr_rules,
        "trainable_by_block": trainable_by_block,
        "frozen_attn_geometry_count": frozen_attn_geometry_count,
        "trainable_attn_geometry_count": trainable_attn_geometry_count,
    }
    return param_groups, param_name_groups, stats


class _TemporaryRequiresGrad:
    def __init__(self, parameters: list[torch.nn.Parameter], enabled: bool):
        self.parameters = parameters
        self.enabled = bool(enabled)
        self.original: list[bool] = []

    def __enter__(self) -> "_TemporaryRequiresGrad":
        self.original = []
        for parameter in self.parameters:
            self.original.append(bool(parameter.requires_grad))
            if parameter.requires_grad != self.enabled:
                parameter.requires_grad_(self.enabled)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for parameter, original in zip(self.parameters, self.original):
            if parameter.requires_grad != original:
                parameter.requires_grad_(original)
        return False


def _is_image_prior_ft_motion_param(
    name: str,
    *,
    target: str,
    motion_blocks: set[int],
) -> bool:
    block_index = _extract_transformer_block_index(name)
    in_motion_block = block_index is not None and block_index in motion_blocks

    if target == "attn_norm_bias":
        return _is_ewc_target_param(name, "attn_norm_bias")
    if target == "attn_geometry":
        return _is_attention_geometry_param(name)
    if target == "self_attn_geometry":
        return _is_self_attention_geometry_param(name)
    if target == "motion_blocks":
        return in_motion_block
    if target == "attn_geometry_or_motion_blocks":
        return _is_attention_geometry_param(name) or in_motion_block
    raise ValueError(f"Unknown image_prior_ft_motion_param_target: {target}")


def _build_image_prior_ft_route_state(
    transformer: torch.nn.Module,
    args: argparse.Namespace,
) -> Optional[dict[str, Any]]:
    if not bool(getattr(args, "image_prior_ft", False)):
        return None

    task_route = str(getattr(args, "image_prior_ft_task_route", "appearance") or "appearance").lower()
    motion_route = str(getattr(args, "image_prior_ft_motion_route", "motion") or "motion").lower()
    if task_route == "all" and motion_route == "all":
        return {
            "enabled": False,
            "task_route": task_route,
            "motion_route": motion_route,
            "motion_params": [],
            "appearance_params": [],
            "motion_param_names": [],
            "appearance_param_names": [],
        }

    target = str(getattr(args, "image_prior_ft_motion_param_target", "attn_geometry") or "attn_geometry").lower()
    motion_blocks = _parse_block_index_spec(getattr(args, "image_prior_ft_motion_blocks", None))
    if target in {"motion_blocks", "attn_geometry_or_motion_blocks"} and not motion_blocks:
        if target == "motion_blocks":
            raise ValueError("--image_prior_ft_motion_param_target motion_blocks requires --image_prior_ft_motion_blocks")

    motion_params: list[torch.nn.Parameter] = []
    appearance_params: list[torch.nn.Parameter] = []
    motion_param_names: list[str] = []
    appearance_param_names: list[str] = []

    for name, parameter in transformer.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_image_prior_ft_motion_param(name, target=target, motion_blocks=motion_blocks):
            motion_params.append(parameter)
            motion_param_names.append(name)
        else:
            appearance_params.append(parameter)
            appearance_param_names.append(name)

    if not motion_params and (task_route == "appearance" or motion_route == "motion"):
        raise ValueError(
            "Image-prior full-FT routing selected no motion-prior parameters. "
            "Adjust --image_prior_ft_motion_param_target or --image_prior_ft_motion_blocks."
        )
    if not appearance_params and (task_route == "appearance" or motion_route == "motion"):
        logger.warning("Image-prior full-FT routing selected no appearance-side parameters.")

    return {
        "enabled": True,
        "task_route": task_route,
        "motion_route": motion_route,
        "motion_param_target": target,
        "motion_blocks": sorted(motion_blocks),
        "motion_params": motion_params,
        "appearance_params": appearance_params,
        "motion_param_names": motion_param_names,
        "appearance_param_names": appearance_param_names,
    }


def _image_prior_ft_route_context(
    route_state: Optional[dict[str, Any]],
    objective: str,
):
    if not route_state or not bool(route_state.get("enabled", False)):
        return contextlib.nullcontext()

    if objective == "task" and route_state.get("task_route") == "appearance":
        return _TemporaryRequiresGrad(route_state.get("motion_params", []), False)
    if objective == "motion" and route_state.get("motion_route") == "motion":
        return _TemporaryRequiresGrad(route_state.get("appearance_params", []), False)
    return contextlib.nullcontext()


def _image_prior_ft_snapshot_task_route_grads(
    route_state: Optional[dict[str, Any]],
) -> Optional[list[tuple[torch.nn.Parameter, Optional[torch.Tensor]]]]:
    if not route_state or not bool(route_state.get("enabled", False)):
        return None
    if route_state.get("task_route") != "appearance":
        return None

    snapshot: list[tuple[torch.nn.Parameter, Optional[torch.Tensor]]] = []
    for parameter in route_state.get("motion_params", []):
        grad = parameter.grad
        snapshot.append((parameter, None if grad is None else grad.detach().clone()))
    return snapshot


def _image_prior_ft_restore_task_route_grads(
    snapshot: Optional[list[tuple[torch.nn.Parameter, Optional[torch.Tensor]]]],
) -> int:
    if not snapshot:
        return 0

    restored = 0
    for parameter, grad in snapshot:
        if grad is None:
            if parameter.grad is not None:
                restored += 1
            parameter.grad = None
        else:
            if parameter.grad is None or parameter.grad.data_ptr() != grad.data_ptr():
                restored += 1
            parameter.grad = grad
    return restored


def _image_prior_ft_routing_active(route_state: Optional[dict[str, Any]]) -> bool:
    return bool(route_state and bool(route_state.get("enabled", False)))


def _apply_image_prior_ft_defaults(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "image_prior_ft", False)):
        return

    if str(getattr(args, "ltx_mode", "video") or "video").lower() != "video":
        raise ValueError("--image_prior_ft requires --ltx2_mode video with image-only cached latents.")

    if bool(getattr(args, "image_prior_ft_apply_preset", True)):
        args.motion_preservation = True
        args.motion_attention_preservation = True
        args.motion_prior_require_temporal = True
        if int(getattr(args, "motion_preservation_anchor_cache_size", 0) or 0) <= 0:
            args.motion_preservation_anchor_cache_size = 32
        if int(getattr(args, "motion_preservation_num_sigmas", 1) or 1) == 1 and not getattr(
            args, "motion_preservation_sigma_values", None
        ):
            args.motion_preservation_num_sigmas = 2
        if float(getattr(args, "motion_preservation_multiplier", 0.0) or 0.0) <= 0.0:
            args.motion_preservation_multiplier = 0.5
        if float(getattr(args, "motion_attention_preservation_weight", 0.0) or 0.0) <= 0.0:
            args.motion_attention_preservation_weight = 0.1

    if (
        getattr(args, "image_prior_ft_task_route", "appearance") != "all"
        or getattr(args, "image_prior_ft_motion_route", "motion") != "all"
    ):
        args.motion_preservation_separate_backward = True
        if bool(getattr(args, "fused_backward_pass", False)):
            args.motion_preservation_fused_defer_step = True

    logger.info(
        "Image-prior full-FT enabled: task_route=%s motion_route=%s motion_target=%s strict=%s",
        getattr(args, "image_prior_ft_task_route", "appearance"),
        getattr(args, "image_prior_ft_motion_route", "motion"),
        getattr(args, "image_prior_ft_motion_param_target", "attn_geometry"),
        bool(getattr(args, "image_prior_ft_strict", True)),
    )


def ltx2_finetune_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--fused_backward_pass",
        action="store_true",
        help=(
            "Use fused backward pass for Adafactor, CAME/CAME8bit, SinkSGD, Q-GaLore, APOLLO, "
            "torchao Adam, torch-optimi, or BAdam (with use_gradient_release=True)"
        ),
    )
    # BAdam (block-coordinate Adam wrapper).
    # All wrapper kwargs flow through --optimizer_args as key=value entries
    # (e.g. base_optimizer_type=CAME8Bit switch_block_every=25 use_gradient_release=True).
    # The inner optimizer's kwargs are passed via --base_optimizer_args.
    parser.add_argument(
        "--base_optimizer_args",
        type=str,
        default=None,
        nargs="*",
        help="Inner optimizer kwargs when --optimizer_type=BAdam (e.g. stochastic_rounding=True use_cautious=False).",
    )
    parser.add_argument(
        "--mem_eff_save",
        action="store_true",
        help=(
            "Enable memory efficient saving (saving states requires normal saving, so it takes same amount of memory "
            "even with this option enabled)"
        ),
    )
    parser.add_argument(
        "--no_final_save",
        action="store_true",
        help="Skip the final checkpoint save. Intended for smoke/stability runs; periodic step/epoch saves still work.",
    )
    parser.add_argument(
        "--qgalore_full_ft",
        action="store_true",
        default=False,
        help=(
            "Enable experimental Q-GaLore full fine-tuning for selected LTX-2 Linear weights. "
            "Use with --optimizer_type QGaLoreAdamW8bit."
        ),
    )
    parser.add_argument(
        "--qgalore_targets",
        type=str,
        default="video",
        help=(
            "Comma-separated Q-GaLore target scopes: video, audio, ff, attn, blocks, non_block, all. "
            "Default: video (attn1/attn2/ff inside transformer blocks)."
        ),
    )
    parser.add_argument("--qgalore_rank", type=int, default=256, help="Q-GaLore low-rank gradient rank.")
    parser.add_argument(
        "--qgalore_update_proj_gap",
        type=int,
        default=200,
        help="Q-GaLore SVD/projection refresh interval in optimizer steps.",
    )
    parser.add_argument(
        "--qgalore_scale",
        type=float,
        default=0.25,
        help="Q-GaLore projected update scale. The upstream Q-GaLore recipes commonly use 0.25.",
    )
    parser.add_argument("--qgalore_proj_type", type=str, default="std", help="Q-GaLore projection type. Only std is supported.")
    parser.add_argument(
        "--qgalore_proj_quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantize Q-GaLore projection matrices. Enabled by default.",
    )
    parser.add_argument("--qgalore_proj_bits", type=int, default=4, help="Projection matrix quantization bits.")
    parser.add_argument("--qgalore_proj_group_size", type=int, default=256, help="Projection quantization group size.")
    parser.add_argument("--qgalore_weight_bits", type=int, default=8, help="Q-GaLore Linear weight quantization bits.")
    parser.add_argument(
        "--qgalore_weight_group_size",
        type=int,
        default=0,
        help="Q-GaLore Linear weight quantization granularity: 0 = row-wise; >0 = flattened groups of this many values.",
    )
    parser.add_argument(
        "--qgalore_stochastic_round",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use stochastic rounding when re-quantizing Q-GaLore Linear weights. Enabled by default.",
    )
    parser.add_argument(
        "--qgalore_min_weight_numel",
        type=int,
        default=16384,
        help="Skip Linear weights smaller than this many elements when replacing with Q-GaLore Linear.",
    )
    parser.add_argument(
        "--qgalore_max_modules",
        type=int,
        default=None,
        help="Optional cap on the number of Linear modules replaced by Q-GaLore, useful for smoke tests.",
    )
    parser.add_argument(
        "--qgalore_load_device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help=(
            "Device used for the initial LTX-2 transformer load before Q-GaLore/QAPOLLO Linear replacement. "
            "cuda is faster and is the default; cpu avoids the temporary bf16 GPU load spike."
        ),
    )
    parser.add_argument("--qgalore_cos_threshold", type=float, default=0.4, help="Q-GaLore lazy subspace cosine threshold.")
    parser.add_argument("--qgalore_gamma_proj", type=float, default=2.0, help="Q-GaLore projection-gap growth factor.")
    parser.add_argument("--qgalore_queue_size", type=int, default=5, help="Q-GaLore lazy subspace cosine queue size.")
    parser.add_argument(
        "--qgalore_svd_method",
        type=str,
        default="full",
        choices=["full", "lowrank"],
        help="Projection SVD method. full matches upstream Q-GaLore; lowrank uses torch.svd_lowrank for large LTX matrices.",
    )
    parser.add_argument(
        "--qgalore_svd_oversampling",
        type=int,
        default=32,
        help="Extra randomized-SVD vectors when --qgalore_svd_method=lowrank.",
    )
    parser.add_argument(
        "--qgalore_svd_niter",
        type=int,
        default=1,
        help="Power iterations for --qgalore_svd_method=lowrank.",
    )
    parser.add_argument(
        "--qgalore_dequantize_save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dequantize Q-GaLore Linear weights back to normal checkpoint tensors when saving. Enabled by default.",
    )
    parser.add_argument(
        "--qgalore_streaming_dequantize_save",
        action="store_true",
        help=(
            "When saving a dequantized Q-GaLore/QAPOLLO checkpoint, materialize and write one dequantized Linear "
            "weight at a time instead of building all dense weights in memory. This reduces save-time VRAM peaks."
        ),
    )
    parser.add_argument(
        "--qgalore_streaming_dequantize_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help=(
            "Device used for each temporary dequantized Q-GaLore save tensor when "
            "--qgalore_streaming_dequantize_save is enabled. Default: cpu."
        ),
    )
    parser.add_argument("--apollo_rank", type=int, default=256, help="APOLLO low-rank auxiliary optimizer-state rank.")
    parser.add_argument(
        "--apollo_update_proj_gap",
        type=int,
        default=200,
        help="APOLLO projection refresh interval in optimizer steps.",
    )
    parser.add_argument(
        "--apollo_scale",
        type=float,
        default=1.0,
        help="APOLLO update scale. Upstream APOLLO uses 1.0 for channel-wise APOLLO.",
    )
    parser.add_argument(
        "--apollo_proj",
        type=str,
        default="random",
        choices=["random", "svd"],
        help="APOLLO projection source. random is the upstream APOLLO default; svd uses the GaLore projector.",
    )
    parser.add_argument(
        "--apollo_proj_type",
        type=str,
        default="std",
        choices=["std", "reverse_std", "left", "right"],
        help="APOLLO projection orientation.",
    )
    parser.add_argument(
        "--apollo_scale_type",
        type=str,
        default="channel",
        choices=["channel", "tensor"],
        help="APOLLO gradient scaling granularity. channel is APOLLO; tensor is APOLLO-Mini style.",
    )
    parser.add_argument(
        "--apollo_update_rule",
        type=str,
        default="apollo",
        choices=["apollo", "fira"],
        help=(
            "APOLLO/QAPOLLO update rule. 'apollo' (default) scales the full-rank gradient by the "
            "channel-wise factor. 'fira' applies the exact projected-back Adam update inside the "
            "low-rank subspace and the channel-scaled residual (G - P P^T G) outside it. 'fira' "
            "requires a projector exposing project_back (both --apollo_proj random and svd do)."
        ),
    )
    add_ltx2_model_parallel_args(parser)
    # EMA arguments
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use Exponential Moving Average of model weights for more stable training",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay rate (higher = slower update, more smoothing). Default: 0.9999",
    )
    parser.add_argument(
        "--ema_update_after_step",
        type=int,
        default=100,
        help="Start EMA updates after this many steps. Default: 100",
    )
    parser.add_argument(
        "--ema_update_every",
        type=int,
        default=1,
        help="Update EMA every N steps. Default: 1",
    )
    parser.add_argument(
        "--save_ema_only",
        action="store_true",
        help="When using EMA, only save EMA weights (not training weights)",
    )
    parser.add_argument(
        "--ema_cpu_offload",
        action="store_true",
        help="Store EMA shadow weights on CPU to save GPU memory (slower updates but no extra VRAM)",
    )
    # Validation arguments
    parser.add_argument(
        "--validation_dataset_config",
        type=str,
        default=None,
        help="Path to validation dataset config (TOML). If not set, validation is disabled.",
    )
    parser.add_argument(
        "--validation_extra_configs",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Extra validation datasets for per-category / OOD tracking. Format: "
            "category:path.toml (e.g. motion:val_motion.toml night:val_night.toml). "
            "Each runs after the main validation pass and logs val/<category>/loss."
        ),
    )
    parser.add_argument(
        "--motion_prior_cache_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build/load motion prior anchor cache (synthetic priors from base model) and exit before optimization.",
    )
    parser.add_argument(
        "--motion_prior_require_temporal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When enabled, require at least one multi-frame anchor in cache. "
            "If none are found, motion preservation is disabled (or cache-only mode fails)."
        ),
    )
    # Note: --validate_every_n_steps and --validate_every_n_epochs are already defined in setup_parser_common()
    parser.add_argument(
        "--num_validation_batches",
        type=int,
        default=None,
        help="Number of validation batches to use (None = all)",
    )
    parser.add_argument(
        "--validation_timesteps",
        type=str,
        default=None,
        help=(
            "Comma-separated list of fixed timesteps for multi-timestep validation "
            "(e.g. '100,300,500,700,900'). When set, runs additional validation passes with "
            "each timestep forced, logging val/t<T>/loss per t. Empty = disabled."
        ),
    )
    parser.add_argument(
        "--motion_preservation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Full-FT motion preservation via base-model output replay using cached anchors. "
            "Default anchor source is synthetic priors (no external video required)."
        ),
    )
    parser.add_argument(
        "--motion_preservation_multiplier",
        type=float,
        default=0.5,
        help="Weight for motion preservation loss.",
    )
    parser.add_argument(
        "--motion_preservation_mode",
        type=str,
        default="temporal",
        choices=["temporal", "full"],
        help="temporal: match frame-to-frame deltas. full: match full output tensors.",
    )
    parser.add_argument(
        "--motion_preservation_loss_type",
        type=str,
        default="mse",
        choices=["mse", "l2", "l1", "mae", "huber", "smooth_l1", "cosine", "kl_softmax"],
        help=(
            "Distance between student and teacher motion outputs. mse (default) = L2 on values; "
            "cosine = 1 - cos on channel dim (directional, amplitude-invariant); "
            "kl_softmax = KL over softmax(channel logits) with temperature. "
            "Chunked CPU-teacher path supports mse only; other types fall through to standard path."
        ),
    )
    parser.add_argument(
        "--motion_preservation_kl_temperature",
        type=float,
        default=1.0,
        help="Temperature for kl_softmax motion loss (lower = sharper distributions).",
    )
    parser.add_argument(
        "--motion_preservation_anchor_cache_size",
        type=int,
        default=32,
        help="Number of base-model rehearsal anchors cached at training start.",
    )
    parser.add_argument(
        "--motion_preservation_anchor_cache_auto",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-size motion anchor cache from dataset size using ratio/min/max settings.",
    )
    parser.add_argument(
        "--motion_preservation_anchor_cache_auto_ratio",
        type=float,
        default=0.2,
        help="When auto-size is enabled: target anchor_count ~= ceil(num_train_items * ratio).",
    )
    parser.add_argument(
        "--motion_preservation_anchor_cache_auto_min",
        type=int,
        default=8,
        help="When auto-size is enabled: minimum anchor cache size.",
    )
    parser.add_argument(
        "--motion_preservation_anchor_cache_auto_max",
        type=int,
        default=256,
        help="When auto-size is enabled: maximum anchor cache size.",
    )
    parser.add_argument(
        "--motion_preservation_anchor_cache_path",
        type=str,
        default=None,
        help="Optional on-disk cache file (.pt) for motion anchors. Signature mismatch auto-rebuilds.",
    )
    parser.add_argument(
        "--motion_preservation_anchor_cache_rebuild",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force rebuild motion anchor cache even if cache file exists.",
    )
    parser.add_argument(
        "--motion_preservation_anchor_source",
        type=str,
        default="synthetic",
        choices=["dataset", "synthetic", "hybrid"],
        help=(
            "Anchor source for motion replay. "
            "dataset=replay on cached dataset latents, synthetic=use generated multi-frame motion priors, "
            "hybrid=mix both."
        ),
    )
    parser.add_argument(
        "--motion_preservation_synthetic_frames",
        type=int,
        default=8,
        help="Frame count for synthetic motion prior anchors (used for synthetic/hybrid source).",
    )
    parser.add_argument(
        "--motion_preservation_synthetic_temporal_corr",
        type=float,
        default=0.92,
        help="Temporal correlation for synthetic motion priors in [0, 0.999]. Higher = smoother motion.",
    )
    parser.add_argument(
        "--motion_preservation_synthetic_dataset_mix",
        type=float,
        default=0.25,
        help="For hybrid source, probability of selecting dataset anchors vs synthetic anchors.",
    )
    parser.add_argument(
        "--motion_preservation_synthetic_content_seeded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Seed synthetic motion priors from the actual image latent instead of pure random noise. "
            "The image becomes frame 0 and subsequent frames evolve from it via the AR(1) process, "
            "giving the base model semantically-structured input so its temporal response encodes "
            "content-aware motion priors. Disable with --no-motion_preservation_synthetic_content_seeded "
            "to revert to the original pure-noise behaviour."
        ),
    )
    parser.add_argument(
        "--motion_preservation_warmup_steps",
        type=int,
        default=0,
        help=(
            "Linearly ramp the motion preservation multiplier from 0 to its full value over this many "
            "global optimizer steps. Allows the model to learn appearance freely in early training "
            "before motion constraints tighten. 0 disables warmup (full multiplier from step 0)."
        ),
    )
    parser.add_argument(
        "--motion_preservation_interval",
        type=int,
        default=1,
        help="Apply motion preservation every N micro-steps.",
    )
    parser.add_argument(
        "--motion_preservation_probability",
        type=float,
        default=None,
        help=(
            "If set, apply motion preservation stochastically each micro-step with this probability in [0,1]. "
            "When provided, this overrides --motion_preservation_interval."
        ),
    )
    parser.add_argument(
        "--motion_preservation_num_sigmas",
        type=int,
        default=1,
        help=(
            "Number of sigma points for rehearsal replay per anchor. "
            "1 keeps current behavior; >1 adds multi-sigma trajectory preservation."
        ),
    )
    parser.add_argument(
        "--motion_preservation_sigma_values",
        type=str,
        default=None,
        help=(
            "Optional comma-separated sigma list in [0,1] for multi-sigma replay "
            "(e.g. 0.25,0.7). Overrides --motion_preservation_num_sigmas/min/max."
        ),
    )
    parser.add_argument(
        "--motion_preservation_sigma_min",
        type=float,
        default=0.2,
        help="Lower bound for auto-generated multi-sigma replay schedule.",
    )
    parser.add_argument(
        "--motion_preservation_sigma_max",
        type=float,
        default=0.8,
        help="Upper bound for auto-generated multi-sigma replay schedule.",
    )
    parser.add_argument(
        "--motion_preservation_sigma_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "logsnr"],
        help=(
            "Multi-sigma replay sampling mode. "
            "uniform = equal probability per cached sigma; "
            "logsnr = bias toward mid-noise points via log-SNR weighting."
        ),
    )
    parser.add_argument(
        "--motion_preservation_sigma_sampling_power",
        type=float,
        default=1.0,
        help="Exponent applied to sigma-sampling weights (logsnr mode only).",
    )
    parser.add_argument(
        "--motion_preservation_second_order_weight",
        type=float,
        default=0.0,
        help="Additional weight for second-order temporal replay term (acceleration matching).",
    )
    parser.add_argument(
        "--motion_preservation_teacher_chunk_frames",
        type=int,
        default=0,
        help=(
            "Stream teacher replay targets from CPU in frame chunks to reduce peak VRAM. 0 disables chunking (faster, higher VRAM)."
        ),
    )
    parser.add_argument(
        "--motion_preservation_separate_backward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reduce peak VRAM by running motion replay in a separate backward pass after task-loss backward. "
            "For fused mode, also enable --motion_preservation_fused_defer_step."
        ),
    )
    parser.add_argument(
        "--motion_preservation_fused_defer_step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional fused-mode support for --motion_preservation_separate_backward. "
            "Defers fused per-parameter stepping on the first backward and steps on the replay backward."
        ),
    )
    parser.add_argument(
        "--motion_attention_preservation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regularize sampled self-attention maps on rehearsal anchors against base-model attention maps.",
    )
    parser.add_argument(
        "--motion_attention_preservation_weight",
        type=float,
        default=0.1,
        help="Weight for attention-map preservation loss on rehearsal anchors.",
    )
    parser.add_argument(
        "--motion_attention_preservation_loss",
        type=str,
        default="kl",
        choices=["kl", "mse"],
        help="Loss type for attention-map preservation.",
    )
    parser.add_argument(
        "--motion_attention_preservation_queries",
        type=int,
        default=32,
        help="Number of sampled query tokens per attention map.",
    )
    parser.add_argument(
        "--motion_attention_preservation_keys",
        type=int,
        default=64,
        help="Number of sampled key tokens per attention map.",
    )
    parser.add_argument(
        "--motion_attention_preservation_per_head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture and regularize per-head attention maps (stronger, slower) instead of head-averaged maps.",
    )
    parser.add_argument(
        "--motion_attention_preservation_temperature",
        type=float,
        default=1.0,
        help="Temperature applied to attention distributions before KL/MSE comparison (must be > 0).",
    )
    parser.add_argument(
        "--motion_attention_preservation_symmetric_kl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When KL loss is used, optimize symmetric KL instead of one-way KL.",
    )
    parser.add_argument(
        "--motion_attention_preservation_blocks",
        type=str,
        default=None,
        help="Optional comma/range block filter for attention-map preservation, e.g. 12-23.",
    )
    parser.add_argument(
        "--ewc_lambda",
        type=float,
        default=0.0,
        help="Fisher/EWC regularization weight. 0 disables EWC.",
    )
    parser.add_argument(
        "--ewc_num_batches",
        type=int,
        default=8,
        help=(
            "Number of batches used to estimate Fisher statistics at training start. "
            "Fisher is computed on whatever data is in the training dataloader, so provide a "
            "video-only dataset config if you want motion-focused Fisher statistics."
        ),
    )
    parser.add_argument(
        "--ewc_target",
        type=str,
        default="attn_norm_bias",
        choices=["attn_norm_bias", "attn_geometry", "all_trainable"],
        help=(
            "Parameter subset for EWC. "
            "attn_norm_bias is safest for VRAM/runtime; attn_geometry is stronger; all_trainable is heaviest."
        ),
    )
    parser.add_argument(
        "--ewc_max_param_tensors",
        type=int,
        default=256,
        help="Maximum number of parameter tensors to include in EWC (0 = no cap).",
    )
    parser.add_argument(
        "--ewc_cache_path",
        type=str,
        default=None,
        help=(
            "Optional on-disk cache file (.pt) for EWC Fisher/theta stats. Signature mismatch auto-rebuilds. "
            "Note: Fisher is computed on whatever data is in the training dataloader, so if you want "
            "motion-focused Fisher statistics, provide a video-only dataset config."
        ),
    )
    parser.add_argument(
        "--ewc_cache_rebuild",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force rebuild EWC cache even if cache file exists.",
    )
    parser.add_argument(
        "--log_weight_drift_every",
        type=int,
        default=0,
        help=(
            "Log per-layer weight drift every N steps (0=disabled). Snapshots initial trainable "
            "params on CPU at training start, then computes ||W - W_0||_F / ||W_0||_F and cosine "
            "similarity per tensor."
        ),
    )
    parser.add_argument(
        "--weight_drift_target",
        type=str,
        default="all_trainable",
        choices=["attn_norm_bias", "attn_geometry", "all_trainable"],
        help=(
            "Parameter subset for drift logging. Default 'all_trainable' tracks ALL trainable "
            "tensors (general catastrophic forgetting detection). 'attn_geometry'/'attn_norm_bias' "
            "narrow to motion-relevant subsets (cheaper, reuses EWC classification)."
        ),
    )
    parser.add_argument(
        "--weight_drift_top_k",
        type=int,
        default=20,
        help="Log top-K highest-drifting tensors per metric (Frobenius/cosine) to avoid TB spam.",
    )
    parser.add_argument(
        "--log_grad_norm_every",
        type=int,
        default=0,
        help=(
            "Log per-layer gradient L2 norm every N steps (0=disabled). Uses post-accumulate-grad "
            "hooks so it works with both fused and non-fused backward paths."
        ),
    )
    parser.add_argument(
        "--grad_norm_target",
        type=str,
        default="all_trainable",
        choices=["attn_norm_bias", "attn_geometry", "all_trainable"],
        help=(
            "Parameter subset for grad-norm logging. Default 'all_trainable' tracks ALL trainable "
            "tensors (general catastrophic forgetting detection). Narrow targets reuse EWC classification."
        ),
    )
    parser.add_argument(
        "--grad_norm_top_k",
        type=int,
        default=20,
        help="Log top-K highest grad-norm tensors to avoid TB spam.",
    )
    parser.add_argument(
        "--log_output_drift_every",
        type=int,
        default=0,
        help=(
            "Log output drift vs initial snapshot every N steps (0=disabled). Captures first K "
            "validation batches + their initial predictions at training start; re-runs forward "
            "with identical noise/timestep every N steps and logs MSE + cosine similarity. "
            "Direct behavioural indicator of catastrophic forgetting; requires --validation_dataset_config."
        ),
    )
    parser.add_argument(
        "--output_drift_batches",
        type=int,
        default=1,
        help="Number of validation batches to use as output-drift probes (keep small; each costs CPU memory).",
    )
    parser.add_argument(
        "--output_drift_timestep",
        type=float,
        default=500.0,
        help="Fixed timestep used for probe batches (middle of schedule by default).",
    )
    parser.add_argument(
        "--freeze_early_blocks",
        type=int,
        default=0,
        help="Freeze transformer blocks [0, N) during full fine-tuning to protect base motion priors.",
    )
    parser.add_argument(
        "--freeze_block_indices",
        type=str,
        default=None,
        help="Additional comma-separated block indices/ranges to freeze, e.g. 0-7,10,12-15.",
    )
    parser.add_argument(
        "--block_lr_scales",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Per-block LR scale rules for full fine-tuning. "
            "Format: start-end:scale, start-:scale, or idx:scale. "
            "Examples: 0-11:0.1 12-23:0.4 24-:1.0"
        ),
    )
    parser.add_argument(
        "--non_block_lr_scale",
        type=float,
        default=1.0,
        help="LR scale for non-transformer-block parameters in full fine-tuning.",
    )
    parser.add_argument(
        "--attn_geometry_lr_scale",
        type=float,
        default=1.0,
        help="Additional LR scale for attention geometry params (to_q/to_k/q_norm/k_norm).",
    )
    parser.add_argument(
        "--freeze_attn_geometry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze attention geometry params (to_q/to_k/q_norm/k_norm) during full fine-tuning.",
    )
    parser.add_argument(
        "--freeze_audio_params",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Full-FT only: freeze all params whose name contains 'audio_' (audio_attn1/audio_attn2/"
            "audio_to_video_attn/video_to_audio_attn inside transformer blocks plus audio_adaln_single/"
            "audio_norm_out at the model level). Lets the video branch keep adapting while audio "
            "remains at its pretrained values."
        ),
    )
    parser.add_argument(
        "--audio_param_lr_scale",
        type=float,
        default=1.0,
        help=(
            "Full-FT only: extra LR scale applied to audio_* params (mirrors LoRA's --audio_lr but "
            "expressed as a scale of --learning_rate). Use values < 1 to slow audio learning; set 0 "
            "to freeze (equivalent to --freeze_audio_params)."
        ),
    )
    parser.add_argument(
        "--image_prior_ft",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable Image-only prior-preserving full fine-tuning. The image/task objective is routed away "
            "from motion-prior parameters while base-model synthetic temporal replay preserves motion behavior."
        ),
    )
    parser.add_argument(
        "--image_prior_ft_strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require checked training/validation batches to have single-frame video latents and no audio latents.",
    )
    parser.add_argument(
        "--image_prior_ft_apply_preset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --image_prior_ft is enabled, automatically enable motion preservation, "
            "motion attention preservation, temporal-anchor requirement, and multi-sigma replay defaults."
        ),
    )
    parser.add_argument(
        "--image_prior_ft_task_route",
        type=str,
        default="appearance",
        choices=["all", "appearance"],
        help=(
            "Gradient routing for the image/task loss. 'appearance' blocks task gradients on selected "
            "motion-prior parameters; 'all' leaves the task loss unchanged."
        ),
    )
    parser.add_argument(
        "--image_prior_ft_motion_route",
        type=str,
        default="motion",
        choices=["all", "motion"],
        help=(
            "Gradient routing for motion replay and attention-preservation losses. 'motion' blocks replay "
            "gradients on appearance-side parameters; 'all' applies replay to all trainable parameters."
        ),
    )
    parser.add_argument(
        "--image_prior_ft_motion_param_target",
        type=str,
        default="attn_geometry",
        choices=[
            "attn_norm_bias",
            "attn_geometry",
            "self_attn_geometry",
            "motion_blocks",
            "attn_geometry_or_motion_blocks",
        ],
        help=(
            "Parameter selector treated as the motion-prior side for Image-prior full-FT routing. "
            "Use attn_geometry_or_motion_blocks with --image_prior_ft_motion_blocks for stronger routing."
        ),
    )
    parser.add_argument(
        "--image_prior_ft_motion_blocks",
        type=str,
        default=None,
        help="Optional comma/range transformer block selector for branch-aware motion-prior routing, e.g. 0-11,16.",
    )
    parser.add_argument(
        "--ltx2_finetune_block_swap_mode",
        type=str,
        default="default",
        choices=["default", "linear", "full"],
        help=(
            "Full fine-tuning only: override LTX-2 block swap mode. "
            "'default' preserves the shared LTX2_SWAP_FULL_BLOCK environment/default behaviour, "
            "'linear' keeps only Linear weights swapped to CPU, and 'full' moves full swapped blocks to CPU. "
            "LoRA training does not use this flag."
        ),
    )
    parser.add_argument(
        "--ltx2_finetune_block_swap_mask",
        type=str,
        default="all",
        help=(
            "Full fine-tuning only: comma-separated block-swap mask. In full mode, only matching module groups "
            "inside swapped blocks are offloaded. In linear mode, only matching Linear weights are offloaded. "
            "Valid tokens: all, ff, attn, self_attn, cross_attn, av_cross_attn. "
            "Example: 'ff' offloads only feed-forward modules/Linear weights inside swapped blocks. "
            "LoRA training does not use this flag."
        ),
    )
    _add_full_ft_text_encoder_args(parser)
    parser.add_argument(
        "--save_comfy_format",
        action="store_true",
        default=False,
        help="Rename checkpoint keys from model.X to model.diffusion_model.X for ComfyUI compatibility.",
    )
    parser.add_argument(
        "--save_merged_checkpoint",
        action="store_true",
        default=False,
        help=(
            "Save a merged checkpoint containing finetuned DIT weights plus all non-overlapping keys "
            "(VAE, audio VAE, vocoder, text_embedding_projection, etc.) from the original --ltx2_checkpoint. "
            "Implies --save_comfy_format."
        ),
    )
    return parser


_LTX2_FT_SWAP_MASK_TOKENS = {"all", "ff", "attn", "self_attn", "cross_attn", "av_cross_attn"}
_LTX2_FT_SWAP_MASK_ALIASES = {
    "mlp": "ff",
    "feedforward": "ff",
    "feed_forward": "ff",
    "self": "self_attn",
    "cross": "cross_attn",
    "av": "av_cross_attn",
}


def _normalize_ltx2_finetune_swap_mask(value: str | None) -> str:
    raw_tokens = [token.strip().lower() for token in (value or "all").replace("+", ",").split(",")]
    tokens = [_LTX2_FT_SWAP_MASK_ALIASES.get(token, token) for token in raw_tokens if token]
    if not tokens:
        return "all"

    invalid = sorted(set(tokens) - _LTX2_FT_SWAP_MASK_TOKENS)
    if invalid:
        raise ValueError(
            "--ltx2_finetune_block_swap_mask contains invalid token(s): "
            f"{', '.join(invalid)}. Valid tokens: {', '.join(sorted(_LTX2_FT_SWAP_MASK_TOKENS))}"
        )
    if "all" in tokens:
        return "all"
    return ",".join(dict.fromkeys(tokens))


def _prepare_state_dict_for_save(state_dict: dict, args) -> tuple[dict, dict[str, str] | None]:
    """Optionally remap keys to ComfyUI format and merge with original checkpoint.

    Returns:
        (state_dict, extra_metadata) where extra_metadata may contain the original
        checkpoint's ``config`` entry (or None if no merging was requested).
    """
    if not getattr(args, "save_comfy_format", False) and not getattr(args, "save_merged_checkpoint", False):
        return state_dict, None

    # Rename keys: model.X -> model.diffusion_model.X
    renamed: dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_key = "model.diffusion_model." + key[len("model.") :]
            renamed[new_key] = value
        else:
            renamed[key] = value

    extra_metadata: dict[str, str] | None = None

    if getattr(args, "save_merged_checkpoint", False):
        ckpt_path = args.ltx2_checkpoint
        logger.info(f"Merging finetuned weights with original checkpoint: {ckpt_path}")
        with MemoryEfficientSafeOpen(ckpt_path) as f:
            all_keys = f.keys()
            missing_keys = [k for k in all_keys if k not in renamed]
            # Restore original dtypes for overlapping keys (e.g. scale_shift_table F32 cast to BF16 by --full_bf16)
            _st_to_torch = {"F32": "torch.float32", "F16": "torch.float16", "BF16": "torch.bfloat16"}
            dtype_fixed = 0
            for key in all_keys:
                if key in renamed:
                    orig_dtype = f.header[key]["dtype"]
                    if _st_to_torch.get(orig_dtype) and str(renamed[key].dtype) != _st_to_torch[orig_dtype]:
                        renamed[key] = renamed[key].to(f.get_tensor(key).dtype)
                        dtype_fixed += 1
            if dtype_fixed:
                logger.info(f"Restored original dtype for {dtype_fixed} overlapping keys")
            # Copy non-overlapping keys (VAE, vocoder, etc.)
            for key in tqdm(missing_keys, desc="Merging original checkpoint keys"):
                renamed[key] = f.get_tensor(key)
            orig_meta = f.metadata()
            if orig_meta and "config" in orig_meta:
                extra_metadata = {"config": orig_meta["config"]}
        logger.info(f"Merged checkpoint has {len(renamed)} keys ({len(missing_keys)} from original)")

    return renamed, extra_metadata


def _setup_ltx2_full_ft_pre_train_hooks(
    args: argparse.Namespace,
    accelerator: Accelerator | None,
    trainer: LTX2NetworkTrainer,
    transformer: torch.nn.Module,
) -> None:
    # Full fine-tuning does not use LTX2NetworkTrainer.pre_train_hook, so
    # opt-in shared hooks that do not add optimizer parameters are installed here.
    trainer._setup_av_cross_grad_surgery(args, accelerator, transformer=transformer, network=None)
    trainer._setup_av_attention_loss_weighting(args, accelerator, transformer=transformer)


def main() -> None:
    parser = setup_parser_common()
    parser = ltx2_setup_parser(parser)
    parser = ltx2_finetune_setup_parser(parser)
    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    trainer = LTX2NetworkTrainer()

    if args.dataset_config is None and getattr(args, "dataset_manifest", None) is None:
        raise ValueError("dataset_config or dataset_manifest is required / dataset_configまたはdataset_manifestが必要です")
    if args.dataset_config is not None and getattr(args, "dataset_manifest", None) is not None:
        logger.info("Both --dataset_config and --dataset_manifest were provided; full fine-tune will use --dataset_manifest.")
    if args.ltx2_checkpoint is None:
        raise ValueError("path to LTX-2 checkpoint is required / LTX-2チェックポイントのパスが必要です")

    if getattr(args, "dit", None) is not None and args.dit != args.ltx2_checkpoint:
        logger.warning("Ignoring --dit for LTX-2; using --ltx2_checkpoint instead")
    args.dit = args.ltx2_checkpoint

    if getattr(args, "vae", None) is not None and args.vae != args.ltx2_checkpoint:
        logger.warning("Ignoring --vae for LTX-2; using --ltx2_checkpoint instead")
    args.vae = args.ltx2_checkpoint

    if getattr(args, "weighting_scheme", None) not in {None, "none"}:
        logger.warning("Ignoring --weighting_scheme for LTX-2; forcing weighting_scheme=none")
    args.weighting_scheme = "none"

    if bool(getattr(args, "qgalore_full_ft", False)):
        from musubi_tuner.optimizers.backends import is_qapollo_optimizer_type
        from musubi_tuner.optimizers.q_galore import is_qgalore_optimizer_type

        if bool(getattr(args, "fp8_base", False)) or bool(getattr(args, "fp8_scaled", False)):
            raise ValueError(
                "--qgalore_full_ft already quantizes selected Linear weights; do not combine it with --fp8_base/--fp8_scaled."
            )
        if bool(getattr(args, "nf4_base", False)):
            raise ValueError("--qgalore_full_ft cannot be combined with --nf4_base.")
        if not getattr(args, "optimizer_type", None):
            args.optimizer_type = "QGaLoreAdamW8bit"
            logger.info("Q-GaLore full-FT: defaulting --optimizer_type QGaLoreAdamW8bit")
        elif not (is_qgalore_optimizer_type(str(args.optimizer_type)) or is_qapollo_optimizer_type(str(args.optimizer_type))):
            raise ValueError("--qgalore_full_ft requires --optimizer_type QGaLoreAdamW8bit or QAPOLLOAdamW")
        if not bool(getattr(args, "fused_backward_pass", False)):
            raise ValueError(
                "--qgalore_full_ft requires --fused_backward_pass so dense per-layer gradients are stepped and released immediately."
            )
        if bool(getattr(args, "fused_backward_pass", False)) and float(getattr(args, "max_grad_norm", 0.0) or 0.0) != 0.0:
            raise ValueError(
                "Q-GaLore fused backward requires --max_grad_norm 0 because uint8 weight float_grad cannot use global clipping."
            )

    if int(getattr(args, "apollo_rank", 256) or 0) <= 0:
        raise ValueError("--apollo_rank must be > 0")
    if int(getattr(args, "apollo_update_proj_gap", 200) or 0) <= 0:
        raise ValueError("--apollo_update_proj_gap must be > 0")
    if float(getattr(args, "apollo_scale", 1.0) or 0.0) <= 0.0:
        raise ValueError("--apollo_scale must be > 0")

    os.environ["LTX2_FULL_FT_OFFLOAD_TRAINABLE_SWAP"] = "1"
    ft_swap_mode = str(getattr(args, "ltx2_finetune_block_swap_mode", "default") or "default").lower()
    raw_swap_mask = getattr(args, "ltx2_finetune_block_swap_mask", "all")
    ft_swap_mask = _normalize_ltx2_finetune_swap_mask(raw_swap_mask)
    os.environ["LTX2_FULL_FT_SWAP_MASK"] = ft_swap_mask
    if ft_swap_mode == "linear":
        os.environ["LTX2_SWAP_FULL_BLOCK"] = "0"
        logger.info("Full fine-tune block swap override: Linear-weight swap mode (mask=%s)", ft_swap_mask)
    elif ft_swap_mode == "full":
        os.environ["LTX2_SWAP_FULL_BLOCK"] = "1"
        logger.info("Full fine-tune block swap override: full-block swap mode (mask=%s)", ft_swap_mask)
    elif ft_swap_mask != "all":
        logger.info(
            "Full fine-tune block swap mask set to %s; it applies when LTX2_SWAP_FULL_BLOCK resolves to either mode",
            ft_swap_mask,
        )

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    short_map = {"v": "video", "a": "audio", "va": "av"}
    if getattr(args, "ltx_mode", None) in short_map:
        args.ltx_mode = short_map[args.ltx_mode]
    if getattr(args, "ltx_mode", "video") == "video":
        all_declared_audio = False
        if getattr(args, "dataset_manifest", None) is not None:
            dataset_manifest_for_mode = config_utils.load_dataset_manifest(args.dataset_manifest)
            all_declared_audio = _all_manifest_datasets_are_audio(dataset_manifest_for_mode)
        elif args.dataset_config is not None:
            user_config = config_utils.load_user_config(args.dataset_config)
            all_declared_audio = _all_declared_datasets_are_audio(user_config)
        if all_declared_audio:
            logger.info("All datasets are audio-only; automatically switching to --ltx2_mode audio")
            args.ltx_mode = "audio"

    _apply_image_prior_ft_defaults(args)

    trainer.handle_model_specific_args(args)
    if getattr(args, "ltx_mode", "video") == "av" and not getattr(args, "av_use_video_prompt_embeds", False):
        logger.info("Enabling av_use_video_prompt_embeds for AV mode compatibility when batches have no audio latents.")
        args.av_use_video_prompt_embeds = True

    if args.motion_preservation and getattr(args, "ltx_mode", "video") != "video":
        logger.warning("motion_preservation is only supported for video mode in full fine-tune. Disabling it.")
        args.motion_preservation = False
    if bool(getattr(args, "motion_prior_cache_only", False)) and getattr(args, "ltx_mode", "video") != "video":
        logger.warning("motion_prior_cache_only is only supported for video mode. Disabling cache-only mode.")
        args.motion_prior_cache_only = False
    if args.motion_preservation and int(args.motion_preservation_interval) <= 0:
        raise ValueError("motion_preservation_interval must be >= 1")
    if args.motion_preservation and args.motion_preservation_probability is not None:
        if float(args.motion_preservation_probability) < 0.0 or float(args.motion_preservation_probability) > 1.0:
            raise ValueError("motion_preservation_probability must be in [0, 1]")
    if args.motion_preservation and int(getattr(args, "motion_preservation_num_sigmas", 1) or 1) <= 0:
        raise ValueError("motion_preservation_num_sigmas must be >= 1")
    if args.motion_preservation and int(getattr(args, "motion_preservation_anchor_cache_size", 0) or 0) < 0:
        raise ValueError("motion_preservation_anchor_cache_size must be >= 0")
    if bool(getattr(args, "motion_preservation_anchor_cache_auto", False)):
        auto_ratio = float(getattr(args, "motion_preservation_anchor_cache_auto_ratio", 0.2) or 0.2)
        auto_min = int(getattr(args, "motion_preservation_anchor_cache_auto_min", 8) or 8)
        auto_max = int(getattr(args, "motion_preservation_anchor_cache_auto_max", 64) or 64)
        if auto_ratio <= 0.0 or auto_ratio > 1.0:
            raise ValueError("motion_preservation_anchor_cache_auto_ratio must be in (0, 1]")
        if auto_min <= 0:
            raise ValueError("motion_preservation_anchor_cache_auto_min must be >= 1")
        if auto_max <= 0:
            raise ValueError("motion_preservation_anchor_cache_auto_max must be >= 1")
        if auto_max < auto_min:
            raise ValueError("motion_preservation_anchor_cache_auto_max must be >= motion_preservation_anchor_cache_auto_min")
    if args.motion_preservation and int(getattr(args, "motion_preservation_teacher_chunk_frames", 0) or 0) < 0:
        raise ValueError("motion_preservation_teacher_chunk_frames must be >= 0")
    if args.motion_preservation:
        anchor_source = str(getattr(args, "motion_preservation_anchor_source", "synthetic") or "synthetic").lower()
        if anchor_source not in {"dataset", "synthetic", "hybrid"}:
            raise ValueError("motion_preservation_anchor_source must be one of: dataset, synthetic, hybrid")
        if int(getattr(args, "motion_preservation_synthetic_frames", 8) or 0) < 2:
            raise ValueError("motion_preservation_synthetic_frames must be >= 2")
        temporal_corr = float(getattr(args, "motion_preservation_synthetic_temporal_corr", 0.92))
        if temporal_corr < 0.0 or temporal_corr > 0.999:
            raise ValueError("motion_preservation_synthetic_temporal_corr must be in [0, 0.999]")
        dataset_mix = float(getattr(args, "motion_preservation_synthetic_dataset_mix", 0.25))
        if dataset_mix < 0.0 or dataset_mix > 1.0:
            raise ValueError("motion_preservation_synthetic_dataset_mix must be in [0, 1]")
        sigma_min = float(getattr(args, "motion_preservation_sigma_min", 0.2))
        sigma_max = float(getattr(args, "motion_preservation_sigma_max", 0.8))
        if sigma_min < 0.0 or sigma_min > 1.0 or sigma_max < 0.0 or sigma_max > 1.0:
            raise ValueError("motion_preservation_sigma_min/max must be in [0, 1]")
        if sigma_max < sigma_min:
            raise ValueError("motion_preservation_sigma_max must be >= motion_preservation_sigma_min")
        sigma_sampling = str(getattr(args, "motion_preservation_sigma_sampling", "uniform") or "uniform").lower()
        if sigma_sampling not in {"uniform", "logsnr"}:
            raise ValueError("motion_preservation_sigma_sampling must be one of: uniform, logsnr")
        sigma_sampling_power = float(getattr(args, "motion_preservation_sigma_sampling_power", 1.0) or 1.0)
        if sigma_sampling_power <= 0.0:
            raise ValueError("motion_preservation_sigma_sampling_power must be > 0")
        second_order_weight = float(getattr(args, "motion_preservation_second_order_weight", 0.0) or 0.0)
        if second_order_weight < 0.0:
            raise ValueError("motion_preservation_second_order_weight must be >= 0")
    if args.motion_preservation and float(args.motion_preservation_multiplier) < 0.0:
        raise ValueError("motion_preservation_multiplier must be >= 0")
    if bool(getattr(args, "motion_preservation_separate_backward", False)) and bool(getattr(args, "fused_backward_pass", False)):
        if not bool(getattr(args, "motion_preservation_fused_defer_step", False)):
            logger.warning(
                "motion_preservation_separate_backward with fused_backward_pass requires "
                "--motion_preservation_fused_defer_step; disabling separate backward."
            )
            args.motion_preservation_separate_backward = False
    if bool(getattr(args, "motion_preservation_fused_defer_step", False)) and not bool(getattr(args, "fused_backward_pass", False)):
        logger.warning("motion_preservation_fused_defer_step is set without fused_backward_pass; ignoring it.")
    if args.motion_attention_preservation and not args.motion_preservation:
        logger.warning("motion_attention_preservation requires motion_preservation anchors. Disabling it.")
        args.motion_attention_preservation = False
    if args.motion_attention_preservation and float(args.motion_attention_preservation_weight) < 0.0:
        raise ValueError("motion_attention_preservation_weight must be >= 0")
    if args.motion_attention_preservation and int(args.motion_attention_preservation_queries) <= 0:
        raise ValueError("motion_attention_preservation_queries must be >= 1")
    if args.motion_attention_preservation and int(args.motion_attention_preservation_keys) <= 0:
        raise ValueError("motion_attention_preservation_keys must be >= 1")
    if args.motion_attention_preservation and float(getattr(args, "motion_attention_preservation_temperature", 1.0) or 1.0) <= 0.0:
        raise ValueError("motion_attention_preservation_temperature must be > 0")
    if args.motion_attention_preservation and bool(getattr(args, "gradient_checkpointing", False)):
        logger.info(
            "motion_attention_preservation with gradient_checkpointing enabled: using native attention-forward capture path."
        )
    if float(getattr(args, "ewc_lambda", 0.0) or 0.0) < 0.0:
        raise ValueError("ewc_lambda must be >= 0")
    if int(getattr(args, "ewc_num_batches", 8) or 0) < 0:
        raise ValueError("ewc_num_batches must be >= 0")
    if int(getattr(args, "ewc_max_param_tensors", 256) or 0) < 0:
        raise ValueError("ewc_max_param_tensors must be >= 0")
    if int(getattr(args, "freeze_early_blocks", 0) or 0) < 0:
        raise ValueError("freeze_early_blocks must be >= 0")
    if float(getattr(args, "non_block_lr_scale", 1.0) or 0.0) < 0.0:
        raise ValueError("non_block_lr_scale must be >= 0")
    if float(getattr(args, "attn_geometry_lr_scale", 1.0) or 0.0) < 0.0:
        raise ValueError("attn_geometry_lr_scale must be >= 0")
    if bool(getattr(args, "freeze_attn_geometry", False)) and float(getattr(args, "attn_geometry_lr_scale", 1.0) or 1.0) != 1.0:
        logger.warning(
            "freeze_attn_geometry is enabled; attn_geometry_lr_scale has no effect on frozen params. "
            "Resetting attn_geometry_lr_scale to 1.0 for clarity."
        )
        args.attn_geometry_lr_scale = 1.0
    if bool(getattr(args, "motion_preservation_anchor_cache_rebuild", False)) and not getattr(
        args, "motion_preservation_anchor_cache_path", None
    ):
        logger.warning(
            "motion_preservation_anchor_cache_rebuild is set but motion_preservation_anchor_cache_path is empty; ignoring rebuild flag."
        )
    if bool(getattr(args, "ewc_cache_rebuild", False)) and not getattr(args, "ewc_cache_path", None):
        logger.warning("ewc_cache_rebuild is set but ewc_cache_path is empty; ignoring rebuild flag.")
    if bool(getattr(args, "motion_prior_cache_only", False)) and not bool(getattr(args, "motion_preservation", False)):
        logger.warning("motion_prior_cache_only requires motion_preservation; enabling motion_preservation for cache build.")
        args.motion_preservation = True

    if getattr(args, "motion_preservation_anchor_cache_path", None):
        args.motion_preservation_anchor_cache_path = _safe_abs_path(args.motion_preservation_anchor_cache_path)
    if getattr(args, "ewc_cache_path", None):
        args.ewc_cache_path = _safe_abs_path(args.ewc_cache_path)

    if args.seed is None:
        args.seed = random.randint(0, 2**32)
    set_seed(args.seed)
    session_id = random.randint(0, 2**32)
    training_started_at = time.time()

    accelerator = prepare_accelerator(args)
    if args.mixed_precision is None:
        args.mixed_precision = accelerator.mixed_precision
    ltx2_model_parallel = is_ltx2_model_parallel_enabled(args)
    ltx2_remote_stage = is_ltx2_remote_stage_enabled(args)
    if ltx2_model_parallel and ltx2_remote_stage:
        raise RuntimeError("--ltx2_model_parallel and --ltx2_remote_stage are experimental paths and cannot be combined")
    if ltx2_model_parallel:
        validate_ltx2_model_parallel_setup(args, accelerator)
    if ltx2_remote_stage:
        if int(getattr(accelerator, "num_processes", 1)) != 1:
            raise RuntimeError("LTX2 remote stage is single-process only; use accelerate --num_processes 1")
        if int(getattr(args, "blocks_to_swap", 0) or 0) > 0:
            raise RuntimeError("LTX2 remote stage is incompatible with --blocks_to_swap in the current implementation")
        if bool(getattr(args, "blockwise_checkpointing", False)):
            raise RuntimeError("LTX2 remote stage is incompatible with --blockwise_checkpointing in the current implementation")
        if bool(getattr(args, "compile", False)):
            raise RuntimeError("LTX2 remote stage is incompatible with --compile in the current implementation")
        validate_ltx2_remote_stage_setup(args)

    # sample prompts (optional)
    sample_parameters = None
    vae = None
    if (
        args.sample_prompts
        or getattr(args, "precache_sample_prompts", False)
        or getattr(args, "use_precached_sample_prompts", False)
    ):
        sample_prompt_path = args.sample_prompts or ""
        sample_parameters = trainer.process_sample_prompts(args, accelerator, sample_prompt_path)
        vae = trainer.load_vae(args, vae_dtype=model_utils.str_to_dtype(args.vae_dtype), vae_path=args.vae)
        vae.requires_grad_(False)
        vae.eval()

    # datasets
    current_epoch = Value("i", 0)
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    manifest_validation_dataset_group = None
    if getattr(args, "dataset_manifest", None) is not None:
        logger.info("Load dataset manifest from %s", args.dataset_manifest)
        dataset_manifest = config_utils.load_dataset_manifest(args.dataset_manifest)
        manifest_architecture = dataset_manifest.get("architecture")
        if manifest_architecture is not None and manifest_architecture != trainer.architecture:
            raise ValueError(
                f"dataset manifest architecture mismatch: expected '{trainer.architecture}', got '{manifest_architecture}'"
            )
        train_dataset_group = config_utils.generate_dataset_group_by_manifest(
            dataset_manifest,
            split="train",
            training=True,
            num_timestep_buckets=args.num_timestep_buckets,
            shared_epoch=current_epoch,
            reference_downscale=getattr(args, "reference_downscale", 1),
        )
        if train_dataset_group is None:
            raise ValueError("dataset manifest contains no training datasets")
        manifest_validation_dataset_group = config_utils.generate_dataset_group_by_manifest(
            dataset_manifest,
            split="validation",
            # Cache-only validation manifests still need bucket preparation; the DataLoader below stays non-shuffled.
            training=True,
            num_timestep_buckets=args.num_timestep_buckets,
            shared_epoch=current_epoch,
            reference_downscale=getattr(args, "reference_downscale", 1),
        )
    else:
        user_config = config_utils.load_user_config(args.dataset_config)
        blueprint = blueprint_generator.generate(user_config, args, architecture=trainer.architecture)
        train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
            blueprint.dataset_group,
            training=True,
            num_timestep_buckets=args.num_timestep_buckets,
            shared_epoch=current_epoch,
            reference_downscale=getattr(args, "reference_downscale", 1),
        )

    if train_dataset_group.num_train_items == 0:
        raise ValueError(
            "No training items found in the dataset. Please ensure that the latent/Text Encoder cache has been created beforehand."
            " / データセットに学習データがありません。latent/Text Encoderキャッシュを事前に作成したか確認してください"
        )
    if bool(getattr(args, "motion_preservation", False)):
        requested_anchor_size = int(getattr(args, "motion_preservation_anchor_cache_size", 0) or 0)
        resolved_anchor_size = _resolve_motion_anchor_cache_size(
            args,
            num_train_items=int(train_dataset_group.num_train_items),
        )
        args.motion_preservation_anchor_cache_size = int(resolved_anchor_size)
        if bool(getattr(args, "motion_preservation_anchor_cache_auto", False)):
            logger.info(
                "Motion anchor cache size (auto): dataset_items=%d ratio=%.3f min=%d max=%d -> %d",
                int(train_dataset_group.num_train_items),
                float(getattr(args, "motion_preservation_anchor_cache_auto_ratio", 0.2) or 0.2),
                int(getattr(args, "motion_preservation_anchor_cache_auto_min", 8) or 8),
                int(getattr(args, "motion_preservation_anchor_cache_auto_max", 64) or 64),
                int(resolved_anchor_size),
            )
        elif requested_anchor_size != resolved_anchor_size:
            logger.info(
                "Motion anchor cache size adjusted: requested=%d -> resolved=%d",
                int(requested_anchor_size),
                int(resolved_anchor_size),
            )

    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = collator_class(current_epoch, ds_for_collator)
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())

    accumulation_sampler, accumulation_sampler_stats = build_accumulation_group_sampler(
        dataset_group=train_dataset_group,
        group_by=getattr(args, "accumulation_group_by", "none"),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        num_processes=int(accelerator.num_processes),
        remainder=getattr(args, "accumulation_group_remainder", "drop"),
        seed=int(args.seed),
        shared_epoch=current_epoch,
        logger=logger,
    )
    train_audio_sampler = None
    train_audio_sampler_mode = None
    train_audio_sampler_stats: dict[str, Any] = {}
    audio_sampler_requested = (
        int(getattr(args, "min_audio_batches_per_accum", 0) or 0) > 0 or getattr(args, "audio_batch_probability", None) is not None
    )
    if accumulation_sampler is not None and audio_sampler_requested:
        raise ValueError(
            "--min_audio_batches_per_accum / --audio_batch_probability cannot be combined with "
            "--accumulation_group_by. Disable accumulation grouping or use dataset repeats/loss balancing instead."
        )

    if accumulation_sampler is None:
        if int(args.gradient_accumulation_steps) > 1 and int(accumulation_sampler_stats.get("bucket_groups", 0)) > 1:
            logger.warning(
                "gradient_accumulation_steps=%d with %d dataset bucket groups: accumulation windows may mix "
                "frame counts/resolutions. Use --accumulation_group_by bucket for opt-in grouped windows.",
                int(args.gradient_accumulation_steps),
                int(accumulation_sampler_stats["bucket_groups"]),
            )

        train_audio_sampler, train_audio_sampler_mode, train_audio_sampler_stats = build_audio_sampler(
            dataset_group=train_dataset_group,
            gradient_accumulation_steps=int(args.gradient_accumulation_steps),
            min_audio_batches_per_accum=int(getattr(args, "min_audio_batches_per_accum", 0) or 0),
            audio_batch_probability=getattr(args, "audio_batch_probability", None),
            seed=int(args.seed),
        )
        if train_audio_sampler_mode == "quota":
            logger.info(
                "Audio quota sampler enabled: min_audio_batches_per_accum=%d, accumulation_steps=%d, "
                "audio_batches=%d, non_audio_batches=%d",
                train_audio_sampler_stats["min_audio_batches_per_accum"],
                train_audio_sampler_stats["accumulation_steps"],
                train_audio_sampler_stats["audio_batches"],
                train_audio_sampler_stats["non_audio_batches"],
            )
        elif train_audio_sampler_mode == "probability":
            logger.info(
                "Audio probability sampler enabled: audio_batch_probability=%.3f, audio_batches=%d, non_audio_batches=%d",
                train_audio_sampler_stats["audio_batch_probability"],
                train_audio_sampler_stats["audio_batches"],
                train_audio_sampler_stats["non_audio_batches"],
            )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=train_audio_sampler is None,
            sampler=train_audio_sampler,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )
    else:
        logger.info(
            "Accumulation group sampler enabled: group_by=%s remainder=%s groups=%d "
            "window_size=%d original_batches=%d planned_batches=%d delta=%+d",
            accumulation_sampler_stats["group_by"],
            accumulation_sampler_stats["remainder"],
            int(accumulation_sampler_stats["groups"]),
            int(accumulation_sampler_stats["window_size"]),
            int(accumulation_sampler_stats["original_batches"]),
            int(accumulation_sampler_stats["planned_batches"]),
            int(accumulation_sampler_stats["dropped_or_added_batches"]),
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=False,
            sampler=accumulation_sampler,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )
    train_dataset_group_for_audio_sampler = train_dataset_group if train_audio_sampler is not None else None

    # Validation dataset (optional)
    val_dataloader = None
    val_dataset_group = None
    if args.validation_dataset_config is not None:
        logger.info("Loading validation dataset from: %s", args.validation_dataset_config)
        val_user_config = config_utils.load_user_config(args.validation_dataset_config)
        val_blueprint = blueprint_generator.generate(val_user_config, args, architecture=trainer.architecture)
        val_dataset_group = config_utils.generate_dataset_group_by_blueprint(
            val_blueprint.dataset_group,
            # Cache-only validation datasets still need bucket preparation; the DataLoader below stays non-shuffled.
            training=True,
            num_timestep_buckets=args.num_timestep_buckets,
            shared_epoch=current_epoch,
            reference_downscale=getattr(args, "reference_downscale", 1),
        )
        if val_dataset_group.num_train_items > 0:
            val_collator = collator_class(current_epoch, val_dataset_group if args.max_data_loader_n_workers == 0 else None)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset_group,
                batch_size=1,
                shuffle=False,
                collate_fn=val_collator,
                num_workers=n_workers,
                persistent_workers=False,
            )
            logger.info("Validation dataset loaded with %d items", val_dataset_group.num_train_items)
        else:
            logger.warning("Validation dataset has no items, validation disabled")
    elif manifest_validation_dataset_group is not None:
        val_dataset_group = manifest_validation_dataset_group
        if val_dataset_group.num_train_items > 0:
            val_collator = collator_class(current_epoch, val_dataset_group if args.max_data_loader_n_workers == 0 else None)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset_group,
                batch_size=1,
                shuffle=False,
                collate_fn=val_collator,
                num_workers=n_workers,
                persistent_workers=False,
            )
            logger.info("Validation dataset loaded from manifest with %d items", val_dataset_group.num_train_items)
        else:
            logger.warning("Manifest validation dataset has no items, validation disabled")

    _run_image_prior_ft_preflight(args, train_dataset_group, split_name="train")
    _run_full_ft_ic_preflight(args, train_dataset_group, split_name="train")
    if val_dataset_group is not None:
        _run_image_prior_ft_preflight(args, val_dataset_group, split_name="validation")
        _run_full_ft_ic_preflight(args, val_dataset_group, split_name="validation")

    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )

    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # model
    blocks_to_swap = int(getattr(args, "blocks_to_swap", 0) or 0)
    trainer.blocks_to_swap = blocks_to_swap
    remote_prune_local_blocks = ltx2_remote_stage and bool(getattr(args, "ltx2_remote_stage_prune_local_blocks", False))
    qgalore_load_device = str(getattr(args, "qgalore_load_device", "cuda") or "cuda").lower()
    if qgalore_load_device not in {"cuda", "cpu"}:
        raise ValueError(f"--qgalore_load_device must be 'cuda' or 'cpu', got {qgalore_load_device!r}")
    args.qgalore_load_device = qgalore_load_device
    qgalore_cpu_load = bool(getattr(args, "qgalore_full_ft", False)) and qgalore_load_device == "cpu"
    loading_device = (
        "cpu" if blocks_to_swap > 0 or ltx2_model_parallel or remote_prune_local_blocks or qgalore_cpu_load else accelerator.device
    )
    if qgalore_cpu_load:
        logger.info("Q-GaLore CPU load enabled: load/replace/quantize transformer on CPU before moving to %s", accelerator.device)

    if args.sdpa:
        attn_mode = "torch"
    elif args.flash_attn:
        attn_mode = "flash"
    elif args.flash3:
        attn_mode = "flash3"
    elif args.xformers:
        attn_mode = "xformers"
    else:
        attn_mode = "torch"

    transformer = trainer.load_transformer(
        accelerator=accelerator,
        args=args,
        dit_path=args.ltx2_checkpoint,
        attn_mode=attn_mode,
        split_attn=bool(getattr(args, "split_attn", False)),
        loading_device=loading_device,
        dit_weight_dtype=None,
    )

    transformer.train()
    transformer.requires_grad_(True)
    qgalore_summary = None
    if bool(getattr(args, "qgalore_full_ft", False)):
        from musubi_tuner.optimizers.q_galore import replace_ltx2_linear_with_qgalore

        qgalore_weight_group_size = getattr(args, "qgalore_weight_group_size", 0)
        if qgalore_weight_group_size is None:
            qgalore_weight_group_size = 0
        qgalore_summary = replace_ltx2_linear_with_qgalore(
            transformer,
            targets=getattr(args, "qgalore_targets", "video"),
            weight_bits=int(getattr(args, "qgalore_weight_bits", 8) or 8),
            weight_group_size=int(qgalore_weight_group_size),
            stochastic_round=bool(getattr(args, "qgalore_stochastic_round", True)),
            min_weight_numel=int(getattr(args, "qgalore_min_weight_numel", 16384) or 0),
            max_modules=getattr(args, "qgalore_max_modules", None),
        )
        logger.info(
            "Q-GaLore full-FT: replaced %d Linear weights (%.3fB params) targets=%s skipped_not_target=%d skipped_small=%d skipped_group_size=%d",
            qgalore_summary.replaced,
            float(qgalore_summary.replaced_numel) / 1_000_000_000.0,
            getattr(args, "qgalore_targets", "video"),
            qgalore_summary.skipped_not_target,
            qgalore_summary.skipped_small,
            qgalore_summary.skipped_group_size,
        )
        qgalore_coverage = _summarize_qgalore_replacement_coverage(qgalore_summary.replaced_names)
        logger.info(
            "Q-GaLore full-FT coverage: touched_blocks=%d ranges=%s linears_per_touched_block=%d-%d non_block_linears=%d",
            qgalore_coverage["touched_block_count"],
            qgalore_coverage["touched_block_ranges"],
            qgalore_coverage["per_block_min"],
            qgalore_coverage["per_block_max"],
            qgalore_coverage["non_block"],
        )
        if qgalore_summary.replaced <= 0:
            raise ValueError(
                "Q-GaLore full-FT did not replace any Linear modules; check --qgalore_targets and --qgalore_min_weight_numel."
            )

    fp8_gemm_summary = None
    if bool(getattr(args, "fp8_gemm", False)):
        if bool(getattr(args, "qgalore_full_ft", False)):
            raise ValueError("--fp8_gemm is mutually exclusive with --qgalore_full_ft (both replace the same Linear layers).")
        if bool(getattr(args, "fp8_scaled", False)):
            raise ValueError(
                "--fp8_gemm is mutually exclusive with --fp8_scaled (fp8_scaled is weight-storage for inference/LoRA, not FP8 training)."
            )
        if ltx2_model_parallel:
            raise ValueError("--fp8_gemm is not yet supported together with --ltx2_model_parallel.")
        from musubi_tuner.modules.fp8_training import (
            assert_fp8_training_supported,
            convert_ltx2_to_fp8_training,
            resolve_fp8_dtype,
        )

        assert_fp8_training_supported()
        fp8_gemm_summary = convert_ltx2_to_fp8_training(
            transformer,
            targets=getattr(args, "fp8_gemm_targets", "video"),
            grad_dtype=resolve_fp8_dtype(getattr(args, "fp8_gemm_grad_dtype", "e4m3")),
            min_weight_numel=int(getattr(args, "fp8_gemm_min_numel", 16384) or 0),
            compile_gemm=bool(getattr(args, "fp8_gemm_compile", True)),
        )
        logger.info(
            "FP8 full-FT: replaced %d Linear layers (%.3fB params) targets=%s grad_dtype=%s compile=%s "
            "skipped_not_target=%d skipped_dims=%d skipped_small=%d",
            fp8_gemm_summary.replaced,
            float(fp8_gemm_summary.replaced_numel) / 1_000_000_000.0,
            getattr(args, "fp8_gemm_targets", "video"),
            getattr(args, "fp8_gemm_grad_dtype", "e4m3"),
            bool(getattr(args, "fp8_gemm_compile", True)),
            fp8_gemm_summary.skipped_not_target,
            fp8_gemm_summary.skipped_dims,
            fp8_gemm_summary.skipped_small,
        )
        if fp8_gemm_summary.replaced <= 0:
            raise ValueError("--fp8_gemm did not replace any Linear modules; check --fp8_gemm_targets / --fp8_gemm_min_numel.")

    int8_weights_summary = None
    if bool(getattr(args, "int8_weights", False)):
        if bool(getattr(args, "fp8_gemm", False)):
            raise ValueError("--int8_weights is mutually exclusive with --fp8_gemm.")
        if bool(getattr(args, "qgalore_full_ft", False)):
            raise ValueError("--int8_weights is mutually exclusive with --qgalore_full_ft (both replace the same Linear weights).")
        if bool(getattr(args, "fp8_scaled", False)):
            raise ValueError("--int8_weights is mutually exclusive with --fp8_scaled.")
        if ltx2_model_parallel:
            raise ValueError("--int8_weights is not yet supported together with --ltx2_model_parallel.")
        if not bool(getattr(args, "fused_backward_pass", False)):
            raise ValueError(
                "--int8_weights requires --fused_backward_pass; otherwise full bf16 gradients for all weights "
                "materialize at once (~2 bytes/param) and negate the int8 weight saving."
            )
        from musubi_tuner.modules.int8_training import convert_to_int8_training
        from musubi_tuner.modules.fp8_training import ltx2_fp8_filter

        _i8_min = int(getattr(args, "int8_weights_min_numel", 16384) or 0)
        _i8_target = ltx2_fp8_filter(getattr(args, "int8_weights_targets", "video"), _i8_min)

        def _i8_keep(mod, fqn):
            return bool(_i8_target(mod, fqn)) and mod.weight.numel() >= _i8_min

        _i8_group = int(getattr(args, "int8_weights_group_size", 0) or 0)
        _i8_outlier_q = float(getattr(args, "int8_weights_outlier_quantile", 1.0) or 1.0)
        _i8_sparse = float(getattr(args, "int8_weights_sparse_ratio", 0.0) or 0.0)
        int8_weights_summary = convert_to_int8_training(
            transformer,
            filter_fn=_i8_keep,
            group_size=_i8_group,
            outlier_clip_quantile=_i8_outlier_q,
            sparse_ratio=_i8_sparse,
        )
        logger.info(
            "int8 weight-only QT: replaced %d Linear layers targets=%s group_size=%d outlier_clip_quantile=%g sparse_ratio=%g (1 byte/param, stochastic-rounding updates)",
            int8_weights_summary,
            getattr(args, "int8_weights_targets", "video"),
            _i8_group,
            _i8_outlier_q,
            _i8_sparse,
        )
        if int8_weights_summary <= 0:
            raise ValueError(
                "--int8_weights did not replace any Linear modules; check --int8_weights_targets / --int8_weights_min_numel."
            )

    ltx2_model_parallel_plan = None
    if ltx2_model_parallel:
        ltx2_model_parallel_plan = enable_ltx2_model_parallel(transformer, args)
    if ltx2_remote_stage:
        enable_ltx2_remote_stage(transformer, args)
        prune_ltx2_remote_stage_local_blocks(transformer, args)
        if remote_prune_local_blocks:
            transformer.to(accelerator.device)

    # Clean up memory after model loading
    clean_memory_on_device(accelerator.device)

    if blocks_to_swap > 0:
        logger.info("enable swap %s blocks to CPU from device: %s", blocks_to_swap, accelerator.device)
        transformer.enable_block_swap(
            blocks_to_swap,
            accelerator.device,
            supports_backward=True,
            use_pinned_memory=getattr(args, "use_pinned_memory_for_block_swap", False),
        )
        transformer.move_to_device_except_swap_blocks(accelerator.device)

    if args.gradient_checkpointing:
        blocks_to_ckpt = getattr(args, "blocks_to_checkpoint", -1)
        if getattr(args, "blockwise_checkpointing", False):
            transformer.enable_gradient_checkpointing(
                args.gradient_checkpointing_cpu_offload,
                weight_cpu_offloading=True,
                blocks_to_checkpoint=blocks_to_ckpt,
            )
            if args.use_pinned_memory_for_block_swap and hasattr(transformer, "transformer_blocks"):
                for block in transformer.transformer_blocks:
                    if hasattr(block, "use_pinned_memory"):
                        block.use_pinned_memory = True
        else:
            transformer.enable_gradient_checkpointing(
                args.gradient_checkpointing_cpu_offload,
                blocks_to_checkpoint=blocks_to_ckpt,
            )

    # Cache-only motion prior build path: do not initialize optimizer/scheduler state.
    if bool(getattr(args, "motion_prior_cache_only", False)):
        motion_attention_modules: list[tuple[str, torch.nn.Module]] = []
        if args.motion_attention_preservation:
            motion_attention_modules = _collect_motion_attention_modules(
                transformer,
                getattr(args, "motion_attention_preservation_blocks", None),
            )
            motion_attention_modules = _filter_motion_attention_modules_for_swap(
                motion_attention_modules,
                transformer=transformer,
                accelerator=accelerator,
                blocks_to_swap=blocks_to_swap,
            )
            if not motion_attention_modules:
                logger.warning("motion_attention_preservation requested but no matching attn1 modules were found; disabling it.")
                args.motion_attention_preservation = False
            else:
                logger.info(
                    "Motion attention preservation enabled on %d attn1 modules "
                    "(queries=%d keys=%d, loss=%s, per_head=%s, temp=%.3f, symmetric_kl=%s)",
                    len(motion_attention_modules),
                    int(getattr(args, "motion_attention_preservation_queries", 32) or 32),
                    int(getattr(args, "motion_attention_preservation_keys", 64) or 64),
                    getattr(args, "motion_attention_preservation_loss", "kl"),
                    str(bool(getattr(args, "motion_attention_preservation_per_head", False))),
                    float(getattr(args, "motion_attention_preservation_temperature", 1.0) or 1.0),
                    str(bool(getattr(args, "motion_attention_preservation_symmetric_kl", False))),
                )

        noise_scheduler = FlowMatchDiscreteScheduler(
            shift=args.discrete_flow_shift,
            reverse=True,
            solver="euler",
        )
        motion_anchor_cache: list[dict[str, Any]] = []
        if args.motion_preservation:
            motion_anchor_cache = _build_motion_anchor_cache(
                trainer=trainer,
                args=args,
                accelerator=accelerator,
                transformer=transformer,
                train_dataloader=train_dataloader,
                noise_scheduler=noise_scheduler,
                attention_modules=motion_attention_modules,
                normalize_batch_fn=_normalize_ltx2_batch_for_call_dit,
            )
            if not motion_anchor_cache:
                logger.warning("motion_preservation requested but no anchors were built.")
            elif bool(getattr(args, "motion_prior_require_temporal", False)):
                temporal_anchor_count = 0
                for entry in motion_anchor_cache:
                    anchor_latents = entry.get("anchor_latents")
                    if isinstance(anchor_latents, torch.Tensor) and anchor_latents.dim() == 5 and int(anchor_latents.shape[2]) > 1:
                        temporal_anchor_count += 1
                if temporal_anchor_count <= 0:
                    raise ValueError(
                        "motion_prior_require_temporal is enabled, but cache has no multi-frame anchors. "
                        "Set --motion_preservation_anchor_source synthetic or hybrid."
                    )

        logger.info(
            "motion_prior_cache_only enabled: cache build phase finished (anchors=%d). Exiting before optimization.",
            len(motion_anchor_cache),
        )
        return

    # optimizer
    text_encoder = None
    if bool(getattr(args, "full_ft_train_text_encoder", False)):
        logger.info("Preparing Gemma text encoder for full fine-tuning.")
        trainer._build_text_encoder(args, accelerator)
        if trainer._text_encoder is None:
            raise RuntimeError("Failed to initialize text encoder despite --full_ft_train_text_encoder.")
        text_encoder = trainer._text_encoder
        text_encoder_model = getattr(text_encoder, "model", None)
        if (
            bool(getattr(text_encoder, "_has_fp8_model", False))
            or bool(getattr(text_encoder_model, "is_loaded_in_8bit", False))
            or bool(getattr(text_encoder_model, "is_loaded_in_4bit", False))
        ):
            raise ValueError(
                "--full_ft_train_text_encoder requires full-precision Gemma weights. "
                "Do not use gemma fp8/8bit/4bit loading for text-encoder full fine-tuning."
            )
        text_encoder.train()

    if bool(getattr(args, "freeze_audio_params", False)) or float(getattr(args, "audio_param_lr_scale", 1.0) or 0.0) == 0.0:
        # Freeze all audio modules. Catches audio_attn1/audio_attn2/audio_to_video_attn/
        # video_to_audio_attn inside transformer blocks AND model-level audio_adaln_single /
        # audio_norm_out / audio_proj. Setting requires_grad=False makes _build_full_ft_param_groups
        # skip them naturally and also prevents gradient computation.
        _frozen_audio_count = 0
        for _name, _param in transformer.named_parameters():
            if "audio_" in _name:
                _param.requires_grad_(False)
                _frozen_audio_count += 1
        if _frozen_audio_count:
            logger.info(
                "Full-FT freezing audio params: %d tensors with 'audio_' in name set requires_grad=False.",
                _frozen_audio_count,
            )

    from musubi_tuner.optimizers.backends import apollo_group_kwargs_from_args, is_apollo_optimizer_type, is_qapollo_optimizer_type

    optimizer_type_for_groups = str(getattr(args, "optimizer_type", "") or "")
    apollo_group_kwargs = None
    if is_apollo_optimizer_type(optimizer_type_for_groups):
        apollo_group_kwargs = apollo_group_kwargs_from_args(args)

    qgalore_group_kwargs = None
    if bool(getattr(args, "qgalore_full_ft", False)):
        from musubi_tuner.optimizers.q_galore import qgalore_group_kwargs_from_args

        qgalore_group_kwargs = (
            apollo_group_kwargs if is_qapollo_optimizer_type(optimizer_type_for_groups) else qgalore_group_kwargs_from_args(args)
        )

    params_to_optimize, param_names, ft_group_stats = _build_full_ft_param_groups(
        transformer,
        args.learning_rate,
        freeze_early_blocks=int(getattr(args, "freeze_early_blocks", 0) or 0),
        freeze_block_indices_spec=getattr(args, "freeze_block_indices", None),
        block_lr_scales_spec=getattr(args, "block_lr_scales", None),
        non_block_lr_scale=float(getattr(args, "non_block_lr_scale", 1.0) or 0.0),
        attn_geometry_lr_scale=float(getattr(args, "attn_geometry_lr_scale", 1.0) or 0.0),
        freeze_attn_geometry=bool(getattr(args, "freeze_attn_geometry", False)),
        qgalore_group_kwargs=qgalore_group_kwargs,
        apollo_group_kwargs=apollo_group_kwargs,
    )

    # Audio param LR scale: post-process param_groups to extract audio_* params
    # into a separate group with scaled LR. Only applies when scale != 1.0 and
    # audio params aren't already frozen (handled above).
    _audio_lr_scale = float(getattr(args, "audio_param_lr_scale", 1.0) or 1.0)
    if _audio_lr_scale != 1.0 and _audio_lr_scale > 0.0:
        param_to_name: dict[int, str] = {}
        for _name, _param in transformer.named_parameters():
            param_to_name[id(_param)] = _name
        audio_groups: dict[float, list[torch.nn.Parameter]] = {}
        new_groups = []
        new_name_groups = []
        for _grp, _names in zip(params_to_optimize, param_names):
            kept_params = []
            kept_names = []
            for _param, _name in zip(_grp["params"], _names):
                if "audio_" in _name:
                    base_lr = float(_grp.get("lr", args.learning_rate))
                    audio_groups.setdefault(base_lr * _audio_lr_scale, []).append(_param)
                else:
                    kept_params.append(_param)
                    kept_names.append(_name)
            if kept_params:
                new_grp = dict(_grp)
                new_grp["params"] = kept_params
                new_groups.append(new_grp)
                new_name_groups.append(kept_names)
        for scaled_lr in sorted(audio_groups.keys()):
            new_groups.append({"params": audio_groups[scaled_lr], "lr": scaled_lr})
            new_name_groups.append(["<audio_group>"])
        n_audio = sum(len(audio_groups[k]) for k in audio_groups)
        if n_audio > 0:
            logger.info(
                "Full-FT audio_param_lr_scale=%.4f applied: %d audio param tensors moved to separate group(s).",
                _audio_lr_scale,
                n_audio,
            )
            params_to_optimize = new_groups
            param_names = new_name_groups
    if text_encoder is not None:
        text_encoder_lr = getattr(args, "full_ft_text_encoder_lr", None)
        if text_encoder_lr is None:
            text_encoder_lr = float(args.learning_rate)
        text_encoder_lr = float(text_encoder_lr)
        if text_encoder_lr <= 0:
            raise ValueError(f"full_ft_text_encoder_lr must be > 0 when provided, got {text_encoder_lr}")
        text_encoder_params: list[torch.nn.Parameter] = []
        text_encoder_param_names: list[str] = []
        for name, param in text_encoder.named_parameters():
            if not param.requires_grad:
                continue
            text_encoder_params.append(param)
            text_encoder_param_names.append(f"text_encoder.{name}")
        if not text_encoder_params:
            raise ValueError("No trainable text-encoder parameters found; check --full_ft_train_text_encoder inputs.")
        params_to_optimize.append({"params": text_encoder_params, "lr": text_encoder_lr})
        param_names.append(text_encoder_param_names)
        ft_group_stats = dict(ft_group_stats)
        ft_group_stats["text_encoder_param_count"] = int(sum(int(p.numel()) for p in text_encoder_params))
        ft_group_stats["text_encoder_trainable_tensor_count"] = int(len(text_encoder_param_names))
        ft_group_stats["text_encoder_lr"] = text_encoder_lr

    image_prior_ft_route_state = _build_image_prior_ft_route_state(transformer, args)
    if image_prior_ft_route_state is not None:
        ft_group_stats = dict(ft_group_stats)
        ft_group_stats["image_prior_ft_motion_param_count"] = len(image_prior_ft_route_state.get("motion_param_names", []))
        ft_group_stats["image_prior_ft_appearance_param_count"] = len(image_prior_ft_route_state.get("appearance_param_names", []))
        logger.info(
            "Image-prior full-FT routing: task=%s motion=%s target=%s motion_params=%d appearance_params=%d",
            image_prior_ft_route_state.get("task_route", "all"),
            image_prior_ft_route_state.get("motion_route", "all"),
            image_prior_ft_route_state.get("motion_param_target", "none"),
            len(image_prior_ft_route_state.get("motion_param_names", [])),
            len(image_prior_ft_route_state.get("appearance_param_names", [])),
        )

    _setup_ltx2_full_ft_pre_train_hooks(args, accelerator, trainer, transformer)

    self_flow_projector_param_count = 0
    if bool(getattr(args, "self_flow", False)):
        self_flow_args = list(getattr(args, "self_flow_args", None) or [])
        if not any(str(arg).startswith("offload_teacher_params=") for arg in self_flow_args):
            self_flow_args.append("offload_teacher_params=true")
            args.self_flow_args = self_flow_args
            logger.info(
                "Self-Flow full-FT: defaulting self_flow_args offload_teacher_params=true "
                "(override with --self_flow_args offload_teacher_params=false)."
            )

        trainer._setup_self_flow(args, accelerator, transformer=transformer, network=None)
        self_flow_module = getattr(trainer, "_self_flow", None)
        if self_flow_module is not None:
            self_flow_params = [p for p in self_flow_module.get_trainable_params() if p.requires_grad]
            if self_flow_params:
                projector_lr = getattr(getattr(self_flow_module, "config", None), "projector_lr", None)
                effective_projector_lr = float(projector_lr) if projector_lr is not None else float(args.learning_rate)
                params_to_optimize.append({"params": self_flow_params, "lr": effective_projector_lr})
                param_names.append([f"self_flow_projector.{idx}" for idx in range(len(self_flow_params))])
                self_flow_projector_param_count = int(sum(p.numel() for p in self_flow_params))
                logger.info(
                    "Self-Flow full-FT: added projector params to optimizer (count=%d tensors=%d lr=%g)",
                    self_flow_projector_param_count,
                    len(self_flow_params),
                    effective_projector_lr,
                )
            shadow_params = getattr(self_flow_module, "_shadow_params", {})
            shadow_bytes = int(
                sum(int(t.numel()) * int(t.element_size()) for t in shadow_params.values() if isinstance(t, torch.Tensor))
            )
            if shadow_bytes > 0:
                logger.info(
                    "Self-Flow full-FT teacher EMA shadow size: %.2f GB across %d tensors",
                    float(shadow_bytes) / (1024.0**3),
                    len(shadow_params),
                )

    logger.info(
        "Full-FT parameter groups: trainable=%d frozen=%d groups=%d scales=%s",
        ft_group_stats["trainable_param_count"],
        ft_group_stats["frozen_param_count"],
        ft_group_stats["num_lr_groups"],
        ft_group_stats["lr_scales"],
    )
    if bool(getattr(args, "qgalore_full_ft", False)):
        if is_qapollo_optimizer_type(optimizer_type_for_groups):
            logger.info(
                "QAPOLLO quantized Linear groups: tensors=%d params=%.3fB groups=%d scales=%s rank=%d gap=%d scale=%g proj=%s scale_type=%s update_rule=%s",
                int(ft_group_stats.get("qgalore_param_count", 0)),
                float(ft_group_stats.get("qgalore_param_numel", 0)) / 1_000_000_000.0,
                int(ft_group_stats.get("num_qgalore_lr_groups", 0)),
                ft_group_stats.get("qgalore_lr_scales", []),
                int(getattr(args, "apollo_rank", 256)),
                int(getattr(args, "apollo_update_proj_gap", 200)),
                float(getattr(args, "apollo_scale", 1.0)),
                str(getattr(args, "apollo_proj", "random")),
                str(getattr(args, "apollo_scale_type", "channel")),
                str(getattr(args, "apollo_update_rule", "apollo")),
            )
        else:
            logger.info(
                "Q-GaLore optimizer groups: tensors=%d params=%.3fB groups=%d scales=%s rank=%d gap=%d scale=%g",
                int(ft_group_stats.get("qgalore_param_count", 0)),
                float(ft_group_stats.get("qgalore_param_numel", 0)) / 1_000_000_000.0,
                int(ft_group_stats.get("num_qgalore_lr_groups", 0)),
                ft_group_stats.get("qgalore_lr_scales", []),
                int(getattr(args, "qgalore_rank", 256)),
                int(getattr(args, "qgalore_update_proj_gap", 200)),
                float(getattr(args, "qgalore_scale", 0.25)),
            )
            logger.info(
                "Q-GaLore projection: svd_method=%s oversampling=%d niter=%d proj_quant=%s",
                str(getattr(args, "qgalore_svd_method", "full")),
                int(getattr(args, "qgalore_svd_oversampling", 32)),
                int(getattr(args, "qgalore_svd_niter", 1)),
                bool(getattr(args, "qgalore_proj_quant", True)),
            )
    if apollo_group_kwargs is not None:
        logger.info(
            "APOLLO optimizer groups: tensors=%d params=%.3fB groups=%d scales=%s rank=%d gap=%d scale=%g proj=%s scale_type=%s update_rule=%s",
            int(ft_group_stats.get("apollo_param_count", 0)),
            float(ft_group_stats.get("apollo_param_numel", 0)) / 1_000_000_000.0,
            int(ft_group_stats.get("num_apollo_lr_groups", 0)),
            ft_group_stats.get("apollo_lr_scales", []),
            int(getattr(args, "apollo_rank", 256)),
            int(getattr(args, "apollo_update_proj_gap", 200)),
            float(getattr(args, "apollo_scale", 1.0)),
            str(getattr(args, "apollo_proj", "random")),
            str(getattr(args, "apollo_scale_type", "channel")),
            str(getattr(args, "apollo_update_rule", "apollo")),
        )
    if ft_group_stats["frozen_blocks"]:
        logger.info("Full-FT frozen blocks: %s", ft_group_stats["frozen_blocks"])
    if ft_group_stats["block_lr_rules"]:
        logger.info("Full-FT block LR rules: %s", ft_group_stats["block_lr_rules"])
    if bool(getattr(args, "freeze_attn_geometry", False)) or float(getattr(args, "attn_geometry_lr_scale", 1.0)) != 1.0:
        logger.info(
            "Attention geometry protection: freeze=%s lr_scale=%.4f trainable=%d frozen=%d",
            bool(getattr(args, "freeze_attn_geometry", False)),
            float(getattr(args, "attn_geometry_lr_scale", 1.0)),
            int(ft_group_stats.get("trainable_attn_geometry_count", 0)),
            int(ft_group_stats.get("frozen_attn_geometry_count", 0)),
        )
    if text_encoder is not None:
        logger.info(
            "Full-FT text encoder params: trainable_tensors=%d trainable_params=%d lr=%.6g",
            int(ft_group_stats.get("text_encoder_trainable_tensor_count", 0)),
            int(ft_group_stats.get("text_encoder_param_count", 0)),
            float(ft_group_stats.get("text_encoder_lr", 0.0)),
        )

    optimizer_name, optimizer_args, optimizer, optimizer_train_fn, optimizer_eval_fn = trainer.get_optimizer(
        args, params_to_optimize
    )

    # BAdam: wrap the freshly-built base optimizer with the block-coordinate wrapper.
    badam_aliases = {"badam", "blockadam", "block_optimizer", "blockoptimizer"}
    is_badam_run = str(getattr(args, "optimizer_type", "")).lower() in badam_aliases
    if is_badam_run:
        from musubi_tuner.optimizers.badam import create_badam_optimizer

        def parse_badam_wrapper_value(raw_value: str) -> Any:
            normalized = raw_value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
            if normalized in {"none", "null"}:
                return None
            try:
                return ast.literal_eval(raw_value)
            except (ValueError, SyntaxError):
                return raw_value

        # Parse wrapper kwargs from --optimizer_args. base_optimizer_type is consumed by
        # the factory and must be excluded here. Values fall back to raw string when
        # they are not Python literals.
        wrapper_kwargs: dict[str, Any] = {}
        for entry in args.optimizer_args or []:
            if "=" not in entry:
                raise ValueError(f"Invalid --optimizer_args entry (expected key=value): {entry}")
            k, v = entry.split("=", 1)
            wrapper_kwargs[k] = parse_badam_wrapper_value(v)
        wrapper_kwargs.pop("base_optimizer_type", None)
        if wrapper_kwargs.get("use_gradient_release", False) and not wrapper_kwargs.get("use_fp32_active_copy", True):
            raise ValueError(
                "use_gradient_release=True requires use_fp32_active_copy=True; the per-param step path operates on HP fp32 copies."
            )

        # create_badam_optimizer reads its config via getattr(args, 'badam_*', default).
        # Build a synthetic namespace populated from wrapper_kwargs to keep that contract
        # without re-introducing 14 individual CLI flags.
        badam_args_ns = argparse.Namespace()
        for short, default in [
            ("switch_block_every", 100),
            ("switch_mode", "random"),
            ("start_block", None),
            ("block_prefix_mode", "transformer_blocks"),
            ("block_prefixes", []),
            ("block_group_size", 1),
            ("always_active_prefixes", []),
            ("active_modules", []),
            ("include_non_block", True),
            ("include_embedding", False),
            ("include_lm_head", False),
            ("use_fp32_active_copy", True),
            ("purge_inactive_state", True),
            ("reset_state_on_switch", True),
            ("use_gradient_release", False),
            ("bread_sgd", False),
            ("bread_sgd_mode", "all"),
            ("bread_sgd_window_blocks", 0),
            ("bread_sgd_lr_scale", 1.0),
            ("bread_sgd_use_sign", False),
            ("allow_distributed", False),
            ("allow_unmatched_params", False),
            ("verbose", 1),
        ]:
            value = wrapper_kwargs.get(short, default)
            setattr(badam_args_ns, f"badam_{short}", value)
            # Mirror onto args so trainer-side hooks (gradient release, fused-step
            # wiring) can inspect them via the standard args namespace.
            setattr(args, f"badam_{short}", value)

        optimizer = create_badam_optimizer(
            badam_args_ns,
            transformer,
            params_to_optimize,
            optimizer,
            logger=logger,
        )
        optimizer_name = type(optimizer).__module__ + "." + type(optimizer).__name__

    # lr scheduler
    lr_scheduler = trainer.get_lr_scheduler(args, optimizer, accelerator.num_processes)

    # prepare accelerator
    if ltx2_model_parallel:
        transformer = accelerator.prepare(transformer, device_placement=[False])
        if text_encoder is not None:
            text_encoder = accelerator.prepare(text_encoder)
    elif blocks_to_swap > 0:
        transformer = accelerator.prepare(transformer, device_placement=[not blocks_to_swap > 0])
        accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        if text_encoder is not None:
            text_encoder = accelerator.prepare(text_encoder)
    else:
        if text_encoder is not None:
            transformer, text_encoder = accelerator.prepare(transformer, text_encoder)
        else:
            transformer = accelerator.prepare(transformer)

    if args.compile:
        transformer = trainer.compile_transformer(args, transformer)
        transformer.__dict__["_orig_mod"] = transformer

    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    if getattr(trainer, "_self_flow", None) is not None:
        trainer._self_flow_network = accelerator.unwrap_model(transformer)

    # Prepare validation dataloader if exists
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    # Extra per-category validation dataloaders
    extra_val_dataloaders: dict[str, Any] = {}
    _extra_cfgs = getattr(args, "validation_extra_configs", None) or []
    for entry in _extra_cfgs:
        s = str(entry).strip()
        if not s or ":" not in s:
            logger.warning("validation_extra_configs: skipping malformed entry %r (expected category:path)", entry)
            continue
        _cat, _path = s.split(":", 1)
        _cat = _cat.strip()
        _path = _path.strip()
        if not _cat or not _path:
            continue
        try:
            _uc = config_utils.load_user_config(_path)
            _bp = blueprint_generator.generate(_uc, args, architecture=trainer.architecture)
            _dg = config_utils.generate_dataset_group_by_blueprint(
                _bp.dataset_group,
                training=False,
                num_timestep_buckets=args.num_timestep_buckets,
                shared_epoch=current_epoch,
                reference_downscale=getattr(args, "reference_downscale", 1),
            )
            if _dg.num_train_items <= 0:
                logger.warning("validation_extra_configs[%s]: no items, skipping.", _cat)
                continue
            _col = collator_class(current_epoch, _dg if args.max_data_loader_n_workers == 0 else None)
            _dl = torch.utils.data.DataLoader(
                _dg,
                batch_size=1,
                shuffle=False,
                collate_fn=_col,
                num_workers=n_workers,
                persistent_workers=False,
            )
            _dl = accelerator.prepare(_dl)
            extra_val_dataloaders[_cat] = _dl
            logger.info("validation_extra[%s]: %d items from %s", _cat, _dg.num_train_items, _path)
        except Exception as exc:
            logger.warning("validation_extra_configs[%s]: failed to load %s: %s", _cat, _path, exc)

    if getattr(args, "autoresume", False) and not args.resume:
        latest = trainer._find_latest_state_dir(args)
        if latest:
            logger.info("autoresume: found latest state directory: %s", latest)
            args.resume = latest
            args._autoresume_selected = True
        else:
            logger.info("autoresume: no saved state found in output_dir, starting from scratch")
            args._autoresume_selected = False
    else:
        args._autoresume_selected = False

    inner_optimizer = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
    saved_param_groups = None
    if getattr(args, "reset_optimizer_params", False):
        saved_param_groups = [{k: v for k, v in pg.items() if k != "params"} for pg in inner_optimizer.param_groups]

    resume_metadata = train_utils.load_resume_metadata(args.resume) if args.resume else None
    initial_global_step = trainer.resume_from_local_or_hf_if_specified(accelerator, args)

    if initial_global_step > 0:
        if getattr(args, "reset_optimizer", False):
            inner_optimizer.state.clear()
            accelerator.print("reset optimizer state (cleared momentum/variance)")

        if getattr(args, "reset_optimizer_params", False) and saved_param_groups is not None:
            for pg, saved in zip(inner_optimizer.param_groups, saved_param_groups):
                for k, v in saved.items():
                    pg[k] = v
            accelerator.print("reset optimizer param groups to CLI values")

        if getattr(args, "reset_optimizer", False) or getattr(args, "reset_optimizer_params", False):
            for pg in inner_optimizer.param_groups:
                if "initial_lr" in pg:
                    pg["lr"] = pg["initial_lr"]
                    del pg["initial_lr"]
            new_inner_scheduler = trainer.get_lr_scheduler(args, inner_optimizer, accelerator.num_processes)
            if hasattr(lr_scheduler, "scheduler"):
                lr_scheduler.scheduler = new_inner_scheduler
            else:
                lr_scheduler = new_inner_scheduler
            accelerator.print("recreated LR scheduler (restarting schedule from step 0)")

    # Initialize EMA after model is prepared
    ema_model = None
    ema_state_path = os.path.join(args.output_dir, "ema_state.pt") if args.output_dir else None
    if args.use_ema:
        if text_encoder is not None:
            logger.warning(
                "EMA is currently tracking transformer parameters only; text-encoder parameters will not be EMA-averaged "
                "even when --full_ft_train_text_encoder is enabled."
            )
        ema_device = torch.device("cpu") if args.ema_cpu_offload else None
        logger.info(
            "Initializing EMA with decay=%.6f, update_after_step=%d, update_every=%d, device=%s",
            args.ema_decay,
            args.ema_update_after_step,
            args.ema_update_every,
            "cpu" if args.ema_cpu_offload else "same as model",
        )
        if not args.ema_cpu_offload:
            logger.warning(
                "EMA shadow weights will be stored on GPU, increasing VRAM usage. "
                "Use --ema_cpu_offload to store EMA on CPU and save GPU memory."
            )
        ema_model = EMAModel(
            accelerator.unwrap_model(transformer),
            decay=args.ema_decay,
            update_after_step=args.ema_update_after_step,
            update_every=args.ema_update_every,
            device=ema_device,
        )
        # Try to load EMA state if resuming
        if args.resume and ema_state_path and os.path.exists(ema_state_path):
            logger.info("Loading EMA state from: %s", ema_state_path)
            ema_state = torch.load(ema_state_path, map_location="cpu", weights_only=True)
            ema_model.load_state_dict(ema_state)
            logger.info("EMA state loaded (step=%d)", ema_model.step)

    fused_step_state: dict[str, bool] | None = None
    badam_gr_active = False
    if args.fused_backward_pass:
        base_optimizer = getattr(optimizer, "optimizer", optimizer)
        # BAdam-with-gradient_release: hooks attach to LP params via the wrapper itself.
        from musubi_tuner.optimizers.badam import BlockOptimizer

        if isinstance(base_optimizer, BlockOptimizer) and bool(getattr(args, "badam_use_gradient_release", False)):
            hooks_step_enabled = args.max_grad_norm == 0.0
            if not hooks_step_enabled:
                logger.info(
                    "BAdam gradient-release: per-param clipping at max_grad_norm=%.6f.",
                    float(args.max_grad_norm),
                )
            fused_step_state = {
                "defer_step": False,
                "suspend_step": False,
                "hook_stepped": False,
                "hooks_step_enabled": hooks_step_enabled,
            }
            base_optimizer.enable_gradient_release(
                accelerator=accelerator,
                max_grad_norm=float(args.max_grad_norm or 0.0),
                fused_step_state=fused_step_state,
            )
            badam_gr_active = True
            logger.info("BAdam gradient-release enabled (post-prepare).")
        else:
            base_optimizer_name = base_optimizer.__class__.__name__.lower()
            if base_optimizer_name == "adafactor":
                import musubi_tuner.modules.adafactor_fused as adafactor_fused

                adafactor_fused.patch_adafactor_fused(optimizer)
                logger.info("Adafactor fused backward pass enabled.")
            else:
                from musubi_tuner.optimizers.backends import (
                    is_apollo_optimizer_instance,
                    is_optimi_optimizer_instance,
                    is_torchao_optimizer_instance,
                    patch_apollo_fused_step_param,
                    patch_optimi_fused_step_param,
                    patch_torchao_fused_step_param,
                )
                from musubi_tuner.optimizers.q_galore import is_qgalore_optimizer_instance

                if is_qgalore_optimizer_instance(base_optimizer):
                    if args.max_grad_norm != 0.0:
                        raise ValueError("Q-GaLore fused backward requires --max_grad_norm 0")
                    if not _attach_fused_step_param(optimizer, base_optimizer):
                        raise ValueError("Q-GaLore fused backward pass requires optimizer.step_param support")
                    logger.info("%s fused backward pass enabled.", base_optimizer.__class__.__name__)
                elif is_torchao_optimizer_instance(base_optimizer):
                    if not patch_torchao_fused_step_param(base_optimizer) or not _attach_fused_step_param(
                        optimizer, base_optimizer
                    ):
                        raise ValueError(
                            f"{base_optimizer.__class__.__name__} fused backward pass requires torchao single-param Adam support"
                        )
                    if not bool(getattr(base_optimizer, "bf16_stochastic_round", False)):
                        logger.warning(
                            "%s fused backward pass is enabled without bf16_stochastic_round=True. "
                            "For BF16 full fine-tuning, pass --optimizer_args bf16_stochastic_round=True "
                            "or omit it to use the LTX2 BF16 default.",
                            base_optimizer.__class__.__name__,
                        )
                    logger.info("%s torchao fused backward pass enabled.", base_optimizer.__class__.__name__)
                elif is_optimi_optimizer_instance(base_optimizer):
                    if not bool(getattr(base_optimizer, "defaults", {}).get("gradient_release", False)):
                        raise ValueError(
                            f"{base_optimizer.__class__.__name__} fused backward pass requires "
                            "gradient_release=True. Omit that optimizer arg to use the LTX2 default."
                        )
                    if not patch_optimi_fused_step_param(base_optimizer) or not _attach_fused_step_param(optimizer, base_optimizer):
                        raise ValueError(
                            f"{base_optimizer.__class__.__name__} fused backward pass requires optimi single-param step support"
                        )
                    logger.info("%s torch-optimi fused backward pass enabled.", base_optimizer.__class__.__name__)
                elif is_apollo_optimizer_instance(base_optimizer):
                    if not patch_apollo_fused_step_param(base_optimizer) or not _attach_fused_step_param(optimizer, base_optimizer):
                        raise ValueError(
                            f"{base_optimizer.__class__.__name__} fused backward pass requires APOLLO step_param support"
                        )
                    logger.info("%s APOLLO fused backward pass enabled.", base_optimizer.__class__.__name__)
                elif base_optimizer_name in {"came", "came8bit", "sinksgd"}:
                    if not _attach_fused_step_param(optimizer, base_optimizer):
                        raise ValueError(
                            f"{base_optimizer.__class__.__name__} fused backward pass requires optimizer.step_param support"
                        )
                    if base_optimizer_name in {"came", "came8bit"} and not any(
                        bool(group.get("stochastic_rounding", False)) for group in base_optimizer.param_groups
                    ):
                        logger.warning(
                            "%s fused backward pass is enabled without stochastic_rounding=True. "
                            "For BF16 full fine-tuning, pass --optimizer_args stochastic_rounding=True "
                            "or omit it to use the LTX2 BF16 default.",
                            base_optimizer.__class__.__name__,
                        )
                    logger.info("%s fused backward pass enabled.", base_optimizer.__class__.__name__)
                elif base_optimizer_name in {"lion", "lion8bit", "lion8bitint8", "smmf"}:
                    if not _attach_fused_step_param(optimizer, base_optimizer):
                        raise ValueError(
                            f"{base_optimizer.__class__.__name__} fused backward pass requires optimizer.step_param support"
                        )
                    logger.info("%s fused backward pass enabled.", base_optimizer.__class__.__name__)
                else:
                    raise ValueError(
                        f"--fused_backward_pass requires Adafactor, CAME/CAME8bit, SinkSGD, Q-GaLore, APOLLO, "
                        f"torchao Adam, torch-optimi, Lion/Lion8bit/Lion8bitInt8, SMMF, or BAdam with badam_use_gradient_release=True; "
                        f"got {base_optimizer.__class__.__name__}"
                    )

            hooks_step_enabled = args.max_grad_norm == 0.0
            if not hooks_step_enabled:
                logger.info(
                    "Fused hooks are disabled because max_grad_norm=%.6f. "
                    "Using sync-point fused stepping to preserve global gradient clipping correctness.",
                    float(args.max_grad_norm),
                )
            fused_step_state = {
                "defer_step": False,
                "suspend_step": False,
                "hook_stepped": False,
                "hooks_step_enabled": hooks_step_enabled,
            }
            from musubi_tuner.optimizers.q_galore import is_qgalore_parameter

            for param_group, param_name_group in zip(optimizer.param_groups, param_names):
                for parameter, param_name in zip(param_group["params"], param_name_group):
                    is_qgalore_weight = is_qgalore_parameter(parameter)
                    if parameter.requires_grad or is_qgalore_weight:

                        def create_grad_hook(p_name, p_group):
                            def grad_hook(tensor: torch.Tensor):
                                if fused_step_state is None or not fused_step_state.get("hooks_step_enabled", False):
                                    return
                                if not accelerator.sync_gradients:
                                    return
                                if fused_step_state is not None and (
                                    fused_step_state.get("defer_step", False) or fused_step_state.get("suspend_step", False)
                                ):
                                    return
                                optimizer.step_param(tensor, p_group)
                                tensor.grad = None
                                fused_step_state["hook_stepped"] = True

                            return grad_hook

                        if is_qgalore_weight:
                            parameter.backward_hook = create_grad_hook(param_name, param_group)
                        else:
                            parameter.register_post_accumulate_grad_hook(create_grad_hook(param_name, param_group))

    # scheduler
    noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")

    num_train_items = train_dataset_group.num_train_items
    metadata = {
        "ss_session_id": session_id,
        "ss_training_started_at": training_started_at,
        "ss_output_name": args.output_name,
        "ss_no_final_save": bool(getattr(args, "no_final_save", False)),
        "ss_learning_rate": args.learning_rate,
        "ss_num_train_items": num_train_items,
        "ss_num_batches_per_epoch": len(train_dataloader),
        "ss_num_epochs": None,
        "ss_gradient_checkpointing": args.gradient_checkpointing,
        "ss_gradient_checkpointing_cpu_offload": args.gradient_checkpointing_cpu_offload,
        "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
        "ss_accumulation_group_by": getattr(args, "accumulation_group_by", "none"),
        "ss_accumulation_group_remainder": getattr(args, "accumulation_group_remainder", "drop"),
        "ss_max_train_steps": args.max_train_steps,
        "ss_lr_warmup_steps": args.lr_warmup_steps,
        "ss_lr_scheduler": args.lr_scheduler,
        SS_METADATA_KEY_BASE_MODEL_VERSION: trainer.architecture_full_name,
        "ss_mixed_precision": args.mixed_precision,
        "ss_seed": args.seed,
        "ss_training_comment": args.training_comment,
        "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
        "ss_max_grad_norm": args.max_grad_norm,
        "ss_fp8_base": bool(getattr(args, "fp8_base", False)),
        "ss_full_fp16": bool(getattr(args, "full_fp16", False)),
        "ss_full_bf16": bool(getattr(args, "full_bf16", False)),
        "ss_qgalore_full_ft": bool(getattr(args, "qgalore_full_ft", False)),
        "ss_qgalore_targets": getattr(args, "qgalore_targets", None),
        "ss_qgalore_rank": getattr(args, "qgalore_rank", None),
        "ss_qgalore_update_proj_gap": getattr(args, "qgalore_update_proj_gap", None),
        "ss_qgalore_scale": getattr(args, "qgalore_scale", None),
        "ss_qgalore_proj_quant": bool(getattr(args, "qgalore_proj_quant", False)),
        "ss_qgalore_proj_bits": getattr(args, "qgalore_proj_bits", None),
        "ss_qgalore_load_device": getattr(args, "qgalore_load_device", "cuda"),
        "ss_qgalore_svd_method": getattr(args, "qgalore_svd_method", None),
        "ss_qgalore_svd_oversampling": getattr(args, "qgalore_svd_oversampling", None),
        "ss_qgalore_svd_niter": getattr(args, "qgalore_svd_niter", None),
        "ss_qgalore_weight_group_size": getattr(args, "qgalore_weight_group_size", None),
        "ss_qgalore_replaced_modules": getattr(qgalore_summary, "replaced", 0) if qgalore_summary is not None else 0,
        "ss_qgalore_replaced_numel": getattr(qgalore_summary, "replaced_numel", 0) if qgalore_summary is not None else 0,
        "ss_apollo_rank": getattr(args, "apollo_rank", None),
        "ss_apollo_update_proj_gap": getattr(args, "apollo_update_proj_gap", None),
        "ss_apollo_scale": getattr(args, "apollo_scale", None),
        "ss_apollo_proj": getattr(args, "apollo_proj", None),
        "ss_apollo_proj_type": getattr(args, "apollo_proj_type", None),
        "ss_apollo_scale_type": getattr(args, "apollo_scale_type", None),
        "ss_apollo_update_rule": getattr(args, "apollo_update_rule", None),
        "ss_weighting_scheme": args.weighting_scheme,
        "ss_logit_mean": args.logit_mean,
        "ss_logit_std": args.logit_std,
        "ss_mode_scale": args.mode_scale,
        "ss_guidance_scale": args.guidance_scale,
        "ss_timestep_sampling": args.timestep_sampling,
        "ss_sigmoid_scale": args.sigmoid_scale,
        "ss_discrete_flow_shift": args.discrete_flow_shift,
        "ss_ltx_version": getattr(args, "ltx_version", "2.3"),
        "ss_shifted_logit_mode": getattr(args, "shifted_logit_mode", None),
        "ss_shifted_logit_eps": getattr(args, "shifted_logit_eps", 1e-3),
        "ss_shifted_logit_uniform_prob": getattr(args, "shifted_logit_uniform_prob", 0.1),
        "ss_shifted_logit_shift": getattr(args, "shifted_logit_shift", None),
        "ss_ltx_mode": args.ltx_mode,
        "ss_split_av_passes": bool(getattr(args, "split_av_passes", False)),
        "ss_video_loss_weight": getattr(args, "video_loss_weight", 1.0),
        "ss_audio_loss_weight": getattr(args, "audio_loss_weight", 1.0),
        "ss_use_ema": args.use_ema,
        "ss_ema_decay": args.ema_decay if args.use_ema else None,
        "ss_caption_dropout_rate": getattr(args, "caption_dropout_rate", 0.0),
        "ss_motion_preservation": bool(getattr(args, "motion_preservation", False)),
        "ss_motion_preservation_mode": getattr(args, "motion_preservation_mode", "temporal"),
        "ss_motion_preservation_multiplier": getattr(args, "motion_preservation_multiplier", 0.0),
        "ss_motion_preservation_anchor_cache_size": getattr(args, "motion_preservation_anchor_cache_size", 0),
        "ss_motion_preservation_anchor_cache_auto": bool(getattr(args, "motion_preservation_anchor_cache_auto", False)),
        "ss_motion_preservation_anchor_cache_auto_ratio": getattr(args, "motion_preservation_anchor_cache_auto_ratio", 0.2),
        "ss_motion_preservation_anchor_cache_auto_min": getattr(args, "motion_preservation_anchor_cache_auto_min", 8),
        "ss_motion_preservation_anchor_cache_auto_max": getattr(args, "motion_preservation_anchor_cache_auto_max", 64),
        "ss_motion_preservation_anchor_cache_path": getattr(args, "motion_preservation_anchor_cache_path", None),
        "ss_motion_preservation_anchor_cache_rebuild": bool(getattr(args, "motion_preservation_anchor_cache_rebuild", False)),
        "ss_motion_preservation_anchor_source": getattr(args, "motion_preservation_anchor_source", "synthetic"),
        "ss_motion_preservation_synthetic_frames": getattr(args, "motion_preservation_synthetic_frames", 8),
        "ss_motion_preservation_synthetic_temporal_corr": getattr(args, "motion_preservation_synthetic_temporal_corr", 0.92),
        "ss_motion_preservation_synthetic_dataset_mix": getattr(args, "motion_preservation_synthetic_dataset_mix", 0.25),
        "ss_motion_preservation_synthetic_content_seeded": getattr(args, "motion_preservation_synthetic_content_seeded", True),
        "ss_motion_preservation_warmup_steps": getattr(args, "motion_preservation_warmup_steps", 0),
        "ss_motion_preservation_interval": getattr(args, "motion_preservation_interval", 1),
        "ss_motion_preservation_probability": getattr(args, "motion_preservation_probability", None),
        "ss_motion_preservation_num_sigmas": getattr(args, "motion_preservation_num_sigmas", 1),
        "ss_motion_preservation_sigma_values": getattr(args, "motion_preservation_sigma_values", None),
        "ss_motion_preservation_sigma_min": getattr(args, "motion_preservation_sigma_min", 0.2),
        "ss_motion_preservation_sigma_max": getattr(args, "motion_preservation_sigma_max", 0.8),
        "ss_motion_preservation_sigma_sampling": getattr(args, "motion_preservation_sigma_sampling", "uniform"),
        "ss_motion_preservation_sigma_sampling_power": getattr(args, "motion_preservation_sigma_sampling_power", 1.0),
        "ss_motion_preservation_second_order_weight": getattr(args, "motion_preservation_second_order_weight", 0.0),
        "ss_motion_preservation_teacher_chunk_frames": getattr(args, "motion_preservation_teacher_chunk_frames", 0),
        "ss_motion_preservation_separate_backward": bool(getattr(args, "motion_preservation_separate_backward", False)),
        "ss_motion_preservation_fused_defer_step": bool(getattr(args, "motion_preservation_fused_defer_step", False)),
        "ss_motion_attention_preservation": bool(getattr(args, "motion_attention_preservation", False)),
        "ss_motion_attention_preservation_weight": getattr(args, "motion_attention_preservation_weight", 0.0),
        "ss_motion_attention_preservation_loss": getattr(args, "motion_attention_preservation_loss", "kl"),
        "ss_motion_attention_preservation_queries": getattr(args, "motion_attention_preservation_queries", 0),
        "ss_motion_attention_preservation_keys": getattr(args, "motion_attention_preservation_keys", 0),
        "ss_motion_attention_preservation_per_head": bool(getattr(args, "motion_attention_preservation_per_head", False)),
        "ss_motion_attention_preservation_temperature": getattr(args, "motion_attention_preservation_temperature", 1.0),
        "ss_motion_attention_preservation_symmetric_kl": bool(getattr(args, "motion_attention_preservation_symmetric_kl", False)),
        "ss_motion_attention_preservation_blocks": getattr(args, "motion_attention_preservation_blocks", None),
        "ss_ewc_lambda": getattr(args, "ewc_lambda", 0.0),
        "ss_ewc_num_batches": getattr(args, "ewc_num_batches", 0),
        "ss_ewc_target": getattr(args, "ewc_target", "attn_norm_bias"),
        "ss_ewc_max_param_tensors": getattr(args, "ewc_max_param_tensors", 0),
        "ss_ewc_cache_path": getattr(args, "ewc_cache_path", None),
        "ss_ewc_cache_rebuild": bool(getattr(args, "ewc_cache_rebuild", False)),
        "ss_self_flow": bool(getattr(args, "self_flow", False)),
        "ss_self_flow_args": getattr(args, "self_flow_args", None),
        "ss_self_flow_projector_params": self_flow_projector_param_count,
        "ss_av_cross_grad_surgery": bool(getattr(args, "av_cross_grad_surgery", False)),
        "ss_av_cross_grad_surgery_config": (
            args.av_cross_grad_surgery_config.format_summary()
            if getattr(args, "av_cross_grad_surgery_config", None) is not None
            else None
        ),
        "ss_av_cross_grad_surgery_args": (
            " ".join(args.av_cross_grad_surgery_args) if getattr(args, "av_cross_grad_surgery_args", None) else None
        ),
        "ss_av_attention_loss_weighting": bool(getattr(args, "av_attention_loss_weighting", False)),
        "ss_av_attention_loss_max": getattr(args, "av_attention_loss_max", 1.5),
        "ss_av_attention_loss_warmup_steps": getattr(args, "av_attention_loss_warmup_steps", 400),
        "ss_freeze_early_blocks": getattr(args, "freeze_early_blocks", 0),
        "ss_freeze_block_indices": getattr(args, "freeze_block_indices", None),
        "ss_block_lr_scales": getattr(args, "block_lr_scales", None),
        "ss_non_block_lr_scale": getattr(args, "non_block_lr_scale", 1.0),
        "ss_attn_geometry_lr_scale": getattr(args, "attn_geometry_lr_scale", 1.0),
        "ss_freeze_attn_geometry": bool(getattr(args, "freeze_attn_geometry", False)),
        "ss_image_prior_ft": bool(getattr(args, "image_prior_ft", False)),
        "ss_image_prior_ft_strict": bool(getattr(args, "image_prior_ft_strict", True)),
        "ss_image_prior_ft_apply_preset": bool(getattr(args, "image_prior_ft_apply_preset", True)),
        "ss_image_prior_ft_task_route": getattr(args, "image_prior_ft_task_route", "appearance"),
        "ss_image_prior_ft_motion_route": getattr(args, "image_prior_ft_motion_route", "motion"),
        "ss_image_prior_ft_motion_param_target": getattr(args, "image_prior_ft_motion_param_target", "attn_geometry"),
        "ss_image_prior_ft_motion_blocks": getattr(args, "image_prior_ft_motion_blocks", None),
        "ss_image_prior_ft_motion_param_count": ft_group_stats.get("image_prior_ft_motion_param_count", 0),
        "ss_image_prior_ft_appearance_param_count": ft_group_stats.get("image_prior_ft_appearance_param_count", 0),
        "ss_full_ft_train_text_encoder": bool(getattr(args, "full_ft_train_text_encoder", False)),
        "ss_full_ft_text_encoder_lr": getattr(args, "full_ft_text_encoder_lr", None),
        "ss_full_ft_text_encoder_fallback": bool(getattr(args, "full_ft_text_encoder_fallback", False)),
        "ss_full_ft_text_encoder_param_count": ft_group_stats.get("text_encoder_param_count", 0),
        "ss_full_ft_lr_group_scales": ft_group_stats.get("lr_scales"),
        "ss_full_ft_frozen_blocks_applied": ft_group_stats.get("frozen_blocks"),
        "ss_full_ft_text_encoder_lr_group": ft_group_stats.get("text_encoder_lr", None),
    }

    if ltx2_model_parallel:
        metadata.update(
            {
                "ss_ltx2_model_parallel": True,
                "ss_ltx2_model_parallel_devices": (
                    ",".join(str(device_id) for device_id in ltx2_model_parallel_plan.device_ids)
                    if ltx2_model_parallel_plan is not None
                    else None
                ),
                "ss_ltx2_model_parallel_splits": (
                    ",".join(str(split) for split in ltx2_model_parallel_plan.split_points)
                    if ltx2_model_parallel_plan is not None
                    else None
                ),
                "ss_ltx2_mp_activation_codec": getattr(args, "ltx2_mp_activation_codec", "none"),
                "ss_ltx2_mp_grad_codec": getattr(args, "ltx2_mp_grad_codec", "none"),
                "ss_ltx2_mp_int8_block_size": getattr(args, "ltx2_mp_int8_block_size", 256),
            }
        )

    if ltx2_remote_stage:
        metadata.update(
            {
                "ss_ltx2_remote_stage": True,
                "ss_ltx2_remote_stage_host": getattr(args, "ltx2_remote_stage_host", None),
                "ss_ltx2_remote_stage_port": getattr(args, "ltx2_remote_stage_port", None),
                "ss_ltx2_remote_stage_split": getattr(args, "ltx2_remote_stage_split", None),
                "ss_ltx2_remote_stage_specs": getattr(args, "ltx2_remote_stage_specs", None),
                "ss_ltx2_remote_stage_codec": getattr(args, "ltx2_remote_stage_codec", "none"),
                "ss_ltx2_remote_stage_grad_codec": getattr(args, "ltx2_remote_stage_grad_codec", "none"),
                "ss_ltx2_remote_stage_metadata_cache": getattr(args, "ltx2_remote_stage_metadata_cache", None),
                "ss_ltx2_remote_stage_metadata_cache_size": getattr(args, "ltx2_remote_stage_metadata_cache_size", None),
                "ss_ltx2_remote_stage_aq_key_mode": getattr(args, "ltx2_remote_stage_aq_key_mode", None),
                "ss_ltx2_remote_stage_trainable": bool(getattr(args, "ltx2_remote_stage_trainable", False)),
                "ss_ltx2_remote_stage_trainable_scope": getattr(args, "ltx2_remote_stage_trainable_scope", None),
                "ss_ltx2_remote_stage_learning_rate": getattr(args, "ltx2_remote_stage_learning_rate", None),
                "ss_ltx2_remote_stage_weight_decay": getattr(args, "ltx2_remote_stage_weight_decay", None),
                "ss_ltx2_remote_stage_max_grad_norm": getattr(args, "ltx2_remote_stage_max_grad_norm", None),
                "ss_ltx2_remote_stage_checkpoint_dir": getattr(args, "ltx2_remote_stage_checkpoint_dir", None),
            }
        )

    checkpoint_extra_metadata = {k: str(v) for k, v in trainer.get_checkpoint_metadata(args).items()}
    if checkpoint_extra_metadata:
        metadata.update(checkpoint_extra_metadata)

    datasets_metadata = []
    for dataset in train_dataset_group.datasets:
        datasets_metadata.append(dataset.get_metadata())

    metadata["ss_datasets"] = json.dumps(datasets_metadata)

    if args.ltx2_checkpoint is not None:
        logger.info("set LTX-2 model name for metadata: %s", args.ltx2_checkpoint)
        sd_model_name = args.ltx2_checkpoint
        if os.path.exists(sd_model_name):
            sd_model_name = os.path.basename(sd_model_name)
        metadata["ss_sd_model_name"] = sd_model_name

    if args.vae is not None:
        logger.info("set VAE model name for metadata: %s", args.vae)
        vae_name = args.vae
        if os.path.exists(vae_name):
            vae_name = os.path.basename(vae_name)
        metadata["ss_vae_name"] = vae_name

    metadata = {k: str(v) for k, v in metadata.items()}

    minimum_metadata = {}
    for key in SS_METADATA_MINIMUM_KEYS:
        if key in metadata:
            minimum_metadata[key] = metadata[key]

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "fine-tuning" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_utils.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )
        # Log full-FT block-level LR setup once so TensorBoard has explicit
        # traces of configured scales/groups even before the first optimizer step.
        setup_logs: dict[str, float] = {
            "setup/frozen_param_count": float(ft_group_stats.get("frozen_param_count", 0)),
            "setup/trainable_param_count": float(ft_group_stats.get("trainable_param_count", 0)),
            "setup/num_lr_groups": float(ft_group_stats.get("num_lr_groups", 0)),
            "setup/frozen_attn_geometry_count": float(ft_group_stats.get("frozen_attn_geometry_count", 0)),
            "setup/trainable_attn_geometry_count": float(ft_group_stats.get("trainable_attn_geometry_count", 0)),
            "setup/attn_geometry_lr_scale": float(getattr(args, "attn_geometry_lr_scale", 1.0)),
            "setup/image_prior_ft": float(bool(getattr(args, "image_prior_ft", False))),
            "setup/image_prior_ft_motion_param_count": float(ft_group_stats.get("image_prior_ft_motion_param_count", 0)),
            "setup/image_prior_ft_appearance_param_count": float(ft_group_stats.get("image_prior_ft_appearance_param_count", 0)),
            "setup/qgalore_param_count": float(ft_group_stats.get("qgalore_param_count", 0)),
            "setup/qgalore_param_numel": float(ft_group_stats.get("qgalore_param_numel", 0)),
        }
        for i, scale in enumerate(ft_group_stats.get("lr_scales", [])):
            setup_logs[f"setup/lr_scale/group_{i}"] = float(scale)
        for i, group in enumerate(params_to_optimize):
            params = list(group.get("params", []))
            setup_logs[f"setup/group_{i}_param_count"] = float(len(params))
            setup_logs[f"setup/group_{i}_param_numel"] = float(sum(int(p.numel()) for p in params))
        accelerator.log(setup_logs, step=0)

    epoch_to_start = 0
    steps_to_skip_in_epoch = 0
    global_step = initial_global_step
    loss_recorder = train_utils.LossRecorder()
    if initial_global_step > 0 and getattr(trainer, "_resume_state_dir", None):
        _meta = train_utils.load_resume_metadata(trainer._resume_state_dir)
        if _meta and "loss_avg" in _meta:
            loss_recorder.prefill(_meta["loss_avg"], _meta.get("loss_count", 0))
            accelerator.print(f"  restored loss average: {_meta['loss_avg']:.4f} (from {_meta.get('loss_count', 0)} steps)")
    self_flow_loss = None
    if train_dataset_group_for_audio_sampler is None:
        del train_dataset_group

    def save_model(ckpt_name: str, unwrapped_model, steps, epoch_no, force_sync_upload=False, use_memory_efficient_saving=False):
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        if torch.cuda.is_available():
            accelerator.print(f"peak VRAM (torch.cuda.max_memory_allocated): {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        metadata["ss_training_finished_at"] = str(time.time())
        metadata["ss_steps"] = str(steps)
        metadata["ss_epoch"] = str(epoch_no)

        metadata_to_save = minimum_metadata if args.no_metadata else metadata

        title = args.metadata_title if args.metadata_title is not None else args.output_name
        if args.min_timestep is not None or args.max_timestep is not None:
            min_time_step = args.min_timestep if args.min_timestep is not None else 0
            max_time_step = args.max_timestep if args.max_timestep is not None else 1000
            md_timesteps = (min_time_step, max_time_step)
        else:
            md_timesteps = None

        sai_metadata = sai_model_spec.build_metadata(
            None,
            ARCHITECTURE_LTX2,
            time.time(),
            title,
            args.metadata_reso,
            args.metadata_author,
            args.metadata_description,
            args.metadata_license,
            args.metadata_tags,
            timesteps=md_timesteps,
            is_lora=False,
            custom_arch=args.metadata_arch,
        )
        metadata_to_save.update(sai_metadata)

        if checkpoint_extra_metadata:
            metadata_to_save.update(checkpoint_extra_metadata)

        save_model_ref = getattr(unwrapped_model, "_orig_mod", None) or unwrapped_model
        # Build a zero-copy dict referencing live parameters (avoids state_dict() VRAM duplication)
        state_dict = {name: param.data for name, param in save_model_ref.named_parameters()}
        state_dict.update({name: buf for name, buf in save_model_ref.named_buffers()})
        if text_encoder is not None:
            unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
            text_encoder_ref = getattr(unwrapped_text_encoder, "_orig_mod", None) or unwrapped_text_encoder
            for name, param in text_encoder_ref.named_parameters():
                state_dict[f"text_encoder.{name}"] = param.data
            for name, buf in text_encoder_ref.named_buffers():
                state_dict[f"text_encoder.{name}"] = buf
        if bool(getattr(args, "qgalore_full_ft", False)) and bool(getattr(args, "qgalore_dequantize_save", True)):
            from musubi_tuner.optimizers.q_galore import dequantize_qgalore_state_dict

            streaming_qgalore_save = bool(getattr(args, "qgalore_streaming_dequantize_save", False))
            if streaming_qgalore_save:
                accelerator.print(
                    "Using streaming Q-GaLore dequantized save "
                    f"(device={getattr(args, 'qgalore_streaming_dequantize_device', 'cpu')})"
                )
            state_dict = dequantize_qgalore_state_dict(
                save_model_ref,
                state_dict,
                lazy=streaming_qgalore_save,
                device=getattr(args, "qgalore_streaming_dequantize_device", "cpu") if streaming_qgalore_save else None,
            )
        if bool(getattr(args, "int8_weights", False)):
            # dequantize Int8QTWeight -> bf16 so the checkpoint is a standard, directly loadable file
            from musubi_tuner.modules.int8_training import Int8QTWeight

            state_dict = {k: (v.dequantize() if isinstance(v, Int8QTWeight) else v) for k, v in state_dict.items()}
        state_dict, extra_meta = _prepare_state_dict_for_save(state_dict, args)
        if extra_meta:
            metadata_to_save.update(extra_meta)
        mem_eff_save_file(state_dict, ckpt_file, metadata_to_save)

        if args.huggingface_repo_id is not None:
            huggingface_utils.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        if getattr(args, "save_checkpoint_metadata", False):
            from datetime import datetime

            _md = {
                "step": steps,
                "epoch": epoch_no,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            try:
                _md["loss"] = current_loss
            except Exception:
                pass
            if loss_recorder.loss_list:
                _md["loss_avg"] = loss_recorder.moving_average
            try:
                _md["lr"] = float(lr_scheduler.get_last_lr()[0])
            except Exception:
                pass
            try:
                if video_loss is not None:
                    _md["loss_video"] = video_loss.detach().item()
            except Exception:
                pass
            try:
                if audio_loss is not None:
                    _md["loss_audio"] = audio_loss.detach().item()
            except Exception:
                pass
            if motion_pres_loss is not None:
                _md["loss_motion_pres"] = motion_pres_loss.detach().item()
            if attn_pres_loss is not None:
                _md["loss_attn_pres"] = attn_pres_loss.detach().item()
            if ewc_loss is not None:
                _md["loss_ewc"] = ewc_loss.detach().item()
            if self_flow_loss is not None:
                _md["loss_self_flow"] = self_flow_loss.detach().item()
            train_utils.save_checkpoint_metadata(ckpt_file, _md)

        if ltx2_remote_stage and bool(getattr(args, "ltx2_remote_stage_trainable", False)):
            remote_checkpoint_dir = getattr(args, "ltx2_remote_stage_checkpoint_dir", None) or args.output_dir
            try:
                responses = save_ltx2_remote_stage_state(
                    transformer,
                    checkpoint_dir=remote_checkpoint_dir,
                    checkpoint_name=ckpt_name,
                )
                for response in responses:
                    path = response.get("path")
                    if path:
                        accelerator.print(f"saved remote stage checkpoint: {path}")
            except Exception as exc:
                logger.warning("Failed to save remote LTX-2 stage checkpoint: %s", exc)

    def remove_model(old_ckpt_name: str) -> None:
        old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
        if os.path.exists(old_ckpt_file):
            accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
            os.remove(old_ckpt_file)
        train_utils.remove_checkpoint_metadata(old_ckpt_file)

    def save_ema_model(ckpt_name: str, steps: int, epoch_no: int) -> None:
        """Save EMA weights as a separate checkpoint."""
        if ema_model is None:
            return
        os.makedirs(args.output_dir, exist_ok=True)
        ema_ckpt_file = os.path.join(args.output_dir, ckpt_name.replace(".safetensors", "_ema.safetensors"))

        accelerator.print(f"\nsaving EMA checkpoint: {ema_ckpt_file}")

        # Create state dict from EMA shadow params
        unwrapped = accelerator.unwrap_model(transformer)
        save_model_ref = getattr(unwrapped, "_orig_mod", None) or unwrapped
        ema_state_dict = {}
        for name, param in save_model_ref.named_parameters():
            if name in ema_model.shadow_params:
                ema_state_dict[name] = ema_model.shadow_params[name].cpu()
            else:
                ema_state_dict[name] = param.data.cpu()

        # Add non-parameter state (buffers)
        for name, buf in save_model_ref.named_buffers():
            ema_state_dict[name] = buf.cpu()

        ema_state_dict, extra_meta = _prepare_state_dict_for_save(ema_state_dict, args)

        ema_metadata = metadata.copy()
        ema_metadata["ss_is_ema"] = "True"
        ema_metadata["ss_ema_decay"] = str(args.ema_decay)
        ema_metadata["ss_steps"] = str(steps)
        ema_metadata["ss_epoch"] = str(epoch_no)
        if extra_meta:
            ema_metadata.update(extra_meta)

        if args.mem_eff_save:
            mem_eff_save_file(ema_state_dict, ema_ckpt_file, ema_metadata)
        else:
            save_file(ema_state_dict, ema_ckpt_file, ema_metadata)

    def save_ema_state() -> None:
        """Save EMA state for resume functionality."""
        if ema_model is None or ema_state_path is None:
            return
        if not accelerator.is_main_process:
            return
        os.makedirs(os.path.dirname(ema_state_path), exist_ok=True)
        torch.save(ema_model.state_dict(), ema_state_path)
        logger.info("EMA state saved to: %s (step=%d)", ema_state_path, ema_model.step)

    def save_self_flow_state() -> None:
        """Save Self-Flow projector + teacher EMA state for resume."""
        if not bool(getattr(args, "self_flow", False)):
            return
        module = getattr(trainer, "_self_flow", None)
        if module is None or not accelerator.is_main_process:
            return
        os.makedirs(args.output_dir, exist_ok=True)
        projector_file = os.path.join(args.output_dir, "self_flow_projector.safetensors")
        teacher_file = os.path.join(args.output_dir, "self_flow_teacher_ema.safetensors")

        try:
            proj_sd = module.state_dict()
            if proj_sd:
                proj_sd_cpu = {k: v.detach().to(device="cpu") for k, v in proj_sd.items() if isinstance(v, torch.Tensor)}
                save_file(proj_sd_cpu, projector_file)
        except Exception as e:
            logger.warning("Failed to save Self-Flow projector state: %s", e)

        try:
            teacher_sd = module.teacher_state_dict()
            if teacher_sd:
                save_file(teacher_sd, teacher_file)
        except Exception as e:
            logger.warning("Failed to save Self-Flow teacher EMA state: %s", e)

    def handle_dashboard_stop_request(global_step: int, epoch: int, step_in_epoch: int) -> bool:
        if not train_utils.dashboard_stop_requested():
            return False

        if train_utils.dashboard_stop_mode() == "force":
            accelerator.print("\nDashboard force stop requested; exiting without saving interrupt state.")
            train_utils.clear_dashboard_stop_request()
            accelerator.end_training()
            return True

        if global_step <= 0:
            accelerator.print("\nDashboard stop requested before training steps completed; exiting without saving state.")
            train_utils.clear_dashboard_stop_request()
            accelerator.end_training()
            return True

        accelerator.print("\nDashboard stop requested; saving interrupt state and exiting training.")
        optimizer_eval_fn()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dir = train_utils.save_state_on_interrupt(
                args,
                accelerator,
                global_step=global_step,
                epoch=epoch + 1,
                step_in_epoch=step_in_epoch,
            )
            train_utils.update_resume_metadata(
                state_dir,
                {
                    "loss_avg": loss_recorder.moving_average,
                    "loss_count": len(loss_recorder.loss_list),
                    "interrupted": True,
                },
            )
            save_ema_state()
            save_self_flow_state()
        train_utils.clear_dashboard_stop_request()
        accelerator.end_training()
        return True

    def run_validation(step: int, epoch: int) -> dict:
        """Run validation and return metrics."""
        if val_dataloader is None:
            return {}

        accelerator.print(f"\nRunning validation at step {step}...")
        transformer.eval()

        # Apply EMA weights for validation if available
        original_params = None
        if ema_model is not None:
            original_params = ema_model.apply_to(accelerator.unwrap_model(transformer))

        self_flow_network = getattr(trainer, "_self_flow_network", None)
        self_flow_network_was_training = (
            bool(getattr(self_flow_network, "training", False)) if self_flow_network is not None else None
        )
        force_self_flow_capture = bool(
            getattr(args, "self_flow", False)
            and getattr(trainer, "_self_flow", None) is not None
            and self_flow_network is not None
            and not bool(getattr(self_flow_network, "training", False))
        )
        if force_self_flow_capture:
            # Self-Flow capture is gated on the top-level module's `.training` flag.
            # Flip only that flag so validation can populate cached teacher/student
            # features without globally switching submodules back to train mode.
            self_flow_network.training = True

        val_losses = []
        val_video_losses = []
        val_audio_losses = []
        val_self_flow_losses = []
        val_ewc_losses = []
        val_motion_pres_losses = []
        val_motion_attn_pres_losses = []
        val_motion_total_losses = []
        num_batches = 0
        max_batches = args.num_validation_batches
        validation_self_flow_enabled = bool(getattr(args, "self_flow", False))

        with torch.no_grad():
            for batch in val_dataloader:
                if max_batches is not None and num_batches >= max_batches:
                    break
                batch = _normalize_ltx2_batch_for_call_dit(batch)

                latents = batch["latents"]
                if isinstance(latents, dict):
                    latents_tensor = latents["latents"]
                else:
                    latents_tensor = latents

                latents_tensor = trainer.scale_shift_latents(latents_tensor)
                noise = torch.randn_like(latents_tensor)

                if validation_self_flow_enabled:
                    args.self_flow = False
                    if getattr(trainer, "_self_flow", None) is not None:
                        trainer._self_flow.cleanup_step()
                    trainer._self_flow_step_context = None
                try:
                    noisy_model_input, timesteps = trainer.get_noisy_model_input_and_timesteps(
                        args,
                        noise,
                        latents_tensor,
                        batch["timesteps"],
                        noise_scheduler,
                        accelerator.device,
                        trainer.dit_dtype,
                    )
                finally:
                    if validation_self_flow_enabled:
                        args.self_flow = True

                weighting = compute_loss_weighting_for_sd3(
                    args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, trainer.dit_dtype
                )

                if validation_self_flow_enabled:
                    args.self_flow = False
                    if getattr(trainer, "_self_flow", None) is not None:
                        trainer._self_flow.cleanup_step()
                    trainer._self_flow_step_context = None
                try:
                    model_pred, target = trainer.call_dit(
                        args,
                        accelerator,
                        transformer,
                        latents_tensor,
                        batch,
                        noise,
                        noisy_model_input,
                        timesteps,
                        trainer.dit_dtype,
                    )
                finally:
                    if validation_self_flow_enabled:
                        args.self_flow = True

                dict_output = isinstance(model_pred, dict)
                if dict_output:
                    out = model_pred
                    if out.get("_skip_step"):
                        continue

                    video_pred = out["video_pred"]
                    video_target = out["video_target"]
                    video_loss_mask = out.get("video_loss_mask")
                    _val_loss_type = getattr(args, "loss_type", "mse")
                    _val_huber_delta = float(getattr(args, "huber_delta", 1.0))
                    video_loss = _masked_mse(
                        video_pred,
                        video_target,
                        video_loss_mask,
                        weighting=weighting,
                        dtype=trainer.dit_dtype,
                        loss_type=_val_loss_type,
                        huber_delta=_val_huber_delta,
                    )
                    val_video_losses.append(video_loss.item())

                    audio_pred = out.get("audio_pred")
                    audio_target = out.get("audio_target")
                    if audio_pred is not None and audio_target is not None:
                        audio_loss_mask = out.get("audio_loss_mask")
                        audio_loss = _masked_mse(
                            audio_pred,
                            audio_target,
                            audio_loss_mask,
                            weighting=weighting,
                            dtype=trainer.dit_dtype,
                            loss_type=_val_loss_type,
                            huber_delta=_val_huber_delta,
                        )
                        val_audio_losses.append(audio_loss.item())
                        video_weight = float(out.get("video_loss_weight", 1.0))
                        audio_weight = float(out.get("audio_loss_weight", 1.0))
                        loss = video_loss * video_weight + audio_loss * audio_weight
                    else:
                        video_weight = float(out.get("video_loss_weight", 1.0))
                        loss = video_loss * video_weight
                else:
                    _val_loss_type = getattr(args, "loss_type", "mse")
                    _val_huber_delta = float(getattr(args, "huber_delta", 1.0))
                    if isinstance(target, torch.Tensor):
                        model_pred = model_pred.to(device=target.device, dtype=trainer.dit_dtype)
                    else:
                        model_pred = model_pred.to(dtype=trainer.dit_dtype)
                    if _val_loss_type in ("mae", "l1"):
                        loss = torch.nn.functional.l1_loss(model_pred, target, reduction="none")
                    elif _val_loss_type in ("huber", "smooth_l1"):
                        loss = torch.nn.functional.smooth_l1_loss(model_pred, target, reduction="none", beta=_val_huber_delta)
                    else:
                        loss = torch.nn.functional.mse_loss(model_pred, target, reduction="none")
                    if weighting is not None:
                        w = weighting
                        if isinstance(w, torch.Tensor) and w.dim() != loss.dim():
                            while w.dim() > loss.dim() and w.shape[-1] == 1:
                                w = w.squeeze(-1)
                        loss = loss * w
                    loss = loss.mean()

                self_flow_loss = None
                if validation_self_flow_enabled:
                    try:
                        sf_noisy_model_input, sf_timesteps = trainer.get_noisy_model_input_and_timesteps(
                            args,
                            noise,
                            latents_tensor,
                            batch["timesteps"],
                            noise_scheduler,
                            accelerator.device,
                            trainer.dit_dtype,
                        )
                        _sf_model_pred, _sf_target = trainer.call_dit(
                            args,
                            accelerator,
                            transformer,
                            latents_tensor,
                            batch,
                            noise,
                            sf_noisy_model_input,
                            sf_timesteps,
                            trainer.dit_dtype,
                        )
                        if getattr(trainer, "_self_flow", None) is not None:
                            trainer._self_flow.on_step(step)
                        self_flow_loss, _self_flow_metrics = trainer.compute_self_flow_addition(
                            args,
                            accelerator,
                            transformer,
                            transformer,
                            trainer.dit_dtype,
                        )
                        if self_flow_loss is not None:
                            val_self_flow_losses.append(float(self_flow_loss.detach().item()))
                    except Exception as e:
                        if getattr(trainer, "_self_flow", None) is not None:
                            trainer._self_flow.cleanup_step()
                        logger.warning(
                            "Self-Flow validation loss computation failed at step=%s batch=%s: %s",
                            step,
                            num_batches,
                            e,
                        )

                ewc_loss = None
                if ewc_state is not None and float(getattr(args, "ewc_lambda", 0.0) or 0.0) > 0.0:
                    ewc_penalty_raw, _ewc_used_tensors, _ewc_skipped_tensors = _compute_ewc_penalty(
                        ewc_state,
                        dtype=trainer.dit_dtype,
                        target_device=loss.device if isinstance(loss, torch.Tensor) else None,
                    )
                    if ewc_penalty_raw is not None:
                        ewc_loss = ewc_penalty_raw * float(getattr(args, "ewc_lambda", 0.0))
                        loss = loss + ewc_loss
                        val_ewc_losses.append(float(ewc_loss.detach().item()))

                motion_pres_loss = None
                attn_pres_loss = None
                motion_total_loss = None
                motion_preservation_prob = getattr(args, "motion_preservation_probability", None)
                if motion_preservation_prob is None:
                    should_apply_motion_replay = (step + num_batches + 1) % int(args.motion_preservation_interval) == 0
                else:
                    should_apply_motion_replay = random.random() < float(motion_preservation_prob)

                if args.motion_preservation and motion_anchor_cache and should_apply_motion_replay:
                    anchor = random.choice(motion_anchor_cache)
                    anchor_latents = _move_to_device(anchor["anchor_latents"], accelerator.device, dtype=trainer.dit_dtype)
                    anchor_noise = _move_to_device(anchor["anchor_noise"], accelerator.device, dtype=trainer.dit_dtype)

                    sigma_idx: Optional[int] = None
                    anchor_sigmas = anchor.get("anchor_sigmas") or []
                    teacher_video_preds = anchor.get("teacher_video_preds")
                    if (
                        isinstance(anchor_sigmas, list)
                        and isinstance(teacher_video_preds, list)
                        and len(anchor_sigmas) > 0
                        and len(anchor_sigmas) == len(teacher_video_preds)
                    ):
                        sigma_idx = _sample_motion_sigma_index(anchor_sigmas, args)
                        sigma_value = float(anchor_sigmas[sigma_idx])
                        anchor_noisy_input, anchor_model_timesteps = _build_noisy_input_for_sigma(
                            anchor_latents,
                            anchor_noise,
                            sigma_value,
                        )
                        teacher_video_pred = teacher_video_preds[sigma_idx]
                    else:
                        anchor_noisy_input = _move_to_device(
                            anchor["anchor_noisy_input"], accelerator.device, dtype=trainer.dit_dtype
                        )
                        anchor_model_timesteps = _move_to_device(anchor["anchor_model_timesteps"], accelerator.device)
                        teacher_video_pred = anchor.get("teacher_video_pred")

                    anchor_batch = _move_to_device(anchor["anchor_batch"], accelerator.device, dtype=trainer.dit_dtype)
                    if isinstance(teacher_video_pred, torch.Tensor):
                        original_first_frame_p = float(getattr(args, "ltx2_first_frame_conditioning_p", 0.0))
                        try:
                            setattr(args, "ltx2_first_frame_conditioning_p", 0.0)
                            if args.motion_attention_preservation and motion_attention_modules:
                                attn_recorder = _AttentionMapRecorder(
                                    motion_attention_modules,
                                    max_queries=int(getattr(args, "motion_attention_preservation_queries", 32) or 32),
                                    max_keys=int(getattr(args, "motion_attention_preservation_keys", 64) or 64),
                                    capture_grad=False,
                                    keep_heads=bool(getattr(args, "motion_attention_preservation_per_head", False)),
                                )
                                attn_recorder.__enter__()
                                try:
                                    motion_pred, _ = trainer.call_dit(
                                        args,
                                        accelerator,
                                        transformer,
                                        anchor_latents,
                                        anchor_batch,
                                        anchor_noise,
                                        anchor_noisy_input,
                                        anchor_model_timesteps,
                                        trainer.dit_dtype,
                                    )
                                    student_attn_maps = attn_recorder.collect_maps()
                                finally:
                                    attn_recorder.__exit__(None, None, None)
                            else:
                                student_attn_maps = {}
                                motion_pred, _ = trainer.call_dit(
                                    args,
                                    accelerator,
                                    transformer,
                                    anchor_latents,
                                    anchor_batch,
                                    anchor_noise,
                                    anchor_noisy_input,
                                    anchor_model_timesteps,
                                    trainer.dit_dtype,
                                )
                        finally:
                            setattr(args, "ltx2_first_frame_conditioning_p", original_first_frame_p)

                        if isinstance(motion_pred, dict) and not motion_pred.get("_skip_step"):
                            student_video_pred = motion_pred["video_pred"]
                            motion_pres_loss_raw = _compute_motion_preservation_loss(
                                args,
                                student_video_pred,
                                teacher_video_pred,
                                motion_pred.get("video_loss_mask"),
                                dtype=trainer.dit_dtype,
                            )
                            motion_multiplier = float(args.motion_preservation_multiplier)
                            motion_warmup = int(getattr(args, "motion_preservation_warmup_steps", 0) or 0)
                            if motion_warmup > 0 and step < motion_warmup:
                                motion_multiplier = motion_multiplier * (step / motion_warmup)
                            motion_pres_loss = motion_pres_loss_raw * motion_multiplier
                            motion_total_loss = motion_pres_loss

                            teacher_attn_maps = anchor.get("teacher_attention_maps")
                            teacher_attn_maps_multi = anchor.get("teacher_attention_maps_multi")
                            if isinstance(teacher_attn_maps_multi, list) and teacher_attn_maps_multi:
                                if sigma_idx is not None and 0 <= sigma_idx < len(teacher_attn_maps_multi):
                                    mapped = teacher_attn_maps_multi[sigma_idx]
                                    if isinstance(mapped, dict):
                                        teacher_attn_maps = mapped
                                elif sigma_idx is None and len(teacher_attn_maps_multi) == 1:
                                    mapped = teacher_attn_maps_multi[0]
                                    if isinstance(mapped, dict):
                                        teacher_attn_maps = mapped

                            if args.motion_attention_preservation and isinstance(teacher_attn_maps, dict) and student_attn_maps:
                                attn_temperature = float(getattr(args, "motion_attention_preservation_temperature", 1.0) or 1.0)
                                symmetric_kl = bool(getattr(args, "motion_attention_preservation_symmetric_kl", False))
                                per_block_losses: list[torch.Tensor] = []
                                for module_name, student_map in student_attn_maps.items():
                                    teacher_map = teacher_attn_maps.get(module_name)
                                    if not isinstance(teacher_map, torch.Tensor):
                                        continue
                                    if teacher_map.shape != student_map.shape:
                                        logger.warning(
                                            "Motion preservation validation: attention map shape mismatch for module %s (student=%s teacher=%s), skipping",
                                            module_name,
                                            student_map.shape,
                                            teacher_map.shape,
                                        )
                                        continue

                                    student_dist = student_map.to(torch.float32)
                                    teacher_dist = teacher_map.to(
                                        device=student_dist.device,
                                        dtype=torch.float32,
                                        non_blocking=True,
                                    )
                                    if attn_temperature != 1.0:
                                        inv_temp = 1.0 / max(1e-6, attn_temperature)
                                        student_dist = student_dist.clamp_min(1e-8).pow(inv_temp)
                                        teacher_dist = teacher_dist.clamp_min(1e-8).pow(inv_temp)
                                    student_dist = student_dist.clamp_min(1e-6)
                                    teacher_dist = teacher_dist.clamp_min(1e-6)
                                    student_dist = student_dist / student_dist.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                                    teacher_dist = teacher_dist / teacher_dist.sum(dim=-1, keepdim=True).clamp_min(1e-6)

                                    if getattr(args, "motion_attention_preservation_loss", "kl") == "kl":
                                        kl_t2s = torch.nn.functional.kl_div(
                                            student_dist.log(),
                                            teacher_dist,
                                            reduction="batchmean",
                                        )
                                        if symmetric_kl:
                                            kl_s2t = torch.nn.functional.kl_div(
                                                teacher_dist.log(),
                                                student_dist,
                                                reduction="batchmean",
                                            )
                                            block_loss = 0.5 * (kl_t2s + kl_s2t)
                                        else:
                                            block_loss = kl_t2s
                                    else:
                                        block_loss = torch.nn.functional.mse_loss(student_dist, teacher_dist)
                                    per_block_losses.append(block_loss)

                                if per_block_losses:
                                    attn_pres_loss_raw = torch.stack(per_block_losses).mean()
                                    attn_pres_loss = attn_pres_loss_raw * float(
                                        getattr(args, "motion_attention_preservation_weight", 0.0)
                                    )
                                    motion_total_loss = motion_total_loss + attn_pres_loss

                if motion_pres_loss is not None:
                    val_motion_pres_losses.append(float(motion_pres_loss.detach().item()))
                if attn_pres_loss is not None:
                    val_motion_attn_pres_losses.append(float(attn_pres_loss.detach().item()))
                if motion_total_loss is not None:
                    loss = loss + motion_total_loss
                    val_motion_total_losses.append(float(motion_total_loss.detach().item()))

                val_losses.append(float(loss.detach().item()))

                num_batches += 1

        # Restore original weights if EMA was applied
        if original_params is not None:
            ema_model.restore(accelerator.unwrap_model(transformer), original_params)
        if force_self_flow_capture and self_flow_network is not None:
            self_flow_network.training = bool(self_flow_network_was_training)

        transformer.train()

        # Compute average metrics
        val_metrics = {}
        if val_losses:
            val_metrics["val_loss"] = sum(val_losses) / len(val_losses)
        if val_video_losses:
            val_metrics["val_video_loss"] = sum(val_video_losses) / len(val_video_losses)
        if val_audio_losses:
            val_metrics["val_audio_loss"] = sum(val_audio_losses) / len(val_audio_losses)
        if val_self_flow_losses:
            val_metrics["val_self_flow_loss"] = sum(val_self_flow_losses) / len(val_self_flow_losses)
        if val_ewc_losses:
            val_metrics["val_ewc_loss"] = sum(val_ewc_losses) / len(val_ewc_losses)
        if val_motion_pres_losses:
            val_metrics["val_motion_pres_loss"] = sum(val_motion_pres_losses) / len(val_motion_pres_losses)
        if val_motion_attn_pres_losses:
            val_metrics["val_motion_attn_pres_loss"] = sum(val_motion_attn_pres_losses) / len(val_motion_attn_pres_losses)
        if val_motion_total_losses:
            val_metrics["val_motion_total_loss"] = sum(val_motion_total_losses) / len(val_motion_total_losses)

        # Multi-timestep validation passes: run additional dataloader sweeps with fixed t.
        mt_raw = getattr(args, "validation_timesteps", None)
        mt_list: list[float] = []
        if mt_raw:
            for tok in str(mt_raw).split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    mt_list.append(float(tok))
                except ValueError:
                    logger.warning("validation_timesteps: skipping non-numeric token %r", tok)

        if mt_list:
            transformer.eval()
            if ema_model is not None and original_params is None:
                original_params = ema_model.apply_to(accelerator.unwrap_model(transformer))
            if validation_self_flow_enabled:
                args.self_flow = False
                if getattr(trainer, "_self_flow", None) is not None:
                    trainer._self_flow.cleanup_step()
                trainer._self_flow_step_context = None
            with torch.no_grad():
                for fixed_t in mt_list:
                    mt_losses: list[float] = []
                    mt_video_losses: list[float] = []
                    mt_audio_losses: list[float] = []
                    mt_batches = 0
                    for batch in val_dataloader:
                        if max_batches is not None and mt_batches >= max_batches:
                            break
                        batch = _normalize_ltx2_batch_for_call_dit(batch)
                        latents = batch["latents"]
                        latents_tensor = latents["latents"] if isinstance(latents, dict) else latents
                        latents_tensor = trainer.scale_shift_latents(latents_tensor)
                        noise = torch.randn_like(latents_tensor)
                        bsz = int(noise.shape[0])
                        override_t = [float(fixed_t)] * bsz
                        try:
                            noisy_model_input, timesteps = trainer.get_noisy_model_input_and_timesteps(
                                args,
                                noise,
                                latents_tensor,
                                override_t,
                                noise_scheduler,
                                accelerator.device,
                                trainer.dit_dtype,
                            )
                        except Exception as exc:
                            logger.warning("val/t%s: get_noisy failed: %s", fixed_t, exc)
                            break
                        weighting = compute_loss_weighting_for_sd3(
                            args.weighting_scheme,
                            noise_scheduler,
                            timesteps,
                            accelerator.device,
                            trainer.dit_dtype,
                        )
                        model_pred, target = trainer.call_dit(
                            args,
                            accelerator,
                            transformer,
                            latents_tensor,
                            batch,
                            noise,
                            noisy_model_input,
                            timesteps,
                            trainer.dit_dtype,
                        )
                        _val_lt = getattr(args, "loss_type", "mse")
                        _val_hd = float(getattr(args, "huber_delta", 1.0))
                        if isinstance(model_pred, dict):
                            if model_pred.get("_skip_step"):
                                continue
                            v_pred = model_pred["video_pred"]
                            v_tgt = model_pred["video_target"]
                            v_mask = model_pred.get("video_loss_mask")
                            v_loss = _masked_mse(
                                v_pred,
                                v_tgt,
                                v_mask,
                                weighting=weighting,
                                dtype=trainer.dit_dtype,
                                loss_type=_val_lt,
                                huber_delta=_val_hd,
                            )
                            mt_video_losses.append(v_loss.item())
                            a_pred = model_pred.get("audio_pred")
                            a_tgt = model_pred.get("audio_target")
                            if a_pred is not None and a_tgt is not None:
                                a_mask = model_pred.get("audio_loss_mask")
                                a_loss = _masked_mse(
                                    a_pred,
                                    a_tgt,
                                    a_mask,
                                    weighting=weighting,
                                    dtype=trainer.dit_dtype,
                                    loss_type=_val_lt,
                                    huber_delta=_val_hd,
                                )
                                mt_audio_losses.append(a_loss.item())
                                v_w = float(model_pred.get("video_loss_weight", 1.0))
                                a_w = float(model_pred.get("audio_loss_weight", 1.0))
                                loss_mt = v_loss * v_w + a_loss * a_w
                            else:
                                v_w = float(model_pred.get("video_loss_weight", 1.0))
                                loss_mt = v_loss * v_w
                        else:
                            if isinstance(target, torch.Tensor):
                                model_pred = model_pred.to(device=target.device, dtype=trainer.dit_dtype)
                            else:
                                model_pred = model_pred.to(dtype=trainer.dit_dtype)
                            if _val_lt in ("mae", "l1"):
                                per_el = torch.nn.functional.l1_loss(model_pred, target, reduction="none")
                            elif _val_lt in ("huber", "smooth_l1"):
                                per_el = torch.nn.functional.smooth_l1_loss(
                                    model_pred,
                                    target,
                                    reduction="none",
                                    beta=_val_hd,
                                )
                            else:
                                per_el = torch.nn.functional.mse_loss(model_pred, target, reduction="none")
                            if weighting is not None:
                                w = weighting
                                if isinstance(w, torch.Tensor) and w.dim() != per_el.dim():
                                    while w.dim() > per_el.dim() and w.shape[-1] == 1:
                                        w = w.squeeze(-1)
                                per_el = per_el * w
                            loss_mt = per_el.mean()
                        mt_losses.append(loss_mt.item())
                        mt_batches += 1
                    t_key = f"t{int(round(fixed_t))}"
                    if mt_losses:
                        val_metrics[f"val/{t_key}/loss"] = sum(mt_losses) / len(mt_losses)
                    if mt_video_losses:
                        val_metrics[f"val/{t_key}/video_loss"] = sum(mt_video_losses) / len(mt_video_losses)
                    if mt_audio_losses:
                        val_metrics[f"val/{t_key}/audio_loss"] = sum(mt_audio_losses) / len(mt_audio_losses)
            if original_params is not None:
                ema_model.restore(accelerator.unwrap_model(transformer), original_params)
                original_params = None
            if validation_self_flow_enabled:
                args.self_flow = True
            transformer.train()

        # Per-category OOD validation: basic loss only, no EWC/motion/self_flow.
        if extra_val_dataloaders:
            transformer.eval()
            if ema_model is not None and original_params is None:
                original_params = ema_model.apply_to(accelerator.unwrap_model(transformer))
            if validation_self_flow_enabled:
                args.self_flow = False
                if getattr(trainer, "_self_flow", None) is not None:
                    trainer._self_flow.cleanup_step()
                trainer._self_flow_step_context = None
            with torch.no_grad():
                for cat_name, cat_loader in extra_val_dataloaders.items():
                    c_losses: list[float] = []
                    c_video: list[float] = []
                    c_audio: list[float] = []
                    c_batches = 0
                    for batch in cat_loader:
                        if max_batches is not None and c_batches >= max_batches:
                            break
                        batch = _normalize_ltx2_batch_for_call_dit(batch)
                        latents = batch["latents"]
                        latents_tensor = latents["latents"] if isinstance(latents, dict) else latents
                        latents_tensor = trainer.scale_shift_latents(latents_tensor)
                        noise = torch.randn_like(latents_tensor)
                        try:
                            noisy_model_input, timesteps = trainer.get_noisy_model_input_and_timesteps(
                                args,
                                noise,
                                latents_tensor,
                                batch.get("timesteps"),
                                noise_scheduler,
                                accelerator.device,
                                trainer.dit_dtype,
                            )
                        except Exception as exc:
                            logger.warning("val/%s: get_noisy failed: %s", cat_name, exc)
                            break
                        weighting = compute_loss_weighting_for_sd3(
                            args.weighting_scheme,
                            noise_scheduler,
                            timesteps,
                            accelerator.device,
                            trainer.dit_dtype,
                        )
                        model_pred, target = trainer.call_dit(
                            args,
                            accelerator,
                            transformer,
                            latents_tensor,
                            batch,
                            noise,
                            noisy_model_input,
                            timesteps,
                            trainer.dit_dtype,
                        )
                        _val_lt = getattr(args, "loss_type", "mse")
                        _val_hd = float(getattr(args, "huber_delta", 1.0))
                        if isinstance(model_pred, dict):
                            if model_pred.get("_skip_step"):
                                continue
                            v_pred = model_pred["video_pred"]
                            v_tgt = model_pred["video_target"]
                            v_mask = model_pred.get("video_loss_mask")
                            v_loss = _masked_mse(
                                v_pred,
                                v_tgt,
                                v_mask,
                                weighting=weighting,
                                dtype=trainer.dit_dtype,
                                loss_type=_val_lt,
                                huber_delta=_val_hd,
                            )
                            c_video.append(v_loss.item())
                            a_pred = model_pred.get("audio_pred")
                            a_tgt = model_pred.get("audio_target")
                            if a_pred is not None and a_tgt is not None:
                                a_mask = model_pred.get("audio_loss_mask")
                                a_loss = _masked_mse(
                                    a_pred,
                                    a_tgt,
                                    a_mask,
                                    weighting=weighting,
                                    dtype=trainer.dit_dtype,
                                    loss_type=_val_lt,
                                    huber_delta=_val_hd,
                                )
                                c_audio.append(a_loss.item())
                                v_w = float(model_pred.get("video_loss_weight", 1.0))
                                a_w = float(model_pred.get("audio_loss_weight", 1.0))
                                loss_c = v_loss * v_w + a_loss * a_w
                            else:
                                v_w = float(model_pred.get("video_loss_weight", 1.0))
                                loss_c = v_loss * v_w
                        else:
                            if isinstance(target, torch.Tensor):
                                model_pred = model_pred.to(device=target.device, dtype=trainer.dit_dtype)
                            else:
                                model_pred = model_pred.to(dtype=trainer.dit_dtype)
                            if _val_lt in ("mae", "l1"):
                                per_el = torch.nn.functional.l1_loss(model_pred, target, reduction="none")
                            elif _val_lt in ("huber", "smooth_l1"):
                                per_el = torch.nn.functional.smooth_l1_loss(
                                    model_pred,
                                    target,
                                    reduction="none",
                                    beta=_val_hd,
                                )
                            else:
                                per_el = torch.nn.functional.mse_loss(model_pred, target, reduction="none")
                            if weighting is not None:
                                w = weighting
                                if isinstance(w, torch.Tensor) and w.dim() != per_el.dim():
                                    while w.dim() > per_el.dim() and w.shape[-1] == 1:
                                        w = w.squeeze(-1)
                                per_el = per_el * w
                            loss_c = per_el.mean()
                        c_losses.append(loss_c.item())
                        c_batches += 1
                    if c_losses:
                        val_metrics[f"val/{cat_name}/loss"] = sum(c_losses) / len(c_losses)
                    if c_video:
                        val_metrics[f"val/{cat_name}/video_loss"] = sum(c_video) / len(c_video)
                    if c_audio:
                        val_metrics[f"val/{cat_name}/audio_loss"] = sum(c_audio) / len(c_audio)
            if original_params is not None:
                ema_model.restore(accelerator.unwrap_model(transformer), original_params)
                original_params = None
            if validation_self_flow_enabled:
                args.self_flow = True
            transformer.train()

        if val_metrics:
            accelerator.print(f"Validation metrics: {val_metrics}")
            accelerator.log(val_metrics, step=step)

        return val_metrics

    def should_validate(step: int, epoch: int, is_epoch_end: bool) -> bool:
        """Check if validation should run."""
        if val_dataloader is None:
            return False
        if args.validate_every_n_steps is not None and step > 0 and step % args.validate_every_n_steps == 0:
            return True
        if is_epoch_end and args.validate_every_n_epochs is not None and (epoch + 1) % args.validate_every_n_epochs == 0:
            return True
        return False

    def run_validation_with_optimizer_offload(step: int, epoch: int) -> dict:
        with offload_optimizer_state_during_validation(
            optimizer,
            accelerator,
            bool(getattr(args, "offload_optimizer_during_validation", False)),
            logger=logger,
        ):
            return run_validation(step, epoch)

    def run_sampling_safely(sample_epoch, sample_step: int) -> None:
        """Run sampling preview without allowing failures to interrupt training."""
        optimizer_eval_fn()
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = None
        try:
            if torch.cuda.is_available():
                cuda_rng_state = torch.cuda.get_rng_state()
        except Exception:
            cuda_rng_state = None

        try:
            with offload_optimizer_state_during_validation(
                optimizer,
                accelerator,
                bool(getattr(args, "offload_optimizer_during_validation", False)),
                logger=logger,
            ):
                trainer.sample_images(
                    accelerator,
                    args,
                    sample_epoch,
                    sample_step,
                    vae,
                    transformer,
                    sample_parameters,
                    trainer.dit_dtype,
                )
        except Exception:
            logger.exception("Sampling failed at step=%s epoch=%s; continuing training.", sample_step, sample_epoch)
            try:
                unwrapped_transformer = accelerator.unwrap_model(transformer)
                if hasattr(unwrapped_transformer, "switch_block_swap_for_training"):
                    unwrapped_transformer.switch_block_swap_for_training()
                if hasattr(unwrapped_transformer, "move_to_device_except_swap_blocks"):
                    unwrapped_transformer.move_to_device_except_swap_blocks(accelerator.device)
                else:
                    unwrapped_transformer.to(accelerator.device)
            except Exception:
                logger.exception("Failed to restore transformer state after sampling failure.")
            try:
                clean_memory_on_device(accelerator.device)
            except Exception:
                pass
        finally:
            try:
                torch.set_rng_state(cpu_rng_state)
            except Exception:
                pass
            if cuda_rng_state is not None:
                try:
                    torch.cuda.set_rng_state(cuda_rng_state)
                except Exception:
                    pass
            optimizer_train_fn()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    metadata["ss_num_epochs"] = str(num_train_epochs)

    resume_batch_offset = 0
    if initial_global_step > 0 and resume_metadata is not None and resume_metadata.get("global_step", 0) > 0:
        saved_epoch = int(resume_metadata.get("epoch", 1) or 1)
        saved_step_in_epoch = int(resume_metadata.get("step_in_epoch", 0) or 0)
        if not getattr(args, "reset_dataloader", False) and saved_step_in_epoch > 0:
            resume_batch_offset = saved_step_in_epoch % max(len(train_dataloader), 1)
            if resume_batch_offset > 0:
                epoch_to_start = max(saved_epoch - 1, 0)
                steps_to_skip_in_epoch = resume_batch_offset
            else:
                epoch_to_start = max(saved_epoch, 0)
        else:
            inferred_batch_count = initial_global_step * args.gradient_accumulation_steps
            resume_batch_offset = inferred_batch_count % max(len(train_dataloader), 1)
            epoch_to_start = inferred_batch_count // max(len(train_dataloader), 1)
            if not getattr(args, "reset_dataloader", False):
                steps_to_skip_in_epoch = resume_batch_offset
    else:
        inferred_batch_count = initial_global_step * args.gradient_accumulation_steps
        epoch_to_start = inferred_batch_count // max(len(train_dataloader), 1) if initial_global_step > 0 else 0
        if initial_global_step > 0 and not getattr(args, "reset_dataloader", False):
            steps_to_skip_in_epoch = inferred_batch_count % max(len(train_dataloader), 1)

    last_resume_epoch = max(epoch_to_start + 1, 1)
    last_resume_step_in_epoch = steps_to_skip_in_epoch

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=initial_global_step,
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc="steps",
    )

    # For --sample_at_first
    if should_sample_images(args, global_step, epoch=0):
        run_sampling_safely(0, global_step)

    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples / サンプル数: {num_train_items}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    # Note: batch_size is hardcoded to 1 in DataLoader (line 337)
    args.train_batch_size = 1
    accelerator.print(f"  batch size per device / バッチサイズ: {args.train_batch_size}")
    accelerator.print(
        f"  total train batch size (with parallel & accumulation) / 総バッチサイズ: {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}"
    )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")
    if initial_global_step > 0:
        msg = f"  resuming from step {initial_global_step}, epoch {epoch_to_start + 1}/{num_train_epochs}"
        if steps_to_skip_in_epoch > 0:
            msg += f", skipping {steps_to_skip_in_epoch} batches in epoch"
        accelerator.print(msg)

    optimizer_train_fn()

    motion_anchor_cache: list[dict[str, Any]] = []
    motion_micro_step = 0
    motion_attention_modules: list[tuple[str, torch.nn.Module]] = []
    if args.motion_attention_preservation:
        motion_attention_modules = _collect_motion_attention_modules(
            transformer,
            getattr(args, "motion_attention_preservation_blocks", None),
        )
        motion_attention_modules = _filter_motion_attention_modules_for_swap(
            motion_attention_modules,
            transformer=transformer,
            accelerator=accelerator,
            blocks_to_swap=blocks_to_swap,
        )
        if not motion_attention_modules:
            logger.warning("motion_attention_preservation requested but no matching attn1 modules were found; disabling it.")
            args.motion_attention_preservation = False
        else:
            logger.info(
                "Motion attention preservation enabled on %d attn1 modules "
                "(queries=%d keys=%d, loss=%s, per_head=%s, temp=%.3f, symmetric_kl=%s)",
                len(motion_attention_modules),
                int(getattr(args, "motion_attention_preservation_queries", 32) or 32),
                int(getattr(args, "motion_attention_preservation_keys", 64) or 64),
                getattr(args, "motion_attention_preservation_loss", "kl"),
                str(bool(getattr(args, "motion_attention_preservation_per_head", False))),
                float(getattr(args, "motion_attention_preservation_temperature", 1.0) or 1.0),
                str(bool(getattr(args, "motion_attention_preservation_symmetric_kl", False))),
            )
    metadata["ss_motion_attention_preservation_active"] = str(bool(args.motion_attention_preservation))
    metadata["ss_motion_attention_preservation_module_count"] = str(len(motion_attention_modules))

    if args.motion_preservation:
        motion_anchor_cache = _build_motion_anchor_cache(
            trainer=trainer,
            args=args,
            accelerator=accelerator,
            transformer=transformer,
            train_dataloader=train_dataloader,
            noise_scheduler=noise_scheduler,
            attention_modules=motion_attention_modules,
            normalize_batch_fn=_normalize_ltx2_batch_for_call_dit,
        )
        if not motion_anchor_cache:
            logger.warning("motion_preservation requested but no anchors were built; disabling.")
            args.motion_preservation = False
        elif bool(getattr(args, "motion_prior_require_temporal", False)):
            temporal_anchor_count = 0
            for entry in motion_anchor_cache:
                anchor_latents = entry.get("anchor_latents")
                if isinstance(anchor_latents, torch.Tensor) and anchor_latents.dim() == 5 and int(anchor_latents.shape[2]) > 1:
                    temporal_anchor_count += 1
            if temporal_anchor_count <= 0:
                message = (
                    "motion_prior_require_temporal is enabled, but cache has no multi-frame anchors. "
                    "Set --motion_preservation_anchor_source synthetic or hybrid."
                )
                if bool(getattr(args, "motion_prior_cache_only", False)):
                    raise ValueError(message)
                logger.warning("%s Disabling motion_preservation.", message)
                args.motion_preservation = False

    if bool(getattr(args, "motion_prior_cache_only", False)):
        logger.info(
            "motion_prior_cache_only enabled: cache build phase finished (anchors=%d). Exiting before optimization.",
            len(motion_anchor_cache),
        )
        return

    ewc_state: Optional[dict[str, Any]] = None
    if float(getattr(args, "ewc_lambda", 0.0) or 0.0) > 0.0:
        ewc_cache_path = getattr(args, "ewc_cache_path", None)
        ewc_cache_rebuild = bool(getattr(args, "ewc_cache_rebuild", False))
        ewc_signature = _build_ewc_cache_signature(args)
        loaded_ewc_cache = False
        if ewc_cache_path and not ewc_cache_rebuild:
            ewc_state = _load_ewc_cache(
                ewc_cache_path,
                ewc_signature,
                transformer,
                target_device=accelerator.device,
            )
            loaded_ewc_cache = ewc_state is not None
        elif ewc_cache_path and ewc_cache_rebuild:
            logger.info("EWC cache rebuild requested; ignoring existing cache: %s", ewc_cache_path)

        if ewc_state is None:
            if fused_step_state is not None:
                logger.info("Suspending fused optimizer hooks during Fisher/EWC stats build.")
                fused_step_state["suspend_step"] = True
            try:
                ewc_state = _build_fisher_ewc_stats(
                    trainer=trainer,
                    args=args,
                    accelerator=accelerator,
                    transformer=transformer,
                    train_dataloader=train_dataloader,
                    noise_scheduler=noise_scheduler,
                    optimizer=optimizer,
                    target_device=accelerator.device,
                )
            finally:
                if fused_step_state is not None:
                    fused_step_state["suspend_step"] = False
                    optimizer.zero_grad(set_to_none=True)
            if ewc_state is not None and ewc_cache_path and not loaded_ewc_cache:
                _save_ewc_cache(ewc_cache_path, ewc_signature, ewc_state)
        if ewc_state is None:
            logger.warning("EWC requested but statistics were not built; disabling EWC loss.")

    weight_drift_state: Optional[dict[str, Any]] = _build_weight_drift_state(args, transformer)
    grad_norm_state: Optional[dict[str, Any]] = _register_grad_norm_hooks(args, accelerator.unwrap_model(transformer))
    output_drift_state: Optional[dict[str, Any]] = _build_output_drift_state(
        args,
        trainer,
        transformer,
        val_dataloader,
        noise_scheduler,
        accelerator,
    )

    audio_loss_balance_mode = str(getattr(args, "audio_loss_balance_mode", "none") or "none").lower()
    audio_loss_balance_beta = float(getattr(args, "audio_loss_balance_beta", 0.01))
    audio_loss_balance_eps = float(getattr(args, "audio_loss_balance_eps", 0.05))
    audio_loss_balance_min = float(getattr(args, "audio_loss_balance_min", 0.05))
    audio_loss_balance_max = float(getattr(args, "audio_loss_balance_max", 4.0))
    audio_loss_balance_ema_init = float(getattr(args, "audio_loss_balance_ema_init", 1.0))
    audio_loss_balance_target_ratio = float(getattr(args, "audio_loss_balance_target_ratio", 0.33))
    audio_loss_balance_ema_decay = float(getattr(args, "audio_loss_balance_ema_decay", 0.99))
    ogm_ge_alpha = float(getattr(args, "ogm_ge_alpha", 0.3))

    cli_video_loss_weight = float(getattr(args, "video_loss_weight", 1.0))
    cli_audio_loss_weight = float(getattr(args, "audio_loss_weight", 1.0))

    audio_presence_ema = min(max(audio_loss_balance_ema_init, 1e-6), 1.0)
    audio_loss_ema = max(audio_loss_balance_ema_init, 1e-6)
    video_loss_ema = max(audio_loss_balance_ema_init, 1e-6)

    uncertainty_log_var_video = uncertainty_log_var_audio = None
    if audio_loss_balance_mode == "uncertainty":
        device_for_uncert = accelerator.device if accelerator is not None else torch.device("cpu")
        uncertainty_log_var_video = torch.nn.Parameter(torch.zeros(1, device=device_for_uncert))
        uncertainty_log_var_audio = torch.nn.Parameter(torch.zeros(1, device=device_for_uncert))
        uncertainty_lr = float(getattr(args, "uncertainty_lr", None) or args.learning_rate)
        try:
            optimizer.add_param_group({"params": [uncertainty_log_var_video, uncertainty_log_var_audio], "lr": uncertainty_lr})
            logger.info(
                "Uncertainty weighting enabled: 2 learnable log-variance params added to optimizer, lr=%g",
                uncertainty_lr,
            )
        except Exception as e:
            logger.warning(
                "Failed to add uncertainty params to optimizer (%s); uncertainty mode will not learn variances.",
                e,
            )

    if audio_loss_balance_mode == "ogm_ge":
        logger.info("OGM-GE audio/video balancing enabled: alpha=%.4f", ogm_ge_alpha)
    elif audio_loss_balance_mode == "ema_mag":
        logger.info(
            "EMA-magnitude audio balancing enabled: target_ratio=%.4f ema_decay=%.4f min=%.4f max=%.4f",
            audio_loss_balance_target_ratio,
            audio_loss_balance_ema_decay,
            audio_loss_balance_min,
            audio_loss_balance_max,
        )
    elif audio_loss_balance_mode == "inv_freq":
        logger.info(
            "Inverse-frequency audio balancing enabled: beta=%.4f eps=%.4f min=%.4f max=%.4f",
            audio_loss_balance_beta,
            audio_loss_balance_eps,
            audio_loss_balance_min,
            audio_loss_balance_max,
        )

    for epoch in range(epoch_to_start, num_train_epochs):
        current_epoch.value = epoch + 1
        if train_audio_sampler is not None:
            if train_dataset_group_for_audio_sampler is None:
                raise RuntimeError("Internal error: audio sampler is enabled but the dataset group was released")
            sync_dataset_group_epoch_without_loading(train_dataset_group_for_audio_sampler, epoch + 1, logger=logger)
            audio_indices, non_audio_indices = split_concat_indices_by_audio(train_dataset_group_for_audio_sampler)
            if len(audio_indices) == 0:
                raise ValueError(
                    f"No audio-bearing batches available at epoch {epoch + 1} while "
                    f"{'--min_audio_batches_per_accum' if train_audio_sampler_mode == 'quota' else '--audio_batch_probability'} is enabled."
                )
            if hasattr(train_audio_sampler, "update_groups"):
                train_audio_sampler.update_groups(audio_indices, non_audio_indices)
            if hasattr(train_audio_sampler, "set_epoch"):
                train_audio_sampler.set_epoch(epoch)

        metadata["ss_epoch"] = str(epoch + 1)
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            if handle_dashboard_stop_request(global_step, epoch, step):
                return

            if steps_to_skip_in_epoch > 0:
                steps_to_skip_in_epoch -= 1
                continue

            with accelerator.accumulate(transformer):
                motion_micro_step += 1
                batch = _normalize_ltx2_batch_for_call_dit(batch)
                latents = batch["latents"]
                if isinstance(latents, dict):
                    if "latents" not in latents:
                        raise ValueError("batch['latents'] is a dict but missing key 'latents'")
                    latents_tensor = latents["latents"]
                else:
                    latents_tensor = latents

                latents_tensor = trainer.scale_shift_latents(latents_tensor)
                noise = torch.randn_like(latents_tensor)

                noisy_model_input, timesteps = trainer.get_noisy_model_input_and_timesteps(
                    args,
                    noise,
                    latents_tensor,
                    batch["timesteps"],
                    noise_scheduler,
                    accelerator.device,
                    trainer.dit_dtype,
                )

                weighting = compute_loss_weighting_for_sd3(
                    args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, trainer.dit_dtype
                )

                trainer._current_train_global_step = global_step
                model_pred, target = trainer.call_dit(
                    args,
                    accelerator,
                    transformer,
                    latents_tensor,
                    batch,
                    noise,
                    noisy_model_input,
                    timesteps,
                    trainer.dit_dtype,
                )

                dict_output = isinstance(model_pred, dict)
                if dict_output:
                    out = model_pred
                    if out.get("_skip_step"):
                        logger.warning(
                            "Skipping step due to non-finite tensor (%s).",
                            out.get("skip_reason", "unknown"),
                        )
                        optimizer.zero_grad(set_to_none=True)
                        if ltx2_remote_stage and bool(getattr(args, "ltx2_remote_stage_trainable", False)):
                            zero_grad_ltx2_remote_stage(transformer)
                        continue

                    _loss_type = getattr(args, "loss_type", "mse")
                    _huber_delta = float(getattr(args, "huber_delta", 1.0))
                    video_pred = out["video_pred"]
                    video_target = out["video_target"]
                    video_loss_mask = out.get("video_loss_mask")
                    video_loss = _masked_mse(
                        video_pred,
                        video_target,
                        video_loss_mask,
                        weighting=weighting,
                        dtype=trainer.dit_dtype,
                        loss_type=_loss_type,
                        huber_delta=_huber_delta,
                        per_elem_modifier=lambda per_elem: trainer.modify_video_loss_per_element(
                            args, per_elem, out, trainer.dit_dtype
                        ),
                    )
                    # Base weights: dataset config × CLI override.
                    video_weight = float(out.get("video_loss_weight", 1.0)) * cli_video_loss_weight

                    audio_pred = out.get("audio_pred")
                    audio_target = out.get("audio_target")
                    audio_loss_mask = out.get("audio_loss_mask")
                    audio_loss = None
                    audio_weight = None
                    has_audio_loss = audio_pred is not None and audio_target is not None
                    if has_audio_loss:
                        audio_loss = _masked_mse(
                            audio_pred,
                            audio_target,
                            audio_loss_mask,
                            weighting=weighting,
                            dtype=trainer.dit_dtype,
                            loss_type=_loss_type,
                            huber_delta=_huber_delta,
                            per_elem_modifier=lambda per_elem: trainer.modify_audio_loss_per_element(
                                args, per_elem, out, trainer.dit_dtype
                            ),
                        )
                        audio_weight = float(out.get("audio_loss_weight", 1.0)) * cli_audio_loss_weight

                    # Uncertainty mode replaces the manual sum with Kendall et al. weighting.
                    if (
                        audio_loss_balance_mode == "uncertainty"
                        and has_audio_loss
                        and uncertainty_log_var_video is not None
                        and uncertainty_log_var_audio is not None
                    ):
                        loss = compute_uncertainty_weighted_loss(
                            video_loss,
                            audio_loss,
                            uncertainty_log_var_video,
                            uncertainty_log_var_audio,
                        )
                        if (global_step % 50) == 0:
                            logger.info(
                                "[Uncertainty step %d] log_var_v=%.4f log_var_a=%.4f",
                                global_step,
                                float(uncertainty_log_var_video.detach().item()),
                                float(uncertainty_log_var_audio.detach().item()),
                            )
                    elif audio_loss_balance_mode == "ogm_ge" and has_audio_loss:
                        ogm_ge_state = compute_ogm_ge_coefficients(
                            video_loss=video_loss,
                            audio_loss=audio_loss,
                            alpha=ogm_ge_alpha,
                        )
                        video_weight = video_weight * float(ogm_ge_state.video_coeff)
                        audio_weight = audio_weight * float(ogm_ge_state.audio_coeff)
                        if (global_step % 50) == 0:
                            logger.info(
                                "[OGM-GE step %d] dominant=%s discrepancy=%.4f v_coeff=%.4f a_coeff=%.4f",
                                global_step,
                                ogm_ge_state.dominant,
                                ogm_ge_state.discrepancy,
                                ogm_ge_state.video_coeff,
                                ogm_ge_state.audio_coeff,
                            )
                        loss = video_loss * video_weight + audio_loss * audio_weight
                    else:
                        # ema_mag / inv_freq / none all use the additive form with possibly-rescaled audio weight.
                        if audio_loss_balance_mode == "ema_mag":
                            video_loss_item = max(float(video_loss.detach().item()), 1e-12)
                            video_loss_ema = update_loss_ema(
                                loss_ema=video_loss_ema,
                                loss_value=video_loss_item,
                                ema_decay=audio_loss_balance_ema_decay,
                            )
                        if audio_loss_balance_mode == "inv_freq":
                            audio_presence_ema = update_audio_presence_ema(
                                audio_presence_ema=audio_presence_ema,
                                balance_beta=audio_loss_balance_beta,
                                has_audio_loss=has_audio_loss,
                            )

                        loss = video_loss * video_weight
                        if has_audio_loss:
                            if audio_loss_balance_mode == "inv_freq":
                                audio_weight = compute_inverse_frequency_audio_weight(
                                    base_audio_weight=audio_weight,
                                    audio_presence_ema=audio_presence_ema,
                                    balance_eps=audio_loss_balance_eps,
                                    balance_min=audio_loss_balance_min,
                                    balance_max=audio_loss_balance_max,
                                )
                                if (global_step % 50) == 0:
                                    logger.info(
                                        "[inv_freq step %d] audio_presence_ema=%.4f audio_weight=%.4f",
                                        global_step,
                                        audio_presence_ema,
                                        audio_weight,
                                    )
                            elif audio_loss_balance_mode == "ema_mag":
                                audio_loss_item = max(float(audio_loss.detach().item()), 1e-12)
                                audio_loss_ema = update_loss_ema(
                                    loss_ema=audio_loss_ema,
                                    loss_value=audio_loss_item,
                                    ema_decay=audio_loss_balance_ema_decay,
                                )
                                audio_weight = compute_ema_magnitude_audio_weight(
                                    base_audio_weight=audio_weight,
                                    audio_loss_ema=audio_loss_ema,
                                    video_loss_ema=video_loss_ema,
                                    target_audio_ratio=audio_loss_balance_target_ratio,
                                    balance_min=audio_loss_balance_min,
                                    balance_max=audio_loss_balance_max,
                                )
                                if (global_step % 50) == 0:
                                    logger.info(
                                        "[ema_mag step %d] v_ema=%.4f a_ema=%.4f audio_weight=%.4f",
                                        global_step,
                                        video_loss_ema,
                                        audio_loss_ema,
                                        audio_weight,
                                    )
                            loss = loss + audio_loss * audio_weight
                else:
                    _loss_type = getattr(args, "loss_type", "mse")
                    _huber_delta = float(getattr(args, "huber_delta", 1.0))
                    if isinstance(target, torch.Tensor):
                        model_pred = model_pred.to(device=target.device, dtype=trainer.dit_dtype)
                    else:
                        model_pred = model_pred.to(dtype=trainer.dit_dtype)
                    if _loss_type in ("mae", "l1"):
                        loss = torch.nn.functional.l1_loss(model_pred, target, reduction="none")
                    elif _loss_type in ("huber", "smooth_l1"):
                        loss = torch.nn.functional.smooth_l1_loss(model_pred, target, reduction="none", beta=_huber_delta)
                    else:
                        loss = torch.nn.functional.mse_loss(model_pred, target, reduction="none")
                    if weighting is not None:
                        w = weighting
                        if isinstance(w, torch.Tensor) and w.dim() != loss.dim():
                            while w.dim() > loss.dim() and w.shape[-1] == 1:
                                w = w.squeeze(-1)
                        loss = loss * w
                    loss = loss.mean()

                self_flow_loss = None
                self_flow_metrics: dict[str, float] = {}
                if bool(getattr(args, "self_flow", False)):
                    try:
                        if getattr(trainer, "_self_flow", None) is not None:
                            trainer._self_flow.on_step(global_step)
                        self_flow_loss, self_flow_metrics = trainer.compute_self_flow_addition(
                            args,
                            accelerator,
                            transformer,
                            transformer,
                            trainer.dit_dtype,
                        )
                        if self_flow_loss is not None:
                            loss = loss + self_flow_loss
                    except Exception as e:
                        if getattr(trainer, "_self_flow", None) is not None:
                            trainer._self_flow.cleanup_step()
                        logger.warning("Self-Flow loss computation failed at step=%s: %s", global_step, e)

                motion_pres_loss = None
                motion_pres_loss_raw = None
                attn_pres_loss = None
                attn_pres_loss_raw = None
                motion_total_loss = None
                ewc_loss = None
                ewc_penalty_raw = None
                ewc_used_tensors = 0
                ewc_skipped_tensors = 0
                replay_sigma_value: Optional[float] = None
                replay_anchor_frames: Optional[int] = None
                motion_to_task_ratio: Optional[float] = None
                image_prior_ft_task_grads_restored = 0
                pending_attn_recorder: Optional[_AttentionMapRecorder] = None
                pending_motion_route_context = None
                motion_preservation_prob = getattr(args, "motion_preservation_probability", None)
                if motion_preservation_prob is None:
                    should_apply_motion_replay = motion_micro_step % int(args.motion_preservation_interval) == 0
                else:
                    should_apply_motion_replay = random.random() < float(motion_preservation_prob)
                if ewc_state is not None and float(getattr(args, "ewc_lambda", 0.0) or 0.0) > 0.0:
                    ewc_penalty_raw, ewc_used_tensors, ewc_skipped_tensors = _compute_ewc_penalty(
                        ewc_state,
                        dtype=trainer.dit_dtype,
                        target_device=loss.device if isinstance(loss, torch.Tensor) else None,
                    )
                    if ewc_penalty_raw is not None:
                        ewc_loss = ewc_penalty_raw * float(getattr(args, "ewc_lambda", 0.0))
                        loss = loss + ewc_loss
                separate_motion_backward = bool(getattr(args, "motion_preservation_separate_backward", False))
                fused_defer_motion_step = bool(
                    args.fused_backward_pass
                    and (
                        bool(getattr(args, "motion_preservation_fused_defer_step", False))
                        or _image_prior_ft_routing_active(image_prior_ft_route_state)
                    )
                    and separate_motion_backward
                    and (
                        _image_prior_ft_routing_active(image_prior_ft_route_state)
                        or (args.motion_preservation and motion_anchor_cache and should_apply_motion_replay)
                    )
                )
                if fused_step_state is not None:
                    fused_step_state["defer_step"] = fused_defer_motion_step
                    fused_step_state["hook_stepped"] = False
                if separate_motion_backward:
                    # Backprop task loss first so replay graph does not overlap it in memory.
                    image_prior_ft_task_grad_snapshot = _image_prior_ft_snapshot_task_route_grads(image_prior_ft_route_state)
                    accelerator.backward(loss)
                    image_prior_ft_task_grads_restored = _image_prior_ft_restore_task_route_grads(image_prior_ft_task_grad_snapshot)
                    total_loss_for_logging = loss.detach()
                if args.motion_preservation and motion_anchor_cache and should_apply_motion_replay:
                    anchor = random.choice(motion_anchor_cache)
                    anchor_latents = _move_to_device(anchor["anchor_latents"], accelerator.device, dtype=trainer.dit_dtype)
                    replay_anchor_frames = int(anchor_latents.shape[2]) if anchor_latents.dim() >= 3 else None
                    anchor_noise = _move_to_device(anchor["anchor_noise"], accelerator.device, dtype=trainer.dit_dtype)
                    sigma_idx: Optional[int] = None
                    anchor_sigmas = anchor.get("anchor_sigmas") or []
                    teacher_video_preds = anchor.get("teacher_video_preds")
                    if (
                        isinstance(anchor_sigmas, list)
                        and isinstance(teacher_video_preds, list)
                        and len(anchor_sigmas) > 0
                        and len(anchor_sigmas) == len(teacher_video_preds)
                    ):
                        sigma_idx = _sample_motion_sigma_index(anchor_sigmas, args)
                        sigma_value = float(anchor_sigmas[sigma_idx])
                        replay_sigma_value = float(sigma_value)
                        anchor_noisy_input, anchor_model_timesteps = _build_noisy_input_for_sigma(
                            anchor_latents,
                            anchor_noise,
                            sigma_value,
                        )
                        teacher_video_pred = teacher_video_preds[sigma_idx]
                    else:
                        anchor_noisy_input = _move_to_device(
                            anchor["anchor_noisy_input"], accelerator.device, dtype=trainer.dit_dtype
                        )
                        anchor_model_timesteps = _move_to_device(anchor["anchor_model_timesteps"], accelerator.device)
                        if isinstance(anchor_model_timesteps, torch.Tensor) and anchor_model_timesteps.numel() > 0:
                            replay_sigma_value = float(anchor_model_timesteps.view(-1)[0].detach().float().item() / 1000.0)
                        teacher_video_pred = anchor.get("teacher_video_pred")
                        if not isinstance(teacher_video_pred, torch.Tensor):
                            logger.warning(
                                "Motion replay: teacher_video_pred is %s instead of Tensor, skipping replay this step",
                                type(teacher_video_pred).__name__,
                            )
                    anchor_batch = _move_to_device(anchor["anchor_batch"], accelerator.device, dtype=trainer.dit_dtype)

                    original_first_frame_p = float(getattr(args, "ltx2_first_frame_conditioning_p", 0.0))
                    # Force first-frame conditioning probability to 0 during motion replay so that
                    # the student prediction is generated under the same conditions as the cached
                    # teacher prediction (which was recorded with p=0).  A mismatch would introduce
                    # a systematic bias between student and teacher outputs, degrading the motion
                    # preservation loss signal.
                    setattr(args, "ltx2_first_frame_conditioning_p", 0.0)
                    try:
                        pending_motion_route_context = _image_prior_ft_route_context(image_prior_ft_route_state, "motion")
                        pending_motion_route_context.__enter__()
                        if args.motion_attention_preservation and motion_attention_modules:
                            attn_recorder = _AttentionMapRecorder(
                                motion_attention_modules,
                                max_queries=int(getattr(args, "motion_attention_preservation_queries", 32) or 32),
                                max_keys=int(getattr(args, "motion_attention_preservation_keys", 64) or 64),
                                capture_grad=True,
                                keep_heads=bool(getattr(args, "motion_attention_preservation_per_head", False)),
                            )
                            attn_recorder.__enter__()
                            pending_attn_recorder = attn_recorder
                            try:
                                motion_pred, _ = trainer.call_dit(
                                    args,
                                    accelerator,
                                    transformer,
                                    anchor_latents,
                                    anchor_batch,
                                    anchor_noise,
                                    anchor_noisy_input,
                                    anchor_model_timesteps,
                                    trainer.dit_dtype,
                                )
                                student_attn_maps = attn_recorder.collect_maps()
                            except Exception:
                                attn_recorder.__exit__(None, None, None)
                                pending_attn_recorder = None
                                if pending_motion_route_context is not None:
                                    pending_motion_route_context.__exit__(None, None, None)
                                    pending_motion_route_context = None
                                raise
                        else:
                            student_attn_maps = {}
                            motion_pred, _ = trainer.call_dit(
                                args,
                                accelerator,
                                transformer,
                                anchor_latents,
                                anchor_batch,
                                anchor_noise,
                                anchor_noisy_input,
                                anchor_model_timesteps,
                                trainer.dit_dtype,
                            )
                    except Exception:
                        if pending_attn_recorder is not None:
                            pending_attn_recorder.__exit__(None, None, None)
                            pending_attn_recorder = None
                        if pending_motion_route_context is not None:
                            pending_motion_route_context.__exit__(None, None, None)
                            pending_motion_route_context = None
                        raise
                    finally:
                        setattr(args, "ltx2_first_frame_conditioning_p", original_first_frame_p)

                    if (
                        isinstance(motion_pred, dict)
                        and not motion_pred.get("_skip_step")
                        and isinstance(teacher_video_pred, torch.Tensor)
                    ):
                        student_video_pred = motion_pred["video_pred"]
                        motion_pres_loss_raw = _compute_motion_preservation_loss(
                            args,
                            student_video_pred,
                            teacher_video_pred,
                            motion_pred.get("video_loss_mask"),
                            dtype=trainer.dit_dtype,
                        )
                        motion_multiplier = float(args.motion_preservation_multiplier)
                        motion_warmup = int(getattr(args, "motion_preservation_warmup_steps", 0) or 0)
                        if motion_warmup > 0 and global_step < motion_warmup:
                            motion_multiplier = motion_multiplier * (global_step / motion_warmup)
                        motion_pres_loss = motion_pres_loss_raw * motion_multiplier
                        motion_total_loss = motion_pres_loss

                        teacher_attn_maps = anchor.get("teacher_attention_maps")
                        teacher_attn_maps_multi = anchor.get("teacher_attention_maps_multi")
                        if isinstance(teacher_attn_maps_multi, list) and teacher_attn_maps_multi:
                            if sigma_idx is not None and 0 <= sigma_idx < len(teacher_attn_maps_multi):
                                mapped = teacher_attn_maps_multi[sigma_idx]
                                if isinstance(mapped, dict):
                                    teacher_attn_maps = mapped
                            elif sigma_idx is None and len(teacher_attn_maps_multi) == 1:
                                mapped = teacher_attn_maps_multi[0]
                                if isinstance(mapped, dict):
                                    teacher_attn_maps = mapped
                        if args.motion_attention_preservation and isinstance(teacher_attn_maps, dict) and student_attn_maps:
                            attn_temperature = float(getattr(args, "motion_attention_preservation_temperature", 1.0) or 1.0)
                            symmetric_kl = bool(getattr(args, "motion_attention_preservation_symmetric_kl", False))
                            per_block_losses: list[torch.Tensor] = []
                            for module_name, student_map in student_attn_maps.items():
                                teacher_map = teacher_attn_maps.get(module_name)
                                if not isinstance(teacher_map, torch.Tensor):
                                    continue
                                if teacher_map.shape != student_map.shape:
                                    logger.warning(
                                        "Motion preservation: attention map shape mismatch for module %s (student=%s teacher=%s), skipping",
                                        module_name,
                                        student_map.shape,
                                        teacher_map.shape,
                                    )
                                    continue

                                student_dist = student_map.to(torch.float32)
                                teacher_dist = teacher_map.to(device=student_dist.device, dtype=torch.float32, non_blocking=True)
                                if attn_temperature != 1.0:
                                    inv_temp = 1.0 / max(1e-6, attn_temperature)
                                    student_dist = student_dist.clamp_min(1e-8).pow(inv_temp)
                                    teacher_dist = teacher_dist.clamp_min(1e-8).pow(inv_temp)
                                student_dist = student_dist.clamp_min(1e-6)
                                teacher_dist = teacher_dist.clamp_min(1e-6)
                                student_dist = student_dist / student_dist.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                                teacher_dist = teacher_dist / teacher_dist.sum(dim=-1, keepdim=True).clamp_min(1e-6)

                                if getattr(args, "motion_attention_preservation_loss", "kl") == "kl":
                                    kl_t2s = torch.nn.functional.kl_div(
                                        student_dist.log(),
                                        teacher_dist,
                                        reduction="batchmean",
                                    )
                                    if symmetric_kl:
                                        kl_s2t = torch.nn.functional.kl_div(
                                            teacher_dist.log(),
                                            student_dist,
                                            reduction="batchmean",
                                        )
                                        block_loss = 0.5 * (kl_t2s + kl_s2t)
                                    else:
                                        block_loss = kl_t2s
                                else:
                                    block_loss = torch.nn.functional.mse_loss(student_dist, teacher_dist)
                                per_block_losses.append(block_loss)

                            if per_block_losses:
                                attn_pres_loss_raw = torch.stack(per_block_losses).mean()
                                attn_pres_loss = attn_pres_loss_raw * float(
                                    getattr(args, "motion_attention_preservation_weight", 0.0)
                                )
                                motion_total_loss = motion_total_loss + attn_pres_loss

                if motion_total_loss is not None and isinstance(loss, torch.Tensor):
                    base_task_abs = loss.detach().to(torch.float32).abs().clamp_min(1e-8)
                    motion_to_task_ratio = (motion_total_loss.detach().to(torch.float32).abs() / base_task_abs).item()

                if separate_motion_backward:
                    if motion_total_loss is not None:
                        if fused_step_state is not None:
                            fused_step_state["defer_step"] = False
                        accelerator.backward(motion_total_loss)
                        total_loss_for_logging = total_loss_for_logging + motion_total_loss.detach()
                    elif fused_defer_motion_step:
                        if fused_step_state is not None:
                            fused_step_state["defer_step"] = False
                    loss_for_step = total_loss_for_logging
                else:
                    if motion_total_loss is not None:
                        loss = loss + motion_total_loss
                    accelerator.backward(loss)
                    if not separate_motion_backward:
                        image_prior_ft_task_grads_restored = 0
                    loss_for_step = loss.detach()
                if pending_motion_route_context is not None:
                    pending_motion_route_context.__exit__(None, None, None)
                    pending_motion_route_context = None
                if pending_attn_recorder is not None:
                    pending_attn_recorder.__exit__(None, None, None)
                    pending_attn_recorder = None
                did_optimizer_step = False
                if not args.fused_backward_pass:
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        _clip_grad_norm_ltx2(
                            transformer.parameters(),
                            accelerator,
                            args.max_grad_norm,
                            ltx2_model_parallel=ltx2_model_parallel,
                            optimizer=optimizer,
                        )
                    optimizer.step()
                    if (
                        accelerator.sync_gradients
                        and ltx2_remote_stage
                        and bool(getattr(args, "ltx2_remote_stage_trainable", False))
                    ):
                        optimizer_step_ltx2_remote_stage(transformer)
                    did_optimizer_step = bool(accelerator.sync_gradients)
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    did_fused_step = False
                    if accelerator.sync_gradients:
                        if fused_step_state is not None:
                            fused_step_state["defer_step"] = False
                        hook_stepped = bool(fused_step_state is not None and fused_step_state.get("hook_stepped", False))
                        if badam_gr_active:
                            # Hooks already applied per-param updates; wrapper.step() advances
                            # the block-switch counter and runs end-of-iteration housekeeping.
                            optimizer.step()
                            did_fused_step = True
                        else:
                            pending_steps = _fused_step_pending_grads(
                                optimizer,
                                accelerator,
                                args.max_grad_norm,
                                ltx2_model_parallel=ltx2_model_parallel,
                            )
                            did_fused_step = hook_stepped or pending_steps > 0
                        if did_fused_step and ltx2_remote_stage and bool(getattr(args, "ltx2_remote_stage_trainable", False)):
                            optimizer_step_ltx2_remote_stage(transformer)
                        optimizer.zero_grad(set_to_none=True)
                        if ltx2_remote_stage and bool(getattr(args, "ltx2_remote_stage_trainable", False)):
                            zero_grad_ltx2_remote_stage(transformer)
                    if did_fused_step:
                        lr_scheduler.step()
                    did_optimizer_step = did_fused_step

                if did_optimizer_step and getattr(trainer, "_self_flow", None) is not None:
                    try:
                        trainer._self_flow.update_teacher(accelerator.unwrap_model(transformer))
                    except Exception as e:
                        logger.warning("Self-Flow teacher update failed at step=%s: %s", global_step, e)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                last_resume_epoch = epoch + 1
                last_resume_step_in_epoch = step + 1
                current_loss = loss_for_step.item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)

                # Update EMA weights
                if ema_model is not None:
                    ema_model.update(accelerator.unwrap_model(transformer))

                # Update progress bar with current metrics
                current_lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, "get_last_lr") else args.learning_rate
                logs = {"loss": current_loss, "lr": current_lr}
                if bool(getattr(args, "image_prior_ft", False)):
                    logs["image_prior_ft/image_only"] = 1.0
                    logs["image_prior_ft/task_motion_blocked"] = float(
                        image_prior_ft_route_state is not None and image_prior_ft_route_state.get("task_route") == "appearance"
                    )
                    logs["image_prior_ft/motion_appearance_blocked"] = float(
                        image_prior_ft_route_state is not None and image_prior_ft_route_state.get("motion_route") == "motion"
                    )
                    logs["image_prior_ft/task_grads_restored"] = float(image_prior_ft_task_grads_restored)
                logs["motion/apply"] = float(bool(args.motion_preservation and motion_anchor_cache and should_apply_motion_replay))
                lr_scales = ft_group_stats.get("lr_scales", [])
                for i, param_group in enumerate(optimizer.param_groups):
                    lr_value = param_group.get("lr", current_lr)
                    logs[f"lr/group_{i}"] = float(lr_value)
                    if i < len(lr_scales):
                        logs[f"lr_scale/group_{i}"] = float(lr_scales[i])
                if dict_output:
                    if "video_pred" in out:
                        logs["v_loss"] = video_loss.item()
                    if audio_pred is not None and audio_target is not None:
                        logs["a_loss"] = audio_loss.item()
                if motion_pres_loss is not None:
                    logs["motion_pres"] = motion_pres_loss.detach().item()
                    logs["motion/pres_weighted"] = motion_pres_loss.detach().item()
                if self_flow_loss is not None:
                    logs["self_flow"] = self_flow_loss.detach().item()
                if self_flow_metrics:
                    logs.update(self_flow_metrics)
                if motion_pres_loss_raw is not None:
                    logs["motion/pres_raw"] = motion_pres_loss_raw.detach().item()
                if motion_pres_loss is not None:
                    _mw = int(getattr(args, "motion_preservation_warmup_steps", 0) or 0)
                    _mm = float(args.motion_preservation_multiplier)
                    if _mw > 0 and global_step < _mw:
                        _mm = _mm * (global_step / _mw)
                    logs["motion/effective_multiplier"] = _mm
                if attn_pres_loss is not None:
                    logs["attn_pres"] = attn_pres_loss.detach().item()
                    logs["motion/attn_weighted"] = attn_pres_loss.detach().item()
                if attn_pres_loss_raw is not None:
                    logs["motion/attn_raw"] = attn_pres_loss_raw.detach().item()
                if motion_total_loss is not None:
                    logs["motion/total"] = motion_total_loss.detach().item()
                if motion_to_task_ratio is not None:
                    logs["motion/total_to_task"] = float(motion_to_task_ratio)
                if replay_sigma_value is not None:
                    logs["motion/sigma"] = float(replay_sigma_value)
                if replay_anchor_frames is not None:
                    logs["motion/anchor_frames"] = float(replay_anchor_frames)
                if ewc_loss is not None:
                    logs["ewc"] = ewc_loss.detach().item()
                if ewc_penalty_raw is not None:
                    logs["ewc/raw"] = ewc_penalty_raw.detach().item()
                if ewc_state is not None:
                    logs["ewc/used_tensors"] = float(ewc_used_tensors)
                    logs["ewc/skipped_tensors"] = float(ewc_skipped_tensors)
                effective_reg_value = 0.0
                has_effective_reg = False
                if motion_total_loss is not None:
                    effective_reg_value += float(motion_total_loss.detach().item())
                    has_effective_reg = True
                if ewc_loss is not None:
                    effective_reg_value += float(ewc_loss.detach().item())
                    has_effective_reg = True
                if self_flow_loss is not None:
                    effective_reg_value += float(self_flow_loss.detach().item())
                    has_effective_reg = True
                if has_effective_reg:
                    logs["motion/effective_regularization"] = effective_reg_value
                if weight_drift_state is not None:
                    _interval = int(weight_drift_state.get("interval", 0) or 0)
                    if _interval > 0 and global_step > 0 and global_step % _interval == 0:
                        logs.update(_compute_weight_drift_logs(weight_drift_state))
                if grad_norm_state is not None:
                    _interval = int(grad_norm_state.get("interval", 0) or 0)
                    if _interval > 0 and global_step > 0 and global_step % _interval == 0:
                        logs.update(_compute_grad_norm_logs(grad_norm_state))
                if output_drift_state is not None:
                    _interval = int(output_drift_state.get("interval", 0) or 0)
                    if _interval > 0 and global_step > 0 and global_step % _interval == 0:
                        logs.update(
                            _compute_output_drift_logs(
                                output_drift_state,
                                trainer,
                                transformer,
                                args,
                                noise_scheduler,
                                accelerator,
                            )
                        )
                progress_logs: dict[str, Any] = {
                    "loss": logs.get("loss"),
                    "lr": logs.get("lr"),
                }
                if "motion/pres_weighted" in logs:
                    progress_logs["m"] = logs["motion/pres_weighted"]
                if "motion/attn_weighted" in logs:
                    progress_logs["a"] = logs["motion/attn_weighted"]
                if "ewc" in logs:
                    progress_logs["e"] = logs["ewc"]
                if "ewc/used_tensors" in logs:
                    progress_logs["eu"] = logs["ewc/used_tensors"]
                if "ewc/skipped_tensors" in logs:
                    progress_logs["es"] = logs["ewc/skipped_tensors"]
                progress_bar.set_postfix(**progress_logs)
                accelerator.log(logs, step=global_step)

                # Run validation at step intervals
                if should_validate(global_step, epoch, is_epoch_end=False):
                    optimizer_eval_fn()
                    run_validation_with_optimizer_offload(global_step, epoch)
                    optimizer_train_fn()

                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    ckpt_name = train_utils.get_step_ckpt_name(args.output_name, global_step)
                    if args.save_ema_only and ema_model is not None:
                        save_ema_model(ckpt_name, global_step, epoch + 1)
                    else:
                        save_model(
                            ckpt_name,
                            accelerator.unwrap_model(transformer),
                            global_step,
                            epoch + 1,
                        )
                        if ema_model is not None:
                            save_ema_model(ckpt_name, global_step, epoch + 1)
                    save_self_flow_state()
                    remove_step_no = train_utils.get_remove_step_no(args, global_step)
                    if remove_step_no is not None:
                        remove_model(train_utils.get_step_ckpt_name(args.output_name, remove_step_no))
                        # Also remove old EMA checkpoint if exists
                        if ema_model is not None:
                            old_ema_name = train_utils.get_step_ckpt_name(args.output_name, remove_step_no).replace(
                                ".safetensors", "_ema.safetensors"
                            )
                            remove_model(old_ema_name)
                    if args.save_state:
                        train_utils.save_and_remove_state_stepwise(
                            args, accelerator, global_step, epoch=epoch + 1, step_in_epoch=step + 1
                        )
                        save_ema_state()
                        save_self_flow_state()
                    clean_memory_on_device(accelerator.device)

                if should_sample_images(args, global_step, epoch=None):
                    run_sampling_safely(None, global_step)

                if global_step >= args.max_train_steps:
                    break
                if handle_dashboard_stop_request(global_step, epoch, step + 1):
                    return

        # Run validation at epoch end
        if should_validate(global_step, epoch, is_epoch_end=True):
            optimizer_eval_fn()
            run_validation_with_optimizer_offload(global_step, epoch + 1)
            optimizer_train_fn()

        if args.save_every_n_epochs is not None and (epoch + 1) % args.save_every_n_epochs == 0:
            ckpt_name = train_utils.get_epoch_ckpt_name(args.output_name, epoch + 1)
            if args.save_ema_only and ema_model is not None:
                # Only save EMA weights
                save_ema_model(ckpt_name, global_step, epoch + 1)
            else:
                # Save training weights
                save_model(
                    ckpt_name,
                    accelerator.unwrap_model(transformer),
                    global_step,
                    epoch + 1,
                )
                # Also save EMA if enabled
                if ema_model is not None:
                    save_ema_model(ckpt_name, global_step, epoch + 1)
            save_self_flow_state()
            remove_epoch_no = train_utils.get_remove_epoch_no(args, epoch + 1)
            if remove_epoch_no is not None:
                remove_model(train_utils.get_epoch_ckpt_name(args.output_name, remove_epoch_no))
            if args.save_state:
                train_utils.save_and_remove_state_on_epoch_end(
                    args, accelerator, epoch + 1, global_step=global_step, step_in_epoch=0
                )
                last_resume_epoch = epoch + 1
                last_resume_step_in_epoch = 0
                save_ema_state()
                save_self_flow_state()
            clean_memory_on_device(accelerator.device)

        if should_sample_images(args, global_step, epoch=epoch + 1):
            run_sampling_safely(epoch + 1, global_step)

        if global_step >= args.max_train_steps:
            break

    metadata["ss_training_finished_at"] = str(time.time())
    optimizer_eval_fn()

    # Final validation
    if val_dataloader is not None:
        accelerator.print("\nRunning final validation...")
        run_validation_with_optimizer_offload(global_step, num_train_epochs)

    if accelerator.is_main_process and (args.save_state or args.save_state_on_train_end):
        train_utils.save_state_on_train_end(
            args,
            accelerator,
            global_step=global_step,
            epoch=last_resume_epoch,
            step_in_epoch=last_resume_step_in_epoch,
        )
        save_ema_state()
        save_self_flow_state()

    if args.no_final_save:
        accelerator.print("Skipping final checkpoint save because --no_final_save is set.")
    else:
        # Save final model
        final_ckpt_name = f"{args.output_name}.safetensors"
        if args.save_ema_only and ema_model is not None:
            # Only save EMA weights as final model
            save_ema_model(final_ckpt_name, global_step, num_train_epochs)
        else:
            # Save training weights
            save_model(
                final_ckpt_name,
                accelerator.unwrap_model(transformer),
                global_step,
                num_train_epochs,
                force_sync_upload=True,
                use_memory_efficient_saving=args.mem_eff_save,
            )
            # Also save EMA if enabled
            if ema_model is not None:
                save_ema_model(final_ckpt_name, global_step, num_train_epochs)
        save_self_flow_state()
    if text_encoder is not None:
        trainer._cleanup_text_encoder(accelerator)

    if accelerator.is_main_process:
        accelerator.end_training()


if __name__ == "__main__":
    main()
