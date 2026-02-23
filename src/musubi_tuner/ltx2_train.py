#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from multiprocessing import Value
from typing import Any, Optional

import toml
import torch
from accelerate import Accelerator
from tqdm import tqdm
from safetensors.torch import save_file

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.hv_train_network import (
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_MINIMUM_KEYS,
    clean_memory_on_device,
    collator_class,
    compute_loss_weighting_for_sd3,
    prepare_accelerator,
    read_config_from_file,
    set_seed,
    setup_parser_common,
    should_sample_images,
)
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.utils import huggingface_utils, model_utils, sai_model_spec, train_utils
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file
from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer, ltx2_setup_parser

import copy
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _is_attention_geometry_param(param_name: str) -> bool:
    # Attention geometry parameters most tied to motion priors.
    return re.search(
        r"(?:^|\.)(?:attn\d+|audio_attn\d+|audio_to_video_attn|video_to_audio_attn)\.(?:to_q|to_k|q_norm|k_norm)\.",
        param_name,
    ) is not None


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
) -> torch.Tensor:
    if isinstance(tgt, torch.Tensor):
        pred = pred.to(device=tgt.device, dtype=dtype)
    else:
        pred = pred.to(dtype=dtype)
    per_elem = torch.nn.functional.mse_loss(pred, tgt, reduction="none")
    if weighting is not None:
        w = weighting
        if isinstance(w, torch.Tensor) and w.dim() != per_elem.dim():
            while w.dim() > per_elem.dim() and w.shape[-1] == 1:
                w = w.squeeze(-1)
        per_elem = per_elem * w
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


def _build_temporal_pair_mask(video_loss_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if video_loss_mask is None:
        return None
    if not isinstance(video_loss_mask, torch.Tensor):
        return None
    if video_loss_mask.dim() == 2:
        if video_loss_mask.shape[1] < 2:
            return None
        pair = video_loss_mask[:, 1:] & video_loss_mask[:, :-1]
        return pair.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    if video_loss_mask.dim() == 5:
        if video_loss_mask.shape[2] < 2:
            return None
        return video_loss_mask[:, :, 1:, :, :] & video_loss_mask[:, :, :-1, :, :]
    return None


def _extract_motion_anchor_batch(batch: dict) -> dict:
    # Keep only fields used by call_dit for text conditioning / frame rate.
    keep_keys = (
        "conditions",
        "text",
        "text_mask",
        "frame_rate",
        "ref_latents",
        "reference_latents",
        "audio_latents",
        "audio_lengths",
    )
    return {k: _clone_to_cpu(batch[k]) for k in keep_keys if k in batch}


def _fused_step_pending_grads(
    optimizer: Any,
    accelerator: Accelerator,
    max_grad_norm: float,
) -> None:
    """Run one fused-style parameter step for any pending grads.

    Used when fused backward hooks defer stepping on the first backward pass
    and no second backward pass happens.
    """
    for param_group in optimizer.param_groups:
        for parameter in param_group.get("params", []):
            if parameter is None or parameter.grad is None:
                continue
            if accelerator.sync_gradients and max_grad_norm != 0.0:
                accelerator.clip_grad_norm_(parameter, max_grad_norm)
            optimizer.step_param(parameter, param_group)
            parameter.grad = None


def _build_motion_anchor_cache(
    trainer: LTX2NetworkTrainer,
    args: argparse.Namespace,
    accelerator: Accelerator,
    transformer: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    noise_scheduler: Any,
    attention_modules: Optional[list[tuple[str, torch.nn.Module]]] = None,
) -> list[dict[str, Any]]:
    cache_size = int(getattr(args, "motion_preservation_anchor_cache_size", 0) or 0)
    if cache_size <= 0:
        return []

    use_attn_pres = bool(getattr(args, "motion_attention_preservation", False) and attention_modules)
    max_queries = int(getattr(args, "motion_attention_preservation_queries", 32) or 32)
    max_keys = int(getattr(args, "motion_attention_preservation_keys", 64) or 64)

    entries: list[dict[str, Any]] = []
    max_attempts = max(cache_size * 4, cache_size)
    attempt = 0
    single_frame_dataset_anchor_count = 0
    original_first_frame_p = float(getattr(args, "ltx2_first_frame_conditioning_p", 0.0))
    was_training = transformer.training

    logger.info(
        "Building motion anchor cache from base model outputs: source=dataset size=%d (replay on real dataset conditioning)",
        cache_size,
    )

    transformer.eval()
    setattr(args, "ltx2_first_frame_conditioning_p", 0.0)

    try:
        with torch.no_grad():
            for batch in train_dataloader:
                if len(entries) >= cache_size or attempt >= max_attempts:
                    break
                attempt += 1

                batch = _normalize_ltx2_batch_for_call_dit(batch)
                latents = batch.get("latents")
                if isinstance(latents, dict):
                    latents = latents.get("latents")
                if not isinstance(latents, torch.Tensor):
                    continue
                if latents.dim() != 5:
                    continue

                latents_tensor = trainer.scale_shift_latents(latents)
                anchor_latents = latents_tensor.clone()
                if int(anchor_latents.shape[2]) <= 1:
                    single_frame_dataset_anchor_count += 1
                anchor_noise = torch.randn_like(anchor_latents)

                batch_timesteps = batch.get("timesteps")
                if not isinstance(batch_timesteps, torch.Tensor):
                    continue

                anchor_noisy_input, anchor_model_timesteps = trainer.get_noisy_model_input_and_timesteps(
                    args,
                    anchor_noise,
                    anchor_latents,
                    batch_timesteps,
                    noise_scheduler,
                    accelerator.device,
                    trainer.dit_dtype,
                )

                teacher_attn_maps = None
                if use_attn_pres:
                    with _AttentionMapRecorder(
                        attention_modules or [],
                        max_queries=max_queries,
                        max_keys=max_keys,
                        capture_grad=False,
                    ) as attn_recorder:
                        teacher_pred, _ = trainer.call_dit(
                            args,
                            accelerator,
                            transformer,
                            anchor_latents,
                            batch,
                            anchor_noise,
                            anchor_noisy_input,
                            anchor_model_timesteps,
                            trainer.dit_dtype,
                        )
                    if attn_recorder.maps:
                        teacher_attn_maps = {
                            k: v.detach().to(dtype=torch.float16).cpu()
                            for k, v in attn_recorder.maps.items()
                        }
                else:
                    teacher_pred, _ = trainer.call_dit(
                        args,
                        accelerator,
                        transformer,
                        anchor_latents,
                        batch,
                        anchor_noise,
                        anchor_noisy_input,
                        anchor_model_timesteps,
                        trainer.dit_dtype,
                    )

                if not isinstance(teacher_pred, dict):
                    continue
                if teacher_pred.get("_skip_step"):
                    continue

                entries.append(
                    {
                        "anchor_latents": _clone_to_cpu(anchor_latents),
                        "anchor_noise": _clone_to_cpu(anchor_noise),
                        "anchor_noisy_input": _clone_to_cpu(anchor_noisy_input),
                        "anchor_model_timesteps": _clone_to_cpu(anchor_model_timesteps),
                        "anchor_batch": _extract_motion_anchor_batch(batch),
                        "teacher_video_pred": _clone_to_cpu(teacher_pred["video_pred"]),
                        "teacher_attention_maps": teacher_attn_maps,
                    }
                )
    finally:
        setattr(args, "ltx2_first_frame_conditioning_p", original_first_frame_p)
        if was_training:
            transformer.train()

    if len(entries) < cache_size:
        logger.warning(
            "Built %d/%d motion anchors (max_attempts=%d).",
            len(entries),
            cache_size,
            max_attempts,
        )
    else:
        logger.info("Built %d motion anchors.", len(entries))
    if getattr(args, "motion_preservation_mode", "temporal") == "temporal" and single_frame_dataset_anchor_count > 0:
        logger.info(
            "Dataset anchor replay: %d/%d anchors are single-frame; temporal mode falls back to full-output replay for those anchors.",
            single_frame_dataset_anchor_count,
            len(entries),
        )

    return entries


def _extract_attn1_block_index(module_name: str) -> Optional[int]:
    match = re.search(r"(?:^|\.)(?:model\.)?transformer_blocks\.(\d+)\.attn1$", module_name)
    if match is None:
        return None
    return int(match.group(1))


def _sample_sequence_indices(length: int, count: int, *, device: torch.device) -> torch.Tensor:
    if length <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    if count <= 0 or count >= length:
        return torch.arange(length, device=device, dtype=torch.long)
    idx = torch.linspace(0, length - 1, steps=count, device=device)
    idx = idx.round().to(torch.long)
    return torch.unique(idx, sorted=True)


def _collect_motion_attention_modules(
    transformer: torch.nn.Module,
    block_spec: Optional[str],
) -> list[tuple[str, torch.nn.Module]]:
    from musubi_tuner.ltx_2.model.transformer.attention import Attention

    block_filter = _parse_block_index_spec(block_spec)
    apply_filter = len(block_filter) > 0
    out: list[tuple[str, torch.nn.Module]] = []
    for module_name, module in transformer.named_modules():
        if not isinstance(module, Attention):
            continue
        block_index = _extract_attn1_block_index(module_name)
        if block_index is None:
            continue
        if apply_filter and block_index not in block_filter:
            continue
        out.append((module_name, module))
    return out


class _AttentionMapRecorder:
    def __init__(
        self,
        modules: list[tuple[str, torch.nn.Module]],
        *,
        max_queries: int,
        max_keys: int,
        capture_grad: bool,
    ) -> None:
        self.modules = modules
        self.max_queries = max(1, int(max_queries))
        self.max_keys = max(1, int(max_keys))
        self.capture_grad = capture_grad
        self.maps: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []

    def _record(self, name: str, module: torch.nn.Module, args: tuple[Any, ...], kwargs: Optional[dict[str, Any]]) -> None:
        if not args:
            return
        x = args[0]
        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            return

        context = kwargs.get("context") if kwargs else None
        pe = kwargs.get("pe") if kwargs else None
        k_pe = kwargs.get("k_pe") if kwargs else None

        if context is None and len(args) > 1 and isinstance(args[1], torch.Tensor):
            context = args[1]
        if pe is None and len(args) > 3 and isinstance(args[3], torch.Tensor):
            pe = args[3]
        if k_pe is None and len(args) > 4 and isinstance(args[4], torch.Tensor):
            k_pe = args[4]

        if context is None:
            context = x
        if not isinstance(context, torch.Tensor) or context.dim() != 3:
            return

        q = module.q_norm(module.to_q(x))
        k = module.k_norm(module.to_k(context))

        if isinstance(pe, torch.Tensor):
            from musubi_tuner.ltx_2.model.transformer.rope import apply_rotary_emb

            q = apply_rotary_emb(q, pe, module.rope_type)
            k = apply_rotary_emb(k, pe if not isinstance(k_pe, torch.Tensor) else k_pe, module.rope_type)

        bsz = q.shape[0]
        q = q.view(bsz, -1, module.heads, module.dim_head).transpose(1, 2)
        k = k.view(bsz, -1, module.heads, module.dim_head).transpose(1, 2)
        if q.shape[2] == 0 or k.shape[2] == 0:
            return

        q_idx = _sample_sequence_indices(q.shape[2], self.max_queries, device=q.device)
        k_idx = _sample_sequence_indices(k.shape[2], self.max_keys, device=k.device)
        if q_idx.numel() == 0 or k_idx.numel() == 0:
            return

        q_sample = q[:, :, q_idx, :].to(torch.float32)
        k_sample = k[:, :, k_idx, :].to(torch.float32)
        logits = torch.matmul(q_sample, k_sample.transpose(-1, -2)) / math.sqrt(float(module.dim_head))
        attn = torch.softmax(logits, dim=-1).mean(dim=1)  # [B, Q, K], average over heads

        if not self.capture_grad:
            attn = attn.detach()
        self.maps[name] = attn

    def _build_hook(self, name: str):
        def _hook(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            self._record(name, module, args, kwargs)

        return _hook

    def _build_legacy_hook(self, name: str):
        def _hook(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            self._record(name, module, args, None)

        return _hook

    def __enter__(self) -> "_AttentionMapRecorder":
        self.maps = {}
        self._handles = []
        for name, module in self.modules:
            try:
                handle = module.register_forward_pre_hook(self._build_hook(name), with_kwargs=True)
            except TypeError:
                handle = module.register_forward_pre_hook(self._build_legacy_hook(name))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        return False


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
    frozen_param_count = 0
    trainable_param_count = 0
    frozen_attn_geometry_count = 0
    trainable_attn_geometry_count = 0
    trainable_by_block: dict[str, int] = {}

    for name, param in transformer.named_parameters():
        if not param.requires_grad:
            continue

        block_index = _extract_transformer_block_index(name)
        if block_index is not None and block_index in frozen_blocks:
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
                param.requires_grad_(False)
                frozen_param_count += 1
                frozen_attn_geometry_count += 1
                continue
            scale *= float(attn_geometry_lr_scale)

        if scale <= 0.0:
            param.requires_grad_(False)
            frozen_param_count += 1
            if is_attn_geometry:
                frozen_attn_geometry_count += 1
            continue

        grouped_params.setdefault(scale, []).append(param)
        grouped_names.setdefault(scale, []).append(name)
        trainable_param_count += 1
        if is_attn_geometry:
            trainable_attn_geometry_count += 1

    if trainable_param_count == 0:
        raise ValueError("No trainable parameters remain after freeze/lr-scale settings.")

    # Keep order deterministic by scale value.
    scales = sorted(grouped_params.keys())
    param_groups = [{"params": grouped_params[s], "lr": base_lr * s} for s in scales]
    param_name_groups = [grouped_names[s] for s in scales]

    stats = {
        "frozen_param_count": frozen_param_count,
        "trainable_param_count": trainable_param_count,
        "num_lr_groups": len(scales),
        "lr_scales": scales,
        "frozen_blocks": sorted(frozen_blocks),
        "block_lr_rules": block_lr_rules,
        "trainable_by_block": trainable_by_block,
        "frozen_attn_geometry_count": frozen_attn_geometry_count,
        "trainable_attn_geometry_count": trainable_attn_geometry_count,
    }
    return param_groups, param_name_groups, stats


def ltx2_finetune_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--fused_backward_pass",
        action="store_true",
        help="Use fused backward pass for Adafactor optimizer",
    )
    parser.add_argument(
        "--mem_eff_save",
        action="store_true",
        help=(
            "Enable memory efficient saving (saving states requires normal saving, so it takes same amount of memory "
            "even with this option enabled)"
        ),
    )
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
    # Note: --validate_every_n_steps and --validate_every_n_epochs are already defined in setup_parser_common()
    parser.add_argument(
        "--num_validation_batches",
        type=int,
        default=None,
        help="Number of validation batches to use (None = all)",
    )
    parser.add_argument(
        "--motion_preservation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Full-FT motion preservation via base-model output replay on real dataset conditioning. "
            "Designed for image-only training to keep existing motion behavior."
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
        "--motion_preservation_anchor_cache_size",
        type=int,
        default=32,
        help="Number of base-model rehearsal anchors cached at training start.",
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
        "--motion_attention_preservation_blocks",
        type=str,
        default=None,
        help="Optional comma/range block filter for attention-map preservation, e.g. 12-23.",
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
    return parser


def main() -> None:
    parser = setup_parser_common()
    parser = ltx2_setup_parser(parser)
    parser = ltx2_finetune_setup_parser(parser)
    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    trainer = LTX2NetworkTrainer()

    if args.dataset_config is None:
        raise ValueError("dataset_config is required / dataset_configが必要です")
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

    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    short_map = {"v": "video", "a": "audio", "va": "av"}
    if getattr(args, "ltx_mode", None) in short_map:
        args.ltx_mode = short_map[args.ltx_mode]

    trainer.handle_model_specific_args(args)
    if getattr(args, "ltx_mode", "video") == "av" and not getattr(args, "av_use_video_prompt_embeds", False):
        logger.info(
            "Enabling av_use_video_prompt_embeds for AV mode compatibility when batches have no audio latents."
        )
        args.av_use_video_prompt_embeds = True

    if args.motion_preservation and getattr(args, "ltx_mode", "video") != "video":
        logger.warning("motion_preservation is only supported for video mode in full fine-tune. Disabling it.")
        args.motion_preservation = False
    if args.motion_preservation and int(args.motion_preservation_interval) <= 0:
        raise ValueError("motion_preservation_interval must be >= 1")
    if args.motion_preservation and args.motion_preservation_probability is not None:
        if float(args.motion_preservation_probability) < 0.0 or float(args.motion_preservation_probability) > 1.0:
            raise ValueError("motion_preservation_probability must be in [0, 1]")
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
    if int(getattr(args, "freeze_early_blocks", 0) or 0) < 0:
        raise ValueError("freeze_early_blocks must be >= 0")
    if float(getattr(args, "non_block_lr_scale", 1.0) or 0.0) < 0.0:
        raise ValueError("non_block_lr_scale must be >= 0")
    if float(getattr(args, "attn_geometry_lr_scale", 1.0) or 0.0) < 0.0:
        raise ValueError("attn_geometry_lr_scale must be >= 0")

    if args.seed is None:
        args.seed = random.randint(0, 2**32)
    set_seed(args.seed)
    session_id = random.randint(0, 2**32)
    training_started_at = time.time()

    accelerator = prepare_accelerator(args)
    if args.mixed_precision is None:
        args.mixed_precision = accelerator.mixed_precision

    # sample prompts (optional)
    sample_parameters = None
    vae = None
    if args.sample_prompts or getattr(args, "precache_sample_prompts", False) or getattr(args, "use_precached_sample_prompts", False):
        sample_prompt_path = args.sample_prompts or ""
        sample_parameters = trainer.process_sample_prompts(args, accelerator, sample_prompt_path)
        vae = trainer.load_vae(args, vae_dtype=model_utils.str_to_dtype(args.vae_dtype), vae_path=args.vae)
        vae.requires_grad_(False)
        vae.eval()

    # datasets
    current_epoch = Value("i", 0)
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=trainer.architecture)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
        blueprint.dataset_group,
        training=True,
        num_timestep_buckets=args.num_timestep_buckets,
        shared_epoch=current_epoch,
    )

    if train_dataset_group.num_train_items == 0:
        raise ValueError(
            "No training items found in the dataset. Please ensure that the latent/Text Encoder cache has been created beforehand."
            " / データセットに学習データがありません。latent/Text Encoderキャッシュを事前に作成したか確認してください"
        )

    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = collator_class(current_epoch, ds_for_collator)
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # Validation dataset (optional)
    val_dataloader = None
    if args.validation_dataset_config is not None:
        logger.info("Loading validation dataset from: %s", args.validation_dataset_config)
        val_user_config = config_utils.load_user_config(args.validation_dataset_config)
        val_blueprint = blueprint_generator.generate(val_user_config, args, architecture=trainer.architecture)
        val_dataset_group = config_utils.generate_dataset_group_by_blueprint(
            val_blueprint.dataset_group,
            training=False,  # validation mode
            num_timestep_buckets=args.num_timestep_buckets,
            shared_epoch=current_epoch,
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

    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )

    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # model
    blocks_to_swap = int(getattr(args, "blocks_to_swap", 0) or 0)
    trainer.blocks_to_swap = blocks_to_swap
    loading_device = "cpu" if blocks_to_swap > 0 else accelerator.device

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

    # Clean up memory after model loading
    clean_memory_on_device(accelerator.device)

    if blocks_to_swap > 0:
        logger.info(
            "enable swap %s blocks to CPU from device: %s", blocks_to_swap, accelerator.device
        )
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

    # optimizer
    params_to_optimize, param_names, ft_group_stats = _build_full_ft_param_groups(
        transformer,
        args.learning_rate,
        freeze_early_blocks=int(getattr(args, "freeze_early_blocks", 0) or 0),
        freeze_block_indices_spec=getattr(args, "freeze_block_indices", None),
        block_lr_scales_spec=getattr(args, "block_lr_scales", None),
        non_block_lr_scale=float(getattr(args, "non_block_lr_scale", 1.0) or 0.0),
        attn_geometry_lr_scale=float(getattr(args, "attn_geometry_lr_scale", 1.0) or 0.0),
        freeze_attn_geometry=bool(getattr(args, "freeze_attn_geometry", False)),
    )
    logger.info(
        "Full-FT parameter groups: trainable=%d frozen=%d groups=%d scales=%s",
        ft_group_stats["trainable_param_count"],
        ft_group_stats["frozen_param_count"],
        ft_group_stats["num_lr_groups"],
        ft_group_stats["lr_scales"],
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

    optimizer_name, optimizer_args, optimizer, optimizer_train_fn, optimizer_eval_fn = trainer.get_optimizer(
        args, params_to_optimize
    )

    # lr scheduler
    lr_scheduler = trainer.get_lr_scheduler(args, optimizer, accelerator.num_processes)

    # prepare accelerator
    if blocks_to_swap > 0:
        transformer = accelerator.prepare(transformer, device_placement=[not blocks_to_swap > 0])
        accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
    else:
        transformer = accelerator.prepare(transformer)

    if args.compile:
        transformer = trainer.compile_transformer(args, transformer)
        transformer.__dict__["_orig_mod"] = transformer

    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    # Prepare validation dataloader if exists
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    trainer.resume_from_local_or_hf_if_specified(accelerator, args)

    # Initialize EMA after model is prepared
    ema_model = None
    ema_state_path = os.path.join(args.output_dir, "ema_state.pt") if args.output_dir else None
    if args.use_ema:
        ema_device = torch.device("cpu") if args.ema_cpu_offload else None
        logger.info(
            "Initializing EMA with decay=%.6f, update_after_step=%d, update_every=%d, device=%s",
            args.ema_decay, args.ema_update_after_step, args.ema_update_every,
            "cpu" if args.ema_cpu_offload else "same as model"
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
    if args.fused_backward_pass:
        import musubi_tuner.modules.adafactor_fused as adafactor_fused

        adafactor_fused.patch_adafactor_fused(optimizer)
        fused_step_state = {"defer_step": False}

        for param_group, param_name_group in zip(optimizer.param_groups, param_names):
            for parameter, param_name in zip(param_group["params"], param_name_group):
                if parameter.requires_grad:

                    def create_grad_hook(p_name, p_group):
                        def grad_hook(tensor: torch.Tensor):
                            if fused_step_state is not None and fused_step_state.get("defer_step", False):
                                return
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                            optimizer.step_param(tensor, p_group)
                            tensor.grad = None

                        return grad_hook

                    parameter.register_post_accumulate_grad_hook(create_grad_hook(param_name, param_group))

    # scheduler
    noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")

    num_train_items = train_dataset_group.num_train_items
    metadata = {
        "ss_session_id": session_id,
        "ss_training_started_at": training_started_at,
        "ss_output_name": args.output_name,
        "ss_learning_rate": args.learning_rate,
        "ss_num_train_items": num_train_items,
        "ss_num_batches_per_epoch": len(train_dataloader),
        "ss_num_epochs": None,
        "ss_gradient_checkpointing": args.gradient_checkpointing,
        "ss_gradient_checkpointing_cpu_offload": args.gradient_checkpointing_cpu_offload,
        "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
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
        "ss_weighting_scheme": args.weighting_scheme,
        "ss_logit_mean": args.logit_mean,
        "ss_logit_std": args.logit_std,
        "ss_mode_scale": args.mode_scale,
        "ss_guidance_scale": args.guidance_scale,
        "ss_timestep_sampling": args.timestep_sampling,
        "ss_sigmoid_scale": args.sigmoid_scale,
        "ss_discrete_flow_shift": args.discrete_flow_shift,
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
        "ss_motion_preservation_interval": getattr(args, "motion_preservation_interval", 1),
        "ss_motion_preservation_probability": getattr(args, "motion_preservation_probability", None),
        "ss_motion_preservation_separate_backward": bool(getattr(args, "motion_preservation_separate_backward", False)),
        "ss_motion_preservation_fused_defer_step": bool(getattr(args, "motion_preservation_fused_defer_step", False)),
        "ss_motion_attention_preservation": bool(getattr(args, "motion_attention_preservation", False)),
        "ss_motion_attention_preservation_weight": getattr(args, "motion_attention_preservation_weight", 0.0),
        "ss_motion_attention_preservation_loss": getattr(args, "motion_attention_preservation_loss", "kl"),
        "ss_motion_attention_preservation_queries": getattr(args, "motion_attention_preservation_queries", 0),
        "ss_motion_attention_preservation_keys": getattr(args, "motion_attention_preservation_keys", 0),
        "ss_motion_attention_preservation_blocks": getattr(args, "motion_attention_preservation_blocks", None),
        "ss_freeze_early_blocks": getattr(args, "freeze_early_blocks", 0),
        "ss_freeze_block_indices": getattr(args, "freeze_block_indices", None),
        "ss_block_lr_scales": getattr(args, "block_lr_scales", None),
        "ss_non_block_lr_scale": getattr(args, "non_block_lr_scale", 1.0),
        "ss_attn_geometry_lr_scale": getattr(args, "attn_geometry_lr_scale", 1.0),
        "ss_freeze_attn_geometry": bool(getattr(args, "freeze_attn_geometry", False)),
        "ss_full_ft_lr_group_scales": ft_group_stats.get("lr_scales"),
        "ss_full_ft_frozen_blocks_applied": ft_group_stats.get("frozen_blocks"),    }

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
        }
        for i, scale in enumerate(ft_group_stats.get("lr_scales", [])):
            setup_logs[f"setup/lr_scale/group_{i}"] = float(scale)
        for i, group in enumerate(params_to_optimize):
            params = list(group.get("params", []))
            setup_logs[f"setup/group_{i}_param_count"] = float(len(params))
            setup_logs[f"setup/group_{i}_param_numel"] = float(sum(int(p.numel()) for p in params))
        accelerator.log(setup_logs, step=0)

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")

    epoch_to_start = 0
    global_step = 0
    loss_recorder = train_utils.LossRecorder()
    del train_dataset_group

    def save_model(
        ckpt_name: str, unwrapped_model, steps, epoch_no, force_sync_upload=False, use_memory_efficient_saving=False
    ):
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
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
            args.metadata_reso,
            title=title,
            author=args.metadata_author,
            description=args.metadata_description,
            license=args.metadata_license,
            tags=args.metadata_tags,
            timesteps=md_timesteps,
            custom_arch=args.metadata_arch,
        )
        metadata_to_save.update(sai_metadata)

        save_model_ref = getattr(unwrapped_model, "_orig_mod", None) or unwrapped_model
        state_dict = save_model_ref.state_dict()
        if use_memory_efficient_saving or args.mem_eff_save:
            mem_eff_save_file(state_dict, ckpt_file, metadata_to_save)
        else:
            save_file(state_dict, ckpt_file, metadata_to_save)

        if args.huggingface_repo_id is not None:
            huggingface_utils.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

    def remove_model(old_ckpt_name: str) -> None:
        old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
        if os.path.exists(old_ckpt_file):
            accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
            os.remove(old_ckpt_file)

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

        ema_metadata = metadata.copy()
        ema_metadata["ss_is_ema"] = "True"
        ema_metadata["ss_ema_decay"] = str(args.ema_decay)
        ema_metadata["ss_steps"] = str(steps)
        ema_metadata["ss_epoch"] = str(epoch_no)

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

        val_losses = []
        val_video_losses = []
        val_audio_losses = []
        num_batches = 0
        max_batches = args.num_validation_batches

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
                        continue

                    video_pred = out["video_pred"]
                    video_target = out["video_target"]
                    video_loss_mask = out.get("video_loss_mask")
                    video_loss = _masked_mse(
                        video_pred, video_target, video_loss_mask,
                        weighting=weighting, dtype=trainer.dit_dtype
                    )
                    val_video_losses.append(video_loss.item())

                    audio_pred = out.get("audio_pred")
                    audio_target = out.get("audio_target")
                    if audio_pred is not None and audio_target is not None:
                        audio_loss_mask = out.get("audio_loss_mask")
                        audio_loss = _masked_mse(
                            audio_pred, audio_target, audio_loss_mask,
                            weighting=weighting, dtype=trainer.dit_dtype
                        )
                        val_audio_losses.append(audio_loss.item())
                        val_losses.append(video_loss.item() * args.video_loss_weight + audio_loss.item() * args.audio_loss_weight)
                    else:
                        val_losses.append(video_loss.item())
                else:
                    if isinstance(target, torch.Tensor):
                        model_pred = model_pred.to(device=target.device, dtype=trainer.dit_dtype)
                    loss = torch.nn.functional.mse_loss(model_pred, target)
                    val_losses.append(loss.item())

                num_batches += 1

        # Restore original weights if EMA was applied
        if original_params is not None:
            ema_model.restore(accelerator.unwrap_model(transformer), original_params)

        transformer.train()

        # Compute average metrics
        val_metrics = {}
        if val_losses:
            val_metrics["val_loss"] = sum(val_losses) / len(val_losses)
        if val_video_losses:
            val_metrics["val_video_loss"] = sum(val_video_losses) / len(val_video_losses)
        if val_audio_losses:
            val_metrics["val_audio_loss"] = sum(val_audio_losses) / len(val_audio_losses)

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

    # For --sample_at_first
    if should_sample_images(args, global_step, epoch=0):
        optimizer_eval_fn()
        trainer.sample_images(
            accelerator,
            args,
            0,
            global_step,
            accelerator.device,
            vae,
            transformer,
            sample_parameters,
        )
        optimizer_train_fn()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    metadata["ss_num_epochs"] = str(num_train_epochs)

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

    optimizer_train_fn()

    motion_anchor_cache: list[dict[str, Any]] = []
    motion_micro_step = 0
    motion_attention_modules: list[tuple[str, torch.nn.Module]] = []
    if args.motion_attention_preservation:
        motion_attention_modules = _collect_motion_attention_modules(
            transformer,
            getattr(args, "motion_attention_preservation_blocks", None),
        )
        if not motion_attention_modules:
            logger.warning(
                "motion_attention_preservation requested but no matching attn1 modules were found; disabling it."
            )
            args.motion_attention_preservation = False
        else:
            logger.info(
                "Motion attention preservation enabled on %d attn1 modules (queries=%d keys=%d, loss=%s)",
                len(motion_attention_modules),
                int(getattr(args, "motion_attention_preservation_queries", 32) or 32),
                int(getattr(args, "motion_attention_preservation_keys", 64) or 64),
                getattr(args, "motion_attention_preservation_loss", "kl"),
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
        )
        if not motion_anchor_cache:
            logger.warning("motion_preservation requested but no anchors were built; disabling.")
            args.motion_preservation = False

    for epoch in range(epoch_to_start, num_train_epochs):
        current_epoch.value = epoch + 1
        metadata["ss_epoch"] = str(epoch + 1)
        transformer.train()
        for step, batch in enumerate(train_dataloader):
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
                        continue

                    video_pred = out["video_pred"]
                    video_target = out["video_target"]
                    video_loss_mask = out.get("video_loss_mask")
                    video_loss = _masked_mse(
                        video_pred,
                        video_target,
                        video_loss_mask,
                        weighting=weighting,
                        dtype=trainer.dit_dtype,
                    )
                    video_weight = float(out.get("video_loss_weight", 1.0))
                    loss = video_loss * video_weight

                    audio_pred = out.get("audio_pred")
                    audio_target = out.get("audio_target")
                    audio_loss_mask = out.get("audio_loss_mask")
                    if audio_pred is not None and audio_target is not None:
                        audio_loss = _masked_mse(
                            audio_pred,
                            audio_target,
                            audio_loss_mask,
                            weighting=weighting,
                            dtype=trainer.dit_dtype,
                        )
                        audio_weight = float(out.get("audio_loss_weight", 1.0))
                        loss = loss + audio_loss * audio_weight
                else:
                    if isinstance(target, torch.Tensor):
                        model_pred = model_pred.to(device=target.device, dtype=trainer.dit_dtype)
                    else:
                        model_pred = model_pred.to(dtype=trainer.dit_dtype)
                    loss = torch.nn.functional.mse_loss(model_pred, target, reduction="none")
                    if weighting is not None:
                        w = weighting
                        if isinstance(w, torch.Tensor) and w.dim() != loss.dim():
                            while w.dim() > loss.dim() and w.shape[-1] == 1:
                                w = w.squeeze(-1)
                        loss = loss * w
                    loss = loss.mean()

                motion_pres_loss = None
                attn_pres_loss = None
                motion_total_loss = None
                motion_preservation_prob = getattr(args, "motion_preservation_probability", None)
                if motion_preservation_prob is None:
                    should_apply_motion_replay = (motion_micro_step % int(args.motion_preservation_interval) == 0)
                else:
                    should_apply_motion_replay = random.random() < float(motion_preservation_prob)
                separate_motion_backward = bool(getattr(args, "motion_preservation_separate_backward", False))
                fused_defer_motion_step = bool(
                    args.fused_backward_pass
                    and bool(getattr(args, "motion_preservation_fused_defer_step", False))
                    and separate_motion_backward
                    and args.motion_preservation
                    and motion_anchor_cache
                    and should_apply_motion_replay
                )
                if fused_step_state is not None:
                    fused_step_state["defer_step"] = fused_defer_motion_step
                if separate_motion_backward:
                    # Backprop task loss first so replay graph does not overlap it in memory.
                    accelerator.backward(loss)
                    total_loss_for_logging = loss.detach()
                if (
                    args.motion_preservation
                    and motion_anchor_cache
                    and should_apply_motion_replay
                ):
                    anchor = random.choice(motion_anchor_cache)
                    anchor_latents = _move_to_device(
                        anchor["anchor_latents"], accelerator.device, dtype=trainer.dit_dtype
                    )
                    anchor_noise = _move_to_device(
                        anchor["anchor_noise"], accelerator.device, dtype=trainer.dit_dtype
                    )
                    anchor_noisy_input = _move_to_device(
                        anchor["anchor_noisy_input"], accelerator.device, dtype=trainer.dit_dtype
                    )
                    anchor_model_timesteps = _move_to_device(
                        anchor["anchor_model_timesteps"], accelerator.device
                    )
                    anchor_batch = _move_to_device(
                        anchor["anchor_batch"], accelerator.device, dtype=trainer.dit_dtype
                    )
                    teacher_video_pred = _move_to_device(
                        anchor["teacher_video_pred"], accelerator.device, dtype=trainer.dit_dtype
                    )

                    original_first_frame_p = float(getattr(args, "ltx2_first_frame_conditioning_p", 0.0))
                    setattr(args, "ltx2_first_frame_conditioning_p", 0.0)
                    try:
                        if args.motion_attention_preservation and motion_attention_modules:
                            with _AttentionMapRecorder(
                                motion_attention_modules,
                                max_queries=int(getattr(args, "motion_attention_preservation_queries", 32) or 32),
                                max_keys=int(getattr(args, "motion_attention_preservation_keys", 64) or 64),
                                capture_grad=True,
                            ) as attn_recorder:
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
                            student_attn_maps = attn_recorder.maps
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
                        if (
                            args.motion_preservation_mode == "temporal"
                            and student_video_pred.dim() == 5
                            and teacher_video_pred.dim() == 5
                            and student_video_pred.shape[2] > 1
                        ):
                            student_delta = student_video_pred[:, :, 1:, :, :] - student_video_pred[:, :, :-1, :, :]
                            teacher_delta = teacher_video_pred[:, :, 1:, :, :] - teacher_video_pred[:, :, :-1, :, :]
                            pair_mask = _build_temporal_pair_mask(motion_pred.get("video_loss_mask"))
                            motion_pres_loss = _masked_mse(
                                student_delta,
                                teacher_delta,
                                pair_mask,
                                weighting=None,
                                dtype=trainer.dit_dtype,
                            )
                        else:
                            motion_pres_loss = _masked_mse(
                                student_video_pred,
                                teacher_video_pred,
                                motion_pred.get("video_loss_mask"),
                                weighting=None,
                                dtype=trainer.dit_dtype,
                            )
                        motion_pres_loss = motion_pres_loss * float(args.motion_preservation_multiplier)
                        motion_total_loss = motion_pres_loss

                        teacher_attn_maps = anchor.get("teacher_attention_maps")
                        if (
                            args.motion_attention_preservation
                            and isinstance(teacher_attn_maps, dict)
                            and student_attn_maps
                        ):
                            teacher_attn_maps = _move_to_device(
                                teacher_attn_maps, accelerator.device, dtype=torch.float32
                            )
                            per_block_losses: list[torch.Tensor] = []
                            for module_name, student_map in student_attn_maps.items():
                                teacher_map = teacher_attn_maps.get(module_name)
                                if not isinstance(teacher_map, torch.Tensor):
                                    continue
                                if teacher_map.shape != student_map.shape:
                                    continue

                                student_dist = student_map.to(torch.float32)
                                teacher_dist = teacher_map.to(device=student_dist.device, dtype=torch.float32)
                                student_dist = student_dist.clamp_min(1e-6)
                                teacher_dist = teacher_dist.clamp_min(1e-6)
                                student_dist = student_dist / student_dist.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                                teacher_dist = teacher_dist / teacher_dist.sum(dim=-1, keepdim=True).clamp_min(1e-6)

                                if getattr(args, "motion_attention_preservation_loss", "kl") == "kl":
                                    block_loss = torch.nn.functional.kl_div(
                                        student_dist.log(),
                                        teacher_dist,
                                        reduction="batchmean",
                                    )
                                else:
                                    block_loss = torch.nn.functional.mse_loss(student_dist, teacher_dist)
                                per_block_losses.append(block_loss)

                            if per_block_losses:
                                attn_pres_loss = (
                                    torch.stack(per_block_losses).mean()
                                    * float(getattr(args, "motion_attention_preservation_weight", 0.0))
                                )
                                motion_total_loss = motion_total_loss + attn_pres_loss

                if separate_motion_backward:
                    if motion_total_loss is not None:
                        if fused_step_state is not None:
                            fused_step_state["defer_step"] = False
                        accelerator.backward(motion_total_loss)
                        total_loss_for_logging = total_loss_for_logging + motion_total_loss.detach()
                    elif fused_defer_motion_step:
                        if fused_step_state is not None:
                            fused_step_state["defer_step"] = False
                        _fused_step_pending_grads(optimizer, accelerator, args.max_grad_norm)
                    loss_for_step = total_loss_for_logging
                else:
                    if motion_total_loss is not None:
                        loss = loss + motion_total_loss
                    accelerator.backward(loss)
                    loss_for_step = loss.detach()
                if not args.fused_backward_pass:
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                current_loss = loss_for_step.item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)

                # Update EMA weights
                if ema_model is not None:
                    ema_model.update(accelerator.unwrap_model(transformer))

                # Update progress bar with current metrics
                current_lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, "get_last_lr") else args.learning_rate
                logs = {"loss": current_loss, "lr": current_lr}
                lr_scales = ft_group_stats.get("lr_scales", [])
                for i, param_group in enumerate(optimizer.param_groups):
                    lr_value = param_group.get("lr", current_lr)
                    logs[f"lr/group_{i}"] = float(lr_value)
                    if i < len(lr_scales):
                        logs[f"lr_scale/group_{i}"] = float(lr_scales[i])
                if dict_output:
                    if "video_pred" in out:
                        logs["v_loss"] = video_loss.item()
                    if audio_pred is not None:
                        logs["a_loss"] = audio_loss.item()
                if motion_pres_loss is not None:
                    logs["motion_pres"] = motion_pres_loss.detach().item()
                if attn_pres_loss is not None:
                    logs["attn_pres"] = attn_pres_loss.detach().item()
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # Run validation at step intervals
                if should_validate(global_step, epoch, is_epoch_end=False):
                    optimizer_eval_fn()
                    run_validation(global_step, epoch)
                    optimizer_train_fn()

                if should_sample_images(args, global_step, epoch=None):
                    optimizer_eval_fn()
                    trainer.sample_images(
                        accelerator,
                        args,
                        None,
                        global_step,
                        accelerator.device,
                        vae,
                        transformer,
                        sample_parameters,
                    )
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
                    remove_step_no = train_utils.get_remove_step_no(args, global_step)
                    if remove_step_no is not None:
                        remove_model(train_utils.get_step_ckpt_name(args.output_name, remove_step_no))
                        # Also remove old EMA checkpoint if exists
                        if ema_model is not None:
                            old_ema_name = train_utils.get_step_ckpt_name(args.output_name, remove_step_no).replace(".safetensors", "_ema.safetensors")
                            remove_model(old_ema_name)
                    if args.save_state:
                        train_utils.save_and_remove_state_stepwise(args, accelerator, global_step)
                        save_ema_state()

                if global_step >= args.max_train_steps:
                    break

        # Run validation at epoch end
        if should_validate(global_step, epoch, is_epoch_end=True):
            optimizer_eval_fn()
            run_validation(global_step, epoch + 1)
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
            remove_epoch_no = train_utils.get_remove_epoch_no(args, epoch + 1)
            if remove_epoch_no is not None:
                remove_model(train_utils.get_epoch_ckpt_name(args.output_name, remove_epoch_no))
            if args.save_state:
                train_utils.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)
                save_ema_state()

        if should_sample_images(args, global_step, epoch=epoch + 1):
            optimizer_eval_fn()
            trainer.sample_images(
                accelerator,
                args,
                epoch + 1,
                global_step,
                accelerator.device,
                vae,
                transformer,
                sample_parameters,
            )
            optimizer_train_fn()

        if global_step >= args.max_train_steps:
            break

    metadata["ss_training_finished_at"] = str(time.time())
    optimizer_eval_fn()

    # Final validation
    if val_dataloader is not None:
        accelerator.print("\nRunning final validation...")
        run_validation(global_step, num_train_epochs)

    if accelerator.is_main_process and (args.save_state or args.save_state_on_train_end):
        train_utils.save_state_on_train_end(args, accelerator)
        save_ema_state()

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

    if accelerator.is_main_process:
        accelerator.end_training()


if __name__ == "__main__":
    main()
