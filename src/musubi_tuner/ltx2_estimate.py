#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Value
from pathlib import Path
from typing import Any, Optional

import torch
from accelerate.utils import set_seed
from safetensors import safe_open
from tqdm import tqdm

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.training.accelerator_setup import (
    clean_memory_on_device,
    collator_class,
    dataloader_extra_kwargs,
    prepare_accelerator,
)
from musubi_tuner.ltx2_train import (
    _all_declared_datasets_are_audio,
    _extract_transformer_block_index,
    _masked_mse,
    _normalize_ltx2_batch_for_call_dit,
)
from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer, ltx2_setup_parser
from musubi_tuner.modules.nf4_optimization_utils import dequantize_nf4_block, is_nf4_module
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.networks.lora_ltx2 import LTX2_LORA_TARGET_PRESETS
from musubi_tuner.training.model_helpers import load_network_state_dict
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


TARGETABLE_FAMILY_PATTERNS: list[tuple[str, str]] = [
    ("audio_to_video_attn", r"(?:^|\.)(?:model\.)?transformer_blocks\.\d+\.audio_to_video_attn\.(?:to_q|to_k|to_v|to_out\.0)$"),
    ("video_to_audio_attn", r"(?:^|\.)(?:model\.)?transformer_blocks\.\d+\.video_to_audio_attn\.(?:to_q|to_k|to_v|to_out\.0)$"),
    ("audio_attn", r"(?:^|\.)(?:model\.)?transformer_blocks\.\d+\.audio_attn(?:1|2)\.(?:to_q|to_k|to_v|to_out\.0)$"),
    ("audio_ff", r"(?:^|\.)(?:model\.)?transformer_blocks\.\d+\.audio_ff\.net\.(?:0\.proj|2)$"),
    ("video_self_attn", r"(?:^|\.)(?:model\.)?transformer_blocks\.\d+\.attn1\.(?:to_q|to_k|to_v|to_out\.0)$"),
    ("video_cross_attn", r"(?:^|\.)(?:model\.)?transformer_blocks\.\d+\.attn2\.(?:to_q|to_k|to_v|to_out\.0)$"),
    ("video_ff", r"(?:^|\.)(?:model\.)?transformer_blocks\.\d+\.ff\.net\.(?:0\.proj|2)$"),
]

PRESET_CANDIDATES_BY_MODE: dict[str, list[str]] = {
    "video": ["video_sa", "video_sa_ff", "video_sa_ca_ff", "t2v", "v2v", "full"],
    "av": ["t2v", "v2v", "av_ic", "video_ref_only_av", "audio", "audio_v2a", "audio_ref_ic", "full"],
    "audio": ["audio", "audio_v2a", "audio_ref_ic", "t2v", "v2v", "full"],
}


@dataclass
class CandidateParam:
    name: str
    module_path: str
    family: str
    block_index: Optional[int]
    param: torch.Tensor
    numel: int
    module: Optional[torch.nn.Module] = None


@dataclass
class ShadowWeightBinding:
    candidate_name: str
    module: torch.nn.Module
    shadow_param: torch.nn.Parameter
    original_weight: torch.Tensor
    original_weight_kind: str
    original_forward: Any


def setup_parser() -> argparse.ArgumentParser:
    parser = setup_parser_common()
    parser = ltx2_setup_parser(parser)

    parser.add_argument(
        "--estimation_batches",
        type=int,
        default=16,
        help="Number of batches used to estimate Fisher-style importance.",
    )
    parser.add_argument(
        "--estimation_output",
        type=str,
        default=None,
        help="Optional JSON output path. Defaults to <output_dir>/ltx2_estimate.json.",
    )
    parser.add_argument(
        "--estimation_top_k",
        type=int,
        default=24,
        help="Top-N module paths to include in the report.",
    )
    parser.add_argument(
        "--estimation_target_coverage",
        type=float,
        default=0.85,
        help="Pick the smallest preset whose Fisher share reaches this coverage target when possible.",
    )
    parser.add_argument(
        "--estimation_keep_caption_dropout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep caption dropout active during estimation. Disabled by default for a more stable score.",
    )
    parser.add_argument(
        "--estimation_block_window",
        type=int,
        default=1,
        help="Number of transformer blocks to score per backward pass. Smaller values reduce VRAM and keep the estimate exact.",
    )
    return parser


def _module_family(module_path: str) -> Optional[str]:
    for family, pattern in TARGETABLE_FAMILY_PATTERNS:
        if re.search(pattern, module_path):
            return family
    return None


def _candidate_params(transformer: torch.nn.Module) -> list[CandidateParam]:
    selected: list[CandidateParam] = []
    for module_path, module in transformer.named_modules():
        family = _module_family(module_path)
        if family is None:
            continue
        param = getattr(module, "weight", None)
        if not isinstance(param, torch.Tensor) or param.ndim != 2:
            continue
        numel = int(param.numel())
        if is_nf4_module(module):
            out_features = int(getattr(module, "nf4_out_features", 0) or 0)
            in_features = int(getattr(module, "nf4_in_features", 0) or 0)
            if out_features > 0 and in_features > 0:
                numel = out_features * in_features
        selected.append(
            CandidateParam(
                name=f"{module_path}.weight",
                module_path=module_path,
                family=family,
                block_index=_extract_transformer_block_index(module_path),
                param=param,
                numel=numel,
                module=module,
            )
        )
    return selected


def _module_path_from_lora_name(lora_name: str) -> Optional[str]:
    if not lora_name.startswith("lora_unet_"):
        return None

    suffix = lora_name[len("lora_unet_") :]
    match = re.match(r"^model_transformer_blocks_(\d+)_(.+)$", suffix)
    if not match:
        return None

    block_index, tail = match.groups()
    tail = re.sub(r"^audio_to_video_attn_", "audio_to_video_attn.", tail)
    tail = re.sub(r"^video_to_audio_attn_", "video_to_audio_attn.", tail)
    tail = re.sub(r"^audio_attn1_", "audio_attn1.", tail)
    tail = re.sub(r"^audio_attn2_", "audio_attn2.", tail)
    tail = re.sub(r"^audio_ff_", "audio_ff.", tail)
    tail = re.sub(r"^attn1_", "attn1.", tail)
    tail = re.sub(r"^attn2_", "attn2.", tail)
    tail = re.sub(r"^ff_", "ff.", tail)
    tail = re.sub(r"to_out_0$", "to_out.0", tail)
    tail = re.sub(r"to_q$", "to_q", tail)
    tail = re.sub(r"to_k$", "to_k", tail)
    tail = re.sub(r"to_v$", "to_v", tail)
    return f"model.transformer_blocks.{block_index}.{tail}"


def _candidate_params_from_network(network: torch.nn.Module) -> list[CandidateParam]:
    unet_loras = list(getattr(network, "unet_loras", []) or [])
    selected: list[CandidateParam] = []
    for lora in unet_loras:
        lora_name = str(getattr(lora, "lora_name", "") or "")
        module_path = _module_path_from_lora_name(lora_name)
        if not module_path:
            continue
        family = _module_family(module_path)
        if family is None:
            continue
        block_index = _extract_transformer_block_index(module_path)
        for param_name, param in lora.named_parameters():
            if not param_name.endswith(".weight"):
                continue
            if not isinstance(param, torch.nn.Parameter) or param.ndim < 2:
                continue
            selected.append(
                CandidateParam(
                    name=f"{lora_name}.{param_name}",
                    module_path=module_path,
                    family=family,
                    block_index=block_index,
                    param=param,
                    numel=int(param.numel()),
                    module=None,
                )
            )
    return selected


def _match_preset(module_path: str, preset_name: str) -> bool:
    patterns = LTX2_LORA_TARGET_PRESETS[preset_name]
    if patterns is None:
        return True
    return any(re.search(pattern, module_path) for pattern in patterns)


def _iter_estimation_weight_specs(args: argparse.Namespace) -> list[tuple[str, float, str]]:
    specs: list[tuple[str, float, str]] = []
    base_weights = list(getattr(args, "base_weights", None) or [])
    base_multipliers = list(getattr(args, "base_weights_multiplier", None) or [])
    for i, weight_path in enumerate(base_weights):
        multiplier = float(base_multipliers[i]) if i < len(base_multipliers) else 1.0
        specs.append((str(weight_path), multiplier, "base_weights"))

    return specs


def _infer_network_module_from_weight(weight_path: str) -> Optional[str]:
    try:
        with safe_open(weight_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception as exc:
        logger.warning("Estimator: failed to inspect LoRA metadata for %s: %s", weight_path, exc)
        return None

    network_module = metadata.get("ss_network_module")
    if network_module:
        return str(network_module)
    return None


def _resolve_estimation_network_module(args: argparse.Namespace, weight_specs: list[tuple[str, float, str]]) -> Optional[str]:
    network_module = getattr(args, "network_module", None)
    if network_module:
        return str(network_module)

    for weight_path, _multiplier, _source in weight_specs:
        inferred = _infer_network_module_from_weight(weight_path)
        if inferred:
            logger.info("Estimator: inferred network_module=%s from %s", inferred, weight_path)
            return inferred

    if weight_specs:
        default_module = "networks.lora_ltx2"
        logger.info("Estimator: defaulting network_module to %s for LoRA merge", default_module)
        return default_module

    return None


def _merge_estimation_weights(
    trainer: LTX2NetworkTrainer,
    transformer: torch.nn.Module,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    weight_specs = _iter_estimation_weight_specs(args)
    if not weight_specs:
        return []

    network_module_name = _resolve_estimation_network_module(args, weight_specs)
    if not network_module_name:
        raise ValueError("Estimator: network_module could not be resolved for LoRA/base weight merge.")

    module_root = os.path.dirname(__file__)
    if module_root not in sys.path:
        sys.path.append(module_root)
    network_module = trainer._resolve_network_module(network_module_name)
    merge_device = next(transformer.parameters()).device
    merged_weights: list[dict[str, Any]] = []

    for weight_path, multiplier, source in weight_specs:
        logger.info(
            "Estimator: merging %s from %s with multiplier %.3f using %s",
            weight_path,
            source,
            multiplier,
            network_module_name,
        )
        weights_sd = trainer.load_network_weights(weight_path, network_module)
        module = network_module.create_arch_network_from_weights(
            multiplier,
            weights_sd,
            unet=transformer,
            for_inference=True,
        )
        try:
            module.merge_to(None, transformer, weights_sd, device=merge_device, non_blocking=True)
        except TypeError:
            merge_dtype = next(transformer.parameters()).dtype
            module.merge_to(None, transformer, weights_sd, merge_dtype, str(merge_device))

        merged_weights.append(
            {
                "path": str(weight_path),
                "multiplier": float(multiplier),
                "source": source,
                "network_module": network_module_name,
            }
        )

    return merged_weights


def _apply_estimation_network(
    trainer: LTX2NetworkTrainer,
    transformer: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[Optional[torch.nn.Module], Optional[dict[str, Any]]]:
    network_weights_path = getattr(args, "network_weights", None)
    if not network_weights_path:
        return None, None

    weight_path = str(network_weights_path)
    network_module_name = _resolve_estimation_network_module(args, [(weight_path, 1.0, "network_weights")])
    if not network_module_name:
        raise ValueError("Estimator: network_module could not be resolved for network_weights.")

    module_root = os.path.dirname(__file__)
    if module_root not in sys.path:
        sys.path.append(module_root)
    network_module = trainer._resolve_network_module(network_module_name)

    logger.info("Estimator: applying network weights from %s using %s", weight_path, network_module_name)
    weights_sd = trainer.load_network_weights(weight_path, network_module)
    network = network_module.create_arch_network_from_weights(
        1.0,
        weights_sd,
        unet=transformer,
        for_inference=False,
    )
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    info = load_network_state_dict(network, weights_sd, False)
    network.train()
    network.requires_grad_(False)

    return network, {
        "path": weight_path,
        "network_module": network_module_name,
        "load_info": str(info),
    }


def _build_dataloader(args: argparse.Namespace, trainer: LTX2NetworkTrainer) -> torch.utils.data.DataLoader:
    current_epoch = Value("i", 0)
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
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
        raise ValueError("No training items found in the dataset. Create latent/text caches first.")

    train_dataset_group.set_max_train_steps(max(1, int(args.estimation_batches)))

    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = collator_class(current_epoch, ds_for_collator)
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() or 1)

    return torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
        **dataloader_extra_kwargs(args, n_workers),
    )


def _resolve_attn_mode(args: argparse.Namespace) -> str:
    if args.sdpa:
        return "torch"
    if args.flash_attn:
        return "flash"
    if args.flash3:
        return "flash3"
    if args.xformers:
        return "xformers"
    return "torch"


def _build_estimation_passes(candidates: list[CandidateParam], block_window: int) -> list[list[int]]:
    block_ids = sorted({cand.block_index for cand in candidates if cand.block_index is not None})
    if not block_ids:
        return [[]]
    window = max(1, int(block_window or 1))
    return [block_ids[i : i + window] for i in range(0, len(block_ids), window)]


def _candidate_needs_shadow_weight(cand: CandidateParam) -> bool:
    module = cand.module
    if module is None:
        return False
    if is_nf4_module(module):
        return True
    weight = cand.param
    return (not weight.is_floating_point()) or weight.dtype.itemsize < 2


def _shadow_weight_dtype(cand: CandidateParam, fallback_dtype: torch.dtype) -> torch.dtype:
    module = cand.module
    if module is not None:
        scale_weight = getattr(module, "scale_weight", None)
        if isinstance(scale_weight, torch.Tensor) and scale_weight.is_floating_point() and scale_weight.dtype.itemsize >= 2:
            return scale_weight.dtype
    if fallback_dtype.is_floating_point and fallback_dtype.itemsize >= 2:
        return fallback_dtype
    return torch.float32


def _dequantize_candidate_weight(cand: CandidateParam, target_dtype: torch.dtype) -> torch.Tensor:
    module = cand.module
    if module is None:
        raise ValueError(f"Estimator: candidate {cand.name} has no module for shadow dequantization.")

    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(f"Estimator: module {cand.module_path} has no tensor weight.")

    if is_nf4_module(module):
        scale_weight = getattr(module, "scale_weight", None)
        if not isinstance(scale_weight, torch.Tensor):
            raise ValueError(f"Estimator: NF4 module {cand.module_path} is missing scale_weight.")
        dense = dequantize_nf4_block(
            weight,
            scale_weight,
            int(getattr(module, "nf4_out_features")),
            int(getattr(module, "nf4_in_features")),
            int(getattr(module, "nf4_block_size")),
            scale_weight.dtype,
        )
        if hasattr(module, "awq_scales"):
            dense = dense / module.awq_scales.unsqueeze(0)
        return dense.to(dtype=target_dtype)

    scale_weight = getattr(module, "scale_weight", None)
    if not isinstance(scale_weight, torch.Tensor):
        return weight.to(dtype=target_dtype)

    scale = scale_weight.to(dtype=target_dtype)
    dense_weight = weight.to(dtype=target_dtype)
    if scale.ndim < 3:
        return dense_weight * scale

    out_features, num_blocks, _ = scale.shape
    dense_weight = dense_weight.contiguous().view(out_features, num_blocks, -1)
    dense_weight = dense_weight * scale
    return dense_weight.view(weight.shape)


def _shadow_linear_forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x, self.weight, self.bias)


def _install_shadow_weight(cand: CandidateParam, fallback_dtype: torch.dtype) -> ShadowWeightBinding:
    module = cand.module
    if module is None:
        raise ValueError(f"Estimator: candidate {cand.name} has no module for shadow-weight estimation.")

    if "weight" in module._parameters:
        original_weight_kind = "parameter"
    elif "weight" in module._buffers:
        original_weight_kind = "buffer"
    else:
        original_weight_kind = "attr"

    original_weight = getattr(module, "weight")
    shadow_dtype = _shadow_weight_dtype(cand, fallback_dtype)
    shadow_param = torch.nn.Parameter(_dequantize_candidate_weight(cand, shadow_dtype).detach(), requires_grad=True)
    original_forward = module.forward

    module._parameters.pop("weight", None)
    module._buffers.pop("weight", None)
    module.register_parameter("weight", shadow_param)
    module.forward = _shadow_linear_forward.__get__(module, type(module))

    return ShadowWeightBinding(
        candidate_name=cand.name,
        module=module,
        shadow_param=shadow_param,
        original_weight=original_weight,
        original_weight_kind=original_weight_kind,
        original_forward=original_forward,
    )


def _restore_shadow_weight(binding: ShadowWeightBinding) -> None:
    module = binding.module
    module.forward = binding.original_forward
    module._parameters.pop("weight", None)
    module._buffers.pop("weight", None)
    if binding.original_weight_kind == "parameter":
        module.register_parameter("weight", binding.original_weight)
    elif binding.original_weight_kind == "buffer":
        module.register_buffer("weight", binding.original_weight)
    else:
        setattr(module, "weight", binding.original_weight)


def _validate_base_model_candidates(candidates: list[CandidateParam]) -> None:
    unsupported = sorted(
        {str(cand.param.dtype) for cand in candidates if _candidate_needs_shadow_weight(cand) and cand.module is None}
    )
    if unsupported:
        raise ValueError(
            "Base-model estimation encountered quantized candidates without module references "
            f"({', '.join(unsupported)}), so shadow dequantization could not be installed."
        )


def _direct_grad_param(cand: CandidateParam) -> Optional[torch.Tensor]:
    if _candidate_needs_shadow_weight(cand):
        return None
    if not isinstance(cand.param, torch.Tensor) or not cand.param.is_floating_point():
        return None
    return cand.param


def _loss_from_prediction(
    args: argparse.Namespace,
    trainer: LTX2NetworkTrainer,
    model_pred: Any,
    target: torch.Tensor,
    weighting: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if isinstance(model_pred, dict):
        out = model_pred
        if out.get("_skip_step"):
            return None
        loss_type = getattr(args, "loss_type", "mse")
        huber_delta = float(getattr(args, "huber_delta", 1.0))
        loss = _masked_mse(
            out["video_pred"],
            out["video_target"],
            out.get("video_loss_mask"),
            weighting=weighting,
            dtype=trainer.dit_dtype,
            loss_type=loss_type,
            huber_delta=huber_delta,
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
                loss_type=loss_type,
                huber_delta=huber_delta,
            ) * float(out.get("audio_loss_weight", 1.0))
        return loss

    loss_type = getattr(args, "loss_type", "mse")
    huber_delta = float(getattr(args, "huber_delta", 1.0))
    pred = model_pred.to(device=target.device, dtype=trainer.dit_dtype)
    if loss_type in ("mae", "l1"):
        loss = torch.nn.functional.l1_loss(pred, target, reduction="none")
    elif loss_type in ("huber", "smooth_l1"):
        loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=huber_delta)
    else:
        loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
    if weighting is not None:
        w = weighting
        if isinstance(w, torch.Tensor) and w.dim() != loss.dim():
            while w.dim() > loss.dim() and w.shape[-1] == 1:
                w = w.squeeze(-1)
        loss = loss * w
    return loss.mean()


def _score_report(
    *,
    args: argparse.Namespace,
    candidates: list[CandidateParam],
    valid_batches: int,
    fisher_sums: dict[str, float],
    grad_norm_sums: dict[str, float],
) -> dict[str, Any]:
    if valid_batches <= 0:
        raise ValueError("Estimator produced 0 valid batches.")

    param_rows: list[dict[str, Any]] = []
    total_fisher = 0.0
    total_numel = 0
    for cand in candidates:
        fisher_sum = float(fisher_sums.get(cand.name, 0.0))
        total_fisher += fisher_sum
        total_numel += cand.numel
        fisher_mean = fisher_sum / float(valid_batches * max(1, cand.numel))
        param_rows.append(
            {
                "name": cand.name,
                "module_path": cand.module_path,
                "family": cand.family,
                "block_index": cand.block_index,
                "numel": cand.numel,
                "fisher_sum": fisher_sum,
                "fisher_mean": fisher_mean,
                "fisher_rms": math.sqrt(max(fisher_mean, 0.0)),
                "grad_l2_mean": float(grad_norm_sums.get(cand.name, 0.0)) / float(valid_batches),
            }
        )

    def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        agg_fisher = sum(float(row["fisher_sum"]) for row in rows)
        agg_numel = sum(int(row["numel"]) for row in rows)
        fisher_mean = agg_fisher / float(valid_batches * max(1, agg_numel))
        fisher_share = agg_fisher / total_fisher if total_fisher > 0 else 0.0
        param_share = agg_numel / total_numel if total_numel > 0 else 0.0
        return {
            "module_count": len(rows),
            "numel": agg_numel,
            "fisher_sum": agg_fisher,
            "fisher_mean": fisher_mean,
            "fisher_rms": math.sqrt(max(fisher_mean, 0.0)),
            "fisher_share": fisher_share,
            "param_share": param_share,
            "efficiency": fisher_share / max(param_share, 1e-12),
        }

    family_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in param_rows:
        family_buckets[row["family"]].append(row)
    family_scores = {family: _aggregate(rows) for family, rows in family_buckets.items()}
    family_scores = dict(
        sorted(
            family_scores.items(),
            key=lambda item: (item[1]["fisher_mean"], item[1]["fisher_share"]),
            reverse=True,
        )
    )

    preset_scores: dict[str, dict[str, Any]] = {}
    for preset_name in PRESET_CANDIDATES_BY_MODE.get(getattr(args, "ltx_mode", "video"), []):
        matched = [row for row in param_rows if _match_preset(str(row["module_path"]), preset_name)]
        if not matched:
            continue
        preset_scores[preset_name] = _aggregate(matched)

    sorted_presets = sorted(
        preset_scores.items(),
        key=lambda item: (item[1]["param_share"], -item[1]["fisher_share"]),
    )
    target_coverage = float(getattr(args, "estimation_target_coverage", 0.85))
    recommended_preset = None
    for preset_name, scores in sorted_presets:
        if scores["fisher_share"] >= target_coverage:
            recommended_preset = preset_name
            break
    if recommended_preset is None and preset_scores:
        recommended_preset = max(
            preset_scores.items(),
            key=lambda item: (item[1]["efficiency"], item[1]["fisher_share"]),
        )[0]

    top_modules = sorted(
        param_rows,
        key=lambda row: (row["fisher_mean"], row["fisher_sum"]),
        reverse=True,
    )[: max(1, int(getattr(args, "estimation_top_k", 24) or 24))]

    return {
        "meta": {
            "ltx_mode": getattr(args, "ltx_mode", "video"),
            "ltx_version": getattr(args, "ltx_version", "2.3"),
            "estimation_batches": valid_batches,
            "dataset_config": str(args.dataset_config),
            "checkpoint": str(args.ltx2_checkpoint),
            "target_coverage": target_coverage,
            "caption_dropout_kept": bool(getattr(args, "estimation_keep_caption_dropout", False)),
        },
        "summary": {
            "candidate_modules": len(param_rows),
            "candidate_params": total_numel,
            "total_fisher_sum": total_fisher,
            "recommended_preset": recommended_preset,
        },
        "family_scores": family_scores,
        "preset_scores": preset_scores,
        "module_scores": param_rows,
        "top_modules": top_modules,
    }


def run_estimation(args: argparse.Namespace) -> dict[str, Any]:
    trainer = LTX2NetworkTrainer()
    trainer.handle_model_specific_args(args)

    blocks_to_swap = int(getattr(args, "blocks_to_swap", 0) or 0)
    if blocks_to_swap > 0:
        current_val = os.environ.get("LTX2_SWAP_TRAIN_FULL")
        os.environ["LTX2_SWAP_TRAIN_FULL"] = "1"
        if current_val is None:
            logger.info("Estimator: auto-enabled LTX2_SWAP_TRAIN_FULL=1 (blocks_to_swap=%d)", blocks_to_swap)
        elif current_val != "1":
            logger.info(
                "Estimator: overriding LTX2_SWAP_TRAIN_FULL from '%s' to '1' (blocks_to_swap=%d)",
                current_val,
                blocks_to_swap,
            )

    accelerator = prepare_accelerator(args)
    if getattr(args, "seed", None) is not None:
        set_seed(args.seed)

    dataloader = _build_dataloader(args, trainer)

    transformer = trainer.load_transformer(
        accelerator=accelerator,
        args=args,
        dit_path=args.ltx2_checkpoint,
        attn_mode=_resolve_attn_mode(args),
        split_attn=bool(getattr(args, "split_attn", False)),
        loading_device="cpu" if int(getattr(args, "blocks_to_swap", 0) or 0) > 0 else accelerator.device,
        dit_weight_dtype=None,
    )
    merged_weights = _merge_estimation_weights(trainer, transformer, args)
    transformer.train()
    transformer.requires_grad_(False)
    trainer.blocks_to_swap = blocks_to_swap
    if blocks_to_swap > 0 and hasattr(transformer, "enable_block_swap"):
        logger.info("Estimator: enabling block swap (%d blocks)", blocks_to_swap)
        transformer.enable_block_swap(
            blocks_to_swap,
            accelerator.device,
            supports_backward=True,
            use_pinned_memory=bool(getattr(args, "use_pinned_memory_for_block_swap", False)),
        )
        transformer.move_to_device_except_swap_blocks(accelerator.device)

    applied_network, applied_network_info = _apply_estimation_network(trainer, transformer, args)

    if getattr(args, "gradient_checkpointing", False):
        blocks_to_ckpt = getattr(args, "blocks_to_checkpoint", -1)
        if getattr(args, "blockwise_checkpointing", False):
            transformer.enable_gradient_checkpointing(
                getattr(args, "gradient_checkpointing_cpu_offload", False),
                weight_cpu_offloading=True,
                blocks_to_checkpoint=blocks_to_ckpt,
            )
            if getattr(args, "use_pinned_memory_for_block_swap", False) and hasattr(transformer, "transformer_blocks"):
                for block in transformer.transformer_blocks:
                    if hasattr(block, "use_pinned_memory"):
                        block.use_pinned_memory = True
        else:
            transformer.enable_gradient_checkpointing(
                getattr(args, "gradient_checkpointing_cpu_offload", False),
                blocks_to_checkpoint=blocks_to_ckpt,
            )

    if blocks_to_swap > 0:
        transformer = accelerator.prepare(transformer, device_placement=[False])
        accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
    else:
        transformer = accelerator.prepare(transformer)

    if getattr(args, "compile", False):
        transformer = trainer.compile_transformer(args, transformer)
        transformer.__dict__["_orig_mod"] = transformer

    if applied_network is not None:
        applied_network = accelerator.prepare(applied_network)
        accelerator.unwrap_model(applied_network).prepare_grad_etc(transformer)

    unwrapped_transformer = accelerator.unwrap_model(transformer)
    unwrapped_network = accelerator.unwrap_model(applied_network) if applied_network is not None else None

    if unwrapped_network is not None:
        candidates = _candidate_params_from_network(unwrapped_network)
        candidate_source = "network"
    else:
        candidates = _candidate_params(unwrapped_transformer)
        candidate_source = "base_model"
        _validate_base_model_candidates(candidates)
    if not candidates:
        raise ValueError("No LoRA-targetable LTX-2 linear weights were found for estimation.")
    for cand in candidates:
        direct_param = _direct_grad_param(cand)
        if direct_param is not None:
            direct_param.requires_grad_(False)

    clean_memory_on_device(accelerator.device)
    noise_scheduler = FlowMatchDiscreteScheduler(
        shift=args.discrete_flow_shift,
        reverse=True,
        solver="euler",
    )

    fisher_sums: dict[str, float] = defaultdict(float)
    grad_norm_sums: dict[str, float] = defaultdict(float)
    estimation_batches = int(args.estimation_batches)
    skipped_batches = 0
    cached_batches: list[dict[str, Any]] = []
    for batch in dataloader:
        if len(cached_batches) >= estimation_batches:
            break
        batch = _normalize_ltx2_batch_for_call_dit(batch)
        latents = batch.get("latents")
        if isinstance(latents, dict):
            latents = latents.get("latents")
        if not isinstance(latents, torch.Tensor) or latents.dim() != 5:
            skipped_batches += 1
            continue
        cached_batches.append(copy.deepcopy(batch))

    valid_batches = len(cached_batches)
    if valid_batches <= 0:
        raise ValueError("Estimator produced 0 valid batches.")

    if candidate_source == "network":
        block_passes: list[Optional[list[int]]] = [None]
    else:
        block_passes = _build_estimation_passes(candidates, int(getattr(args, "estimation_block_window", 1) or 1))
    used_shadow_dequant = candidate_source == "base_model" and any(_candidate_needs_shadow_weight(cand) for cand in candidates)
    total_steps = len(block_passes) * valid_batches

    progress = tqdm(
        total=total_steps,
        desc="estimate",
        leave=False,
        disable=not accelerator.is_local_main_process,
    )

    original_caption_dropout = float(getattr(args, "caption_dropout_rate", 0.0))
    if not bool(getattr(args, "estimation_keep_caption_dropout", False)):
        args.caption_dropout_rate = 0.0

    start_time = time.time()
    try:
        for pass_block_ids in block_passes:
            if pass_block_ids is None:
                active_candidates = list(candidates)
            else:
                active_candidates = [
                    cand for cand in candidates if cand.block_index is not None and cand.block_index in pass_block_ids
                ]
            if not active_candidates:
                continue

            active_param_names = {cand.name for cand in active_candidates}
            for cand in candidates:
                direct_param = _direct_grad_param(cand)
                if direct_param is not None:
                    direct_param.requires_grad_(cand.name in active_param_names)

            shadow_bindings: dict[str, ShadowWeightBinding] = {}
            for cand in active_candidates:
                if _candidate_needs_shadow_weight(cand):
                    shadow_bindings[cand.name] = _install_shadow_weight(cand, trainer.dit_dtype)

            if blocks_to_swap > 0 and hasattr(unwrapped_transformer, "switch_block_swap_for_training"):
                unwrapped_transformer.switch_block_swap_for_training()
            elif blocks_to_swap > 0 and hasattr(unwrapped_transformer, "prepare_block_swap_before_forward"):
                unwrapped_transformer.prepare_block_swap_before_forward()

            active_grad_params = [
                shadow_bindings[cand.name].shadow_param if cand.name in shadow_bindings else cand.param
                for cand in active_candidates
            ]
            optimizer = torch.optim.SGD(active_grad_params, lr=0.0)
            try:
                for batch in cached_batches:
                    batch = copy.deepcopy(batch)
                    latents = batch.get("latents")
                    if isinstance(latents, dict):
                        latents = latents.get("latents")
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
                        args.weighting_scheme,
                        noise_scheduler,
                        timesteps,
                        accelerator.device,
                        trainer.dit_dtype,
                    )
                    if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
                        trainer._ensure_fp8_buffers_on_device(transformer)
                    elif getattr(args, "nf4_base", False):
                        trainer._ensure_nf4_buffers_on_device(transformer)
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
                    loss = _loss_from_prediction(args, trainer, model_pred, target, weighting)
                    if loss is None:
                        optimizer.zero_grad(set_to_none=True)
                        progress.update(1)
                        continue

                    accelerator.backward(loss)
                    for cand in active_candidates:
                        grad_param = shadow_bindings[cand.name].shadow_param if cand.name in shadow_bindings else cand.param
                        grad = grad_param.grad
                        if grad is None:
                            continue
                        grad_f = grad.detach().float()
                        fisher_sums[cand.name] += float(grad_f.square().sum().item())
                        grad_norm_sums[cand.name] += float(grad_f.norm().item())
                    optimizer.zero_grad(set_to_none=True)
                    progress.update(1)
                    if accelerator.is_local_main_process:
                        if pass_block_ids is None:
                            progress.set_postfix(valid=valid_batches, skipped=skipped_batches, blocks="all")
                        else:
                            progress.set_postfix(
                                valid=valid_batches, skipped=skipped_batches, blocks=",".join(map(str, pass_block_ids))
                            )
            finally:
                optimizer.zero_grad(set_to_none=True)
                for cand in active_candidates:
                    direct_param = _direct_grad_param(cand)
                    if direct_param is not None:
                        direct_param.requires_grad_(False)
                for binding in shadow_bindings.values():
                    _restore_shadow_weight(binding)
                if blocks_to_swap > 0 and hasattr(unwrapped_transformer, "prepare_block_swap_before_forward"):
                    unwrapped_transformer.prepare_block_swap_before_forward()
                clean_memory_on_device(accelerator.device)
    finally:
        progress.close()
        args.caption_dropout_rate = original_caption_dropout

    report = _score_report(
        args=args,
        candidates=candidates,
        valid_batches=valid_batches,
        fisher_sums=fisher_sums,
        grad_norm_sums=grad_norm_sums,
    )
    report["meta"]["elapsed_sec"] = time.time() - start_time
    report["meta"]["skipped_batches"] = skipped_batches
    report["meta"]["compile"] = bool(getattr(args, "compile", False))
    report["meta"]["blocks_to_swap"] = int(getattr(args, "blocks_to_swap", 0) or 0)
    report["meta"]["fp8_base"] = bool(getattr(args, "fp8_base", False))
    report["meta"]["fp8_scaled"] = bool(getattr(args, "fp8_scaled", False))
    report["meta"]["nf4_base"] = bool(getattr(args, "nf4_base", False))
    report["meta"]["merged_weights"] = merged_weights
    report["meta"]["applied_network"] = applied_network_info
    report["meta"]["candidate_source"] = candidate_source
    report["meta"]["shadow_dequantized_candidates"] = used_shadow_dequant
    return report


def _default_output_path(args: argparse.Namespace) -> Path:
    output_dir = Path(getattr(args, "output_dir", ".") or ".")
    return output_dir / "ltx2_estimate.json"


def _print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    logger.info(
        "LTX-2 estimate complete: preset=%s candidates=%d params=%d",
        summary.get("recommended_preset"),
        int(summary.get("candidate_modules", 0)),
        int(summary.get("candidate_params", 0)),
    )

    family_scores = report.get("family_scores", {})
    if family_scores:
        logger.info("Top families by normalized Fisher score:")
        for family, scores in list(family_scores.items())[:5]:
            logger.info(
                "  %s: fisher_mean=%.6e fisher_share=%.3f efficiency=%.3f",
                family,
                float(scores["fisher_mean"]),
                float(scores["fisher_share"]),
                float(scores["efficiency"]),
            )

    preset_scores = report.get("preset_scores", {})
    if preset_scores:
        logger.info("Preset coverage:")
        for preset_name, scores in preset_scores.items():
            logger.info(
                "  %s: fisher_share=%.3f param_share=%.3f efficiency=%.3f",
                preset_name,
                float(scores["fisher_share"]),
                float(scores["param_share"]),
                float(scores["efficiency"]),
            )


def _normalize_estimation_args(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "dataset_config", None) is None:
        raise ValueError("dataset_config is required")
    if getattr(args, "ltx2_checkpoint", None) is None:
        raise ValueError("ltx2_checkpoint is required")

    short_map = {"v": "video", "a": "audio", "va": "av"}
    if getattr(args, "ltx_mode", None) in short_map:
        args.ltx_mode = short_map[args.ltx_mode]
    if getattr(args, "ltx_mode", "video") == "video":
        user_config = config_utils.load_user_config(args.dataset_config)
        if _all_declared_datasets_are_audio(user_config):
            logger.info("All datasets are audio-only; automatically switching to --ltx2_mode audio")
            args.ltx_mode = "audio"

    if int(args.estimation_batches) <= 0:
        raise ValueError("estimation_batches must be >= 1")
    if float(args.estimation_target_coverage) <= 0.0 or float(args.estimation_target_coverage) > 1.0:
        raise ValueError("estimation_target_coverage must be in (0, 1]")
    args.max_train_steps = max(1, int(args.estimation_batches))
    return args


def build_estimation_args_from_training_args(
    source_args: Any,
    *,
    estimation_output: Optional[str | Path] = None,
) -> argparse.Namespace:
    if not hasattr(source_args, "__dict__"):
        raise TypeError("source_args must provide attributes via __dict__")

    estimate_args = argparse.Namespace(**vars(source_args).copy())
    defaults = {
        "estimation_batches": 16,
        "estimation_output": None,
        "estimation_top_k": 24,
        "estimation_target_coverage": 0.85,
        "estimation_keep_caption_dropout": False,
        "estimation_block_window": 1,
    }
    for key, default_value in defaults.items():
        if getattr(estimate_args, key, None) is None:
            setattr(estimate_args, key, default_value)

    if estimation_output is not None:
        estimate_args.estimation_output = str(estimation_output)

    return _normalize_estimation_args(estimate_args)


def write_estimation_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    temp_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temp_path, output_path)
    return output_path


def generate_estimation_report(
    source_args: Any,
    *,
    estimation_output: Optional[str | Path] = None,
) -> Path:
    args = build_estimation_args_from_training_args(source_args, estimation_output=estimation_output)
    report = run_estimation(args)
    output_path = Path(args.estimation_output) if args.estimation_output else _default_output_path(args)
    output_path = write_estimation_report(report, output_path)
    _print_summary(report)
    logger.info("Saved estimate report to %s", output_path)
    return output_path


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()
    args = read_config_from_file(args, parser)
    generate_estimation_report(args)


if __name__ == "__main__":
    main()
