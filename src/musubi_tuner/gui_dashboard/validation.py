"""Shared GUI validation rules for process launch."""

from __future__ import annotations

import shlex
import math
from pathlib import Path
from typing import Any

from musubi_tuner.gui_dashboard.project_schema import DatasetEntry, ProjectConfig


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())


def _parse_network_args(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    try:
        parts = shlex.split(raw, posix=False)
    except ValueError:
        return {}

    parsed: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if not key:
            continue
        parsed[key.strip()] = value.strip().strip("\"'")
    return parsed


def _lokr_factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
    """Mirror networks.lokr.factorization without importing the torch-heavy backend."""
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def _lokr_dense_threshold_for_pair(in_dim: int, out_dim: int, factor: int) -> int:
    _, in_n = _lokr_factorization(in_dim, factor)
    _, out_k = _lokr_factorization(out_dim, factor)
    return int(max(out_k, in_n) / 2)


def _get_lokr_dim_warning(t) -> str | None:
    network_module = getattr(t, "network_module", None) or ""
    if network_module != "networks.lokr":
        return None

    network_dim = getattr(t, "network_dim", None)
    if network_dim is None:
        return None

    try:
        dim = int(network_dim)
    except (TypeError, ValueError):
        return None
    if dim <= 0:
        return None

    factor = -1
    gui_factor = getattr(t, "lokr_factor", None)
    if gui_factor is not None:
        try:
            factor = int(gui_factor)
        except (TypeError, ValueError):
            factor = -1
    raw_factor = _parse_network_args(getattr(t, "network_args", "")).get("factor")
    if gui_factor is None and raw_factor is not None:
        try:
            factor = int(float(raw_factor))
        except ValueError:
            factor = -1

    attention_pairs = [
        (4096, 4096),
        (2048, 2048),
        (4096, 2048),
        (2048, 4096),
    ]
    ffn_pairs = [
        (4096, 16384),
        (16384, 4096),
        (2048, 8192),
        (8192, 2048),
    ]
    include_ffn = getattr(t, "lora_target_preset", "t2v") in {
        "v2v",
        "video_sa_ff",
        "video_sa_ca_ff",
        "character_training",
        "audio",
        "audio_ref_only_ic",
        "av_ic",
        "video_ref_only_av",
        "full",
    }
    target_pairs = attention_pairs + (ffn_pairs if include_ffn else [])
    threshold = max(_lokr_dense_threshold_for_pair(in_dim, out_dim, factor) for in_dim, out_dim in target_pairs)
    if dim < threshold:
        return None

    target_label = "attention+FFN" if include_ffn else "attention-only"
    factor_label = f"factor={factor}" if factor > 0 else "balanced factorization"
    return (
        f"For the current LoKr/DoKr target set ({target_label}, {factor_label}), dims >= {threshold} already use "
        f"dense LoKr blocks. network_dim={dim} will not increase effective LoKr capacity beyond that threshold."
    )


def _make_issue(severity: str, field: str | None, message: str, *, label: str | None = None, page: str | None = None) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "severity": severity,
        "message": message,
    }
    if field:
        issue["field"] = field
    if label:
        issue["label"] = label
    if page:
        issue["page"] = page
    return issue


def _group_by_field(issues: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for issue in issues:
        field = issue.get("field")
        if not field:
            continue
        grouped.setdefault(field, []).append(issue["message"])
    return grouped


def _build_report(errors: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> dict[str, Any]:
    if errors:
        summary = f"Fix {len(errors)} validation error{'s' if len(errors) != 1 else ''} before launch."
    elif warnings:
        summary = f"Validation passed with {len(warnings)} warning{'s' if len(warnings) != 1 else ''}."
    else:
        summary = "Validation passed."

    return {
        "ok": not errors,
        "summary": summary,
        "errors": errors,
        "warnings": warnings,
        "field_errors": _group_by_field(errors),
        "field_warnings": _group_by_field(warnings),
    }


def _dataset_source_label(index: int) -> str:
    return f"Dataset #{index + 1}"


def _validate_dataset_entry(entry: DatasetEntry, index: int, *, errors: list[dict[str, Any]], warnings: list[dict[str, Any]]) -> None:
    field_base = f"dataset.datasets[{index}]"
    label = _dataset_source_label(index)

    has_directory = _has_text(entry.directory)
    has_jsonl = _has_text(entry.jsonl_file)

    if not has_directory and not has_jsonl:
        errors.append(
            _make_issue(
                "error",
                f"{field_base}.source",
                f"{label}: fill either the media directory or the JSONL file.",
                label=label,
                page="dataset",
            )
        )

    if has_directory and has_jsonl:
        warnings.append(
            _make_issue(
                "warning",
                f"{field_base}.jsonl_file",
                f"{label}: both directory and JSONL file are set. JSONL will take precedence.",
                label=label,
                page="dataset",
            )
        )

    if entry.type != "audio" and entry.reference_frames is not None and entry.reference_frames < 1:
        errors.append(
            _make_issue(
                "error",
                f"{field_base}.reference_frames",
                f"{label}: reference frames must be at least 1.",
                label=label,
                page="dataset",
            )
        )


def _has_training_gemma_source(config: ProjectConfig) -> bool:
    t = config.training
    return any(
        _has_text(value)
        for value in (
            t.gemma_root,
            t.gemma_safetensors,
            config.default_gemma_root,
            config.default_gemma_safetensors,
        )
    )


def _has_inference_gemma_source(config: ProjectConfig) -> bool:
    i = config.inference
    return any(
        _has_text(value)
        for value in (
            i.gemma_root,
            i.gemma_safetensors,
            config.default_gemma_root,
            config.default_gemma_safetensors,
        )
    )


def _has_cache_text_gemma_source(config: ProjectConfig) -> bool:
    c = config.caching
    return any(
        _has_text(value)
        for value in (
            c.gemma_root,
            c.gemma_safetensors,
            config.default_gemma_root,
            config.default_gemma_safetensors,
        )
    )


def _has_training_checkpoint(config: ProjectConfig) -> bool:
    return _has_text(config.training.ltx2_checkpoint) or _has_text(config.default_ltx2_checkpoint)


def _has_inference_checkpoint(config: ProjectConfig) -> bool:
    return _has_text(config.inference.ltx2_checkpoint) or _has_text(config.default_ltx2_checkpoint)


def _has_cache_text_checkpoint(config: ProjectConfig) -> bool:
    return _has_text(config.caching.ltx2_checkpoint) or _has_text(config.default_ltx2_checkpoint)


def _effective_gemma_safetensors(raw_path: str | None, default_path: str | None) -> str:
    if _has_text(raw_path):
        return str(raw_path).strip()
    if _has_text(default_path):
        return str(default_path).strip()
    return ""


def _validate_gemma_quantization_combo(
    *,
    errors: list[dict[str, Any]],
    gemma_load_in_8bit: bool,
    gemma_load_in_4bit: bool,
    gemma_safetensors: str,
    field_prefix: str,
    page: str,
) -> None:
    if gemma_load_in_8bit and gemma_load_in_4bit:
        message = "Gemma 8-bit and Gemma 4-bit cannot be enabled together."
        errors.append(_make_issue("error", f"{field_prefix}.gemma_load_in_8bit", message, label="Gemma 8b", page=page))
        errors.append(_make_issue("error", f"{field_prefix}.gemma_load_in_4bit", message, label="Gemma 4b", page=page))

    if gemma_safetensors and (gemma_load_in_8bit or gemma_load_in_4bit):
        errors.append(
            _make_issue(
                "error",
                f"{field_prefix}.gemma_safetensors",
                "Gemma Safetensors cannot be combined with Gemma 8-bit or 4-bit loading.",
                label="Gemma Safetensors",
                page=page,
            )
        )


def _has_inline_training_sample_prompts(config: ProjectConfig) -> bool:
    return _has_text(config.training.sample_prompts_text)


def _has_any_sample_prompts(config: ProjectConfig) -> bool:
    return any(
        (
            _has_text(config.caching.sample_prompts),
            _has_text(config.training.sample_prompts),
            _has_inline_training_sample_prompts(config),
        )
    )


def _resolve_project_path(config: ProjectConfig, raw_path: str | None) -> Path | None:
    if not _has_text(raw_path):
        return None
    path = Path(str(raw_path).strip())
    if path.is_absolute() or not _has_text(config.project_dir):
        return path
    return Path(config.project_dir) / path


def validate_training_config(config: ProjectConfig) -> dict[str, Any]:
    """Validate GUI training config before launch."""
    t = config.training
    c = config.caching
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    effective_gemma_safetensors = _effective_gemma_safetensors(t.gemma_safetensors, config.default_gemma_safetensors)

    if not _has_training_checkpoint(config):
        errors.append(
            _make_issue(
                "error",
                "training.ltx2_checkpoint",
                "LTX-2 Checkpoint is required.",
                label="LTX-2 Checkpoint",
                page="training",
            )
        )

    if t.log_with == "tensorboard" and not _has_text(t.logging_dir):
        errors.append(
            _make_issue(
                "error",
                "training.logging_dir",
                "Log Dir is required when Logger is set to TensorBoard.",
                label="Log Dir",
                page="training",
            )
        )

    _validate_gemma_quantization_combo(
        errors=errors,
        gemma_load_in_8bit=t.gemma_load_in_8bit,
        gemma_load_in_4bit=t.gemma_load_in_4bit,
        gemma_safetensors=effective_gemma_safetensors,
        field_prefix="training",
        page="training",
    )

    if t.full_fp16 and t.full_bf16:
        message = "Full FP16 and Full BF16 cannot be enabled together."
        errors.append(_make_issue("error", "training.full_fp16", message, label="Full FP16", page="training"))
        errors.append(_make_issue("error", "training.full_bf16", message, label="Full BF16", page="training"))

    if t.fp8_w8a8 and not t.fp8_scaled:
        message = "W8A8 requires FP8 Scaled."
        errors.append(_make_issue("error", "training.fp8_w8a8", message, label="W8A8", page="training"))
        errors.append(_make_issue("error", "training.fp8_scaled", message, label="FP8 Scaled", page="training"))

    if t.fp8_w8a8 and not t.fp8_base:
        message = "W8A8 requires FP8 Base."
        errors.append(_make_issue("error", "training.fp8_w8a8", message, label="W8A8", page="training"))
        errors.append(_make_issue("error", "training.fp8_base", message, label="FP8 Base", page="training"))

    if t.loftq_init and not t.nf4_base:
        message = "LoftQ Init requires NF4 Base."
        errors.append(_make_issue("error", "training.loftq_init", message, label="LoftQ Init", page="training"))
        errors.append(_make_issue("error", "training.nf4_base", message, label="NF4 Base", page="training"))

    dora_mode_count = int(bool(getattr(t, "use_dora", False))) + int(bool(getattr(t, "use_dokr", False))) + int(bool(getattr(t, "use_dora_oft", False)))
    if dora_mode_count > 1:
        message = "Use either DoRA, DokR, or DoRA-OFT, not multiple DoRA modes."
        errors.append(_make_issue("error", "training.use_dora", message, label="DoRA", page="training"))
        errors.append(_make_issue("error", "training.use_dokr", message, label="DokR", page="training"))
        errors.append(_make_issue("error", "training.use_dora_oft", message, label="DoRA-OFT", page="training"))

    if t.use_lokr and (t.use_dora or t.use_dokr or t.use_dora_oft):
        message = "Use either LoKr, DoRA, DokR, or DoRA-OFT, not multiple network modes."
        errors.append(_make_issue("error", "training.use_lokr", message, label="LoKr", page="training"))
        if t.use_dora:
            errors.append(_make_issue("error", "training.use_dora", message, label="DoRA", page="training"))
        if t.use_dokr:
            errors.append(_make_issue("error", "training.use_dokr", message, label="DokR", page="training"))
        if t.use_dora_oft:
            errors.append(_make_issue("error", "training.use_dora_oft", message, label="DoRA-OFT", page="training"))

    if t.use_dora_oft and not t.scaled_oft:
        message = "DoRA-OFT requires Scaled OFT."
        errors.append(_make_issue("error", "training.use_dora_oft", message, label="DoRA-OFT", page="training"))
        errors.append(_make_issue("error", "training.scaled_oft", message, label="Scaled OFT", page="training"))

    if t.use_dora_oft and t.oft_block_size is not None and int(t.oft_block_size) <= 0:
        errors.append(
            _make_issue(
                "error",
                "training.oft_block_size",
                "OFT Block Size must be a positive integer.",
                label="OFT Block Size",
                page="training",
            )
        )

    raw_lokr_factor = getattr(t, "lokr_factor", None)
    if raw_lokr_factor is not None:
        try:
            parsed_lokr_factor = int(raw_lokr_factor)
        except (TypeError, ValueError):
            parsed_lokr_factor = 0
        if parsed_lokr_factor <= 0 and parsed_lokr_factor != -1:
            errors.append(
                _make_issue(
                    "error",
                    "training.lokr_factor",
                    "LoKr Factor must be -1 for balanced factorization, or a positive integer.",
                    label="LoKr Factor",
                    page="training",
                )
            )

    lokr_dim_warning = _get_lokr_dim_warning(t)
    if lokr_dim_warning is not None:
        warnings.append(
            _make_issue(
                "warning",
                "training.network_dim",
                lokr_dim_warning,
                label="Dim",
                page="training",
            )
        )

    if t.awq_calibration and not t.nf4_base:
        message = "AWQ Calibration requires NF4 Base."
        errors.append(_make_issue("error", "training.awq_calibration", message, label="AWQ Calibration", page="training"))
        errors.append(_make_issue("error", "training.nf4_base", message, label="NF4 Base", page="training"))

    if t.tread and (t.ltx2_mode == "audio" or t.ltx2_audio_only_model):
        errors.append(
            _make_issue(
                "error",
                "training.tread",
                "TREAD requires a video-enabled LTX path and cannot be used with audio-only training.",
                label="TREAD",
                page="techniques",
            )
        )

    if t.differential_guidance:
        try:
            differential_guidance_scale = float(t.differential_guidance_scale)
        except (TypeError, ValueError):
            differential_guidance_scale = float("nan")
        if not math.isfinite(differential_guidance_scale):
            errors.append(
                _make_issue(
                    "error",
                    "training.differential_guidance_scale",
                    "Differential guidance scale must be a finite number.",
                    label="Differential Guidance Scale",
                    page="techniques",
                )
            )
        if t.ltx2_mode == "audio" or t.ltx2_audio_only_model:
            errors.append(
                _make_issue(
                    "error",
                    "training.differential_guidance",
                    "Differential guidance requires a video/main prediction loss and cannot be used with audio-only training.",
                    label="Differential Guidance",
                    page="techniques",
                )
            )

    if t.crepa_similarity_threshold is not None and not 0.0 <= t.crepa_similarity_threshold <= 0.99:
        errors.append(
            _make_issue(
                "error",
                "training.crepa_similarity_threshold",
                "CREPA similarity threshold must be between 0 and 0.99, or blank to disable.",
                label="CREPA Similarity Threshold",
                page="techniques",
            )
        )

    if t.crepa_lambda < 0.0:
        errors.append(
            _make_issue(
                "error",
                "training.crepa_lambda",
                "CREPA lambda must be >= 0.",
                label="CREPA Lambda",
                page="techniques",
            )
        )

    if t.crepa_lambda_end < 0.0:
        errors.append(
            _make_issue(
                "error",
                "training.crepa_lambda_end",
                "CREPA lambda end must be >= 0.",
                label="CREPA Lambda End",
                page="techniques",
            )
        )

    if t.crepa_warmup_steps < 0:
        errors.append(
            _make_issue(
                "error",
                "training.crepa_warmup_steps",
                "CREPA warmup steps must be >= 0.",
                label="CREPA Warmup Steps",
                page="techniques",
            )
        )

    if t.crepa_decay_steps < 0:
        errors.append(
            _make_issue(
                "error",
                "training.crepa_decay_steps",
                "CREPA decay steps must be >= 0.",
                label="CREPA Decay Steps",
                page="techniques",
            )
        )

    if not 0.0 <= t.crepa_similarity_ema_decay < 1.0:
        errors.append(
            _make_issue(
                "error",
                "training.crepa_similarity_ema_decay",
                "CREPA similarity EMA decay must be >= 0 and < 1.",
                label="CREPA Similarity EMA",
                page="techniques",
            )
        )

    if not t.save_every_n_steps and not t.save_every_n_epochs:
        warnings.append(
            _make_issue(
                "warning",
                "training.save_every_n_epochs",
                "No checkpoint save frequency is set. Set Save Every N Epochs or Save Every N Steps to make checkpoint output explicit.",
                label="Checkpoint Save Frequency",
                page="training",
            )
        )

    if t.sample_two_stage and not _has_text(t.spatial_upsampler_path):
        errors.append(
            _make_issue(
                "error",
                "training.spatial_upsampler_path",
                "Upsampler Path is required when Two-Stage sampling is enabled.",
                label="Upsampler Path",
                page="training",
            )
        )

    sampling_enabled = bool(t.sample_at_first or t.sample_every_n_steps or t.sample_every_n_epochs)
    has_inline_sample_prompts = _has_inline_training_sample_prompts(config)
    has_sample_prompt_source = _has_text(t.sample_prompts) or has_inline_sample_prompts

    if sampling_enabled and not has_sample_prompt_source:
        errors.append(
            _make_issue(
                "error",
                "training.sample_prompts",
                "Define sample prompts on the Samples page or set Sample Prompts File when training sampling is enabled.",
                label="Sample Prompts",
                page="training",
            )
        )

    if t.use_precached_sample_prompts and not has_sample_prompt_source:
        errors.append(
            _make_issue(
                "error",
                "training.sample_prompts",
                "Define sample prompts on the Samples page or set Sample Prompts File when Precached sample prompts is enabled.",
                label="Sample Prompts",
                page="training",
            )
        )

    sample_prompts_path = _resolve_project_path(config, t.sample_prompts)
    if sample_prompts_path is not None and not sample_prompts_path.exists():
        errors.append(
            _make_issue(
                "error",
                "training.sample_prompts",
                f"Sample Prompts file not found: {sample_prompts_path}",
                label="Sample Prompts",
                page="training",
            )
        )

    if sampling_enabled and not t.use_precached_sample_prompts and not _has_training_gemma_source(config):
        message = "Gemma Root or Gemma Safetensors is required for non-precached sample prompts."
        errors.append(_make_issue("error", "training.gemma_root", message, label="Gemma Root", page="training"))
        errors.append(_make_issue("error", "training.gemma_safetensors", message, label="Gemma Safetensors", page="training"))

    if has_sample_prompt_source and not sampling_enabled:
        warnings.append(
            _make_issue(
                "warning",
                "training.sample_prompts",
                "Sample Prompts are defined, but no sampling trigger is enabled.",
                label="Sample Prompts",
                page="training",
            )
        )

    if t.validate_every_n_steps or t.validate_every_n_epochs:
        if not config.dataset.validation_datasets:
            warnings.append(
                _make_issue(
                    "warning",
                    "dataset.validation_datasets",
                    "Validation frequency is set, but no validation datasets are configured.",
                    label="Validation Datasets",
                    page="dataset",
                )
            )

    if _has_text(t.dataset_manifest):
        if config.dataset.datasets:
            warnings.append(
                _make_issue(
                    "warning",
                    "training.dataset_manifest",
                    "Dataset Manifest is set, so training datasets from the Dataset page will be ignored.",
                    label="Dataset Manifest",
                    page="training",
                )
            )
    else:
        if not config.dataset.datasets:
            errors.append(
                _make_issue(
                    "error",
                    "dataset.datasets",
                    "Add at least one training dataset or set Dataset Manifest.",
                    label="Training Datasets",
                    page="dataset",
                )
            )
        for index, entry in enumerate(config.dataset.datasets):
            _validate_dataset_entry(entry, index, errors=errors, warnings=warnings)

    return _build_report(errors, warnings)


def _validate_sample_prompt_path(config: ProjectConfig, raw_path: str | None, *, errors: list[dict[str, Any]], field: str, label: str, page: str) -> None:
    sample_prompts_path = _resolve_project_path(config, raw_path)
    if sample_prompts_path is not None and not sample_prompts_path.exists():
        errors.append(
            _make_issue(
                "error",
                field,
                f"{label} file not found: {sample_prompts_path}",
                label=label,
                page=page,
            )
        )


def validate_cache_latents_config(config: ProjectConfig) -> dict[str, Any]:
    c = config.caching
    t = config.training
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if c.precache_sample_latents and not _has_any_sample_prompts(config):
        errors.append(
            _make_issue(
                "error",
                "caching.sample_prompts",
                "Define sample prompts on the Samples page or set an external prompts file before precaching sample latents.",
                label="Sample Prompts",
                page="caching",
            )
        )

    _validate_sample_prompt_path(
        config,
        c.sample_prompts,
        errors=errors,
        field="caching.sample_prompts",
        label="Caching Sample Prompts",
        page="caching",
    )
    _validate_sample_prompt_path(
        config,
        config.training.sample_prompts,
        errors=errors,
        field="training.sample_prompts",
        label="Sample Prompts",
        page="training",
    )

    return _build_report(errors, warnings)


def validate_cache_text_config(config: ProjectConfig) -> dict[str, Any]:
    c = config.caching
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    effective_gemma_safetensors = _effective_gemma_safetensors(c.gemma_safetensors, config.default_gemma_safetensors)

    if not _has_cache_text_checkpoint(config):
        errors.append(
            _make_issue(
                "error",
                "caching.ltx2_checkpoint",
                "LTX-2 Checkpoint is required.",
                label="LTX-2 Checkpoint",
                page="caching",
            )
        )
    if not _has_cache_text_gemma_source(config):
        message = "Gemma Root or Gemma Safetensors is required for text encoder caching."
        errors.append(_make_issue("error", "caching.gemma_root", message, label="Gemma Root", page="caching"))
        errors.append(_make_issue("error", "caching.gemma_safetensors", message, label="Gemma Safetensors", page="caching"))

    _validate_gemma_quantization_combo(
        errors=errors,
        gemma_load_in_8bit=c.gemma_load_in_8bit,
        gemma_load_in_4bit=c.gemma_load_in_4bit,
        gemma_safetensors=effective_gemma_safetensors,
        field_prefix="caching",
        page="caching",
    )

    if c.precache_sample_prompts and not _has_any_sample_prompts(config):
        errors.append(
            _make_issue(
                "error",
                "caching.sample_prompts",
                "Define sample prompts on the Samples page or set an external prompts file before precaching sample prompts.",
                label="Sample Prompts",
                page="caching",
            )
        )

    _validate_sample_prompt_path(
        config,
        c.sample_prompts,
        errors=errors,
        field="caching.sample_prompts",
        label="Caching Sample Prompts",
        page="caching",
    )
    _validate_sample_prompt_path(
        config,
        config.training.sample_prompts,
        errors=errors,
        field="training.sample_prompts",
        label="Sample Prompts",
        page="training",
    )

    return _build_report(errors, warnings)


def validate_inference_config(config: ProjectConfig) -> dict[str, Any]:
    i = config.inference
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    effective_gemma_safetensors = _effective_gemma_safetensors(i.gemma_safetensors, config.default_gemma_safetensors)

    if not _has_inference_checkpoint(config):
        errors.append(
            _make_issue(
                "error",
                "inference.ltx2_checkpoint",
                "LTX-2 Checkpoint is required.",
                label="LTX-2 Checkpoint",
                page="inference",
            )
        )

    if not _has_text(i.prompt) and not _has_text(i.from_file):
        message = "Prompt or Sample Prompts File is required."
        errors.append(_make_issue("error", "inference.prompt", message, label="Prompt", page="inference"))
        errors.append(_make_issue("error", "inference.from_file", message, label="Sample Prompts File", page="inference"))

    if (i.sample_two_stage or i.sampling_preset == "distilled_two_stage") and not _has_text(i.spatial_upsampler_path):
        errors.append(
            _make_issue(
                "error",
                "inference.spatial_upsampler_path",
                "Upsampler Path is required when Two-Stage sampling is enabled.",
                label="Upsampler Path",
                page="inference",
            )
        )

    if i.use_precached_sample_prompts and not _has_text(i.from_file):
        errors.append(
            _make_issue(
                "error",
                "inference.from_file",
                "Sample Prompts File is required when Precached sample prompts is enabled.",
                label="Sample Prompts File",
                page="inference",
            )
        )

    sample_prompts_cache_path = _resolve_project_path(config, i.sample_prompts_cache)
    if i.use_precached_sample_prompts and sample_prompts_cache_path is not None and not sample_prompts_cache_path.exists():
        errors.append(
            _make_issue(
                "error",
                "inference.sample_prompts_cache",
                f"Sample Prompts Cache file not found: {sample_prompts_cache_path}",
                label="Sample Prompts Cache",
                page="inference",
            )
        )

    if not i.use_precached_sample_prompts and not _has_inference_gemma_source(config):
        message = "Gemma Root or Gemma Safetensors is required for inference."
        errors.append(_make_issue("error", "inference.gemma_root", message, label="Gemma Root", page="inference"))
        errors.append(_make_issue("error", "inference.gemma_safetensors", message, label="Gemma Safetensors", page="inference"))

    _validate_gemma_quantization_combo(
        errors=errors,
        gemma_load_in_8bit=i.gemma_load_in_8bit,
        gemma_load_in_4bit=i.gemma_load_in_4bit,
        gemma_safetensors=effective_gemma_safetensors,
        field_prefix="inference",
        page="inference",
    )

    _validate_sample_prompt_path(
        config,
        i.from_file,
        errors=errors,
        field="inference.from_file",
        label="Sample Prompts File",
        page="inference",
    )

    return _build_report(errors, warnings)


def validate_process_config(proc_type: str, config: ProjectConfig) -> dict[str, Any]:
    if proc_type == "training":
        return validate_training_config(config)
    if proc_type == "cache_latents":
        return validate_cache_latents_config(config)
    if proc_type == "cache_text":
        return validate_cache_text_config(config)
    if proc_type == "inference":
        return validate_inference_config(config)
    return _build_report([], [])
