"""Shared GUI validation rules for process launch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from musubi_tuner.gui_dashboard.project_schema import DatasetEntry, ProjectConfig


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())


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
        summary = f"Fix {len(errors)} validation error{'s' if len(errors) != 1 else ''} before training."
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


def _has_training_checkpoint(config: ProjectConfig) -> bool:
    return _has_text(config.training.ltx2_checkpoint) or _has_text(config.default_ltx2_checkpoint)


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
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

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

    if t.gemma_load_in_8bit and t.gemma_load_in_4bit:
        message = "Gemma 8-bit and Gemma 4-bit cannot be enabled together."
        errors.append(_make_issue("error", "training.gemma_load_in_8bit", message, label="Gemma 8b", page="training"))
        errors.append(_make_issue("error", "training.gemma_load_in_4bit", message, label="Gemma 4b", page="training"))

    if _has_text(t.gemma_safetensors) and (t.gemma_load_in_8bit or t.gemma_load_in_4bit):
        errors.append(
            _make_issue(
                "error",
                "training.gemma_safetensors",
                "Gemma Safetensors cannot be combined with Gemma 8-bit or 4-bit loading.",
                label="Gemma Safetensors",
                page="training",
            )
        )

    if t.full_fp16 and t.full_bf16:
        message = "Full FP16 and Full BF16 cannot be enabled together."
        errors.append(_make_issue("error", "training.full_fp16", message, label="Full FP16", page="training"))
        errors.append(_make_issue("error", "training.full_bf16", message, label="Full BF16", page="training"))

    if t.fp8_w8a8 and not t.fp8_scaled:
        message = "W8A8 requires FP8 Scaled."
        errors.append(_make_issue("error", "training.fp8_w8a8", message, label="W8A8", page="training"))
        errors.append(_make_issue("error", "training.fp8_scaled", message, label="FP8 Scaled", page="training"))

    if t.loftq_init and not t.nf4_base:
        message = "LoftQ Init requires NF4 Base."
        errors.append(_make_issue("error", "training.loftq_init", message, label="LoftQ Init", page="training"))
        errors.append(_make_issue("error", "training.nf4_base", message, label="NF4 Base", page="training"))

    if t.awq_calibration and not t.nf4_base:
        message = "AWQ Calibration requires NF4 Base."
        errors.append(_make_issue("error", "training.awq_calibration", message, label="AWQ Calibration", page="training"))
        errors.append(_make_issue("error", "training.nf4_base", message, label="NF4 Base", page="training"))

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


def validate_process_config(proc_type: str, config: ProjectConfig) -> dict[str, Any]:
    if proc_type == "training":
        return validate_training_config(config)
    if proc_type == "cache_latents":
        return validate_cache_latents_config(config)
    if proc_type == "cache_text":
        return validate_cache_text_config(config)
    return _build_report([], [])
