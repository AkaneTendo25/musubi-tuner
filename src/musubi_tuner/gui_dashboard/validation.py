"""Shared GUI validation rules for process launch."""

from __future__ import annotations

import math
import re
import shlex
from pathlib import Path
from typing import Any

from musubi_tuner.gui_dashboard.cli_defaults import get_ltx2_training_network_module_default
from musubi_tuner.gui_dashboard.project_schema import DatasetEntry, ProjectConfig

try:
    from musubi_tuner.ltx2_av_cross_grad_surgery import parse_av_cross_grad_surgery_args
except ModuleNotFoundError:  # LTX-2 is optional on the minimax-h3 branch.

    def parse_av_cross_grad_surgery_args(*args, **kwargs):
        del args, kwargs
        return None


from musubi_tuner.modules.group_lr_scheduler import parse_group_lr_scheduler_args
from musubi_tuner.tread import default_ltx_tread_route, parse_tread_args


def _parse_reward_spec_for_validation(spec: str, reward_plugins: str = "") -> tuple[dict[str, float] | None, str | None]:
    """Parse a reward spec, returning ``(weights, error)``.

    Defers the ``ltx2_rewards`` import (which pulls in the zoo registry) into the function
    so importing this validation module stays lightweight. ``reward_plugins`` (whitespace-
    separated .py paths) is loaded first so plugin-defined reward names resolve, mirroring
    the drivers. ``parse_reward_spec`` raises ``KeyError`` for an unregistered reward name
    and ``ValueError`` for a malformed weight.
    """
    try:
        from musubi_tuner.ltx2_rewards import load_reward_plugins, parse_reward_spec, registered_rewards
    except Exception as exc:  # pragma: no cover - registry import should always succeed
        return None, f"could not import the reward registry: {exc}"
    for plugin in (reward_plugins or "").split():
        try:
            load_reward_plugins([plugin])
        except FileNotFoundError:
            return None, f"reward plugin file not found: {plugin}"
        except ValueError:
            pass  # already registered (validation may run repeatedly in one process)
        except Exception as exc:
            return None, f"reward plugin {plugin} failed to load: {exc}"
    try:
        return parse_reward_spec(spec), None
    except KeyError as exc:
        registered = ", ".join(registered_rewards())
        return None, f"{exc.args[0] if exc.args else exc}. Registered rewards: {registered}"
    except (ValueError, TypeError) as exc:
        return None, f"reward spec is malformed: {exc}"


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())


def _split_cli_args(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        return shlex.split(raw, posix=False)
    except ValueError:
        return []


def _accelerate_num_processes(raw: str | None) -> int | None:
    args = _split_cli_args(raw)
    for index, arg in enumerate(args):
        if arg == "--num_processes" and index + 1 < len(args):
            try:
                return int(args[index + 1])
            except ValueError:
                return None
        if arg.startswith("--num_processes="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _parse_csv_ints(raw: str | None) -> list[int] | None:
    if not _has_text(raw):
        return []
    values: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            return None
    return values


def _parse_remote_stage_specs(raw: str | None) -> tuple[list[tuple[str, int, int, int]], str | None]:
    if not _has_text(raw):
        return [], None
    specs: list[tuple[str, int, int, int]] = []
    previous_end: int | None = None
    for index, entry in enumerate(str(raw).split(";")):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.rsplit(":", 3)
        if len(parts) != 4:
            return [], "Remote Stage Specs entries must use host:port:start:end."
        host, port_raw, start_raw, end_raw = (part.strip() for part in parts)
        if not host:
            return [], f"Remote stage #{index + 1} host must not be empty."
        try:
            port = int(port_raw)
            start = int(start_raw)
            end = int(end_raw)
        except ValueError:
            return [], f"Remote stage #{index + 1} port/start/end must be integers."
        if not (0 < port < 65536):
            return [], f"Remote stage #{index + 1} port must be in 1..65535."
        if start < 0:
            return [], f"Remote stage #{index + 1} start block must be >= 0."
        if end <= start:
            return [], f"Remote stage #{index + 1} end block must be greater than start block."
        if previous_end is not None and start != previous_end:
            return [], "Remote Stage Specs must be contiguous."
        previous_end = end
        specs.append((host, port, start, end))
    if not specs:
        return [], "Remote Stage Specs must contain at least one stage."
    return specs, None


def _make_issue(
    severity: str, field: str | None, message: str, *, label: str | None = None, page: str | None = None
) -> dict[str, Any]:
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


def _validate_dataset_entry(
    entry: DatasetEntry,
    index: int,
    *,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    collection: str = "datasets",
) -> None:
    field_base = f"dataset.{collection}[{index}]"
    label = ("Validation " if collection == "validation_datasets" else "") + _dataset_source_label(index)

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


def _validate_h3_dataset_entry(
    entry: DatasetEntry,
    index: int,
    *,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    collection: str = "datasets",
) -> None:
    _validate_dataset_entry(
        entry,
        index,
        errors=errors,
        warnings=warnings,
        collection=collection,
    )
    field_base = f"dataset.{collection}[{index}]"
    label = ("Validation " if collection == "validation_datasets" else "") + _dataset_source_label(index)
    modalities = [part.strip().lower() for part in entry.control_modalities.replace(",", ";").split(";") if part.strip()]
    invalid_modalities = sorted(set(modalities) - {"av", "video", "audio"})
    if invalid_modalities:
        errors.append(
            _make_issue(
                "error",
                f"{field_base}.control_modalities",
                f"{label}: unknown H3 reference modality: {', '.join(invalid_modalities)}.",
                label=label,
                page="dataset",
            )
        )
    if entry.control_modality and modalities:
        errors.append(
            _make_issue(
                "error",
                f"{field_base}.control_modalities",
                f"{label}: choose one fixed reference modality or an ordered modality list, not both.",
                label=label,
                page="dataset",
            )
        )
    probabilities = (
        entry.control_modality_probability_av,
        entry.control_modality_probability_video,
        entry.control_modality_probability_audio,
    )
    if any(value is not None for value in probabilities):
        if any(value is None for value in probabilities):
            errors.append(
                _make_issue(
                    "error",
                    f"{field_base}.control_modality_probability_av",
                    f"{label}: AV, video, and audio reference probabilities must all be set.",
                    label=label,
                    page="dataset",
                )
            )
        else:
            numeric = tuple(float(value) for value in probabilities)
            if any(not math.isfinite(value) or value < 0 for value in numeric) or not math.isclose(
                sum(numeric), 1.0, rel_tol=0.0, abs_tol=1e-6
            ):
                errors.append(
                    _make_issue(
                        "error",
                        f"{field_base}.control_modality_probability_av",
                        f"{label}: H3 reference probabilities must be non-negative and sum to 1.",
                        label=label,
                        page="dataset",
                    )
                )
        if entry.control_modality or modalities:
            errors.append(
                _make_issue(
                    "error",
                    f"{field_base}.control_modality_probability_av",
                    f"{label}: stochastic reference probabilities cannot be combined with a fixed modality policy.",
                    label=label,
                    page="dataset",
                )
            )
    if entry.control_directory and (entry.control_video_directory or entry.control_audio_directory):
        errors.append(
            _make_issue(
                "error",
                f"{field_base}.control_directory",
                f"{label}: use the legacy combined control directory or paired video/audio directories, not both.",
                label=label,
                page="dataset",
            )
        )
    if bool(entry.control_video_directory) != bool(entry.control_audio_directory):
        errors.append(
            _make_issue(
                "error",
                f"{field_base}.control_video_directory",
                f"{label}: paired H3 reference video and audio directories must be specified together.",
                label=label,
                page="dataset",
            )
        )
    if entry.h3_image_frame_count is not None and (entry.h3_image_frame_count < 5 or (entry.h3_image_frame_count - 5) % 17 != 0):
        errors.append(
            _make_issue(
                "error",
                f"{field_base}.h3_image_frame_count",
                f"{label}: H3 conditioned-image frame count must satisfy frame_count % 17 == 5.",
                label=label,
                page="dataset",
            )
        )
    if entry.type == "video" and (entry.target_frames < 5 or (entry.target_frames - 5) % 17 != 0):
        errors.append(
            _make_issue(
                "error",
                f"{field_base}.target_frames",
                f"{label}: H3 video target frames must satisfy frame_count % 17 == 5 (for example 124).",
                label=label,
                page="dataset",
            )
        )
    if entry.type == "audio" and entry.h3_target_mode != "audio":
        errors.append(
            _make_issue(
                "error",
                f"{field_base}.h3_target_mode",
                f"{label}: an audio dataset requires Audio only target modalities.",
                label=label,
                page="dataset",
            )
        )


def _h3_required_vaes(config: ProjectConfig) -> tuple[bool, bool]:
    requires_video = False
    requires_audio = False
    rows = list(config.dataset.datasets or []) + list(config.dataset.validation_datasets or [])
    for entry in rows:
        target_mode = "video" if entry.type == "image" else entry.h3_target_mode
        requires_video |= target_mode != "audio"
        requires_audio |= entry.type in {"video", "audio"} and target_mode != "video"

        # Paired directories always describe one video stream plus its explicit
        # audio stream.  A combined reference directory can contain images,
        # videos, or audio, so use its modality policy when one is provided and
        # conservatively require both decoders when its contents are unknown.
        if entry.control_video_directory or entry.control_audio_directory:
            requires_video = True
            requires_audio = True
        has_combined_references = bool(entry.control_directory or entry.extra_control_directories)
        if has_combined_references:
            fixed_modes = [entry.control_modality] if entry.control_modality else []
            fixed_modes.extend(
                value.strip().lower() for value in re.split(r"[;,]", entry.control_modalities or "") if value.strip()
            )
            probabilities = (
                entry.control_modality_probability_av,
                entry.control_modality_probability_video,
                entry.control_modality_probability_audio,
            )
            if fixed_modes:
                requires_video |= any(mode in {"av", "video"} for mode in fixed_modes)
                requires_audio |= any(mode in {"av", "audio"} for mode in fixed_modes)
            elif any(value is not None for value in probabilities):
                av_probability, video_probability, audio_probability = probabilities
                requires_video |= bool((av_probability or 0) > 0 or (video_probability or 0) > 0)
                requires_audio |= bool((av_probability or 0) > 0 or (audio_probability or 0) > 0)
            else:
                requires_video = True
                requires_audio = True

        # Uncached reference sources need their respective encoder. Cached
        # references themselves do not create an additional VAE requirement.
        requires_audio |= any(
            _has_text(value)
            for value in (
                entry.reference_audio_directory,
                entry.extra_reference_audio_directories,
            )
        )
    return requires_video, requires_audio


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
    if config.training.model_type == "minimax_h3":
        return _has_text(config.training.h3_model)
    return _has_text(config.training.ltx2_checkpoint) or _has_text(config.default_ltx2_checkpoint)


def _has_full_finetune_checkpoint(config: ProjectConfig) -> bool:
    return _has_text(config.full_finetune.ltx2_checkpoint) or _has_text(config.default_ltx2_checkpoint)


def _has_inference_checkpoint(config: ProjectConfig) -> bool:
    if config.inference.model_type == "minimax_h3":
        return _has_text(config.inference.h3_model)
    return _has_text(config.inference.ltx2_checkpoint) or _has_text(config.default_ltx2_checkpoint)


def _has_cache_text_checkpoint(config: ProjectConfig) -> bool:
    if config.caching.model_type == "minimax_h3":
        return _has_text(config.caching.h3_text_encoder)
    return _has_text(config.caching.ltx2_checkpoint) or _has_text(config.default_ltx2_checkpoint)


def _effective_gemma_safetensors(raw_path: str | None, default_path: str | None, explicit_gemma_root: str | None = None) -> str:
    if _has_text(raw_path):
        return str(raw_path).strip()
    # If the user explicitly set gemma_root to a valid directory, suppress the
    # default_gemma_safetensors fallback so gemma_root takes priority.
    if _has_text(explicit_gemma_root) and Path(str(explicit_gemma_root).strip()).is_dir():
        return ""
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


def _default_dataset_cache_directory(config: ProjectConfig) -> str:
    datasets = list(config.dataset.datasets or []) + list(config.dataset.validation_datasets or [])
    for entry in datasets:
        if _has_text(getattr(entry, "cache_directory", "")):
            return str(entry.cache_directory).strip()
        if _has_text(getattr(entry, "directory", "")):
            return str(Path(str(entry.directory).strip()) / "cache")
        if _has_text(getattr(entry, "jsonl_file", "")):
            return str(Path(str(entry.jsonl_file).strip()).parent / "cache")
    return ""


def _effective_cache_preview_input(config: ProjectConfig) -> str:
    return config.caching.cache_preview_input or _default_dataset_cache_directory(config)


def _validate_generated_av_metrics(
    training,
    *,
    page: str,
    errors: list[dict[str, Any]],
) -> None:
    fad_enabled = bool(getattr(training, "audio_metrics_clap_fad", False))
    desync_enabled = bool(getattr(training, "audio_metrics_av_desync", False))
    if not (fad_enabled or desync_enabled):
        return

    if not training.audio_metrics:
        errors.append(
            _make_issue(
                "error",
                f"{page}.audio_metrics",
                "Audio Metrics must be enabled for generated AV validation.",
                label="Audio Metrics",
                page=page,
            )
        )
    has_sample_manifest = _has_text(getattr(training, "sample_prompts", "")) or _has_text(
        getattr(training, "sample_prompts_text", "")
    )
    if not has_sample_manifest:
        errors.append(
            _make_issue(
                "error",
                f"{page}.sample_prompts",
                "A deterministic sample manifest is required for generated AV validation.",
                label="Sample Prompts",
                page=page,
            )
        )
    if fad_enabled and int(getattr(training, "audio_metrics_clap_fad_min_samples", 513)) < 2:
        errors.append(
            _make_issue(
                "error",
                f"{page}.audio_metrics_clap_fad_min_samples",
                "FAD-CLAP minimum samples must be at least 2; the runtime also enforces embedding_dimension + 1.",
                label="FAD-CLAP Minimum Samples",
                page=page,
            )
        )
    if desync_enabled:
        checkpoint = str(getattr(training, "audio_metrics_av_desync_checkpoint", "") or "").strip()
        if not checkpoint or not Path(checkpoint).is_file():
            errors.append(
                _make_issue(
                    "error",
                    f"{page}.audio_metrics_av_desync_checkpoint",
                    "AV DeSync requires a readable Synchformer checkpoint.",
                    label="AV DeSync Checkpoint",
                    page=page,
                )
            )
        if float(getattr(training, "audio_metrics_av_desync_max_length_s", 8.0)) <= 0:
            errors.append(
                _make_issue(
                    "error",
                    f"{page}.audio_metrics_av_desync_max_length_s",
                    "AV DeSync maximum duration must be greater than 0.",
                    label="AV DeSync Maximum Duration",
                    page=page,
                )
            )


def validate_training_config(config: ProjectConfig) -> dict[str, Any]:
    """Validate GUI training config before launch."""
    t = config.training
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if t.max_data_loader_n_workers is not None and t.max_data_loader_n_workers < 0:
        errors.append(
            _make_issue(
                "error",
                "training.max_data_loader_n_workers",
                "DataLoader worker count cannot be negative.",
                label="DataLoader Workers",
                page="training",
            )
        )
    if t.dataloader_prefetch_factor is not None and t.dataloader_prefetch_factor < 1:
        errors.append(
            _make_issue(
                "error",
                "training.dataloader_prefetch_factor",
                "DataLoader prefetch factor must be at least 1.",
                label="DataLoader Prefetch Factor",
                page="training",
            )
        )
    if t.max_data_loader_n_workers == 0:
        if t.dataloader_prefetch_factor is not None:
            warnings.append(
                _make_issue(
                    "warning",
                    "training.dataloader_prefetch_factor",
                    "DataLoader prefetch factor is ignored when the worker count is 0.",
                    label="DataLoader Prefetch Factor",
                    page="training",
                )
            )
        if t.persistent_data_loader_workers:
            warnings.append(
                _make_issue(
                    "warning",
                    "training.persistent_data_loader_workers",
                    "Persistent DataLoader workers are ignored when the worker count is 0.",
                    label="Persistent DataLoader Workers",
                    page="training",
                )
            )
    if t.model_type == "minimax_h3":
        allowed_h3_timestep_sampling = {"uniform", "sigmoid", "shift", "logsnr", "sigma"}
        if t.h3_timestep_sampling not in allowed_h3_timestep_sampling:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_timestep_sampling",
                    "H3 timestep sampling must use an unshifted base distribution: uniform, sigmoid, shift, logsnr, or sigma.",
                    label="H3 Timestep Sampling",
                    page="training",
                )
            )
        if t.num_timestep_buckets is not None and t.h3_timestep_sampling == "sigma":
            errors.append(
                _make_issue(
                    "error",
                    "training.num_timestep_buckets",
                    "H3 sigma sampling does not consume timestep buckets; use uniform or disable bucketing.",
                    label="Timestep Buckets",
                    page="training",
                )
            )
        cache_task = config.caching.h3_task
        compatible_tasks = {"t2va", "i2va", "fl2va", "l2va"} if t.h3_training_mode == "fl2va" else {t.h3_training_mode}
        if cache_task not in compatible_tasks:
            errors.append(
                _make_issue(
                    "error",
                    "caching.h3_task",
                    f"H3 {t.h3_training_mode.upper()} training is incompatible with {cache_task.upper()} text caches.",
                    label="H3 Conditioning Task",
                    page="caching",
                )
            )
        if not _has_text(t.h3_model):
            errors.append(
                _make_issue(
                    "error", "training.h3_model", "MiniMax H3 DiT checkpoint is required.", label="H3 Model", page="training"
                )
            )
        if t.split_attn:
            errors.append(
                _make_issue(
                    "error",
                    "training.split_attn",
                    "MiniMax H3 training does not support split attention.",
                    label="Split Attention",
                    page="training",
                )
            )
        selected_attention_backends = sum(bool(value) for value in (t.sdpa, t.flash_attn, t.flash3))
        if selected_attention_backends > 1:
            errors.append(
                _make_issue(
                    "error",
                    "training.flash_attn",
                    "Choose one H3 attention backend: SDPA, FlashAttention 2, or FlashAttention 3.",
                    label="H3 Attention Backend",
                    page="training",
                )
            )
        if t.int8_convrot_base and (t.fp8_base or t.fp8_scaled):
            message = "MiniMax H3 INT8 ConvRot and FP8 base loading are mutually exclusive."
            errors.append(
                _make_issue("error", "training.int8_convrot_base", message, label="Pruned INT8 ConvRot Base", page="training")
            )
            errors.append(_make_issue("error", "training.fp8_base", message, label="FP8 Base", page="training"))
        if t.h3_guidance_distillation_scale is not None:
            scale = float(t.h3_guidance_distillation_scale)
            if not math.isfinite(scale) or scale <= 1.0:
                errors.append(
                    _make_issue(
                        "error",
                        "training.h3_guidance_distillation_scale",
                        "H3 guidance distillation scale must be finite and greater than 1.",
                        label="H3 Guidance Distillation Scale",
                        page="training",
                    )
                )
            if not config.caching.h3_cache_guidance_empty:
                errors.append(
                    _make_issue(
                        "error",
                        "caching.h3_cache_guidance_empty",
                        "H3 guidance training requires cached empty-text conditioning.",
                        label="H3 Cache Guidance Empty",
                        page="caching",
                    )
                )
            if float(t.network_dropout or 0.0) > 0:
                errors.append(
                    _make_issue(
                        "error",
                        "training.network_dropout",
                        "H3 guidance-consistent training cannot replay neuron dropout across different prompt lengths. Use rank or module dropout instead.",
                        label="Network Dropout",
                        page="training",
                    )
                )
        preservation_weight = float(t.h3_base_preservation_loss_weight)
        if not math.isfinite(preservation_weight) or preservation_weight < 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_base_preservation_loss_weight",
                    "H3 base preservation loss weight must be finite and non-negative.",
                    label="H3 Base Preservation Loss Weight",
                    page="training",
                )
            )
        preservation_probability = float(t.h3_base_preservation_probability)
        if not math.isfinite(preservation_probability) or not 0 < preservation_probability <= 1:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_base_preservation_probability",
                    "H3 base preservation probability must be finite and lie in (0, 1].",
                    label="H3 Base Preservation Probability",
                    page="training",
                )
            )
        spatial_density_jitter = float(t.h3_spatial_density_jitter)
        if not math.isfinite(spatial_density_jitter) or spatial_density_jitter < 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_spatial_density_jitter",
                    "H3 spatial density jitter must be finite and non-negative.",
                    label="H3 Spatial Density Jitter",
                    page="training",
                )
            )
        modality_loss_weights = (
            ("h3_video_loss_weight", float(t.h3_video_loss_weight), "H3 Video Loss Weight"),
            ("h3_audio_loss_weight", float(t.h3_audio_loss_weight), "H3 Audio Loss Weight"),
        )
        for field, value, label in modality_loss_weights:
            if not math.isfinite(value) or value < 0:
                errors.append(
                    _make_issue(
                        "error",
                        f"training.{field}",
                        f"{label} must be finite and non-negative.",
                        label=label,
                        page="training",
                    )
                )
        if all(math.isfinite(value) and value == 0 for _, value, _ in modality_loss_weights):
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_video_loss_weight",
                    "At least one H3 modality loss weight must be positive.",
                    label="H3 Modality Loss Weights",
                    page="training",
                )
            )
        if t.h3_observed_modality == "video" and t.h3_audio_loss_weight == 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_audio_loss_weight",
                    "Video-observed H3 training generates audio and requires a positive audio loss weight.",
                    label="H3 Audio Loss Weight",
                    page="training",
                )
            )
        if t.h3_observed_modality == "audio" and t.h3_video_loss_weight == 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_video_loss_weight",
                    "Audio-observed H3 training generates video and requires a positive video loss weight.",
                    label="H3 Video Loss Weight",
                    page="training",
                )
            )
        focus_probability = float(t.h3_timestep_focus_probability)
        if not 0.0 <= focus_probability <= 1.0:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_timestep_focus_probability",
                    "H3 timestep focus probability must be between 0 and 1.",
                    label="H3 Timestep Focus Probability",
                    page="training",
                )
            )
        if focus_probability > 0:
            if not 0.0 <= t.h3_timestep_focus_min < t.h3_timestep_focus_max <= 1.0:
                errors.append(
                    _make_issue(
                        "error",
                        "training.h3_timestep_focus_min",
                        "H3 timestep focus must satisfy 0 <= minimum < maximum <= 1.",
                        label="H3 Timestep Focus Band",
                        page="training",
                    )
                )
            if t.h3_timestep_sampling != "uniform":
                errors.append(
                    _make_issue(
                        "error",
                        "training.h3_timestep_sampling",
                        "H3 timestep focus requires uniform timestep sampling.",
                        label="H3 Timestep Sampling",
                        page="training",
                    )
                )
            if t.min_timestep is not None or t.max_timestep is not None:
                errors.append(
                    _make_issue(
                        "error",
                        "training.h3_timestep_focus_probability",
                        "H3 timestep focus cannot be combined with minimum or maximum timestep clipping.",
                        label="H3 Timestep Focus Probability",
                        page="training",
                    )
                )
        for field, value, label in (
            ("h3_frame_sigma_jitter", t.h3_frame_sigma_jitter, "H3 Frame Sigma Jitter"),
            ("h3_caption_dropout_rate", t.h3_caption_dropout_rate, "H3 Caption Dropout"),
        ):
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0 or numeric > 1:
                errors.append(
                    _make_issue(
                        "error",
                        f"training.{field}",
                        f"{label} must be finite and lie in [0, 1].",
                        label=label,
                        page="training",
                    )
                )
        if t.h3_caption_dropout_rate > 0 and not config.caching.h3_cache_guidance_empty:
            errors.append(
                _make_issue(
                    "error",
                    "caching.h3_cache_guidance_empty",
                    "H3 caption dropout requires cached empty-text conditioning.",
                    label="H3 Cache Guidance Empty",
                    page="caching",
                )
            )
        if (t.h3_mask_mode != "off" or t.h3_mask_audio) and not (0 < t.h3_mask_min_fraction <= t.h3_mask_max_fraction <= 1):
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_mask_min_fraction",
                    "H3 mask fractions must satisfy 0 <= minimum <= maximum <= 1.",
                    label="H3 Mask Fractions",
                    page="training",
                )
            )
        if t.h3_keyframe_anchors and t.h3_keyframe_random_count:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_keyframe_random_count",
                    "Choose explicit H3 keyframe anchors or a random count, not both.",
                    label="H3 Keyframes",
                    page="training",
                )
            )
        if t.h3_keyframe_random_count < 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_keyframe_random_count",
                    "H3 random keyframe count cannot be negative.",
                    label="H3 Random Keyframes",
                    page="training",
                )
            )
        keyframes = bool(t.h3_keyframe_anchors or t.h3_keyframe_random_count)
        extension = bool(t.h3_extension_video_frames or t.h3_extension_audio_latents)
        masking = bool(t.h3_mask_mode != "off" or t.h3_mask_audio)
        if keyframes and extension:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_keyframe_anchors",
                    "H3 keyframes and extension both claim observed rows; enable only one.",
                    label="H3 Conditioning",
                    page="training",
                )
            )
        if keyframes and masking:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_keyframe_anchors",
                    "H3 keyframes and masked conditioning both claim observed rows; enable only one.",
                    label="H3 Conditioning",
                    page="training",
                )
            )
        if extension and masking:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_extension_video_frames",
                    "H3 extension and masked conditioning both claim observed rows; enable only one.",
                    label="H3 Conditioning",
                    page="training",
                )
            )
        if (keyframes or extension or masking) and t.h3_training_mode != "fl2va":
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_training_mode",
                    "H3 keyframes, extension, and masked conditioning require FL2VA training mode.",
                    label="H3 Training Mode",
                    page="training",
                )
            )
        if (extension or masking) and config.caching.h3_task != "t2va":
            errors.append(
                _make_issue(
                    "error",
                    "caching.h3_task",
                    "H3 extension and masked conditioning require T2VA caches.",
                    label="H3 Conditioning Task",
                    page="caching",
                )
            )
        if t.h3_frame_sigma_jitter > 0 and (keyframes or extension or masking or t.h3_observed_modality is not None):
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_frame_sigma_jitter",
                    "H3 frame sigma jitter cannot be combined with observed-modality, extension, keyframe, or masked conditioning.",
                    label="H3 Frame Sigma Jitter",
                    page="training",
                )
            )
        if t.h3_frame_sigma_jitter > 0 and t.weighting_scheme in {"sigma_sqrt", "cosmap"}:
            errors.append(
                _make_issue(
                    "error",
                    "training.weighting_scheme",
                    "H3 frame sigma jitter does not support per-frame sigma_sqrt or cosmap weighting.",
                    label="H3 Weighting Scheme",
                    page="training",
                )
            )
        if t.h3_sigma_sqrt_max_weight <= 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_sigma_sqrt_max_weight",
                    "H3 sigma-sqrt maximum weight must be positive.",
                    label="Sigma-sqrt Max Weight",
                    page="training",
                )
            )
        if (t.h3_keyframe_anchors or t.h3_keyframe_random_count) and config.caching.h3_task != "t2va":
            errors.append(
                _make_issue(
                    "error",
                    "caching.h3_task",
                    "Custom H3 keyframe anchors require T2VA caches.",
                    label="H3 Conditioning Task",
                    page="caching",
                )
            )
        if t.h3_adaln_rank is not None and t.h3_adaln_rank < 1:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_adaln_rank",
                    "H3 AdaLN rank must be at least 1.",
                    label="H3 AdaLN Rank",
                    page="training",
                )
            )
        if t.h3_image_flow_shift is not None and t.h3_image_flow_shift <= 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_image_flow_shift",
                    "H3 image flow shift must be positive.",
                    label="H3 Image Flow Shift",
                    page="training",
                )
            )
        for field, value, label in (
            ("h3_shift_video", t.h3_shift_video, "H3 Video Shift"),
            ("h3_shift_audio", t.h3_shift_audio, "H3 Audio Shift"),
        ):
            if not math.isfinite(float(value)) or not 0.01 <= float(value) <= 100.0:
                errors.append(
                    _make_issue(
                        "error",
                        f"training.{field}",
                        f"{label} must be finite and lie in [0.01, 100].",
                        label=label,
                        page="training",
                    )
                )
        if t.h3_extension_video_frames < 0 or t.h3_extension_audio_latents < 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_extension_video_frames",
                    "H3 extension context lengths cannot be negative.",
                    label="H3 Extension Context",
                    page="training",
                )
            )
        if t.reference_image_short_edge < 32:
            errors.append(
                _make_issue(
                    "error",
                    "training.reference_image_short_edge",
                    "H3 reference image short edge must be at least 32 pixels.",
                    label="H3 Reference Short Edge",
                    page="training",
                )
            )
        if t.h3_attn_auto_dispatch and not t.sdpa:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_attn_auto_dispatch",
                    "H3 attention auto-dispatch requires SDPA.",
                    label="H3 Attention Auto-dispatch",
                    page="training",
                )
            )
        if t.blocks_to_swap is not None and t.blocks_to_swap < 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.blocks_to_swap",
                    "H3 blocks to swap cannot be negative.",
                    label="Blocks to Swap",
                    page="training",
                )
            )
        if t.h3_int8_attention != "off" and t.compile:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_int8_attention",
                    "H3 INT8 attention cannot currently be combined with torch.compile.",
                    label="H3 INT8 Attention",
                    page="training",
                )
            )
        if t.h3_convrot_int8 and (t.fp8_base or t.fp8_scaled or t.int8_convrot_base):
            message = (
                "Online H3 ConvRot INT8 requires the BF16 checkpoint and cannot be combined with FP8 or a pruned INT8 checkpoint."
            )
            errors.append(_make_issue("error", "training.h3_convrot_int8", message, label="Online ConvRot INT8", page="training"))
        convrot_int8_active = t.h3_convrot_int8 or t.int8_convrot_base
        if t.h3_convrot_int8_bwd == "int8" and not convrot_int8_active:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_convrot_int8_bwd",
                    "INT8 ConvRot backward requires online or pre-quantized ConvRot INT8 weights.",
                    label="ConvRot Backward",
                    page="training",
                )
            )
        if t.h3_convrot_int8_fwd == "bf16" and not convrot_int8_active:
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_convrot_int8_fwd",
                    "BF16 ConvRot forward requires online or pre-quantized ConvRot INT8 weights.",
                    label="ConvRot Forward",
                    page="training",
                )
            )
        if t.h3_convrot_int8_fwd == "bf16" and t.h3_convrot_int8_bwd == "int8":
            message = "BF16 ConvRot forward cannot be combined with INT8 ConvRot backward."
            errors.append(_make_issue("error", "training.h3_convrot_int8_fwd", message, label="ConvRot Forward", page="training"))
            errors.append(_make_issue("error", "training.h3_convrot_int8_bwd", message, label="ConvRot Backward", page="training"))
        if t.h3_convrot_int8_lora_fused and not (
            convrot_int8_active and t.h3_convrot_int8_fwd == "int8" and t.h3_convrot_int8_bwd == "int8"
        ):
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_convrot_int8_lora_fused",
                    "Fused ConvRot LoRA requires online or pre-quantized ConvRot weights with INT8 forward and backward.",
                    label="Fused ConvRot LoRA",
                    page="training",
                )
            )
        checkpoint_blocks = t.h3_gradient_checkpointing_blocks
        if checkpoint_blocks is not None:
            if not 0 <= checkpoint_blocks <= 50:
                errors.append(
                    _make_issue(
                        "error",
                        "training.h3_gradient_checkpointing_blocks",
                        "H3 checkpointed blocks must be between 0 and 50.",
                        label="Checkpointed H3 Blocks",
                        page="training",
                    )
                )
            if not t.gradient_checkpointing:
                errors.append(
                    _make_issue(
                        "error",
                        "training.h3_gradient_checkpointing_blocks",
                        "Checkpointed H3 blocks requires gradient checkpointing.",
                        label="Checkpointed H3 Blocks",
                        page="training",
                    )
                )
            if checkpoint_blocks < 50 and (t.blocks_to_swap or 0) > 0:
                errors.append(
                    _make_issue(
                        "error",
                        "training.h3_gradient_checkpointing_blocks",
                        "Partial H3 gradient checkpointing cannot be combined with block swap.",
                        label="Checkpointed H3 Blocks",
                        page="training",
                    )
                )
            if checkpoint_blocks < 50 and t.compile:
                errors.append(
                    _make_issue(
                        "error",
                        "training.h3_gradient_checkpointing_blocks",
                        "Partial H3 gradient checkpointing cannot be combined with torch.compile.",
                        label="Checkpointed H3 Blocks",
                        page="training",
                    )
                )
        if t.h3_gradient_checkpointing_cpu_offload_pin_memory and not (
            t.gradient_checkpointing and t.gradient_checkpointing_cpu_offload
        ):
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_gradient_checkpointing_cpu_offload_pin_memory",
                    "Pinned H3 checkpoint offload requires gradient checkpointing and checkpoint CPU offload.",
                    label="Pinned Checkpoint Offload",
                    page="training",
                )
            )
        if t.h3_reusable_activation_offload and not (t.gradient_checkpointing and t.gradient_checkpointing_cpu_offload):
            errors.append(
                _make_issue(
                    "error",
                    "training.h3_reusable_activation_offload",
                    "Reusable H3 activation offload requires gradient checkpointing with CPU offload.",
                    label="Reusable Activation Offload",
                    page="training",
                )
            )
        if t.discrete_flow_shift != 1.0:
            errors.append(
                _make_issue(
                    "error",
                    "training.discrete_flow_shift",
                    "H3 requires discrete flow shift 1.0; use H3 Video Shift and H3 Audio Shift for modality schedules.",
                    label="Discrete Flow Shift",
                    page="training",
                )
            )
        if t.min_timestep is not None and not 0 <= t.min_timestep <= 999:
            errors.append(
                _make_issue(
                    "error",
                    "training.min_timestep",
                    "H3 minimum timestep must be between 0 and 999.",
                    label="Minimum Timestep",
                    page="training",
                )
            )
        if t.max_timestep is not None and not 1 <= t.max_timestep <= 1000:
            errors.append(
                _make_issue(
                    "error",
                    "training.max_timestep",
                    "H3 maximum timestep must be between 1 and 1000.",
                    label="Maximum Timestep",
                    page="training",
                )
            )
        if t.min_timestep is not None and t.max_timestep is not None and t.min_timestep >= t.max_timestep:
            errors.append(
                _make_issue(
                    "error",
                    "training.max_timestep",
                    "H3 maximum timestep must be greater than the minimum timestep.",
                    label="Maximum Timestep",
                    page="training",
                )
            )
        if t.num_timestep_buckets is not None and t.num_timestep_buckets < 2:
            errors.append(
                _make_issue(
                    "error",
                    "training.num_timestep_buckets",
                    "H3 timestep bucketing requires at least 2 buckets.",
                    label="Timestep Buckets",
                    page="training",
                )
            )
        unsupported_quantization = (
            ("int8_base", "int8 Base"),
            ("int8_base_dynamic", "int8 Base (dynamic)"),
            ("int8_convrot_dynamic", "INT8 ConvRot (dynamic)"),
            ("nf4_base", "NF4 Base"),
            ("fp8_w8a8", "FP8 W8A8"),
            ("int4_convrot_base", "INT4 ConvRot Base"),
            ("int4_convrot_dynamic", "INT4 ConvRot (dynamic)"),
            ("w4a4g4", "W4A4G4"),
            ("w4a8", "W4A8"),
            ("w4a4g8", "W4A4G8"),
        )
        for field, label in unsupported_quantization:
            if getattr(t, field, False):
                errors.append(
                    _make_issue(
                        "error",
                        f"training.{field}",
                        f"MiniMax H3 training does not support {label}.",
                        label=label,
                        page="training",
                    )
                )
        has_sample_prompts = _has_text(t.sample_prompts) or _has_text(t.sample_prompts_text)
        sampling_scheduled = bool(t.sample_at_first or t.sample_every_n_steps or t.sample_every_n_epochs)
        if has_sample_prompts and t.h3_training_mode != "fl2va":
            errors.append(
                _make_issue(
                    "error",
                    "training.sample_prompts",
                    "MiniMax H3 training-time sampling currently supports only FL2VA; use the Inference page for Ref2VA samples.",
                    label="Sample Prompts",
                    page="training",
                )
            )
        if sampling_scheduled and not has_sample_prompts:
            errors.append(
                _make_issue(
                    "error",
                    "training.sample_prompts",
                    "MiniMax H3 training-time sampling requires a prompt file or prompts from the Samples page.",
                    label="Sample Prompts",
                    page="training",
                )
            )
        if has_sample_prompts:
            for field, label in (
                ("h3_text_encoder", "H3 Text Encoder"),
                ("h3_tokenizer", "H3 Tokenizer"),
                ("h3_video_vae", "H3 Video VAE"),
                ("h3_audio_vae", "H3 Audio VAE"),
            ):
                if not _has_text(getattr(config.caching, field)):
                    errors.append(
                        _make_issue(
                            "error",
                            f"caching.{field}",
                            f"{label} is required for MiniMax H3 training-time sampling.",
                            label=label,
                            page="caching",
                        )
                    )
        if not config.dataset.datasets and not any(
            _has_text(value) for value in (t.dataset_config, t.dataset_manifest, t.config_file)
        ):
            errors.append(
                _make_issue(
                    "error",
                    "dataset.datasets",
                    "Add at least one H3 training dataset or provide an external dataset/config manifest.",
                    label="Training Datasets",
                    page="dataset",
                )
            )
        for index, entry in enumerate(config.dataset.datasets):
            _validate_h3_dataset_entry(entry, index, errors=errors, warnings=warnings)
            target_mode = "video" if entry.type == "image" else "audio" if entry.type == "audio" else entry.h3_target_mode
            has_active_loss = (target_mode in {"av", "video"} and t.h3_video_loss_weight > 0) or (
                target_mode in {"av", "audio"} and t.h3_audio_loss_weight > 0
            )
            if not has_active_loss:
                errors.append(
                    _make_issue(
                        "error",
                        f"dataset.datasets.{index}.h3_target_mode",
                        f"Dataset {index + 1} targets {target_mode}, but its configured H3 modality loss weight is zero.",
                        label=f"Dataset {index + 1} Target Mode",
                        page="dataset",
                    )
                )
        for index, entry in enumerate(config.dataset.validation_datasets):
            _validate_h3_dataset_entry(
                entry,
                index,
                errors=errors,
                warnings=warnings,
                collection="validation_datasets",
            )
        validation_requested = bool(t.validate_at_start or t.validate_every_n_steps or t.validate_every_n_epochs)
        has_validation_config = bool(_has_text(t.validation_dataset_config) or config.dataset.validation_datasets)
        if validation_requested and not has_validation_config:
            errors.append(
                _make_issue(
                    "error",
                    "dataset.validation_datasets",
                    "H3 validation requires a validation dataset row or an external validation dataset config.",
                    label="Validation Datasets",
                    page="dataset",
                )
            )
        if _has_text(t.validation_dataset_config) and not validation_requested:
            errors.append(
                _make_issue(
                    "error",
                    "training.validation_dataset_config",
                    "H3 validation dataset config requires Validate at Start or a validation interval.",
                    label="Validation Dataset Config",
                    page="training",
                )
            )
        validation_options_changed = bool(
            t.validation_seed is not None
            or t.validation_timestep_bins != 4
            or t.validation_min_timestep != 0
            or t.validation_max_timestep != 1000
            or t.max_validation_items is not None
        )
        if validation_options_changed and not validation_requested:
            errors.append(
                _make_issue(
                    "error",
                    "training.validation_timestep_bins",
                    "H3 validation options require Validate at Start or a validation interval.",
                    label="Validation Options",
                    page="training",
                )
            )
        if t.validate_every_n_steps is not None and t.validate_every_n_steps < 1:
            errors.append(
                _make_issue(
                    "error",
                    "training.validate_every_n_steps",
                    "Validate Every N Steps must be at least 1.",
                    label="Validation Step Interval",
                    page="training",
                )
            )
        if t.validate_every_n_epochs is not None and t.validate_every_n_epochs < 1:
            errors.append(
                _make_issue(
                    "error",
                    "training.validate_every_n_epochs",
                    "Validate Every N Epochs must be at least 1.",
                    label="Validation Epoch Interval",
                    page="training",
                )
            )
        if t.validation_timestep_bins < 1:
            errors.append(
                _make_issue(
                    "error",
                    "training.validation_timestep_bins",
                    "Validation timestep bins must be at least 1.",
                    label="Validation Timestep Bins",
                    page="training",
                )
            )
        if not 0 <= t.validation_min_timestep < t.validation_max_timestep <= 1000:
            errors.append(
                _make_issue(
                    "error",
                    "training.validation_min_timestep",
                    "Validation timestep range must satisfy 0 <= minimum < maximum <= 1000.",
                    label="Validation Timestep Range",
                    page="training",
                )
            )
        if t.max_validation_items is not None and t.max_validation_items < 1:
            errors.append(
                _make_issue(
                    "error",
                    "training.max_validation_items",
                    "Maximum validation items must be at least 1.",
                    label="Maximum Validation Items",
                    page="training",
                )
            )
        return _build_report(errors, warnings)
    _validate_generated_av_metrics(t, page="training", errors=errors)
    if _has_text(getattr(t, "lr_group_scheduler_args", "")):
        try:
            parse_group_lr_scheduler_args(_split_cli_args(t.lr_group_scheduler_args))
        except (TypeError, ValueError, re.error) as exc:
            errors.append(
                _make_issue(
                    "error",
                    "training.lr_group_scheduler_args",
                    f"Per-group LR scheduler rules are invalid: {exc}",
                    label="Per-Group LR Schedulers",
                    page="training",
                )
            )
        optimizer_type = str(t.optimizer_type or "").lower()
        if optimizer_type.endswith("schedulefree") or optimizer_type == "automagic":
            errors.append(
                _make_issue(
                    "error",
                    "training.lr_group_scheduler_args",
                    "Per-group LR schedulers cannot be combined with a schedule-free optimizer.",
                    label="Per-Group LR Schedulers",
                    page="training",
                )
            )
        if bool(getattr(t, "ltx2_remote_stage", False)):
            errors.append(
                _make_issue(
                    "error",
                    "training.lr_group_scheduler_args",
                    "Per-group LR schedulers are not supported with remote-stage training.",
                    label="Per-Group LR Schedulers",
                    page="training",
                )
            )
    modality_control_names = [
        f"{modality}_{suffix}"
        for modality in ("video", "audio", "cross_modal")
        for suffix in ("rank_dropout", "module_dropout", "max_grad_norm", "weight_decay")
    ]
    active_modality_controls = [name for name in modality_control_names if getattr(t, name, None) is not None]
    for name in active_modality_controls:
        value = float(getattr(t, name))
        is_dropout = name.endswith(("rank_dropout", "module_dropout"))
        valid = math.isfinite(value) and (0.0 <= value < 1.0 if is_dropout else value >= 0.0)
        if not valid:
            expected = "finite and in [0, 1)" if is_dropout else "finite and non-negative"
            errors.append(
                _make_issue(
                    "error",
                    f"training.{name}",
                    f"{name.replace('_', ' ').title()} must be {expected}.",
                    label=name.replace("_", " ").title(),
                    page="training",
                )
            )
    if active_modality_controls and "lycoris" in str(getattr(t, "network_module", "") or "").lower():
        errors.append(
            _make_issue(
                "error",
                f"training.{active_modality_controls[0]}",
                "Per-modality dropout and optimizer controls require the built-in LTX LoRA network.",
                label="Per-Modality Optimization",
                page="training",
            )
        )
    active_modality_clip = [name for name in active_modality_controls if name.endswith("max_grad_norm")]
    if active_modality_clip and bool(getattr(t, "ltx2_model_parallel", False)):
        errors.append(
            _make_issue(
                "error",
                f"training.{active_modality_clip[0]}",
                "Per-modality gradient clipping is not supported with model parallel training.",
                label="Per-Modality Gradient Clipping",
                page="training",
            )
        )
    active_modality_optimizer = [name for name in active_modality_controls if name.endswith(("max_grad_norm", "weight_decay"))]
    if active_modality_optimizer and bool(getattr(t, "ltx2_remote_stage", False)):
        errors.append(
            _make_issue(
                "error",
                f"training.{active_modality_optimizer[0]}",
                "Per-modality clipping and weight decay are not supported with remote-stage training.",
                label="Per-Modality Optimizer Controls",
                page="training",
            )
        )
    effective_gemma_safetensors = _effective_gemma_safetensors(t.gemma_safetensors, config.default_gemma_safetensors, t.gemma_root)

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

    _quantized_base_modes = [
        ("training.int8_base", getattr(t, "int8_base", False), "int8 Base"),
        ("training.int8_base_dynamic", getattr(t, "int8_base_dynamic", False), "int8 Base (dynamic)"),
        ("training.int8_convrot_base", getattr(t, "int8_convrot_base", False), "Pruned INT8 ConvRot Base"),
        ("training.int8_convrot_dynamic", getattr(t, "int8_convrot_dynamic", False), "INT8 ConvRot (dynamic)"),
        ("training.w4a4g4", getattr(t, "w4a4g4", False), "W4A4G4"),
        ("training.w4a8", getattr(t, "w4a8", False), "W4A8"),
        ("training.w4a4g8", getattr(t, "w4a4g8", False), "W4A4G8"),
    ]
    _enabled_quantized_base_modes = [mode for mode in _quantized_base_modes if mode[1]]
    if len(_enabled_quantized_base_modes) > 1:
        message = "Only one quantized base mode can be enabled."
        for field, _enabled, label in _enabled_quantized_base_modes:
            errors.append(_make_issue("error", field, message, label=label, page="training"))
    if len(_enabled_quantized_base_modes) == 1:
        _quantized_field, _enabled, _quantized_label = _enabled_quantized_base_modes[0]
        for _conf_field, _conf_val, _conf_label in (
            ("training.fp8_base", t.fp8_base, "FP8 Base"),
            ("training.fp8_scaled", t.fp8_scaled, "FP8 Scaled"),
            ("training.nf4_base", t.nf4_base, "NF4 Base"),
        ):
            if _conf_val:
                message = f"{_quantized_label} is mutually exclusive with {_conf_label}."
                errors.append(_make_issue("error", _quantized_field, message, label=_quantized_label, page="training"))
                errors.append(_make_issue("error", _conf_field, message, label=_conf_label, page="training"))
    _int4_scale_refine_steps = int(getattr(t, "int4_convrot_scale_refine_steps", 0) or 0)
    if _int4_scale_refine_steps < 0:
        errors.append(
            _make_issue(
                "error",
                "training.int4_convrot_scale_refine_steps",
                "INT4 scale refinement steps must be zero or greater.",
                label="INT4 Scale Refine Steps",
                page="training",
            )
        )
    elif _int4_scale_refine_steps and not (getattr(t, "w4a4g4", False) or getattr(t, "w4a8", False) or getattr(t, "w4a4g8", False)):
        errors.append(
            _make_issue(
                "error",
                "training.int4_convrot_scale_refine_steps",
                "INT4 scale refinement requires a dynamic W4A4G4, W4A8, or W4A4G8 base mode.",
                label="INT4 Scale Refine Steps",
                page="training",
            )
        )
    _int4_group_scales = int(getattr(t, "int4_convrot_group_scales", 0) or 0)
    if _int4_group_scales < 0 or (
        _int4_group_scales and (_int4_group_scales < 16 or _int4_group_scales & (_int4_group_scales - 1))
    ):
        errors.append(
            _make_issue(
                "error",
                "training.int4_convrot_group_scales",
                "INT4 group scales must be 0/off or a power of two of at least 16.",
                label="INT4 Group Scales",
                page="training",
            )
        )
    elif _int4_group_scales and not (getattr(t, "w4a4g4", False) or getattr(t, "w4a8", False) or getattr(t, "w4a4g8", False)):
        errors.append(
            _make_issue(
                "error",
                "training.int4_convrot_group_scales",
                "INT4 group scales require a dynamic W4A4G4, W4A8, or W4A4G8 base mode.",
                label="INT4 Group Scales",
                page="training",
            )
        )
    if getattr(t, "int4_convrot_group_ratio_q8", False) and not _int4_group_scales:
        errors.append(
            _make_issue(
                "error",
                "training.int4_convrot_group_ratio_q8",
                "INT4 Q8.8 group-ratio storage requires INT4 group scales.",
                label="INT4 Q8.8 Group Ratios",
                page="training",
            )
        )
    _int4_compare_raw = str(getattr(t, "int4_convrot_compare_group_scales", "") or "").strip()
    if _int4_compare_raw:
        try:
            _int4_compare_values = [int(part.strip()) for part in _int4_compare_raw.replace(";", ",").split(",") if part.strip()]
        except ValueError:
            _int4_compare_values = []
        _int4_compare_invalid = not _int4_compare_values or any(
            value < 0 or (value and (value < 16 or value & (value - 1))) for value in _int4_compare_values
        )
        if _int4_compare_invalid:
            errors.append(
                _make_issue(
                    "error",
                    "training.int4_convrot_compare_group_scales",
                    "INT4 comparison sizes must be a comma-separated list of 0 or powers of two of at least 16.",
                    label="INT4 Group-Scale Comparison",
                    page="training",
                )
            )
        if not (getattr(t, "w4a4g4", False) or getattr(t, "w4a8", False) or getattr(t, "w4a4g8", False)):
            errors.append(
                _make_issue(
                    "error",
                    "training.int4_convrot_compare_group_scales",
                    "INT4 group-scale comparison requires a dynamic W4A4G4, W4A8, or W4A4G8 base mode.",
                    label="INT4 Group-Scale Comparison",
                    page="training",
                )
            )
        if not getattr(t, "int4_convrot_quality_report", ""):
            errors.append(
                _make_issue(
                    "error",
                    "training.int4_convrot_compare_group_scales",
                    "INT4 group-scale comparison requires an INT4 quality report path.",
                    label="INT4 Group-Scale Comparison",
                    page="training",
                )
            )
    if getattr(t, "convrot_policy", "") and not (
        getattr(t, "int8_convrot_base", False)
        or getattr(t, "int8_convrot_dynamic", False)
        or getattr(t, "w4a4g4", False)
        or getattr(t, "w4a8", False)
        or getattr(t, "w4a4g8", False)
    ):
        errors.append(
            _make_issue(
                "error",
                "training.convrot_policy",
                "ConvRot Policy requires an INT8 or INT4 ConvRot base mode.",
                label="ConvRot Policy",
                page="training",
            )
        )

    network_module = t.network_module or get_ltx2_training_network_module_default()
    lycoris_requested = (
        "lycoris" in network_module.lower()
        or bool(getattr(t, "lycoris_config", ""))
        or bool(getattr(t, "lycoris_algo", ""))
        or getattr(t, "lycoris_factor", None) is not None
        or getattr(t, "lycoris_conv_dim", None) is not None
        or getattr(t, "lycoris_conv_alpha", None) is not None
        or getattr(t, "lycoris_dropout", None) is not None
        or str(getattr(t, "lora_target_preset", "") or "").lower() == "lycoris"
    )

    if t.use_dora:
        if network_module in {"networks.loha", "lycoris.kohya"} or lycoris_requested:
            errors.append(
                _make_issue(
                    "error",
                    "training.use_dora",
                    "DoRA/DokR is currently available only with the native LoRA or native LoKr backend.",
                    label="DoRA/DokR",
                    page="training",
                )
            )

    if t.use_dora_oft:
        if network_module in {"networks.loha", "lycoris.kohya"} or lycoris_requested:
            errors.append(
                _make_issue(
                    "error",
                    "training.use_dora_oft",
                    "DoRA-OFT/DoKr-OFT is currently available only with the native LoRA or native LoKr backend.",
                    label="DoRA-OFT/DoKr-OFT",
                    page="training",
                )
            )
        if t.use_dora:
            message = "DoRA/DokR and DoRA-OFT/DoKr-OFT cannot be enabled together."
            errors.append(_make_issue("error", "training.use_dora", message, label="DoRA/DokR", page="training"))
            errors.append(_make_issue("error", "training.use_dora_oft", message, label="DoRA-OFT/DoKr-OFT", page="training"))
        if t.adaptive_rank:
            message = "Adaptive rank is not supported with DoRA-OFT/DoKr-OFT."
            errors.append(_make_issue("error", "training.use_dora_oft", message, label="DoRA-OFT/DoKr-OFT", page="training"))
            errors.append(_make_issue("error", "training.adaptive_rank", message, label="Adaptive Rank", page="training"))

    if getattr(t, "use_oft", False):
        # Plain OFT is honored only by the native LoRA backend. The LoKr backend
        # ignores use_oft and exposes OFT only through DoKr-OFT (use_dora_oft).
        if lycoris_requested or network_module not in {
            "networks.lora",
            "networks.lora_ltx2",
            "musubi_tuner.networks.lora",
            "musubi_tuner.networks.lora_ltx2",
        }:
            errors.append(
                _make_issue(
                    "error",
                    "training.use_oft",
                    "OFT is currently available only with the native LoRA backend. Use DoRA-OFT/DoKr-OFT for the LoKr backend.",
                    label="OFT",
                    page="training",
                )
            )
        if t.use_dora_oft:
            message = "OFT and DoRA-OFT/DoKr-OFT cannot be enabled together."
            errors.append(_make_issue("error", "training.use_oft", message, label="OFT", page="training"))
            errors.append(_make_issue("error", "training.use_dora_oft", message, label="DoRA-OFT/DoKr-OFT", page="training"))
        if t.use_dora:
            message = "OFT and DoRA/DokR cannot be enabled together."
            errors.append(_make_issue("error", "training.use_oft", message, label="OFT", page="training"))
            errors.append(_make_issue("error", "training.use_dora", message, label="DoRA/DokR", page="training"))

    if getattr(t, "use_rslora", False):
        if t.use_dora_oft:
            message = "rsLoRA is not supported with DoRA-OFT/DoKr-OFT."
            errors.append(_make_issue("error", "training.use_rslora", message, label="rsLoRA", page="training"))
            errors.append(_make_issue("error", "training.use_dora_oft", message, label="DoRA-OFT/DoKr-OFT", page="training"))
        if getattr(t, "use_oft", False):
            message = "rsLoRA is not supported with OFT."
            errors.append(_make_issue("error", "training.use_rslora", message, label="rsLoRA", page="training"))
            errors.append(_make_issue("error", "training.use_oft", message, label="OFT", page="training"))
        if lycoris_requested or network_module not in {
            "networks.lora",
            "networks.lora_ltx2",
            "musubi_tuner.networks.lora",
            "musubi_tuner.networks.lora_ltx2",
        }:
            errors.append(
                _make_issue(
                    "error",
                    "training.use_rslora",
                    "rsLoRA is currently available only with the native LoRA backend.",
                    label="rsLoRA",
                    page="training",
                )
            )

    if getattr(t, "lycoris_factor", None) is not None and t.lycoris_factor <= 0:
        errors.append(
            _make_issue(
                "error",
                "training.lycoris_factor",
                "LyCORIS factor must be greater than 0.",
                label="LyCORIS Factor",
                page="training",
            )
        )
    if getattr(t, "lycoris_conv_dim", None) is not None and t.lycoris_conv_dim <= 0:
        errors.append(
            _make_issue(
                "error",
                "training.lycoris_conv_dim",
                "LyCORIS conv dim must be greater than 0.",
                label="LyCORIS Conv Dim",
                page="training",
            )
        )
    if getattr(t, "lycoris_conv_alpha", None) is not None and t.lycoris_conv_alpha < 0:
        errors.append(
            _make_issue(
                "error",
                "training.lycoris_conv_alpha",
                "LyCORIS conv alpha cannot be negative.",
                label="LyCORIS Conv Alpha",
                page="training",
            )
        )
    if getattr(t, "lycoris_dropout", None) is not None and not (0 <= t.lycoris_dropout <= 1):
        errors.append(
            _make_issue(
                "error",
                "training.lycoris_dropout",
                "LyCORIS dropout must be between 0 and 1.",
                label="LyCORIS Dropout",
                page="training",
            )
        )

    if t.blockwise_checkpointing:
        warnings.append(
            _make_issue(
                "warning",
                "training.blockwise_checkpointing",
                "Blockwise checkpointing checkpoints blocks individually and reloads state during backward. On the 832x480x49 video dataset, peak VRAM is typically 4-6 GiB with --blocks_to_swap 47.",
                label="Blockwise Checkpointing",
                page="training",
            )
        )

    if t.block_swap_h2d_only:
        if t.blocks_to_swap in (None, 0):
            errors.append(
                _make_issue(
                    "error",
                    "training.block_swap_h2d_only",
                    "H2D-only block swap requires Blocks To Swap to be greater than 0.",
                    label="H2D-only Block Swap",
                    page="training",
                )
            )
        if not (t.gradient_checkpointing or t.blockwise_checkpointing):
            errors.append(
                _make_issue(
                    "error",
                    "training.block_swap_h2d_only",
                    "H2D-only block swap requires gradient checkpointing for training.",
                    label="H2D-only Block Swap",
                    page="training",
                )
            )
        if t.block_swap_ring_size < 1:
            errors.append(
                _make_issue(
                    "error",
                    "training.block_swap_ring_size",
                    "Block swap ring size must be at least 1.",
                    label="Block Swap Ring Size",
                    page="training",
                )
            )

    if getattr(t, "ltx2_compile_inner_blocks", False):
        if not t.compile:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_compile_inner_blocks",
                    "Compile inner LTX blocks requires torch.compile.",
                    label="Compile Inner LTX Blocks",
                    page="training",
                )
            )
        if t.blocks_to_swap not in (None, 0) or t.blockwise_checkpointing or t.gradient_checkpointing_cpu_offload:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_compile_inner_blocks",
                    "Compile inner LTX blocks requires resident weights and activations.",
                    label="Compile Inner LTX Blocks",
                    page="training",
                )
            )

    if getattr(t, "ltx2_bounded_activation_offload", False):
        conflicts = []
        if not t.gradient_checkpointing:
            conflicts.append("Gradient Checkpointing must be enabled")
        if t.gradient_checkpointing_cpu_offload:
            conflicts.append("Checkpoint CPU Offload")
        if t.blockwise_checkpointing:
            conflicts.append("Blockwise Checkpointing")
        if t.ltx2_partial_gradient_checkpointing:
            conflicts.append("Partial Gradient Checkpointing")
        if t.compile:
            conflicts.append("torch.compile")
        if t.ltx2_model_parallel:
            conflicts.append("Model Parallel")
        if t.ltx2_remote_stage:
            conflicts.append("Remote Stage")
        if conflicts:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_bounded_activation_offload",
                    "Bounded Activation Offload configuration conflict: " + ", ".join(conflicts) + ".",
                    label="Bounded Activation Offload",
                    page="training",
                )
            )
        max_inflight = t.ltx2_activation_offload_max_inflight
        keep_trailing = t.ltx2_activation_offload_keep_trailing
        min_mb = t.ltx2_activation_offload_min_mb
        if max_inflight is not None and max_inflight < 1:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_activation_offload_max_inflight",
                    "Activation offload maximum in-flight calls must be at least 1.",
                    label="Activation Offload In-Flight Limit",
                    page="training",
                )
            )
        if keep_trailing is not None and keep_trailing < 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_activation_offload_keep_trailing",
                    "Activation offload trailing block count cannot be negative.",
                    label="Activation Offload Trailing Blocks",
                    page="training",
                )
            )
        if min_mb is not None and min_mb < 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_activation_offload_min_mb",
                    "Activation offload minimum size cannot be negative.",
                    label="Activation Offload Minimum Size",
                    page="training",
                )
            )

    try:
        video_anchor_strength = float(t.video_anchor_strength)
    except (TypeError, ValueError):
        video_anchor_strength = float("nan")
    if not math.isfinite(video_anchor_strength) or not 0.0 <= video_anchor_strength <= 1.0:
        errors.append(
            _make_issue(
                "error",
                "training.video_anchor_strength",
                "Video Anchor Strength must be a finite number in the inclusive range 0.0 to 1.0.",
                label="Video Anchor Strength",
                page="conditioning",
            )
        )
    elif video_anchor_strength != 1.0 and not t.ltx2_graded_conditioning:
        errors.append(
            _make_issue(
                "error",
                "training.ltx2_graded_conditioning",
                "Enable Graded Conditioning to use Video Anchor Strength below 1.0.",
                label="Graded Conditioning",
                page="conditioning",
            )
        )

    if t.ltx2_causal_temporal_attention:
        if t.ltx2_mode == "audio" or t.ltx2_audio_only_model:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_causal_temporal_attention",
                    "Causal Temporal Attention requires a video training path.",
                    label="Causal Temporal Attention",
                    page="conditioning",
                )
            )
        if not t.sdpa:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_causal_temporal_attention",
                    "Causal Temporal Attention requires SDPA.",
                    label="Causal Temporal Attention",
                    page="conditioning",
                )
            )

    try:
        soft_av_sigma = float(t.ltx2_soft_av_alignment_sigma)
    except (TypeError, ValueError):
        soft_av_sigma = float("nan")
    if not math.isfinite(soft_av_sigma) or soft_av_sigma <= 0.0:
        errors.append(
            _make_issue(
                "error",
                "training.ltx2_soft_av_alignment_sigma",
                "Soft AV Alignment Sigma must be finite and greater than zero.",
                label="Soft AV Alignment Sigma",
                page="conditioning",
            )
        )
    if t.ltx2_soft_av_alignment:
        if t.ltx2_mode != "av" or t.ltx2_audio_only_model:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_soft_av_alignment",
                    "Soft AV Alignment requires AV mode.",
                    label="Soft AV Alignment",
                    page="conditioning",
                )
            )
        if not t.sdpa:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_soft_av_alignment",
                    "Soft AV Alignment requires SDPA.",
                    label="Soft AV Alignment",
                    page="conditioning",
                )
            )

    if t.ltx2_model_parallel:
        if t.ltx2_remote_stage:
            message = "LTX2 Model Parallel and LTX2 Remote Stage cannot be enabled together."
            errors.append(_make_issue("error", "training.ltx2_model_parallel", message, label="Model Parallel", page="training"))
            errors.append(_make_issue("error", "training.ltx2_remote_stage", message, label="Remote Stage", page="training"))

        if t.blocks_to_swap not in (None, 0):
            errors.append(
                _make_issue(
                    "error",
                    "training.blocks_to_swap",
                    "LTX2 Model Parallel is incompatible with block swapping.",
                    label="Blocks To Swap",
                    page="training",
                )
            )
        if t.blockwise_checkpointing:
            errors.append(
                _make_issue(
                    "error",
                    "training.blockwise_checkpointing",
                    "LTX2 Model Parallel is not compatible with blockwise checkpointing yet.",
                    label="Blockwise Checkpointing",
                    page="training",
                )
            )
        if t.compile:
            errors.append(
                _make_issue(
                    "error",
                    "training.compile",
                    "LTX2 Model Parallel is not compatible with torch.compile yet.",
                    label="Compile",
                    page="training",
                )
            )

        accelerate_args = _split_cli_args(t.accelerate_extra_args)
        if "--multi_gpu" in accelerate_args:
            errors.append(
                _make_issue(
                    "error",
                    "training.accelerate_extra_args",
                    "LTX2 Model Parallel is single-process; remove --multi_gpu from Accelerate Extra Args.",
                    label="Accelerate Extra Args",
                    page="training",
                )
            )
        num_processes = _accelerate_num_processes(t.accelerate_extra_args)
        if num_processes is not None and num_processes != 1:
            errors.append(
                _make_issue(
                    "error",
                    "training.accelerate_extra_args",
                    "LTX2 Model Parallel requires Accelerate --num_processes 1.",
                    label="Accelerate Extra Args",
                    page="training",
                )
            )
        elif num_processes is None:
            warnings.append(
                _make_issue(
                    "warning",
                    "training.accelerate_extra_args",
                    "LTX2 Model Parallel should be launched with --num_processes 1.",
                    label="Accelerate Extra Args",
                    page="training",
                )
            )

        device_ids = _parse_csv_ints(t.ltx2_model_parallel_devices)
        if device_ids is None:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_model_parallel_devices",
                    "Model Parallel Devices must be a comma-separated integer list.",
                    label="Model Parallel Devices",
                    page="training",
                )
            )
        elif device_ids:
            if len(device_ids) < 2:
                errors.append(
                    _make_issue(
                        "error",
                        "training.ltx2_model_parallel_devices",
                        "LTX2 Model Parallel requires at least two CUDA devices.",
                        label="Model Parallel Devices",
                        page="training",
                    )
                )
            if len(set(device_ids)) != len(device_ids):
                errors.append(
                    _make_issue(
                        "error",
                        "training.ltx2_model_parallel_devices",
                        "Model Parallel Devices must be unique.",
                        label="Model Parallel Devices",
                        page="training",
                    )
                )
            if device_ids[0] != 0:
                warnings.append(
                    _make_issue(
                        "warning",
                        "training.ltx2_model_parallel_devices",
                        "The first model-parallel device should usually be 0 after CUDA_VISIBLE_DEVICES remapping.",
                        label="Model Parallel Devices",
                        page="training",
                    )
                )

        split_points = _parse_csv_ints(t.ltx2_model_parallel_splits)
        if split_points is None:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_model_parallel_splits",
                    "Model Parallel Splits must be a comma-separated integer list.",
                    label="Model Parallel Splits",
                    page="training",
                )
            )
        elif split_points:
            if split_points != sorted(split_points) or len(set(split_points)) != len(split_points):
                errors.append(
                    _make_issue(
                        "error",
                        "training.ltx2_model_parallel_splits",
                        "Model Parallel Splits must be strictly increasing.",
                        label="Model Parallel Splits",
                        page="training",
                    )
                )
            if split_points[0] <= 0 or split_points[-1] >= 48:
                errors.append(
                    _make_issue(
                        "error",
                        "training.ltx2_model_parallel_splits",
                        "Model Parallel Splits must be inside the LTX2 transformer block range 1..47.",
                        label="Model Parallel Splits",
                        page="training",
                    )
                )
            if device_ids and len(split_points) != len(device_ids) - 1:
                errors.append(
                    _make_issue(
                        "error",
                        "training.ltx2_model_parallel_splits",
                        "Model Parallel Splits must contain one fewer value than Model Parallel Devices.",
                        label="Model Parallel Splits",
                        page="training",
                    )
                )

        if t.ltx2_mp_profile_log_every <= 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_mp_profile_log_every",
                    "Model-parallel profile log interval must be greater than 0.",
                    label="MP Profile Log Every",
                    page="training",
                )
            )
        if t.ltx2_mp_int8_block_size <= 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_mp_int8_block_size",
                    "Model-parallel codec block size must be greater than 0.",
                    label="MP Codec Block Size",
                    page="training",
                )
            )

    if t.ltx2_remote_stage:
        if t.blocks_to_swap not in (None, 0):
            errors.append(
                _make_issue(
                    "error",
                    "training.blocks_to_swap",
                    "LTX2 Remote Stage is incompatible with block swapping.",
                    label="Blocks To Swap",
                    page="training",
                )
            )
        if t.blockwise_checkpointing:
            errors.append(
                _make_issue(
                    "error",
                    "training.blockwise_checkpointing",
                    "LTX2 Remote Stage is not compatible with blockwise checkpointing yet.",
                    label="Blockwise Checkpointing",
                    page="training",
                )
            )
        if t.compile:
            errors.append(
                _make_issue(
                    "error",
                    "training.compile",
                    "LTX2 Remote Stage is not compatible with torch.compile yet.",
                    label="Compile",
                    page="training",
                )
            )

        accelerate_args = _split_cli_args(t.accelerate_extra_args)
        if "--multi_gpu" in accelerate_args:
            errors.append(
                _make_issue(
                    "error",
                    "training.accelerate_extra_args",
                    "LTX2 Remote Stage is single-process; remove --multi_gpu from Accelerate Extra Args.",
                    label="Accelerate Extra Args",
                    page="training",
                )
            )
        num_processes = _accelerate_num_processes(t.accelerate_extra_args)
        if num_processes is not None and num_processes != 1:
            errors.append(
                _make_issue(
                    "error",
                    "training.accelerate_extra_args",
                    "LTX2 Remote Stage requires Accelerate --num_processes 1.",
                    label="Accelerate Extra Args",
                    page="training",
                )
            )
        elif num_processes is None:
            warnings.append(
                _make_issue(
                    "warning",
                    "training.accelerate_extra_args",
                    "LTX2 Remote Stage should be launched with --num_processes 1.",
                    label="Accelerate Extra Args",
                    page="training",
                )
            )

        specs, specs_error = _parse_remote_stage_specs(t.ltx2_remote_stage_specs)
        if specs_error:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_remote_stage_specs",
                    specs_error,
                    label="Remote Stage Specs",
                    page="training",
                )
            )
        elif specs:
            first_start = specs[0][2]
            last_end = specs[-1][3]
            if first_start < 0 or last_end > 48:
                errors.append(
                    _make_issue(
                        "error",
                        "training.ltx2_remote_stage_specs",
                        "Remote Stage Specs block ranges must stay inside 0..48 for LTX-2.",
                        label="Remote Stage Specs",
                        page="training",
                    )
                )
            if last_end != 48:
                errors.append(
                    _make_issue(
                        "error",
                        "training.ltx2_remote_stage_specs",
                        "Remote Stage Specs must currently cover the suffix through block 48.",
                        label="Remote Stage Specs",
                        page="training",
                    )
                )
        else:
            if not _has_text(t.ltx2_remote_stage_host):
                errors.append(
                    _make_issue(
                        "error",
                        "training.ltx2_remote_stage_host",
                        "Remote Stage Host is required when specs are empty.",
                        label="Remote Stage Host",
                        page="training",
                    )
                )
            if not (0 < int(t.ltx2_remote_stage_port) < 65536):
                errors.append(
                    _make_issue(
                        "error",
                        "training.ltx2_remote_stage_port",
                        "Remote Stage Port must be in 1..65535.",
                        label="Remote Stage Port",
                        page="training",
                    )
                )
            if t.ltx2_remote_stage_split < 0 or t.ltx2_remote_stage_split >= 48:
                errors.append(
                    _make_issue(
                        "error",
                        "training.ltx2_remote_stage_split",
                        "Remote Stage Split must be a block index in 0..47.",
                        label="Remote Stage Split",
                        page="training",
                    )
                )

        if t.ltx2_remote_stage_timeout <= 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_remote_stage_timeout",
                    "Remote Stage Timeout must be greater than 0.",
                    label="Remote Stage Timeout",
                    page="training",
                )
            )
        if t.ltx2_remote_stage_int8_block_size <= 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_remote_stage_int8_block_size",
                    "Remote Stage codec block size must be greater than 0.",
                    label="Remote Stage Codec Block Size",
                    page="training",
                )
            )
        if t.ltx2_remote_stage_metadata_cache_size <= 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_remote_stage_metadata_cache_size",
                    "Remote Stage Metadata Cache Size must be greater than 0.",
                    label="Remote Metadata Cache Size",
                    page="training",
                )
            )
        if t.ltx2_remote_stage_aq_cache_size < 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_remote_stage_aq_cache_size",
                    "Remote Stage AQ Cache Size must be >= 0.",
                    label="Remote AQ Cache Size",
                    page="training",
                )
            )
        if t.ltx2_remote_stage_trainable_scope != "auto" and not t.ltx2_remote_stage_trainable:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_remote_stage_trainable_scope",
                    "Remote Stage Trainable Scope requires Remote Stage Trainable.",
                    label="Remote Trainable Scope",
                    page="training",
                )
            )

    if t.awq_calibration and not t.nf4_base:
        message = "AWQ Calibration requires NF4 Base."
        errors.append(_make_issue("error", "training.awq_calibration", message, label="AWQ Calibration", page="training"))
        errors.append(_make_issue("error", "training.nf4_base", message, label="NF4 Base", page="training"))

    if t.av_cross_grad_surgery:
        if t.ltx2_mode != "av":
            errors.append(
                _make_issue(
                    "error",
                    "training.av_cross_grad_surgery",
                    "AV Cross Grad Surgery requires LTX2 Mode = av.",
                    label="AV Cross Grad Surgery",
                    page="training",
                )
            )
        if t.ltx2_audio_only_model:
            errors.append(
                _make_issue(
                    "error",
                    "training.av_cross_grad_surgery",
                    "AV Cross Grad Surgery requires a video+audio transformer, not Audio-only Model.",
                    label="AV Cross Grad Surgery",
                    page="training",
                )
            )
        if t.ltx2_remote_stage:
            errors.append(
                _make_issue(
                    "error",
                    "training.av_cross_grad_surgery",
                    "AV Cross Grad Surgery is not supported with Remote Stage.",
                    label="AV Cross Grad Surgery",
                    page="training",
                )
            )
        try:
            parse_av_cross_grad_surgery_args(_split_cli_args(t.av_cross_grad_surgery_args), total_layers=48)
        except (TypeError, ValueError) as exc:
            errors.append(
                _make_issue(
                    "error",
                    "training.av_cross_grad_surgery_args",
                    f"AV Cross Grad Surgery Args are invalid: {exc}",
                    label="AV Cross Grad Surgery Args",
                    page="training",
                )
            )

    if t.av_attention_loss_weighting:
        if t.ltx2_mode != "av":
            errors.append(
                _make_issue(
                    "error",
                    "training.av_attention_loss_weighting",
                    "AV Attention Loss Weighting requires LTX2 Mode = av.",
                    label="AV Attention Loss Weighting",
                    page="training",
                )
            )
        if t.ltx2_audio_only_model:
            errors.append(
                _make_issue(
                    "error",
                    "training.av_attention_loss_weighting",
                    "AV Attention Loss Weighting requires a video+audio transformer, not Audio-only Model.",
                    label="AV Attention Loss Weighting",
                    page="training",
                )
            )
        if t.av_attention_loss_max < 1.0:
            errors.append(
                _make_issue(
                    "error",
                    "training.av_attention_loss_max",
                    "AV Attention Loss Max must be >= 1.0.",
                    label="AV Attention Loss Max",
                    page="training",
                )
            )
        if t.av_attention_loss_warmup_steps < 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.av_attention_loss_warmup_steps",
                    "AV Attention Loss Warmup Steps must be >= 0.",
                    label="AV Attention Loss Warmup",
                    page="training",
                )
            )

    if t.av_curriculum_mode != "none":
        if t.ltx2_mode != "av" or t.ltx2_audio_only_model:
            errors.append(
                _make_issue(
                    "error",
                    "training.av_curriculum_mode",
                    "AV Curriculum requires LTX2 Mode = av and a joint video+audio transformer.",
                    label="AV Curriculum",
                    page="techniques",
                )
            )
        if t.av_curriculum_interval_steps <= 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.av_curriculum_interval_steps",
                    "AV Curriculum Interval Steps must be greater than 0.",
                    label="AV Curriculum Interval",
                    page="techniques",
                )
            )
        if t.av_curriculum_mode == "two_stage":
            if t.av_curriculum_stage1_steps <= 0:
                errors.append(
                    _make_issue(
                        "error",
                        "training.av_curriculum_stage1_steps",
                        "Two-stage AV Curriculum requires Stage 1 Steps greater than 0.",
                        label="AV Curriculum Stage 1 Steps",
                        page="techniques",
                    )
                )
            if t.av_curriculum_stage1_policy == t.av_curriculum_stage2_policy:
                errors.append(
                    _make_issue(
                        "error",
                        "training.av_curriculum_stage2_policy",
                        "Two-stage AV Curriculum requires different stage policies.",
                        label="AV Curriculum Stage 2 Policy",
                        page="techniques",
                    )
                )
            covered_policies = {t.av_curriculum_stage1_policy, t.av_curriculum_stage2_policy}
            if not (
                ("video" in covered_policies or "joint" in covered_policies)
                and ("audio" in covered_policies or "joint" in covered_policies)
            ):
                errors.append(
                    _make_issue(
                        "error",
                        "training.av_curriculum_stage2_policy",
                        "Two-stage AV Curriculum policies must collectively train video and audio.",
                        label="AV Curriculum Stage 2 Policy",
                        page="techniques",
                    )
                )
        conflicts = []
        inferred_ic_strategy = t.ic_lora_strategy
        if inferred_ic_strategy == "auto" and t.lora_target_preset in {
            "v2v",
            "audio_ref_ic",
            "av_ic",
            "video_ref_only_av",
        }:
            inferred_ic_strategy = t.lora_target_preset
        if inferred_ic_strategy not in {"auto", "none"}:
            conflicts.append("IC-LoRA")
        if t.ltx2_train_direction != "joint":
            conflicts.append("Directional Training")
        if t.audio_loss_balance_mode != "none":
            conflicts.append("Audio Loss Balance")
        if t.modality_freeze_check_interval > 0:
            conflicts.append("Modality Freeze")
        if t.audio_silence_regularizer:
            conflicts.append("Audio Silence Regularizer")
        if t.self_flow:
            conflicts.append("Self-Flow")
        if t.crepa:
            conflicts.append("CREPA")
        if t.cts_lambda_video_driven > 0 or t.cts_lambda_audio_driven > 0:
            conflicts.append("Cross-Task Synergy")
        if conflicts:
            errors.append(
                _make_issue(
                    "error",
                    "training.av_curriculum_mode",
                    f"AV Curriculum cannot be combined with: {', '.join(conflicts)}.",
                    label="AV Curriculum",
                    page="techniques",
                )
            )
        if t.video_loss_weight <= 0 or t.audio_loss_weight <= 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.av_curriculum_mode",
                    "AV Curriculum requires positive Video Loss Weight and Audio Loss Weight.",
                    label="AV Curriculum",
                    page="techniques",
                )
            )

    if t.tread:
        tread_args_parts = _split_cli_args(t.tread_args)
        if t.tread_target != "video":
            tread_args_parts.append(f"target={t.tread_target}")
        if t.tread_selection_ratio != 0.5:
            tread_args_parts.append(f"selection_ratio={t.tread_selection_ratio}")
        if t.tread_start_layer_idx is not None:
            tread_args_parts.append(f"start_layer_idx={t.tread_start_layer_idx}")
        if t.tread_end_layer_idx is not None:
            tread_args_parts.append(f"end_layer_idx={t.tread_end_layer_idx}")
        parsed_tread_config = None
        try:
            parsed_tread_config = parse_tread_args(
                tread_args_parts,
                total_layers=48,
                default_route=default_ltx_tread_route(t.ltx_version),
            )
        except (TypeError, ValueError) as exc:
            errors.append(
                _make_issue(
                    "error",
                    "training.tread_args",
                    f"TREAD Args are invalid: {exc}",
                    label="TREAD Args",
                    page="techniques",
                )
            )
        tread_targets = {
            str(route.get("target", "video")).lower()
            for route in ((parsed_tread_config or {}).get("routes") or [{"target": "video"}])
        }
        wants_video_tread = any(target in {"video", "both"} for target in tread_targets)
        wants_audio_tread = any(target in {"audio", "both"} for target in tread_targets)
        try:
            tread_selection_ratio = float(t.tread_selection_ratio)
        except (TypeError, ValueError):
            tread_selection_ratio = float("nan")
        if not math.isfinite(tread_selection_ratio) or not 0.0 <= tread_selection_ratio < 1.0:
            errors.append(
                _make_issue(
                    "error",
                    "training.tread_selection_ratio",
                    "TREAD Selection Ratio must be at least 0.0 and less than 1.0.",
                    label="TREAD Selection Ratio",
                    page="techniques",
                )
            )
        if wants_video_tread and (t.ltx2_mode == "audio" or t.ltx2_audio_only_model):
            errors.append(
                _make_issue(
                    "error",
                    "training.tread",
                    "TREAD target=video requires a video-enabled LTX path. Use target=audio for audio-only training.",
                    label="TREAD",
                    page="techniques",
                )
            )
        if wants_audio_tread and t.ltx2_mode == "video":
            errors.append(
                _make_issue(
                    "error",
                    "training.tread",
                    "TREAD target=audio requires an audio-enabled LTX mode.",
                    label="TREAD",
                    page="techniques",
                )
            )
        if t.ltx2_remote_stage:
            errors.append(
                _make_issue(
                    "error",
                    "training.tread",
                    "TREAD cannot be combined with this execution mode because routing changes token lengths across blocks.",
                    label="TREAD",
                    page="techniques",
                )
            )

    if t.differential_guidance:
        try:
            from musubi_tuner.differential_guidance import DifferentialGuidanceConfig

            DifferentialGuidanceConfig.from_args(t)
        except (TypeError, ValueError) as exc:
            errors.append(
                _make_issue(
                    "error",
                    "training.differential_guidance",
                    str(exc),
                    label="Differential Guidance",
                    page="techniques",
                )
            )
        if t.ltx2_mode == "audio" or t.ltx2_audio_only_model:
            errors.append(
                _make_issue(
                    "error",
                    "training.differential_guidance",
                    "Differential Guidance requires a video/main prediction loss and cannot be used with audio-only training.",
                    label="Differential Guidance",
                    page="techniques",
                )
            )
        if t.hfato:
            errors.append(
                _make_issue(
                    "error",
                    "training.differential_guidance",
                    "Differential Guidance cannot be combined with HFATO because HFATO replaces the video target loss.",
                    label="Differential Guidance",
                    page="techniques",
                )
            )

    if t.keyframe_endpoint_training:
        keyframe_probs = [
            ("training.keyframe_first_frame_p", "Keyframe First Frame Probability", t.keyframe_first_frame_p),
            ("training.keyframe_last_frame_p", "Keyframe Last Frame Probability", t.keyframe_last_frame_p),
            ("training.keyframe_random_interior_p", "Keyframe Random Interior Probability", t.keyframe_random_interior_p),
        ]
        parsed_keyframe_probs: list[float] = []
        for field, label, raw_value in keyframe_probs:
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = float("nan")
            parsed_keyframe_probs.append(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                errors.append(
                    _make_issue(
                        "error",
                        field,
                        f"{label} must be a finite number in the inclusive range 0.0 to 1.0.",
                        label=label,
                        page="conditioning",
                    )
                )
        try:
            keyframe_max_random_interior = int(t.keyframe_max_random_interior)
        except (TypeError, ValueError):
            keyframe_max_random_interior = -1
        if keyframe_max_random_interior < 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.keyframe_max_random_interior",
                    "Keyframe Max Random Interior must be at least 0.",
                    label="Keyframe Max Random Interior",
                    page="conditioning",
                )
            )
        if parsed_keyframe_probs == [0.0, 0.0, 0.0]:
            warnings.append(
                _make_issue(
                    "warning",
                    "training.keyframe_endpoint_training",
                    "Endpoint Keyframe Training is enabled, but all keyframe probabilities are 0.",
                    label="Endpoint Keyframe Training",
                    page="conditioning",
                )
            )

    if t.video_anchor_training:
        try:
            video_anchor_probability = float(t.video_anchor_probability)
        except (TypeError, ValueError):
            video_anchor_probability = float("nan")
        if not math.isfinite(video_anchor_probability) or not 0.0 <= video_anchor_probability <= 1.0:
            errors.append(
                _make_issue(
                    "error",
                    "training.video_anchor_probability",
                    "Video Anchor Probability must be a finite number in the inclusive range 0.0 to 1.0.",
                    label="Video Anchor Probability",
                    page="techniques",
                )
            )
        try:
            video_anchor_count = int(t.video_anchor_count)
        except (TypeError, ValueError):
            video_anchor_count = -1
        if video_anchor_count < 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.video_anchor_count",
                    "Video Anchor Count must be at least 0.",
                    label="Video Anchor Count",
                    page="techniques",
                )
            )
        if str(t.video_anchor_strategy or "endpoints_random") == "random" and video_anchor_count < 1:
            errors.append(
                _make_issue(
                    "error",
                    "training.video_anchor_count",
                    "Video Anchor Count must be at least 1 when Video Anchor Strategy is random.",
                    label="Video Anchor Count",
                    page="techniques",
                )
            )
        if t.ltx2_mode == "audio" or t.ltx2_audio_only_model:
            errors.append(
                _make_issue(
                    "error",
                    "training.video_anchor_training",
                    "Video Anchor Training requires a video-target training path and cannot be used with audio-only training.",
                    label="Video Anchor Training",
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

    if t.sample_two_stage and not (_has_text(t.spatial_upsampler_path) or _has_text(getattr(t, "temporal_upsampler_path", ""))):
        errors.append(
            _make_issue(
                "error",
                "training.spatial_upsampler_path",
                "Spatial or Temporal Upsampler Path is required when Two-Stage sampling is enabled.",
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


def validate_full_finetune_config(config: ProjectConfig) -> dict[str, Any]:
    """Validate GUI full fine-tune config before launch."""
    t = config.full_finetune
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    effective_gemma_safetensors = _effective_gemma_safetensors(t.gemma_safetensors, config.default_gemma_safetensors, t.gemma_root)

    if not _has_full_finetune_checkpoint(config):
        errors.append(
            _make_issue(
                "error",
                "full_finetune.ltx2_checkpoint",
                "LTX-2 Checkpoint is required.",
                label="LTX-2 Checkpoint",
                page="full_finetune",
            )
        )

    if t.log_with == "tensorboard" and not _has_text(t.logging_dir):
        errors.append(
            _make_issue(
                "error",
                "full_finetune.logging_dir",
                "Log Dir is required when Logger is set to TensorBoard.",
                label="Log Dir",
                page="full_finetune",
            )
        )

    _validate_gemma_quantization_combo(
        errors=errors,
        gemma_load_in_8bit=t.gemma_load_in_8bit,
        gemma_load_in_4bit=t.gemma_load_in_4bit,
        gemma_safetensors=effective_gemma_safetensors,
        field_prefix="full_finetune",
        page="full_finetune",
    )

    if t.full_fp16 and t.full_bf16:
        message = "Full FP16 and Full BF16 cannot be enabled together."
        errors.append(_make_issue("error", "full_finetune.full_fp16", message, label="Full FP16", page="full_finetune"))
        errors.append(_make_issue("error", "full_finetune.full_bf16", message, label="Full BF16", page="full_finetune"))

    if getattr(t, "ltx2_compile_inner_blocks", False):
        if not t.compile:
            errors.append(
                _make_issue(
                    "error",
                    "full_finetune.ltx2_compile_inner_blocks",
                    "Compile inner LTX blocks requires torch.compile.",
                    label="Compile Inner LTX Blocks",
                    page="full_finetune",
                )
            )
        if t.blocks_to_swap not in (None, 0) or t.blockwise_checkpointing or t.gradient_checkpointing_cpu_offload:
            errors.append(
                _make_issue(
                    "error",
                    "full_finetune.ltx2_compile_inner_blocks",
                    "Compile inner LTX blocks requires resident weights and activations.",
                    label="Compile Inner LTX Blocks",
                    page="full_finetune",
                )
            )

    if getattr(t, "ltx2_bounded_activation_offload", False):
        conflicts = []
        if not t.gradient_checkpointing:
            conflicts.append("Gradient Checkpointing must be enabled")
        if t.gradient_checkpointing_cpu_offload:
            conflicts.append("Checkpoint CPU Offload")
        if t.blockwise_checkpointing:
            conflicts.append("Blockwise Checkpointing")
        if t.ltx2_partial_gradient_checkpointing:
            conflicts.append("Partial Gradient Checkpointing")
        if t.compile:
            conflicts.append("torch.compile")
        if t.ltx2_model_parallel:
            conflicts.append("Model Parallel")
        if t.ltx2_remote_stage:
            conflicts.append("Remote Stage")
        if conflicts:
            errors.append(
                _make_issue(
                    "error",
                    "full_finetune.ltx2_bounded_activation_offload",
                    "Bounded Activation Offload configuration conflict: " + ", ".join(conflicts) + ".",
                    label="Bounded Activation Offload",
                    page="full_finetune",
                )
            )
        max_inflight = t.ltx2_activation_offload_max_inflight
        keep_trailing = t.ltx2_activation_offload_keep_trailing
        min_mb = t.ltx2_activation_offload_min_mb
        if max_inflight is not None and max_inflight < 1:
            errors.append(
                _make_issue(
                    "error",
                    "full_finetune.ltx2_activation_offload_max_inflight",
                    "Activation offload maximum in-flight calls must be at least 1.",
                    label="Activation Offload In-Flight Limit",
                    page="full_finetune",
                )
            )
        if keep_trailing is not None and keep_trailing < 0:
            errors.append(
                _make_issue(
                    "error",
                    "full_finetune.ltx2_activation_offload_keep_trailing",
                    "Activation offload trailing block count cannot be negative.",
                    label="Activation Offload Trailing Blocks",
                    page="full_finetune",
                )
            )
        if min_mb is not None and min_mb < 0:
            errors.append(
                _make_issue(
                    "error",
                    "full_finetune.ltx2_activation_offload_min_mb",
                    "Activation offload minimum size cannot be negative.",
                    label="Activation Offload Minimum Size",
                    page="full_finetune",
                )
            )

    try:
        video_anchor_strength = float(t.video_anchor_strength)
    except (TypeError, ValueError):
        video_anchor_strength = float("nan")
    if not math.isfinite(video_anchor_strength) or not 0.0 <= video_anchor_strength <= 1.0:
        errors.append(
            _make_issue(
                "error",
                "full_finetune.video_anchor_strength",
                "Video Anchor Strength must be a finite number in the inclusive range 0.0 to 1.0.",
                label="Video Anchor Strength",
                page="full_finetune",
            )
        )
    elif video_anchor_strength != 1.0 and not t.ltx2_graded_conditioning:
        errors.append(
            _make_issue(
                "error",
                "full_finetune.ltx2_graded_conditioning",
                "Enable Graded Conditioning to use Video Anchor Strength below 1.0.",
                label="Graded Conditioning",
                page="full_finetune",
            )
        )

    if t.ltx2_causal_temporal_attention:
        if t.ltx2_mode == "audio" or t.ltx2_audio_only_model:
            errors.append(
                _make_issue(
                    "error",
                    "full_finetune.ltx2_causal_temporal_attention",
                    "Causal Temporal Attention requires a video training path.",
                    label="Causal Temporal Attention",
                    page="full_finetune",
                )
            )
        if not t.sdpa:
            errors.append(
                _make_issue(
                    "error",
                    "full_finetune.ltx2_causal_temporal_attention",
                    "Causal Temporal Attention requires SDPA.",
                    label="Causal Temporal Attention",
                    page="full_finetune",
                )
            )

    try:
        soft_av_sigma = float(t.ltx2_soft_av_alignment_sigma)
    except (TypeError, ValueError):
        soft_av_sigma = float("nan")
    if not math.isfinite(soft_av_sigma) or soft_av_sigma <= 0.0:
        errors.append(
            _make_issue(
                "error",
                "full_finetune.ltx2_soft_av_alignment_sigma",
                "Soft AV Alignment Sigma must be finite and greater than zero.",
                label="Soft AV Alignment Sigma",
                page="full_finetune",
            )
        )
    if t.ltx2_soft_av_alignment:
        if t.ltx2_mode != "av" or t.ltx2_audio_only_model:
            errors.append(
                _make_issue(
                    "error",
                    "full_finetune.ltx2_soft_av_alignment",
                    "Soft AV Alignment requires AV mode.",
                    label="Soft AV Alignment",
                    page="full_finetune",
                )
            )
        if not t.sdpa:
            errors.append(
                _make_issue(
                    "error",
                    "full_finetune.ltx2_soft_av_alignment",
                    "Soft AV Alignment requires SDPA.",
                    label="Soft AV Alignment",
                    page="full_finetune",
                )
            )

    if getattr(t, "int8_weights_w8a8", False):
        if not getattr(t, "int8_weights", False):
            message = "Int8 W8A8 requires Int8 Weights."
            errors.append(_make_issue("error", "full_finetune.int8_weights_w8a8", message, label="Int8 W8A8", page="full_finetune"))
            errors.append(_make_issue("error", "full_finetune.int8_weights", message, label="Int8 Weights", page="full_finetune"))
        if int(getattr(t, "int8_weights_group_size", 0) or 0) != 0:
            message = "Int8 W8A8 requires Int8 Weights Group Size = 0."
            errors.append(_make_issue("error", "full_finetune.int8_weights_w8a8", message, label="Int8 W8A8", page="full_finetune"))
            errors.append(
                _make_issue(
                    "error",
                    "full_finetune.int8_weights_group_size",
                    message,
                    label="Int8 Weights Group Size",
                    page="full_finetune",
                )
            )
        if float(getattr(t, "int8_weights_sparse_ratio", 0.0) or 0.0) != 0.0:
            message = "Int8 W8A8 requires Int8 Weights Sparse Ratio = 0."
            errors.append(_make_issue("error", "full_finetune.int8_weights_w8a8", message, label="Int8 W8A8", page="full_finetune"))
            errors.append(
                _make_issue(
                    "error",
                    "full_finetune.int8_weights_sparse_ratio",
                    message,
                    label="Int8 Weights Sparse Ratio",
                    page="full_finetune",
                )
            )

    if t.qgalore_full_ft:
        opt = str(t.optimizer_type or "").lower()
        qgalore_aliases = {
            "",
            "qgalore",
            "q_galore",
            "qgaloreadamw8bit",
            "q_galore_adamw8bit",
            "q-galore-adamw8bit",
            "qapollo",
            "q_apollo",
            "qapollo_adamw",
            "qapolloadamw",
            "q_apollo_adamw",
            "apollo_torch.qapolloadamw",
            "apollo_torch.q_apollo.adamw",
        }
        if opt not in qgalore_aliases:
            errors.append(
                _make_issue(
                    "error",
                    "full_finetune.optimizer_type",
                    "Quantized full fine-tuning requires optimizer type QGaLoreAdamW8bit or QAPOLLOAdamW.",
                    label="Optimizer",
                    page="full_finetune",
                )
            )
        if not t.fused_backward_pass:
            errors.append(
                _make_issue(
                    "error",
                    "full_finetune.fused_backward_pass",
                    "Q-GaLore full fine-tuning requires fused backward.",
                    label="Fused Backward",
                    page="full_finetune",
                )
            )
        if float(t.max_grad_norm or 0.0) != 0.0:
            errors.append(
                _make_issue(
                    "error",
                    "full_finetune.max_grad_norm",
                    "Q-GaLore fused backward requires Max Grad Norm = 0.",
                    label="Max Grad Norm",
                    page="full_finetune",
                )
            )
        if t.fp8_base or t.fp8_scaled:
            message = "Q-GaLore full fine-tuning cannot be combined with FP8 base/scaled loading."
            errors.append(_make_issue("error", "full_finetune.fp8_base", message, label="FP8 Base", page="full_finetune"))
            errors.append(_make_issue("error", "full_finetune.fp8_scaled", message, label="FP8 Scaled", page="full_finetune"))
        if t.nf4_base:
            errors.append(
                _make_issue(
                    "error",
                    "full_finetune.nf4_base",
                    "Q-GaLore full fine-tuning cannot be combined with NF4 base loading.",
                    label="NF4 Base",
                    page="full_finetune",
                )
            )

    return _build_report(errors, warnings)


def _has_remote_stage_server_checkpoint(config: ProjectConfig) -> bool:
    return _has_text(config.remote_stage_server.ltx2_checkpoint) or _has_text(config.default_ltx2_checkpoint)


def validate_remote_stage_server_config(config: ProjectConfig) -> dict[str, Any]:
    """Validate the remote stage server launcher config."""
    r = config.remote_stage_server
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if not _has_remote_stage_server_checkpoint(config):
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.ltx2_checkpoint",
                "LTX-2 Checkpoint is required.",
                label="LTX-2 Checkpoint",
                page="training",
            )
        )

    if _has_text(r.ltx2_checkpoint):
        checkpoint_path = _resolve_project_path(config, r.ltx2_checkpoint)
        if checkpoint_path is not None and not checkpoint_path.exists():
            errors.append(
                _make_issue(
                    "error",
                    "remote_stage_server.ltx2_checkpoint",
                    f"LTX-2 Checkpoint file not found: {checkpoint_path}",
                    label="LTX-2 Checkpoint",
                    page="training",
                )
            )

    if not _has_text(r.bind):
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.bind",
                "Bind host is required.",
                label="Bind",
                page="training",
            )
        )

    if not (0 < int(r.port) < 65536):
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.port",
                "Port must be in 1..65535.",
                label="Port",
                page="training",
            )
        )

    if not _has_text(r.device):
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.device",
                "Device is required.",
                label="Device",
                page="training",
            )
        )

    if r.split < 0:
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.split",
                "Split block index must be >= 0.",
                label="Split",
                page="training",
            )
        )
    if r.end != -1 and r.end <= r.split:
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.end",
                "End block index must be greater than Split, or left at -1.",
                label="End",
                page="training",
            )
        )

    if r.stage_only_device_placement and r.full_model_device_placement:
        message = "Stage-only device placement and full-model device placement cannot both be enabled."
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.stage_only_device_placement",
                message,
                label="Stage Only Device Placement",
                page="training",
            )
        )
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.full_model_device_placement",
                message,
                label="Full Model Device Placement",
                page="training",
            )
        )

    if r.block_only_load and r.full_model_device_placement:
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.block_only_load",
                "Block-only load is incompatible with full-model device placement.",
                label="Block Only Load",
                page="training",
            )
        )

    if not r.trainable and r.trainable_scope != "auto":
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.trainable_scope",
                "Trainable scope requires trainable mode.",
                label="Trainable Scope",
                page="training",
            )
        )

    if r.int8_block_size <= 0:
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.int8_block_size",
                "Int8 block size must be greater than 0.",
                label="Int8 Block Size",
                page="training",
            )
        )

    if r.nf4_block_size <= 0:
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.nf4_block_size",
                "NF4 block size must be greater than 0.",
                label="NF4 Block Size",
                page="training",
            )
        )

    if r.split_attn_chunk_size < 0:
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.split_attn_chunk_size",
                "Split-attention chunk size must be >= 0.",
                label="Split Attn Chunk Size",
                page="training",
            )
        )

    if r.ffn_chunk_size < 0:
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.ffn_chunk_size",
                "FFN chunk size must be >= 0.",
                label="FFN Chunk Size",
                page="training",
            )
        )

    if r.network_lr is not None and r.network_lr <= 0:
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.network_lr",
                "Network learning rate must be greater than 0.",
                label="Network LR",
                page="training",
            )
        )

    if r.learning_rate is not None and r.learning_rate <= 0:
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.learning_rate",
                "Learning rate must be greater than 0.",
                label="Learning Rate",
                page="training",
            )
        )

    return _build_report(errors, warnings)


def validate_remote_stage_launcher_config(config: ProjectConfig) -> dict[str, Any]:
    """Validate the master-side SSH orchestration config."""
    launcher = config.remote_stage_launcher
    server = config.remote_stage_server
    training = config.training
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if not training.ltx2_remote_stage:
        errors.append(
            _make_issue(
                "error",
                "training.ltx2_remote_stage",
                "Enable Remote Stage before launching remote slaves.",
                label="Remote Stage",
                page="training",
            )
        )

    if not _has_text(launcher.remote_root):
        errors.append(
            _make_issue(
                "error",
                "remote_stage_launcher.remote_root",
                "Remote root path is required.",
                label="Remote Root",
                page="training",
            )
        )

    if not _has_text(launcher.remote_python):
        errors.append(
            _make_issue(
                "error",
                "remote_stage_launcher.remote_python",
                "Remote Python executable is required.",
                label="Remote Python",
                page="training",
            )
        )

    if not (0 < int(launcher.ssh_port) < 65536):
        errors.append(
            _make_issue(
                "error",
                "remote_stage_launcher.ssh_port",
                "SSH port must be in 1..65535.",
                label="SSH Port",
                page="training",
            )
        )

    if launcher.ready_timeout <= 0:
        errors.append(
            _make_issue(
                "error",
                "remote_stage_launcher.ready_timeout",
                "Ready timeout must be greater than 0.",
                label="Ready Timeout",
                page="training",
            )
        )

    if launcher.ready_poll_interval <= 0:
        errors.append(
            _make_issue(
                "error",
                "remote_stage_launcher.ready_poll_interval",
                "Ready poll interval must be greater than 0.",
                label="Ready Poll Interval",
                page="training",
            )
        )

    if server.bind in {"127.0.0.1", "localhost", "::1"}:
        errors.append(
            _make_issue(
                "error",
                "remote_stage_server.bind",
                "Remote stage servers are bound to loopback only; remote orchestration will not be reachable from other machines.",
                label="Bind",
                page="training",
            )
        )

    try:
        specs, specs_error = _parse_remote_stage_specs(training.ltx2_remote_stage_specs)
    except Exception as exc:
        errors.append(
            _make_issue(
                "error",
                "training.ltx2_remote_stage_specs",
                str(exc),
                label="Remote Stage Specs",
                page="training",
            )
        )
        specs = []
    else:
        if specs_error is not None:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_remote_stage_specs",
                    specs_error,
                    label="Remote Stage Specs",
                    page="training",
                )
            )
        elif not specs and training.ltx2_remote_stage_split < 0:
            errors.append(
                _make_issue(
                    "error",
                    "training.ltx2_remote_stage_split",
                    "Set either Remote Stage Specs or a single remote stage split before launching slaves.",
                    label="Remote Stage Split",
                    page="training",
                )
            )
        else:
            for idx, spec in enumerate(specs):
                host = spec[0]
                if host in {"127.0.0.1", "localhost", "::1"}:
                    errors.append(
                        _make_issue(
                            "error",
                            "training.ltx2_remote_stage_specs",
                            f"Remote stage spec {idx} targets loopback host {host!r}; use the machine's reachable LAN address or DNS name.",
                            label="Remote Stage Specs",
                            page="training",
                        )
                    )
                    break

    if launcher.ssh_extra_args:
        try:
            shlex.split(launcher.ssh_extra_args, posix=False)
        except ValueError as exc:
            errors.append(
                _make_issue(
                    "error",
                    "remote_stage_launcher.ssh_extra_args",
                    f"Invalid SSH extra args: {exc}",
                    label="SSH Extra Args",
                    page="training",
                )
            )

    return _build_report(errors, warnings)


def _validate_sample_prompt_path(
    config: ProjectConfig, raw_path: str | None, *, errors: list[dict[str, Any]], field: str, label: str, page: str
) -> None:
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
    if c.model_type == "minimax_h3":
        requires_video, requires_audio = _h3_required_vaes(config)
        if requires_video and not _has_text(c.h3_video_vae):
            errors.append(
                _make_issue(
                    "error",
                    "caching.h3_video_vae",
                    "MiniMax H3 video VAE is required by the configured visual targets or references.",
                    label="H3 Video VAE",
                    page="caching",
                )
            )
        if requires_audio and not _has_text(c.h3_audio_vae):
            errors.append(
                _make_issue(
                    "error",
                    "caching.h3_audio_vae",
                    "MiniMax H3 audio VAE is required by the configured audio targets or references.",
                    label="H3 Audio VAE",
                    page="caching",
                )
            )
        if not config.dataset.datasets and not config.dataset.validation_datasets:
            errors.append(
                _make_issue(
                    "error",
                    "dataset.datasets",
                    "Add at least one H3 training or validation dataset before caching latents.",
                    label="Datasets",
                    page="dataset",
                )
            )
        for collection, rows in (
            ("datasets", config.dataset.datasets),
            ("validation_datasets", config.dataset.validation_datasets),
        ):
            for index, entry in enumerate(rows):
                _validate_h3_dataset_entry(
                    entry,
                    index,
                    errors=errors,
                    warnings=warnings,
                    collection=collection,
                )
        if c.h3_image_mode != "none" and c.h3_task != "fl2va":
            errors.append(
                _make_issue(
                    "error",
                    "caching.h3_image_mode",
                    "H3 conditioned-image caching requires the FL2VA task.",
                    label="H3 Image Mode",
                    page="caching",
                )
            )
        if c.h3_image_frame_count is not None and (c.h3_image_frame_count < 5 or (c.h3_image_frame_count - 5) % 17 != 0):
            errors.append(
                _make_issue(
                    "error",
                    "caching.h3_image_frame_count",
                    "H3 image frame count must satisfy frame_count % 17 == 5.",
                    label="H3 Image Frame Count",
                    page="caching",
                )
            )
        return _build_report(errors, warnings)

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
    if c.model_type == "minimax_h3":
        if not _has_text(c.h3_text_encoder):
            errors.append(
                _make_issue(
                    "error",
                    "caching.h3_text_encoder",
                    "MiniMax H3 text encoder is required.",
                    label="H3 Text Encoder",
                    page="caching",
                )
            )
        if not _has_text(c.h3_tokenizer):
            errors.append(
                _make_issue(
                    "error",
                    "caching.h3_tokenizer",
                    "MiniMax H3 tokenizer directory is required.",
                    label="H3 Tokenizer",
                    page="caching",
                )
            )
        if c.h3_text_encoder_dtype != "bfloat16":
            errors.append(
                _make_issue(
                    "error",
                    "caching.h3_text_encoder_dtype",
                    "MiniMax H3 Qwen3-VL conditioning and cache outputs require bfloat16.",
                    label="H3 Text Encoder Dtype",
                    page="caching",
                )
            )
        if c.h3_text_encoder_quantization != "none" and c.device and not c.device.lower().startswith("cuda"):
            errors.append(
                _make_issue(
                    "error",
                    "caching.device",
                    "MiniMax H3 quantized text-encoder modes require a CUDA device.",
                    label="Device",
                    page="caching",
                )
            )
        if not 0 <= c.h3_text_encoder_blocks_to_stream <= 50:
            errors.append(
                _make_issue(
                    "error",
                    "caching.h3_text_encoder_blocks_to_stream",
                    "H3 text-encoder streamed blocks must be between 0 and 50.",
                    label="H3 Text Encoder Blocks To Stream",
                    page="caching",
                )
            )
        if c.h3_nvfp4_scaled_mm and c.h3_text_encoder_quantization != "nvfp4_awq":
            errors.append(
                _make_issue(
                    "error",
                    "caching.h3_nvfp4_scaled_mm",
                    "H3 NVFP4 scaled_mm requires NVFP4/AWQ text-encoder quantization.",
                    label="H3 NVFP4 scaled_mm",
                    page="caching",
                )
            )
        if c.h3_text_visual_max_pixels < 0:
            errors.append(
                _make_issue(
                    "error",
                    "caching.h3_text_visual_max_pixels",
                    "H3 text visual pixel cap must be non-negative.",
                    label="H3 Text Visual Max Pixels",
                    page="caching",
                )
            )
        if c.h3_image_mode != "none" and c.h3_task != "fl2va":
            errors.append(
                _make_issue(
                    "error",
                    "caching.h3_image_mode",
                    "H3 conditioned-image text caching requires the FL2VA task.",
                    label="H3 Image Mode",
                    page="caching",
                )
            )
        return _build_report(errors, warnings)
    effective_gemma_safetensors = _effective_gemma_safetensors(c.gemma_safetensors, config.default_gemma_safetensors, c.gemma_root)

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


def validate_cache_preview_config(config: ProjectConfig) -> dict[str, Any]:
    c = config.caching
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    preview_input = _effective_cache_preview_input(config)
    if not _has_text(preview_input):
        errors.append(
            _make_issue(
                "error",
                "caching.cache_preview_input",
                "Cache Preview Input is required, or at least one dataset must define a cache directory.",
                label="Cache Preview Input",
                page="caching",
            )
        )
    else:
        path = _resolve_project_path(config, preview_input)
        if path is not None and not path.exists():
            errors.append(
                _make_issue(
                    "error",
                    "caching.cache_preview_input",
                    f"Cache Preview Input not found: {path}",
                    label="Cache Preview Input",
                    page="caching",
                )
            )

    if c.cache_preview_decode and not _has_cache_text_checkpoint(config):
        errors.append(
            _make_issue(
                "error",
                "caching.ltx2_checkpoint",
                "LTX-2 Checkpoint is required when Decode Previews is enabled.",
                label="LTX-2 Checkpoint",
                page="caching",
            )
        )

    if c.cache_preview_limit is not None and c.cache_preview_limit < 1:
        errors.append(
            _make_issue(
                "error",
                "caching.cache_preview_limit",
                "Cache Preview Limit must be at least 1.",
                label="Preview Limit",
                page="caching",
            )
        )

    if c.cache_preview_fps <= 0:
        errors.append(
            _make_issue(
                "error",
                "caching.cache_preview_fps",
                "Cache Preview FPS must be greater than 0.",
                label="Preview FPS",
                page="caching",
            )
        )

    companion_roles = {part.strip().lower() for part in c.cache_preview_require_companions.split(",") if part.strip()}
    invalid_companion_roles = sorted(companion_roles - {"video", "audio", "text"})
    if invalid_companion_roles:
        errors.append(
            _make_issue(
                "error",
                "caching.cache_preview_require_companions",
                f"Unknown required companion role(s): {', '.join(invalid_companion_roles)}.",
                label="Required Companions",
                page="caching",
            )
        )

    if c.cache_preview_av_duration_tolerance < 0:
        errors.append(
            _make_issue(
                "error",
                "caching.cache_preview_av_duration_tolerance",
                "AV Duration Tolerance must be zero or greater.",
                label="AV Duration Tolerance",
                page="caching",
            )
        )

    return _build_report(errors, warnings)


def validate_inference_config(config: ProjectConfig) -> dict[str, Any]:
    i = config.inference
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if i.model_type == "minimax_h3":
        if not _has_text(i.h3_model):
            errors.append(
                _make_issue(
                    "error", "inference.h3_model", "MiniMax H3 model checkpoint is required.", label="H3 Model", page="inference"
                )
            )
        if not _has_text(i.prompt):
            errors.append(
                _make_issue(
                    "error", "inference.prompt", "Prompt is required for MiniMax H3 inference.", label="Prompt", page="inference"
                )
            )
        required_components = [
            ("h3_text_encoder", config.caching.h3_text_encoder, "H3 Text Encoder"),
            ("h3_tokenizer", config.caching.h3_tokenizer, "H3 Tokenizer"),
            ("h3_video_vae", config.caching.h3_video_vae, "H3 Video VAE"),
        ]
        if i.h3_image_mode == "none":
            required_components.append(("h3_audio_vae", config.caching.h3_audio_vae, "H3 Audio VAE"))
        for field, fallback, label in required_components:
            if not _has_text(getattr(i, field)) and not _has_text(fallback):
                errors.append(_make_issue("error", f"inference.{field}", f"{label} is required.", label=label, page="inference"))
        if not 5 <= i.h3_duration <= 15:
            errors.append(
                _make_issue(
                    "error",
                    "inference.h3_duration",
                    "MiniMax H3 duration must be between 5 and 15 seconds.",
                    label="Duration",
                    page="inference",
                )
            )
        if (i.width is None) != (i.height is None):
            errors.append(
                _make_issue(
                    "error",
                    "inference.width",
                    "H3 explicit canvas requires both width and height.",
                    label="H3 Canvas",
                    page="inference",
                )
            )
        if i.fp8_base and i.h3_int8_convrot_base:
            errors.append(
                _make_issue(
                    "error",
                    "inference.h3_int8_convrot_base",
                    "Choose scaled FP8 or pruned INT8 ConvRot, not both.",
                    label="H3 Quantized Base",
                    page="inference",
                )
            )
        if not 0 <= i.h3_blocks_to_swap <= 50:
            errors.append(
                _make_issue(
                    "error",
                    "inference.h3_blocks_to_swap",
                    "H3 inference block swap count must be between 0 and 50.",
                    label="H3 Blocks To Swap",
                    page="inference",
                )
            )
        if i.h3_block_swap_ring_size < 1:
            errors.append(
                _make_issue(
                    "error",
                    "inference.h3_block_swap_ring_size",
                    "H3 block-swap ring size must be at least 1.",
                    label="H3 Block Swap Ring Size",
                    page="inference",
                )
            )
        if not 0 <= i.h3_text_encoder_blocks_to_stream <= 50:
            errors.append(
                _make_issue(
                    "error",
                    "inference.h3_text_encoder_blocks_to_stream",
                    "H3 text-encoder streamed blocks must be between 0 and 50.",
                    label="H3 Text Encoder Blocks To Stream",
                    page="inference",
                )
            )
        effective_text_quantization = (
            i.h3_text_encoder_quantization
            if i.h3_text_encoder_quantization != "none"
            else config.caching.h3_text_encoder_quantization
        )
        if (i.h3_nvfp4_scaled_mm or config.caching.h3_nvfp4_scaled_mm) and effective_text_quantization != "nvfp4_awq":
            errors.append(
                _make_issue(
                    "error",
                    "inference.h3_nvfp4_scaled_mm",
                    "H3 NVFP4 scaled_mm requires NVFP4/AWQ text-encoder quantization.",
                    label="H3 NVFP4 scaled_mm",
                    page="inference",
                )
            )
        if i.h3_text_visual_max_pixels < 0:
            errors.append(
                _make_issue(
                    "error",
                    "inference.h3_text_visual_max_pixels",
                    "H3 text visual pixel cap must be non-negative.",
                    label="H3 Text Visual Max Pixels",
                    page="inference",
                )
            )
        if i.h3_image_mode == "first":
            if not _has_text(i.h3_first_frame):
                errors.append(
                    _make_issue(
                        "error",
                        "inference.h3_first_frame",
                        "H3 first-image mode requires a first-frame image.",
                        label="First Frame",
                        page="inference",
                    )
                )
            if _has_text(i.h3_last_frame) and i.h3_last_frame != i.h3_first_frame:
                errors.append(
                    _make_issue(
                        "error",
                        "inference.h3_last_frame",
                        "H3 first-image mode does not accept a different last-frame image.",
                        label="Last Frame",
                        page="inference",
                    )
                )
        if i.h3_image_mode == "first_last" and not (_has_text(i.h3_first_frame) and _has_text(i.h3_last_frame)):
            errors.append(
                _make_issue(
                    "error",
                    "inference.h3_image_mode",
                    "H3 first/last-image mode requires both endpoint images.",
                    label="H3 Image Mode",
                    page="inference",
                )
            )
        if i.h3_image_mode != "none" and (i.h3_image_frame_count < 5 or (i.h3_image_frame_count - 5) % 17 != 0):
            errors.append(
                _make_issue(
                    "error",
                    "inference.h3_image_frame_count",
                    "H3 image frame count must satisfy frame_count % 17 == 5.",
                    label="H3 Image Frame Count",
                    page="inference",
                )
            )
        if i.h3_image_mode != "none" and not 0 <= i.h3_select_frame < i.h3_image_frame_count:
            errors.append(
                _make_issue(
                    "error",
                    "inference.h3_select_frame",
                    "H3 selected output frame must be inside the generated image-frame range.",
                    label="H3 Select Frame",
                    page="inference",
                )
            )
        return _build_report(errors, warnings)
    effective_gemma_safetensors = _effective_gemma_safetensors(i.gemma_safetensors, config.default_gemma_safetensors, i.gemma_root)

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

    if (i.sample_two_stage or i.sampling_preset == "distilled_two_stage") and not (
        _has_text(i.spatial_upsampler_path) or _has_text(getattr(i, "temporal_upsampler_path", ""))
    ):
        errors.append(
            _make_issue(
                "error",
                "inference.spatial_upsampler_path",
                "Spatial or Temporal Upsampler Path is required when Two-Stage sampling is enabled.",
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


def _reward_routes(weights: dict[str, float]) -> dict[str, str]:
    """Map each selected reward name to its declared route (video|audio|sync)."""
    try:
        from musubi_tuner.ltx2_rewards import get_reward_cls
    except Exception:  # pragma: no cover - registry import should always succeed
        return {}
    routes: dict[str, str] = {}
    for name in weights:
        try:
            routes[name] = getattr(get_reward_cls(name), "route", "")
        except Exception:
            routes[name] = ""
    return routes


def validate_rl_config(config: ProjectConfig, phase: str | None = None) -> dict[str, Any]:
    """Validate the NFT/GRPO RL post-training config before launch.

    ``phase`` ("cache_rollouts" | "train_rl") selects which phase's path requirements to check;
    when omitted it falls back to ``config.rl.phase``. The dashboard passes it per process type so
    the Phase A and Phase B forms each validate their own phase independently.
    """
    rl = config.rl
    t = config.training
    rl_loss = getattr(rl, "rl_loss", "nft") or "nft"
    effective_phase = phase or rl.phase
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if not _has_training_checkpoint(config):
        errors.append(
            _make_issue(
                "error",
                "training.ltx2_checkpoint",
                "LTX-2 Checkpoint is required.",
                label="LTX-2 Checkpoint",
                page="rl",
            )
        )

    # Phase A may start from either an existing LoRA or a fresh adapter. Offline Phase B is stricter:
    # it must load the exact `old` snapshot that generated the cache, or the snapshot hash invariant
    # will fail before training.
    if effective_phase == "train_rl" and not rl.online and rl_loss != "refl" and not _has_text(t.network_weights):
        errors.append(
            _make_issue(
                "error",
                "training.network_weights",
                "Offline Phase B requires Network Weights to point at the Phase A `old` snapshot "
                "(the cache snapshot hash must match before training starts).",
                label="Network Weights",
                page="rl",
            )
        )
    elif effective_phase == "cache_rollouts" and not _has_text(t.network_weights):
        warnings.append(
            _make_issue(
                "warning",
                "training.network_weights",
                "Phase A will start from a fresh LoRA because Network Weights is empty. Set Save `old` "
                "Snapshot, then load that snapshot as Network Weights for Phase B.",
                label="Network Weights",
                page="rl",
            )
        )

    # Reward spec must be parseable and reference only registered rewards (incl. plugins).
    weights, reward_error = _parse_reward_spec_for_validation(rl.reward_fn, getattr(rl, "reward_plugins", ""))
    if reward_error is not None:
        errors.append(
            _make_issue(
                "error",
                "rl.reward_fn",
                f"Reward spec is invalid: {reward_error}",
                label="Reward Function",
                page="rl",
            )
        )
    elif not weights:
        errors.append(
            _make_issue(
                "error",
                "rl.reward_fn",
                "Reward spec must select at least one reward (e.g. 'hpsv3:1.0').",
                label="Reward Function",
                page="rl",
            )
        )
    else:
        # Branch-aware: audio/sync rewards need an audio-enabled (AV) transformer to route their
        # advantage to the audio branch.
        routes = _reward_routes(weights)
        needs_audio = {name for name, route in routes.items() if route in {"audio", "sync"}}
        if needs_audio and t.ltx2_mode == "video":
            errors.append(
                _make_issue(
                    "error",
                    "rl.reward_fn",
                    f"Reward(s) {sorted(needs_audio)} route to the audio/sync branch and require LTX2 Mode = av "
                    "(or audio). The narrow video-only RL path supports video-route rewards only.",
                    label="Reward Function",
                    page="rl",
                )
            )

    # refl (differentiable-reward backprop) is online-only and every reward must be differentiable.
    if rl_loss == "refl":
        if not _has_text(rl.rl_prompts):
            errors.append(
                _make_issue(
                    "error",
                    "rl.rl_prompts",
                    "refl (differentiable-reward backprop) generates rollouts inline and requires a Prompts "
                    "file (there is no rollout cache to replay).",
                    label="Prompts",
                    page="rl",
                )
            )
        if reward_error is None and weights:
            try:
                from musubi_tuner.ltx2_rewards import get_reward_cls

                blackbox = sorted(n for n in weights if getattr(get_reward_cls(n), "kind", "blackbox") != "differentiable")
            except Exception:
                blackbox = []
            if blackbox:
                errors.append(
                    _make_issue(
                        "error",
                        "rl.reward_fn",
                        f"refl backprops the reward, so every selected reward must be differentiable, but "
                        f"{blackbox} is/are black-box. Use a differentiable reward (e.g. latent_energy) or "
                        "switch to a policy-gradient rule (nft/rwr/dpo/ppo).",
                        label="Reward Function",
                        page="rl",
                    )
                )
        if getattr(rl, "refl_av", False) and t.ltx2_mode != "av":
            errors.append(
                _make_issue(
                    "error",
                    "rl.refl_av",
                    "AV ReFL requires LTX2 Mode = av.",
                    label="AV ReFL",
                    page="rl",
                )
            )
        if t.ltx2_mode == "av" and not getattr(rl, "refl_av", False):
            errors.append(
                _make_issue(
                    "error",
                    "rl.refl_av",
                    "Enable AV ReFL to use the refl update rule in AV mode.",
                    label="AV ReFL",
                    page="rl",
                )
            )

    # --reward_args must be parseable key=value entries.
    for entry in _split_cli_args(rl.reward_args):
        if "=" not in entry:
            errors.append(
                _make_issue(
                    "error",
                    "rl.reward_args",
                    f"Reward Args entry '{entry}' must be key=value.",
                    label="Reward Args",
                    page="rl",
                )
            )

    # K == GRPO group size: group-relative advantages need at least 2 samples for non-zero variance.
    if rl.rl_group_size < 2:
        errors.append(
            _make_issue(
                "error",
                "rl.rl_group_size",
                "RL Group Size (K) must be at least 2 for GRPO group-relative advantages.",
                label="RL Group Size",
                page="rl",
            )
        )

    # NFT coefficients
    if rl.nft_beta_mix <= 0.0:
        errors.append(_make_issue("error", "rl.nft_beta_mix", "NFT Beta Mix must be > 0.", label="NFT Beta Mix", page="rl"))
    if rl.nft_kl_beta < 0.0:
        errors.append(
            _make_issue(
                "error",
                "rl.nft_kl_beta",
                "Reference MSE Beta must be >= 0.",
                label="Reference MSE Beta",
                page="rl",
            )
        )
    if rl.nft_adv_clip_max <= 0.0:
        errors.append(
            _make_issue(
                "error",
                "rl.nft_adv_clip_max",
                "NFT Advantage Clip Max must be > 0.",
                label="NFT Adv Clip Max",
                page="rl",
            )
        )

    # Update-rule hyperparameters (only the active rule's value is used at train time).
    rl_loss = getattr(rl, "rl_loss", "nft") or "nft"
    if rl_loss == "rwr" and rl.rwr_temperature <= 0.0:
        errors.append(
            _make_issue("error", "rl.rwr_temperature", "RWR Temperature must be > 0.", label="RWR Temperature", page="rl")
        )
    if rl_loss == "dpo" and rl.dpo_beta <= 0.0:
        errors.append(_make_issue("error", "rl.dpo_beta", "DPO Beta must be > 0.", label="DPO Beta", page="rl"))
    if rl_loss == "ppo":
        if rl.ppo_clip_eps <= 0.0:
            errors.append(_make_issue("error", "rl.ppo_clip_eps", "PPO Clip Epsilon must be > 0.", label="PPO Clip Eps", page="rl"))
        if not (0.0 < getattr(rl, "rl_sde_eta", 1.0) <= 1.0):
            errors.append(
                _make_issue(
                    "error",
                    "rl.rl_sde_eta",
                    "SDE eta must be in (0, 1] for PPO: at eta=0 the step is deterministic and the PPO gradient is zero.",
                    label="SDE eta",
                    page="rl",
                )
            )

    if rl.rl_timesteps_per_sample < 1:
        errors.append(
            _make_issue(
                "error",
                "rl.rl_timesteps_per_sample",
                "RL Timesteps Per Sample must be at least 1.",
                label="RL Timesteps Per Sample",
                page="rl",
            )
        )
    if rl.rl_max_steps < 0:
        errors.append(
            _make_issue("error", "rl.rl_max_steps", "RL Max Steps must be >= 0 (0 = one pass).", label="RL Max Steps", page="rl")
        )

    # Phase-specific path requirements.
    if effective_phase == "cache_rollouts":
        if not _has_text(rl.rl_rollout_cache):
            errors.append(
                _make_issue(
                    "error",
                    "rl.rl_rollout_cache",
                    "Rollout Cache directory is required for Phase A (cache_rollouts).",
                    label="Rollout Cache",
                    page="rl",
                )
            )
        if not _has_text(rl.rl_prompts):
            errors.append(
                _make_issue(
                    "error",
                    "rl.rl_prompts",
                    "Prompts file is required for Phase A (cache_rollouts).",
                    label="RL Prompts",
                    page="rl",
                )
            )
        else:
            prompts_path = _resolve_project_path(config, rl.rl_prompts)
            if prompts_path is not None and not prompts_path.exists():
                errors.append(
                    _make_issue(
                        "error",
                        "rl.rl_prompts",
                        f"RL Prompts file not found: {prompts_path}",
                        label="RL Prompts",
                        page="rl",
                    )
                )
        if not _has_training_gemma_source(config):
            message = "Gemma Root or Gemma Safetensors is required to encode RL prompts in Phase A."
            errors.append(_make_issue("error", "training.gemma_root", message, label="Gemma Root", page="rl"))
            errors.append(_make_issue("error", "training.gemma_safetensors", message, label="Gemma Safetensors", page="rl"))
    else:  # train_rl
        if rl.online:
            if not _has_text(rl.rl_prompts):
                errors.append(
                    _make_issue(
                        "error",
                        "rl.rl_prompts",
                        "Prompts file is required for online RL training.",
                        label="RL Prompts",
                        page="rl",
                    )
                )
            warnings.append(
                _make_issue(
                    "warning",
                    "rl.online",
                    "Online RL is experimental and not cleanly VRAM-flat. Prefer the offline (cache replay) path: "
                    "run Phase A (cache_rollouts) then Phase B with a rollout cache.",
                    label="Online RL",
                    page="rl",
                )
            )
        else:
            if not _has_text(rl.rl_rollout_cache):
                errors.append(
                    _make_issue(
                        "error",
                        "rl.rl_rollout_cache",
                        "Rollout Cache directory is required for offline Phase B (train_rl). "
                        "Generate it with Phase A, or enable Online RL.",
                        label="Rollout Cache",
                        page="rl",
                    )
                )

    return _build_report(errors, warnings)


def validate_process_config(proc_type: str, config: ProjectConfig) -> dict[str, Any]:
    if proc_type == "training":
        return validate_training_config(config)
    if proc_type == "full_finetune":
        return validate_full_finetune_config(config)
    if proc_type == "remote_stage_server":
        return validate_remote_stage_server_config(config)
    if proc_type == "remote_stage_launcher":
        return validate_remote_stage_launcher_config(config)
    if proc_type == "cache_latents":
        return validate_cache_latents_config(config)
    if proc_type == "cache_text":
        return validate_cache_text_config(config)
    if proc_type == "cache_preview":
        return validate_cache_preview_config(config)
    if proc_type == "inference":
        return validate_inference_config(config)
    if proc_type in ("rl", "rl_cache_rollouts", "rl_train"):
        phase = {"rl_cache_rollouts": "cache_rollouts", "rl_train": "train_rl"}.get(proc_type)
        return validate_rl_config(config, phase=phase)
    return _build_report([], [])
