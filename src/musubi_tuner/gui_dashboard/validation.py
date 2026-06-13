"""Shared GUI validation rules for process launch."""

from __future__ import annotations

import math
import shlex
from pathlib import Path
from typing import Any

from musubi_tuner.gui_dashboard.cli_defaults import get_ltx2_training_network_module_default
from musubi_tuner.gui_dashboard.project_schema import DatasetEntry, ProjectConfig
from musubi_tuner.ltx2_av_cross_grad_surgery import parse_av_cross_grad_surgery_args
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
    entry: DatasetEntry, index: int, *, errors: list[dict[str, Any]], warnings: list[dict[str, Any]]
) -> None:
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


def _has_full_finetune_checkpoint(config: ProjectConfig) -> bool:
    return _has_text(config.full_finetune.ltx2_checkpoint) or _has_text(config.default_ltx2_checkpoint)


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

    if getattr(t, "use_rslora", False):
        if t.use_dora_oft:
            message = "rsLoRA is not supported with DoRA-OFT/DoKr-OFT."
            errors.append(_make_issue("error", "training.use_rslora", message, label="rsLoRA", page="training"))
            errors.append(_make_issue("error", "training.use_dora_oft", message, label="DoRA-OFT/DoKr-OFT", page="training"))
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
            differential_guidance_scale = float(t.differential_guidance_scale)
        except (TypeError, ValueError):
            differential_guidance_scale = float("nan")
        if not math.isfinite(differential_guidance_scale):
            errors.append(
                _make_issue(
                    "error",
                    "training.differential_guidance_scale",
                    "Differential Guidance Scale must be a finite number.",
                    label="Differential Guidance Scale",
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


def validate_full_finetune_config(config: ProjectConfig) -> dict[str, Any]:
    """Validate GUI full fine-tune config before launch."""
    t = config.full_finetune
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    effective_gemma_safetensors = _effective_gemma_safetensors(t.gemma_safetensors, config.default_gemma_safetensors)

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
    if effective_phase == "train_rl" and not rl.online and not _has_text(t.network_weights):
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
        errors.append(_make_issue("error", "rl.nft_kl_beta", "NFT KL Beta must be >= 0.", label="NFT KL Beta", page="rl"))
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
    if proc_type == "inference":
        return validate_inference_config(config)
    if proc_type in ("rl", "rl_cache_rollouts", "rl_train"):
        phase = {"rl_cache_rollouts": "cache_rollouts", "rl_train": "train_rl"}.get(proc_type)
        return validate_rl_config(config, phase=phase)
    return _build_report([], [])
