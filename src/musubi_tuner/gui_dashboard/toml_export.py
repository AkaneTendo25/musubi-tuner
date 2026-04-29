"""TOML serialization for dataset and slider configs."""

from __future__ import annotations

import json
from pathlib import Path

try:
    import tomli_w
except ImportError:
    tomli_w = None

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

from musubi_tuner.gui_dashboard.project_schema import ProjectConfig


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        items = ", ".join(_toml_value(i) for i in v)
        return f"[{items}]"
    return str(v)


def _render_toml_fallback(doc: dict) -> str:
    """Simple TOML renderer for when tomli_w is not available."""
    lines = []

    if "general" in doc:
        lines.append("[general]")
        for k, v in doc["general"].items():
            lines.append(f"{k} = {_toml_value(v)}")
        lines.append("")

    for section_name in ("datasets", "validation_datasets"):
        if section_name not in doc:
            continue
        for entry in doc[section_name]:
            lines.append(f"[[{section_name}]]")
            for k, v in entry.items():
                lines.append(f"{k} = {_toml_value(v)}")
            lines.append("")

    return "\n".join(lines)


def render_toml(doc: dict) -> str:
    if tomli_w is not None:
        return tomli_w.dumps(doc)
    return _render_toml_fallback(doc)


def _default_cache_directory(entry) -> str:
    if entry.cache_directory:
        return entry.cache_directory
    if entry.directory:
        return str(Path(entry.directory) / "cache")
    if entry.jsonl_file:
        return str(Path(entry.jsonl_file).parent / "cache")
    return ""


def _split_path_list(raw: str) -> list[str]:
    if not raw:
        return []
    values: list[str] = []
    seen: set[str] = set()
    normalized = raw.replace(";", "\n").replace(",", "\n")
    for part in normalized.splitlines():
        candidate = part.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        values.append(candidate)
    return values


def _merge_path_values(*groups: list[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            if value in seen:
                continue
            seen.add(value)
            values.append(value)
    return values


def _dataset_entry_to_dict(entry) -> dict:
    """Convert a DatasetEntry to a TOML-ready dict."""
    d: dict = {}
    # Directory or JSONL source
    if entry.jsonl_file:
        key_prefix = {"video": "video", "image": "image", "audio": "audio"}[entry.type]
        d[f"{key_prefix}_jsonl_file"] = entry.jsonl_file
    else:
        if entry.type == "video":
            d["video_directory"] = entry.directory
        elif entry.type == "image":
            d["image_directory"] = entry.directory
        elif entry.type == "audio":
            d["audio_directory"] = entry.directory

    d["cache_directory"] = _default_cache_directory(entry)
    reference_cache_directories = _merge_path_values(
        [entry.reference_cache_directory] if entry.reference_cache_directory else [],
        _split_path_list(getattr(entry, "extra_reference_cache_directories", "")),
    )
    if len(reference_cache_directories) == 1:
        d["reference_cache_directory"] = reference_cache_directories[0]
    elif len(reference_cache_directories) > 1:
        d["reference_cache_directories"] = reference_cache_directories

    reference_audio_cache_directories = _merge_path_values(
        [entry.reference_audio_cache_directory] if getattr(entry, "reference_audio_cache_directory", "") else [],
        _split_path_list(getattr(entry, "extra_reference_audio_cache_directories", "")),
    )
    if len(reference_audio_cache_directories) == 1:
        d["reference_audio_cache_directory"] = reference_audio_cache_directories[0]
    elif len(reference_audio_cache_directories) > 1:
        d["reference_audio_cache_directories"] = reference_audio_cache_directories

    reference_directories = _merge_path_values(
        [entry.control_directory] if entry.control_directory else [],
        _split_path_list(getattr(entry, "extra_control_directories", "")),
    )
    if reference_directories:
        if reference_cache_directories:
            if len(reference_directories) == 1:
                d["reference_directory"] = reference_directories[0]
            else:
                d["reference_directories"] = reference_directories
        else:
            if len(reference_directories) == 1:
                d["control_directory"] = reference_directories[0]
            else:
                d["control_directory"] = reference_directories[0]
                d["extra_control_directories"] = ", ".join(reference_directories[1:])

    reference_audio_directories = _merge_path_values(
        [entry.reference_audio_directory] if getattr(entry, "reference_audio_directory", "") else [],
        _split_path_list(getattr(entry, "extra_reference_audio_directories", "")),
    )
    if len(reference_audio_directories) == 1:
        d["reference_audio_directory"] = reference_audio_directories[0]
    elif len(reference_audio_directories) > 1:
        d["reference_audio_directories"] = reference_audio_directories
    if entry.type != "audio":
        d["resolution"] = [entry.resolution_w, entry.resolution_h]
    d["batch_size"] = entry.batch_size
    d["num_repeats"] = entry.num_repeats
    d["caption_extension"] = entry.caption_extension
    if getattr(entry, "caption_field", ""):
        d["caption_field"] = entry.caption_field

    if getattr(entry, "loss_mask_directory", ""):
        d["loss_mask_directory"] = entry.loss_mask_directory
    if getattr(entry, "default_loss_mask_path", ""):
        d["default_loss_mask_path"] = entry.default_loss_mask_path
    if getattr(entry, "loss_mask_use_alpha", False):
        d["loss_mask_use_alpha"] = True
    if getattr(entry, "loss_mask_invert", False):
        d["loss_mask_invert"] = True

    if entry.type == "video":
        d["target_frames"] = [entry.target_frames]
        d["frame_extraction"] = entry.frame_extraction
        if entry.frame_sample is not None:
            d["frame_sample"] = entry.frame_sample
        if entry.max_frames is not None:
            d["max_frames"] = entry.max_frames
        if entry.frame_stride is not None:
            d["frame_stride"] = entry.frame_stride
        if entry.source_fps is not None:
            d["source_fps"] = entry.source_fps
        if entry.target_fps is not None:
            d["target_fps"] = entry.target_fps

    return d


def build_dataset_toml_document(config: ProjectConfig) -> dict:
    doc: dict = {
        "general": {
            "enable_bucket": config.dataset.general.enable_bucket,
            "bucket_no_upscale": config.dataset.general.bucket_no_upscale,
        },
        "datasets": [_dataset_entry_to_dict(e) for e in config.dataset.datasets],
    }

    if config.dataset.validation_datasets:
        doc["validation_datasets"] = [
            _dataset_entry_to_dict(e) for e in config.dataset.validation_datasets
        ]

    return doc


def _write_dataset_toml(config: ProjectConfig, output_path: Path) -> Path:
    """Generate dataset_config.toml from project config and return path."""
    doc = build_dataset_toml_document(config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_toml(doc), encoding="utf-8")

    return output_path


def build_dataset_toml_path(config: ProjectConfig) -> Path:
    return Path(config.project_dir) / "dataset_config.toml"


def export_dataset_toml(config: ProjectConfig) -> Path:
    return _write_dataset_toml(config, build_dataset_toml_path(config))


def _write_slider_toml(config: ProjectConfig, output_path: Path) -> Path:
    """Generate slider_config.toml from project config and return path."""
    s = config.slider
    doc: dict = {
        "mode": s.mode,
        "guidance_strength": s.guidance_strength,
    }
    if s.reference_modality:
        doc["reference_modality"] = s.reference_modality
    if s.pos_cache_dir:
        doc["pos_cache_dir"] = s.pos_cache_dir
    if s.neg_cache_dir:
        doc["neg_cache_dir"] = s.neg_cache_dir
    if s.text_cache_dir:
        doc["text_cache_dir"] = s.text_cache_dir
    if s.reference_cache_dir:
        doc["reference_cache_dir"] = s.reference_cache_dir

    # Parse sample_slider_range
    try:
        doc["sample_slider_range"] = [float(x.strip()) for x in s.sample_slider_range.split(",")]
    except (ValueError, AttributeError):
        doc["sample_slider_range"] = [-2.0, -1.0, 0.0, 1.0, 2.0]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write TOML with targets as [[targets]] array
    lines = []
    for k, v in doc.items():
        lines.append(f"{k} = {_toml_value(v)}")
    lines.append("")

    if s.mode == "text":
        for target in s.targets:
            lines.append("[[targets]]")
            lines.append(f'positive = "{target.positive}"')
            lines.append(f'negative = "{target.negative}"')
            if target.target_class:
                lines.append(f'target_class = "{target.target_class}"')
            lines.append(f"weight = {target.weight}")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_slider_toml_path(config: ProjectConfig) -> Path:
    return Path(config.project_dir) / "slider_config.toml"
