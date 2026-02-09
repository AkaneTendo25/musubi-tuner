"""TOML serialization for dataset and slider configs."""

from __future__ import annotations

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
        return f'"{v}"'
    if isinstance(v, list):
        items = ", ".join(_toml_value(i) for i in v)
        return f"[{items}]"
    return str(v)


def _write_toml_fallback(doc: dict, path: Path):
    """Simple TOML writer for when tomli_w is not available."""
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

    path.write_text("\n".join(lines), encoding="utf-8")


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

    d["cache_directory"] = entry.cache_directory
    if entry.reference_cache_directory:
        d["reference_cache_directory"] = entry.reference_cache_directory
    if entry.control_directory:
        d["control_directory"] = entry.control_directory
    if entry.type != "audio":
        d["resolution"] = [entry.resolution_w, entry.resolution_h]
    d["batch_size"] = entry.batch_size
    d["num_repeats"] = entry.num_repeats
    d["caption_extension"] = entry.caption_extension

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


def _write_dataset_toml(config: ProjectConfig, output_path: Path) -> Path:
    """Generate dataset_config.toml from project config and return path."""
    doc: dict = {}

    # General section
    doc["general"] = {
        "enable_bucket": config.dataset.general.enable_bucket,
        "bucket_no_upscale": config.dataset.general.bucket_no_upscale,
    }

    # Datasets
    doc["datasets"] = [_dataset_entry_to_dict(e) for e in config.dataset.datasets]

    # Validation datasets
    if config.dataset.validation_datasets:
        doc["validation_datasets"] = [
            _dataset_entry_to_dict(e) for e in config.dataset.validation_datasets
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if tomli_w is not None:
        output_path.write_bytes(tomli_w.dumps(doc).encode("utf-8"))
    else:
        _write_toml_fallback(doc, output_path)

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
