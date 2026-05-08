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
    reference_frames = getattr(entry, "reference_frames", None)
    if entry.type != "audio" and reference_frames is not None:
        d["reference_frames"] = int(reference_frames)

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
                raise ValueError(
                    "Multiple control/reference directories require matching reference cache directories. "
                    "Dataset TOML supports one control_directory without reference caches."
                )

    reference_audio_directories = _merge_path_values(
        [entry.reference_audio_directory] if getattr(entry, "reference_audio_directory", "") else [],
        _split_path_list(getattr(entry, "extra_reference_audio_directories", "")),
    )
    if len(reference_audio_directories) == 1:
        d["reference_audio_directory"] = reference_audio_directories[0]
    elif len(reference_audio_directories) > 1:
        d["reference_audio_directories"] = reference_audio_directories

    # Latent guides — only emitted when the source directory is set so existing
    # TOMLs without these fields stay byte-identical.
    if getattr(entry, "latent_idx_guide_directory", ""):
        d["latent_idx_guide_directory"] = entry.latent_idx_guide_directory
        if getattr(entry, "latent_idx_guide_cache_directory", ""):
            d["latent_idx_guide_cache_directory"] = entry.latent_idx_guide_cache_directory
        latidx_fi = getattr(entry, "latent_idx_guide_frame_idx", None)
        if latidx_fi is not None:
            d["latent_idx_guide_frame_idx"] = int(latidx_fi)
        latidx_st = getattr(entry, "latent_idx_guide_strength", None)
        if latidx_st is not None and float(latidx_st) != 1.0:
            d["latent_idx_guide_strength"] = float(latidx_st)
    if getattr(entry, "keyframe_guide_directory", ""):
        d["keyframe_guide_directory"] = entry.keyframe_guide_directory
        if getattr(entry, "keyframe_guide_cache_directory", ""):
            d["keyframe_guide_cache_directory"] = entry.keyframe_guide_cache_directory
        kf_fi = getattr(entry, "keyframe_guide_frame_idx", None)
        if kf_fi is not None:
            d["keyframe_guide_frame_idx"] = int(kf_fi)
        kf_st = getattr(entry, "keyframe_guide_strength", None)
        if kf_st is not None and float(kf_st) != 1.0:
            d["keyframe_guide_strength"] = float(kf_st)

        # Multi-keyframe extras: parallel semicolon-separated lists in the GUI
        # (flat JSON friendly) become parallel TOML arrays. All four lists must
        # have the same length and parse cleanly; mismatches raise so the user
        # gets a clear error instead of a silently-dropped extras config.
        def _split_csv(s: str) -> list[str]:
            return [p.strip() for p in (s or "").replace(",", ";").split(";") if p.strip()]

        extra_dirs = _split_csv(getattr(entry, "keyframe_guide_extra_directories", ""))
        extra_caches = _split_csv(getattr(entry, "keyframe_guide_extra_cache_directories", ""))
        extra_fis = _split_csv(getattr(entry, "keyframe_guide_extra_frame_idxs", ""))
        extra_sts = _split_csv(getattr(entry, "keyframe_guide_extra_strengths", ""))
        any_set = any((extra_dirs, extra_caches, extra_fis, extra_sts))
        if any_set:
            if not (len(extra_dirs) == len(extra_caches) == len(extra_fis) == len(extra_sts)):
                raise ValueError(
                    "keyframe_guide_extra_* fields must all have the same number of entries (use ';' to separate). "
                    f"Got directories={len(extra_dirs)}, cache_directories={len(extra_caches)}, "
                    f"frame_idxs={len(extra_fis)}, strengths={len(extra_sts)}."
                )
            try:
                d["keyframe_guide_extra_directories"] = extra_dirs
                d["keyframe_guide_extra_cache_directories"] = extra_caches
                d["keyframe_guide_extra_frame_idxs"] = [int(x) for x in extra_fis]
                d["keyframe_guide_extra_strengths"] = [float(x) for x in extra_sts]
            except ValueError as exc:
                raise ValueError(
                    f"keyframe_guide_extra_frame_idxs / keyframe_guide_extra_strengths must parse as int / float: {exc}"
                ) from exc

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
            lines.append(f"positive = {_toml_value(target.positive)}")
            lines.append(f"negative = {_toml_value(target.negative)}")
            if target.target_class:
                lines.append(f"target_class = {_toml_value(target.target_class)}")
            lines.append(f"weight = {target.weight}")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_slider_toml_path(config: ProjectConfig) -> Path:
    return Path(config.project_dir) / "slider_config.toml"
