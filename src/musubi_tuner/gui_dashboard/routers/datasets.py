"""Dataset configuration API router."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from musubi_tuner.gui_dashboard.toml_export import export_dataset_toml
from musubi_tuner.gui_dashboard.project_schema import DatasetConfig, ProjectConfig

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


def _get_config(request: Request) -> ProjectConfig:
    config = request.app.state.project_config
    if config is None:
        raise HTTPException(status_code=400, detail="No project loaded")
    return config


@router.get("/config")
async def get_dataset_config(request: Request):
    config = _get_config(request)
    return config.dataset.model_dump()


@router.put("/config")
async def update_dataset_config(body: dict, request: Request):
    config = _get_config(request)
    try:
        config.dataset = DatasetConfig(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    config.save()
    return {"ok": True, "config": config.dataset.model_dump()}


@router.post("/export-toml")
async def export_toml(request: Request):
    config = _get_config(request)
    if not config.project_dir:
        raise HTTPException(status_code=400, detail="project_dir not set")

    try:
        path = export_dataset_toml(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"ok": True, "path": str(path)}


@router.get("/preview-toml")
async def preview_toml(request: Request):
    """Return what the TOML file would look like without writing it."""
    config = _get_config(request)

    from musubi_tuner.gui_dashboard.toml_export import _write_toml_fallback, build_dataset_toml_path
    import io

    # Generate TOML content as string
    doc: dict = {}
    doc["general"] = {
        "enable_bucket": config.dataset.general.enable_bucket,
        "bucket_no_upscale": config.dataset.general.bucket_no_upscale,
    }

    from musubi_tuner.gui_dashboard.toml_export import _toml_value

    datasets = []
    for entry in config.dataset.datasets:
        d: dict = {}
        if entry.type == "video":
            d["video_directory"] = entry.directory
        elif entry.type == "image":
            d["image_directory"] = entry.directory
        elif entry.type == "audio":
            d["audio_directory"] = entry.directory
        d["cache_directory"] = entry.cache_directory
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
        datasets.append(d)
    doc["datasets"] = datasets

    if config.dataset.validation_datasets:
        val = []
        for entry in config.dataset.validation_datasets:
            d = {}
            if entry.type == "video":
                d["video_directory"] = entry.directory
            elif entry.type == "image":
                d["image_directory"] = entry.directory
            elif entry.type == "audio":
                d["audio_directory"] = entry.directory
            d["cache_directory"] = entry.cache_directory
            if entry.type != "audio":
                d["resolution"] = [entry.resolution_w, entry.resolution_h]
            d["batch_size"] = entry.batch_size
            d["num_repeats"] = entry.num_repeats
            d["caption_extension"] = entry.caption_extension
            if entry.type == "video":
                d["target_frames"] = [entry.target_frames]
                d["frame_extraction"] = entry.frame_extraction
            val.append(d)
        doc["validation_datasets"] = val

    # Build TOML string
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

    return {"toml": "\n".join(lines)}
