"""Dataset configuration API router."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from musubi_tuner.gui_dashboard.toml_export import _dataset_entry_to_dict, _toml_value, export_dataset_toml
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

    doc: dict = {
        "general": {
            "enable_bucket": config.dataset.general.enable_bucket,
            "bucket_no_upscale": config.dataset.general.bucket_no_upscale,
        },
        "datasets": [_dataset_entry_to_dict(entry) for entry in config.dataset.datasets],
    }

    if config.dataset.validation_datasets:
        doc["validation_datasets"] = [
            _dataset_entry_to_dict(entry) for entry in config.dataset.validation_datasets
        ]

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
