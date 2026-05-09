"""Project configuration API router."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from musubi_tuner.gui_dashboard.project_schema import ProjectConfig

router = APIRouter(prefix="/api/project", tags=["project"])


def _get_state(request: Request):
    return request.app.state


def _resolve_project_json(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_dir():
        path = path / "project.json"
    return path


def _project_summary(config: ProjectConfig, path: Path) -> dict:
    training = config.training
    datasets = config.dataset.datasets
    validation_datasets = config.dataset.validation_datasets
    dataset_types: dict[str, int] = {}
    for entry in datasets:
        dataset_types[entry.type] = dataset_types.get(entry.type, 0) + 1

    ic_targets = {"v2v", "audio_ref_only_ic", "av_ic", "video_ref_only_av"}
    uses_ic_lora = training.ic_lora_strategy not in ("auto", "none") or training.lora_target_preset in ic_targets
    uses_lycoris = bool(training.lycoris_config) or training.lora_target_preset == "lycoris"
    if uses_lycoris:
        lora_kind = "LyCORIS"
    elif uses_ic_lora:
        lora_kind = "IC-LoRA"
    else:
        lora_kind = "LoRA"

    return {
        "name": config.name,
        "path": str(path),
        "project_dir": config.project_dir,
        "datasets": len(datasets),
        "validation_datasets": len(validation_datasets),
        "dataset_types": dataset_types,
        "mode": training.ltx2_mode,
        "ltx_version": training.ltx_version,
        "lora_kind": lora_kind,
        "lora_target_preset": training.lora_target_preset,
        "ic_lora_strategy": training.ic_lora_strategy,
        "network_dim": training.network_dim,
        "network_alpha": training.network_alpha,
        "max_train_epochs": training.max_train_epochs,
        "max_train_steps": training.max_train_steps,
        "output_name": training.output_name,
    }


@router.get("")
async def get_project(request: Request):
    state = _get_state(request)
    config: ProjectConfig | None = state.project_config
    if config is None:
        return {"loaded": False, "config": None, "project_path": None}
    project_path = state.project_path
    return {
        "loaded": True,
        "config": config.model_dump(),
        "project_path": str(project_path) if project_path else None,
    }


@router.get("/defaults")
async def get_project_defaults():
    return {"config": ProjectConfig().model_dump()}


@router.post("/summary")
async def summarize_project(body: dict):
    path_str = body.get("path", "")
    if not path_str:
        raise HTTPException(status_code=400, detail="path is required")

    path = _resolve_project_json(path_str)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        config = ProjectConfig.load(path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to load: {e}")
    return {"ok": True, "summary": _project_summary(config, path)}


@router.post("")
async def create_project(body: dict, request: Request):
    state = _get_state(request)
    try:
        config = ProjectConfig(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not config.project_dir:
        raise HTTPException(status_code=400, detail="project_dir is required")

    project_dir = Path(config.project_dir)
    project_json = project_dir / "project.json"

    if project_dir.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Project folder already exists: {project_dir}",
        )

    project_dir.mkdir(parents=True, exist_ok=True)
    config.save(project_json)
    state.project_config = config
    state.project_path = project_json
    return {
        "ok": True,
        "config": config.model_dump(),
        "project_path": str(project_json),
    }


@router.put("")
async def update_project(body: dict, request: Request):
    state = _get_state(request)
    config: ProjectConfig | None = state.project_config
    if config is None:
        raise HTTPException(status_code=400, detail="No project loaded")

    try:
        updated = ProjectConfig(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    project_path = state.project_path
    updated.save(project_path)
    state.project_config = updated
    return {"ok": True, "config": updated.model_dump()}


@router.delete("")
async def close_project(request: Request):
    state = _get_state(request)
    state.project_config = None
    state.project_path = None
    return {"ok": True}


@router.post("/load")
async def load_project(body: dict, request: Request):
    state = _get_state(request)
    path_str = body.get("path", "")
    if not path_str:
        raise HTTPException(status_code=400, detail="path is required")

    path = _resolve_project_json(path_str)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        config = ProjectConfig.load(path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to load: {e}")

    state.project_config = config
    state.project_path = path
    return {
        "ok": True,
        "config": config.model_dump(),
        "project_path": str(path),
    }
