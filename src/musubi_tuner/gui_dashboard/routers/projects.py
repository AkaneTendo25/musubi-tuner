"""Project configuration API router."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from musubi_tuner.gui_dashboard.project_schema import ProjectConfig

router = APIRouter(prefix="/api/project", tags=["project"])


def _get_state(request: Request):
    return request.app.state


def _slugify(name: str) -> str:
    """Convert a project name to a safe filename (without extension)."""
    s = name.strip().lower()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s-]+', '_', s)
    return s or "project"


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


@router.post("")
async def create_project(body: dict, request: Request):
    state = _get_state(request)
    try:
        config = ProjectConfig(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not config.project_dir:
        raise HTTPException(status_code=400, detail="project_dir is required")

    # Keep canonical filename for compatibility with directory-based loading.
    project_json = Path(config.project_dir) / "project.json"

    # If canonical file exists, add a numeric suffix based on project name.
    if project_json.exists():
        for i in range(2, 100):
            candidate = Path(config.project_dir) / f"{_slugify(config.name)}_{i}.json"
            if not candidate.exists():
                project_json = candidate
                break
        else:
            raise HTTPException(
                status_code=409,
                detail=f"Too many project files with name '{config.name}' in {config.project_dir}.",
            )

    Path(config.project_dir).mkdir(parents=True, exist_ok=True)
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

    path = Path(path_str)

    # If user passes a directory, look for project.json inside it
    if path.is_dir():
        path = path / "project.json"

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
