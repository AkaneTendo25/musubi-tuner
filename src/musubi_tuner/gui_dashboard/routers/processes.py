"""Process management API router."""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse

from musubi_tuner.gui_dashboard.command_builder import (
    build_cache_dino_cmd,
    build_cache_latents_cmd,
    build_cache_text_cmd,
    build_inference_cmd,
    build_slider_training_cmd,
    build_training_cmd,
)
from musubi_tuner.gui_dashboard.process_manager import ProcessManager

router = APIRouter(tags=["processes"])

VALID_TYPES = ("cache_latents", "cache_text", "cache_dino", "training", "inference", "slider_training")


def _get_pm(request: Request) -> ProcessManager:
    return request.app.state.process_manager


def _get_config(request: Request):
    config = request.app.state.project_config
    if config is None:
        raise HTTPException(status_code=400, detail="No project loaded")
    return config


def _validate_type(proc_type: str):
    if proc_type not in VALID_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid type: {proc_type}. Must be one of {VALID_TYPES}")


def _build_cmd(proc_type: str, config):
    """Build command list for a given process type."""
    if proc_type == "cache_latents":
        return build_cache_latents_cmd(config)
    elif proc_type == "cache_text":
        return build_cache_text_cmd(config)
    elif proc_type == "cache_dino":
        return build_cache_dino_cmd(config)
    elif proc_type == "training":
        return build_training_cmd(config)
    elif proc_type == "inference":
        return build_inference_cmd(config)
    elif proc_type == "slider_training":
        return build_slider_training_cmd(config)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown type: {proc_type}")


@router.post("/api/processes/{proc_type}/start")
async def start_process(proc_type: str, request: Request):
    _validate_type(proc_type)
    pm = _get_pm(request)
    config = _get_config(request)

    try:
        cmd = _build_cmd(proc_type, config)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build command: {e}")

    try:
        pm.start(proc_type, cmd, cwd=config.project_dir or None)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {"ok": True, "state": pm.get_status(proc_type)["state"]}


@router.post("/api/processes/{proc_type}/stop")
async def stop_process(proc_type: str, request: Request):
    _validate_type(proc_type)
    pm = _get_pm(request)
    pm.stop(proc_type)
    return {"ok": True}


@router.get("/api/processes/{proc_type}/status")
async def get_process_status(proc_type: str, request: Request):
    _validate_type(proc_type)
    pm = _get_pm(request)
    return pm.get_status(proc_type)


@router.get("/api/processes/{proc_type}/logs")
async def get_process_logs(
    proc_type: str,
    request: Request,
    last_n: Optional[int] = Query(None, description="Return last N lines"),
):
    _validate_type(proc_type)
    pm = _get_pm(request)
    return {"lines": pm.get_logs(proc_type, last_n=last_n)}


@router.get("/api/processes/{proc_type}/command-preview")
async def get_command_preview(proc_type: str, request: Request):
    """Return the CLI command that would be run for the given process type."""
    _validate_type(proc_type)
    config = _get_config(request)

    try:
        cmd = _build_cmd(proc_type, config)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build command: {e}")

    return {"command": " ".join(cmd)}


@router.get("/api/processes/status")
async def get_all_process_statuses(request: Request):
    pm = _get_pm(request)
    return pm.get_all_statuses()


@router.get("/sse/processes")
async def sse_process_stream(request: Request):
    """SSE stream that emits process status changes and new log lines."""
    pm = _get_pm(request)

    async def event_generator():
        last_statuses = {}
        last_log_counts = {t: 0 for t in VALID_TYPES}

        while True:
            await asyncio.sleep(1)

            # Check for status changes
            current = pm.get_all_statuses()
            if current != last_statuses:
                last_statuses = current
                yield {"event": "status", "data": json.dumps(current)}

            # Check for new log lines
            for proc_type in VALID_TYPES:
                logs = pm.get_logs(proc_type)
                count = len(logs)
                if count > last_log_counts[proc_type]:
                    new_lines = logs[last_log_counts[proc_type]:]
                    last_log_counts[proc_type] = count
                    yield {
                        "event": "logs",
                        "data": json.dumps({
                            "type": proc_type,
                            "lines": new_lines,
                        }),
                    }

    return EventSourceResponse(event_generator())
