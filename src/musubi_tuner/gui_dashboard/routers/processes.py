"""Process management API router."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse

from musubi_tuner.gui_dashboard.command_builder import (
    build_cache_dino_cmd,
    build_cache_latents_cmd,
    build_cache_text_cmd,
    build_full_finetune_cmd,
    build_inference_cmd,
    build_remote_stage_launcher_cmd,
    build_remote_stage_server_cmd,
    build_rl_cache_rollouts_cmd,
    build_rl_train_cmd,
    build_slider_training_cmd,
    build_training_cmd,
)
from musubi_tuner.gui_dashboard.process_manager import ProcessManager
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.gui_dashboard.training_dashboard_state import (
    clear_training_dashboard_files,
    training_is_resume,
    write_training_launch_status,
)
from musubi_tuner.gui_dashboard.validation import validate_process_config

router = APIRouter(tags=["processes"])

VALID_TYPES = (
    "cache_latents",
    "cache_text",
    "cache_dino",
    "training",
    "full_finetune",
    "remote_stage_launcher",
    "remote_stage_server",
    "inference",
    "slider_training",
    "rl_cache_rollouts",
    "rl_train",
)


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
    elif proc_type == "full_finetune":
        return build_full_finetune_cmd(config)
    elif proc_type == "remote_stage_launcher":
        return build_remote_stage_launcher_cmd(config)
    elif proc_type == "remote_stage_server":
        return build_remote_stage_server_cmd(config)
    elif proc_type == "inference":
        return build_inference_cmd(config)
    elif proc_type == "slider_training":
        return build_slider_training_cmd(config)
    elif proc_type == "rl_cache_rollouts":
        return build_rl_cache_rollouts_cmd(config)
    elif proc_type == "rl_train":
        return build_rl_train_cmd(config)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown type: {proc_type}")


def _render_command_script(cmd: list[str], cwd: str | None = None) -> str:
    if os.name == "nt":
        lines = ["@echo off"]
        if cwd:
            lines.append(f"cd /d {subprocess.list2cmdline([str(cwd)])}")
        lines.append(subprocess.list2cmdline([str(part) for part in cmd]))
        return "\r\n".join(lines) + "\r\n"

    lines = ["#!/usr/bin/env sh", "set -e"]
    if cwd:
        lines.append(f"cd {shlex.quote(str(cwd))}")
    lines.append(shlex.join([str(part) for part in cmd]))
    return "\n".join(lines) + "\n"


def _validate_process_or_raise(proc_type: str, config) -> dict:
    report = validate_process_config(proc_type, config)
    if not report["ok"]:
        raise HTTPException(status_code=422, detail=report)
    return report


@router.post("/api/processes/{proc_type}/validate")
async def validate_process(proc_type: str, request: Request, project_config: ProjectConfig | None = Body(default=None)):
    _validate_type(proc_type)
    config = project_config or _get_config(request)
    return validate_process_config(proc_type, config)


@router.post("/api/processes/{proc_type}/start")
async def start_process(proc_type: str, request: Request):
    _validate_type(proc_type)
    pm = _get_pm(request)
    config = _get_config(request)
    _validate_process_or_raise(proc_type, config)

    if proc_type in {"training", "full_finetune"}:
        clear_training_dashboard_files(config, keep_history=training_is_resume(config, proc_type), process_type=proc_type)
        write_training_launch_status(config, process_type=proc_type)

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

    return {"command": _render_command_script(cmd, cwd=config.project_dir or None)}


@router.get("/api/processes/status")
async def get_all_process_statuses(request: Request):
    pm = _get_pm(request)
    return pm.get_all_statuses()


@router.get("/sse/processes")
async def sse_process_stream(request: Request):
    """SSE stream that emits process status changes."""
    pm = _get_pm(request)

    async def event_generator():
        last_statuses = {}

        while True:
            await asyncio.sleep(1)

            # Check for status changes
            current = pm.get_all_statuses()
            if current != last_statuses:
                last_statuses = current
                yield {"event": "status", "data": json.dumps(current)}

    return EventSourceResponse(event_generator())
