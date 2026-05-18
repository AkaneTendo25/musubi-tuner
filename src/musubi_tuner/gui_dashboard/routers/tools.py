"""Dashboard utility jobs such as adapter conversion."""

from __future__ import annotations

import dataclasses
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from musubi_tuner.gui_dashboard.command_builder import _effective_ltx2_checkpoint
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.ltx2_model_loading import detect_ltx2_dtype
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tools", tags=["tools"])


class ConvertComfyRequest(BaseModel):
    checkpoint_path: str
    output_path: str = ""
    base_model_path: str = ""
    device: str = "cpu"


@dataclasses.dataclass
class ConvertComfyJob:
    job_id: str
    checkpoint_path: str
    output_path: str = ""
    base_model_path: str = ""
    state: str = "queued"
    message: str = "Queued"
    error: str = ""
    created_at: float = dataclasses.field(default_factory=time.time)
    updated_at: float = dataclasses.field(default_factory=time.time)
    finished_at: float | None = None
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


def _get_config(request: Request) -> ProjectConfig:
    config = request.app.state.project_config
    if config is None:
        raise HTTPException(status_code=400, detail="No project loaded")
    return config


def _get_jobs(request: Request) -> dict[str, ConvertComfyJob]:
    jobs = getattr(request.app.state, "convert_comfy_jobs", None)
    if jobs is None:
        jobs = {}
        request.app.state.convert_comfy_jobs = jobs
    return jobs


def _snapshot_job(job: ConvertComfyJob) -> dict:
    with job.lock:
        return {
            "job_id": job.job_id,
            "checkpoint_path": job.checkpoint_path,
            "output_path": job.output_path,
            "base_model_path": job.base_model_path,
            "state": job.state,
            "message": job.message,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "finished_at": job.finished_at,
        }


def _set_job_state(job: ConvertComfyJob, **fields) -> None:
    with job.lock:
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = time.time()


def _prune_finished_jobs(jobs: dict[str, ConvertComfyJob], keep: int = 20) -> None:
    finished = [
        job for job in jobs.values()
        if job.state in {"completed", "failed"}
    ]
    finished.sort(key=lambda job: job.updated_at, reverse=True)
    for job in finished[keep:]:
        jobs.pop(job.job_id, None)


def _resolve_project_path(config: ProjectConfig, raw_path: str) -> Path:
    clean_path = str(raw_path or "").strip()
    if not clean_path:
        raise ValueError("Path is required")
    path = Path(clean_path)
    if not path.is_absolute() and config.project_dir:
        path = Path(config.project_dir) / path
    return path


def _base_dtype_for_training(config: ProjectConfig, base_model_path: str) -> torch.dtype:
    try:
        dtype = detect_ltx2_dtype(base_model_path)
    except Exception:
        logger.warning("Could not detect LTX-2 dtype for conversion; falling back to float32", exc_info=True)
        return torch.float32

    mixed_precision = str(config.training.mixed_precision or "no")
    if dtype is not None and dtype.itemsize == 1:
        if mixed_precision == "fp16":
            return torch.float16
        if mixed_precision == "bf16":
            return torch.bfloat16
        return torch.float32
    if dtype == torch.float32 and mixed_precision in {"fp16", "bf16"}:
        return torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return dtype or torch.float32


def _checkpoint_needs_base_model(checkpoint_path: Path) -> bool:
    from safetensors import safe_open

    with safe_open(str(checkpoint_path), framework="pt") as handle:
        return any(".lora_magnitude_vector.weight" in key for key in handle.keys())


def _converter_env() -> dict[str, str]:
    import musubi_tuner

    env = os.environ.copy()
    python_bin_dir = os.path.dirname(sys.executable)
    path_key = "Path" if "Path" in env else "PATH"
    env[path_key] = python_bin_dir + os.pathsep + env.get(path_key, "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    src_root = str(Path(musubi_tuner.__file__).resolve().parents[1])
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_root + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    return env


def _build_convert_comfy_cmd(job: ConvertComfyJob, config: ProjectConfig, device: str, base_dtype: torch.dtype) -> list[str]:
    t = config.training
    cmd = [
        sys.executable,
        "-m",
        "musubi_tuner.ltx_2.convert_lora_to_comfy",
        job.checkpoint_path,
        "--base_dtype",
        model_utils.dtype_to_str(base_dtype),
        "--device",
        device or "cpu",
        "--dora_ff_only",
    ]
    if job.output_path:
        cmd += ["--output", job.output_path]
    if job.base_model_path:
        cmd += ["--base_model", job.base_model_path]
    if t.ltx2_mode in {"av", "audio"}:
        cmd.append("--audio_video")
    if t.ltx2_audio_only_model:
        cmd.append("--audio_only_model")
    if t.fp8_base:
        cmd.append("--fp8_base")
    if t.fp8_scaled:
        cmd.append("--fp8_scaled")
    if t.fp8_w8a8:
        cmd.append("--fp8_w8a8")
        if t.w8a8_mode != "int8":
            cmd += ["--w8a8_mode", t.w8a8_mode]
    if t.fp8_keep_blocks:
        cmd += ["--fp8_keep_blocks", t.fp8_keep_blocks]
    if t.nf4_base:
        cmd.append("--nf4_base")
        if t.nf4_block_size != 32:
            cmd += ["--nf4_block_size", str(t.nf4_block_size)]
    if t.quantize_device:
        cmd += ["--quantize_device", t.quantize_device]
    return cmd


def _default_comfy_output_path(input_path: str) -> str:
    path = Path(input_path)
    return str(path.parent / f"{path.stem}.comfy{path.suffix}")


def _run_convert_comfy_job(job: ConvertComfyJob, config: ProjectConfig, device: str) -> None:
    try:
        needs_base = bool(job.base_model_path)
        _set_job_state(job, state="running", message="Loading base checkpoint in converter process" if needs_base else "Converting checkpoint")
        base_dtype = _base_dtype_for_training(config, job.base_model_path) if needs_base else torch.float32
        cmd = _build_convert_comfy_cmd(job, config, device, base_dtype)
        result = subprocess.run(
            cmd,
            cwd=config.project_dir or None,
            env=_converter_env(),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            output_tail = "\n".join((result.stdout or "").splitlines()[-40:])
            raise RuntimeError(output_tail or f"Converter exited with code {result.returncode}")

        output_path = job.output_path or _default_comfy_output_path(job.checkpoint_path)
        _set_job_state(
            job,
            state="completed",
            message=f"Saved to {output_path}",
            output_path=str(output_path),
            finished_at=time.time(),
        )
    except Exception as exc:
        logger.exception("ComfyUI conversion failed")
        _set_job_state(
            job,
            state="failed",
            message=str(exc),
            error=str(exc),
            finished_at=time.time(),
        )


@router.post("/convert-comfy")
async def start_convert_comfy(req: ConvertComfyRequest, request: Request):
    config = _get_config(request)
    checkpoint_path = _resolve_project_path(config, req.checkpoint_path)
    if not checkpoint_path.is_file():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix.casefold() != ".safetensors":
        raise HTTPException(status_code=400, detail="Checkpoint must be a .safetensors file")

    output_path: Optional[Path] = _resolve_project_path(config, req.output_path) if req.output_path.strip() else None
    if output_path is not None and output_path.suffix.casefold() != ".safetensors":
        raise HTTPException(status_code=400, detail="Output path must be a .safetensors file")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    needs_base = _checkpoint_needs_base_model(checkpoint_path)
    base_model_path: Optional[Path] = None
    if needs_base or req.base_model_path.strip():
        base_model = req.base_model_path.strip() or _effective_ltx2_checkpoint(config, config.training.ltx2_checkpoint)
        base_model_path = _resolve_project_path(config, base_model)
        if not base_model_path.is_file():
            raise HTTPException(status_code=404, detail=f"Base LTX-2 checkpoint not found: {base_model_path}")

    jobs = _get_jobs(request)
    _prune_finished_jobs(jobs)
    job = ConvertComfyJob(
        job_id=uuid.uuid4().hex,
        checkpoint_path=str(checkpoint_path),
        output_path=str(output_path) if output_path is not None else "",
        base_model_path=str(base_model_path) if base_model_path is not None else "",
    )
    jobs[job.job_id] = job
    thread = threading.Thread(target=_run_convert_comfy_job, args=(job, config, req.device or "cpu"), daemon=True)
    thread.start()
    return _snapshot_job(job)


@router.get("/convert-comfy/{job_id}")
async def get_convert_comfy_status(job_id: str, request: Request):
    jobs = _get_jobs(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Conversion job not found")
    return _snapshot_job(job)
