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

import safetensors
import torch
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from musubi_tuner.gui_dashboard.command_builder import _effective_ltx2_checkpoint
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig

try:
    from musubi_tuner.ltx2_model_loading import detect_ltx2_dtype
except ModuleNotFoundError:
    detect_ltx2_dtype = None

try:
    from musubi_tuner.networks import lora_ltx2
except ImportError:
    lora_ltx2 = None

from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tools", tags=["tools"])


class ConvertComfyRequest(BaseModel):
    checkpoint_path: str
    output_path: str = ""
    base_model_path: str = ""
    device: str = "cpu"


class ConvertLoRARequest(BaseModel):
    checkpoint_path: str
    output_path: str = ""
    target_format: str = "other"
    diffusers_prefix: str = ""


class ExtractLoRARequest(BaseModel):
    base_model_path: str = ""
    finetuned_model_path: str
    output_path: str = ""
    target_preset: str = "full"
    connector_lora: bool = False
    extract_mode: str = "lora"
    rank_mode: str = "fro"
    dim: int = 64
    max_rank: int = 128
    fro_target: float = 0.98
    unsupported_tensors: str = "report"
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


@dataclasses.dataclass
class ConvertLoRAJob:
    job_id: str
    checkpoint_path: str
    output_path: str = ""
    target_format: str = "other"
    diffusers_prefix: str = ""
    state: str = "queued"
    message: str = "Queued"
    error: str = ""
    created_at: float = dataclasses.field(default_factory=time.time)
    updated_at: float = dataclasses.field(default_factory=time.time)
    finished_at: float | None = None
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


@dataclasses.dataclass
class ExtractLoRAJob:
    job_id: str
    base_model_path: str
    finetuned_model_path: str
    output_path: str = ""
    target_preset: str = "full"
    connector_lora: bool = False
    extract_mode: str = "lora"
    rank_mode: str = "fro"
    dim: int = 64
    max_rank: int = 128
    fro_target: float = 0.98
    unsupported_tensors: str = "report"
    device: str = "cpu"
    report_json: str = ""
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


def _get_lora_jobs(request: Request) -> dict[str, ConvertLoRAJob]:
    jobs = getattr(request.app.state, "convert_lora_jobs", None)
    if jobs is None:
        jobs = {}
        request.app.state.convert_lora_jobs = jobs
    return jobs


def _get_extract_lora_jobs(request: Request) -> dict[str, ExtractLoRAJob]:
    jobs = getattr(request.app.state, "extract_lora_jobs", None)
    if jobs is None:
        jobs = {}
        request.app.state.extract_lora_jobs = jobs
    return jobs


def _snapshot_job(job: ConvertComfyJob) -> dict:
    with job.lock:
        snapshot = {
            "job_id": job.job_id,
            "checkpoint_path": job.checkpoint_path,
            "output_path": job.output_path,
            "state": job.state,
            "message": job.message,
            "error": job.error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "finished_at": job.finished_at,
        }
        if hasattr(job, "base_model_path"):
            snapshot["base_model_path"] = job.base_model_path
        if hasattr(job, "target_format"):
            snapshot["target_format"] = job.target_format
        if hasattr(job, "diffusers_prefix"):
            snapshot["diffusers_prefix"] = job.diffusers_prefix
        if hasattr(job, "finetuned_model_path"):
            snapshot["finetuned_model_path"] = job.finetuned_model_path
        if hasattr(job, "target_preset"):
            snapshot["target_preset"] = job.target_preset
        if hasattr(job, "connector_lora"):
            snapshot["connector_lora"] = job.connector_lora
        if hasattr(job, "extract_mode"):
            snapshot["extract_mode"] = job.extract_mode
        if hasattr(job, "rank_mode"):
            snapshot["rank_mode"] = job.rank_mode
        if hasattr(job, "report_json"):
            snapshot["report_json"] = job.report_json
        return snapshot


def _set_job_state(job: ConvertComfyJob, **fields) -> None:
    with job.lock:
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = time.time()


def _prune_finished_jobs(jobs: dict[str, ConvertComfyJob], keep: int = 20) -> None:
    finished = [job for job in jobs.values() if job.state in {"completed", "failed"}]
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


def _default_comfy_output_path(input_path: str) -> str:
    path = Path(input_path)
    return str(path.parent / f"{path.stem}.comfy{path.suffix}")


def _default_lora_output_path(input_path: str, target_format: str) -> str:
    path = Path(input_path)
    suffix = "musubi" if target_format == "default" else "diffusers"
    return str(path.parent / f"{path.stem}.{suffix}{path.suffix}")


def _default_extract_lora_output_path(input_path: str) -> str:
    path = Path(input_path)
    return str(path.parent / f"{path.stem}.extracted_lora{path.suffix}")


def _checkpoint_needs_base_model(path: Path) -> bool:
    try:
        with safetensors.safe_open(str(path), framework="pt") as handle:
            keys = list(handle.keys())
    except Exception as exc:
        raise ValueError(f"Could not inspect checkpoint: {exc}") from exc
    return any(str(key).endswith(".lora_magnitude_vector.weight") for key in keys)


def _base_dtype_for_training(config: ProjectConfig, base_model_path: str) -> torch.dtype:
    try:
        if detect_ltx2_dtype is not None:
            return detect_ltx2_dtype(base_model_path)
    except Exception:
        pass
    mixed_precision = str(getattr(config.training, "mixed_precision", "") or "").lower()
    if mixed_precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if mixed_precision in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _build_convert_comfy_cmd(job: ConvertComfyJob, config: ProjectConfig, device: str, base_dtype: torch.dtype) -> list[str]:
    t = config.training
    cmd = [
        sys.executable,
        "-m",
        "musubi_tuner.ltx2_convert_lora_to_comfy",
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


def _run_convert_comfy_job(job: ConvertComfyJob, config: ProjectConfig, device: str) -> None:
    try:
        needs_base = bool(job.base_model_path)
        _set_job_state(
            job,
            state="running",
            message="Loading base checkpoint in converter process" if needs_base else "Converting checkpoint",
        )
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
        _set_job_state(job, state="failed", message=str(exc), error=str(exc), finished_at=time.time())


def _run_convert_lora_job(job: ConvertLoRAJob, config: ProjectConfig) -> None:
    try:
        _set_job_state(job, state="running", message="Converting adapter checkpoint")
        cmd = [
            sys.executable,
            "-m",
            "musubi_tuner.convert_lora",
            "--input",
            job.checkpoint_path,
            "--output",
            job.output_path,
            "--target",
            job.target_format,
        ]
        if job.diffusers_prefix:
            cmd += ["--diffusers_prefix", job.diffusers_prefix]

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

        _set_job_state(
            job,
            state="completed",
            message=f"Saved to {job.output_path}",
            finished_at=time.time(),
        )
    except Exception as exc:
        logger.exception("Generic LoRA conversion failed")
        _set_job_state(job, state="failed", message=str(exc), error=str(exc), finished_at=time.time())


def _build_extract_lora_cmd(job: ExtractLoRAJob) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "musubi_tuner.ltx2_extract_lora",
        "--base_model",
        job.base_model_path,
        "--finetuned_model",
        job.finetuned_model_path,
        "--save_to",
        job.output_path,
        "--target_preset",
        job.target_preset,
        "--extract_mode",
        job.extract_mode,
        "--rank_mode",
        job.rank_mode,
        "--dim",
        str(job.dim),
        "--max_rank",
        str(job.max_rank),
        "--fro_target",
        str(job.fro_target),
        "--unsupported_tensors",
        job.unsupported_tensors,
        "--report_json",
        job.report_json,
        "--device",
        job.device or "cpu",
    ]
    if job.connector_lora:
        cmd.append("--connector_lora")
    return cmd


def _run_extract_lora_job(job: ExtractLoRAJob, config: ProjectConfig) -> None:
    try:
        _set_job_state(job, state="running", message="Extracting LoRA from fine-tuned checkpoint")
        cmd = _build_extract_lora_cmd(job)
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
            raise RuntimeError(output_tail or f"Extractor exited with code {result.returncode}")

        _set_job_state(
            job,
            state="completed",
            message=f"Saved to {job.output_path}",
            finished_at=time.time(),
        )
    except Exception as exc:
        logger.exception("LoRA extraction failed")
        _set_job_state(job, state="failed", message=str(exc), error=str(exc), finished_at=time.time())


@router.post("/convert-comfy")
async def start_convert_comfy(req: ConvertComfyRequest, request: Request):
    config = _get_config(request)
    try:
        checkpoint_path = _resolve_project_path(config, req.checkpoint_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not checkpoint_path.is_file():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix.casefold() != ".safetensors":
        raise HTTPException(status_code=400, detail="Checkpoint must be a .safetensors file")

    try:
        output_path: Optional[Path] = _resolve_project_path(config, req.output_path) if req.output_path.strip() else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if output_path is not None and output_path.suffix.casefold() != ".safetensors":
        raise HTTPException(status_code=400, detail="Output path must be a .safetensors file")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        needs_base = _checkpoint_needs_base_model(checkpoint_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    base_model_path: Optional[Path] = None
    if needs_base or req.base_model_path.strip():
        base_model = req.base_model_path.strip() or _effective_ltx2_checkpoint(config, config.training.ltx2_checkpoint)
        try:
            base_model_path = _resolve_project_path(config, base_model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
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


@router.post("/convert-lora")
async def start_convert_lora(req: ConvertLoRARequest, request: Request):
    config = _get_config(request)
    try:
        checkpoint_path = _resolve_project_path(config, req.checkpoint_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not checkpoint_path.is_file():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix.casefold() != ".safetensors":
        raise HTTPException(status_code=400, detail="Checkpoint must be a .safetensors file")

    target_format = (req.target_format or "other").strip().lower()
    if target_format not in {"other", "default"}:
        raise HTTPException(status_code=400, detail="Target format must be 'other' or 'default'")

    try:
        output_path = (
            _resolve_project_path(config, req.output_path)
            if req.output_path.strip()
            else Path(_default_lora_output_path(str(checkpoint_path), target_format))
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if output_path.suffix.casefold() != ".safetensors":
        raise HTTPException(status_code=400, detail="Output path must be a .safetensors file")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = _get_lora_jobs(request)
    _prune_finished_jobs(jobs)
    job = ConvertLoRAJob(
        job_id=uuid.uuid4().hex,
        checkpoint_path=str(checkpoint_path),
        output_path=str(output_path),
        target_format=target_format,
        diffusers_prefix=(req.diffusers_prefix or "").strip(),
    )
    jobs[job.job_id] = job
    thread = threading.Thread(target=_run_convert_lora_job, args=(job, config), daemon=True)
    thread.start()
    return _snapshot_job(job)


@router.get("/convert-lora/{job_id}")
async def get_convert_lora_status(job_id: str, request: Request):
    jobs = _get_lora_jobs(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Conversion job not found")
    return _snapshot_job(job)


@router.post("/extract-lora")
async def start_extract_lora(req: ExtractLoRARequest, request: Request):
    config = _get_config(request)
    base_model_raw = req.base_model_path.strip() or _effective_ltx2_checkpoint(config, config.training.ltx2_checkpoint)
    try:
        base_model_path = _resolve_project_path(config, base_model_raw)
        finetuned_model_path = _resolve_project_path(config, req.finetuned_model_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not base_model_path.is_file():
        raise HTTPException(status_code=404, detail=f"Base LTX-2 checkpoint not found: {base_model_path}")
    if not finetuned_model_path.is_file():
        raise HTTPException(status_code=404, detail=f"Fine-tuned checkpoint not found: {finetuned_model_path}")
    if base_model_path.suffix.casefold() != ".safetensors" or finetuned_model_path.suffix.casefold() != ".safetensors":
        raise HTTPException(status_code=400, detail="Input checkpoints must be .safetensors files")

    try:
        output_path = (
            _resolve_project_path(config, req.output_path)
            if req.output_path.strip()
            else Path(_default_extract_lora_output_path(str(finetuned_model_path)))
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if output_path.suffix.casefold() != ".safetensors":
        raise HTTPException(status_code=400, detail="Output path must be a .safetensors file")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_preset = (req.target_preset or "full").strip()
    valid_presets = ({"custom"} if lora_ltx2 is None else set(lora_ltx2.LTX2_LORA_TARGET_PRESETS.keys()) | {"custom"})
    if target_preset not in valid_presets:
        raise HTTPException(status_code=400, detail=f"Unknown target preset: {target_preset}")
    extract_mode = (req.extract_mode or "lora").strip()
    if extract_mode not in {"lora", "dora"}:
        raise HTTPException(status_code=400, detail="extract_mode must be 'lora' or 'dora'")
    rank_mode = (req.rank_mode or "fro").strip()
    if rank_mode not in {"fixed", "fro", "quantile", "knee", "relative_drop"}:
        raise HTTPException(status_code=400, detail="Invalid rank mode")
    unsupported_tensors = (req.unsupported_tensors or "report").strip()
    if unsupported_tensors not in {"report", "skip", "error", "sidecar"}:
        raise HTTPException(status_code=400, detail="Invalid unsupported tensor mode")

    report_json = str(output_path.with_suffix(".report.json"))
    jobs = _get_extract_lora_jobs(request)
    _prune_finished_jobs(jobs)
    job = ExtractLoRAJob(
        job_id=uuid.uuid4().hex,
        base_model_path=str(base_model_path),
        finetuned_model_path=str(finetuned_model_path),
        output_path=str(output_path),
        target_preset=target_preset,
        connector_lora=bool(req.connector_lora),
        extract_mode=extract_mode,
        rank_mode=rank_mode,
        dim=max(1, int(req.dim)),
        max_rank=max(1, int(req.max_rank)),
        fro_target=float(req.fro_target),
        unsupported_tensors=unsupported_tensors,
        device=(req.device or "cpu").strip(),
        report_json=report_json,
    )
    jobs[job.job_id] = job
    thread = threading.Thread(target=_run_extract_lora_job, args=(job, config), daemon=True)
    thread.start()
    return _snapshot_job(job)


@router.get("/extract-lora/{job_id}")
async def get_extract_lora_status(job_id: str, request: Request):
    jobs = _get_extract_lora_jobs(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Extraction job not found")
    return _snapshot_job(job)


class QuantizeInt8ConvRotRequest(BaseModel):
    checkpoint_path: str
    output_path: str = ""
    groupsize: str = "auto"
    mse_clip: bool = True
    calc_device: str = "cpu"
    quality_report: bool = True
    convrot_policy: str = ""


@dataclasses.dataclass
class QuantizeInt8ConvRotJob:
    job_id: str
    checkpoint_path: str
    output_path: str = ""
    groupsize: str = "auto"
    mse_clip: bool = True
    calc_device: str = "cpu"
    quality_report: bool = True
    convrot_policy: str = ""
    state: str = "queued"
    message: str = "Queued"
    error: str = ""
    created_at: float = dataclasses.field(default_factory=time.time)
    updated_at: float = dataclasses.field(default_factory=time.time)
    finished_at: float | None = None
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


def _get_quantize_int8cr_jobs(request: Request) -> dict[str, QuantizeInt8ConvRotJob]:
    jobs = getattr(request.app.state, "quantize_int8cr_jobs", None)
    if jobs is None:
        jobs = {}
        request.app.state.quantize_int8cr_jobs = jobs
    return jobs


def _default_int8cr_output_path(input_path: str) -> str:
    path = Path(input_path)
    return str(path.parent / f"{path.stem}.int8cr{path.suffix}")


def _run_quantize_int8cr_job(job: QuantizeInt8ConvRotJob, config: ProjectConfig) -> None:
    try:
        _set_job_state(job, state="running", message="Pre-quantizing transformer weights to INT8 ConvRot")
        cmd = [
            sys.executable,
            "-m",
            "musubi_tuner.ltx2_quantize_int8_convrot",
            "--input_model",
            job.checkpoint_path,
            "--output_model",
            job.output_path,
            "--groupsize",
            job.groupsize or "auto",
            "--calc_device",
            job.calc_device or "cpu",
        ]
        if not job.mse_clip:
            cmd.append("--no_mse_clip")
        if not job.quality_report:
            cmd.append("--no_quality_report")
        if job.convrot_policy:
            cmd += ["--convrot_policy", job.convrot_policy]

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
            raise RuntimeError(output_tail or f"Quantizer exited with code {result.returncode}")

        _set_job_state(
            job,
            state="completed",
            message=f"Saved to {job.output_path}",
            finished_at=time.time(),
        )
    except Exception as exc:
        logger.exception("INT8 ConvRot pre-quantization failed")
        _set_job_state(job, state="failed", message=str(exc), error=str(exc), finished_at=time.time())


@router.post("/quantize-int8cr")
async def start_quantize_int8cr(req: QuantizeInt8ConvRotRequest, request: Request):
    config = _get_config(request)
    try:
        checkpoint_path = _resolve_project_path(config, req.checkpoint_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not checkpoint_path.is_file():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix.casefold() != ".safetensors":
        raise HTTPException(status_code=400, detail="Checkpoint must be a .safetensors file")

    try:
        output_path = (
            _resolve_project_path(config, req.output_path)
            if req.output_path.strip()
            else Path(_default_int8cr_output_path(str(checkpoint_path)))
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if output_path.suffix.casefold() != ".safetensors":
        raise HTTPException(status_code=400, detail="Output path must be a .safetensors file")
    if output_path == checkpoint_path:
        raise HTTPException(status_code=400, detail="Output path must differ from the input checkpoint")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = (req.calc_device or "cpu").strip().lower()
    if device not in {"cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="calc_device must be 'cpu' or 'cuda'")
    convrot_policy = ""
    if req.convrot_policy.strip():
        try:
            policy_path = _resolve_project_path(config, req.convrot_policy)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not policy_path.is_file():
            raise HTTPException(status_code=404, detail=f"ConvRot policy not found: {policy_path}")
        convrot_policy = str(policy_path)

    jobs = _get_quantize_int8cr_jobs(request)
    _prune_finished_jobs(jobs)
    job = QuantizeInt8ConvRotJob(
        job_id=uuid.uuid4().hex,
        checkpoint_path=str(checkpoint_path),
        output_path=str(output_path),
        groupsize=(req.groupsize or "auto").strip(),
        mse_clip=bool(req.mse_clip),
        calc_device=device,
        quality_report=bool(req.quality_report),
        convrot_policy=convrot_policy,
    )
    jobs[job.job_id] = job
    thread = threading.Thread(target=_run_quantize_int8cr_job, args=(job, config), daemon=True)
    thread.start()
    return _snapshot_job(job)


@router.get("/quantize-int8cr/{job_id}")
async def get_quantize_int8cr_status(job_id: str, request: Request):
    jobs = _get_quantize_int8cr_jobs(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Quantization job not found")
    return _snapshot_job(job)


class QuantizeInt4ConvRotRequest(BaseModel):
    checkpoint_path: str
    output_path: str = ""
    groupsize: str = "auto"
    mse_clip: bool = True
    calc_device: str = "cpu"
    quality_report: bool = True
    stabilizer_rank: int = 0
    scale_refine_steps: int = 0
    int4_convrot_group_scales: int = 0
    int4_convrot_group_ratio_q8: bool = False
    int4_convrot_compare_group_scales: str = ""
    convrot_policy: str = ""


@dataclasses.dataclass
class QuantizeInt4ConvRotJob:
    job_id: str
    checkpoint_path: str
    output_path: str = ""
    groupsize: str = "auto"
    mse_clip: bool = True
    calc_device: str = "cpu"
    quality_report: bool = True
    stabilizer_rank: int = 0
    scale_refine_steps: int = 0
    int4_convrot_group_scales: int = 0
    int4_convrot_group_ratio_q8: bool = False
    int4_convrot_compare_group_scales: str = ""
    convrot_policy: str = ""
    state: str = "queued"
    message: str = "Queued"
    error: str = ""
    created_at: float = dataclasses.field(default_factory=time.time)
    updated_at: float = dataclasses.field(default_factory=time.time)
    finished_at: float | None = None
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


def _get_quantize_int4cr_jobs(request: Request) -> dict[str, QuantizeInt4ConvRotJob]:
    jobs = getattr(request.app.state, "quantize_int4cr_jobs", None)
    if jobs is None:
        jobs = {}
        request.app.state.quantize_int4cr_jobs = jobs
    return jobs


def _default_int4cr_output_path(input_path: str) -> str:
    path = Path(input_path)
    return str(path.parent / f"{path.stem}.int4cr{path.suffix}")


def _run_quantize_int4cr_job(job: QuantizeInt4ConvRotJob, config: ProjectConfig) -> None:
    try:
        _set_job_state(job, state="running", message="Pre-quantizing transformer weights to INT4 ConvRot")
        cmd = [
            sys.executable,
            "-m",
            "musubi_tuner.ltx2_quantize_int4_convrot",
            "--input_model",
            job.checkpoint_path,
            "--output_model",
            job.output_path,
            "--groupsize",
            job.groupsize or "auto",
            "--calc_device",
            job.calc_device or "cpu",
        ]
        if not job.mse_clip:
            cmd.append("--no_mse_clip")
        if not job.quality_report:
            cmd.append("--no_quality_report")
        if job.stabilizer_rank > 0:
            cmd += ["--stabilizer_rank", str(job.stabilizer_rank)]
        if job.scale_refine_steps > 0:
            cmd += ["--scale_refine_steps", str(job.scale_refine_steps)]
        if job.int4_convrot_group_scales > 0:
            cmd += ["--int4_convrot_group_scales", str(job.int4_convrot_group_scales)]
        if job.int4_convrot_group_ratio_q8:
            cmd.append("--int4_convrot_group_ratio_q8")
        if job.int4_convrot_compare_group_scales:
            cmd += ["--int4_convrot_compare_group_scales", job.int4_convrot_compare_group_scales]
        if job.convrot_policy:
            cmd += ["--convrot_policy", job.convrot_policy]

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
            raise RuntimeError(output_tail or f"Quantizer exited with code {result.returncode}")

        _set_job_state(
            job,
            state="completed",
            message=f"Saved to {job.output_path}",
            finished_at=time.time(),
        )
    except Exception as exc:
        logger.exception("INT4 ConvRot pre-quantization failed")
        _set_job_state(job, state="failed", message=str(exc), error=str(exc), finished_at=time.time())


@router.post("/quantize-int4cr")
async def start_quantize_int4cr(req: QuantizeInt4ConvRotRequest, request: Request):
    config = _get_config(request)
    try:
        checkpoint_path = _resolve_project_path(config, req.checkpoint_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not checkpoint_path.is_file():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix.casefold() != ".safetensors":
        raise HTTPException(status_code=400, detail="Checkpoint must be a .safetensors file")

    try:
        output_path = (
            _resolve_project_path(config, req.output_path)
            if req.output_path.strip()
            else Path(_default_int4cr_output_path(str(checkpoint_path)))
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if output_path.suffix.casefold() != ".safetensors":
        raise HTTPException(status_code=400, detail="Output path must be a .safetensors file")
    if output_path == checkpoint_path:
        raise HTTPException(status_code=400, detail="Output path must differ from the input checkpoint")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = (req.calc_device or "cpu").strip().lower()
    if device not in {"cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="calc_device must be 'cpu' or 'cuda'")

    stabilizer_rank = int(req.stabilizer_rank or 0)
    if stabilizer_rank < 0:
        raise HTTPException(status_code=400, detail="stabilizer_rank must be >= 0")
    scale_refine_steps = int(req.scale_refine_steps or 0)
    if scale_refine_steps < 0:
        raise HTTPException(status_code=400, detail="scale_refine_steps must be >= 0")
    group_scales = int(req.int4_convrot_group_scales or 0)
    if group_scales < 0 or (group_scales and (group_scales < 16 or group_scales & (group_scales - 1))):
        raise HTTPException(
            status_code=400,
            detail="int4_convrot_group_scales must be 0 or a power of two >= 16",
        )
    if req.int4_convrot_group_ratio_q8 and not group_scales:
        raise HTTPException(
            status_code=400,
            detail="int4_convrot_group_ratio_q8 requires int4_convrot_group_scales",
        )
    compare_group_scales = (req.int4_convrot_compare_group_scales or "").strip()
    if compare_group_scales:
        try:
            compare_values = [int(part.strip()) for part in compare_group_scales.replace(";", ",").split(",") if part.strip()]
        except ValueError:
            compare_values = []
        if not compare_values or any(value < 0 or (value and (value < 16 or value & (value - 1))) for value in compare_values):
            raise HTTPException(
                status_code=400,
                detail="int4_convrot_compare_group_scales must be a comma-separated list of 0 or powers of two >= 16",
            )
        if not req.quality_report:
            raise HTTPException(
                status_code=400,
                detail="int4_convrot_compare_group_scales requires quality_report",
            )
    convrot_policy = ""
    if req.convrot_policy.strip():
        try:
            policy_path = _resolve_project_path(config, req.convrot_policy)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not policy_path.is_file():
            raise HTTPException(status_code=404, detail=f"ConvRot policy not found: {policy_path}")
        convrot_policy = str(policy_path)

    jobs = _get_quantize_int4cr_jobs(request)
    _prune_finished_jobs(jobs)
    job = QuantizeInt4ConvRotJob(
        job_id=uuid.uuid4().hex,
        checkpoint_path=str(checkpoint_path),
        output_path=str(output_path),
        groupsize=(req.groupsize or "auto").strip(),
        mse_clip=bool(req.mse_clip),
        calc_device=device,
        quality_report=bool(req.quality_report),
        stabilizer_rank=stabilizer_rank,
        scale_refine_steps=scale_refine_steps,
        int4_convrot_group_scales=group_scales,
        int4_convrot_group_ratio_q8=bool(req.int4_convrot_group_ratio_q8),
        int4_convrot_compare_group_scales=compare_group_scales,
        convrot_policy=convrot_policy,
    )
    jobs[job.job_id] = job
    thread = threading.Thread(target=_run_quantize_int4cr_job, args=(job, config), daemon=True)
    thread.start()
    return _snapshot_job(job)


@router.get("/quantize-int4cr/{job_id}")
async def get_quantize_int4cr_status(job_id: str, request: Request):
    jobs = _get_quantize_int4cr_jobs(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Quantization job not found")
    return _snapshot_job(job)


class QuantizeInt8WeightsRequest(BaseModel):
    checkpoint_path: str
    output_path: str = ""
    targets: str = "video"
    group_size: str = "0"
    convrot: str = "auto"
    dtype: str = "bfloat16"
    calc_device: str = "cpu"


@dataclasses.dataclass
class QuantizeInt8WeightsJob:
    job_id: str
    checkpoint_path: str
    output_path: str = ""
    targets: str = "video"
    group_size: str = "0"
    convrot: str = "auto"
    dtype: str = "bfloat16"
    calc_device: str = "cpu"
    state: str = "queued"
    message: str = "Queued"
    error: str = ""
    created_at: float = dataclasses.field(default_factory=time.time)
    updated_at: float = dataclasses.field(default_factory=time.time)
    finished_at: float | None = None
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


def _get_quantize_int8w_jobs(request: Request) -> dict[str, QuantizeInt8WeightsJob]:
    jobs = getattr(request.app.state, "quantize_int8w_jobs", None)
    if jobs is None:
        jobs = {}
        request.app.state.quantize_int8w_jobs = jobs
    return jobs


def _default_int8w_output_path(input_path: str) -> str:
    path = Path(input_path)
    return str(path.parent / f"{path.stem}.int8w{path.suffix}")


def _run_quantize_int8w_job(job: QuantizeInt8WeightsJob, config: ProjectConfig) -> None:
    try:
        _set_job_state(job, state="running", message="Pre-quantizing transformer weights to the INT8 weight-only grid")
        cmd = [
            sys.executable,
            "-m",
            "musubi_tuner.ltx2_export_int8_weights",
            "--input_model",
            job.checkpoint_path,
            "--output_model",
            job.output_path,
            "--targets",
            job.targets or "video",
            "--group_size",
            str(job.group_size or "0"),
            "--convrot",
            job.convrot or "auto",
            "--dtype",
            job.dtype or "bfloat16",
            "--calc_device",
            job.calc_device or "cpu",
        ]
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
            raise RuntimeError(output_tail or f"Exporter exited with code {result.returncode}")

        _set_job_state(job, state="completed", message=f"Saved to {job.output_path}", finished_at=time.time())
    except Exception as exc:
        logger.exception("INT8 weight-only pre-quantization failed")
        _set_job_state(job, state="failed", message=str(exc), error=str(exc), finished_at=time.time())


@router.post("/quantize-int8w")
async def start_quantize_int8w(req: QuantizeInt8WeightsRequest, request: Request):
    config = _get_config(request)
    try:
        checkpoint_path = _resolve_project_path(config, req.checkpoint_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not checkpoint_path.is_file():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix.casefold() != ".safetensors":
        raise HTTPException(status_code=400, detail="Checkpoint must be a .safetensors file")

    try:
        output_path = (
            _resolve_project_path(config, req.output_path)
            if req.output_path.strip()
            else Path(_default_int8w_output_path(str(checkpoint_path)))
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if output_path.suffix.casefold() != ".safetensors":
        raise HTTPException(status_code=400, detail="Output path must be a .safetensors file")
    if output_path == checkpoint_path:
        raise HTTPException(status_code=400, detail="Output path must differ from the input checkpoint")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = (req.calc_device or "cpu").strip().lower()
    if device not in {"cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="calc_device must be 'cpu' or 'cuda'")
    dtype = (req.dtype or "bfloat16").strip().lower()
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise HTTPException(status_code=400, detail="dtype must be bfloat16, float16, or float32")

    jobs = _get_quantize_int8w_jobs(request)
    _prune_finished_jobs(jobs)
    job = QuantizeInt8WeightsJob(
        job_id=uuid.uuid4().hex,
        checkpoint_path=str(checkpoint_path),
        output_path=str(output_path),
        targets=(req.targets or "video").strip(),
        group_size=(str(req.group_size) or "0").strip(),
        convrot=(req.convrot or "auto").strip(),
        dtype=dtype,
        calc_device=device,
    )
    jobs[job.job_id] = job
    thread = threading.Thread(target=_run_quantize_int8w_job, args=(job, config), daemon=True)
    thread.start()
    return _snapshot_job(job)


@router.get("/quantize-int8w/{job_id}")
async def get_quantize_int8w_status(job_id: str, request: Request):
    jobs = _get_quantize_int8w_jobs(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Quantization job not found")
    return _snapshot_job(job)
