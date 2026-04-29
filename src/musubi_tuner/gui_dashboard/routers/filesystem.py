"""Filesystem browsing API router."""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import requests
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fs", tags=["filesystem"])


@router.get("/cwd")
async def get_cwd():
    """Return the server's current working directory."""
    return {"cwd": os.getcwd()}


@router.get("/browse")
async def browse_directory(
    path: str = Query("", description="Directory path to browse"),
    show_files: bool = Query(True, description="Include files in results"),
):
    """List directory contents for the path picker UI."""
    if not path:
        # Return root drives on Windows, / on Unix
        if os.name == "nt":
            import string
            drives = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append({"name": drive, "path": drive, "is_dir": True})
            return {"path": "", "entries": drives, "parent": None}
        else:
            path = "/"

    p = Path(path)
    if not p.exists():
        return {"path": str(p), "entries": [], "parent": str(p.parent)}
    if not p.is_dir():
        return {"path": str(p.parent), "entries": [], "parent": str(p.parent.parent)}

    entries = []
    try:
        for item in sorted(p.iterdir()):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                entries.append({
                    "name": item.name,
                    "path": str(item),
                    "is_dir": True,
                })
            elif show_files:
                entries.append({
                    "name": item.name,
                    "path": str(item),
                    "is_dir": False,
                })
    except PermissionError:
        pass

    parent = str(p.parent) if p.parent != p else None
    return {"path": str(p), "entries": entries, "parent": parent}


@router.get("/exists")
async def check_exists(path: str = Query(..., description="Path to check")):
    """Check if a path exists and its type."""
    p = Path(path)
    return {
        "exists": p.exists(),
        "is_dir": p.is_dir() if p.exists() else False,
        "is_file": p.is_file() if p.exists() else False,
    }


@router.get("/read-file")
async def read_file(path: str = Query(..., description="File path to read")):
    """Read a text file and return its contents."""
    p = Path(path)
    if not p.is_file():
        return {"content": ""}
    try:
        return {"content": p.read_text(encoding="utf-8")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class WriteFileRequest(BaseModel):
    path: str
    content: str


@router.post("/write-file")
async def write_file(req: WriteFileRequest):
    """Write content to a text file."""
    p = Path(req.path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(req.content, encoding="utf-8")
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Checkpoint scanning
# ---------------------------------------------------------------------------

_LTX_PATTERNS = ["*ltx*", "*LTX*"]
_LTX_SUFFIXES = {".safetensors"}
_GEMMA_MARKERS = ["tokenizer.model", "config.json"]  # files inside gemma dirs
_GEMMA_NAME_HINTS = ["gemma"]


def _scan_ltx_checkpoints(roots: list[Path], max_depth: int = 3) -> list[str]:
    """Recursively search *roots* for safetensors files with 'ltx' in the name."""
    results: list[str] = []
    seen: set[str] = set()

    def _walk(p: Path, depth: int):
        if depth > max_depth:
            return
        try:
            for item in p.iterdir():
                if item.name.startswith("."):
                    continue
                if item.is_file() and item.suffix in _LTX_SUFFIXES:
                    low = item.name.lower()
                    if "ltx" in low:
                        s = str(item)
                        if s not in seen:
                            seen.add(s)
                            results.append(s)
                elif item.is_dir():
                    _walk(item, depth + 1)
        except (PermissionError, OSError):
            pass

    for root in roots:
        if root.is_dir():
            _walk(root, 0)
    return results


def _scan_gemma_roots(roots: list[Path], max_depth: int = 3) -> list[str]:
    """Recursively search for directories that look like Gemma model dirs."""
    results: list[str] = []
    seen: set[str] = set()

    def _is_gemma(p: Path) -> bool:
        # Must contain at least one marker file
        for marker in _GEMMA_MARKERS:
            if (p / marker).exists():
                return True
        return False

    def _walk(p: Path, depth: int):
        if depth > max_depth:
            return
        try:
            for item in p.iterdir():
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    low = item.name.lower()
                    if any(h in low for h in _GEMMA_NAME_HINTS) and _is_gemma(item):
                        s = str(item)
                        if s not in seen:
                            seen.add(s)
                            results.append(s)
                    else:
                        _walk(item, depth + 1)
        except (PermissionError, OSError):
            pass

    for root in roots:
        if root.is_dir():
            _walk(root, 0)
    return results


def _gather_scan_roots() -> list[Path]:
    """Collect common directories to scan for model files."""
    roots: list[Path] = []
    cwd = Path.cwd()
    roots.append(cwd)
    if cwd.parent != cwd:
        roots.append(cwd.parent)

    # HuggingFace cache
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.is_dir():
        roots.append(hf_cache)

    # Common model directories relative to cwd
    for sub in ("models", "checkpoints", "weights", "ckpts"):
        d = cwd / sub
        if d.is_dir():
            roots.append(d)

    return roots


@router.get("/scan-checkpoints")
async def scan_checkpoints(
    type: str = Query(..., description="Type: 'ltx2' or 'gemma'"),
    extra_paths: str = Query("", description="Comma-separated extra directories to scan"),
):
    """Scan common locations for model checkpoints."""
    roots = _gather_scan_roots()
    if extra_paths:
        for p in extra_paths.split(","):
            p = p.strip()
            if p:
                pp = Path(p)
                if pp.is_dir():
                    roots.append(pp)
    if type == "ltx2":
        return {"results": _scan_ltx_checkpoints(roots)}
    elif type == "gemma":
        return {"results": _scan_gemma_roots(roots)}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown type: {type}")


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

# Predefined download sources
_DOWNLOAD_PRESETS: dict[str, dict] = {
    "ltxav": {
        "label": "LTX-2.3 22B Dev",
        "repo": "Lightricks/LTX-2.3",
        "filename": "ltx-2.3-22b-dev.safetensors",
    },
    "gemma-unsloth": {
        "label": "Gemma 3 12B IT QAT Q4_0 Unquantized",
        "repo": "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized",
    },
}


def _download_preset_url(preset: dict) -> str:
    repo = preset["repo"]
    filename = preset.get("filename")
    if filename:
        return f"https://huggingface.co/{repo}/resolve/main/{filename}"
    return f"https://huggingface.co/{repo}"


def _nearest_existing_parent(path: Path) -> Path:
    current = path if path.is_dir() else path.parent
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _format_bytes_for_message(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    amount = float(value)
    index = 0
    while amount >= 1024 and index < len(units) - 1:
        amount /= 1024
        index += 1
    return f"{amount:.1f} {units[index]}" if index else f"{int(amount)} B"


def _estimate_download_size(preset: dict) -> tuple[int | None, str | None]:
    try:
        from huggingface_hub import HfApi, hf_hub_url
        from huggingface_hub.file_download import get_hf_file_metadata
        from huggingface_hub.hf_api import RepoFile
    except ImportError:
        return None, "huggingface_hub is not installed; download size cannot be estimated."

    repo = preset["repo"]
    filename = preset.get("filename")
    try:
        if filename:
            metadata = get_hf_file_metadata(hf_hub_url(repo_id=repo, filename=filename), token=None)
            return metadata.size or None, None

        api = HfApi()
        total = 0
        found_size = False
        for entry in api.list_repo_tree(repo_id=repo, recursive=True, expand=True, repo_type="model"):
            if isinstance(entry, RepoFile) and entry.size is not None:
                total += int(entry.size)
                found_size = True
        return (total if found_size else None), None
    except Exception as exc:
        logger.debug("Could not estimate download size for %s: %s", repo, exc)
        return None, f"Download size could not be estimated: {exc}"


def _build_download_preflight(preset_key: str, dest_path: str) -> dict:
    preset = _DOWNLOAD_PRESETS[preset_key]
    target_path = Path(dest_path)
    partial_path = target_path.with_name(f"{target_path.name}.partial")
    target_exists = target_path.exists()
    partial_exists = partial_path.exists()
    size_bytes, size_warning = _estimate_download_size(preset)

    disk_root = _nearest_existing_parent(target_path)
    free_bytes = None
    disk_warning = None
    try:
        free_bytes = shutil.disk_usage(str(disk_root)).free
    except Exception as exc:
        disk_warning = f"Free disk space could not be checked: {exc}"

    enough_space = None
    if size_bytes is not None and free_bytes is not None:
        enough_space = free_bytes >= int(size_bytes * 1.05)

    warnings = [w for w in (size_warning, disk_warning) if w]
    errors: list[str] = []
    if target_exists:
        errors.append(f"Target already exists: {target_path}")
    if partial_exists:
        errors.append(f"Partial download already exists: {partial_path}")
    if enough_space is False:
        errors.append(
            f"Not enough free disk space: need about {_format_bytes_for_message(size_bytes)}, "
            f"available {_format_bytes_for_message(free_bytes)}."
        )

    return {
        "ok": not errors,
        "preset": preset_key,
        "url": _download_preset_url(preset),
        "target_path": str(target_path),
        "target_exists": target_exists,
        "partial_path": str(partial_path),
        "partial_exists": partial_exists,
        "total_bytes": size_bytes,
        "free_bytes": free_bytes,
        "enough_space": enough_space,
        "warnings": warnings,
        "errors": errors,
    }


@router.get("/download-presets")
async def get_download_presets():
    """Return available model download presets."""
    return {
        "presets": {
            k: {
                "label": v["label"],
                "repo": v["repo"],
                "filename": v.get("filename"),
                "url": _download_preset_url(v),
            }
            for k, v in _DOWNLOAD_PRESETS.items()
        }
    }


class DownloadRequest(BaseModel):
    preset: str
    dest_path: str


@dataclasses.dataclass
class DownloadJob:
    job_id: str
    preset: str
    target_path: str
    state: str = "queued"
    message: str = "Queued"
    path: str | None = None
    error: str | None = None
    bytes_downloaded: int = 0
    total_bytes: int | None = None
    created_at: float = dataclasses.field(default_factory=time.time)
    updated_at: float = dataclasses.field(default_factory=time.time)
    cancel_event: threading.Event = dataclasses.field(default_factory=threading.Event)
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    thread: threading.Thread | None = None


class DownloadCancelled(Exception):
    """Raised when a download job is cancelled."""


def _get_download_jobs(request: Request) -> dict[str, DownloadJob]:
    jobs = getattr(request.app.state, "download_jobs", None)
    if jobs is None:
        jobs = {}
        request.app.state.download_jobs = jobs
    return jobs


def _prune_finished_downloads(jobs: dict[str, DownloadJob], keep: int = 20) -> None:
    finished = [
        job for job in jobs.values()
        if job.state in {"completed", "failed", "cancelled"}
    ]
    finished.sort(key=lambda job: job.updated_at, reverse=True)
    for job in finished[keep:]:
        jobs.pop(job.job_id, None)


def _set_job_state(job: DownloadJob, **fields) -> None:
    with job.lock:
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = time.time()


def _add_job_progress(job: DownloadJob, chunk_size: int) -> None:
    with job.lock:
        job.bytes_downloaded += chunk_size
        job.updated_at = time.time()


def _snapshot_download_job(job: DownloadJob) -> dict:
    with job.lock:
        progress_percent = None
        if job.total_bytes:
            progress_percent = round((job.bytes_downloaded / job.total_bytes) * 100, 1)
        return {
            "job_id": job.job_id,
            "preset": job.preset,
            "target_path": job.target_path,
            "state": job.state,
            "message": job.message,
            "path": job.path,
            "error": job.error,
            "bytes_downloaded": job.bytes_downloaded,
            "total_bytes": job.total_bytes,
            "progress_percent": progress_percent,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }


def _ensure_target_available(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Target already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _download_file(
    *,
    session: requests.Session,
    url: str,
    destination: Path,
    job: DownloadJob,
) -> None:
    _ensure_target_available(destination)
    temp_path = destination.with_name(f"{destination.name}.partial")
    if temp_path.exists():
        raise FileExistsError(f"Partial download already exists: {temp_path}")

    try:
        with session.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if job.cancel_event.is_set():
                        raise DownloadCancelled()
                    if not chunk:
                        continue
                    handle.write(chunk)
                    _add_job_progress(job, len(chunk))

        temp_path.replace(destination)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _run_download_job(job: DownloadJob) -> None:
    try:
        from huggingface_hub import HfApi, hf_hub_url
        from huggingface_hub.file_download import get_hf_file_metadata
        from huggingface_hub.hf_api import RepoFile
        from huggingface_hub.utils import build_hf_headers
    except ImportError as e:
        _set_job_state(
            job,
            state="failed",
            error="huggingface_hub not installed. Run: pip install huggingface_hub",
            message="huggingface_hub not installed",
        )
        return

    preset = _DOWNLOAD_PRESETS[job.preset]
    repo = preset["repo"]
    filename = preset.get("filename")
    headers = build_hf_headers(token=None, library_name="musubi-tuner")
    session = requests.Session()
    session.headers.update(headers)

    try:
        _set_job_state(job, state="running", message="Preparing download", error=None)
        if filename:
            target_path = Path(job.target_path)
            metadata = get_hf_file_metadata(hf_hub_url(repo_id=repo, filename=filename), token=None)
            _set_job_state(
                job,
                message=f"Downloading {filename}",
                total_bytes=metadata.size or None,
                path=str(target_path),
            )
            _download_file(
                session=session,
                url=hf_hub_url(repo_id=repo, filename=filename),
                destination=target_path,
                job=job,
            )
            _set_job_state(job, state="completed", message=f"Saved to {target_path}", path=str(target_path))
            return

        api = HfApi()
        repo_files = [
            entry
            for entry in api.list_repo_tree(repo_id=repo, recursive=True, expand=True, repo_type="model")
            if isinstance(entry, RepoFile)
        ]
        target_dir = Path(job.target_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        total_bytes = sum(entry.size or 0 for entry in repo_files) or None
        _set_job_state(
            job,
            total_bytes=total_bytes,
            path=str(target_dir),
            message=f"Downloading {len(repo_files)} files",
        )
        for index, entry in enumerate(repo_files, start=1):
            if job.cancel_event.is_set():
                raise DownloadCancelled()
            _set_job_state(job, message=f"Downloading {entry.path} ({index}/{len(repo_files)})")
            _download_file(
                session=session,
                url=hf_hub_url(repo_id=repo, filename=entry.path),
                destination=target_dir / entry.path,
                job=job,
            )
        _set_job_state(job, state="completed", message=f"Saved to {target_dir}", path=str(target_dir))
    except DownloadCancelled:
        _set_job_state(job, state="cancelled", message="Download cancelled")
    except Exception as e:
        logger.exception("Model download failed for preset %s", job.preset)
        _set_job_state(job, state="failed", error=str(e), message=str(e))
    finally:
        session.close()


@router.post("/download-model")
async def download_model(req: DownloadRequest, request: Request):
    """Start a model download in the background."""
    if req.preset not in _DOWNLOAD_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {req.preset}")

    preflight = _build_download_preflight(req.preset, req.dest_path)
    if not preflight["ok"]:
        raise HTTPException(status_code=409, detail="; ".join(preflight["errors"]))

    target_path = Path(preflight["target_path"])

    job = DownloadJob(
        job_id=uuid.uuid4().hex,
        preset=req.preset,
        target_path=str(target_path),
    )
    jobs = _get_download_jobs(request)
    _prune_finished_downloads(jobs)
    jobs[job.job_id] = job
    job.thread = threading.Thread(target=_run_download_job, args=(job,), daemon=True)
    job.thread.start()
    return _snapshot_download_job(job)


@router.post("/download-preflight")
async def download_preflight(req: DownloadRequest):
    """Check target conflicts and disk space before starting a model download."""
    if req.preset not in _DOWNLOAD_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {req.preset}")
    return _build_download_preflight(req.preset, req.dest_path)


@router.get("/download-model/{job_id}")
async def get_download_model_status(job_id: str, request: Request):
    """Return download job state."""
    jobs = _get_download_jobs(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Download job not found")
    return _snapshot_download_job(job)


@router.post("/download-model/{job_id}/cancel")
async def cancel_download_model(job_id: str, request: Request):
    """Cancel a running download job."""
    jobs = _get_download_jobs(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Download job not found")
    if job.state in {"completed", "failed", "cancelled"}:
        return _snapshot_download_job(job)
    job.cancel_event.set()
    _set_job_state(job, state="cancelling", message="Cancelling download")
    return _snapshot_download_job(job)
