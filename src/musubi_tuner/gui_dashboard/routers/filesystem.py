"""Filesystem browsing API router."""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import string
import threading
import time
import uuid
from pathlib import Path

import requests
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

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

_SAFETENSORS_SUFFIXES = {".safetensors"}
_LTX_SUFFIXES = _SAFETENSORS_SUFFIXES
_SCAN_TYPES = {"ltx2", "gemma", "gemma_safetensors"}
_GEMMA_MARKERS = ["tokenizer.model", "config.json"]  # files inside gemma dirs
_GEMMA_NAME_HINTS = ["gemma"]
_SCAN_MAX_DEPTH = 5
_SCAN_SKIP_DIRS = {
    "$recycle.bin",
    "__pycache__",
    "node_modules",
    "program files",
    "program files (x86)",
    "programdata",
    "system volume information",
    "venv",
    "windows",
}


@dataclasses.dataclass
class ScanJob:
    job_id: str
    scan_type: str
    target_name: str = ""
    related_specs: dict[tuple[str, str], tuple[str, str]] = dataclasses.field(default_factory=dict)
    state: str = "queued"
    current_path: str = ""
    results: list[str] = dataclasses.field(default_factory=list)
    error: str = ""
    cancel_requested: bool = False
    started_at: float = dataclasses.field(default_factory=time.time)
    finished_at: float | None = None
    last_progress_at: float = 0.0
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


class ScanCheckpointRequest(BaseModel):
    type: str
    extra_paths: str = ""
    target_path: str = ""
    related_targets: dict[str, str] = Field(default_factory=dict)


_SCAN_JOBS: dict[str, ScanJob] = {}
_SCAN_JOBS_LOCK = threading.Lock()
_SCAN_RESULT_CACHE: dict[tuple[str, str], list[str]] = {}
_SCAN_RESULT_CACHE_LOCK = threading.Lock()


def _scan_job_payload(job: ScanJob) -> dict:
    with job.lock:
        state = "cancelling" if job.state == "running" and job.cancel_requested else job.state
        return {
            "job_id": job.job_id,
            "type": job.scan_type,
            "target_name": job.target_name,
            "state": state,
            "current_path": job.current_path,
            "results": list(job.results),
            "error": job.error,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        }


def _set_scan_progress(job: ScanJob, path: Path, force: bool = False):
    now = time.monotonic()
    with job.lock:
        if force or now - job.last_progress_at >= 1.0:
            job.current_path = str(path)
            job.last_progress_at = now


def _scan_roots(extra_paths: str = "") -> list[Path]:
    roots = _gather_scan_roots()
    if extra_paths:
        for p in extra_paths.split(","):
            p = p.strip()
            if p:
                pp = Path(p)
                if pp.is_dir():
                    roots.append(pp)

    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)
    return unique_roots


def _target_leaf_name(path_or_name: str = "") -> str:
    normalized = str(path_or_name or "").replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1].strip()


def _same_leaf_name(candidate_name: str, target_name: str) -> bool:
    return bool(target_name) and candidate_name.casefold() == target_name.casefold()


def _add_scan_result(results: list[str], seen: set[str], path: Path):
    s = str(path)
    key = s.casefold()
    if key not in seen:
        seen.add(key)
        results.append(s)


def _scan_cache_key(scan_type: str, target_name: str) -> tuple[str, str]:
    return (scan_type, target_name.casefold())


def _scan_result_still_valid(scan_type: str, path: Path, target_name: str) -> bool:
    if not _same_leaf_name(path.name, target_name):
        return False
    if scan_type == "gemma":
        return path.is_dir()
    if scan_type in {"ltx2", "gemma_safetensors"}:
        return path.is_file() and path.suffix.casefold() in _SAFETENSORS_SUFFIXES
    return False


def _get_cached_scan_results(scan_type: str, target_name: str) -> list[str]:
    if not target_name:
        return []
    key = _scan_cache_key(scan_type, target_name)
    with _SCAN_RESULT_CACHE_LOCK:
        cached = list(_SCAN_RESULT_CACHE.get(key, []))
    if not cached:
        return []
    valid = [path for path in cached if _scan_result_still_valid(scan_type, Path(path), target_name)]
    with _SCAN_RESULT_CACHE_LOCK:
        if valid:
            _SCAN_RESULT_CACHE[key] = valid
        else:
            _SCAN_RESULT_CACHE.pop(key, None)
    return valid


def _store_scan_results(scan_type: str, target_name: str, results: list[str]):
    if not target_name or not results:
        return
    valid = [path for path in results if _scan_result_still_valid(scan_type, Path(path), target_name)]
    if not valid:
        return
    with _SCAN_RESULT_CACHE_LOCK:
        _SCAN_RESULT_CACHE[_scan_cache_key(scan_type, target_name)] = valid


def _completed_scan_payload(scan_type: str, target_name: str, results: list[str]) -> dict:
    job = ScanJob(
        job_id="",
        scan_type=scan_type,
        target_name=target_name,
        state="completed",
        results=results,
        started_at=time.time(),
        finished_at=time.time(),
    )
    return _scan_job_payload(job)


def _scan_target_specs(scan_type: str, target_path: str, related_targets: dict[str, str] | None = None) -> dict[tuple[str, str], tuple[str, str]]:
    specs: dict[tuple[str, str], tuple[str, str]] = {}

    def add(target_type: str, path_or_name: str):
        if target_type not in _SCAN_TYPES:
            return
        target_name = _target_leaf_name(path_or_name)
        if not target_name:
            return
        specs[_scan_cache_key(target_type, target_name)] = (target_type, target_name)

    add(scan_type, target_path)
    for target_type, path_or_name in (related_targets or {}).items():
        add(target_type, path_or_name)
    return specs


def _scan_exact_targets(
    roots: list[Path],
    specs: dict[tuple[str, str], tuple[str, str]],
    max_depth: int = _SCAN_MAX_DEPTH,
    should_cancel=None,
    on_progress=None,
) -> dict[tuple[str, str], list[str]]:
    """Recursively search *roots* once for multiple exact file/folder targets."""
    results = {key: [] for key in specs}
    seen = {key: set() for key in specs}

    if not specs:
        return results

    def _matches_file(scan_type: str, item: Path, target_name: str) -> bool:
        return scan_type in {"ltx2", "gemma_safetensors"} and _same_leaf_name(item.name, target_name) and item.suffix.casefold() in _SAFETENSORS_SUFFIXES

    def _matches_dir(scan_type: str, item: Path, target_name: str) -> bool:
        return scan_type == "gemma" and _same_leaf_name(item.name, target_name)

    def _walk(p: Path, depth: int):
        if should_cancel and should_cancel():
            return
        if depth > max_depth:
            return
        if on_progress:
            on_progress(p)
        try:
            for item in p.iterdir():
                if should_cancel and should_cancel():
                    return
                if item.name.startswith("."):
                    continue
                if item.is_file():
                    for key, (scan_type, target_name) in specs.items():
                        if _matches_file(scan_type, item, target_name):
                            _add_scan_result(results[key], seen[key], item)
                elif item.is_dir():
                    for key, (scan_type, target_name) in specs.items():
                        if _matches_dir(scan_type, item, target_name):
                            _add_scan_result(results[key], seen[key], item)
                    if item.name.lower() in _SCAN_SKIP_DIRS:
                        continue
                    _walk(item, depth + 1)
        except (PermissionError, OSError):
            pass

    for root in roots:
        if should_cancel and should_cancel():
            break
        if root.is_dir():
            _walk(root, 0)
    return results


def _scan_exact_files(
    roots: list[Path],
    target_name: str,
    suffixes: set[str] | None = None,
    max_depth: int = _SCAN_MAX_DEPTH,
    should_cancel=None,
    on_progress=None,
) -> list[str]:
    """Recursively search *roots* for files whose basename exactly matches *target_name*."""
    results: list[str] = []
    seen: set[str] = set()
    target_suffix = Path(target_name).suffix.casefold()
    suffixes = {suffix.casefold() for suffix in suffixes or set()}
    if not target_name or (suffixes and target_suffix not in suffixes):
        return results

    def _walk(p: Path, depth: int):
        if should_cancel and should_cancel():
            return
        if depth > max_depth:
            return
        if on_progress:
            on_progress(p)
        try:
            for item in p.iterdir():
                if should_cancel and should_cancel():
                    return
                if item.name.startswith("."):
                    continue
                if item.is_file():
                    if _same_leaf_name(item.name, target_name) and (not suffixes or item.suffix.casefold() in suffixes):
                        _add_scan_result(results, seen, item)
                elif item.is_dir():
                    if item.name.lower() in _SCAN_SKIP_DIRS:
                        continue
                    _walk(item, depth + 1)
        except (PermissionError, OSError):
            pass

    for root in roots:
        if should_cancel and should_cancel():
            break
        if root.is_dir():
            _walk(root, 0)
    return results


def _scan_ltx_checkpoints(
    roots: list[Path],
    target_name: str = "",
    max_depth: int = _SCAN_MAX_DEPTH,
    should_cancel=None,
    on_progress=None,
) -> list[str]:
    """Recursively search *roots* for LTX checkpoint files."""
    if target_name:
        return _scan_exact_files(roots, target_name, _LTX_SUFFIXES, max_depth, should_cancel, on_progress)

    results: list[str] = []
    seen: set[str] = set()

    def _walk(p: Path, depth: int):
        if should_cancel and should_cancel():
            return
        if depth > max_depth:
            return
        if on_progress:
            on_progress(p)
        try:
            for item in p.iterdir():
                if should_cancel and should_cancel():
                    return
                if item.name.startswith("."):
                    continue
                if item.is_file() and item.suffix in _LTX_SUFFIXES:
                    low = item.name.lower()
                    if "ltx" in low:
                        _add_scan_result(results, seen, item)
                elif item.is_dir():
                    if item.name.lower() in _SCAN_SKIP_DIRS:
                        continue
                    _walk(item, depth + 1)
        except (PermissionError, OSError):
            pass

    for root in roots:
        if should_cancel and should_cancel():
            break
        if root.is_dir():
            _walk(root, 0)
    return results


def _scan_gemma_roots(
    roots: list[Path],
    target_name: str = "",
    max_depth: int = _SCAN_MAX_DEPTH,
    should_cancel=None,
    on_progress=None,
) -> list[str]:
    """Recursively search for Gemma model directories."""
    results: list[str] = []
    seen: set[str] = set()

    def _is_gemma(p: Path) -> bool:
        # Must contain at least one marker file
        for marker in _GEMMA_MARKERS:
            if (p / marker).exists():
                return True
        return False

    def _walk(p: Path, depth: int):
        if should_cancel and should_cancel():
            return
        if depth > max_depth:
            return
        if on_progress:
            on_progress(p)
        try:
            for item in p.iterdir():
                if should_cancel and should_cancel():
                    return
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    if item.name.lower() in _SCAN_SKIP_DIRS:
                        continue
                    if target_name and _same_leaf_name(item.name, target_name):
                        _add_scan_result(results, seen, item)
                    low = item.name.lower()
                    if not target_name and any(h in low for h in _GEMMA_NAME_HINTS) and _is_gemma(item):
                        _add_scan_result(results, seen, item)
                    else:
                        _walk(item, depth + 1)
        except (PermissionError, OSError):
            pass

    for root in roots:
        if should_cancel and should_cancel():
            break
        if root.is_dir():
            _walk(root, 0)
    return results


def _gather_scan_roots() -> list[Path]:
    """Collect common directories and mounted drives to scan for model files."""
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

    if os.name == "nt":
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            if drive.is_dir():
                roots.append(drive)
    else:
        roots.append(Path("/"))

    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)
    return unique_roots


@router.get("/scan-checkpoints")
async def scan_checkpoints(
    type: str = Query(..., description="Type: 'ltx2', 'gemma', or 'gemma_safetensors'"),
    extra_paths: str = Query("", description="Comma-separated extra directories to scan"),
    target_path: str = Query("", description="Target path/name to match exactly by basename"),
):
    """Scan common locations for model checkpoints."""
    roots = _scan_roots(extra_paths)
    target_name = _target_leaf_name(target_path)
    cached = _get_cached_scan_results(type, target_name)
    if cached:
        return {"results": cached}
    if type == "ltx2":
        results = _scan_ltx_checkpoints(roots, target_name=target_name)
    elif type == "gemma":
        results = _scan_gemma_roots(roots, target_name=target_name)
    elif type == "gemma_safetensors":
        results = _scan_exact_files(roots, target_name, _SAFETENSORS_SUFFIXES)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown type: {type}")
    _store_scan_results(type, target_name, results)
    return {"results": results}


def _run_scan_job(job: ScanJob, extra_paths: str):
    with job.lock:
        job.state = "running"
    try:
        roots = _scan_roots(extra_paths)
        _set_scan_progress(job, roots[0] if roots else Path.cwd(), force=True)

        def should_cancel() -> bool:
            with job.lock:
                return job.cancel_requested

        def on_progress(path: Path):
            _set_scan_progress(job, path)

        related_specs = getattr(job, "related_specs", None)
        if related_specs:
            grouped_results = _scan_exact_targets(roots, related_specs, should_cancel=should_cancel, on_progress=on_progress)
            results = grouped_results.get(_scan_cache_key(job.scan_type, job.target_name), [])
            if not should_cancel():
                for key, paths in grouped_results.items():
                    scan_type, target_name = related_specs[key]
                    _store_scan_results(scan_type, target_name, paths)
        elif job.scan_type == "ltx2":
            results = _scan_ltx_checkpoints(roots, target_name=job.target_name, should_cancel=should_cancel, on_progress=on_progress)
            if not should_cancel():
                _store_scan_results(job.scan_type, job.target_name, results)
        elif job.scan_type == "gemma":
            results = _scan_gemma_roots(roots, target_name=job.target_name, should_cancel=should_cancel, on_progress=on_progress)
            if not should_cancel():
                _store_scan_results(job.scan_type, job.target_name, results)
        elif job.scan_type == "gemma_safetensors":
            results = _scan_exact_files(roots, job.target_name, _SAFETENSORS_SUFFIXES, should_cancel=should_cancel, on_progress=on_progress)
            if not should_cancel():
                _store_scan_results(job.scan_type, job.target_name, results)
        else:
            raise ValueError(f"Unknown type: {job.scan_type}")

        with job.lock:
            job.results = results
            job.finished_at = time.time()
            job.state = "cancelled" if job.cancel_requested else "completed"
    except Exception as e:
        logger.warning("Checkpoint scan failed: %s", e)
        with job.lock:
            job.error = str(e)
            job.finished_at = time.time()
            job.state = "failed"


@router.post("/scan-checkpoints/start")
async def start_scan_checkpoints(req: ScanCheckpointRequest):
    """Start a cancellable checkpoint scan job."""
    if req.type not in _SCAN_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown type: {req.type}")

    target_name = _target_leaf_name(req.target_path)
    cached = _get_cached_scan_results(req.type, target_name)
    if cached:
        return _completed_scan_payload(req.type, target_name, cached)

    job = ScanJob(job_id=str(uuid.uuid4()), scan_type=req.type, target_name=target_name)
    job.related_specs = _scan_target_specs(req.type, req.target_path, req.related_targets)
    with _SCAN_JOBS_LOCK:
        _SCAN_JOBS[job.job_id] = job

    thread = threading.Thread(target=_run_scan_job, args=(job, req.extra_paths), daemon=True)
    thread.start()
    return _scan_job_payload(job)


@router.get("/scan-checkpoints/{job_id}")
async def get_scan_checkpoint_status(job_id: str):
    with _SCAN_JOBS_LOCK:
        job = _SCAN_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Scan job not found")
    return _scan_job_payload(job)


@router.post("/scan-checkpoints/{job_id}/cancel")
async def cancel_scan_checkpoint(job_id: str):
    with _SCAN_JOBS_LOCK:
        job = _SCAN_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Scan job not found")
    with job.lock:
        if job.state in {"queued", "running"}:
            job.cancel_requested = True
    return _scan_job_payload(job)


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
