"""Filesystem browsing API router."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
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
# Model download (huggingface-cli)
# ---------------------------------------------------------------------------

# Predefined download sources
_DOWNLOAD_PRESETS: dict[str, dict] = {
    "ltxav": {
        "label": "LTXAV 19B (audio+video)",
        "repo": "Lightricks/LTX-2",
        "filename": "ltx-2-19b-dev.safetensors",
    },
    "gemma-unsloth": {
        "label": "Gemma 3 12B IT (unsloth)",
        "repo": "unsloth/gemma-3-12b-it",
    },
}


@router.get("/download-presets")
async def get_download_presets():
    """Return available model download presets."""
    return {"presets": {k: {"label": v["label"]} for k, v in _DOWNLOAD_PRESETS.items()}}


class DownloadRequest(BaseModel):
    preset: str
    dest_dir: str


@router.post("/download-model")
async def download_model(req: DownloadRequest):
    """Download a model from HuggingFace using huggingface-cli."""
    if req.preset not in _DOWNLOAD_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {req.preset}")

    preset = _DOWNLOAD_PRESETS[req.preset]
    dest = Path(req.dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    repo = preset["repo"]
    filename = preset.get("filename")

    # Build huggingface-cli download command
    cmd = [sys.executable, "-m", "huggingface_hub", "download", repo]
    if filename:
        cmd.append(filename)
    cmd += ["--local-dir", str(dest)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "Download failed"
            raise HTTPException(status_code=500, detail=detail)

        # Find the downloaded file path
        if filename:
            downloaded = dest / filename
            return {"ok": True, "path": str(downloaded)}
        else:
            # For full repo downloads (gemma), return the directory
            return {"ok": True, "path": str(dest)}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Download timed out (10 min limit)")
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="huggingface_hub not installed. Run: pip install huggingface_hub",
        )
