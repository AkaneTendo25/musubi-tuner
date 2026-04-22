"""System information API — hardware stats for the dashboard."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import locale
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request

from musubi_tuner.gui_dashboard.management_tools import get_management_status, launch_setup_tool, open_repo_in_file_browser

router = APIRouter(prefix="/api/system", tags=["system"])


def _decode_subprocess_output(data: bytes) -> str:
    for encoding in ("utf-8", locale.getpreferredencoding(False), "cp1251", "cp866"):
        if not encoding:
            continue
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


@router.get("/info")
async def get_system_info():
    return {
        "cpu": _get_cpu_info(),
        "ram": _get_ram_info(),
        "gpus": _get_gpu_info(),
        "disk": _get_disk_info(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
    }


@router.get("/management-status")
async def get_management_status_route(request: Request):
    try:
        return get_management_status(project_config=getattr(request.app.state, "project_config", None))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load management status: {exc}") from exc


@router.post("/management/open-setup")
async def open_setup_tool(
    request: Request,
    branch: Literal["ltx-2", "ltx-2-dev"] | None = Query(default=None),
):
    process_manager = getattr(request.app.state, "process_manager", None)
    if process_manager:
        statuses = process_manager.get_all_statuses()
        active = [name for name, status in statuses.items() if status.get("state") in {"running", "stopping"}]
        if active:
            raise HTTPException(status_code=409, detail=f"Cannot open Setup / Update while processes are active: {', '.join(active)}")
    try:
        return launch_setup_tool(branch_override=branch)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to open Setup / Update: {exc}") from exc


@router.post("/management/open-repo")
async def open_repo_folder():
    try:
        return open_repo_in_file_browser()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to open repository folder: {exc}") from exc


def _get_cpu_info() -> dict:
    return {
        "model": platform.processor() or platform.machine() or "Unknown",
        "cores": os.cpu_count() or 0,
    }


def _get_ram_info() -> dict | None:
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 1),
            "used_gb": round(mem.used / (1024**3), 1),
            "available_gb": round(mem.available / (1024**3), 1),
            "percent": mem.percent,
        }
    except ImportError:
        pass

    # Fallback: parse /proc/meminfo on Linux
    try:
        if os.path.exists("/proc/meminfo"):
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val_kb = int(parts[1].strip().split()[0])
                        info[key] = val_kb
            total = info.get("MemTotal", 0) / (1024 * 1024)
            available = info.get("MemAvailable", 0) / (1024 * 1024)
            return {
                "total_gb": round(total, 1),
                "used_gb": round(total - available, 1),
                "available_gb": round(available, 1),
                "percent": round((1 - available / total) * 100, 1) if total > 0 else 0,
            }
    except Exception:
        pass

    return None


def _get_gpu_info() -> list[dict]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,fan.speed,utilization.gpu,power.draw,clocks.current.graphics",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            stdout = _decode_subprocess_output(result.stdout or b"")
            gpus = []
            for line in stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 9:
                    gpus.append(
                        {
                            "name": parts[0],
                            "vram_total_mb": int(parts[1]),
                            "vram_used_mb": int(parts[2]),
                            "vram_free_mb": int(parts[3]),
                            "temperature": int(parts[4]) if parts[4] not in ("N/A", "[N/A]") else None,
                            "fan_speed_percent": int(parts[5]) if parts[5] not in ("N/A", "[N/A]") else None,
                            "utilization": int(parts[6]) if parts[6] not in ("N/A", "[N/A]") else None,
                            "power_draw_w": float(parts[7]) if parts[7] not in ("N/A", "[N/A]") else None,
                            "graphics_clock_mhz": int(float(parts[8])) if parts[8] not in ("N/A", "[N/A]") else None,
                        }
                    )
            return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return []


def _get_disk_info() -> dict | None:
    try:
        usage = shutil.disk_usage(os.getcwd())
        return {
            "total_gb": round(usage.total / (1024**3), 1),
            "used_gb": round(usage.used / (1024**3), 1),
            "free_gb": round(usage.free / (1024**3), 1),
        }
    except Exception:
        return None
