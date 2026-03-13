"""System information API — hardware stats for the dashboard."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

from fastapi import APIRouter

router = APIRouter(prefix="/api/system", tags=["system"])


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
                "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpus.append(
                        {
                            "name": parts[0],
                            "vram_total_mb": int(parts[1]),
                            "vram_used_mb": int(parts[2]),
                            "vram_free_mb": int(parts[3]),
                            "temperature": int(parts[4]) if parts[4] not in ("N/A", "[N/A]") else None,
                            "utilization": int(parts[5]) if parts[5] not in ("N/A", "[N/A]") else None,
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
