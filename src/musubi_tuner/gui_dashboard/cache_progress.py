"""Lightweight JSON status writer for dashboard-launched cache jobs."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional


class CacheProgressWriter:
    def __init__(self, path: str, process_type: str):
        self.path = Path(path)
        self.process_type = process_type
        self._lock = threading.Lock()

    def update(
        self,
        *,
        dataset_index: int,
        current_items: int,
        total_items: Optional[int],
        status: str = "running",
    ) -> None:
        payload = {
            "process_type": self.process_type,
            "dataset_index": dataset_index,
            "current_items": int(current_items),
            "total_items": int(total_items) if total_items is not None and total_items > 0 else None,
            "status": status,
            "time": time.time(),
        }
        self._write_json(payload)

    def _write_json(self, payload: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f"{self.path.name}.{threading.get_ident()}.tmp")
            with self._lock:
                tmp.write_text(json.dumps(payload), encoding="utf-8")
                os.replace(tmp, self.path)
        except OSError:
            pass


def create_cache_progress_writer_from_env() -> CacheProgressWriter | None:
    path = os.environ.get("MUSUBI_DASHBOARD_CACHE_STATUS_FILE")
    if not path:
        return None
    process_type = os.environ.get("MUSUBI_DASHBOARD_PROCESS_TYPE") or "cache"
    return CacheProgressWriter(path, process_type)


def cache_progress_total(dataset) -> Optional[int]:
    if getattr(dataset, "frame_extraction", None) is not None:
        return None
    datasource = getattr(dataset, "datasource", None)
    if datasource is None:
        return None
    try:
        if hasattr(datasource, "is_indexable") and not datasource.is_indexable():
            return None
        return len(datasource)
    except Exception:
        return None


def should_update_cache_progress(current: int, total: Optional[int], last_updated: int) -> bool:
    if current <= last_updated:
        return False
    if last_updated == 0:
        return True
    if total is not None and total > 0:
        return current >= total or current - last_updated >= max(1, total // 200)
    return current - last_updated >= 10
