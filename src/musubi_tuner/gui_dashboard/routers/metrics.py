"""Metrics data API router — extracted from server.py for the management app."""

from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

router = APIRouter(tags=["metrics"])


def create_metrics_router(run_dir: str) -> APIRouter:
    """Create a metrics router bound to a specific run directory."""
    r = APIRouter(tags=["metrics"])
    dashboard_dir = os.path.join(run_dir, "dashboard")
    sample_dir = os.path.join(run_dir, "sample")

    @r.get("/data/metrics.parquet")
    async def get_metrics():
        path = os.path.join(dashboard_dir, "metrics.parquet")
        if not os.path.exists(path):
            return Response(status_code=204)
        return FileResponse(path, media_type="application/octet-stream")

    @r.get("/data/status.json")
    async def get_status():
        path = os.path.join(dashboard_dir, "status.json")
        if not os.path.exists(path):
            return Response(status_code=204)
        return FileResponse(path, media_type="application/json")

    @r.get("/data/events.json")
    async def get_events():
        path = os.path.join(dashboard_dir, "events.json")
        if not os.path.exists(path):
            return Response(status_code=204)
        return FileResponse(path, media_type="application/json")

    @r.get("/sse")
    async def sse_stream():
        metrics_path = os.path.join(dashboard_dir, "metrics.parquet")

        async def event_generator():
            last_mtime = 0.0
            while True:
                await asyncio.sleep(2)
                try:
                    mtime = os.path.getmtime(metrics_path) if os.path.exists(metrics_path) else 0.0
                except OSError:
                    mtime = 0.0
                if mtime > last_mtime:
                    last_mtime = mtime
                    yield {"event": "update", "data": f'{{"mtime": {mtime}}}'}

        return EventSourceResponse(event_generator())

    return r


def mount_samples_dir(app, run_dir: str):
    """Mount the samples static file directory."""
    sample_dir = os.path.join(run_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)
    app.mount("/data/samples", StaticFiles(directory=sample_dir), name="samples")
