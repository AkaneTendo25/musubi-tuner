"""Standalone management FastAPI app for the full training dashboard."""

from __future__ import annotations

import logging
import math
import mimetypes
import os
from pathlib import Path
from typing import Optional

# Windows registry often maps .js to text/plain — fix it
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
import pyarrow.parquet as pq

from musubi_tuner.gui_dashboard.process_manager import ProcessManager
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.gui_dashboard.routers import datasets, filesystem, processes, projects, stats, system

logger = logging.getLogger(__name__)

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")
NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def create_management_app(project_path: Optional[str] = None) -> FastAPI:
    """Create the management FastAPI application."""
    app = FastAPI(title="LTX-2 Training Manager")

    # App state
    app.state.process_manager = ProcessManager()
    app.state.project_config = None
    app.state.project_path = None  # Path to the actual .json file

    # Load project if path provided
    if project_path:
        p = Path(project_path)
        if p.is_dir():
            p = p / "project.json"
        if p.exists():
            try:
                app.state.project_config = ProjectConfig.load(p)
                app.state.project_path = p
                logger.info(f"Loaded project: {p}")
            except Exception as e:
                logger.warning(f"Failed to load project {p}: {e}")

    # Mount API routers
    app.include_router(projects.router)
    app.include_router(datasets.router)
    app.include_router(processes.router)
    app.include_router(filesystem.router)
    app.include_router(system.router)
    app.include_router(stats.router)

    # Metrics router — dynamically bound to training output_dir
    @app.get("/data/metrics.parquet")
    async def get_metrics():
        if not _training_dashboard_active(app):
            return Response(status_code=204)
        run_dir = _get_run_dir(app)
        if not run_dir:
            return Response(status_code=204)
        path = os.path.join(run_dir, "dashboard", "metrics.parquet")
        if not os.path.exists(path):
            return Response(status_code=204)
        return Response(Path(path).read_bytes(), media_type="application/octet-stream", headers=NO_CACHE_HEADERS)

    @app.get("/data/status.json")
    async def get_status():
        if not _training_dashboard_active(app):
            return Response(status_code=204)
        run_dir = _get_run_dir(app)
        if not run_dir:
            return Response(status_code=204)
        path = os.path.join(run_dir, "dashboard", "status.json")
        if not os.path.exists(path):
            return Response(status_code=204)
        return Response(Path(path).read_bytes(), media_type="application/json", headers=NO_CACHE_HEADERS)

    @app.get("/data/events.json")
    async def get_events():
        if not _training_dashboard_active(app):
            return Response(status_code=204)
        run_dir = _get_run_dir(app)
        if not run_dir:
            return Response(status_code=204)
        path = os.path.join(run_dir, "dashboard", "events.json")
        if not os.path.exists(path):
            return Response(status_code=204)
        return Response(Path(path).read_bytes(), media_type="application/json", headers=NO_CACHE_HEADERS)

    @app.get("/api/dashboard/metrics")
    async def get_metrics_json():
        if not _training_dashboard_active(app):
            return Response(status_code=204)
        run_dir = _get_run_dir(app)
        if not run_dir:
            return Response(status_code=204)
        path = os.path.join(run_dir, "dashboard", "metrics.parquet")
        if not os.path.exists(path):
            return Response(status_code=204)
        try:
            table = pq.read_table(path)
            rows = []
            for row in table.to_pylist():
                clean_row = {}
                for key, value in row.items():
                    if isinstance(value, float) and math.isnan(value):
                        clean_row[key] = None
                    else:
                        clean_row[key] = value
                rows.append(clean_row)
            return JSONResponse({"rows": rows}, headers=NO_CACHE_HEADERS)
        except Exception:
            return Response(status_code=204)

    @app.api_route("/data/samples/{file_path:path}", methods=["GET", "HEAD"])
    async def get_sample(file_path: str):
        if not _training_dashboard_active(app):
            return HTMLResponse("no active training run", status_code=404)
        run_dir = _get_run_dir(app)
        if not run_dir:
            return HTMLResponse("no training output configured", status_code=404)
        full_path = os.path.join(run_dir, "sample", file_path)
        if not os.path.exists(full_path):
            return HTMLResponse("not found", status_code=404)
        return FileResponse(full_path)

    # SSE for metrics updates
    import asyncio
    from sse_starlette.sse import EventSourceResponse

    @app.get("/sse")
    async def sse_stream():
        async def event_generator():
            last_mtime = 0.0
            while True:
                await asyncio.sleep(2)
                if not _training_dashboard_active(app):
                    continue
                run_dir = _get_run_dir(app)
                if not run_dir:
                    continue
                metrics_path = os.path.join(run_dir, "dashboard", "metrics.parquet")
                status_path = os.path.join(run_dir, "dashboard", "status.json")
                events_path = os.path.join(run_dir, "dashboard", "events.json")
                try:
                    mtime = max(
                        os.path.getmtime(metrics_path) if os.path.exists(metrics_path) else 0.0,
                        os.path.getmtime(status_path) if os.path.exists(status_path) else 0.0,
                        os.path.getmtime(events_path) if os.path.exists(events_path) else 0.0,
                    )
                except OSError:
                    mtime = 0.0
                if mtime > last_mtime:
                    last_mtime = mtime
                    yield {"event": "update", "data": f'{{"mtime": {mtime}}}'}

        return EventSourceResponse(event_generator())

    # SvelteKit frontend (must be last — catches all routes)
    if os.path.isdir(FRONTEND_DIST):
        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            normalized = (full_path or "").lstrip("/")
            if normalized.startswith("api/"):
                return JSONResponse({"detail": f"API route not found: /{normalized}"}, status_code=404, headers=NO_CACHE_HEADERS)
            if normalized.startswith("data/"):
                return Response(status_code=404, headers=NO_CACHE_HEADERS)
            if normalized == "sse" or normalized.startswith("sse/"):
                return Response(status_code=404, headers=NO_CACHE_HEADERS)

            file_path = os.path.join(FRONTEND_DIST, full_path)
            if full_path and os.path.isfile(file_path):
                return FileResponse(file_path, headers=NO_CACHE_HEADERS)
            return FileResponse(os.path.join(FRONTEND_DIST, "index.html"), headers=NO_CACHE_HEADERS)
    else:
        @app.get("/")
        async def no_frontend():
            return HTMLResponse(
                "<h2>LTX-2 Training Manager</h2>"
                "<p>Frontend not built. Run <code>npm run build</code> in <code>gui_dashboard/frontend/</code></p>"
            )

    return app


def _get_run_dir(app: FastAPI) -> Optional[str]:
    """Get the current training output directory from project config."""
    config: ProjectConfig | None = app.state.project_config
    if config and config.training.output_dir:
        run_dir = Path(config.training.output_dir)
        if not run_dir.is_absolute() and config.project_dir:
            run_dir = Path(config.project_dir) / run_dir
        return str(run_dir)
    return None


def _training_dashboard_active(app: FastAPI) -> bool:
    pm: ProcessManager = app.state.process_manager
    state = pm.get_status("training").get("state")
    return state in {"running", "stopping"}
