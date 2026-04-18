"""Standalone management FastAPI app for the full training dashboard."""

from __future__ import annotations

import logging
import mimetypes
import os
from pathlib import Path
from typing import Optional

# Windows registry often maps .js to text/plain — fix it
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, Response

from musubi_tuner.gui_dashboard.process_manager import ProcessManager
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.gui_dashboard.routers import datasets, filesystem, processes, projects, stats, system

logger = logging.getLogger(__name__)

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")


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
        run_dir = _get_run_dir(app)
        if not run_dir:
            return Response(status_code=204)
        path = os.path.join(run_dir, "dashboard", "metrics.parquet")
        if not os.path.exists(path):
            return Response(status_code=204)
        return FileResponse(path, media_type="application/octet-stream")

    @app.get("/data/status.json")
    async def get_status():
        run_dir = _get_run_dir(app)
        if not run_dir:
            return Response(status_code=204)
        path = os.path.join(run_dir, "dashboard", "status.json")
        if not os.path.exists(path):
            return Response(status_code=204)
        return FileResponse(path, media_type="application/json")

    @app.get("/data/events.json")
    async def get_events():
        run_dir = _get_run_dir(app)
        if not run_dir:
            return Response(status_code=204)
        path = os.path.join(run_dir, "dashboard", "events.json")
        if not os.path.exists(path):
            return Response(status_code=204)
        return FileResponse(path, media_type="application/json")

    @app.api_route("/data/samples/{file_path:path}", methods=["GET", "HEAD"])
    async def get_sample(file_path: str):
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
                run_dir = _get_run_dir(app)
                if not run_dir:
                    continue
                metrics_path = os.path.join(run_dir, "dashboard", "metrics.parquet")
                try:
                    mtime = os.path.getmtime(metrics_path) if os.path.exists(metrics_path) else 0.0
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
            file_path = os.path.join(FRONTEND_DIST, full_path)
            if full_path and os.path.isfile(file_path):
                return FileResponse(file_path)
            return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
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
        return config.training.output_dir
    return None
