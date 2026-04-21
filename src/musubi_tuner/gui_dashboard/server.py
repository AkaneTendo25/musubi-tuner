import asyncio
import logging
import mimetypes
import os
import threading

# Windows registry often maps .js to text/plain — fix it
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")
NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def create_app(run_dir: str) -> FastAPI:
    app = FastAPI(title="Training Dashboard")
    dashboard_dir = os.path.join(run_dir, "dashboard")
    sample_dir = os.path.join(run_dir, "sample")

    # -- data endpoints --

    @app.get("/data/metrics.parquet")
    async def get_metrics():
        path = os.path.join(dashboard_dir, "metrics.parquet")
        if not os.path.exists(path):
            return Response(status_code=204)
        return FileResponse(path, media_type="application/octet-stream")

    @app.get("/data/status.json")
    async def get_status():
        path = os.path.join(dashboard_dir, "status.json")
        if not os.path.exists(path):
            return Response(status_code=204)
        return FileResponse(path, media_type="application/json")

    @app.get("/data/events.json")
    async def get_events():
        path = os.path.join(dashboard_dir, "events.json")
        if not os.path.exists(path):
            return Response(status_code=204)
        return FileResponse(path, media_type="application/json")

    # -- SSE --

    @app.get("/sse")
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

    # -- samples static files --

    if os.path.isdir(sample_dir):
        app.mount("/data/samples", StaticFiles(directory=sample_dir), name="samples")
    else:
        # Create the dir so the mount doesn't fail; samples will appear later
        os.makedirs(sample_dir, exist_ok=True)
        app.mount("/data/samples", StaticFiles(directory=sample_dir), name="samples")

    # -- SvelteKit frontend (must be last - catches all other routes) --

    if os.path.isdir(FRONTEND_DIST):
        # Serve index.html for SPA routing
        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            file_path = os.path.join(FRONTEND_DIST, full_path)
            if full_path and os.path.isfile(file_path):
                return FileResponse(file_path, headers=NO_CACHE_HEADERS)
            return FileResponse(os.path.join(FRONTEND_DIST, "index.html"), headers=NO_CACHE_HEADERS)
    else:
        @app.get("/")
        async def no_frontend():
            return HTMLResponse(
                "<h2>Training Dashboard</h2>"
                "<p>Frontend not built. Run <code>npm run build</code> in <code>gui_dashboard/frontend/</code></p>"
                "<p>API available at <code>/data/metrics.parquet</code>, <code>/data/status.json</code>, <code>/data/events.json</code></p>"
            )

    return app


def start_server(run_dir: str, host: str = "0.0.0.0", port: int = 7860):
    import uvicorn

    app = create_app(run_dir)

    def _run():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    t = threading.Thread(target=_run, daemon=True, name="dashboard-server")
    t.start()
    logger.info(f"Training dashboard started at http://{host}:{port}")
    return t
