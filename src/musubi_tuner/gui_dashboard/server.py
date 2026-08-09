import asyncio
import logging
import mimetypes
import os
import re
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

    # Matches save_path produced by ltx2_sampling.py:
    #   {output_name?}_{numsuffix}_{prompt_idx:02d}_{ts(14)}{_seed?}.{ext}
    # where numsuffix is "e######" (epoch) or "######" (step).
    _SAMPLE_RE = re.compile(r"^(?P<prefix>.+?_)?(?P<step_kind>e?\d+)_(?P<prompt>\d{2})_(?P<ts>\d{14})(?:_(?P<seed>\d+))?$")
    # Per-prompt IC-LoRA / I2V source files copied by ltx2_sampling._publish_sample_sources.
    _SOURCE_RE = re.compile(r"^_source_p(?P<prompt>\d{2})_(?P<kind>i2v|v2v|refaudio)$")
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
    _VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
    _AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    @app.get("/data/samples-index")
    async def get_samples_index():
        if not os.path.isdir(sample_dir):
            return {"samples": []}
        groups: dict[tuple, dict] = {}
        sources_by_prompt: dict[int, dict] = {}
        try:
            entries = os.listdir(sample_dir)
        except OSError:
            return {"samples": []}
        for name in entries:
            full = os.path.join(sample_dir, name)
            if not os.path.isfile(full):
                continue
            stem, ext = os.path.splitext(name)
            ext = ext.lower()
            src_match = _SOURCE_RE.match(stem)
            if src_match:
                prompt_idx = int(src_match.group("prompt"))
                src_kind = src_match.group("kind")
                if ext in _IMAGE_EXTS:
                    media = "image"
                elif ext in _VIDEO_EXTS:
                    media = "video"
                elif ext in _AUDIO_EXTS:
                    media = "audio"
                else:
                    continue
                sources_by_prompt.setdefault(prompt_idx, {})[src_kind] = {
                    "url": f"/data/samples/{name}",
                    "media": media,
                }
                continue
            if ext not in (".mp4", ".wav", ".png", ".jpg", ".jpeg", ".webp"):
                continue
            kind = ext.lstrip(".")
            if ext == ".mp4" and stem.endswith("_av"):
                stem = stem[:-3]
                kind = "av"
            m = _SAMPLE_RE.match(stem)
            if not m:
                continue
            step_kind = m.group("step_kind")
            try:
                step = int(step_kind.lstrip("e"))
            except ValueError:
                continue
            prompt_idx = int(m.group("prompt"))
            ts = m.group("ts")
            key = (step, prompt_idx, ts)
            entry = groups.setdefault(
                key,
                {
                    "step": step,
                    "prompt_idx": prompt_idx,
                    "is_epoch": step_kind.startswith("e"),
                    "ts": ts,
                    "files": {},
                },
            )
            entry["files"][kind] = f"/data/samples/{name}"
        items = []
        for entry in groups.values():
            files = entry["files"]
            video_url = files.get("av") or files.get("mp4")
            image_url = files.get("png") or files.get("jpg") or files.get("jpeg") or files.get("webp")
            audio_url = files.get("wav")
            if video_url is not None:
                kind = "video"
                url = video_url
                has_audio = "av" in files or audio_url is not None
            elif image_url is not None:
                kind = "image"
                url = image_url
                has_audio = audio_url is not None
            elif audio_url is not None:
                kind = "audio"
                url = audio_url
                has_audio = True
            else:
                continue
            items.append(
                {
                    "step": entry["step"],
                    "prompt_idx": entry["prompt_idx"],
                    "is_epoch": entry["is_epoch"],
                    "ts": entry["ts"],
                    "kind": kind,
                    "url": url,
                    "audio_url": audio_url,
                    "has_audio": has_audio,
                    "sources": sources_by_prompt.get(entry["prompt_idx"], {}),
                }
            )
        items.sort(key=lambda i: (-i["step"], i["prompt_idx"]))
        return {"samples": items}

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
