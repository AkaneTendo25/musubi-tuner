"""Entry point for `python -m musubi_tuner.gui_dashboard`."""

import argparse
import logging
import os
import signal
import subprocess
import sys

import uvicorn

from musubi_tuner.gui_dashboard.management_server import create_management_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")


def main():
    parser = argparse.ArgumentParser(description="LTX-2 Training Manager")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--project", type=str, default=None, help="Path to project.json to load on startup")
    parser.add_argument("--dev", action="store_true", help="Start Vite dev server alongside the API backend")
    parser.add_argument("--dev-port", type=int, default=5173, help="Vite dev server port (default: 5173)")
    args = parser.parse_args()

    vite_proc = None
    if args.dev:
        vite_proc = _start_vite(args.dev_port)

    dev_frontend_url = f"http://127.0.0.1:{args.dev_port}" if args.dev else None
    app = create_management_app(project_path=args.project, dev_frontend_url=dev_frontend_url)

    url = f"http://127.0.0.1:{args.dev_port}" if args.dev else f"http://{args.host}:{args.port}"
    logger.info(f"Starting LTX-2 Training Manager — open {url}")

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if vite_proc is not None:
            _stop_vite(vite_proc)


def _start_vite(port: int) -> subprocess.Popen:
    """Spawn `npm run dev` in the frontend directory."""
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    logger.info(f"Starting Vite dev server on port {port} ...")
    proc = subprocess.Popen(
        [npm_cmd, "run", "dev", "--", "--host", "127.0.0.1", "--port", str(port)],
        cwd=FRONTEND_DIR,
        # Let Vite output go to the same console
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return proc


def _stop_vite(proc: subprocess.Popen):
    """Terminate the Vite dev server."""
    if proc.poll() is not None:
        return
    logger.info("Stopping Vite dev server ...")
    if sys.platform == "win32":
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


if __name__ == "__main__":
    main()
