from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request

import uvicorn

from musubi_tuner.gui_dashboard.management_server import create_management_app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(url: str, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except Exception as exc:  # pragma: no cover - startup polling
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"Server did not start: {last_error}")


def test_unknown_api_route_returns_json_404_instead_of_frontend_html():
    app = create_management_app()
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    try:
        _wait_for_server(f"http://127.0.0.1:{port}/api/project")

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/settings", timeout=5) as response:
            assert response.status == 200
            assert "text/html" in response.headers.get("Content-Type", "").lower()

        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/does-not-exist", timeout=5)
            raise AssertionError("Expected HTTPError for unknown API route")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            payload = json.loads(body)
            assert exc.code == 404
            assert "application/json" in exc.headers.get("Content-Type", "").lower()
            assert payload["detail"] == "API route not found: /api/does-not-exist"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
