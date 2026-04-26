from __future__ import annotations

import socket
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn

from musubi_tuner.gui_dashboard.management_server import create_management_app
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig


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


def test_sample_route_supports_head(tmp_path: Path):
	run_dir = tmp_path / "run"
	project_dir = tmp_path / "project"
	sample_path = run_dir / "sample" / "000002.png"
	sample_path.parent.mkdir(parents=True, exist_ok=True)
	project_dir.mkdir(parents=True, exist_ok=True)
	sample_path.write_bytes(b"fake-image")

	config = ProjectConfig(name="sample-head", project_dir=str(project_dir))
	config.training.output_dir = str(run_dir)
	project_json = project_dir / "project.json"
	config.save(project_json)

	app = create_management_app(project_path=str(project_json))
	port = _free_port()
	server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
	thread = threading.Thread(target=server.run, daemon=True)
	thread.start()

	try:
		_wait_for_server(f"http://127.0.0.1:{port}/api/project")

		head_request = urllib.request.Request(
			f"http://127.0.0.1:{port}/data/samples/000002.png",
			method="HEAD",
		)
		with urllib.request.urlopen(head_request, timeout=5) as response:
			assert response.status == 200

		with urllib.request.urlopen(f"http://127.0.0.1:{port}/data/samples/000002.png", timeout=5) as response:
			assert response.status == 200
			assert response.read() == b"fake-image"
	finally:
		server.should_exit = True
		thread.join(timeout=5)
