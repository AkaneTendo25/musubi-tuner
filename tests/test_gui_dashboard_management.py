import asyncio
import json
import os
import sys
import time
from pathlib import Path

from fastapi.responses import FileResponse

from musubi_tuner.gui_dashboard import management_server
from musubi_tuner.gui_dashboard.process_manager import ProcessManager
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.gui_dashboard.training_dashboard_state import training_is_resume
from musubi_tuner.utils import train_utils


def _config(tmp_path: Path) -> ProjectConfig:
    config = ProjectConfig(project_dir=str(tmp_path))
    config.training.output_dir = "training-output"
    return config


def _endpoint(app, path: str):
    return next(route.endpoint for route in app.routes if getattr(route, "path", None) == path)


def test_management_server_serves_persisted_dashboard_after_process_exit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    dashboard_dir = tmp_path / "training-output" / "dashboard"
    dashboard_dir.mkdir(parents=True)
    status = {"status": "finished", "step": 12}
    (dashboard_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")

    app = management_server.create_management_app()
    app.state.project_config = config

    response = asyncio.run(_endpoint(app, "/data/status.json")())

    assert response.status_code == 200
    assert json.loads(response.body) == status


def test_management_server_rejects_sample_path_traversal(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "training-output"
    sample_dir = run_dir / "sample"
    sample_dir.mkdir(parents=True)
    (sample_dir / "preview.png").write_bytes(b"preview")
    (run_dir / "secret.txt").write_text("secret", encoding="utf-8")

    app = management_server.create_management_app()
    app.state.project_config = config

    get_sample = _endpoint(app, "/data/samples/{file_path:path}")
    preview = asyncio.run(get_sample("preview.png"))
    response = asyncio.run(get_sample("../secret.txt"))

    assert isinstance(preview, FileResponse)
    assert Path(preview.path) == sample_dir / "preview.png"
    assert response.status_code == 404
    assert response.body != b"secret"


def test_dev_mode_redirects_without_a_prebuilt_frontend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(management_server, "FRONTEND_DIST", str(tmp_path / "missing-dist"))
    app = management_server.create_management_app(dev_frontend_url="http://127.0.0.1:5173")

    response = asyncio.run(_endpoint(app, "/{full_path:path}")("training"))

    assert response.status_code == 307
    assert response.headers["location"] == "http://127.0.0.1:5173/training"


def test_training_process_stop_uses_graceful_stop_file(tmp_path: Path) -> None:
    manager = ProcessManager()
    script = (
        "import os,time; "
        "path=os.environ['MUSUBI_DASHBOARD_STOP_FILE']; "
        "deadline=time.time()+10; "
        "\nwhile not os.path.isfile(path) and time.time()<deadline: time.sleep(0.01); "
        "\nraise SystemExit(0 if os.path.isfile(path) else 2)"
    )
    manager.start("training", [sys.executable, "-c", script], cwd=str(tmp_path))

    manager.stop("training")
    deadline = time.time() + 5
    while manager.get_status("training")["state"] == "stopping" and time.time() < deadline:
        time.sleep(0.01)

    status = manager.get_status("training")
    assert status["state"] == "finished"
    assert status["exit_code"] == 0
    assert status["stop_requested"] is True


def test_process_uses_absolute_checkout_src_before_inherited_pythonpath(tmp_path: Path, monkeypatch) -> None:
    manager = ProcessManager()
    output = tmp_path / "pythonpath.txt"
    monkeypatch.setenv("PYTHONPATH", "src")
    script = f"import os,pathlib; pathlib.Path({str(output)!r}).write_text(os.environ['PYTHONPATH'])"

    manager.start("inference", [sys.executable, "-c", script], cwd=str(tmp_path))
    deadline = time.time() + 5
    while manager.get_status("inference")["state"] == "running" and time.time() < deadline:
        time.sleep(0.01)

    pythonpath = output.read_text(encoding="utf-8").split(os.pathsep)
    expected_src = Path(__file__).resolve().parents[1] / "src"
    assert Path(pythonpath[0]) == expected_src
    assert pythonpath[1] == "src"


def test_process_does_not_inherit_empty_cuda_visible_devices(tmp_path: Path, monkeypatch) -> None:
    manager = ProcessManager()
    output = tmp_path / "cuda_visible_devices.txt"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    script = f"import os,pathlib; pathlib.Path({str(output)!r}).write_text(os.environ.get('CUDA_VISIBLE_DEVICES', 'missing'))"

    manager.start("inference", [sys.executable, "-c", script], cwd=str(tmp_path))
    deadline = time.time() + 5
    while manager.get_status("inference")["state"] == "running" and time.time() < deadline:
        time.sleep(0.01)

    assert output.read_text(encoding="utf-8") == "missing"


def test_dashboard_autoresume_recognizes_manifest_state(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.training.autoresume = True
    state_dir = tmp_path / "training-output" / "model-step00000012-state"
    state_dir.mkdir(parents=True)
    for filename in ("model.safetensors", "optimizer.bin", "random_states_0.pkl"):
        (state_dir / filename).write_bytes(b"state")
    train_utils.save_resume_metadata(str(state_dir), global_step=12, step_in_epoch=4, epoch=2, data_seed=42)
    train_utils.save_state_manifest(
        str(state_dir),
        state_type="step",
        global_step=12,
        step_in_epoch=4,
        epoch=2,
    )

    assert training_is_resume(config) is True
