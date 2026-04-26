from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess

from musubi_tuner.gui_dashboard import management_tools


def _repo_status_stub(**overrides):
    status = {
        "exists": True,
        "git_available": True,
        "is_git_repo": True,
        "origin_configured": True,
        "head": "0123456789abcdef",
        "head_short": "01234567",
        "branch": "ltx-2-dev",
        "dirty": False,
        "remote_head": "fedcba9876543210",
        "remote_head_short": "fedcba98",
        "local_ahead_count": 0,
        "remote_ahead_count": 0,
        "diverged": False,
        "update_available": False,
        "can_auto_update": False,
        "fetch_attempted": False,
        "fetch_succeeded": False,
        "offline": False,
        "origin_error": "",
        "summary": "Repository is up to date",
    }
    status.update(overrides)
    return status


def test_get_management_status_without_state_recommends_setup(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "src").mkdir()
    (repo_root / "scripts" / "install.ps1").write_text("# installer", encoding="utf-8")

    desktop = tmp_path / "desktop"
    desktop.mkdir()
    monkeypatch.setattr(management_tools, "_desktop_dir", lambda: desktop)

    status = management_tools.get_management_status(repo_root)

    assert status["actions"]["can_launch_setup"] is True
    assert status["install_state_present"] is False
    assert status["install"]["venv_exists"] is False
    assert status["install"]["frontend_dist_exists"] is False
    assert any("Install state has not been recorded" in item for item in status["warnings"])
    assert any("Reinstall Python dependencies" in item for item in status["recommendations"])
    assert any("Recreate desktop shortcuts" in item for item in status["recommendations"])


def test_get_repository_status_uses_recent_cached_remote_state(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()

    def fake_git_run(_repo_root: Path, *args: str, timeout: int = 10):
        class Result:
            def __init__(self, stdout: str):
                self.stdout = stdout
                self.returncode = 0

        if args == ("rev-parse", "HEAD"):
            return Result("0123456789abcdef\n")
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return Result("ltx-2-dev\n")
        if args == ("status", "--porcelain"):
            return Result("")
        raise AssertionError(f"Unexpected git call: {args}")

    monkeypatch.setattr(management_tools, "_git_run", fake_git_run)

    state = {
        "repo": {
            "last_checked_utc": datetime.now(timezone.utc).isoformat(),
            "remote_head": "fedcba9876543210",
            "remote_head_short": "fedcba98",
            "local_ahead_count": 0,
            "remote_ahead_count": 3,
            "diverged": False,
            "update_available": True,
        }
    }

    status = management_tools.get_repository_status(repo_root, "ltx-2-dev", "https://github.com/example/repo.git", state)

    assert status["fetch_attempted"] is False
    assert status["update_available"] is True
    assert status["remote_ahead_count"] == 3
    assert "Update available" in status["summary"]


def test_get_management_status_falls_back_to_git_metadata_when_state_missing(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "src").mkdir()
    (repo_root / ".git").mkdir()
    (repo_root / "scripts" / "install.ps1").write_text("# installer", encoding="utf-8")

    desktop = tmp_path / "desktop"
    desktop.mkdir()
    monkeypatch.setattr(management_tools, "_desktop_dir", lambda: desktop)
    monkeypatch.setattr(management_tools, "_github_reachable", lambda remote_url: False)

    def fake_git_run(_repo_root: Path, *args: str, timeout: int = 10):
        class Result:
            def __init__(self, stdout: str, returncode: int = 0):
                self.stdout = stdout
                self.returncode = returncode

        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return Result("feature/setup-manager\n")
        if args == ("remote", "get-url", "origin"):
            return Result("https://github.com/example/musubi-tuner.git\n")
        if args == ("rev-parse", "HEAD"):
            return Result("0123456789abcdef\n")
        if args == ("status", "--porcelain"):
            return Result("")
        raise AssertionError(f"Unexpected git call: {args}")

    monkeypatch.setattr(management_tools, "_git_run", fake_git_run)

    status = management_tools.get_management_status(repo_root)

    assert status["install_state_present"] is False
    assert status["branch"] == "feature/setup-manager"
    assert status["remote_url"] == "https://github.com/example/musubi-tuner.git"


def test_get_management_status_warns_for_corrupt_state_and_missing_origin(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "src").mkdir()
    (repo_root / ".git").mkdir()
    (repo_root / "scripts" / "install.ps1").write_text("# installer", encoding="utf-8")
    (repo_root / management_tools.INSTALL_STATE_NAME).write_text("{not-json", encoding="utf-8")

    desktop = tmp_path / "desktop"
    desktop.mkdir()
    monkeypatch.setattr(management_tools, "_desktop_dir", lambda: desktop)

    def fake_git_run(_repo_root: Path, *args: str, timeout: int = 10):
        class Result:
            def __init__(self, stdout: str, returncode: int = 0):
                self.stdout = stdout
                self.returncode = returncode

        if args == ("rev-parse", "HEAD"):
            return Result("0123456789abcdef\n")
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return Result("ltx-2-dev\n")
        if args == ("status", "--porcelain"):
            return Result("")
        if args == ("remote", "get-url", "origin"):
            return Result("", 1)
        raise AssertionError(f"Unexpected git call: {args}")

    monkeypatch.setattr(management_tools, "_git_run", fake_git_run)

    status = management_tools.get_management_status(repo_root)

    assert status["repo"]["origin_configured"] is False
    assert any("could not be parsed" in item for item in status["warnings"])
    assert any("No origin remote is configured" in item for item in status["warnings"])
    assert any("records its state" in item for item in status["recommendations"])


def test_get_management_status_without_project_includes_doctor_summary(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "src").mkdir()
    (repo_root / "scripts" / "install.ps1").write_text("# installer", encoding="utf-8")

    desktop = tmp_path / "desktop"
    desktop.mkdir()
    monkeypatch.setattr(management_tools, "_desktop_dir", lambda: desktop)
    monkeypatch.setattr(management_tools, "get_repository_status", lambda *args, **kwargs: _repo_status_stub())
    monkeypatch.setattr(management_tools, "_command_exists", lambda *names: True)
    monkeypatch.setattr(management_tools, "_has_hf_token", lambda: False)

    status = management_tools.get_management_status(repo_root)

    assert status["doctor"]["loaded_project"] is False
    assert "Load or create a project" in status["doctor"]["summary"]
    assert status["doctor"]["environment_checks"]
    assert status["doctor"]["asset_checks"][0]["label"] == "Project"


def test_get_management_status_with_project_reports_missing_assets_in_doctor(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "src").mkdir()
    (repo_root / "scripts" / "install.ps1").write_text("# installer", encoding="utf-8")
    (repo_root / "venv" / "Scripts").mkdir(parents=True)
    (repo_root / "venv" / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    frontend_dist = repo_root / "src" / "musubi_tuner" / "gui_dashboard" / "frontend" / "dist"
    frontend_dist.mkdir(parents=True)
    (frontend_dist / "index.html").write_text("<!doctype html>", encoding="utf-8")

    project_dir = repo_root / "projects" / "demo"
    model_dir = repo_root / "models"
    project_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    desktop = tmp_path / "desktop"
    desktop.mkdir()
    monkeypatch.setattr(management_tools, "_desktop_dir", lambda: desktop)
    monkeypatch.setattr(management_tools, "get_repository_status", lambda *args, **kwargs: _repo_status_stub())
    monkeypatch.setattr(management_tools, "_command_exists", lambda *names: True)
    monkeypatch.setattr(management_tools, "_has_hf_token", lambda: True)
    monkeypatch.setattr(
        management_tools,
        "validate_process_config",
        lambda key, config: {"ok": True, "errors": [], "warnings": [], "summary": "Validation passed."},
    )

    config = management_tools.ProjectConfig.model_validate(
        {
            "name": "Demo",
            "project_dir": str(project_dir),
            "model_dir": str(model_dir),
            "dataset": {
                "datasets": [
                    {
                        "type": "video",
                        "directory": "missing_dataset",
                    }
                ]
            },
            "caching": {
                "ltx2_checkpoint": "missing-ltx.safetensors",
                "gemma_root": "missing-gemma",
            },
            "training": {
                "ltx2_checkpoint": "missing-ltx.safetensors",
                "gemma_root": "missing-gemma",
            },
            "inference": {
                "ltx2_checkpoint": "missing-ltx.safetensors",
                "gemma_root": "missing-gemma",
            },
        }
    )

    status = management_tools.get_management_status(repo_root, project_config=config)
    assets = {item["key"]: item for item in status["doctor"]["asset_checks"]}
    processes = {item["key"]: item for item in status["doctor"]["processes"]}

    assert status["doctor"]["loaded_project"] is True
    assert assets["datasets"]["status"] == "error"
    assert assets["cache_latents_checkpoint"]["status"] == "error"
    assert assets["cache_text_gemma"]["status"] == "error"
    assert processes["training"]["status"] == "error"
    assert processes["inference"]["status"] == "error"


def test_launch_setup_tool_uses_branch_override_with_launcher(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    launcher = repo_root / management_tools.SETUP_LAUNCHER_NAME
    launcher.write_text("@echo off\n", encoding="utf-8")

    called: dict[str, object] = {}

    def fake_popen(cmd, cwd=None, creationflags=0):
        called["cmd"] = cmd
        called["cwd"] = cwd
        called["creationflags"] = creationflags

        class DummyProcess:
            pass

        return DummyProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = management_tools.launch_setup_tool(repo_root, branch_override="ltx-2")

    assert result["mode"] == "launcher"
    assert result["branch"] == "ltx-2"
    assert called["cmd"] == ["cmd", "/c", str(launcher), "-Branch", "ltx-2"]
    assert called["cwd"] == str(repo_root)


def test_launch_setup_tool_direct_mode_uses_git_remote_fallback(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "scripts" / "install.ps1").write_text("# installer", encoding="utf-8")

    called: dict[str, object] = {}

    def fake_popen(cmd, cwd=None, creationflags=0):
        called["cmd"] = cmd
        called["cwd"] = cwd
        called["creationflags"] = creationflags

        class DummyProcess:
            pass

        return DummyProcess()

    def fake_git_run(_repo_root: Path, *args: str, timeout: int = 10):
        class Result:
            def __init__(self, stdout: str, returncode: int = 0):
                self.stdout = stdout
                self.returncode = returncode

        if args == ("remote", "get-url", "origin"):
            return Result("https://github.com/example/musubi-tuner.git\n")
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return Result("ltx-2-dev\n")
        raise AssertionError(f"Unexpected git call: {args}")

    monkeypatch.setattr(management_tools, "_git_run", fake_git_run)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = management_tools.launch_setup_tool(repo_root, branch_override="ltx-2")

    assert result["mode"] == "direct"
    assert result["branch"] == "ltx-2"
    assert "-RepoUrl" in called["cmd"]
    assert "https://github.com/example/musubi-tuner.git" in called["cmd"]
    assert "-Branch" in called["cmd"]
    assert "ltx-2" in called["cmd"]
    assert called["cwd"] == str(repo_root)
