from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "ruff_pre_commit.py"
INSTALLER = ROOT / "tools" / "install_git_hooks.py"
RUFF_DIR = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin")


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    if not (RUFF_DIR / ("ruff.exe" if os.name == "nt" else "ruff")).exists():
        pytest.skip("repository Ruff executable is unavailable")
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "Test User")
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 88\n", encoding="utf-8")
    (tmp_path / "example.py").write_text("value = 1\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "initial")
    return tmp_path


def run_gate(repo: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join((str(RUFF_DIR), env.get("PATH", "")))
    return subprocess.run([sys.executable, str(RUNNER)], cwd=repo, env=env, capture_output=True, text=True)


def test_healthy_staged_snapshot_passes(repo: Path) -> None:
    assert run_gate(repo).returncode == 0


def test_lint_error_is_rejected(repo: Path) -> None:
    (repo / "example.py").write_text("import os\n", encoding="utf-8")
    git(repo, "add", "example.py")

    result = run_gate(repo)

    assert result.returncode != 0
    assert "F401" in result.stdout


def test_format_error_is_rejected(repo: Path) -> None:
    (repo / "example.py").write_text("items = {'a':1,'b':2}\n", encoding="utf-8")
    git(repo, "add", "example.py")

    result = run_gate(repo)

    assert result.returncode != 0
    assert "would be reformatted" in result.stdout


def test_partially_staged_error_uses_index_not_clean_worktree(repo: Path) -> None:
    (repo / "example.py").write_text("import os\n", encoding="utf-8")
    git(repo, "add", "example.py")
    (repo / "example.py").write_text("value = 1\n", encoding="utf-8")

    result = run_gate(repo)

    assert result.returncode != 0
    assert "F401" in result.stdout


def test_installer_preserves_hooks_path_and_refuses_overwrite(repo: Path) -> None:
    hook_source = repo / ".githooks" / "pre-commit"
    hook_source.parent.mkdir()
    hook_source.write_bytes(b"#!/bin/sh\nexit 0\n")
    git(repo, "config", "core.hooksPath", "global-forwarder")

    installed = subprocess.run([sys.executable, str(INSTALLER)], cwd=repo, capture_output=True, text=True)

    destination = repo / ".git" / "hooks" / "pre-commit"
    assert installed.returncode == 0
    assert destination.read_bytes() == hook_source.read_bytes()
    assert git(repo, "config", "--get", "core.hooksPath").stdout.strip() == "global-forwarder"

    destination.write_bytes(b"#!/bin/sh\nexit 42\n")
    refused = subprocess.run([sys.executable, str(INSTALLER)], cwd=repo, capture_output=True, text=True)
    assert refused.returncode != 0
    assert destination.read_bytes() == b"#!/bin/sh\nexit 42\n"
    assert "Refusing to overwrite" in refused.stderr
