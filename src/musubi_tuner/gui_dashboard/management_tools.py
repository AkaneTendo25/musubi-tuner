from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.gui_dashboard.validation import validate_process_config

INSTALL_STATE_NAME = ".musubi_install_state.json"
DASHBOARD_LAUNCHER_NAME = "launch_musubi_dashboard.cmd"
SETUP_LAUNCHER_NAME = "launch_musubi_setup.cmd"
DASHBOARD_SHORTCUT_NAME = "Musubi Tuner Dashboard.lnk"
SETUP_SHORTCUT_NAME = "Musubi Tuner Setup and Update.lnk"
CHECK_CACHE_WINDOW = timedelta(minutes=15)
DEFAULT_BRANCH = "ltx-2"
DEFAULT_REPO_URL = "https://github.com/AkaneTendo25/musubi-tuner.git"
SUPPORTED_BRANCHES = ("ltx-2", "ltx-2-dev")
PROCESS_LABELS: dict[str, tuple[str, str]] = {
    "cache_latents": ("Cache Latents", "/caching"),
    "cache_text": ("Cache Text", "/caching"),
    "training": ("Training", "/training"),
    "inference": ("Inference", "/inference"),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_timestamp(value: str | None) -> str:
    parsed = _parse_iso(value)
    if not parsed:
        return "never"
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M")


def _short_sha(value: str | None) -> str:
    if not value:
        return ""
    return value[:8]


def find_repo_root(start: Path | None = None) -> Path:
    candidate = (start or Path.cwd()).resolve()
    search = [candidate, *candidate.parents]
    for root in search:
        if (root / "scripts" / "install.ps1").exists() and (root / "src").exists():
            return root
    return candidate


def load_install_state_info(repo_root: Path) -> tuple[dict[str, Any], str]:
    path = repo_root / INSTALL_STATE_NAME
    if not path.exists():
        return {}, ""
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except (OSError, json.JSONDecodeError):
        return {}, f"Install state file at {path} could not be parsed. Setup / Update can recreate it."


def load_install_state(repo_root: Path) -> dict[str, Any]:
    state, _ = load_install_state_info(repo_root)
    return state


def _desktop_dir() -> Path:
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        return Path(user_profile) / "Desktop"
    return Path.home() / "Desktop"


def _git_run(repo_root: Path, *args: str, timeout: int = 10) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _git_status_path(line: str) -> str:
    return line[3:] if len(line) >= 4 else ""


def _format_git_status_entry(line: str) -> str:
    status = line[:2] if len(line) >= 2 else line
    path = _git_status_path(line)
    labels = {
        "??": "untracked",
        " M": "modified",
        "M ": "staged",
        "MM": "modified",
        " A": "added",
        "A ": "staged add",
        " D": "deleted",
        "D ": "staged delete",
    }
    label = labels.get(status, status.strip() or "changed")
    return f"{label}: {path}"


def _format_file_list(items: list[str], max_items: int = 5) -> str:
    shown = items[:max_items]
    suffix = f" (+{len(items) - max_items} more)" if len(items) > max_items else ""
    return f"{', '.join(shown)}{suffix}"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _stash_recovery_commands(target: str) -> list[str]:
    if not target:
        return []
    return [
        f"git stash show --include-untracked --name-only {target}",
        f"git stash show --include-untracked -p {target}",
        f"git stash apply {target}",
        f"git stash pop {target}",
    ]


def _build_backup_status(repo_root: Path, install_state: dict[str, Any]) -> dict[str, Any]:
    raw = install_state.get("last_backup")
    if not isinstance(raw, dict):
        return {
            "present": False,
            "available": False,
            "created_utc": "",
            "created_label": "never",
            "message": "",
            "stash_ref": "",
            "stash_sha": "",
            "stash_subject": "",
            "recovery_target": "",
            "file_backup_path": "",
            "file_backup_available": False,
            "files": [],
            "commands": [],
            "error": "",
        }

    target = str(raw.get("recovery_target") or raw.get("stash_sha") or raw.get("stash_ref") or "").strip()
    file_backup_path = str(raw.get("file_backup_path") or "").strip()
    files = _string_list(raw.get("files"))
    commands = _string_list(raw.get("commands")) or _stash_recovery_commands(target)
    error = ""
    available = False
    file_backup_available = bool(file_backup_path and Path(file_backup_path).exists())

    if target and target != file_backup_path:
        show_proc = _git_run(repo_root, "stash", "show", "--include-untracked", "--name-only", target)
        if show_proc and show_proc.returncode == 0:
            available = True
            fresh_files = [line.strip() for line in show_proc.stdout.splitlines() if line.strip()]
            if fresh_files:
                files = fresh_files
        else:
            error = "The recorded stash backup was not found. It may have been applied, popped, or dropped."
    if file_backup_available:
        available = True
    elif file_backup_path and not error:
        error = "The recorded local-file backup folder was not found. It may have been moved or deleted."

    return {
        "present": True,
        "available": available,
        "created_utc": str(raw.get("created_utc") or ""),
        "created_label": _format_timestamp(str(raw.get("created_utc") or "")),
        "message": str(raw.get("message") or ""),
        "stash_ref": str(raw.get("stash_ref") or ""),
        "stash_sha": str(raw.get("stash_sha") or ""),
        "stash_subject": str(raw.get("stash_subject") or ""),
        "recovery_target": target,
        "file_backup_path": file_backup_path,
        "file_backup_available": file_backup_available,
        "files": files,
        "commands": commands,
        "error": error,
    }


def _git_value(repo_root: Path, *args: str, timeout: int = 10) -> str:
    result = _git_run(repo_root, *args, timeout=timeout)
    if not result or result.returncode != 0:
        return ""
    return result.stdout.strip()


def _doctor_item(
    key: str,
    label: str,
    status: str,
    detail: str,
    *,
    page: str = "",
    path: str = "",
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "status": status,
        "detail": detail,
        "page": page,
        "path": path,
    }


def _resolve_project_path(project_dir: str | None, raw_path: str | None) -> Path | None:
    value = str(raw_path or "").strip()
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute() or not str(project_dir or "").strip():
        return candidate
    return Path(project_dir) / candidate


def _command_exists(*names: str) -> bool:
    for name in names:
        if shutil.which(name):
            return True
    return False


def _has_hf_token() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        return True
    token_paths = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path(os.environ.get("APPDATA", "")) / "huggingface" / "token" if os.environ.get("APPDATA") else None,
    ]
    return any(path and path.is_file() for path in token_paths)


def _effective_checkpoint(config: ProjectConfig, section_name: str) -> str:
    section = getattr(config, section_name)
    return str(getattr(section, "ltx2_checkpoint", "") or config.default_ltx2_checkpoint or "").strip()


def _effective_gemma_source(config: ProjectConfig, section_name: str) -> tuple[str, str]:
    section = getattr(config, section_name)
    safetensors = str(getattr(section, "gemma_safetensors", "") or config.default_gemma_safetensors or "").strip()
    if safetensors:
        return "file", safetensors
    root = str(getattr(section, "gemma_root", "") or config.default_gemma_root or "").strip()
    return "dir", root


def _check_required_path(
    *,
    key: str,
    label: str,
    path: Path | None,
    expected: str,
    page: str,
    optional: bool = False,
) -> dict[str, Any]:
    if path is None:
        return _doctor_item(
            key,
            label,
            "warning" if optional else "error",
            "Not configured." if optional else "Not configured.",
            page=page,
        )

    exists = path.is_file() if expected == "file" else path.is_dir()
    if exists:
        return _doctor_item(key, label, "ok", f"Found at {path}", page=page, path=str(path))

    noun = "file" if expected == "file" else "directory"
    return _doctor_item(
        key, label, "warning" if optional else "error", f"Expected {noun} is missing: {path}", page=page, path=str(path)
    )


def _dataset_source_status(config: ProjectConfig) -> dict[str, Any]:
    entries = list(config.dataset.datasets or [])
    if not entries:
        return _doctor_item("datasets", "Training Datasets", "error", "No training datasets are configured.", page="/dataset")

    missing: list[str] = []
    for index, entry in enumerate(entries, start=1):
        directory = _resolve_project_path(config.project_dir, entry.directory)
        jsonl_file = _resolve_project_path(config.project_dir, entry.jsonl_file)
        source_ok = False
        if directory and directory.is_dir():
            source_ok = True
        if jsonl_file and jsonl_file.is_file():
            source_ok = True
        if not source_ok:
            wanted = []
            if directory:
                wanted.append(str(directory))
            if jsonl_file:
                wanted.append(str(jsonl_file))
            if not wanted:
                wanted.append("no directory or JSONL configured")
            missing.append(f"Dataset #{index}: {' | '.join(wanted)}")

    if missing:
        detail = f"{len(missing)} dataset source(s) are missing. " + " ; ".join(missing[:3])
        return _doctor_item("datasets", "Training Datasets", "error", detail, page="/dataset")

    return _doctor_item("datasets", "Training Datasets", "ok", f"{len(entries)} training dataset source(s) found.", page="/dataset")


def _process_doctor_entries(
    config: ProjectConfig,
    requirement_statuses: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    requirements_by_process = {
        "cache_latents": ["venv_python", "project_dir", "datasets", "cache_latents_checkpoint"],
        "cache_text": ["venv_python", "project_dir", "datasets", "cache_text_checkpoint", "cache_text_gemma"],
        "training": ["venv_python", "project_dir", "datasets", "training_checkpoint"],
        "inference": ["venv_python", "project_dir", "inference_checkpoint", "inference_gemma"],
    }
    process_entries: list[dict[str, Any]] = []

    for key, (label, page) in PROCESS_LABELS.items():
        report = validate_process_config(key, config)
        blocking = [
            requirement_statuses[item_key]["detail"]
            for item_key in requirements_by_process.get(key, [])
            if requirement_statuses.get(item_key, {}).get("status") == "error"
        ]
        warnings = [
            requirement_statuses[item_key]["detail"]
            for item_key in requirements_by_process.get(key, [])
            if requirement_statuses.get(item_key, {}).get("status") == "warning"
        ]
        status = "ok"
        if (not report["ok"]) or blocking:
            status = "error"
        elif report["warnings"] or warnings:
            status = "warning"

        detail = report["summary"]
        if blocking:
            detail = f"{detail} Blocking: {' | '.join(blocking[:2])}"
        elif warnings:
            detail = f"{detail} Warning: {' | '.join(warnings[:2])}"

        process_entries.append(
            {
                "key": key,
                "label": label,
                "page": page,
                "status": status,
                "summary": detail,
                "validation_ok": report["ok"],
                "validation_errors": len(report["errors"]),
                "validation_warnings": len(report["warnings"]),
                "blocking_requirements": blocking,
                "requirement_warnings": warnings,
            }
        )

    return process_entries


def build_doctor_status(
    project_config: ProjectConfig | None,
    management_status: dict[str, Any],
) -> dict[str, Any]:
    repo = management_status["repo"]
    install = management_status["install"]

    environment_checks = [
        _doctor_item(
            "venv_python",
            "Virtual Environment",
            "ok" if install["venv_exists"] else "error",
            f"Using {install['venv_python_path']}" if install["venv_exists"] else "Virtual environment Python is missing.",
            page="/settings",
            path=install["venv_python_path"],
        ),
        _doctor_item(
            "git",
            "Git",
            "ok" if repo["git_available"] else "error",
            "Git is available." if repo["git_available"] else "Git is not available on PATH.",
            page="/settings",
        ),
        _doctor_item(
            "node",
            "Node.js / npm",
            "ok" if _command_exists("npm.cmd", "npm") else "warning",
            "npm is available for frontend rebuilds."
            if _command_exists("npm.cmd", "npm")
            else "npm is not available. Setup / Update can still repair this.",
            page="/settings",
        ),
        _doctor_item(
            "frontend_dist",
            "Frontend Build",
            "ok" if install["frontend_dist_exists"] else "warning",
            "Dashboard frontend build is present."
            if install["frontend_dist_exists"]
            else "Frontend dist is missing and should be rebuilt through Setup / Update.",
            page="/settings",
            path=install["frontend_dist_path"],
        ),
        _doctor_item(
            "hf_token",
            "Hugging Face Auth",
            "ok" if _has_hf_token() else "warning",
            "HF token detected."
            if _has_hf_token()
            else "No HF token detected. Public downloads may work, gated/private repos will not.",
            page="/settings",
        ),
    ]

    if repo["origin_error"]:
        environment_checks.append(
            _doctor_item(
                "origin",
                "Repository Updates",
                "warning",
                repo["origin_error"],
                page="/settings",
            )
        )

    if project_config is None:
        warning_count = sum(1 for item in environment_checks if item["status"] == "warning") + 1
        error_count = sum(1 for item in environment_checks if item["status"] == "error")
        summary = "Environment checked. Load or create a project to validate caching, training, and inference readiness."
        return {
            "loaded_project": False,
            "project_name": "",
            "ready": error_count == 0,
            "summary": summary,
            "error_count": error_count,
            "warning_count": warning_count,
            "environment_checks": environment_checks,
            "asset_checks": [
                _doctor_item("project", "Project", "warning", "No project is currently loaded.", page="/"),
            ],
            "processes": [],
        }

    asset_checks = [
        _check_required_path(
            key="project_dir",
            label="Project Directory",
            path=_resolve_project_path(None, project_config.project_dir),
            expected="dir",
            page="/",
        ),
        _check_required_path(
            key="model_dir",
            label="Model Directory",
            path=_resolve_project_path(project_config.project_dir, project_config.model_dir),
            expected="dir",
            page="/",
            optional=True,
        ),
        _dataset_source_status(project_config),
        _check_required_path(
            key="cache_latents_checkpoint",
            label="Cache Latents Checkpoint",
            path=_resolve_project_path(project_config.project_dir, _effective_checkpoint(project_config, "caching")),
            expected="file",
            page="/caching",
        ),
        _check_required_path(
            key="cache_text_checkpoint",
            label="Cache Text Checkpoint",
            path=_resolve_project_path(project_config.project_dir, _effective_checkpoint(project_config, "caching")),
            expected="file",
            page="/caching",
        ),
    ]

    cache_text_gemma_kind, cache_text_gemma_path = _effective_gemma_source(project_config, "caching")
    training_gemma_kind, training_gemma_path = _effective_gemma_source(project_config, "training")
    inference_gemma_kind, inference_gemma_path = _effective_gemma_source(project_config, "inference")

    asset_checks.extend(
        [
            _check_required_path(
                key="cache_text_gemma",
                label="Cache Text Gemma",
                path=_resolve_project_path(project_config.project_dir, cache_text_gemma_path),
                expected=cache_text_gemma_kind,
                page="/caching",
            ),
            _check_required_path(
                key="training_checkpoint",
                label="Training Checkpoint",
                path=_resolve_project_path(project_config.project_dir, _effective_checkpoint(project_config, "training")),
                expected="file",
                page="/training",
            ),
            _check_required_path(
                key="training_gemma",
                label="Training Gemma",
                path=_resolve_project_path(project_config.project_dir, training_gemma_path),
                expected=training_gemma_kind,
                page="/training",
                optional=True,
            ),
            _check_required_path(
                key="inference_checkpoint",
                label="Inference Checkpoint",
                path=_resolve_project_path(project_config.project_dir, _effective_checkpoint(project_config, "inference")),
                expected="file",
                page="/inference",
            ),
            _check_required_path(
                key="inference_gemma",
                label="Inference Gemma",
                path=_resolve_project_path(project_config.project_dir, inference_gemma_path),
                expected=inference_gemma_kind,
                page="/inference",
            ),
        ]
    )

    requirement_statuses = {item["key"]: item for item in [*environment_checks, *asset_checks]}
    process_entries = _process_doctor_entries(project_config, requirement_statuses)

    error_count = sum(1 for item in [*environment_checks, *asset_checks] if item["status"] == "error")
    error_count += sum(1 for item in process_entries if item["status"] == "error")
    warning_count = sum(1 for item in [*environment_checks, *asset_checks] if item["status"] == "warning")
    warning_count += sum(1 for item in process_entries if item["status"] == "warning")

    ready_labels = [item["label"] for item in process_entries if item["status"] == "ok"]
    if error_count == 0 and warning_count == 0:
        summary = "Environment, project assets, and launch validation all look ready."
    elif error_count == 0:
        summary = f"No blocking issues found. {warning_count} warning(s) remain."
    else:
        summary = f"{error_count} blocking issue(s) and {warning_count} warning(s) found."
    if ready_labels:
        summary += f" Ready now: {', '.join(ready_labels)}."

    return {
        "loaded_project": True,
        "project_name": project_config.name,
        "ready": error_count == 0,
        "summary": summary,
        "error_count": error_count,
        "warning_count": warning_count,
        "environment_checks": environment_checks,
        "asset_checks": asset_checks,
        "processes": process_entries,
    }


def _github_reachable(remote_url: str) -> bool:
    if "github.com" not in remote_url.lower():
        return True
    try:
        with socket.create_connection(("github.com", 443), timeout=2):
            return True
    except OSError:
        return False


def _git_paths_changed(repo_root: Path, from_ref: str, to_ref: str, pathspecs: list[str]) -> bool | None:
    if not from_ref or not to_ref or not pathspecs:
        return None

    from_ok = _git_run(repo_root, "rev-parse", "--verify", from_ref)
    to_ok = _git_run(repo_root, "rev-parse", "--verify", to_ref)
    if not from_ok or from_ok.returncode != 0 or not to_ok or to_ok.returncode != 0:
        return None

    diff = _git_run(repo_root, "diff", "--name-only", f"{from_ref}..{to_ref}", "--", *pathspecs)
    if not diff or diff.returncode != 0:
        return None
    return bool(diff.stdout.strip())


def get_repository_status(repo_root: Path, branch: str, remote_url: str, state: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": repo_root.exists(),
        "git_available": False,
        "is_git_repo": False,
        "origin_configured": False,
        "head": "",
        "head_short": "",
        "branch": "",
        "dirty": False,
        "remote_head": "",
        "remote_head_short": "",
        "local_ahead_count": 0,
        "remote_ahead_count": 0,
        "diverged": False,
        "update_available": False,
        "can_auto_update": False,
        "tracked_dirty": False,
        "has_untracked": False,
        "changed_files": [],
        "untracked_files": [],
        "local_files": [],
        "fetch_attempted": False,
        "fetch_succeeded": False,
        "offline": False,
        "origin_error": "",
        "origin_url": "",
        "expected_remote_url": remote_url.strip(),
        "summary": "Repository not found",
    }

    if not repo_root.exists():
        return result
    if not (repo_root / ".git").exists():
        result["summary"] = "Repository exists but is not a git checkout"
        return result

    result["is_git_repo"] = True
    head = _git_run(repo_root, "rev-parse", "HEAD")
    if not head or head.returncode != 0:
        result["summary"] = "Git is not available"
        return result

    result["git_available"] = True
    result["head"] = head.stdout.strip()
    result["head_short"] = _short_sha(result["head"])

    branch_proc = _git_run(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    if branch_proc and branch_proc.returncode == 0:
        result["branch"] = branch_proc.stdout.strip()

    status_proc = _git_run(repo_root, "status", "--porcelain", "--", ".")
    if status_proc and status_proc.returncode == 0:
        status_lines = [line for line in status_proc.stdout.splitlines() if line.strip()]
        tracked_lines = [line for line in status_lines if not line.startswith("?? ")]
        untracked_lines = [line for line in status_lines if line.startswith("?? ")]
        result["changed_files"] = [_format_git_status_entry(line) for line in tracked_lines]
        result["untracked_files"] = [_format_git_status_entry(line) for line in untracked_lines]
        result["local_files"] = [_format_git_status_entry(line) for line in status_lines]
        result["tracked_dirty"] = bool(tracked_lines)
        result["has_untracked"] = bool(untracked_lines)
        result["dirty"] = bool(tracked_lines)

    origin_url = _git_value(repo_root, "remote", "get-url", "origin")
    result["origin_url"] = origin_url
    result["origin_configured"] = bool(origin_url)

    repo_state = state.get("repo", {})
    last_checked = _parse_iso(repo_state.get("last_checked_utc"))
    cached_recent = last_checked and (datetime.now(timezone.utc) - last_checked) < CHECK_CACHE_WINDOW

    offline = False

    if cached_recent and result["origin_configured"]:
        result["remote_head"] = repo_state.get("remote_head", "") or ""
        result["remote_head_short"] = repo_state.get("remote_head_short", "") or _short_sha(result["remote_head"])
        result["local_ahead_count"] = int(repo_state.get("local_ahead_count") or 0)
        result["remote_ahead_count"] = int(repo_state.get("remote_ahead_count") or 0)
        result["diverged"] = bool(repo_state.get("diverged"))
        result["update_available"] = bool(repo_state.get("update_available"))
    elif not result["origin_configured"]:
        result["origin_error"] = "No origin remote is configured"
    elif _github_reachable(origin_url or remote_url):
        result["fetch_attempted"] = True
        fetch_proc = _git_run(repo_root, "fetch", "--quiet", "origin", branch, timeout=20)
        if fetch_proc and fetch_proc.returncode == 0:
            result["fetch_succeeded"] = True
            remote_ref = f"origin/{branch}"
            remote_proc = _git_run(repo_root, "rev-parse", remote_ref)
            if remote_proc and remote_proc.returncode == 0:
                result["remote_head"] = remote_proc.stdout.strip()
                result["remote_head_short"] = _short_sha(result["remote_head"])
            counts_proc = _git_run(repo_root, "rev-list", "--left-right", "--count", f"HEAD...{remote_ref}")
            if counts_proc and counts_proc.returncode == 0:
                parts = counts_proc.stdout.strip().split()
                if len(parts) >= 2:
                    result["local_ahead_count"] = int(parts[0])
                    result["remote_ahead_count"] = int(parts[1])
                    result["diverged"] = result["local_ahead_count"] > 0 and result["remote_ahead_count"] > 0
                    result["update_available"] = result["remote_ahead_count"] > 0
        else:
            result["origin_error"] = "Fetching origin failed. Check repository access, branch name, and network connectivity."
    else:
        offline = True
        result["offline"] = True
        result["origin_error"] = "Could not reach the origin host to check for updates"

    if result["dirty"]:
        result["summary"] = f"Tracked local changes detected ({len(result['changed_files'])} file(s)); safe auto-update is disabled"
    elif result["diverged"]:
        result["summary"] = "Local branch and origin have diverged"
    elif result["update_available"]:
        result["summary"] = f"Update available: origin/{branch} is {result['remote_ahead_count']} commit(s) ahead"
    elif result["local_ahead_count"] > 0:
        result["summary"] = f"Local branch is {result['local_ahead_count']} commit(s) ahead of origin"
    elif not result["origin_configured"]:
        result["summary"] = "No origin remote configured; update checks are unavailable"
    elif result["fetch_attempted"] and not result["fetch_succeeded"]:
        result["summary"] = "Origin fetch failed; review git remote configuration and network access"
    elif offline:
        result["summary"] = "Could not reach the origin host to check for updates"
    elif result["has_untracked"]:
        result["summary"] = "Repository is up to date; untracked local files are present and will be left in place"
    else:
        result["summary"] = "Repository is up to date"

    result["can_auto_update"] = result["update_available"] and not result["tracked_dirty"] and not result["diverged"]
    return result


def _setup_problem(
    severity: str,
    area: str,
    message: str,
    *,
    repairable: bool = False,
    repair: str = "",
) -> dict[str, Any]:
    return {
        "severity": severity,
        "area": area,
        "message": message,
        "repairable": repairable,
        "repair": repair,
    }


def _build_setup_problems(
    *,
    repo_root: Path,
    branch: str,
    remote_url: str,
    repo_status: dict[str, Any],
    install_state_present: bool,
    install_state_error: str,
    venv_exists: bool,
    frontend_dist_exists: bool,
    deps_recommended: bool,
    frontend_recommended: bool,
    dashboard_shortcut_exists: bool,
    setup_shortcut_exists: bool,
) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []

    if not repo_status["exists"]:
        problems.append(
            _setup_problem(
                "error",
                "Repository",
                f"Repository directory does not exist: {repo_root}",
                repairable=True,
                repair="Open Setup / Update to clone the repository.",
            )
        )
    elif not repo_status["is_git_repo"]:
        problems.append(
            _setup_problem(
                "error",
                "Repository",
                f"Repository directory exists but is not a Git checkout: {repo_root}",
                repair="Move the folder aside or choose a clean install path before cloning.",
            )
        )
    else:
        if not repo_status["git_available"]:
            problems.append(
                _setup_problem(
                    "error",
                    "Git",
                    "Git is not available or cannot read this repository.",
                    repairable=True,
                    repair="Install Git, then rerun Setup / Update.",
                )
            )
        else:
            if not repo_status["origin_configured"]:
                repair = f"Repair setup can add origin -> {remote_url}." if remote_url else "Set a repository URL, then add origin."
                problems.append(
                    _setup_problem(
                        "warning",
                        "Repository",
                        "The Git origin remote is missing, so update checks cannot fetch remote commits.",
                        repairable=bool(remote_url),
                        repair=repair,
                    )
                )
            elif repo_status["fetch_attempted"] and not repo_status["fetch_succeeded"]:
                problems.append(
                    _setup_problem(
                        "warning",
                        "Repository",
                        f"Fetching origin/{branch} failed.",
                        repair="Check network access, credentials, remote URL, and branch name.",
                    )
                )
            elif repo_status["offline"]:
                problems.append(
                    _setup_problem(
                        "warning",
                        "Network",
                        "The origin host was not reachable during the update check.",
                        repair="Reconnect to the network and refresh status.",
                    )
                )

            if repo_status["branch"] and repo_status["branch"] != branch:
                problems.append(
                    _setup_problem(
                        "warning",
                        "Repository",
                        f"Setup is configured for '{branch}', but the checkout is on '{repo_status['branch']}'.",
                        repairable=not repo_status["tracked_dirty"],
                        repair="Setup / Update can switch branches when tracked local changes do not block checkout.",
                    )
                )
            if repo_status["diverged"]:
                problems.append(
                    _setup_problem(
                        "warning",
                        "Repository",
                        f"The local branch and origin/{branch} have diverged.",
                        repair="Resolve the Git history manually; setup only performs fast-forward updates.",
                    )
                )
            elif repo_status["update_available"]:
                problems.append(
                    _setup_problem(
                        "info",
                        "Repository",
                        f"origin/{branch} is {repo_status['remote_ahead_count']} commit(s) ahead.",
                        repairable=True,
                        repair="Open Setup / Update to fast-forward the checkout.",
                    )
                )
            if repo_status["tracked_dirty"]:
                problems.append(
                    _setup_problem(
                        "warning",
                        "Repository",
                        f"Tracked local changes are present: {_format_file_list(repo_status['changed_files'])}",
                        repairable=True,
                        repair="Use StashAndUpdate to back them up before updating, or handle them manually.",
                    )
                )

    if not venv_exists:
        problems.append(
            _setup_problem(
                "error",
                "Python",
                "Virtual environment Python is missing.",
                repairable=True,
                repair="Open Setup / Update to create the virtual environment and install dependencies.",
            )
        )
    elif deps_recommended:
        problems.append(
            _setup_problem(
                "warning",
                "Python",
                "Python dependency inputs changed or the CUDA target changed.",
                repairable=True,
                repair="Open Setup / Update to reinstall Python dependencies.",
            )
        )

    if not frontend_dist_exists:
        problems.append(
            _setup_problem(
                "warning",
                "Dashboard",
                "Dashboard frontend build is missing.",
                repairable=True,
                repair="Open Setup / Update to rebuild the dashboard frontend.",
            )
        )
    elif frontend_recommended:
        problems.append(
            _setup_problem(
                "warning",
                "Dashboard",
                "Dashboard frontend sources changed since the recorded build.",
                repairable=True,
                repair="Open Setup / Update to rebuild the dashboard frontend.",
            )
        )

    if not dashboard_shortcut_exists or not setup_shortcut_exists:
        problems.append(
            _setup_problem(
                "warning",
                "Shortcuts",
                "One or more desktop shortcuts are missing.",
                repairable=True,
                repair="Open Setup / Update to recreate desktop shortcuts.",
            )
        )
    if not install_state_present:
        problems.append(
            _setup_problem(
                "warning",
                "Setup State",
                "Install state has not been recorded yet.",
                repairable=True,
                repair="Completing Setup / Update will recreate .musubi_install_state.json.",
            )
        )
    if install_state_error:
        problems.append(
            _setup_problem(
                "warning",
                "Setup State",
                install_state_error,
                repairable=True,
                repair="Completing Setup / Update will rewrite the install state file.",
            )
        )

    return problems


def get_management_status(repo_root: Path | None = None, project_config: ProjectConfig | None = None) -> dict[str, Any]:
    root = find_repo_root(repo_root)
    state, state_error = load_install_state_info(root)
    branch = str(state.get("branch") or "").strip() or _git_value(root, "rev-parse", "--abbrev-ref", "HEAD") or DEFAULT_BRANCH
    remote_url = str(state.get("repo_url") or "").strip() or _git_value(root, "remote", "get-url", "origin") or DEFAULT_REPO_URL
    repo_status = get_repository_status(root, branch, remote_url, state)

    install_state = state.get("install", {})
    venv_python = root / "venv" / "Scripts" / "python.exe"
    frontend_dist = root / "src" / "musubi_tuner" / "gui_dashboard" / "frontend" / "dist" / "index.html"
    dashboard_launcher = root / DASHBOARD_LAUNCHER_NAME
    setup_launcher = root / SETUP_LAUNCHER_NAME
    desktop = _desktop_dir()
    dashboard_shortcut = desktop / DASHBOARD_SHORTCUT_NAME
    setup_shortcut = desktop / SETUP_SHORTCUT_NAME

    compare_target = (
        repo_status["remote_head"] if repo_status["can_auto_update"] and repo_status["remote_head"] else repo_status["head"]
    )
    deps_changed = _git_paths_changed(
        root, str(install_state.get("deps_commit") or ""), compare_target, ["pyproject.toml", "uv.lock"]
    )
    frontend_changed = _git_paths_changed(
        root,
        str(install_state.get("frontend_commit") or ""),
        compare_target,
        [
            "src/musubi_tuner/gui_dashboard/frontend/package.json",
            "src/musubi_tuner/gui_dashboard/frontend/package-lock.json",
            "src/musubi_tuner/gui_dashboard/frontend/src",
            "src/musubi_tuner/gui_dashboard/frontend/svelte.config.js",
            "src/musubi_tuner/gui_dashboard/frontend/vite.config.js",
        ],
    )

    deps_recommended = (not venv_python.exists()) or bool(deps_changed)
    frontend_recommended = (not frontend_dist.exists()) or bool(frontend_changed)
    backup_status = _build_backup_status(root, install_state)
    setup_problems = _build_setup_problems(
        repo_root=root,
        branch=branch,
        remote_url=remote_url,
        repo_status=repo_status,
        install_state_present=bool(state),
        install_state_error=state_error,
        venv_exists=venv_python.exists(),
        frontend_dist_exists=frontend_dist.exists(),
        deps_recommended=deps_recommended,
        frontend_recommended=frontend_recommended,
        dashboard_shortcut_exists=dashboard_shortcut.exists(),
        setup_shortcut_exists=setup_shortcut.exists(),
    )
    repairable_problem_count = sum(1 for item in setup_problems if item["repairable"])

    notices: list[str] = []
    recommendations: list[str] = []
    warnings: list[str] = []

    if repo_status["diverged"]:
        recommendations.append(
            "Resolve the diverged Git branch manually; Setup / Update only performs fast-forward repository updates."
        )
    elif repo_status["update_available"] and repo_status["tracked_dirty"]:
        recommendations.append(
            "Open Setup / Update to choose whether to skip the repo update or stash local changes before updating."
        )
    elif repo_status["update_available"]:
        recommendations.append("Run Setup / Update to pull the latest repository changes.")
    if deps_recommended:
        recommendations.append("Reinstall Python dependencies so the environment matches the selected repo revision.")
    if frontend_recommended:
        recommendations.append("Rebuild the dashboard frontend so the shipped UI matches the source tree.")
    if not dashboard_shortcut.exists() or not setup_shortcut.exists():
        recommendations.append("Recreate desktop shortcuts so both Dashboard and Setup / Update remain easy to access.")
    if not state:
        recommendations.append("Run Setup / Update once so this installation records its state and saved launcher settings.")
    if not repo_status["origin_configured"] and repo_status["is_git_repo"]:
        recommendations.append("Configure an origin remote or reclone through Setup / Update if you want automatic update checks.")
    if repo_status["branch"] and branch and repo_status["branch"] != branch and not repo_status["dirty"]:
        recommendations.append(f"Open Setup / Update on '{branch}' to switch the working copy away from '{repo_status['branch']}'.")
    if repo_status["untracked_files"]:
        untracked_files = _format_file_list(repo_status["untracked_files"])
        notices.append(f"Untracked local files are present and will be left in place by safe updates: {untracked_files}.")
    if backup_status["present"] and backup_status["available"]:
        notices.append("A setup backup is available below with file list and recovery commands.")
    if backup_status["present"] and backup_status["error"]:
        warnings.append(backup_status["error"])
    if repo_status["dirty"]:
        changed_files = _format_file_list(repo_status["changed_files"])
        warnings.append(f"Repository has tracked local changes. Safe update will not overwrite them: {changed_files}.")
    if not install_state:
        warnings.append("Install state has not been recorded yet. Rerunning Setup / Update will register this installation.")
    if state_error:
        warnings.append(state_error)
    if repo_status["branch"] and branch and repo_status["branch"] != branch:
        warnings.append(
            f"Repository is currently checked out on '{repo_status['branch']}', but setup is configured for '{branch}'."
        )
    if not repo_status["origin_configured"] and repo_status["is_git_repo"]:
        warnings.append("No origin remote is configured, so update availability cannot be checked automatically.")
    if not setup_launcher.exists() and (root / "scripts" / "install.ps1").exists():
        warnings.append(
            "Setup launcher is missing. The dashboard can still open Setup / Update directly and recreate the launcher."
        )
    if not dashboard_launcher.exists():
        warnings.append("Dashboard launcher is missing. Rerunning Setup / Update will recreate it.")

    result = {
        "repo_root": str(root),
        "install_state_path": str(root / INSTALL_STATE_NAME),
        "install_state_present": bool(state),
        "branch": branch,
        "remote_url": remote_url,
        "repo": repo_status,
        "install": {
            "venv_python_path": str(venv_python),
            "venv_exists": venv_python.exists(),
            "dashboard_launcher_path": str(dashboard_launcher),
            "dashboard_launcher_exists": dashboard_launcher.exists(),
            "setup_launcher_path": str(setup_launcher),
            "setup_launcher_exists": setup_launcher.exists(),
            "dashboard_shortcut_path": str(dashboard_shortcut),
            "dashboard_shortcut_exists": dashboard_shortcut.exists(),
            "setup_shortcut_path": str(setup_shortcut),
            "setup_shortcut_exists": setup_shortcut.exists(),
            "frontend_dist_path": str(frontend_dist),
            "frontend_dist_exists": frontend_dist.exists(),
            "last_success_utc": install_state.get("last_success_utc", ""),
            "last_success_label": _format_timestamp(install_state.get("last_success_utc")),
            "deps_commit": install_state.get("deps_commit", ""),
            "deps_timestamp_utc": install_state.get("deps_timestamp_utc", ""),
            "deps_timestamp_label": _format_timestamp(install_state.get("deps_timestamp_utc")),
            "frontend_commit": install_state.get("frontend_commit", ""),
            "frontend_timestamp_utc": install_state.get("frontend_timestamp_utc", ""),
            "frontend_timestamp_label": _format_timestamp(install_state.get("frontend_timestamp_utc")),
            "backup": backup_status,
            "deps_recommended": deps_recommended,
            "frontend_recommended": frontend_recommended,
        },
        "actions": {
            "can_launch_setup": (setup_launcher.exists() or (root / "scripts" / "install.ps1").exists()),
            "can_repair_setup": (setup_launcher.exists() or (root / "scripts" / "install.ps1").exists())
            and repairable_problem_count > 0,
            "can_open_repo": root.exists(),
            "setup_launch_mode": "launcher"
            if setup_launcher.exists()
            else ("direct" if (root / "scripts" / "install.ps1").exists() else "unavailable"),
        },
        "problems": setup_problems,
        "problem_count": len(setup_problems),
        "repairable_problem_count": repairable_problem_count,
        "notices": notices,
        "warnings": warnings,
        "recommendations": recommendations,
        "checked_utc": _utc_now_iso(),
    }
    result["doctor"] = build_doctor_status(project_config, result)
    return result


def launch_setup_tool(
    repo_root: Path | None = None,
    branch_override: str | None = None,
    repair_mode: bool = False,
) -> dict[str, Any]:
    root = find_repo_root(repo_root)
    state, _ = load_install_state_info(root)
    setup_launcher = root / SETUP_LAUNCHER_NAME
    branch = (
        branch_override
        or str(state.get("branch") or "").strip()
        or _git_value(root, "rev-parse", "--abbrev-ref", "HEAD")
        or DEFAULT_BRANCH
    ).strip()

    if setup_launcher.exists():
        cmd = ["cmd", "/c", str(setup_launcher), "-Branch", branch]
        mode = "launcher"
    else:
        install_script = root / "scripts" / "install.ps1"
        if not install_script.exists():
            raise FileNotFoundError(f"Installer script not found at {install_script}")
        remote_url = str(state.get("repo_url") or "").strip() or _git_value(root, "remote", "get-url", "origin") or DEFAULT_REPO_URL
        cmd = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(install_script),
            "-InstallRoot",
            str(root.parent),
            "-RepoDir",
            str(root),
            "-RepoUrl",
            remote_url,
            "-Branch",
            branch,
            "-Cuda",
            str(state.get("cuda") or "cu128"),
            "-PythonVersion",
            str(state.get("python_version") or "3.12"),
            "-DashboardHost",
            str(state.get("dashboard_host") or "127.0.0.1"),
            "-Port",
            str(state.get("port") or 7860),
        ]
        mode = "direct"
    if repair_mode:
        cmd.append("-RepairMode")

    creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    try:
        subprocess.Popen(cmd, cwd=str(root), creationflags=creationflags)
    except OSError as exc:
        raise RuntimeError(f"Could not start Setup / Update: {exc}") from exc
    return {"launched": True, "mode": mode, "repo_root": str(root), "branch": branch, "repair_mode": repair_mode}


def open_repo_in_file_browser(repo_root: Path | None = None) -> dict[str, Any]:
    root = find_repo_root(repo_root)
    try:
        if hasattr(os, "startfile"):
            os.startfile(str(root))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(root)], cwd=str(root))
    except OSError as exc:
        raise RuntimeError(f"Could not open the repository folder in the system file browser: {exc}") from exc
    return {"opened": True, "repo_root": str(root)}
