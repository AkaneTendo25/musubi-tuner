from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Optional

from musubi_tuner.gui_dashboard.cli_defaults import get_ltx2_training_output_dir_default
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.utils import train_utils


def get_training_run_dir(config: ProjectConfig) -> Optional[Path]:
    if not config.project_dir:
        return None

    output_dir = config.training.output_dir or get_ltx2_training_output_dir_default()
    run_dir = Path(output_dir)
    if not run_dir.is_absolute():
        run_dir = Path(config.project_dir) / run_dir
    return run_dir


def has_local_autoresume_state(config: ProjectConfig) -> bool:
    run_dir = get_training_run_dir(config)
    if run_dir is None or not run_dir.is_dir():
        return False

    try:
        state_paths = list(run_dir.iterdir())
    except OSError:
        return False

    for path in state_paths:
        try:
            if path.name.endswith("-state") and train_utils.is_complete_state_dir(str(path)):
                return True
        except OSError:
            continue
    return False


def training_is_resume(config: ProjectConfig) -> bool:
    t = config.training
    if t.resume or t.resume_from_huggingface:
        return True
    return bool(t.autoresume and has_local_autoresume_state(config))


def clear_training_dashboard_files(config: ProjectConfig, *, keep_history: bool = False) -> None:
    run_dir = get_training_run_dir(config)
    if run_dir is None:
        return

    dashboard_dir = run_dir / "dashboard"
    filenames = ("status.json",) if keep_history else ("metrics.parquet", "status.json", "events.json")
    for filename in filenames:
        path = dashboard_dir / filename
        try:
            if path.exists():
                os.remove(path)
        except OSError:
            pass


def write_training_launch_status(config: ProjectConfig) -> None:
    run_dir = get_training_run_dir(config)
    if run_dir is None:
        return

    dashboard_dir = run_dir / "dashboard"
    status_path = dashboard_dir / "status.json"
    try:
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps(
                {
                    "status": "launching",
                    "step": 0,
                    "max_steps": int(config.training.max_train_steps or 0),
                    "epoch": 0,
                    "max_epochs": 0,
                    "elapsed_sec": 0.0,
                    "speed_steps_per_sec": 0.0,
                    "time": time.time(),
                }
            ),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError):
        pass
