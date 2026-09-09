from pathlib import Path

from musubi_tuner.gui_dashboard.command_builder import build_training_cmd
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.gui_dashboard.validation import validate_training_config


def _config(tmp_path: Path) -> ProjectConfig:
    config = ProjectConfig(project_dir=str(tmp_path))
    config.training.model_type = "minimax_h3"
    config.training.h3_model = "models/minimax-h3.safetensors"
    config.training.network_dim = 16
    return config


def test_overlay_training_only_is_disabled_by_default(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert config.training.h3_overlay_training_only is False
    assert "--h3_overlay_training_only" not in build_training_cmd(config)


def test_overlay_training_only_is_emitted_with_overlay_weights(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.training.h3_overlay_weights = "models/frozen-overlay.safetensors"
    config.training.h3_overlay_training_only = True

    command = build_training_cmd(config)

    assert "--h3_overlay_weights" in command
    assert "--h3_overlay_training_only" in command
    report = validate_training_config(config)
    assert "training.h3_overlay_training_only" not in report["field_errors"]


def test_overlay_training_only_requires_overlay_weights(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.training.h3_overlay_training_only = True

    report = validate_training_config(config)

    assert "training.h3_overlay_training_only" in report["field_errors"]


def test_overlay_training_only_rejects_preview_validation_probes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.training.h3_overlay_weights = "models/frozen-overlay.safetensors"
    config.training.h3_overlay_training_only = True

    config.training.h3_validation_field_probe = True
    field_report = validate_training_config(config)
    assert "training.h3_overlay_training_only" in field_report["field_errors"]

    config.training.h3_validation_field_probe = False
    config.training.h3_validation_rollout_probe = 8
    rollout_report = validate_training_config(config)
    assert "training.h3_overlay_training_only" in rollout_report["field_errors"]
