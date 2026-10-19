from pathlib import Path

import pytest

from musubi_tuner.gui_dashboard.command_builder import build_training_cmd
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.gui_dashboard.validation import validate_training_config
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser


NEW_FIELDS = {
    "h3_field_preservation_weight": 0.2,
    "h3_field_preservation_steps": 8,
    "h3_projection_scale": 1.5,
    "h3_repair_reference": "caption",
    "h3_repair_scale": "1.5",
    "h3_repair_reference_weight": 0.1,
    "h3_repair_null_anchor_weight": 0.2,
    "h3_repair_preservation_probability": 0.25,
    "h3_repair_caption_anchors": 3,
    "h3_repair_caption_probability": 0.5,
    "h3_repair_sigma_weight": "1,2",
    "h3_repair_target_multiplier": "1,1,2",
    "h3_reamplify_scale": "2,3",
    "h3_reamplify_after": 10,
    "h3_reamplify_probability": 0.5,
    "h3_rollout_inactive_supervision": "teacher_match",
    "h3_rollout_inactive_weight": 0.7,
    "h3_rollout_inactive_probability": 0.6,
    "h3_teacher_ema_decay": 0.9,
    "h3_rollout_teacher_match_sigma_max": 0.7,
    "h3_rollout_teacher_match_dc_weight": 0.8,
    "h3_rollout_teacher_match_mag_weight": 0.9,
    "h3_rollout_teacher_match_preservation_weight": 1.2,
    "h3_base_weights_runtime_attach": True,
    "h3_sparse_router": "router.pt",
    "h3_sparse_exact_blocks": 4,
    "h3_sparse_temporal_radius": 1,
    "h3_sparse_data_path": "mapped_int64",
    "h3_sparse_teacher_loss_weight": 0.3,
    "h3_sparse_teacher_only": True,
    "h3_sparse_teacher_eval_only": True,
    "h3_sparse_teacher_budgets": "2,4",
    "h3_sparse_teacher_budget_offset": 1,
}


def _config(tmp_path: Path) -> ProjectConfig:
    config = ProjectConfig(project_dir=str(tmp_path))
    config.training.model_type = "minimax_h3"
    config.training.h3_model = "h3.safetensors"
    config.training.network_dim = 2
    config.caching.h3_task = "ref2va"
    return config


def _trainer_argv(command: list[str]) -> list[str]:
    script = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    return command[script + 1 :]


def test_new_h3_backend_fields_roundtrip_and_parse_actual_cli(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.training.base_weights = "base.safetensors"
    config.training.h3_training_mode = "ref2va"
    for field, value in NEW_FIELDS.items():
        setattr(config.training, field, value)
    restored = ProjectConfig.model_validate_json(config.model_dump_json())
    for field, value in NEW_FIELDS.items():
        assert getattr(restored.training, field) == value
    parsed = create_parser().parse_args(_trainer_argv(build_training_cmd(restored)))
    for field, value in NEW_FIELDS.items():
        actual = getattr(parsed, field)
        assert str(actual) == value if field == "h3_sparse_router" else actual == value


def test_new_h3_defaults_emit_no_flags(tmp_path: Path) -> None:
    command = build_training_cmd(_config(tmp_path))
    assert not {f"--{field}" for field in NEW_FIELDS}.intersection(command)


@pytest.mark.parametrize(
    ("updates", "field"),
    [
        ({"h3_repair_scale": "2"}, "h3_repair_reference"),
        ({"h3_repair_reference": "empty", "h3_repair_scale": "2", "h3_repair_reference_weight": 0.2}, "h3_repair_reference_weight"),
        ({"h3_field_preservation_steps": 2}, "h3_field_preservation_steps"),
        ({"h3_projection_scale": -1.0}, "h3_projection_scale"),
        ({"h3_reamplify_scale": "2"}, "h3_reamplify_after"),
        ({"h3_rollout_inactive_weight": 0.5}, "h3_rollout_inactive_supervision"),
        ({"h3_rollout_inactive_supervision": "teacher_gap"}, "h3_rollout_inactive_supervision"),
        ({"h3_teacher_ema_decay": 0.9}, "h3_teacher_ema_decay"),
        ({"h3_base_weights_runtime_attach": True}, "h3_base_weights_runtime_attach"),
        (
            {
                "h3_base_weights_runtime_attach": True,
                "base_weights": "base.safetensors",
                "base_weights_multiplier": "1 nan",
            },
            "base_weights_multiplier",
        ),
        ({"h3_sparse_teacher_loss_weight": 0.2}, "h3_sparse_teacher_loss_weight"),
        ({"h3_sparse_teacher_eval_only": True}, "h3_sparse_teacher_eval_only"),
        ({"h3_sparse_teacher_loss_weight": -0.2}, "h3_sparse_teacher_loss_weight"),
        ({"h3_sparse_teacher_budgets": "bad", "h3_sparse_teacher_loss_weight": 0.2}, "h3_sparse_teacher_budgets"),
        ({"h3_reamplify_scale": "1", "h3_reamplify_after": 2}, "h3_reamplify_scale"),
        ({"h3_repair_reference": "caption", "h3_repair_scale": "2", "h3_rollout_supervision": True}, "h3_repair_reference"),
        ({"h3_repair_reference": "caption", "h3_repair_scale": "broken"}, "h3_repair_scale"),
        (
            {"h3_repair_reference": "caption", "h3_repair_scale": "2", "network_args": '"dropout=0.1"'},
            "h3_repair_reference",
        ),
        (
            {"h3_repair_reference": "caption", "h3_repair_scale": "2", "network_args": "'dropout=0.1'"},
            "h3_repair_reference",
        ),
        ({"h3_projection_scale": 2, "h3_rollout_supervision": True}, "h3_projection_scale"),
        (
            {
                "h3_rollout_supervision": True,
                "h3_rollout_inactive_supervision": "teacher_match",
                "h3_rollout_teacher_match_sigma_max": -1,
            },
            "h3_rollout_teacher_match_sigma_max",
        ),
        (
            {
                "h3_rollout_supervision": True,
                "h3_rollout_inactive_supervision": "teacher_match",
                "h3_guidance_distillation_scale": 2,
            },
            "h3_rollout_inactive_supervision",
        ),
        (
            {
                "h3_rollout_supervision": True,
                "h3_rollout_inactive_supervision": "teacher_gap",
                "h3_adapter_prompt_only": True,
            },
            "h3_rollout_inactive_supervision",
        ),
        (
            {
                "h3_rollout_supervision": True,
                "h3_rollout_inactive_supervision": "teacher_gap",
                "h3_teacher_ema_decay": 0.9,
                "network_module": "custom.network",
            },
            "h3_teacher_ema_decay",
        ),
        ({"h3_training_type": "learned_context", "h3_field_preservation_weight": 0.2}, "h3_training_type"),
        (
            {
                "h3_field_preservation_weight": 0.2,
                "blocks_to_swap": 1,
                "h3_fused_backward_pass": True,
            },
            "h3_fused_backward_pass",
        ),
    ],
)
def test_new_h3_dependency_validation(tmp_path: Path, updates: dict, field: str) -> None:
    config = _config(tmp_path)
    for name, value in updates.items():
        setattr(config.training, name, value)
    assert f"training.{field}" in validate_training_config(config)["field_errors"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--sdpa", "--h3_repair_reference", "empty"],
        ["--sdpa", "--h3_sparse_teacher_eval_only"],
    ],
)
def test_actual_h3_trainer_rejects_gui_dependency_examples(argv: list[str]) -> None:
    args = create_parser().parse_args(argv)
    with pytest.raises(ValueError):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_runtime_base_multiplier_accepts_one_quoted_numeric_value(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.training.h3_base_weights_runtime_attach = True
    config.training.base_weights = "base.safetensors"
    config.training.base_weights_multiplier = "'0.5'"
    assert "training.base_weights_multiplier" not in validate_training_config(config)["field_errors"]
