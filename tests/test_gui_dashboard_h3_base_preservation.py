from pathlib import Path

from musubi_tuner.gui_dashboard.command_builder import build_training_cmd
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.gui_dashboard.validation import validate_training_config
from musubi_tuner.minimax_h3_train_network import create_parser


def _h3_config(tmp_path: Path) -> ProjectConfig:
    config = ProjectConfig(project_dir=str(tmp_path))
    config.caching.model_type = "minimax_h3"
    config.caching.h3_video_vae = "models/h3-video-vae.safetensors"
    config.caching.h3_audio_vae = "models/h3-audio-vae.safetensors"
    config.caching.h3_text_encoder = "models/qwen3-vl.safetensors"
    config.caching.h3_tokenizer = "models/MiniMax-H3/FL2VA/text_encoder"
    config.training.model_type = "minimax_h3"
    config.training.h3_model = "models/minimax-h3.safetensors"
    config.training.network_dim = 16
    return config


def test_h3_spatial_density_jitter_defaults_to_point_two(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])

    assert config.training.h3_spatial_density_jitter == 0.2
    assert parsed.h3_spatial_density_jitter == 0.2


def test_h3_spatial_density_jitter_custom_value_is_forwarded(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_spatial_density_jitter = 0.35

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])

    assert parsed.h3_spatial_density_jitter == 0.35


def test_h3_spatial_density_jitter_rejects_negative_value(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_spatial_density_jitter = -0.01

    report = validate_training_config(config)

    assert "training.h3_spatial_density_jitter" in report["field_errors"]


def test_h3_base_preservation_default_is_not_forwarded(tmp_path: Path) -> None:
    command = build_training_cmd(_h3_config(tmp_path))

    assert "--h3_base_preservation_loss_weight" not in command


def test_h3_base_preservation_is_forwarded_to_trainer(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_base_preservation_loss_weight = 0.05

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])

    assert parsed.h3_base_preservation_loss_weight == 0.05


def test_h3_base_preservation_rejects_negative_weight(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_base_preservation_loss_weight = -0.01

    report = validate_training_config(config)

    assert "training.h3_base_preservation_loss_weight" in report["field_errors"]


def test_h3_exact_resume_controls_are_forwarded(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.save_state = True
    config.training.autoresume = True

    command = build_training_cmd(config)

    assert "--save_state" in command
    assert "--autoresume" in command


def test_h3_compile_controls_are_forwarded_and_allowed(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.compile = True
    config.training.compile_auto_cache_size_limit = True
    config.training.compile_fallback_to_eager = True

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])
    report = validate_training_config(config)

    assert parsed.compile is True
    assert parsed.compile_auto_cache_size_limit is True
    assert parsed.compile_fallback_to_eager is True
    assert "training.compile" not in report["field_errors"]


def test_h3_performance_controls_are_forwarded_to_trainer(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    training = config.training
    training.h3_fused_qk_norm_rope = True
    training.h3_attn_auto_dispatch = True
    training.h3_adaln_rank = 16
    training.gradient_checkpointing = True
    training.gradient_checkpointing_cpu_offload = True
    training.h3_gradient_checkpointing_blocks = 50
    training.h3_gradient_checkpointing_cpu_offload_pin_memory = True
    training.blocks_to_swap = 8
    training.block_swap_h2d_only = True
    training.block_swap_granularity = "layer"

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])

    assert parsed.h3_fused_qk_norm_rope is True
    assert parsed.h3_attn_auto_dispatch is True
    assert parsed.h3_adaln_rank == 16
    assert parsed.gradient_checkpointing_cpu_offload is True
    assert parsed.h3_gradient_checkpointing_blocks == 50
    assert parsed.h3_gradient_checkpointing_cpu_offload_pin_memory is True
    assert parsed.block_swap_granularity == "layer"


def test_h3_performance_quantization_controls_are_forwarded(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    training = config.training
    training.h3_convrot_int8 = True
    training.h3_convrot_int8_bwd = "int8"
    training.h3_convrot_int8_lora_fused = True

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])

    assert parsed.h3_convrot_int8 is True
    assert parsed.h3_convrot_int8_fwd == "int8"
    assert parsed.h3_convrot_int8_bwd == "int8"
    assert parsed.h3_convrot_int8_lora_fused is True

    training.h3_convrot_int8 = False
    training.h3_convrot_int8_bwd = "bf16"
    training.h3_convrot_int8_lora_fused = False
    training.fp8_base = True
    training.h3_fp8_quantization_mode = "channel"

    command = build_training_cmd(config)
    parsed = create_parser().parse_args(command[script_index + 1 :])

    assert parsed.fp8_base is True
    assert parsed.h3_fp8_quantization_mode == "channel"


def test_h3_performance_validation_matches_trainer_constraints(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    training = config.training
    training.int8_convrot_base = True
    training.h3_convrot_int8 = True
    training.gradient_checkpointing = True
    training.h3_gradient_checkpointing_blocks = 24
    training.compile = True
    training.h3_gradient_checkpointing_cpu_offload_pin_memory = True

    report = validate_training_config(config)

    assert "training.h3_convrot_int8" in report["field_errors"]
    assert "training.h3_gradient_checkpointing_blocks" in report["field_errors"]
    assert "training.h3_gradient_checkpointing_cpu_offload_pin_memory" in report["field_errors"]
