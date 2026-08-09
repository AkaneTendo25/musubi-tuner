import sys
from pathlib import Path

from musubi_tuner.gui_dashboard.command_builder import (
    build_cache_latents_cmd,
    build_cache_text_cmd,
    build_inference_cmd,
    build_training_cmd,
)
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.gui_dashboard.validation import validate_inference_config, validate_training_config
from musubi_tuner.minimax_h3_generate_video import create_parser as create_inference_parser
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


def test_dashboard_project_defaults_are_h3_only() -> None:
    config = ProjectConfig()

    assert config.caching.model_type == "minimax_h3"
    assert config.training.model_type == "minimax_h3"
    assert config.inference.model_type == "minimax_h3"


def test_default_h3_workflow_never_builds_ltx_commands(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.inference.model_type = "minimax_h3"
    config.inference.h3_model = config.training.h3_model
    config.inference.prompt = "test prompt"
    config.inference.lora_weight = "output/h3.safetensors"

    commands = (
        build_cache_latents_cmd(config),
        build_cache_text_cmd(config),
        build_training_cmd(config),
        build_inference_cmd(config),
    )

    rendered = [" ".join(command).lower() for command in commands]
    assert "minimax_h3_cache_latents.py" in rendered[0]
    assert "minimax_h3_cache_text_encoder_outputs.py" in rendered[1]
    assert "minimax_h3_train_network.py" in rendered[2]
    assert "minimax_h3_generate_video.py" in rendered[3]
    assert "--text_encoder models/qwen3-vl.safetensors" in rendered[3]
    assert "--tokenizer models/minimax-h3/fl2va/text_encoder" in rendered[3]
    assert "--vae models/h3-video-vae.safetensors" in rendered[3]
    assert "--audio_vae models/h3-audio-vae.safetensors" in rendered[3]
    assert "--lora_weight output/h3.safetensors" in rendered[3]
    assert all("ltx2_" not in command and "gemma" not in command for command in rendered)


def test_training_uses_accelerate_from_dashboard_python(tmp_path: Path) -> None:
    command = build_training_cmd(_h3_config(tmp_path))

    assert command[:3] == [sys.executable, "-m", "accelerate.commands.launch"]


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


def test_h3_native_conditioning_controls_round_trip_through_real_parser(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    training = config.training
    training.h3_training_mode = "ref2va_omni"
    training.h3_observed_modality = "random"
    training.h3_frame_sigma_jitter = 0.1
    training.h3_shift_video = 10.0
    training.h3_shift_audio = 2.5
    training.h3_caption_dropout_rate = 0.2
    training.h3_image_flow_shift = 3.0
    training.h3_extension_video_frames = 2
    training.h3_extension_audio_latents = 4
    training.h3_extension_route = "per_row_sigma"
    training.h3_keyframe_anchors = "first,last"
    training.reference_image_short_edge = 448
    training.h3_mask_mode = "segment"
    training.h3_mask_audio = True
    training.h3_mask_min_fraction = 0.2
    training.h3_mask_max_fraction = 0.6
    training.crepa = True
    training.h3_crepa_teacher_block_idx = 34
    training.crepa_lambda = 0.2

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])

    assert parsed.h3_training_mode == "ref2va_omni"
    assert parsed.h3_observed_modality == "random"
    assert parsed.h3_frame_sigma_jitter == 0.1
    assert parsed.h3_shift_video == 10.0
    assert parsed.h3_shift_audio == 2.5
    assert parsed.h3_caption_dropout_rate == 0.2
    assert parsed.h3_image_flow_shift == 3.0
    assert parsed.h3_extension_video_frames == 2
    assert parsed.h3_extension_audio_latents == 4
    assert parsed.h3_extension_route == "per_row_sigma"
    assert parsed.h3_keyframe_anchors == "first,last"
    assert parsed.reference_image_short_edge == 448
    assert parsed.h3_mask_mode == "segment"
    assert parsed.h3_mask_audio is True
    assert parsed.h3_mask_min_fraction == 0.2
    assert parsed.h3_mask_max_fraction == 0.6
    assert parsed.crepa == ["teacher_block=34", "weight=0.2"]


def test_h3_inference_controls_round_trip_through_real_parser(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    inference = config.inference
    inference.model_type = "minimax_h3"
    inference.h3_model = config.training.h3_model
    inference.prompt = "dashboard parser coverage"
    inference.sample_steps = 28
    inference.width = 1280
    inference.height = 720
    inference.h3_keyframes = "4:one.png 8:two.png"
    inference.h3_reference_image_short_edge = 448
    inference.h3_dtype = "float16"
    inference.h3_int8_convrot_base = True
    inference.h3_blocks_to_swap = 12
    inference.h3_block_swap_h2d_only = True
    inference.h3_block_swap_ring_size = 3
    inference.h3_block_swap_granularity = "layer"
    inference.h3_use_pinned_memory_for_block_swap = True
    inference.h3_compile = True
    inference.h3_compile_dynamic = "auto"
    inference.h3_compile_fullgraph = True
    inference.h3_compile_cache_size_limit = 64
    inference.h3_compile_fallback_to_eager = True
    inference.h3_inductor_config = "max_autotune=true"
    inference.h3_fused_qk_norm_rope = True

    command = build_inference_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_generate_video.py"))
    parsed = create_inference_parser().parse_args(command[script_index + 1 :])

    assert parsed.steps == 28
    assert (parsed.width, parsed.height) == (1280, 720)
    assert parsed.keyframe == ["4:one.png", "8:two.png"]
    assert parsed.reference_image_short_edge == 448
    assert parsed.dtype == "float16"
    assert parsed.int8_convrot_base is True
    assert parsed.blocks_to_swap == 12
    assert parsed.block_swap_h2d_only is True
    assert parsed.block_swap_ring_size == 3
    assert parsed.block_swap_granularity == "layer"
    assert parsed.use_pinned_memory_for_block_swap is True
    assert parsed.compile is True
    assert parsed.compile_dynamic == "auto"
    assert parsed.compile_fullgraph is True
    assert parsed.compile_cache_size_limit == 64
    assert parsed.compile_fallback_to_eager is True
    assert parsed.inductor_config == ["max_autotune=true"]
    assert parsed.h3_fused_qk_norm_rope is True


def test_advanced_cli_escape_hatches_reach_each_h3_process(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.caching.cache_latents_extra_args = "--batch_size 2 --reference_image_short_edge 448"
    config.caching.cache_text_extra_args = "--batch_size 3 --reference_image_short_edge 448"
    config.training.extra_args = "--cuda_allow_tf32 --max_data_loader_n_workers 2"
    config.inference.model_type = "minimax_h3"
    config.inference.h3_model = config.training.h3_model
    config.inference.prompt = "escape hatch"
    config.inference.extra_args = "--inspect"

    assert build_cache_latents_cmd(config)[-4:] == ["--batch_size", "2", "--reference_image_short_edge", "448"]
    assert build_cache_text_cmd(config)[-4:] == ["--batch_size", "3", "--reference_image_short_edge", "448"]
    assert build_training_cmd(config)[-3:] == ["--cuda_allow_tf32", "--max_data_loader_n_workers", "2"]
    assert build_inference_cmd(config)[-1] == "--inspect"


def test_h3_inference_rejects_incompatible_memory_and_canvas_controls(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    inference = config.inference
    inference.model_type = "minimax_h3"
    inference.h3_model = config.training.h3_model
    inference.prompt = "validation"
    inference.width = 1280
    inference.fp8_base = True
    inference.h3_int8_convrot_base = True
    inference.h3_blocks_to_swap = 51
    inference.h3_block_swap_ring_size = 0

    report = validate_inference_config(config)

    assert "inference.width" in report["field_errors"]
    assert "inference.h3_int8_convrot_base" in report["field_errors"]
    assert "inference.h3_blocks_to_swap" in report["field_errors"]
    assert "inference.h3_block_swap_ring_size" in report["field_errors"]


def test_h3_layout_contract_prevents_clipped_errors_and_ltx_routes() -> None:
    frontend = Path(__file__).parents[1] / "src" / "musubi_tuner" / "gui_dashboard" / "frontend" / "src"
    css = (frontend / "app.css").read_text(encoding="utf-8")
    dataset_entry = (frontend / "lib" / "components" / "DatasetEntry.svelte").read_text(encoding="utf-8")
    layout = (frontend / "routes" / "+layout.svelte").read_text(encoding="utf-8")

    error_rule = css.split(".app-field-error {", 1)[1].split("}", 1)[0]
    assert "border: 1px solid" in error_rule
    assert "overflow-wrap: anywhere" in error_rule
    assert 'class="overflow-visible"' in dataset_entry
    for route in (
        "/samples",
        "/settings",
        "/tools",
        "/training/conditioning",
        "/training/full-finetune",
        "/training/rl",
        "/training/techniques",
    ):
        assert f"'{route}'" in layout
