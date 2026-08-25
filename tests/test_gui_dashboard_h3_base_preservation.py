import sys
from pathlib import Path

import tomllib
import pytest

from musubi_tuner.gui_dashboard.command_builder import (
    build_cache_latents_cmd,
    build_cache_text_cmd,
    build_inference_cmd,
    build_training_cmd,
)
from musubi_tuner.gui_dashboard.project_schema import DatasetEntry, ProjectConfig
from musubi_tuner.gui_dashboard.validation import (
    validate_cache_latents_config,
    validate_cache_text_config,
    validate_inference_config,
    validate_training_config,
)
from musubi_tuner.minimax_h3.dataset import _normalize_explicit_modality_config
from musubi_tuner.minimax_h3_cache_latents import create_parser as create_cache_latents_parser
from musubi_tuner.minimax_h3_cache_text_encoder_outputs import create_parser as create_cache_text_parser
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


def test_dashboard_accepts_all_h3_conditioning_cache_tasks(tmp_path: Path) -> None:
    for task in ("t2va", "i2va", "fl2va", "l2va", "ref2va", "ref2va_omni"):
        config = _h3_config(tmp_path)
        config.caching.h3_task = task
        command = build_cache_text_cmd(config)

        assert command[command.index("--task") + 1] == task


def test_dashboard_rejects_conditioning_cache_from_another_h3_partition(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_training_mode = "ref2va"
    config.caching.h3_task = "fl2va"

    report = validate_training_config(config)

    assert "caching.h3_task" in report["field_errors"]


def test_dashboard_validates_dataloader_transfer_settings(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.max_data_loader_n_workers = -1
    config.training.dataloader_prefetch_factor = 0

    report = validate_training_config(config)

    assert "training.max_data_loader_n_workers" in report["field_errors"]
    assert "training.dataloader_prefetch_factor" in report["field_errors"]


def test_dashboard_warns_about_worker_only_settings_with_no_workers(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.max_data_loader_n_workers = 0
    config.training.dataloader_prefetch_factor = 3
    config.training.persistent_data_loader_workers = True

    report = validate_training_config(config)

    assert "training.dataloader_prefetch_factor" not in report["field_errors"]
    assert "training.persistent_data_loader_workers" not in report["field_errors"]
    assert "training.dataloader_prefetch_factor" in report["field_warnings"]
    assert "training.persistent_data_loader_workers" in report["field_warnings"]


def test_h3_dashboard_dataloader_transfer_settings_round_trip(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.max_data_loader_n_workers = 2
    config.training.persistent_data_loader_workers = True
    config.training.dataloader_pin_memory = True
    config.training.dataloader_prefetch_factor = 3

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])

    assert parsed.max_data_loader_n_workers == 2
    assert parsed.persistent_data_loader_workers is True
    assert parsed.dataloader_pin_memory is True
    assert parsed.dataloader_prefetch_factor == 3


def test_dashboard_reference_size_defaults_match_h3_cli(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)

    assert config.caching.h3_reference_image_short_edge == 2048
    assert config.training.reference_image_short_edge == 2048
    assert config.inference.h3_reference_image_short_edge == 2048
    assert config.caching.h3_reference_video_short_edge == 768
    assert config.training.reference_video_short_edge == 768
    assert config.inference.h3_reference_video_short_edge == 768
    assert config.caching.h3_reference_video_max_pixels == 768 * 1344
    assert config.training.reference_video_max_pixels == 768 * 1344
    assert config.inference.h3_reference_video_max_pixels == 768 * 1344
    assert "--reference_image_short_edge" not in build_cache_latents_cmd(config)
    assert "--reference_image_short_edge" not in build_cache_text_cmd(config)
    assert "--reference_image_short_edge" not in build_training_cmd(config)
    assert "--reference_video_short_edge" not in build_cache_latents_cmd(config)
    assert "--reference_video_short_edge" not in build_cache_text_cmd(config)
    assert "--reference_video_short_edge" not in build_training_cmd(config)


def test_dashboard_rejects_custom_keyframes_with_endpoint_task_cache(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.caching.h3_task = "i2va"
    config.training.h3_keyframe_anchors = "first,last"

    report = validate_training_config(config)

    assert "caching.h3_task" in report["field_errors"]


@pytest.mark.parametrize("task", ["t2va", "ref2va", "ref2va_omni"])
def test_dashboard_accepts_custom_keyframes_with_supported_task_cache(tmp_path: Path, task: str) -> None:
    config = _h3_config(tmp_path)
    config.caching.h3_task = task
    config.training.h3_training_mode = task if task != "t2va" else "fl2va"
    config.training.h3_keyframe_anchors = "first,last"

    report = validate_training_config(config)

    assert "caching.h3_task" not in report["field_errors"]
    assert "training.h3_training_mode" not in report["field_errors"]


def test_dashboard_accepts_valid_ref2va_av_guide_specs(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.caching.h3_task = "ref2va"
    config.training.h3_training_mode = "ref2va"
    config.training.h3_guide_specs = "0:2:4;21:0:8"

    report = validate_training_config(config)

    assert "training.h3_guide_specs" not in report["field_errors"]
    assert "caching.h3_task" not in report["field_errors"]
    assert "--h3_guide_specs" in build_training_cmd(config)


def test_dashboard_rejects_malformed_or_non_ref2va_av_guides(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.caching.h3_task = "t2va"
    config.training.h3_training_mode = "fl2va"
    config.training.h3_guide_specs = "0:0:0"

    report = validate_training_config(config)

    assert "training.h3_guide_specs" in report["field_errors"]
    assert "training.h3_training_mode" in report["field_errors"]
    assert "caching.h3_task" in report["field_errors"]


@pytest.mark.parametrize("mode", ["ref2va", "ref2va_omni"])
def test_dashboard_accepts_ref2va_masking_and_per_row_extension(tmp_path: Path, mode: str) -> None:
    config = _h3_config(tmp_path)
    config.caching.h3_task = mode
    config.training.h3_training_mode = mode
    config.training.h3_extension_video_frames = 1
    config.training.h3_extension_route = "per_row_sigma"

    report = validate_training_config(config)

    assert "training.h3_training_mode" not in report["field_errors"]
    assert "training.h3_extension_route" not in report["field_errors"]
    assert "caching.h3_task" not in report["field_errors"]

    config.training.h3_extension_video_frames = 0
    config.training.h3_mask_mode = "box"
    report = validate_training_config(config)
    assert "training.h3_training_mode" not in report["field_errors"]
    assert "caching.h3_task" not in report["field_errors"]


def test_dashboard_rejects_ref2va_condition_row_extension(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.caching.h3_task = "ref2va"
    config.training.h3_training_mode = "ref2va"
    config.training.h3_extension_video_frames = 1
    config.training.h3_extension_route = "condition_rows"

    report = validate_training_config(config)

    assert "training.h3_extension_route" in report["field_errors"]


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


def test_h3_spatial_density_jitter_defaults_to_off(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])

    assert config.training.h3_spatial_density_jitter == 0.0
    assert parsed.h3_spatial_density_jitter == 0.0


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


def test_h3_modality_loss_weights_must_be_nonnegative_and_not_both_zero(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_audio_loss_weight = -0.1

    report = validate_training_config(config)
    assert "training.h3_audio_loss_weight" in report["field_errors"]

    config.training.h3_audio_loss_weight = 0.0
    config.training.h3_video_loss_weight = 0.0
    report = validate_training_config(config)
    assert "training.h3_video_loss_weight" in report["field_errors"]


def test_h3_modality_loss_weights_cover_observed_and_dataset_targets(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_observed_modality = "video"
    config.training.h3_audio_loss_weight = 0.0

    report = validate_training_config(config)
    assert "training.h3_audio_loss_weight" in report["field_errors"]

    config.training.h3_observed_modality = None
    config.training.h3_audio_loss_weight = 1.0
    config.training.h3_video_loss_weight = 0.0
    config.dataset.datasets = [DatasetEntry(type="image", directory=str(tmp_path / "images"))]
    report = validate_training_config(config)
    assert "dataset.datasets.0.h3_target_modalities" in report["field_errors"]


def test_h3_base_preservation_default_is_not_forwarded(tmp_path: Path) -> None:
    command = build_training_cmd(_h3_config(tmp_path))

    assert "--h3_base_preservation_loss_weight" not in command


def test_h3_base_preservation_is_forwarded_to_trainer(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_base_preservation_loss_weight = 0.05
    config.training.h3_base_preservation_probability = 0.25

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])

    assert parsed.h3_base_preservation_loss_weight == 0.05
    assert parsed.h3_base_preservation_probability == 0.25


def test_h3_base_preservation_rejects_negative_weight(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_base_preservation_loss_weight = -0.01

    report = validate_training_config(config)

    assert "training.h3_base_preservation_loss_weight" in report["field_errors"]


def test_h3_base_preservation_rejects_invalid_probability(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_base_preservation_probability = 0.0

    report = validate_training_config(config)

    assert "training.h3_base_preservation_probability" in report["field_errors"]


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
    training.h3_fused_indexed_adaln = True
    training.h3_fused_swiglu = True
    training.h3_attn_auto_dispatch = True
    training.h3_int8_attention = "train"
    training.h3_lora_token_refiner = True
    training.h3_adaln_rank = 16
    training.gradient_checkpointing = True
    training.gradient_checkpointing_cpu_offload = True
    training.h3_gradient_checkpointing_blocks = 50
    training.h3_gradient_checkpointing_cpu_offload_pin_memory = True
    training.h3_reusable_activation_offload = True
    training.blocks_to_swap = 8
    training.block_swap_h2d_only = True
    training.block_swap_granularity = "layer"

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])

    assert parsed.h3_fused_qk_norm_rope is True
    assert parsed.h3_fused_indexed_adaln is True
    assert parsed.h3_fused_swiglu is True
    assert parsed.h3_attn_auto_dispatch is True
    assert parsed.h3_int8_attention == "train"
    assert parsed.h3_lora_token_refiner is True
    assert parsed.h3_adaln_rank == 16
    assert parsed.gradient_checkpointing_cpu_offload is True
    assert parsed.h3_gradient_checkpointing_blocks == 50
    assert parsed.h3_gradient_checkpointing_cpu_offload_pin_memory is True
    assert parsed.h3_reusable_activation_offload is True
    assert parsed.block_swap_granularity == "layer"


def test_h3_int8_attention_rejects_compile_in_dashboard(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_int8_attention = "train"
    config.training.compile = True

    report = validate_training_config(config)

    assert "training.h3_int8_attention" in report["field_errors"]


def test_h3_reusable_activation_offload_requires_cpu_checkpoint_offload(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_reusable_activation_offload = True

    report = validate_training_config(config)

    assert "training.h3_reusable_activation_offload" in report["field_errors"]


def test_h3_ref2va_training_sampling_is_rejected_by_dashboard(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.caching.h3_task = "ref2va"
    config.training.h3_training_mode = "ref2va"
    config.training.sample_prompts_text = "a test prompt"

    report = validate_training_config(config)

    assert "training.sample_prompts" in report["field_errors"]


def test_h3_frame_sigma_jitter_rejects_out_of_range_and_conditioning(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_frame_sigma_jitter = 1.01
    report = validate_training_config(config)
    assert "training.h3_frame_sigma_jitter" in report["field_errors"]

    config.training.h3_frame_sigma_jitter = 0.1
    config.training.h3_observed_modality = "video"
    report = validate_training_config(config)
    assert "training.h3_frame_sigma_jitter" in report["field_errors"]


def test_h3_timestep_focus_round_trips_and_validates(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    training = config.training
    training.h3_timestep_focus_probability = 0.25
    training.h3_timestep_focus_min = 0.4
    training.h3_timestep_focus_max = 0.8

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])
    assert parsed.h3_timestep_focus_probability == 0.25
    assert parsed.h3_timestep_focus_min == 0.4
    assert parsed.h3_timestep_focus_max == 0.8

    training.h3_timestep_sampling = "logsnr"
    report = validate_training_config(config)
    assert "training.h3_timestep_sampling" in report["field_errors"]

    training.h3_timestep_focus_probability = 0
    training.h3_timestep_sampling = "sigma"
    training.num_timestep_buckets = 8
    report = validate_training_config(config)
    assert "training.num_timestep_buckets" in report["field_errors"]


def test_h3_conditioning_modes_reject_conflicts_and_wrong_cache(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.caching.h3_task = "i2va"
    config.training.h3_extension_video_frames = 1
    config.training.h3_mask_mode = "box"

    report = validate_training_config(config)

    assert "training.h3_extension_video_frames" in report["field_errors"]
    assert "caching.h3_task" in report["field_errors"]


def test_h3_mask_minimum_must_be_positive(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_mask_mode = "box"
    config.training.h3_mask_min_fraction = 0.0

    report = validate_training_config(config)

    assert "training.h3_mask_min_fraction" in report["field_errors"]


def test_h3_audio_only_masking_round_trips_without_video_mask_mode(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_mask_audio = True
    config.training.h3_mask_min_fraction = 0.2
    config.training.h3_mask_max_fraction = 0.6

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])

    assert parsed.h3_mask_mode == "off"
    assert parsed.h3_mask_audio is True
    assert parsed.h3_mask_min_fraction == 0.2
    assert parsed.h3_mask_max_fraction == 0.6


def test_h3_paired_reference_directories_must_be_complete(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.dataset.datasets = [
        DatasetEntry(
            type="image",
            directory="train",
            source_video_directory="reference_video",
            source_video_audio_paired=True,
        )
    ]

    report = validate_cache_latents_config(config)

    assert "dataset.datasets[0].source_video_audio_paired" in report["field_errors"]


def test_h3_dataset_frame_grid_is_validated_for_video_and_audio(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.dataset.datasets = [
        DatasetEntry(type="video", directory="video", target_frames=33),
        DatasetEntry(type="audio", directory="audio", target_frames=34),
    ]

    report = validate_cache_latents_config(config)

    assert "dataset.datasets[0].target_frames" in report["field_errors"]
    assert "dataset.datasets[1].target_frames" in report["field_errors"]


def test_h3_audio_dashboard_row_forces_audio_target_and_exports_timeline(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.caching.h3_video_vae = ""
    config.dataset.datasets = [DatasetEntry(type="audio", directory="audio", target_frames=124)]

    report = validate_cache_latents_config(config)
    command = build_cache_latents_cmd(config)
    document = tomllib.loads(Path(command[command.index("--dataset_config") + 1]).read_text(encoding="utf-8"))

    assert "caching.h3_video_vae" not in report["field_errors"]
    assert "dataset.datasets[0].h3_target_modalities" not in report["field_errors"]
    assert document["datasets"][0]["target_modalities"] == ["audio"]
    assert document["datasets"][0]["target_frames"] == [124]
    _normalize_explicit_modality_config(document)


def test_h3_explicit_source_directories_control_required_vaes(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.caching.h3_audio_vae = ""
    config.dataset.datasets = [DatasetEntry(type="image", directory="train", source_image_directory="references")]

    assert "caching.h3_audio_vae" not in validate_cache_latents_config(config)["field_errors"]

    config.dataset.datasets[0] = DatasetEntry(
        type="image",
        directory="train",
        source_video_directory="references",
        source_video_audio_embedded=True,
    )
    assert "caching.h3_audio_vae" in validate_cache_latents_config(config)["field_errors"]


def test_h3_paired_source_allows_an_additional_image_source(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.dataset.datasets = [
        DatasetEntry(
            type="image",
            directory="train",
            source_image_directory="reference_images",
            source_video_directory="reference_videos",
            source_audio_directory="reference_audio",
            source_video_audio_paired=True,
        )
    ]

    report = validate_cache_latents_config(config)

    assert "dataset.datasets[0].source_video_audio_paired" not in report["field_errors"]


def test_h3_source_probability_validation_rejects_impossible_variants(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.dataset.datasets = [
        DatasetEntry(
            type="image",
            directory="train-audio",
            source_audio_directory="reference_audio",
            control_modality_probability_av=0.5,
            control_modality_probability_video=0.25,
            control_modality_probability_audio=0.25,
        ),
        DatasetEntry(
            type="image",
            directory="train-video",
            source_video_directory="reference_video",
            control_modality_probability_av=0.5,
            control_modality_probability_video=0.25,
            control_modality_probability_audio=0.25,
        ),
    ]

    report = validate_cache_latents_config(config)

    assert "dataset.datasets[0].control_modality_probability_video" in report["field_errors"]
    assert "dataset.datasets[1].control_modality_probability_audio" in report["field_errors"]


def test_dashboard_migrates_unambiguous_h3_project_dataset_fields() -> None:
    config = ProjectConfig.model_validate(
        {
            "version": 3,
            "dataset": {
                "datasets": [
                    {
                        "type": "video",
                        "h3_target_mode": "video",
                        "control_video_directory": "reference_video",
                        "control_audio_directory": "reference_audio",
                    }
                ]
            },
        }
    )

    entry = config.dataset.datasets[0]
    assert config.version == 4
    assert entry.h3_target_modalities == "video"
    assert entry.source_video_directory == "reference_video"
    assert entry.source_audio_directory == "reference_audio"
    assert entry.source_video_audio_paired is True


def test_h3_training_sample_text_visual_limit_round_trips(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.caching.h3_text_visual_max_pixels = 1_048_576
    config.training.sample_prompts = "sample_prompts.txt"

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])

    assert parsed.h3_text_visual_max_pixels == 1_048_576


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


def test_h3_prequantized_convrot_controls_are_forwarded_without_reapplying_adaln(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    training = config.training
    training.int8_convrot_base = True
    training.h3_adaln_rank = 16
    training.h3_convrot_int8_fwd = "bf16"
    training.h3_convrot_int8_bwd = "bf16"

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])
    report = validate_training_config(config)

    assert parsed.int8_convrot_base is True
    assert parsed.h3_adaln_rank is None
    assert parsed.h3_convrot_int8_fwd == "bf16"
    assert "training.h3_convrot_int8_fwd" not in report["field_errors"]


def test_h3_performance_validation_matches_trainer_constraints(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    training = config.training
    training.int8_convrot_base = True
    training.h3_convrot_int8 = True
    training.gradient_checkpointing = True
    training.h3_gradient_checkpointing_blocks = 24
    training.blocks_to_swap = 10
    training.compile = True
    training.h3_gradient_checkpointing_cpu_offload_pin_memory = True

    report = validate_training_config(config)

    assert "training.h3_convrot_int8" in report["field_errors"]
    assert "training.h3_gradient_checkpointing_blocks" in report["field_errors"]
    assert "training.h3_gradient_checkpointing_cpu_offload_pin_memory" in report["field_errors"]


def test_h3_dashboard_rejects_nonpositive_sigma_sqrt_cap(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_sigma_sqrt_max_weight = 0

    report = validate_training_config(config)

    assert "training.h3_sigma_sqrt_max_weight" in report["field_errors"]


def test_h3_native_conditioning_controls_round_trip_through_real_parser(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    training = config.training
    training.h3_training_mode = "ref2va_omni"
    training.h3_observed_modality = "random"
    training.h3_frame_sigma_jitter = 0.1
    training.h3_shift_video = 10.0
    training.h3_shift_audio = 2.5
    training.h3_sigma_sqrt_max_weight = 8.0
    training.h3_caption_dropout_rate = 0.2
    training.h3_image_flow_shift = 3.0
    training.h3_extension_video_frames = 2
    training.h3_extension_audio_latents = 4
    training.h3_extension_route = "per_row_sigma"
    training.h3_keyframe_anchors = "first,last"
    training.h3_guide_specs = "0:2:4"
    training.reference_image_short_edge = 448
    training.reference_video_short_edge = 384
    training.reference_video_max_pixels = 384 * 672
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
    assert parsed.h3_sigma_sqrt_max_weight == 8.0
    assert parsed.h3_caption_dropout_rate == 0.2
    assert parsed.h3_image_flow_shift == 3.0
    assert parsed.h3_extension_video_frames == 2
    assert parsed.h3_extension_audio_latents == 4
    assert parsed.h3_extension_route == "per_row_sigma"
    assert parsed.h3_keyframe_anchors == "first,last"
    assert parsed.h3_guide_specs == "0:2:4"
    assert parsed.reference_image_short_edge == 448
    assert parsed.reference_video_short_edge == 384
    assert parsed.reference_video_max_pixels == 384 * 672
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
    inference.h3_reference_video_short_edge = 384
    inference.h3_reference_video_max_pixels = 384 * 672
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
    assert parsed.reference_video_short_edge == 384
    assert parsed.reference_video_max_pixels == 384 * 672
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


def test_h3_dashboard_emits_visible_attention_schedule_and_network_controls(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    training = config.training
    training.flash_attn = True
    training.h3_timestep_sampling = "logsnr"
    training.weighting_scheme = "cosmap"
    training.min_timestep = 100
    training.max_timestep = 900
    training.preserve_distribution_shape = True
    training.num_timestep_buckets = 8
    training.lr_warmup_steps = 32
    training.lr_scheduler_num_cycles = 2
    training.network_dropout = 0.1
    training.rank_dropout = 0.2
    training.module_dropout = 0.3
    training.network_args = "include_patterns=['.*to_v.*']"
    training.save_precision = "bf16"
    training.log_grad_metrics = True

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])

    assert parsed.flash_attn is True
    assert parsed.sdpa is False
    assert parsed.timestep_sampling == "logsnr"
    assert parsed.weighting_scheme == "cosmap"
    assert (parsed.min_timestep, parsed.max_timestep) == (100, 900)
    assert parsed.preserve_distribution_shape is True
    assert parsed.num_timestep_buckets == 8
    assert parsed.lr_warmup_steps == 32
    assert parsed.lr_scheduler_num_cycles == 2
    assert parsed.network_dropout == 0.1
    assert "rank_dropout=0.2" in parsed.network_args
    assert "module_dropout=0.3" in parsed.network_args
    assert parsed.save_precision == "bf16"
    assert parsed.log_grad_metrics is True


def test_h3_dashboard_rejects_multiple_attention_backends(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.sdpa = True
    config.training.flash_attn = True

    report = validate_training_config(config)

    assert "training.flash_attn" in report["field_errors"]


def test_h3_cache_toml_contains_training_and_validation_rows_under_datasets(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.dataset.datasets = [
        DatasetEntry(type="image", directory="train", source_video_directory="train_refs", h3_target_modalities="video")
    ]
    config.dataset.validation_datasets = [
        DatasetEntry(
            type="audio",
            directory="validation",
            target_frames=124,
            h3_target_modalities="audio",
            cache_directory="validation_cache",
        )
    ]

    command = build_cache_latents_cmd(config)
    dataset_path = Path(command[command.index("--dataset_config") + 1])
    document = tomllib.loads(dataset_path.read_text(encoding="utf-8"))

    assert "validation_datasets" not in document
    assert len(document["datasets"]) == 2
    assert document["datasets"][0]["source_video_directory"] == "train_refs"
    assert document["datasets"][1]["target_modalities"] == ["audio"]
    assert document["datasets"][1]["target_frames"] == [124]
    _normalize_explicit_modality_config(document)
    legacy_keys = {
        "video_directory",
        "image_directory",
        "audio_directory",
        "control_directory",
        "control_video_directory",
        "control_audio_directory",
        "control_modality",
        "control_modalities",
        "control_modality_probabilities",
        "h3_target_mode",
    }
    assert not any(legacy_keys.intersection(dataset) for dataset in document["datasets"])


def test_h3_dashboard_exports_source_modality_probabilities(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.dataset.datasets = [
        DatasetEntry(
            type="video",
            directory="train",
            target_frames=124,
            source_image_directory="references",
            control_modality_probability_av=0.5,
            control_modality_probability_video=0.25,
            control_modality_probability_audio=0.25,
        )
    ]

    command = build_cache_latents_cmd(config)
    document = tomllib.loads(Path(command[command.index("--dataset_config") + 1]).read_text(encoding="utf-8"))
    dataset = document["datasets"][0]

    assert dataset["source_modality_probabilities"] == [0.5, 0.25, 0.25]
    assert "control_modality_probabilities" not in dataset
    _normalize_explicit_modality_config(document)


def test_h3_dashboard_rejects_removed_fixed_control_modality_fields(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.dataset.datasets = [DatasetEntry(type="video", directory="train", target_frames=124, control_modality="video")]

    report = validate_cache_latents_config(config)

    assert "dataset.datasets[0].control_modalities" in report["field_errors"]


def test_h3_validation_uses_separate_dataset_toml_and_full_cli(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.dataset.datasets = [DatasetEntry(type="video", directory="train", target_frames=124)]
    config.dataset.validation_datasets = [DatasetEntry(type="video", directory="validation", target_frames=124)]
    training = config.training
    training.validate_at_start = True
    training.validate_every_n_steps = 25
    training.validation_seed = 123
    training.validation_timestep_bins = 6
    training.validation_min_timestep = 100
    training.validation_max_timestep = 900
    training.max_validation_items = 8

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])
    training_doc = tomllib.loads(Path(parsed.dataset_config).read_text(encoding="utf-8"))
    validation_doc = tomllib.loads(Path(parsed.validation_dataset_config).read_text(encoding="utf-8"))

    assert len(training_doc["datasets"]) == 1
    assert training_doc["datasets"][0]["target_video_directory"] == "train"
    assert training_doc["datasets"][0]["target_modalities"] == ["video", "audio"]
    assert len(validation_doc["datasets"]) == 1
    assert validation_doc["datasets"][0]["target_video_directory"] == "validation"
    assert validation_doc["datasets"][0]["target_modalities"] == ["video", "audio"]
    _normalize_explicit_modality_config(training_doc)
    _normalize_explicit_modality_config(validation_doc)
    assert parsed.validate_at_start is True
    assert parsed.validate_every_n_steps == 25
    assert parsed.validation_seed == 123
    assert parsed.validation_timestep_bins == 6
    assert (parsed.validation_min_timestep, parsed.validation_max_timestep) == (100, 900)
    assert parsed.max_validation_items == 8
    assert validate_training_config(config)["ok"] is True


def test_h3_cache_streaming_and_conditioned_image_controls_round_trip(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.dataset.datasets = [DatasetEntry(type="image", directory="images", h3_image_frame_count=22)]
    caching = config.caching
    caching.h3_task = "fl2va"
    caching.h3_image_mode = "first"
    caching.h3_image_frame_count = 22
    caching.h3_text_visual_max_pixels = 1_048_576
    caching.h3_text_encoder_quantization = "nvfp4_awq"
    caching.h3_text_encoder_blocks_to_stream = 40
    caching.h3_nvfp4_scaled_mm = True

    latent_command = build_cache_latents_cmd(config)
    latent_script = next(index for index, value in enumerate(latent_command) if value.endswith("minimax_h3_cache_latents.py"))
    latent_args = create_cache_latents_parser().parse_args(latent_command[latent_script + 1 :])
    text_command = build_cache_text_cmd(config)
    text_script = next(
        index for index, value in enumerate(text_command) if value.endswith("minimax_h3_cache_text_encoder_outputs.py")
    )
    text_args = create_cache_text_parser().parse_args(text_command[text_script + 1 :])

    assert (latent_args.h3_image_mode, latent_args.h3_image_frame_count) == ("first", 22)
    assert text_args.h3_image_mode == "first"
    assert text_args.h3_text_encoder_blocks_to_stream == 40
    assert text_args.h3_nvfp4_scaled_mm is True
    assert text_args.h3_text_visual_max_pixels == 1_048_576
    assert validate_cache_latents_config(config)["ok"] is True
    assert validate_cache_text_config(config)["ok"] is True


def test_h3_image_inference_uses_image_output_and_does_not_require_audio_vae(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.caching.h3_audio_vae = ""
    inference = config.inference
    inference.model_type = "minimax_h3"
    inference.h3_model = config.training.h3_model
    inference.prompt = "still image"
    inference.output_name = "still"
    inference.h3_image_mode = "first"
    inference.h3_first_frame = "first.png"
    inference.h3_image_frame_count = 22
    inference.h3_select_frame = 5
    inference.h3_text_encoder_quantization = "nvfp4_awq"
    inference.h3_text_encoder_blocks_to_stream = 30
    inference.h3_nvfp4_scaled_mm = True

    command = build_inference_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_generate_video.py"))
    parsed = create_inference_parser().parse_args(command[script_index + 1 :])

    assert parsed.output.suffix == ".png"
    assert parsed.audio_vae is None
    assert parsed.h3_image_mode == "first"
    assert parsed.h3_image_frame_count == 22
    assert parsed.h3_select_frame == 5
    assert parsed.h3_text_encoder_blocks_to_stream == 30
    assert parsed.h3_nvfp4_scaled_mm is True
    assert validate_inference_config(config)["ok"] is True


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


def test_h3_dashboard_guidance_null_field_options_round_trip(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.dataset.datasets = [DatasetEntry(type="video", directory="train", target_frames=124)]
    config.caching.h3_cache_guidance_empty = True
    config.training.h3_guidance_distillation_scale = 4.0
    config.training.h3_guidance_null_source = "frozen"
    config.training.h3_guidance_cfg_zero = True

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])

    assert parsed.h3_guidance_null_source == "frozen"
    assert parsed.h3_guidance_cfg_zero is True
    assert validate_training_config(config)["field_errors"] == {}


def test_h3_dashboard_guidance_null_field_defaults_stay_off_the_command_line(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.caching.h3_cache_guidance_empty = True
    config.training.h3_guidance_distillation_scale = 4.0

    command = build_training_cmd(config)

    assert "--h3_guidance_null_source" not in command
    assert "--h3_guidance_cfg_zero" not in command
    assert "--h3_fuse_frozen_teachers" not in command


def test_h3_dashboard_fused_frozen_teachers_round_trip(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.dataset.datasets = [DatasetEntry(type="video", directory="train", target_frames=124)]
    config.caching.h3_cache_guidance_empty = True
    config.training.h3_guidance_distillation_scale = 4.0
    config.training.h3_guidance_null_source = "frozen"
    config.training.h3_base_preservation_loss_weight = 0.02
    config.training.h3_fuse_frozen_teachers = True

    command = build_training_cmd(config)
    script_index = next(index for index, value in enumerate(command) if value.endswith("minimax_h3_train_network.py"))
    parsed = create_parser().parse_args(command[script_index + 1 :])

    assert parsed.h3_fuse_frozen_teachers is True
    assert validate_training_config(config)["field_errors"] == {}


def test_h3_dashboard_rejects_fused_teachers_without_both_frozen_objectives(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_fuse_frozen_teachers = True

    report = validate_training_config(config)

    assert "training.h3_fuse_frozen_teachers" in report["field_errors"]


def test_h3_dashboard_rejects_null_field_options_without_a_guidance_scale(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.h3_guidance_null_source = "frozen"
    config.training.h3_guidance_cfg_zero = True

    report = validate_training_config(config)

    assert "training.h3_guidance_null_source" in report["field_errors"]
    assert "training.h3_guidance_cfg_zero" in report["field_errors"]


def test_h3_dashboard_training_page_exposes_the_null_field_controls() -> None:
    page = (
        Path(__file__).parents[1]
        / "src"
        / "musubi_tuner"
        / "gui_dashboard"
        / "frontend"
        / "src"
        / "routes"
        / "training"
        / "+page.svelte"
    ).read_text(encoding="utf-8")

    assert "training.h3_guidance_null_source" in page
    assert "training.h3_guidance_cfg_zero" in page
    assert "training.h3_fuse_frozen_teachers" in page


def test_dashboard_warns_that_automagic3_ignores_the_lr_schedule(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.optimizer_type = "Automagic3"
    config.training.lr_scheduler = "cosine"

    report = validate_training_config(config)

    assert "training.lr_scheduler" in report["field_warnings"]


def test_dashboard_accepts_automagic3_with_a_constant_schedule(tmp_path: Path) -> None:
    config = _h3_config(tmp_path)
    config.training.optimizer_type = "Automagic3"
    config.training.lr_scheduler = "constant"

    report = validate_training_config(config)

    assert "training.lr_scheduler" not in report["field_warnings"]
