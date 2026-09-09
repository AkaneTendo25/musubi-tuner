from pathlib import Path

from musubi_tuner.gui_dashboard.command_builder import (
    build_cache_latents_cmd,
    build_cache_text_cmd,
    build_inference_cmd,
    build_training_cmd,
)
from musubi_tuner.gui_dashboard.project_schema import DatasetEntry, ProjectConfig
from musubi_tuner.gui_dashboard.toml_export import build_h3_dataset_toml_document
from musubi_tuner.minimax_h3_generate_video import create_parser as create_inference_parser


def _config(tmp_path: Path) -> ProjectConfig:
    config = ProjectConfig(project_dir=str(tmp_path))
    config.dataset.datasets = [
        DatasetEntry(
            type="image",
            directory=str(tmp_path / "targets"),
            cache_directory=str(tmp_path / "cache"),
            source_image_directory=str(tmp_path / "controls"),
            fp_1f_clean_indices=[0, 48, 96],
            fp_1f_target_index=24,
        )
    ]
    config.caching.h3_one_frame = True
    config.caching.h3_task = "fl2va"
    config.caching.h3_video_vae = "video_vae.safetensors"
    config.caching.h3_audio_vae = "audio_vae.safetensors"
    config.caching.h3_text_encoder = "text_encoder.safetensors"
    config.caching.h3_tokenizer = "tokenizer"
    config.training.h3_one_frame = True
    config.training.h3_model = "dit.safetensors"
    config.inference.h3_one_frame = True
    config.inference.h3_model = "dit.safetensors"
    config.inference.prompt = "an intermediate frame"
    config.inference.h3_condition_images = "first image.png\nmiddle.png\nlast.png"
    config.inference.h3_one_frame_options = "target_index=24,control_index=0;48;96"
    return config


def test_one_frame_project_exports_dataset_times_and_all_workflow_flags(tmp_path):
    config = _config(tmp_path)

    document = build_h3_dataset_toml_document(config, include_training=True, include_validation=False)
    dataset = document["datasets"][0]
    assert dataset["fp_1f_clean_indices"] == [0, 48, 96]
    assert dataset["fp_1f_target_index"] == 24

    latent = build_cache_latents_cmd(config)
    text = build_cache_text_cmd(config)
    training = build_training_cmd(config)
    assert "--one_frame" in latent and latent[latent.index("--task") + 1] == "fl2va"
    assert "--one_frame" in text
    assert "--one_frame" in training


def test_one_frame_inference_command_round_trips_through_real_parser(tmp_path):
    config = _config(tmp_path)

    command = build_inference_cmd(config)
    args = create_inference_parser().parse_args(command[3:])

    assert args.frame_count == 1
    assert args.one_frame == "target_index=24,control_index=0;48;96"
    assert args.condition_image == ["first image.png", "middle.png", "last.png"]
    assert Path(args.output).suffix == ".png"
    assert "--duration" not in command
