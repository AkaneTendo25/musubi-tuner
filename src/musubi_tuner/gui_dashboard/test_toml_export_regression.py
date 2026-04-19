import sys
from pathlib import Path

import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from musubi_tuner.gui_dashboard import toml_export
from musubi_tuner.gui_dashboard.command_builder import build_slider_training_cmd, build_training_cmd
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig


WINDOWS_VIDEO_DIR = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\videos"
WINDOWS_CACHE_DIR = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\cache"
WINDOWS_REF_CACHE_DIR = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\ref_cache"
WINDOWS_REF_DIR = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\refs"
WINDOWS_REF_CACHE_DIR_2 = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\ref_cache_b"
WINDOWS_REF_DIR_2 = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\refs_b"
WINDOWS_REF_AUDIO_CACHE_DIR = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\ref_audio_cache"
WINDOWS_REF_AUDIO_CACHE_DIR_2 = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\ref_audio_cache_b"
WINDOWS_REF_AUDIO_DIR = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\ref_audio"
WINDOWS_REF_AUDIO_DIR_2 = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\ref_audio_b"


def _build_config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        name="GUI TOML Test",
        project_dir=str(tmp_path),
        dataset={
            "general": {
                "enable_bucket": True,
                "bucket_no_upscale": True,
            },
            "datasets": [
                {
                    "type": "video",
                    "directory": WINDOWS_VIDEO_DIR,
                    "cache_directory": WINDOWS_CACHE_DIR,
                    "reference_cache_directory": WINDOWS_REF_CACHE_DIR,
                    "extra_reference_cache_directories": WINDOWS_REF_CACHE_DIR_2,
                    "reference_audio_cache_directory": WINDOWS_REF_AUDIO_CACHE_DIR,
                    "extra_reference_audio_cache_directories": WINDOWS_REF_AUDIO_CACHE_DIR_2,
                    "control_directory": WINDOWS_REF_DIR,
                    "extra_control_directories": WINDOWS_REF_DIR_2,
                    "reference_audio_directory": WINDOWS_REF_AUDIO_DIR,
                    "extra_reference_audio_directories": WINDOWS_REF_AUDIO_DIR_2,
                    "resolution_w": 768,
                    "resolution_h": 512,
                    "batch_size": 1,
                    "num_repeats": 1,
                    "caption_extension": ".txt",
                    "target_frames": 33,
                    "frame_extraction": "full",
                    "max_frames": 129,
                    "source_fps": 24.0,
                }
            ],
            "validation_datasets": [
                {
                    "type": "audio",
                    "directory": r"E:\audio\clips",
                    "cache_directory": r"E:\audio\cache",
                    "batch_size": 2,
                    "num_repeats": 1,
                    "caption_extension": ".txt",
                }
            ],
        },
    )


def test_toml_value_escapes_windows_paths():
    parsed = toml.loads(f"video_directory = {toml_export._toml_value(WINDOWS_VIDEO_DIR)}")
    assert parsed["video_directory"] == WINDOWS_VIDEO_DIR


def test_render_toml_fallback_handles_windows_paths(monkeypatch, tmp_path):
    config = _build_config(tmp_path)
    monkeypatch.setattr(toml_export, "tomli_w", None)

    rendered = toml_export.render_toml(toml_export.build_dataset_toml_document(config))
    parsed = toml.loads(rendered)

    dataset = parsed["datasets"][0]
    assert dataset["video_directory"] == WINDOWS_VIDEO_DIR
    assert dataset["cache_directory"] == WINDOWS_CACHE_DIR
    assert dataset["reference_cache_directories"] == [WINDOWS_REF_CACHE_DIR, WINDOWS_REF_CACHE_DIR_2]
    assert dataset["reference_directories"] == [WINDOWS_REF_DIR, WINDOWS_REF_DIR_2]
    assert dataset["reference_audio_cache_directories"] == [WINDOWS_REF_AUDIO_CACHE_DIR, WINDOWS_REF_AUDIO_CACHE_DIR_2]
    assert dataset["reference_audio_directories"] == [WINDOWS_REF_AUDIO_DIR, WINDOWS_REF_AUDIO_DIR_2]
    assert dataset["frame_extraction"] == "full"
    assert dataset["max_frames"] == 129
    assert dataset["source_fps"] == 24.0


def test_export_dataset_toml_writes_parseable_file_without_tomli_w(monkeypatch, tmp_path):
    config = _build_config(tmp_path)
    monkeypatch.setattr(toml_export, "tomli_w", None)

    output_path = toml_export.export_dataset_toml(config)
    parsed = toml.load(output_path)

    assert output_path.name == "dataset_config.toml"
    assert parsed["general"]["enable_bucket"] is True
    assert parsed["datasets"][0]["video_directory"] == WINDOWS_VIDEO_DIR
    assert parsed["validation_datasets"][0]["audio_directory"] == r"E:\audio\clips"


def test_dataset_cache_directory_defaults_to_dataset_cache_subdir(tmp_path):
    config = ProjectConfig(
        name="Default Cache Dir",
        project_dir=str(tmp_path),
        dataset={
            "datasets": [
                {
                    "type": "video",
                    "directory": r"E:\datasets\clips",
                    "cache_directory": "",
                }
            ]
        },
    )

    doc = toml_export.build_dataset_toml_document(config)

    assert doc["datasets"][0]["cache_directory"] == r"E:\datasets\clips\cache"


def test_dataset_toml_export_splits_extra_reference_dirs(tmp_path):
    config = ProjectConfig(
        name="Extra Reference Dirs",
        project_dir=str(tmp_path),
        dataset={
            "datasets": [
                {
                    "type": "video",
                    "directory": WINDOWS_VIDEO_DIR,
                    "cache_directory": WINDOWS_CACHE_DIR,
                    "reference_cache_directory": WINDOWS_REF_CACHE_DIR,
                    "extra_reference_cache_directories": f"{WINDOWS_REF_CACHE_DIR_2}, {WINDOWS_REF_CACHE_DIR}",
                    "control_directory": WINDOWS_REF_DIR,
                    "extra_control_directories": f"{WINDOWS_REF_DIR_2}; {WINDOWS_REF_DIR}",
                }
            ]
        },
    )

    doc = toml_export.build_dataset_toml_document(config)

    assert doc["datasets"][0]["reference_cache_directories"] == [WINDOWS_REF_CACHE_DIR, WINDOWS_REF_CACHE_DIR_2]
    assert doc["datasets"][0]["reference_directories"] == [WINDOWS_REF_DIR, WINDOWS_REF_DIR_2]


def test_dataset_toml_export_splits_extra_control_dirs_without_reference_cache(tmp_path):
    config = ProjectConfig(
        name="Extra Control Dirs",
        project_dir=str(tmp_path),
        dataset={
            "datasets": [
                {
                    "type": "video",
                    "directory": WINDOWS_VIDEO_DIR,
                    "cache_directory": WINDOWS_CACHE_DIR,
                    "control_directory": WINDOWS_REF_DIR,
                    "extra_control_directories": f"{WINDOWS_REF_DIR_2}; {WINDOWS_REF_CACHE_DIR_2}",
                }
            ]
        },
    )

    doc = toml_export.build_dataset_toml_document(config)

    assert doc["datasets"][0]["control_directory"] == WINDOWS_REF_DIR
    assert doc["datasets"][0]["extra_control_directories"] == f"{WINDOWS_REF_DIR_2}, {WINDOWS_REF_CACHE_DIR_2}"


def test_training_command_builder_includes_av_ic_modifier_args(tmp_path):
    config = ProjectConfig(
        name="AV IC Training Cmd",
        project_dir=str(tmp_path),
        dataset={
            "datasets": [
                {
                    "type": "video",
                    "directory": WINDOWS_VIDEO_DIR,
                    "cache_directory": WINDOWS_CACHE_DIR,
                }
            ]
        },
        training={
            "ltx2_checkpoint": str(tmp_path / "ltx2.safetensors"),
            "gemma_root": str(tmp_path / "gemma"),
            "mixed_precision": "bf16",
            "lora_target_preset": "av_ic",
            "ic_lora_strategy": "av_ic",
            "av_cross_attention_mode": "v2a_only",
            "av_multi_ref": True,
        },
    )

    cmd = build_training_cmd(config)

    assert "--ic_lora_strategy" in cmd
    assert cmd[cmd.index("--ic_lora_strategy") + 1] == "av_ic"
    assert "--av_cross_attention_mode" in cmd
    assert cmd[cmd.index("--av_cross_attention_mode") + 1] == "v2a_only"
    assert "--av_multi_ref" in cmd


def test_training_command_builder_omits_default_av_ic_modifier_args(tmp_path):
    config = ProjectConfig(
        name="Default AV IC Training Cmd",
        project_dir=str(tmp_path),
        dataset={
            "datasets": [
                {
                    "type": "video",
                    "directory": WINDOWS_VIDEO_DIR,
                    "cache_directory": WINDOWS_CACHE_DIR,
                }
            ]
        },
        training={
            "ltx2_checkpoint": str(tmp_path / "ltx2.safetensors"),
            "gemma_root": str(tmp_path / "gemma"),
            "mixed_precision": "bf16",
            "lora_target_preset": "av_ic",
            "ic_lora_strategy": "av_ic",
        },
    )

    cmd = build_training_cmd(config)

    assert "--av_cross_attention_mode" not in cmd
    assert "--av_multi_ref" not in cmd


def test_slider_toml_export_writes_reference_slider_fields(tmp_path):
    config = ProjectConfig(
        name="Slider IC Export",
        project_dir=str(tmp_path),
        slider={
            "mode": "ic_reference",
            "reference_modality": "video",
            "pos_cache_dir": r"E:\slider\pos",
            "neg_cache_dir": r"E:\slider\neg",
            "text_cache_dir": r"E:\slider\text",
            "reference_cache_dir": r"E:\slider\ref",
            "sample_slider_range": "-3,-1,0,1,3",
        },
    )

    output_path = toml_export._write_slider_toml(config, toml_export.build_slider_toml_path(config))
    parsed = toml.load(output_path)

    assert parsed["mode"] == "ic_reference"
    assert parsed["reference_modality"] == "video"
    assert parsed["pos_cache_dir"] == r"E:\slider\pos"
    assert parsed["neg_cache_dir"] == r"E:\slider\neg"
    assert parsed["text_cache_dir"] == r"E:\slider\text"
    assert parsed["reference_cache_dir"] == r"E:\slider\ref"
    assert parsed["sample_slider_range"] == [-3.0, -1.0, 0.0, 1.0, 3.0]
    assert "targets" not in parsed


def test_slider_command_builder_includes_mode_relevant_training_args(tmp_path):
    config = ProjectConfig(
        name="Slider Cmd",
        project_dir=str(tmp_path),
        training={
            "ltx2_checkpoint": str(tmp_path / "ltx2.safetensors"),
            "gemma_root": str(tmp_path / "gemma"),
            "mixed_precision": "bf16",
            "ltx2_mode": "video",
            "lora_target_preset": "v2v",
        },
        slider={
            "mode": "ic_reference",
            "pos_cache_dir": str(tmp_path / "pos"),
            "neg_cache_dir": str(tmp_path / "neg"),
            "text_cache_dir": str(tmp_path / "text"),
            "reference_cache_dir": str(tmp_path / "ref"),
        },
    )

    cmd = build_slider_training_cmd(config)

    assert "--ltx2_mode" in cmd
    assert cmd[cmd.index("--ltx2_mode") + 1] == "video"
    assert "--lora_target_preset" in cmd
    assert cmd[cmd.index("--lora_target_preset") + 1] == "v2v"
