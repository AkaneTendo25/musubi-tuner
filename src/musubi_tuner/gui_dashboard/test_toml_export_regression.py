import sys
from pathlib import Path

import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from musubi_tuner.gui_dashboard import toml_export
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig


WINDOWS_VIDEO_DIR = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\videos"
WINDOWS_CACHE_DIR = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\cache"
WINDOWS_REF_CACHE_DIR = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\ref_cache"
WINDOWS_REF_DIR = r"E:\DaVinciResolve_Files_cache.gallery\LORA_STUFF\musubi-tuner-ltx2\projects\Kom\refs"


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
                    "control_directory": WINDOWS_REF_DIR,
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
    assert dataset["reference_cache_directory"] == WINDOWS_REF_CACHE_DIR
    assert dataset["reference_directory"] == WINDOWS_REF_DIR
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
