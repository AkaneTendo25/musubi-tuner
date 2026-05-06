from __future__ import annotations

import pytest

from musubi_tuner.gui_dashboard.project_schema import DatasetEntry, ProjectConfig, SliderTargetConfig
from musubi_tuner.gui_dashboard.toml_export import _write_slider_toml, build_dataset_toml_document, tomllib


def test_dataset_toml_export_includes_masked_loss_fields():
    config = ProjectConfig()
    config.dataset.datasets = [
        DatasetEntry(
            type="video",
            directory="G:/data/videos",
            cache_directory="G:/data/cache",
            loss_mask_directory="G:/data/masks",
            default_loss_mask_path="G:/data/default_mask.png",
            loss_mask_invert=True,
        )
    ]

    dataset = build_dataset_toml_document(config)["datasets"][0]

    assert dataset["loss_mask_directory"] == "G:/data/masks"
    assert dataset["default_loss_mask_path"] == "G:/data/default_mask.png"
    assert dataset["loss_mask_invert"] is True
    assert "loss_mask_use_alpha" not in dataset


def test_dataset_toml_export_includes_alpha_mask_flag_only_when_enabled():
    config = ProjectConfig()
    config.dataset.datasets = [
        DatasetEntry(
            type="image",
            directory="G:/data/images",
            cache_directory="G:/data/cache",
            loss_mask_use_alpha=True,
        )
    ]

    dataset = build_dataset_toml_document(config)["datasets"][0]

    assert dataset["loss_mask_use_alpha"] is True
    assert "loss_mask_invert" not in dataset


def test_dataset_toml_export_rejects_unsupported_extra_control_directories():
    config = ProjectConfig()
    config.dataset.datasets = [
        DatasetEntry(
            type="video",
            directory="G:/data/videos",
            cache_directory="G:/data/cache",
            control_directory="G:/data/control_a",
            extra_control_directories="G:/data/control_b",
        )
    ]

    with pytest.raises(ValueError, match="Multiple control/reference directories"):
        build_dataset_toml_document(config)


def test_dataset_toml_export_uses_reference_directories_with_reference_caches():
    config = ProjectConfig()
    config.dataset.datasets = [
        DatasetEntry(
            type="video",
            directory="G:/data/videos",
            cache_directory="G:/data/cache",
            control_directory="G:/data/ref_a",
            extra_control_directories="G:/data/ref_b",
            reference_cache_directory="G:/data/ref_cache_a",
            extra_reference_cache_directories="G:/data/ref_cache_b",
        )
    ]

    dataset = build_dataset_toml_document(config)["datasets"][0]

    assert dataset["reference_directories"] == ["G:/data/ref_a", "G:/data/ref_b"]
    assert dataset["reference_cache_directories"] == ["G:/data/ref_cache_a", "G:/data/ref_cache_b"]
    assert "extra_control_directories" not in dataset


def test_slider_toml_export_escapes_target_strings(tmp_path):
    if tomllib is None:
        pytest.skip("tomllib/tomli is not available")

    config = ProjectConfig()
    config.slider.targets = [
        SliderTargetConfig(
            positive='quoted "positive"\nnext line',
            negative='negative \\ value',
            target_class='class "name"',
            weight=1.0,
        )
    ]
    path = _write_slider_toml(config, tmp_path / "slider_config.toml")

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))

    assert parsed["targets"][0]["positive"] == 'quoted "positive"\nnext line'
    assert parsed["targets"][0]["negative"] == "negative \\ value"
    assert parsed["targets"][0]["target_class"] == 'class "name"'
