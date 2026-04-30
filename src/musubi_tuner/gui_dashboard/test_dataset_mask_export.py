from __future__ import annotations

from musubi_tuner.gui_dashboard.project_schema import DatasetEntry, ProjectConfig
from musubi_tuner.gui_dashboard.toml_export import build_dataset_toml_document


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
