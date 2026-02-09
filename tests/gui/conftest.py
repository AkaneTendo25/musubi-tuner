"""Shared fixtures for GUI dashboard tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the package importable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from musubi_tuner.gui_dashboard.project_schema import (
    CachingConfig,
    DatasetConfig,
    DatasetEntry,
    GeneralConfig,
    ProjectConfig,
    TrainingConfig,
)


@pytest.fixture
def tmp_project_dir(tmp_path):
    return str(tmp_path / "test_project")


def _make_config(project_dir: str, **overrides) -> ProjectConfig:
    """Create a ProjectConfig with sensible test defaults."""
    config = ProjectConfig(
        name="Test Project",
        project_dir=project_dir,
        dataset=DatasetConfig(
            general=GeneralConfig(enable_bucket=True, bucket_no_upscale=True),
            datasets=[
                DatasetEntry(
                    type="video",
                    directory="/data/videos",
                    cache_directory="/data/cache",
                    resolution_w=768,
                    resolution_h=512,
                    batch_size=1,
                    num_repeats=1,
                    caption_extension=".txt",
                    target_frames=33,
                    frame_extraction="head",
                )
            ],
        ),
        caching=CachingConfig(
            ltx2_checkpoint="/models/ltx2.safetensors",
            gemma_root="/models/gemma",
            ltx2_mode="video",
            vae_dtype="bfloat16",
            device="cuda",
            skip_existing=True,
        ),
        training=TrainingConfig(
            ltx2_checkpoint="/models/ltx2.safetensors",
            gemma_root="/models/gemma",
            ltx2_mode="video",
            learning_rate=1e-4,
            optimizer_type="adamw8bit",
            lr_scheduler="constant_with_warmup",
            lr_warmup_steps=100,
            max_train_steps=1600,
            network_dim=16,
            network_alpha=16,
            output_dir="/output/lora",
            output_name="test_lora",
            flash_attn=True,
            gradient_checkpointing=True,
            mixed_precision="bf16",
        ),
    )
    # Apply overrides
    for key, value in overrides.items():
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return config


# Mock _find_script to avoid needing actual script files
MOCK_SCRIPTS = {
    "ltx2_cache_latents.py": "/pkg/ltx2_cache_latents.py",
    "ltx2_cache_text_encoder_outputs.py": "/pkg/ltx2_cache_text_encoder_outputs.py",
    "ltx2_train_network.py": "/pkg/ltx2_train_network.py",
}


@pytest.fixture(autouse=True)
def mock_find_script():
    def fake_find(name):
        if name in MOCK_SCRIPTS:
            return MOCK_SCRIPTS[name]
        raise FileNotFoundError(f"Script not found: {name}")

    with patch("musubi_tuner.gui_dashboard.command_builder._find_script", side_effect=fake_find):
        yield
