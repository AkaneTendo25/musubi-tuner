"""
Tests for gui_dashboard backend: project schema, command builder, TOML export,
and API routers.

Run:  python -m pytest tests/test_gui_dashboard.py -v
  or: python tests/test_gui_dashboard.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the package importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from musubi_tuner.gui_dashboard.project_schema import (
    CachingConfig,
    DatasetConfig,
    DatasetEntry,
    GeneralConfig,
    ProjectConfig,
    TrainingConfig,
)
from musubi_tuner.gui_dashboard.command_builder import (
    _toml_value,
    _write_toml_fallback,
    build_cache_latents_cmd,
    build_cache_text_cmd,
    build_training_cmd,
    export_dataset_toml,
)
from musubi_tuner.gui_dashboard.process_manager import ManagedProcess, ProcessManager, ProcessState


# ─── Fixtures ────────────────────────────────────────────────────────────


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


# ─── Schema Tests ────────────────────────────────────────────────────────


class TestProjectSchema:
    def test_default_creation(self):
        config = ProjectConfig()
        assert config.version == 1
        assert config.name == "New Project"
        assert config.dataset.general.enable_bucket is True
        assert config.training.learning_rate == 1e-4
        assert config.training.network_dim == 16

    def test_json_roundtrip(self):
        config = ProjectConfig(name="Roundtrip Test", project_dir="/tmp/test")
        json_str = config.model_dump_json()
        restored = ProjectConfig.model_validate_json(json_str)
        assert restored.name == "Roundtrip Test"
        assert restored.project_dir == "/tmp/test"
        assert restored.dataset.general.enable_bucket == config.dataset.general.enable_bucket
        assert restored.training.learning_rate == config.training.learning_rate

    def test_save_and_load(self, tmp_path):
        config = ProjectConfig(name="Save Test", project_dir=str(tmp_path))
        config.save()

        loaded = ProjectConfig.load(tmp_path / "project.json")
        assert loaded.name == "Save Test"
        assert loaded.project_dir == str(tmp_path)
        assert loaded.training.max_train_steps == config.training.max_train_steps

    def test_dataset_entry_types(self):
        video = DatasetEntry(type="video", directory="/v", cache_directory="/vc")
        assert video.target_frames == 33

        image = DatasetEntry(type="image", directory="/i", cache_directory="/ic")
        assert image.type == "image"

        audio = DatasetEntry(type="audio", directory="/a", cache_directory="/ac")
        assert audio.type == "audio"

    def test_full_config_roundtrip(self, tmp_path):
        config = _make_config(str(tmp_path))
        config.save()
        loaded = ProjectConfig.load(tmp_path / "project.json")
        assert loaded.dataset.datasets[0].type == "video"
        assert loaded.dataset.datasets[0].directory == "/data/videos"
        assert loaded.caching.ltx2_checkpoint == "/models/ltx2.safetensors"
        assert loaded.training.optimizer_type == "adamw8bit"

    def test_optional_fields_none(self):
        config = ProjectConfig()
        assert config.training.max_train_epochs is None
        assert config.training.seed is None
        assert config.training.blocks_to_swap is None
        assert config.training.sample_every_n_steps is None


# ─── TOML Export Tests ───────────────────────────────────────────────────


class TestTomlExport:
    def test_toml_value_primitives(self):
        assert _toml_value(True) == "true"
        assert _toml_value(False) == "false"
        assert _toml_value(42) == "42"
        assert _toml_value(3.14) == "3.14"
        assert _toml_value("hello") == '"hello"'
        assert _toml_value([1, 2, 3]) == "[1, 2, 3]"
        assert _toml_value([768, 512]) == "[768, 512]"

    def test_export_video_dataset(self, tmp_path):
        config = _make_config(str(tmp_path))
        path = export_dataset_toml(config)
        assert path.exists()

        content = path.read_text(encoding="utf-8")
        assert "[general]" in content
        assert "enable_bucket = true" in content
        assert "[[datasets]]" in content
        assert 'video_directory = "/data/videos"' in content
        assert "resolution" in content
        assert "768" in content
        assert "512" in content
        assert "target_frames" in content
        assert "33" in content
        assert 'frame_extraction = "head"' in content

    def test_export_image_dataset(self, tmp_path):
        config = ProjectConfig(
            project_dir=str(tmp_path),
            dataset=DatasetConfig(
                datasets=[
                    DatasetEntry(
                        type="image",
                        directory="/data/images",
                        cache_directory="/data/img_cache",
                        resolution_w=1024,
                        resolution_h=1024,
                    )
                ]
            ),
        )
        path = export_dataset_toml(config)
        content = path.read_text(encoding="utf-8")
        assert 'image_directory = "/data/images"' in content
        assert "resolution" in content
        assert "1024" in content
        # Should NOT have video-specific fields
        assert "target_frames" not in content
        assert "frame_extraction" not in content

    def test_export_audio_dataset(self, tmp_path):
        config = ProjectConfig(
            project_dir=str(tmp_path),
            dataset=DatasetConfig(
                datasets=[
                    DatasetEntry(
                        type="audio",
                        directory="/data/audio",
                        cache_directory="/data/audio_cache",
                    )
                ]
            ),
        )
        path = export_dataset_toml(config)
        content = path.read_text(encoding="utf-8")
        assert 'audio_directory = "/data/audio"' in content
        # Audio datasets should NOT have resolution
        assert "resolution" not in content

    def test_export_validation_datasets(self, tmp_path):
        config = ProjectConfig(
            project_dir=str(tmp_path),
            dataset=DatasetConfig(
                datasets=[DatasetEntry(type="video", directory="/train", cache_directory="/tc")],
                validation_datasets=[DatasetEntry(type="video", directory="/val", cache_directory="/vc")],
            ),
        )
        path = export_dataset_toml(config)
        content = path.read_text(encoding="utf-8")
        assert "[[datasets]]" in content
        assert "[[validation_datasets]]" in content
        assert 'video_directory = "/val"' in content

    def test_export_multiple_datasets(self, tmp_path):
        config = ProjectConfig(
            project_dir=str(tmp_path),
            dataset=DatasetConfig(
                datasets=[
                    DatasetEntry(type="video", directory="/v1", cache_directory="/c1"),
                    DatasetEntry(type="image", directory="/i1", cache_directory="/c2"),
                    DatasetEntry(type="audio", directory="/a1", cache_directory="/c3"),
                ]
            ),
        )
        path = export_dataset_toml(config)
        content = path.read_text(encoding="utf-8")
        assert content.count("[[datasets]]") == 3
        assert 'video_directory = "/v1"' in content
        assert 'image_directory = "/i1"' in content
        assert 'audio_directory = "/a1"' in content


# ─── Command Builder Tests ───────────────────────────────────────────────


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


class TestCacheLatentsCmd:
    def test_basic_video_mode(self, tmp_path):
        config = _make_config(str(tmp_path))
        cmd = build_cache_latents_cmd(config)

        assert sys.executable == cmd[0]
        assert MOCK_SCRIPTS["ltx2_cache_latents.py"] in cmd
        assert "--dataset_config" in cmd
        assert "--ltx2_checkpoint" in cmd
        assert "/models/ltx2.safetensors" in cmd
        assert "--ltx2_mode" in cmd
        assert "video" in cmd
        assert "--skip_existing" in cmd
        assert "--vae_dtype" in cmd
        assert "bfloat16" in cmd

    def test_av_mode_with_audio_source(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "caching.ltx2_mode": "av",
                "caching.ltx2_audio_source": "audio_files",
                "caching.ltx2_audio_dir": "/data/audio",
                "caching.ltx2_audio_ext": ".flac",
            },
        )
        cmd = build_cache_latents_cmd(config)
        assert "--ltx2_mode" in cmd
        idx = cmd.index("--ltx2_mode")
        assert cmd[idx + 1] == "av"
        assert "--ltx2_audio_source" in cmd
        assert "audio_files" in cmd
        assert "--ltx2_audio_dir" in cmd
        assert "/data/audio" in cmd
        assert "--ltx2_audio_ext" in cmd
        assert ".flac" in cmd

    def test_vae_tiling_options(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "caching.vae_chunk_size": 16,
                "caching.vae_spatial_tile_size": 512,
            },
        )
        cmd = build_cache_latents_cmd(config)
        assert "--vae_chunk_size" in cmd
        assert "16" in cmd
        assert "--vae_spatial_tile_size" in cmd
        assert "512" in cmd

    def test_skip_existing_off(self, tmp_path):
        config = _make_config(str(tmp_path), **{"caching.skip_existing": False})
        cmd = build_cache_latents_cmd(config)
        assert "--skip_existing" not in cmd

    def test_writes_toml_file(self, tmp_path):
        config = _make_config(str(tmp_path))
        build_cache_latents_cmd(config)
        toml_path = Path(str(tmp_path)) / "dataset_config.toml"
        assert toml_path.exists()


class TestCacheTextCmd:
    def test_basic(self, tmp_path):
        config = _make_config(str(tmp_path))
        cmd = build_cache_text_cmd(config)

        assert MOCK_SCRIPTS["ltx2_cache_text_encoder_outputs.py"] in cmd
        assert "--gemma_root" in cmd
        assert "/models/gemma" in cmd
        assert "--ltx2_mode" in cmd
        assert "video" in cmd

    def test_mixed_precision(self, tmp_path):
        config = _make_config(str(tmp_path), **{"caching.mixed_precision": "bf16"})
        cmd = build_cache_text_cmd(config)
        assert "--mixed_precision" in cmd
        assert "bf16" in cmd

    def test_no_mixed_precision(self, tmp_path):
        config = _make_config(str(tmp_path), **{"caching.mixed_precision": "no"})
        cmd = build_cache_text_cmd(config)
        assert "--mixed_precision" not in cmd

    def test_gemma_8bit(self, tmp_path):
        config = _make_config(str(tmp_path), **{"caching.gemma_load_in_8bit": True})
        cmd = build_cache_text_cmd(config)
        assert "--gemma_load_in_8bit" in cmd

    def test_gemma_4bit_with_quant_type(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "caching.gemma_load_in_4bit": True,
                "caching.gemma_bnb_4bit_quant_type": "fp4",
            },
        )
        cmd = build_cache_text_cmd(config)
        assert "--gemma_load_in_4bit" in cmd
        assert "--gemma_bnb_4bit_quant_type" in cmd
        assert "fp4" in cmd

    def test_precache_sample_prompts(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "caching.precache_sample_prompts": True,
                "caching.sample_prompts": "/data/prompts.txt",
            },
        )
        cmd = build_cache_text_cmd(config)
        assert "--precache_sample_prompts" in cmd
        assert "--sample_prompts" in cmd
        assert "/data/prompts.txt" in cmd

    def test_precache_preservation_with_dop(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "caching.precache_preservation_prompts": True,
                "caching.blank_preservation": True,
                "caching.dop": True,
                "caching.dop_class_prompt": "woman",
            },
        )
        cmd = build_cache_text_cmd(config)
        assert "--precache_preservation_prompts" in cmd
        assert "--blank_preservation" in cmd
        assert "--dop" in cmd
        assert "--dop_class_prompt" in cmd
        assert "woman" in cmd


class TestTrainingCmd:
    def test_basic_training(self, tmp_path):
        config = _make_config(str(tmp_path))
        cmd = build_training_cmd(config)

        # Should use accelerate launch
        assert "-m" in cmd
        assert "accelerate.commands.launch" in cmd
        assert MOCK_SCRIPTS["ltx2_train_network.py"] in cmd

        # Core args
        assert "--dataset_config" in cmd
        assert "--ltx2_checkpoint" in cmd
        assert "--learning_rate" in cmd
        assert "--optimizer_type" in cmd
        assert "--lr_scheduler" in cmd
        assert "--network_dim" in cmd
        assert "--network_alpha" in cmd
        assert "--output_dir" in cmd
        assert "--output_name" in cmd
        assert "--gui" in cmd  # Always appended

    def test_training_always_has_gui_flag(self, tmp_path):
        config = _make_config(str(tmp_path))
        cmd = build_training_cmd(config)
        assert "--gui" in cmd

    def test_lora_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.network_dim": 32,
                "training.network_alpha": 16,
                "training.lora_target_preset": "v2v",
            },
        )
        cmd = build_training_cmd(config)
        idx = cmd.index("--network_dim")
        assert cmd[idx + 1] == "32"
        idx = cmd.index("--network_alpha")
        assert cmd[idx + 1] == "16"
        idx = cmd.index("--lora_target_preset")
        assert cmd[idx + 1] == "v2v"

    def test_optimizer_args(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{"training.optimizer_args": "weight_decay=0.01 betas=0.9,0.999"},
        )
        cmd = build_training_cmd(config)
        idx = cmd.index("--optimizer_args")
        assert cmd[idx + 1] == "weight_decay=0.01"
        assert cmd[idx + 2] == "betas=0.9,0.999"

    def test_epochs_instead_of_steps(self, tmp_path):
        config = _make_config(str(tmp_path), **{"training.max_train_epochs": 10})
        cmd = build_training_cmd(config)
        assert "--max_train_epochs" in cmd
        assert "10" in cmd
        assert "--max_train_steps" not in cmd

    def test_steps_when_no_epochs(self, tmp_path):
        config = _make_config(str(tmp_path))
        cmd = build_training_cmd(config)
        assert "--max_train_steps" in cmd
        assert "1600" in cmd
        assert "--max_train_epochs" not in cmd

    def test_memory_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.blocks_to_swap": 20,
                "training.gradient_checkpointing": True,
                "training.blockwise_checkpointing": True,
                "training.split_attn_target": "all",
                "training.split_attn_mode": "batch",
            },
        )
        cmd = build_training_cmd(config)
        assert "--blocks_to_swap" in cmd
        assert "20" in cmd
        assert "--gradient_checkpointing" in cmd
        assert "--blockwise_checkpointing" in cmd
        assert "--split_attn_target" in cmd
        assert "all" in cmd
        assert "--split_attn_mode" in cmd
        assert "batch" in cmd

    def test_sampling_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.sample_every_n_steps": 100,
                "training.sample_prompts": "/data/prompts.txt",
                "training.height": 480,
                "training.width": 640,
                "training.sample_num_frames": 25,
                "training.sample_with_offloading": True,
                "training.sample_merge_audio": True,
            },
        )
        cmd = build_training_cmd(config)
        assert "--sample_every_n_steps" in cmd
        assert "100" in cmd
        assert "--sample_prompts" in cmd
        assert "--height" in cmd
        assert "480" in cmd
        assert "--width" in cmd
        assert "640" in cmd
        assert "--sample_num_frames" in cmd
        assert "25" in cmd
        assert "--sample_with_offloading" in cmd
        assert "--sample_merge_audio" in cmd

    def test_preservation_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.blank_preservation": True,
                "training.blank_preservation_args": "multiplier=0.5",
                "training.dop": True,
                "training.dop_args": "class=woman multiplier=1.0",
                "training.prior_divergence": True,
                "training.prior_divergence_args": "multiplier=0.1",
            },
        )
        cmd = build_training_cmd(config)
        assert "--blank_preservation" in cmd
        assert "--blank_preservation_args" in cmd
        assert "multiplier=0.5" in cmd
        assert "--dop" in cmd
        assert "--dop_args" in cmd
        assert "class=woman" in cmd
        assert "multiplier=1.0" in cmd
        assert "--prior_divergence" in cmd
        assert "--prior_divergence_args" in cmd
        assert "multiplier=0.1" in cmd

    def test_loss_weighting(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.video_loss_weight": 0.5,
                "training.audio_loss_weight": 2.0,
            },
        )
        cmd = build_training_cmd(config)
        assert "--video_loss_weight" in cmd
        assert "0.5" in cmd
        assert "--audio_loss_weight" in cmd
        assert "2.0" in cmd

    def test_default_loss_weights_omitted(self, tmp_path):
        config = _make_config(str(tmp_path))
        cmd = build_training_cmd(config)
        assert "--video_loss_weight" not in cmd
        assert "--audio_loss_weight" not in cmd

    def test_fp8_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.fp8_base": True,
                "training.fp8_scaled": True,
            },
        )
        cmd = build_training_cmd(config)
        assert "--fp8_base" in cmd
        assert "--fp8_scaled" in cmd

    def test_av_mode_training(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.ltx2_mode": "av",
                "training.separate_audio_buckets": True,
                "training.lora_target_preset": "audio",
            },
        )
        cmd = build_training_cmd(config)
        idx = cmd.index("--ltx2_mode")
        assert cmd[idx + 1] == "av"
        assert "--separate_audio_buckets" in cmd
        idx = cmd.index("--lora_target_preset")
        assert cmd[idx + 1] == "audio"

    def test_save_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.save_every_n_epochs": 5,
                "training.save_every_n_steps": 500,
                "training.save_state": True,
            },
        )
        cmd = build_training_cmd(config)
        assert "--save_every_n_epochs" in cmd
        assert "5" in cmd
        assert "--save_every_n_steps" in cmd
        assert "500" in cmd
        assert "--save_state" in cmd

    def test_logging_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.log_with": "tensorboard",
                "training.logging_dir": "/logs",
            },
        )
        cmd = build_training_cmd(config)
        assert "--log_with" in cmd
        assert "tensorboard" in cmd
        assert "--logging_dir" in cmd
        assert "/logs" in cmd

    def test_seed(self, tmp_path):
        config = _make_config(str(tmp_path), **{"training.seed": 42})
        cmd = build_training_cmd(config)
        assert "--seed" in cmd
        assert "42" in cmd

    def test_no_seed_when_none(self, tmp_path):
        config = _make_config(str(tmp_path))
        cmd = build_training_cmd(config)
        assert "--seed" not in cmd

    def test_mixed_precision_in_accelerate(self, tmp_path):
        config = _make_config(str(tmp_path), **{"training.mixed_precision": "fp16"})
        cmd = build_training_cmd(config)
        # mixed_precision should be passed to accelerate launch
        mp_idx = cmd.index("--mixed_precision")
        assert cmd[mp_idx + 1] == "fp16"
        # It should come before the training script
        script_idx = cmd.index(MOCK_SCRIPTS["ltx2_train_network.py"])
        assert mp_idx < script_idx

    def test_ffn_chunk_settings(self, tmp_path):
        config = _make_config(
            str(tmp_path),
            **{
                "training.ffn_chunk_target": "all",
                "training.ffn_chunk_size": 2048,
            },
        )
        cmd = build_training_cmd(config)
        assert "--ffn_chunk_target" in cmd
        assert "all" in cmd
        assert "--ffn_chunk_size" in cmd
        assert "2048" in cmd


# ─── Process Manager Tests ───────────────────────────────────────────────


class TestProcessManager:
    def test_initial_state(self):
        pm = ProcessManager()
        status = pm.get_status("training")
        assert status["state"] == "idle"
        assert status["exit_code"] is None

    def test_all_statuses(self):
        pm = ProcessManager()
        statuses = pm.get_all_statuses()
        assert "cache_latents" in statuses
        assert "cache_text" in statuses
        assert "training" in statuses
        for s in statuses.values():
            assert s["state"] == "idle"

    def test_empty_logs(self):
        pm = ProcessManager()
        logs = pm.get_logs("training")
        assert logs == []

    def test_start_simple_command(self):
        pm = ProcessManager()
        # Run a simple command that exits quickly
        pm.start("training", [sys.executable, "-c", "print('hello')"])
        status = pm.get_status("training")
        assert status["state"] in ("running", "finished")

    def test_cannot_start_twice(self):
        pm = ProcessManager()
        # Use a command that takes a bit of time
        pm.start("training", [sys.executable, "-c", "import time; time.sleep(5)"])
        with pytest.raises(RuntimeError, match="already running"):
            pm.start("training", [sys.executable, "-c", "print('second')"])
        pm.stop("training")

    def test_stop_idle_is_noop(self):
        pm = ProcessManager()
        pm.stop("training")  # Should not raise

    def test_managed_process_state_machine(self):
        mp = ManagedProcess([sys.executable, "-c", "print('test')"])
        assert mp.state == ProcessState.IDLE
        mp.start()
        # Wait for completion
        import time
        for _ in range(50):
            if mp.state != ProcessState.RUNNING:
                break
            time.sleep(0.1)
        assert mp.state == ProcessState.FINISHED
        assert mp.exit_code == 0
        logs = mp.get_logs()
        assert any("test" in line for line in logs)

    def test_managed_process_logs_command(self):
        mp = ManagedProcess([sys.executable, "-c", "print('hello world')"])
        mp.start()
        import time
        for _ in range(50):
            if mp.state != ProcessState.RUNNING:
                break
            time.sleep(0.1)
        logs = mp.get_logs()
        # First line should be the command
        assert logs[0].startswith("$")


# ─── API Router Tests (using httpx + FastAPI TestClient) ─────────────────


class TestAPIRouters:
    @pytest.fixture
    def client(self, tmp_path):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi[testclient] or httpx not installed")

        from musubi_tuner.gui_dashboard.management_server import create_management_app

        app = create_management_app()
        return TestClient(app)

    def test_get_project_no_project(self, client):
        resp = client.get("/api/project")
        assert resp.status_code == 200
        data = resp.json()
        assert data["loaded"] is False

    def test_create_project(self, client, tmp_path):
        resp = client.post(
            "/api/project",
            json={"name": "API Test", "project_dir": str(tmp_path / "proj")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["config"]["name"] == "API Test"

        # project.json should exist
        assert (tmp_path / "proj" / "project.json").exists()

    def test_create_project_requires_dir(self, client):
        resp = client.post("/api/project", json={"name": "No Dir"})
        assert resp.status_code == 400

    def test_update_project(self, client, tmp_path):
        # First create
        client.post(
            "/api/project",
            json={"name": "Before", "project_dir": str(tmp_path / "proj")},
        )
        # Then update
        resp = client.put(
            "/api/project",
            json={"name": "After", "project_dir": str(tmp_path / "proj")},
        )
        assert resp.status_code == 200
        assert resp.json()["config"]["name"] == "After"

    def test_load_project(self, client, tmp_path):
        # Save a project file
        config = ProjectConfig(name="LoadMe", project_dir=str(tmp_path))
        config.save()

        resp = client.post(
            "/api/project/load",
            json={"path": str(tmp_path / "project.json")},
        )
        assert resp.status_code == 200
        assert resp.json()["config"]["name"] == "LoadMe"

    def test_load_nonexistent(self, client):
        resp = client.post(
            "/api/project/load",
            json={"path": "/nonexistent/project.json"},
        )
        assert resp.status_code == 404

    def test_filesystem_browse(self, client, tmp_path):
        # Create some dirs
        (tmp_path / "subdir1").mkdir()
        (tmp_path / "subdir2").mkdir()
        (tmp_path / "file.txt").write_text("test")

        resp = client.get(f"/api/fs/browse?path={tmp_path}&show_files=true")
        assert resp.status_code == 200
        data = resp.json()
        names = [e["name"] for e in data["entries"]]
        assert "subdir1" in names
        assert "subdir2" in names
        assert "file.txt" in names

    def test_filesystem_exists(self, client, tmp_path):
        (tmp_path / "exists.txt").write_text("yes")
        resp = client.get(f"/api/fs/exists?path={tmp_path / 'exists.txt'}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["exists"] is True
        assert data["is_file"] is True

        resp = client.get(f"/api/fs/exists?path={tmp_path / 'nope.txt'}")
        data = resp.json()
        assert data["exists"] is False

    def test_process_status_all(self, client):
        resp = client.get("/api/processes/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "cache_latents" in data
        assert "cache_text" in data
        assert "training" in data

    def test_process_invalid_type(self, client):
        resp = client.post("/api/processes/invalid/start")
        assert resp.status_code == 400

    def test_dataset_config_no_project(self, client):
        resp = client.get("/api/dataset/config")
        assert resp.status_code == 400

    def test_dataset_export_toml(self, client, tmp_path):
        # Create project first
        client.post(
            "/api/project",
            json={
                "name": "TOML Test",
                "project_dir": str(tmp_path / "proj"),
                "dataset": {
                    "general": {"enable_bucket": True, "bucket_no_upscale": True},
                    "datasets": [
                        {
                            "type": "video",
                            "directory": "/data/videos",
                            "cache_directory": "/data/cache",
                            "resolution_w": 768,
                            "resolution_h": 512,
                            "batch_size": 1,
                            "num_repeats": 1,
                            "caption_extension": ".txt",
                            "target_frames": 33,
                            "frame_extraction": "head",
                        }
                    ],
                    "validation_datasets": [],
                },
            },
        )

        resp = client.post("/api/dataset/export-toml")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "dataset_config.toml" in data["path"]

        # Verify file content
        toml_content = Path(data["path"]).read_text(encoding="utf-8")
        assert "[[datasets]]" in toml_content
        assert 'video_directory = "/data/videos"' in toml_content

    def test_dataset_preview_toml(self, client, tmp_path):
        client.post(
            "/api/project",
            json={
                "name": "Preview Test",
                "project_dir": str(tmp_path / "proj"),
                "dataset": {
                    "general": {"enable_bucket": True, "bucket_no_upscale": False},
                    "datasets": [
                        {
                            "type": "image",
                            "directory": "/images",
                            "cache_directory": "/ic",
                            "resolution_w": 1024,
                            "resolution_h": 1024,
                            "batch_size": 2,
                            "num_repeats": 3,
                            "caption_extension": ".txt",
                            "target_frames": 33,
                            "frame_extraction": "head",
                        }
                    ],
                    "validation_datasets": [],
                },
            },
        )

        resp = client.get("/api/dataset/preview-toml")
        assert resp.status_code == 200
        data = resp.json()
        assert "bucket_no_upscale = false" in data["toml"]
        assert 'image_directory = "/images"' in data["toml"]
        assert "batch_size = 2" in data["toml"]
        assert "num_repeats = 3" in data["toml"]


# ─── End-to-End: Config → Command → Parseable ───────────────────────────


class TestEndToEnd:
    """Tests that configs produce commands with the right structure for the actual CLI."""

    def test_full_video_workflow(self, tmp_path):
        """Full video training workflow: config → TOML + all 3 commands."""
        config = _make_config(str(tmp_path))

        # Export TOML
        toml_path = export_dataset_toml(config)
        assert toml_path.exists()
        content = toml_path.read_text(encoding="utf-8")
        assert "[[datasets]]" in content

        # Build all 3 commands
        lat_cmd = build_cache_latents_cmd(config)
        txt_cmd = build_cache_text_cmd(config)
        trn_cmd = build_training_cmd(config)

        # All should reference the same TOML
        assert str(toml_path) in lat_cmd
        assert str(toml_path) in txt_cmd
        assert str(toml_path) in trn_cmd

        # Training should be via accelerate
        assert "accelerate.commands.launch" in trn_cmd

    def test_full_av_workflow(self, tmp_path):
        """Full audio-video workflow with preservation."""
        config = _make_config(
            str(tmp_path),
            **{
                "caching.ltx2_mode": "av",
                "caching.precache_sample_prompts": True,
                "caching.sample_prompts": "/prompts.txt",
                "caching.precache_preservation_prompts": True,
                "caching.blank_preservation": True,
                "training.ltx2_mode": "av",
                "training.lora_target_preset": "full",
                "training.separate_audio_buckets": True,
                "training.blank_preservation": True,
                "training.use_precached_preservation": True,
                "training.sample_every_n_steps": 200,
                "training.sample_merge_audio": True,
            },
        )

        lat_cmd = build_cache_latents_cmd(config)
        txt_cmd = build_cache_text_cmd(config)
        trn_cmd = build_training_cmd(config)

        # Latents should be av mode
        idx = lat_cmd.index("--ltx2_mode")
        assert lat_cmd[idx + 1] == "av"

        # Text should precache
        assert "--precache_sample_prompts" in txt_cmd
        assert "--precache_preservation_prompts" in txt_cmd
        assert "--blank_preservation" in txt_cmd

        # Training should have all av flags
        assert "--separate_audio_buckets" in trn_cmd
        assert "--blank_preservation" in trn_cmd
        assert "--use_precached_preservation" in trn_cmd
        assert "--sample_merge_audio" in trn_cmd
        idx = trn_cmd.index("--lora_target_preset")
        assert trn_cmd[idx + 1] == "full"


# ─── Run directly ────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
