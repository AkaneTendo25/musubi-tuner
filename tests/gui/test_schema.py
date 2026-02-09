"""Tests for ProjectConfig schema: creation, serialization, roundtrip."""

from musubi_tuner.gui_dashboard.project_schema import DatasetEntry, ProjectConfig

from .conftest import _make_config


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
