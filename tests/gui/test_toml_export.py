"""Tests for TOML export: value serialization and dataset export."""

from musubi_tuner.gui_dashboard.project_schema import DatasetConfig, DatasetEntry, ProjectConfig
from musubi_tuner.gui_dashboard.toml_export import _toml_value, export_dataset_toml

from .conftest import _make_config


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
