"""Tests for API routers and end-to-end workflows."""

from pathlib import Path

import pytest

from musubi_tuner.gui_dashboard.command_builder import (
    build_cache_latents_cmd,
    build_cache_text_cmd,
    build_training_cmd,
)
from musubi_tuner.gui_dashboard.project_schema import ProjectConfig
from musubi_tuner.gui_dashboard.toml_export import export_dataset_toml

from .conftest import _make_config


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


class TestEndToEnd:
    """Tests that configs produce commands with the right structure for the actual CLI."""

    def test_full_video_workflow(self, tmp_path):
        """Full video training workflow: config -> TOML + all 3 commands."""
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
