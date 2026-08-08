import asyncio
from types import SimpleNamespace

from musubi_tuner.gui_dashboard.project_schema import DatasetEntry, ProjectConfig
from musubi_tuner.gui_dashboard.routers.cache_status import cache_status


def _request(config: ProjectConfig) -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(project_config=config)))


def test_h3_cache_status_uses_mmh3_layout_and_embedded_audio(tmp_path):
    image_dir = tmp_path / "images"
    image_cache = tmp_path / "image_cache"
    video_dir = tmp_path / "videos"
    video_cache = tmp_path / "video_cache"
    for path in (image_dir, image_cache, video_dir, video_cache):
        path.mkdir()

    (image_dir / "still.png").write_bytes(b"image")
    (image_cache / "still_1024x0768_mmh3.safetensors").write_bytes(b"latent")
    (image_cache / "still_mmh3_te.safetensors").write_bytes(b"text")

    (video_dir / "clip.mp4").write_bytes(b"video")
    (video_cache / "clip_00000-022_0384x0384_mmh3.safetensors").write_bytes(b"video-and-audio-latents")
    (video_cache / "clip_00000-022_mmh3_te.safetensors").write_bytes(b"text")

    config = ProjectConfig()
    config.caching.model_type = "minimax_h3"
    config.dataset.datasets = [
        DatasetEntry(type="image", directory=str(image_dir), cache_directory=str(image_cache)),
        DatasetEntry(type="video", directory=str(video_dir), cache_directory=str(video_cache)),
    ]

    result = asyncio.run(cache_status(_request(config)))

    assert result.totals["source_count"] == 2
    assert result.totals["latent_count"] == 2
    assert result.totals["text_count"] == 2
    assert result.totals["audio_count"] == 1
    assert result.totals["missing_latent"] == 0
    assert result.totals["missing_text"] == 0
    assert result.totals["missing_audio"] == 0
    assert result.rows[0].audio_count == 0
    assert result.rows[1].audio_count == 1


def test_ltx2_cache_status_keeps_separate_audio_layout(tmp_path):
    source_dir = tmp_path / "videos"
    cache_dir = tmp_path / "cache"
    source_dir.mkdir()
    cache_dir.mkdir()

    (source_dir / "clip.mp4").write_bytes(b"video")
    (cache_dir / "clip_0512x0288_ltx2.safetensors").write_bytes(b"latent")
    (cache_dir / "clip_ltx2_te.safetensors").write_bytes(b"text")
    (cache_dir / "clip_ltx2_audio.safetensors").write_bytes(b"audio")

    config = ProjectConfig()
    config.caching.model_type = "ltx2"
    config.caching.ltx2_mode = "av"
    config.dataset.datasets = [DatasetEntry(type="video", directory=str(source_dir), cache_directory=str(cache_dir))]

    result = asyncio.run(cache_status(_request(config)))

    assert result.totals["latent_count"] == 1
    assert result.totals["text_count"] == 1
    assert result.totals["audio_count"] == 1
    assert result.totals["missing_latent"] == 0
    assert result.totals["missing_text"] == 0
    assert result.totals["missing_audio"] == 0
