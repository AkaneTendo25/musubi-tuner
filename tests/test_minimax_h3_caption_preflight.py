import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from musubi_tuner.dataset.datasources import ImageDirectoryDatasource, ImageJsonlDatasource, VideoDirectoryDatasource
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.caption_preflight import require_caption_preflight, scan_caption_preflight
from musubi_tuner.minimax_h3.dataset import create_h3_dataset_group


def group(source):
    return SimpleNamespace(datasets=[SimpleNamespace(datasource=source)])


def test_directory_reports_missing_and_blank_without_media(tmp_path):
    (tmp_path / "missing.png").write_bytes(b"not an image")
    (tmp_path / "blank.png").write_bytes(b"not an image")
    (tmp_path / "blank.txt").write_text(" \n\t", encoding="utf-8")
    source = ImageDirectoryDatasource(str(tmp_path), ".txt")
    report = scan_caption_preflight(group(source))
    assert report.checked == 2
    assert {issue.kind for issue in report.issues} == {"empty", "missing"}
    assert str(tmp_path / "missing.txt") in report.summary()
    assert str(tmp_path / "blank.txt") in report.summary()
    with pytest.raises(ValueError, match="caption preflight"):
        require_caption_preflight(group(source), report=report)


def test_jsonl_reports_all_invalid_captions_without_mutation(tmp_path):
    path = tmp_path / "data.jsonl"
    records = [
        {"image_path": "missing.png"},
        {"image_path": "null.png", "caption": None},
        {"image_path": "blank.png", "caption": "  "},
        {"image_path": "bad.png", "caption": 42},
        {"image_path": "ok.png", "caption": "hello"},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    source = ImageJsonlDatasource(str(path))
    original = [record.copy() for record in source.data]
    report = scan_caption_preflight(group(source))
    assert report.checked == 5
    assert [issue.kind for issue in report.issues] == ["missing", "missing", "empty", "non-string"]
    assert str(path) in report.summary()
    assert "record 1" in report.summary()
    assert "record 4" in report.summary()
    with pytest.raises(ValueError):
        require_caption_preflight(group(source), report)
    assert source.data == original


def test_encoding_error_reports_without_media_decode(tmp_path):
    (tmp_path / "x.mp4").write_bytes(b"not video")
    (tmp_path / "x.txt").write_bytes(b"\xff")
    source = VideoDirectoryDatasource(str(tmp_path), ".txt")
    report = scan_caption_preflight(group(source))
    assert report.issues[0].kind == "unreadable"
    with pytest.raises(ValueError, match="UnicodeDecodeError"):
        require_caption_preflight(group(source), report)


def test_permission_errors_are_reported_for_every_item():
    class Source:
        def __len__(self):
            return 2

        def get_item_key(self, index):
            return f"item{index}.mp4"

        def get_caption(self, index):
            raise PermissionError("denied")

    report = scan_caption_preflight(group(Source()))
    assert len(report.issues) == 2
    assert all(issue.kind == "unreadable" for issue in report.issues)


def test_audio_records_checked_without_mutation():
    dataset = SimpleNamespace(records=[("x.wav", "  "), ("y.wav", "valid")], caption_extension=".txt")
    with pytest.raises(ValueError, match="x.txt"):
        require_caption_preflight(SimpleNamespace(datasets=[dataset]))
    assert dataset.records == [("x.wav", "  "), ("y.wav", "valid")]


def test_real_h3_group_retains_all_missing_image_targets(tmp_path):
    image = tmp_path / "missing.png"
    image.write_bytes(b"media must not be decoded by caption preflight")
    config = {
        "general": {"resolution": [64, 64], "caption_extension": ".txt"},
        "datasets": [
            {"target_image_directory": str(tmp_path), "target_modalities": ["image"], "cache_directory": str(tmp_path / "cache")}
        ],
    }
    dataset_group, adapter = create_h3_dataset_group(config, Namespace(debug_dataset=False, h3_caption_preflight=True))
    assert dataset_group.datasets[0].datasource.image_paths == [str(image)]
    report = scan_caption_preflight(dataset_group)
    assert report.checked == 1
    assert report.issues[0].kind == "missing"
    with pytest.raises(ValueError, match="missing.txt"):
        require_caption_preflight(dataset_group, report)
    item = ItemInfo(str(image), "valid", (64, 64), (64, 64))
    assert adapter.attach(item)[0].path == image
    assert not image.with_suffix(".txt").exists()


@pytest.mark.parametrize("caption,loads", [(None, False), (" \n", False), (42, False), ("valid caption", True)])
def test_cli_preflight_runs_before_encoder_factory(tmp_path, monkeypatch, caption, loads):
    import musubi_tuner.minimax_h3_cache_text_encoder_outputs as cli

    path = tmp_path / "data.jsonl"
    record = {"image_path": "missing.png"}
    if caption is not None:
        record["caption"] = caption
    path.write_text(json.dumps(record), encoding="utf-8")
    source = ImageJsonlDatasource(str(path))
    monkeypatch.setattr(cli.config_utils, "load_user_config", lambda path: {})
    monkeypatch.setattr(cli, "create_h3_dataset_group", lambda config, args: (group(source), SimpleNamespace()))
    monkeypatch.setattr(cli.cache_text_encoder_outputs, "prepare_cache_files_and_paths", lambda datasets: ({}, {}))
    calls = []

    class EncoderReached(Exception):
        pass

    def factory(**kwargs):
        calls.append(kwargs)
        raise EncoderReached

    monkeypatch.setattr(cli, "create_conditioning_encoder", factory)
    with pytest.raises(EncoderReached if loads else SystemExit):
        cli.main(["--dataset_config", "dataset.toml", "--text_encoder", "encoder", "--device", "cpu"])
    assert bool(calls) is loads
