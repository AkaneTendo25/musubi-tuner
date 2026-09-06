import json
from pathlib import Path

import av
import numpy as np
import pytest
import toml
from PIL import Image

from musubi_tuner.minimax_h3_prepare_rollout_teacher import _uniform_indices, prepare_teacher


def _write_config(path: Path, dataset: dict) -> None:
    path.write_text(toml.dumps({"general": {"batch_size": 1}, "datasets": [dataset]}), encoding="utf-8")


def test_uniform_indices_are_distinct_and_deterministic():
    assert _uniform_indices(12, 2) == (3, 9)
    assert _uniform_indices(5, 5) == (0, 1, 2, 3, 4)


def test_prepare_fl2va_reuses_student_latents(tmp_path: Path):
    config = tmp_path / "student.toml"
    student_cache = tmp_path / "student_cache"
    _write_config(
        config,
        {"target_video_directory": str(tmp_path / "targets"), "cache_directory": str(student_cache)},
    )

    teacher_config, mode = prepare_teacher(config, tmp_path / "teacher")

    result = toml.load(teacher_config)["datasets"][0]
    assert mode == "fl2va"
    assert result["latent_cache_directory"] == str(student_cache.resolve())
    assert result["cache_directory"] != str(student_cache.resolve())


def test_explicit_mode_cannot_override_detected_dataset_type(tmp_path: Path):
    config = tmp_path / "student.toml"
    _write_config(config, {"cache_directory": str(tmp_path / "student_cache")})

    with pytest.raises(ValueError, match="conflicts with the detected fl2va"):
        prepare_teacher(config, tmp_path / "teacher", mode="ref2va")


def test_prepare_ref2va_copies_shared_and_adds_target_frames(tmp_path: Path):
    targets = tmp_path / "targets"
    references = tmp_path / "student_refs"
    targets.mkdir()
    references.mkdir()
    (targets / "clip.mp4").write_bytes(b"video")
    (references / "clip_0.png").write_bytes(b"shared")
    config = tmp_path / "student.toml"
    _write_config(
        config,
        {
            "target_video_directory": str(targets),
            "source_image_directory": str(references),
            "source_modalities": ["image"],
            "cache_directory": str(tmp_path / "student_cache"),
        },
    )

    def fake_extract(_video: Path, destinations: list[Path]) -> None:
        for index, destination in enumerate(destinations):
            destination.write_bytes(f"frame {index}".encode())

    teacher_config, mode = prepare_teacher(
        config, tmp_path / "teacher", teacher_frames=2, extractor=fake_extract
    )

    result = toml.load(teacher_config)["datasets"][0]
    teacher_refs = Path(result["source_image_directory"])
    assert mode == "ref2va"
    assert sorted(path.name for path in teacher_refs.iterdir()) == ["clip_0.png", "clip_1.png", "clip_2.png"]
    assert (teacher_refs / "clip_0.png").read_bytes() == b"shared"
    assert "latent_cache_directory" not in result
    assert result["cache_directory"] != str((tmp_path / "student_cache").resolve())


def test_prepare_ref2va_extracts_real_video_frames(tmp_path: Path):
    targets = tmp_path / "targets"
    targets.mkdir()
    video = targets / "clip.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for value in range(8):
            pixels = np.full((16, 16, 3), value * 24, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    config = tmp_path / "student.toml"
    _write_config(
        config,
        {
            "target_video_directory": str(targets),
            "source_video_directory": str(tmp_path / "video_refs"),
            "source_modalities": ["video"],
            "cache_directory": str(tmp_path / "student_cache"),
        },
    )

    teacher_config, _ = prepare_teacher(config, tmp_path / "teacher", teacher_frames=2)

    result = toml.load(teacher_config)["datasets"][0]
    extracted = sorted(Path(result["source_image_directory"]).glob("clip_*.png"))
    assert len(extracted) == 2
    assert all(Image.open(path).size == (16, 16) for path in extracted)


def test_prepare_ref2va_jsonl_preserves_references_and_appends_frames(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    shared = tmp_path / "shared.png"
    shared.write_bytes(b"shared")
    jsonl = tmp_path / "student.jsonl"
    jsonl.write_text(
        json.dumps({"video_path": "clip.mp4", "caption": "a clip", "control_path_0": "shared.png"}) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "student.toml"
    _write_config(config, {"video_jsonl_file": str(jsonl), "cache_directory": str(tmp_path / "student_cache")})

    def fake_extract(_video: Path, destinations: list[Path]) -> None:
        for destination in destinations:
            destination.write_bytes(b"frame")

    teacher_config, mode = prepare_teacher(
        config, tmp_path / "teacher", teacher_frames=2, extractor=fake_extract
    )

    dataset = toml.load(teacher_config)["datasets"][0]
    teacher_row = json.loads(Path(dataset["video_jsonl_file"]).read_text(encoding="utf-8"))
    assert mode == "ref2va"
    assert teacher_row["video_path"] == str(video.resolve())
    assert teacher_row["control_path_0"] == str(shared.resolve())
    assert Path(teacher_row["control_path_1"]).read_bytes() == b"frame"
    assert Path(teacher_row["control_path_2"]).read_bytes() == b"frame"


def test_prepare_ref2va_rejects_too_many_image_references_before_extraction(tmp_path: Path):
    targets = tmp_path / "targets"
    references = tmp_path / "student_refs"
    targets.mkdir()
    references.mkdir()
    (targets / "clip.mp4").write_bytes(b"video")
    for index in range(8):
        (references / f"clip_{index}.png").write_bytes(b"shared")
    config = tmp_path / "student.toml"
    _write_config(
        config,
        {
            "target_video_directory": str(targets),
            "source_image_directory": str(references),
            "cache_directory": str(tmp_path / "student_cache"),
        },
    )

    with pytest.raises(ValueError, match="would have 10 image references"):
        prepare_teacher(config, tmp_path / "teacher", teacher_frames=2)
