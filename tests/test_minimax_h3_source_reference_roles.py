import json
from argparse import Namespace
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.gui_dashboard.project_schema import DatasetEntry, ProjectConfig
from musubi_tuner.gui_dashboard.routers.stats import _h3_dataset_rows
from musubi_tuner.gui_dashboard.toml_export import build_h3_dataset_toml_document, render_toml
from musubi_tuner.gui_dashboard.validation import _h3_required_vaes, validate_cache_latents_config
from musubi_tuner.minimax_h3 import dataset as h3_dataset
from musubi_tuner.minimax_h3.media import MediaModality
from musubi_tuner.minimax_h3.references import reference_fingerprint


def _media_dir(root: Path, *names: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(name.encode())
    return root


def _native_config(target_kind: str, target_dir: Path, cache_dir: Path, **source) -> dict:
    target_modalities = ["audio"] if target_kind == "audio" else [target_kind]
    return {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                f"target_{target_kind}_directory": str(target_dir),
                "target_modalities": target_modalities,
                "cache_directory": str(cache_dir),
                "target_frames": [22],
                **source,
            }
        ],
    }


def _attached_assets(adapter, target: Path):
    is_image = target.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    bucket_resolution = (512, 512) if is_image else (512, 512, 22)
    item = ItemInfo(
        str(target),
        "prompt",
        (512, 512),
        bucket_resolution,
        frame_count=1 if is_image else 22,
    )
    assets = adapter.attach(item)
    return assets, item


def _dashboard_config(tmp_path: Path, entry: DatasetEntry) -> ProjectConfig:
    config = ProjectConfig(project_dir=str(tmp_path))
    config.caching.model_type = "minimax_h3"
    config.caching.h3_task = "ref2va"
    config.caching.h3_video_vae = "video-vae.safetensors"
    config.caching.h3_audio_vae = "audio-vae.safetensors"
    config.dataset.datasets = [entry]
    return config


def test_ordered_roles_resolve_two_video_references_and_change_required_vaes(tmp_path: Path) -> None:
    targets = _media_dir(tmp_path / "targets", "scene.wav")
    videos = _media_dir(tmp_path / "videos", "scene.mp4", "scene_0.mp4")
    config = _native_config(
        "audio",
        targets,
        tmp_path / "cache",
        source_video_directory=str(videos),
        source_modalities=["video"],
        source_reference_modalities=["video", "audio"],
    )

    adapter = h3_dataset.H3DatasetAdapter(config, Namespace(h3_image_mode="none"))
    assets, item = _attached_assets(adapter, targets / "scene.wav")
    references = assets[1:]

    assert [(asset.path, asset.modality) for asset in references] == [
        (videos / "scene.mp4", MediaModality.VIDEO),
        (videos / "scene_0.mp4", MediaModality.AUDIO),
    ]
    assert references[0].metadata.get("include_audio") is False
    assert "audio_path" not in references[0].metadata
    assert adapter.requires_video is True
    assert adapter.requires_audio is True
    assert item.h3_cache_metadata["reference_fingerprint"] == reference_fingerprint(references)


def test_role_list_uses_image_video_audio_group_order(tmp_path: Path) -> None:
    targets = _media_dir(tmp_path / "targets", "scene.wav")
    images = _media_dir(tmp_path / "images", "scene.png")
    videos = _media_dir(tmp_path / "videos", "scene.mp4", "scene_0.mp4")
    audios = _media_dir(tmp_path / "audios", "scene.wav")
    config = _native_config(
        "audio",
        targets,
        tmp_path / "cache",
        source_image_directory=str(images),
        source_video_directory=str(videos),
        source_audio_directory=str(audios),
        source_modalities=["image", "video", "audio"],
        source_reference_modalities=["video", "av", "audio", "audio"],
    )

    adapter = h3_dataset.H3DatasetAdapter(config, Namespace(h3_image_mode="none"))
    references = _attached_assets(adapter, targets / "scene.wav")[0][1:]

    assert [asset.path for asset in references] == [
        images / "scene.png",
        videos / "scene.mp4",
        videos / "scene_0.mp4",
        audios / "scene.wav",
    ]
    assert [asset.modality for asset in references] == [
        MediaModality.IMAGE,
        MediaModality.VIDEO,
        MediaModality.AUDIO,
        MediaModality.AUDIO,
    ]
    assert references[1].metadata["include_audio"] is True


@pytest.mark.parametrize(
    ("mode", "expected_modalities", "requires_video", "requires_audio"),
    [
        ("video", [MediaModality.IMAGE, MediaModality.VIDEO], True, False),
        ("av", [MediaModality.IMAGE, MediaModality.VIDEO], True, True),
        ("audio", [MediaModality.IMAGE, MediaModality.AUDIO], True, True),
    ],
)
def test_video_directory_mode_preserves_independent_images_and_effective_requirements(
    tmp_path: Path,
    mode: str,
    expected_modalities: list[MediaModality],
    requires_video: bool,
    requires_audio: bool,
) -> None:
    targets = _media_dir(tmp_path / "targets", "scene.png")
    images = _media_dir(tmp_path / "images", "scene.png")
    videos = _media_dir(tmp_path / "videos", "scene.mp4")
    config = _native_config(
        "image",
        targets,
        tmp_path / "cache",
        source_image_directory=str(images),
        source_video_directory=str(videos),
        source_modalities=[
            "image",
            *([] if mode == "audio" else ["video"]),
            *([] if mode == "video" else ["audio"]),
        ],
        source_video_audio_mode=mode,
    )

    adapter = h3_dataset.H3DatasetAdapter(config, Namespace(h3_image_mode="none"))
    references = _attached_assets(adapter, targets / "scene.png")[0][1:]

    assert [asset.modality for asset in references] == expected_modalities
    assert adapter.requires_video is requires_video
    assert adapter.requires_audio is requires_audio
    if mode == "av":
        assert references[1].metadata["include_audio"] is True


def test_fixed_roles_on_paired_source_preserve_external_audio_contract(tmp_path: Path) -> None:
    targets = _media_dir(tmp_path / "targets", "scene.wav")
    videos = _media_dir(tmp_path / "videos", "scene.mp4")
    audios = _media_dir(tmp_path / "audios", "scene.wav")
    config = _native_config(
        "audio",
        targets,
        tmp_path / "cache",
        source_video_directory=str(videos),
        source_audio_directory=str(audios),
        source_modalities=["video", "audio"],
        source_video_audio_paired=True,
        source_reference_modality="audio",
    )

    adapter = h3_dataset.H3DatasetAdapter(config, Namespace(h3_image_mode="none"))
    reference = _attached_assets(adapter, targets / "scene.wav")[0][1]

    assert reference.path == audios / "scene.wav"
    assert reference.modality is MediaModality.AUDIO
    assert adapter.requires_video is False
    assert adapter.requires_audio is True

    config["datasets"][0]["source_reference_modality"] = "av"
    av_adapter = h3_dataset.H3DatasetAdapter(config, Namespace(h3_image_mode="none"))
    av_reference = _attached_assets(av_adapter, targets / "scene.wav")[0][1]
    assert av_reference.path == videos / "scene.mp4"
    assert av_reference.modality is MediaModality.VIDEO
    assert av_reference.metadata["audio_path"] == str(audios / "scene.wav")
    assert av_reference.metadata["include_audio"] is True
    assert av_adapter.requires_video is True
    assert av_adapter.requires_audio is True


@pytest.mark.parametrize(
    ("carrier", "role"),
    [("image", "audio"), ("image", "av"), ("audio", "video"), ("audio", "av")],
)
def test_incompatible_reference_carrier_roles_fail_at_native_resolution(tmp_path: Path, carrier: str, role: str) -> None:
    suffix = {"image": ".png", "audio": ".wav"}[carrier]
    targets = _media_dir(tmp_path / "targets", "scene.mp4")
    sources = _media_dir(tmp_path / carrier, f"scene{suffix}")
    config = _native_config(
        "video",
        targets,
        tmp_path / "cache",
        **{
            f"source_{carrier}_directory": str(sources),
            "source_modalities": [carrier],
            "source_reference_modality": role,
        },
    )

    with pytest.raises(ValueError, match="source reference modality"):
        h3_dataset.H3DatasetAdapter(config, Namespace(h3_image_mode="none"))


def test_ordered_roles_require_the_same_exact_reference_count_for_every_target(tmp_path: Path) -> None:
    targets = _media_dir(tmp_path / "targets", "one.mp4", "two.mp4")
    videos = _media_dir(tmp_path / "videos", "one.mp4", "two.mp4", "two_0.mp4")
    config = _native_config(
        "video",
        targets,
        tmp_path / "cache",
        source_video_directory=str(videos),
        source_modalities=["video"],
        source_reference_modalities=["video", "audio"],
    )

    with pytest.raises(ValueError, match="exactly 1 entries"):
        h3_dataset.H3DatasetAdapter(config, Namespace(h3_image_mode="none"))


def test_reference_roles_inherit_from_general_and_reject_cross_section_conflict(tmp_path: Path) -> None:
    targets = _media_dir(tmp_path / "targets", "scene.wav")
    videos = _media_dir(tmp_path / "videos", "scene.mp4")
    config = _native_config(
        "audio",
        targets,
        tmp_path / "cache",
        source_video_directory=str(videos),
        source_modalities=["video"],
    )
    config["general"]["source_reference_modality"] = "audio"

    adapter = h3_dataset.H3DatasetAdapter(config, Namespace(h3_image_mode="none"))
    assert _attached_assets(adapter, targets / "scene.wav")[0][1].modality is MediaModality.AUDIO

    config["datasets"][0]["source_reference_modalities"] = ["video"]
    with pytest.raises(ValueError, match="source_reference_modality.*source_reference_modalities"):
        h3_dataset.H3DatasetAdapter(config, Namespace(h3_image_mode="none"))


def test_dataset_role_rejects_jsonl_per_record_role_layer(tmp_path: Path) -> None:
    target = tmp_path / "scene.mp4"
    reference = tmp_path / "reference.mp4"
    target.write_bytes(b"target")
    reference.write_bytes(b"reference")
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_path": str(target),
                "caption": "prompt",
                "control_path": str(reference),
                "control_modality_0": "video",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "video_jsonl_file": str(manifest),
                "target_modalities": ["video"],
                "source_reference_modality": "av",
                "target_frames": [22],
            }
        ],
    }

    with pytest.raises(ValueError, match="source_reference_modality.*control_modality_N"):
        h3_dataset.H3DatasetAdapter(config, Namespace(h3_image_mode="none"))


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            {
                "source_video_audio_mode": "audio",
                "source_modality_probabilities": [0.0, 1.0, 0.0],
            },
            "video variant",
        ),
        (
            {
                "source_video_audio_mode": "video",
                "source_modality_probabilities": [0.0, 0.0, 1.0],
            },
            "audio variant",
        ),
        (
            {
                "source_reference_modality": "video",
                "source_modality_probabilities": [0.0, 1.0, 0.0],
            },
            "cannot be combined",
        ),
        (
            {"source_reference_modality": "audio", "aligned_guide_indices": [0]},
            "must select a video reference",
        ),
    ],
)
def test_role_layer_enforces_dropout_and_aligned_guide_constraints(tmp_path: Path, source: dict, message: str) -> None:
    targets = _media_dir(tmp_path / "targets", "scene.wav")
    videos = _media_dir(tmp_path / "videos", "scene.mp4")
    source_modalities = ["audio"] if source.get("source_video_audio_mode") == "audio" else ["video"]
    config = _native_config(
        "audio",
        targets,
        tmp_path / "cache",
        source_video_directory=str(videos),
        source_modalities=source_modalities,
        **source,
    )

    with pytest.raises(ValueError, match=message):
        h3_dataset.H3DatasetAdapter(config, Namespace(h3_image_mode="none"))


@pytest.mark.parametrize("ordered", [False, True])
def test_av_role_changes_same_path_fingerprint_and_alignment_strips_audio(tmp_path: Path, caplog, ordered: bool) -> None:
    targets = _media_dir(tmp_path / "targets", "scene.png")
    videos = _media_dir(tmp_path / "videos", "scene.mp4")

    def resolved(role: str, *, aligned: bool = False):
        config = _native_config(
            "image",
            targets,
            tmp_path / f"cache-{role}-{aligned}",
            source_video_directory=str(videos),
            source_modalities=["video"],
            **({"source_reference_modalities": [role]} if ordered else {"source_reference_modality": role}),
            **({"aligned_guide_indices": [0]} if aligned else {}),
        )
        adapter = h3_dataset.H3DatasetAdapter(config, Namespace(h3_image_mode="none"))
        return adapter, _attached_assets(adapter, targets / "scene.png")[0][1]

    video_adapter, video = resolved("video")
    av_adapter, av = resolved("av")
    aligned_adapter, aligned = resolved("av", aligned=True)

    assert video.path == av.path == aligned.path == videos / "scene.mp4"
    assert video.metadata.get("include_audio") is False
    assert av.metadata["include_audio"] is True
    assert av_adapter.requires_audio is True
    assert video_adapter.requires_audio is False
    assert reference_fingerprint((video,)) != reference_fingerprint((av,))
    assert aligned.metadata["include_audio"] is False
    assert "audio_path" not in aligned.metadata
    assert aligned.metadata["aligned_to_target"] is True
    assert reference_fingerprint((aligned,)) != reference_fingerprint((av,))
    assert aligned_adapter.requires_video is True
    assert aligned_adapter.requires_audio is False
    assert "video-only guide" in caplog.text
    assert "audio contribution is ignored" in caplog.text


def test_dashboard_required_vaes_account_for_aligned_ordered_av_roles(tmp_path: Path) -> None:
    def required_vaes(aligned_guide_indices: str) -> tuple[bool, bool]:
        return _h3_required_vaes(
            _dashboard_config(
                tmp_path,
                DatasetEntry(
                    type="image",
                    directory="targets",
                    source_video_directory="references",
                    source_reference_modalities="av;video",
                    aligned_guide_indices=aligned_guide_indices,
                ),
            )
        )

    assert required_vaes("") == (True, True)
    assert required_vaes("0") == (True, False)


def test_v5_export_round_trips_ordered_roles_into_native_resolution(tmp_path: Path) -> None:
    targets = _media_dir(tmp_path / "targets", "scene.wav")
    videos = _media_dir(tmp_path / "videos", "scene.mp4", "scene_0.mp4")
    entry = DatasetEntry(
        type="audio",
        directory=str(targets),
        cache_directory=str(tmp_path / "cache"),
        target_frames=22,
        h3_target_modalities="audio",
        source_video_directory=str(videos),
        source_reference_modalities="video;audio",
    )
    config = _dashboard_config(tmp_path, entry)

    document = build_h3_dataset_toml_document(config, include_training=True, include_validation=False)
    round_tripped = tomllib.loads(render_toml(document))
    dataset = round_tripped["datasets"][0]

    assert dataset["source_reference_modalities"] == ["video", "audio"]
    assert dataset["source_modalities"] == ["video"]
    h3_dataset._normalize_explicit_modality_config(round_tripped)
    adapter = h3_dataset.H3DatasetAdapter(round_tripped, Namespace(h3_image_mode="none"))
    references = _attached_assets(adapter, targets / "scene.wav")[0][1:]
    assert [asset.modality for asset in references] == [MediaModality.VIDEO, MediaModality.AUDIO]


def test_export_keeps_training_and_validation_reference_roles_independent(tmp_path: Path) -> None:
    config = _dashboard_config(
        tmp_path,
        DatasetEntry(
            type="video",
            directory="train",
            target_frames=22,
            source_video_directory="train-references",
            source_reference_modality="video",
        ),
    )
    config.dataset.validation_datasets = [
        DatasetEntry(
            type="audio",
            directory="validation",
            target_frames=22,
            source_video_directory="validation-references",
            source_reference_modalities="av;audio",
        )
    ]

    document = build_h3_dataset_toml_document(config, include_training=True, include_validation=True)

    assert document["datasets"][0]["source_reference_modality"] == "video"
    assert "source_reference_modalities" not in document["datasets"][0]
    assert document["datasets"][1]["source_reference_modalities"] == ["av", "audio"]
    assert "source_reference_modality" not in document["datasets"][1]


@pytest.mark.parametrize(
    ("version", "row", "expected"),
    [
        (
            3,
            {
                "type": "video",
                "directory": "targets",
                "control_video_directory": "videos",
                "control_audio_directory": "audios",
                "control_modality": "audio",
            },
            {
                "source_video_directory": "videos",
                "source_audio_directory": "audios",
                "source_video_audio_paired": True,
                "source_reference_modality": "audio",
            },
        ),
        (
            3,
            {
                "type": "video",
                "directory": "targets",
                "control_video_directory": "videos",
                "control_modality": "audio",
            },
            {"source_video_directory": "videos", "source_video_audio_mode": "audio"},
        ),
        (
            3,
            {
                "type": "video",
                "directory": "targets",
                "control_video_directory": "videos",
                "control_modality": "av",
            },
            {"source_video_directory": "videos", "source_video_audio_mode": "av"},
        ),
        (
            3,
            {
                "type": "audio",
                "directory": "targets",
                "control_audio_directory": "audios",
                "control_modality": "audio",
            },
            {"source_audio_directory": "audios"},
        ),
        (
            4,
            {
                "type": "video",
                "directory": "targets",
                "source_video_directory": "videos",
                "control_modality": "video",
            },
            {"source_reference_modality": "video"},
        ),
        (
            4,
            {
                "type": "video",
                "directory": "targets",
                "source_video_directory": "videos",
                "control_modalities": "video;audio",
            },
            {"source_reference_modalities": "video;audio"},
        ),
        (
            4,
            {
                "type": "video",
                "directory": "targets",
                "source_video_directory": "videos",
                "source_video_audio_embedded": True,
            },
            {"source_video_audio_embedded": True},
        ),
    ],
)
def test_project_v5_migration_preserves_legacy_reference_intent(version: int, row: dict, expected: dict) -> None:
    config = ProjectConfig.model_validate({"version": version, "dataset": {"datasets": [row]}})
    migrated = config.dataset.datasets[0].model_dump()

    assert config.version == 5
    for key, value in expected.items():
        assert migrated[key] == value
    assert migrated["control_modality"] == ""
    assert migrated["control_modalities"] == ""


def test_default_v4_project_migration_preserves_legacy_visual_video_semantics() -> None:
    config = ProjectConfig.model_validate(
        {
            "version": 4,
            "dataset": {
                "datasets": [
                    {
                        "type": "video",
                        "directory": "targets",
                        "source_video_directory": "references",
                    }
                ]
            },
        }
    )
    row = config.dataset.datasets[0]
    document = build_h3_dataset_toml_document(config, include_training=True, include_validation=False)["datasets"][0]

    assert config.version == 5
    assert row.source_video_audio_mode == ""
    assert row.source_reference_modality == ""
    assert row.source_reference_modalities == ""
    assert "source_video_audio_mode" not in document
    assert "source_reference_modality" not in document
    assert document["source_modalities"] == ["video"]


@pytest.mark.parametrize("collection", ["datasets", "validation_datasets"])
def test_v4_conflicting_legacy_roles_migrate_ordered_roles_with_warning(tmp_path: Path, caplog, collection: str) -> None:
    config = ProjectConfig.model_validate(
        {
            "version": 4,
            "dataset": {
                collection: [
                    {
                        "type": "video",
                        "directory": "targets",
                        "source_video_directory": "references",
                        "control_modality": "video",
                        "control_modalities": "video;audio",
                    }
                ]
            },
        }
    )
    config.project_dir = str(tmp_path)
    config.caching.model_type = "minimax_h3"
    row = getattr(config.dataset, collection)[0]

    assert row.control_modality == ""
    assert row.control_modalities == ""
    assert row.source_reference_modality == ""
    assert row.source_reference_modalities == "video;audio"
    assert "explicit ordered roles take precedence" in caplog.text
    document = build_h3_dataset_toml_document(config, include_training=True, include_validation=True)
    assert document["datasets"][0]["source_reference_modalities"] == ["video", "audio"]


@pytest.mark.parametrize("roles", ["video;;audio", "video;depth", ";audio", "video;", "video audio"])
def test_v4_legacy_ordered_roles_reject_invalid_entries(roles: str) -> None:
    with pytest.raises(ValueError, match="control_modalities must contain only"):
        ProjectConfig.model_validate(
            {"version": 4, "dataset": {"datasets": [{"control_modality": "video", "control_modalities": roles}]}}
        )


@pytest.mark.parametrize(
    "roles",
    [
        {"control_modalities": "video;audio", "source_reference_modality": "video"},
        {"control_modalities": "video;audio", "source_reference_modalities": "av;audio"},
        {"control_modality": "video", "source_reference_modality": "audio"},
        {"control_modality": "video", "source_reference_modalities": "video;audio"},
    ],
)
def test_v4_legacy_roles_reject_conflicting_new_fields(roles: dict) -> None:
    with pytest.raises(ValueError, match="conflicts with existing source reference roles"):
        ProjectConfig.model_validate({"version": 4, "dataset": {"datasets": [roles]}})


def test_v4_legacy_ordered_roles_preserve_matching_new_roles() -> None:
    config = ProjectConfig.model_validate(
        {
            "version": 4,
            "dataset": {"datasets": [{"control_modalities": " VIDEO, audio ", "source_reference_modalities": "video;audio"}]},
        }
    )
    row = config.dataset.datasets[0]
    assert row.source_reference_modalities == "video;audio"
    assert row.control_modalities == ""


@pytest.mark.parametrize(
    ("changes", "expected_field"),
    [
        ({"source_reference_modalities": "video;;audio"}, "source_reference_modalities"),
        ({"source_reference_modalities": "video;depth"}, "source_reference_modalities"),
        ({"source_reference_modality": "depth"}, "source_reference_modality"),
        ({"source_video_audio_mode": "depth"}, "source_video_audio_mode"),
        (
            {"source_reference_modality": "video", "source_reference_modalities": "video"},
            "source_reference_modalities",
        ),
        (
            {"source_reference_modality": "video", "source_video_audio_mode": "video"},
            "source_video_audio_mode",
        ),
        (
            {
                "source_reference_modality": "video",
                "control_modality_probability_av": 0.0,
                "control_modality_probability_video": 1.0,
                "control_modality_probability_audio": 0.0,
            },
            "control_modality_probability_av",
        ),
        (
            {"source_video_audio_mode": "video", "source_video_directory": ""},
            "source_video_audio_mode",
        ),
        ({"source_video_audio_mode": "audio", "source_audio_directory": "audio"}, "source_video_audio_mode"),
        ({"source_video_audio_mode": "av", "source_video_audio_embedded": True}, "source_video_audio_mode"),
        (
            {
                "source_video_audio_mode": "video",
                "source_video_audio_paired": True,
                "source_audio_directory": "audio",
            },
            "source_video_audio_mode",
        ),
        ({"source_reference_modality": "audio", "aligned_guide_indices": "0"}, "aligned_guide_indices"),
    ],
)
def test_dashboard_validation_reports_reference_role_fields(changes: dict, expected_field: str, tmp_path: Path) -> None:
    entry = DatasetEntry(
        type="video",
        directory="targets",
        target_frames=22,
        source_video_directory="references",
    )
    for key, value in changes.items():
        setattr(entry, key, value)
    config = _dashboard_config(tmp_path, entry)
    report = validate_cache_latents_config(config)

    assert f"dataset.datasets[0].{expected_field}" in report["field_errors"]


def test_server_stats_use_effective_fixed_and_ordered_reference_roles() -> None:
    training = {"reference_video_short_edge": 768, "reference_video_max_pixels": 768 * 1344}
    caching = {"h3_task": "ref2va"}
    base = {
        "type": "audio",
        "target_frames": 124,
        "resolution_w": 832,
        "resolution_h": 480,
        "source_video_directory": "references",
    }

    video_rows = _h3_dataset_rows(training, caching, {**base, "source_reference_modality": "video"})
    av_rows = _h3_dataset_rows(training, caching, {**base, "source_reference_modality": "av"})
    audio_rows = _h3_dataset_rows(training, caching, {**base, "source_reference_modality": "audio"})
    ordered_rows = _h3_dataset_rows(
        training,
        caching,
        {**base, "source_reference_modalities": "video;av;audio"},
    )

    assert av_rows > video_rows
    assert av_rows > audio_rows
    assert ordered_rows > av_rows
