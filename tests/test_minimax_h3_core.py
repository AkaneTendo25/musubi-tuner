import json
import math
import os
import re
import struct
import wave
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from musubi_tuner import minimax_h3_cache_latents as h3_cache_latents
from musubi_tuner import minimax_h3_cache_text_encoder_outputs as h3_cache_text
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.config_utils import (
    BlueprintGenerator,
    ConfigSanitizer,
)
from musubi_tuner.dataset.datasources import ImageDirectoryDatasource, ImageJsonlDatasource, VideoJsonlDatasource
from musubi_tuner.dataset.image_video_dataset import ItemInfo, _validate_h3_cache_pair
from musubi_tuner.minimax_h3 import backend as h3_backend
from musubi_tuner.minimax_h3 import integration as h3_integration
from musubi_tuner.minimax_h3 import references as h3_references
from musubi_tuner.minimax_h3.architecture import (
    AUDIO_FLOW_SHIFT,
    AUDIO_HOP_LENGTH,
    AUDIO_LATENT_FPS,
    AUDIO_SAMPLE_RATE,
    CANVAS_MULTIPLE,
    VIDEO_FLOW_SHIFT,
    VIDEO_FPS,
    temporal_shape,
)
from musubi_tuner.minimax_h3.assets import default_text_encoder_assets
from musubi_tuner.minimax_h3.audio import (
    AudioDecodeError,
    audio_valid_mask_to_latent_mask,
    load_audio_asset,
    target_audio_processing_spec,
)
from musubi_tuner.minimax_h3.audio_dataset import H3AudioDataset
from musubi_tuner.minimax_h3.cache import (
    H3_AUDIO_LATENTS_KEY,
    H3_CONDITIONING_TASK_IDS,
    H3_CONDITIONING_TASK_KEY,
    H3_KEYFRAME_VIDEO_ROWS_KEY,
    H3_REFERENCE_IMAGE_MAX_PIXELS_KEY,
    H3_REFERENCE_IMAGE_SHORT_EDGE_KEY,
    H3_REFERENCE_IMAGE_SIZE_MODE_KEY,
    H3_REFERENCE_KINDS_KEY,
    H3_REFERENCE_TEMPORAL_CONTRACT_KEY,
    H3_REFERENCE_TEMPORAL_CONTRACT_VERSION,
    H3_REFERENCE_VIDEO_MAX_PIXELS_KEY,
    H3_REFERENCE_VIDEO_SHORT_EDGE_KEY,
    H3_TEXT_VISUAL_MAX_PIXELS_KEY,
    reference_key_suffix,
    save_latent_cache_minimax_h3,
)
from musubi_tuner.minimax_h3 import dataset as h3_dataset
from musubi_tuner.minimax_h3.dataset import create_h3_dataset_group
from musubi_tuner.minimax_h3.media import (
    AudioProcessingSpec,
    CropMode,
    MediaAsset,
    MediaModality,
    MissingMediaPolicy,
    PadMode,
    fit_audio_length,
    slice_media_asset,
)
from musubi_tuner.minimax_h3 import packing as h3_packing
from musubi_tuner.minimax_h3.model import MiniMaxH3TokenTag
from musubi_tuner.minimax_h3.packing import (
    MiniMaxH3ReferenceGeometry,
    build_ref2va_packed_sequence,
    build_t2va_packed_sequence,
)
from musubi_tuner.minimax_h3.references import (
    REFERENCE_IMAGE_SHORT_EDGE,
    REFERENCE_VIDEO_MAX_PIXELS,
    REFERENCE_VIDEO_SHORT_EDGE,
    H3PreparedReference,
    H3ReferenceKind,
    _reference_audio_asset,
    _source_frame_limit,
    reference_modality_variant,
    resample_reference_frames,
    resolve_reference_image_size,
    resolve_reference_image_area_size,
    resolve_reference_video_size,
    trim_reference_frames,
    validate_reference_image_short_edge,
)
from musubi_tuner.minimax_h3.request import H3GenerationRequest, H3Reference, ReferenceKind, ReferenceRole
from musubi_tuner.minimax_h3.weights import CheckpointInspectionError, inspect_checkpoint
from musubi_tuner.minimax_h3_cache_latents import create_parser as create_cache_latents_parser
from musubi_tuner.minimax_h3_cache_text_encoder_outputs import create_parser as create_cache_text_parser
from musubi_tuner.minimax_h3_generate_video import create_parser, request_from_args


def _write_safetensors_header(path: Path, tensors: dict) -> None:
    header = json.dumps(tensors).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def test_h3_inference_validates_lora_training_metadata(tmp_path, caplog):
    path = tmp_path / "adapter.safetensors"
    save_file(
        {"probe": torch.zeros(1)},
        path,
        metadata={
            "ss_h3_training_mode": "ref2va",
            "ss_h3_reference_image_short_edge": "448",
            "ss_h3_adaln_rank": "16",
        },
    )

    with pytest.raises(ValueError, match="trained with reference_image_short_edge=448"):
        h3_integration._validate_inference_lora_metadata(path, "ref2va", 2048)
    h3_integration._validate_inference_lora_metadata(path, "ref2va", 448)

    assert "rank-16 frozen AdaLN approximation" in caplog.text


def test_h3_inference_validates_ref2va_qwen_visual_cap_metadata(tmp_path):
    path = tmp_path / "adapter.safetensors"
    save_file(
        {"probe": torch.zeros(1)},
        path,
        metadata={
            "ss_h3_training_mode": "ref2va",
            "ss_h3_reference_image_short_edge": "2048",
            "ss_h3_text_visual_max_pixels": "65536",
        },
    )

    with pytest.raises(ValueError, match="trained with h3_text_visual_max_pixels=65536"):
        h3_integration._validate_inference_lora_metadata(path, "ref2va", 2048)
    h3_integration._validate_inference_lora_metadata(path, "ref2va", 2048, text_visual_max_pixels=65_536)


def test_h3_inference_validates_ref2va_video_sizing_metadata(tmp_path):
    path = tmp_path / "adapter.safetensors"
    save_file(
        {"probe": torch.zeros(1)},
        path,
        metadata={
            "ss_h3_training_mode": "ref2va",
            "ss_h3_reference_video_short_edge": "384",
            "ss_h3_reference_video_max_pixels": str(384 * 672),
        },
    )

    with pytest.raises(ValueError, match="reference-video sizing"):
        h3_integration._validate_inference_lora_metadata(path, "ref2va", 2048)
    h3_integration._validate_inference_lora_metadata(
        path,
        "ref2va",
        2048,
        reference_video_short_edge=384,
        reference_video_max_pixels=384 * 672,
    )


def test_public_request_modes_and_limits(tmp_path):
    t2v = H3GenerationRequest("prompt", tmp_path / "out.mp4")
    assert t2v.mode == "text_to_video"

    first = H3Reference(tmp_path / "first.png", ReferenceKind.IMAGE, ReferenceRole.FIRST_FRAME)
    assert H3GenerationRequest("prompt", tmp_path / "out.mp4", references=(first,)).mode == "first_last_frame"

    image = H3Reference(tmp_path / "style.png", ReferenceKind.IMAGE)
    audio = H3Reference(tmp_path / "voice.wav", ReferenceKind.AUDIO)
    assert H3GenerationRequest("prompt", tmp_path / "out.mp4", references=(image, audio)).mode == "reference"
    with pytest.raises(ValueError, match="5 through 15"):
        H3GenerationRequest("prompt", tmp_path / "out.mp4", duration=4)
    with pytest.raises(ValueError, match="5 through 15"):
        H3GenerationRequest("prompt", tmp_path / "out.mp4", duration=16)


def test_keyframe_conditioning_carries_its_latent_index(tmp_path):
    # Measured on the release: an interior anchor is honoured about as strongly
    # as the trained ends, so the index has to survive to the packing.
    from musubi_tuner.minimax_h3.request import make_references

    references = make_references(keyframes=[(11, str(tmp_path / "k.png"))])
    request = H3GenerationRequest("prompt", tmp_path / "out.mp4", references=references)

    assert request.mode == "first_last_frame"
    assert [(ref.role, ref.latent_index) for ref in request.references] == [(ReferenceRole.KEYFRAME, 11)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"role": ReferenceRole.KEYFRAME}, "requires a latent index"),
        ({"role": ReferenceRole.KEYFRAME, "latent_index": -1}, "non-negative integer"),
        ({"latent_index": 3}, "only to keyframe"),
    ],
)
def test_keyframe_index_is_validated(tmp_path, kwargs, message):
    from musubi_tuner.minimax_h3.request import H3Reference

    with pytest.raises(ValueError, match=message):
        H3Reference(tmp_path / "k.png", ReferenceKind.IMAGE, **kwargs)


def test_keyframe_indices_must_be_distinct(tmp_path):
    from musubi_tuner.minimax_h3.request import make_references

    references = make_references(keyframes=[(4, str(tmp_path / "a.png")), (4, str(tmp_path / "b.png"))])
    with pytest.raises(ValueError, match="only once"):
        H3GenerationRequest("prompt", tmp_path / "out.mp4", references=references)


def test_keyframe_cli_parses_index_and_path(tmp_path):
    args = create_parser().parse_args(
        ["--model", str(tmp_path), "--prompt", "p", "--output", str(tmp_path / "o.mp4"), "--keyframe", "11:/k.png"]
    )
    request = request_from_args(args)
    assert [(ref.role, ref.latent_index) for ref in request.references] == [(ReferenceRole.KEYFRAME, 11)]

    bad = create_parser().parse_args(
        ["--model", str(tmp_path), "--prompt", "p", "--output", str(tmp_path / "o.mp4"), "--keyframe", "/k.png"]
    )
    with pytest.raises(ValueError, match="INDEX:PATH"):
        request_from_args(bad)


def test_conditioned_image_cli_duplicates_first_control_and_uses_short_frame_grid(tmp_path):
    first = tmp_path / "first.png"
    args = create_parser().parse_args(
        [
            "--model",
            str(tmp_path),
            "--prompt",
            "p",
            "--output",
            str(tmp_path / "out.png"),
            "--h3_image_mode",
            "first",
            "--first_frame",
            str(first),
        ]
    )

    request = request_from_args(args)

    assert request.frame_count_override == 5
    assert request.temporal_shape.frame_count == 5
    assert [reference.role for reference in request.references] == [ReferenceRole.FIRST_FRAME, ReferenceRole.LAST_FRAME]
    assert request.references[0].path == request.references[1].path == first


def test_public_request_enforces_released_reference_caps_and_audio_pairing(tmp_path):
    audio = H3Reference(tmp_path / "voice.wav", ReferenceKind.AUDIO)
    with pytest.raises(ValueError, match="requires at least one reference image or video"):
        H3GenerationRequest("prompt", tmp_path / "out.mp4", references=(audio,))

    references = tuple(H3Reference(tmp_path / f"image_{index}.png", ReferenceKind.IMAGE) for index in range(9))
    references += tuple(H3Reference(tmp_path / f"video_{index}.mp4", ReferenceKind.VIDEO) for index in range(3))
    assert H3GenerationRequest("prompt", tmp_path / "out.mp4", references=references).mode == "reference"

    with pytest.raises(ValueError, match="at most 12 references"):
        H3GenerationRequest(
            "prompt",
            tmp_path / "out.mp4",
            references=references + (H3Reference(tmp_path / "audio.wav", ReferenceKind.AUDIO),),
        )

    too_many_images = references + (H3Reference(tmp_path / "image_9.png", ReferenceKind.IMAGE),)
    with pytest.raises(ValueError, match="9 reference images"):
        H3GenerationRequest("prompt", tmp_path / "out.mp4", references=too_many_images)


def test_verified_h3_temporal_contract():
    shape = temporal_shape(124)
    assert shape.video_latent_frames == 37
    assert shape.audio_latent_frames == 207
    assert shape.audio_samples == 207 * 800
    assert AUDIO_SAMPLE_RATE == 32000
    assert AUDIO_LATENT_FPS == 40
    assert CANVAS_MULTIPLE == 32
    assert (VIDEO_FLOW_SHIFT, AUDIO_FLOW_SHIFT) == (12.0, 3.0)
    assert temporal_shape(5).audio_latent_frames == 8
    assert temporal_shape(22).audio_latent_frames == 37
    assert temporal_shape(39).audio_latent_frames == 65
    assert temporal_shape(56).audio_latent_frames == 93

    assert H3GenerationRequest("prompt", Path("out.mp4"), duration=5).temporal_shape == shape
    with pytest.raises(ValueError, match="% 17 == 5"):
        temporal_shape(121)


def test_first_frame_cannot_be_mixed_with_reference_mode(tmp_path):
    first = H3Reference(tmp_path / "first.png", ReferenceKind.IMAGE, ReferenceRole.FIRST_FRAME)
    reference = H3Reference(tmp_path / "style.png", ReferenceKind.IMAGE)
    with pytest.raises(ValueError, match="separate H3 modes"):
        H3GenerationRequest("prompt", tmp_path / "out.mp4", references=(first, reference))


def test_cli_maps_multimodal_references(tmp_path):
    parser = create_parser()
    args = parser.parse_args(
        [
            "--model",
            str(tmp_path),
            "--prompt",
            "prompt",
            "--output",
            str(tmp_path / "out.mp4"),
            "--reference_image",
            str(tmp_path / "subject.png"),
            "--reference_audio",
            str(tmp_path / "voice.wav"),
        ]
    )
    request = request_from_args(args)
    assert request.mode == "reference"
    assert [reference.kind for reference in request.references] == [ReferenceKind.IMAGE, ReferenceKind.AUDIO]


def test_checkpoint_inventory_reads_headers_without_tensor_data(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    _write_safetensors_header(
        shard,
        {
            "transformer.blocks.0.weight": {"dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 64]},
            "vae.decoder.weight": {"dtype": "F16", "shape": [2, 3, 3], "data_offsets": [64, 100]},
        },
    )
    index = {"weight_map": {"transformer.blocks.0.weight": shard.name, "vae.decoder.weight": shard.name}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    inventory = inspect_checkpoint(tmp_path)

    assert inventory.tensors == 2
    assert inventory.parameters == 50
    assert inventory.prefixes == {"transformer": 1, "vae": 1}
    assert inventory.config_files == ("config.json",)


def test_checkpoint_inventory_rejects_missing_index_shard(tmp_path):
    index = {"weight_map": {"model.weight": "missing.safetensors"}}
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(CheckpointInspectionError, match="missing shard"):
        inspect_checkpoint(path)


def test_checkpoint_inventory_supports_multiple_component_indexes(tmp_path):
    transformer = tmp_path / "transformer"
    vae = tmp_path / "vae"
    transformer.mkdir()
    vae.mkdir()
    transformer_shard = transformer / "model.safetensors"
    vae_shard = vae / "model.safetensors"
    _write_safetensors_header(
        transformer_shard,
        {"blocks.0.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}},
    )
    _write_safetensors_header(
        vae_shard,
        {"blocks.0.weight": {"dtype": "F16", "shape": [3, 2], "data_offsets": [0, 12]}},
    )
    for directory in (transformer, vae):
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"blocks.0.weight": "model.safetensors"}}), encoding="utf-8"
        )
        (directory / "config.json").write_text("{}", encoding="utf-8")

    inventory = inspect_checkpoint(tmp_path)

    assert inventory.index_file is None
    assert inventory.index_files == (
        "transformer/model.safetensors.index.json",
        "vae/model.safetensors.index.json",
    )
    assert inventory.shards == ("transformer/model.safetensors", "vae/model.safetensors")
    assert inventory.components == {"transformer": 1, "vae": 1}
    assert inventory.config_files == ("transformer/config.json", "vae/config.json")


def test_h3_reuses_control_directory_for_mixed_reference_media(tmp_path):
    videos = tmp_path / "videos"
    controls = tmp_path / "controls"
    videos.mkdir()
    controls.mkdir()
    target = videos / "target.mp4"
    target.write_bytes(b"target")
    (controls / "target.png").write_bytes(b"image")
    (controls / "target_0.mp4").write_bytes(b"video")
    (controls / "target_1.wav").write_bytes(b"audio")

    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "video_directory": str(videos),
                "control_directory": str(controls),
                "cache_directory": str(tmp_path / "cache"),
                "target_frames": [22],
                "frame_extraction": "uniform",
            }
        ],
    }
    group, adapter = create_h3_dataset_group(config, Namespace(debug_dataset=False))
    dataset = group.datasets[0]
    assert dataset.architecture == ARCHITECTURE_MINIMAX_H3
    assert dataset.control_directory is None
    assert dataset.has_control is False

    item = ItemInfo(
        str(videos / "target_00012-022.mp4"),
        "prompt",
        (512, 512),
        (512, 512, 22),
        frame_count=22,
    )
    assets = adapter.attach(item)
    assert [(asset.modality, asset.role) for asset in assets] == [
        (MediaModality.VIDEO, "target"),
        (MediaModality.IMAGE, "reference"),
        (MediaModality.VIDEO, "reference"),
        (MediaModality.AUDIO, "reference"),
    ]
    assert assets[0].start_seconds == 0.5
    assert assets[0].duration_seconds == pytest.approx(22 / 24)
    assert assets[0].metadata == {"frame_count": 22, "fps": 24.0}


def _write_control_directory(root: Path, names) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(name.encode())
    return root


def test_h3_control_matching_never_steals_controls_from_a_sibling_target(tmp_path):
    controls = _write_control_directory(tmp_path / "controls", ("a_0.png", "a_3.png"))
    targets = [str(tmp_path / "videos" / "a_1.mp4"), str(tmp_path / "videos" / "a_2.mp4")]

    with pytest.raises(ValueError, match="ambiguous H3 controls"):
        h3_dataset._references_from_directory(str(controls), targets)


def test_h3_control_matching_keeps_blessed_layouts(tmp_path):
    videos = tmp_path / "videos"

    multi = _write_control_directory(tmp_path / "multi", ("a_0.png", "a_1.png"))
    assert h3_dataset._references_from_directory(str(multi), [str(videos / "a.mp4")]) == {
        str(videos / "a.mp4"): (multi / "a_0.png", multi / "a_1.png")
    }

    shared = _write_control_directory(tmp_path / "shared", ("a.png", "a_0.png", "a_0_0.png"))
    assert h3_dataset._references_from_directory(str(shared), [str(videos / "a.mp4"), str(videos / "a_0.mp4")]) == {
        str(videos / "a.mp4"): (shared / "a.png",),
        str(videos / "a_0.mp4"): (shared / "a_0.png", shared / "a_0_0.png"),
    }

    fallback = _write_control_directory(tmp_path / "fallback", ("b.png", "b_1.png"))
    assert h3_dataset._references_from_directory(str(fallback), [str(videos / "b_0.png")]) == {
        str(videos / "b_0.png"): (fallback / "b.png", fallback / "b_1.png")
    }

    with pytest.raises(ValueError, match="no matching H3 controls"):
        h3_dataset._references_from_directory(str(fallback), [str(videos / "c.mp4")])


def test_h3_video_jsonl_resolves_relative_control_paths_against_the_manifest(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "refs").mkdir(parents=True)
    target = data / "target.mp4"
    reference = data / "refs" / "reference.png"
    for path in (target, reference):
        path.write_bytes(b"media")
    manifest = data / "videos.jsonl"
    manifest.write_text(
        json.dumps({"video_path": str(target), "control_path_0": "refs/reference.png", "caption": "prompt"}),
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "video_jsonl_file": str(manifest),
                "cache_directory": str(tmp_path / "cache"),
                "target_frames": [22],
                "frame_extraction": "uniform",
            }
        ],
    }
    _, adapter = create_h3_dataset_group(config, Namespace(debug_dataset=False))
    item = ItemInfo(str(target), "prompt", (512, 512), (512, 512), frame_count=22)

    assets = adapter.attach(item)

    assert assets[1].path == reference.resolve()


def test_h3_image_jsonl_resolves_relative_control_paths_without_conditioned_image_mode(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "refs").mkdir(parents=True)
    target = data / "target.png"
    reference = data / "refs" / "reference.png"
    for path in (target, reference):
        path.write_bytes(b"media")
    manifest = data / "images.jsonl"
    manifest.write_text(
        json.dumps({"image_path": str(target), "control_path": "refs/reference.png", "caption": "prompt"}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [{"image_jsonl_file": str(manifest), "cache_directory": str(tmp_path / "cache")}],
    }
    _, adapter = create_h3_dataset_group(config, Namespace(debug_dataset=False, h3_image_mode="none"))
    item = ItemInfo(str(target), "prompt", (512, 512), (512, 512))

    assets = adapter.attach(item)

    assert assets[1].path == reference.resolve()


def test_h3_reference_modality_probabilities_reject_unsatisfiable_variants(tmp_path):
    targets = tmp_path / "targets"
    controls = tmp_path / "controls"
    for directory in (targets, controls):
        directory.mkdir()
    (targets / "scene.mp4").write_bytes(b"target")
    (controls / "scene.mp4").write_bytes(b"reference")
    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "video_directory": str(targets),
                "control_directory": str(controls),
                "control_modality_probabilities": [0.5, 0.25, 0.25],
                "cache_directory": str(tmp_path / "cache"),
                "target_frames": [22],
            }
        ],
    }

    with pytest.raises(ValueError, match="audio variant probability 0.25"):
        create_h3_dataset_group(config, Namespace(debug_dataset=False))

    config["datasets"][0]["control_modality_probabilities"] = [0.5, 0.5, 0.0]
    _, adapter = create_h3_dataset_group(config, Namespace(debug_dataset=False))
    item = ItemInfo(str(targets / "scene.mp4"), "prompt", (512, 512), (512, 512), frame_count=22)
    adapter.attach(item)

    assert item.h3_reference_modality_probabilities == (0.5, 0.5, 0.0)


def test_h3_image_target_keeps_basename_matched_ref2va_reference(tmp_path):
    images = tmp_path / "images"
    controls = tmp_path / "controls"
    images.mkdir()
    controls.mkdir()
    (images / "monster.png").write_bytes(b"target")
    (controls / "monster.png").write_bytes(b"reference")

    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "image_directory": str(images),
                "control_directory": str(controls),
                "cache_directory": str(tmp_path / "cache"),
            }
        ],
    }
    group, adapter = create_h3_dataset_group(config, Namespace(debug_dataset=False))
    item = ItemInfo(str(images / "monster.png"), "monster", (512, 512), (512, 512))

    assets = adapter.attach(item)

    assert [(asset.modality, asset.role) for asset in assets] == [
        (MediaModality.IMAGE, "target"),
        (MediaModality.IMAGE, "reference"),
    ]


@pytest.mark.parametrize(
    ("mode", "control_names"), [("first", ("portrait.png",)), ("first_last", ("portrait.png", "portrait_0.png"))]
)
def test_h3_conditioned_image_mode_builds_temporal_target_and_cache_identity(tmp_path, mode, control_names):
    images = tmp_path / "images"
    controls = tmp_path / "controls"
    images.mkdir()
    controls.mkdir()
    target = images / "portrait.png"
    target.write_bytes(b"target")
    for name in control_names:
        (controls / name).write_bytes(name.encode())
    config = {
        "general": {"resolution": [64, 64], "batch_size": 1},
        "datasets": [
            {
                "image_directory": str(images),
                "control_directory": str(controls),
                "cache_directory": str(tmp_path / "cache"),
                "h3_image_frame_count": 22,
            }
        ],
    }
    args = Namespace(debug_dataset=False, h3_image_mode=mode, h3_image_frame_count=None)
    group, adapter = create_h3_dataset_group(config, args)
    dataset = group.datasets[0]
    item = ItemInfo(str(target), "portrait", (64, 64), (64, 64), content=np.full((64, 64, 3), 17, dtype=np.uint8))

    assets = adapter.attach(item)

    assert [(asset.modality, asset.role) for asset in assets] == [(MediaModality.IMAGE, "target")]
    assert item.content.shape == (22, 64, 64, 3)
    assert item.frame_count == 22
    assert item.h3_image_mode == mode
    assert len(item.h3_condition_paths) == len(control_names)
    assert len(item.h3_cache_metadata["sample_fingerprint"]) == 64
    assert dataset.h3_image_frame_count == 22
    item.latent_cache_path = dataset.get_latent_cache_path(item)
    item.text_encoder_output_cache_path = dataset.get_text_encoder_output_cache_path(item)
    assert "_00000-022_0064x0064_" in item.latent_cache_path
    assert "_00000-022_mmh3_te.safetensors" in item.text_encoder_output_cache_path


def test_h3_conditioned_image_rejects_wrong_control_count(tmp_path):
    images = tmp_path / "images"
    controls = tmp_path / "controls"
    images.mkdir()
    controls.mkdir()
    target = images / "portrait.png"
    target.write_bytes(b"target")
    (controls / "portrait.png").write_bytes(b"control")
    config = {
        "general": {"resolution": [64, 64]},
        "datasets": [
            {
                "image_directory": str(images),
                "control_directory": str(controls),
                "cache_directory": str(tmp_path / "cache"),
            }
        ],
    }
    _, adapter = create_h3_dataset_group(config, Namespace(debug_dataset=False, h3_image_mode="first_last", h3_image_frame_count=5))
    item = ItemInfo(str(target), "portrait", (64, 64), (64, 64), content=np.zeros((64, 64, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="exactly 2 control"):
        adapter.attach(item)


def test_h3_indexed_image_sequence_accepts_caption_alias_and_ordered_controls(tmp_path):
    images = tmp_path / "images"
    controls = tmp_path / "controls"
    images.mkdir()
    controls.mkdir()
    for index, value in enumerate((10, 20, 30)):
        Image.fromarray(np.full((32, 32, 3), value, dtype=np.uint8)).save(images / f"scene_{index:02d}.png")
    (images / "scene.txt").write_text("a transformation", encoding="utf-8")
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(controls / "scene.png")
    Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8)).save(controls / "scene_00.png")

    with pytest.raises(ValueError, match="no multiple-target images found"):
        ImageDirectoryDatasource(str(images), ".txt", str(controls), None, True)

    datasource = ImageDirectoryDatasource(
        str(images),
        ".txt",
        str(controls),
        None,
        True,
        allow_indexed_caption_alias=True,
    )
    key, targets, caption, condition_images, _mask = datasource.get_image_data(0)

    assert Path(key).name == "scene_00.png"
    assert [int(np.asarray(image)[0, 0, 0]) for image in targets] == [10, 20, 30]
    assert caption == "a transformation"
    assert len(condition_images) == 2


def test_image_jsonl_keeps_legacy_relative_paths_without_conditioned_image_opt_in(tmp_path):
    jsonl = tmp_path / "dataset.jsonl"
    jsonl.write_text('{"image_path_0000":"relative/target.png","caption":"caption"}\n', encoding="utf-8")

    legacy = ImageJsonlDatasource(str(jsonl), multiple_target=True)
    conditioned = ImageJsonlDatasource(str(jsonl), multiple_target=True, normalize_indexed_paths=True)

    assert legacy.data[0]["image_path_0000"] == "relative/target.png"
    assert "image_path_0" not in legacy.data[0]
    assert conditioned.data[0]["image_path_0"] == str((tmp_path / "relative/target.png").resolve())


def test_h3_reuses_numbered_jsonl_control_paths(tmp_path):
    target = tmp_path / "target.mp4"
    image = tmp_path / "reference.png"
    audio = tmp_path / "reference.wav"
    for path in (target, image, audio):
        path.write_bytes(b"media")
    manifest = tmp_path / "videos.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_path": str(target),
                "control_path_0": str(image),
                "control_path_1": str(audio),
                "caption": "prompt",
            }
        ),
        encoding="utf-8",
    )
    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "video_jsonl_file": str(manifest),
                "cache_directory": str(tmp_path / "cache"),
                "target_frames": [22],
                "frame_extraction": "uniform",
            }
        ],
    }
    group, adapter = create_h3_dataset_group(config, Namespace(debug_dataset=False))
    item = ItemInfo(str(target), "prompt", (512, 512), (512, 512))
    assets = adapter.attach(item)
    assert [asset.path for asset in assets[1:]] == [image, audio]
    assert group.datasets[0].datasource.has_control is False


def test_h3_paired_video_audio_paths_form_one_av_reference(tmp_path):
    target = tmp_path / "target.mp4"
    reference_video = tmp_path / "motion.mp4"
    reference_audio = tmp_path / "voice.wav"
    for path in (target, reference_video, reference_audio):
        path.write_bytes(b"media")
    manifest = tmp_path / "videos.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_path": str(target),
                "control_video_path_0": str(reference_video),
                "control_audio_path_0": str(reference_audio),
                "caption": "prompt",
            }
        ),
        encoding="utf-8",
    )
    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "video_jsonl_file": str(manifest),
                "cache_directory": str(tmp_path / "cache"),
                "target_frames": [22],
                "frame_extraction": "uniform",
            }
        ],
    }

    group, adapter = create_h3_dataset_group(config, Namespace(debug_dataset=False))
    item = ItemInfo(str(target), "prompt", (512, 512), (512, 512), frame_count=22)
    assets = adapter.attach(item)

    assert len(assets) == 2
    assert assets[1].path == reference_video
    assert assets[1].modality is MediaModality.VIDEO
    assert assets[1].metadata == {"audio_path": str(reference_audio)}
    assert adapter.requires_audio is True
    assert group.datasets[0].datasource.has_control is False

    audio_asset = _reference_audio_asset(assets[1])
    assert audio_asset.path == reference_audio
    assert audio_asset.modality is MediaModality.AUDIO


def test_h3_paired_reference_requires_matching_indices(tmp_path):
    target = tmp_path / "target.mp4"
    target.write_bytes(b"media")
    manifest = tmp_path / "videos.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video_path": str(target),
                "control_video_path_0": str(tmp_path / "motion.mp4"),
                "caption": "prompt",
            }
        ),
        encoding="utf-8",
    )
    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "video_jsonl_file": str(manifest),
                "cache_directory": str(tmp_path / "cache"),
                "target_frames": [22],
            }
        ],
    }

    with pytest.raises(ValueError, match="must use the same indices"):
        create_h3_dataset_group(config, Namespace(debug_dataset=False))


@pytest.mark.parametrize(
    ("mode", "expected_modality", "expected_path", "include_audio"),
    [
        ("av", MediaModality.VIDEO, "scene.mp4", True),
        ("video", MediaModality.VIDEO, "scene.mp4", False),
        ("audio", MediaModality.AUDIO, "scene.wav", None),
    ],
)
def test_h3_toml_paired_reference_directories_select_modality(tmp_path, mode, expected_modality, expected_path, include_audio):
    targets = tmp_path / "targets"
    reference_video = tmp_path / "reference_video"
    reference_audio = tmp_path / "reference_audio"
    for directory in (targets, reference_video, reference_audio):
        directory.mkdir()
    (targets / "scene.mp4").write_bytes(b"target")
    (reference_video / "scene.mp4").write_bytes(b"video")
    (reference_audio / "scene.wav").write_bytes(b"audio")
    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "video_directory": str(targets),
                "control_video_directory": str(reference_video),
                "control_audio_directory": str(reference_audio),
                "control_modality": mode,
                "cache_directory": str(tmp_path / "cache"),
                "target_frames": [22],
            }
        ],
    }

    group, adapter = create_h3_dataset_group(config, Namespace(debug_dataset=False))
    item = ItemInfo(str(targets / "scene.mp4"), "prompt", (512, 512), (512, 512), frame_count=22)
    reference = adapter.attach(item)[1]

    assert reference.modality is expected_modality
    assert reference.path.name == expected_path
    if include_audio is not None:
        assert bool(reference.metadata.get("include_audio", True)) is include_audio
    assert not hasattr(group.datasets[0], "control_video_directory")


def test_h3_toml_control_modalities_apply_per_reference(tmp_path):
    videos = tmp_path / "videos"
    controls = tmp_path / "controls"
    videos.mkdir()
    controls.mkdir()
    target = videos / "scene.mp4"
    target.write_bytes(b"target")
    (controls / "scene.png").write_bytes(b"image")
    (controls / "scene_0.mp4").write_bytes(b"video")
    (controls / "scene_1.wav").write_bytes(b"audio")
    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "video_directory": str(videos),
                "control_directory": str(controls),
                "control_modalities": ["video", "video", "audio"],
                "cache_directory": str(tmp_path / "cache"),
                "target_frames": [22],
            }
        ],
    }

    _, adapter = create_h3_dataset_group(config, Namespace(debug_dataset=False))
    item = ItemInfo(str(target), "prompt", (512, 512), (512, 512), frame_count=22)
    references = adapter.attach(item)[1:]

    assert [reference.modality for reference in references] == [
        MediaModality.IMAGE,
        MediaModality.VIDEO,
        MediaModality.AUDIO,
    ]
    assert references[0].metadata["include_audio"] is False
    assert references[1].metadata["include_audio"] is False


def test_h3_toml_reference_modality_probabilities_are_attached(tmp_path):
    targets = tmp_path / "targets"
    controls = tmp_path / "controls"
    targets.mkdir()
    controls.mkdir()
    target = targets / "scene.mp4"
    target.write_bytes(b"target")
    (controls / "scene.png").write_bytes(b"reference")
    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "video_directory": str(targets),
                "control_directory": str(controls),
                "control_modality_probabilities": [0.5, 0.25, 0.25],
                "cache_directory": str(tmp_path / "cache"),
                "target_frames": [22],
            }
        ],
    }

    group, adapter = create_h3_dataset_group(config, Namespace(debug_dataset=False))
    item = ItemInfo(str(target), "prompt", (512, 512), (512, 512), frame_count=22)
    adapter.attach(item)

    assert item.h3_reference_modality_probabilities == (0.5, 0.25, 0.25)
    assert not hasattr(group.datasets[0], "control_modality_probabilities")


def test_h3_audio_target_resolves_control_directory_references(tmp_path):
    # Foley and reference-voice training: the conditioning video is arbitrary, not the
    # target's own synchronized track, so an audio-only dataset must reach the Ref2VA
    # reference path the video datasets use.
    audio = tmp_path / "audio"
    controls = tmp_path / "controls"
    audio.mkdir()
    controls.mkdir()
    target = audio / "tone.wav"
    target.write_bytes(b"target")
    (controls / "tone.png").write_bytes(b"image")
    (controls / "tone_0.mp4").write_bytes(b"video")
    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "audio_directory": str(audio),
                "h3_target_mode": "audio",
                "control_directory": str(controls),
                "cache_directory": str(tmp_path / "cache"),
                "target_frames": [22],
            }
        ],
    }

    adapter = h3_dataset.H3DatasetAdapter(config, Namespace(debug_dataset=False))
    item = ItemInfo(str(target), "prompt", (512, 512), (512, 512, 22), frame_count=22)
    assets = adapter.attach(item)

    assert [asset.role for asset in assets] == ["target", "reference", "reference"]
    assert [asset.modality for asset in assets] == [MediaModality.AUDIO, MediaModality.IMAGE, MediaModality.VIDEO]
    assert item.h3_target_mode == "audio"
    # The reference bundle takes part in the cache identity exactly as it does for video targets.
    assert item.h3_cache_metadata[h3_references.REFERENCE_FINGERPRINT_KEY]
    # A visual reference makes the video VAE mandatory even though the target has no video.
    assert adapter.requires_video and adapter.requires_audio


def test_h3_audio_target_without_controls_keeps_the_empty_reference_fast_path(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    target = audio / "tone.wav"
    target.write_bytes(b"target")
    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "audio_directory": str(audio),
                "h3_target_mode": "audio",
                "cache_directory": str(tmp_path / "cache"),
                "target_frames": [22],
            }
        ],
    }

    adapter = h3_dataset.H3DatasetAdapter(config, Namespace(debug_dataset=False))
    item = ItemInfo(str(target), "prompt", (512, 512), (512, 512, 22), frame_count=22)
    assets = adapter.attach(item)

    assert adapter._targets[h3_dataset._normal_path(target)].references == ()
    assert len(assets) == 1 and assets[0].role == "target"
    assert not hasattr(item, "h3_cache_metadata")
    assert not adapter.requires_video


def test_h3_audio_jsonl_control_paths_select_reference_modality(tmp_path):
    audio = tmp_path / "audio"
    controls = tmp_path / "controls"
    audio.mkdir()
    controls.mkdir()
    target = audio / "tone.wav"
    target.write_bytes(b"target")
    (controls / "anchor.png").write_bytes(b"image")
    (controls / "scene.mp4").write_bytes(b"video")
    jsonl = tmp_path / "audio.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "audio_path": str(target),
                "caption": "a clean tone",
                "control_path_0": "controls/anchor.png",
                "control_path_1": "controls/scene.mp4",
                "control_modality_1": "video",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "audio_jsonl_file": str(jsonl),
                "h3_target_mode": "audio",
                "cache_directory": str(tmp_path / "cache"),
                "target_frames": [22],
            }
        ],
    }

    adapter = h3_dataset.H3DatasetAdapter(config, Namespace(debug_dataset=False))
    item = ItemInfo(str(target), "prompt", (512, 512), (512, 512, 22), frame_count=22)
    references = adapter.attach(item)[1:]

    assert [reference.path.name for reference in references] == ["anchor.png", "scene.mp4"]
    assert references[1].metadata["include_audio"] is False


def test_h3_audio_dataset_rejects_controls_in_both_directory_and_jsonl(tmp_path):
    audio = tmp_path / "audio"
    controls = tmp_path / "controls"
    audio.mkdir()
    controls.mkdir()
    target = audio / "tone.wav"
    target.write_bytes(b"target")
    (controls / "tone.png").write_bytes(b"image")
    jsonl = tmp_path / "audio.jsonl"
    jsonl.write_text(
        json.dumps({"audio_path": str(target), "caption": "c", "control_path_0": str(controls / "tone.png")}) + "\n",
        encoding="utf-8",
    )
    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "audio_jsonl_file": str(jsonl),
                "h3_target_mode": "audio",
                "control_directory": str(controls),
                "cache_directory": str(tmp_path / "cache"),
                "target_frames": [22],
            }
        ],
    }

    with pytest.raises(ValueError, match="control_directory or the audio JSONL"):
        h3_dataset.H3DatasetAdapter(config, Namespace(debug_dataset=False))


def test_native_latent_encoder_caches_references_for_an_audio_target():
    encoder = h3_integration._NativeLatentEncoder(None, None, torch.float32)
    latent_frames = temporal_shape(124).audio_latent_frames
    encoder._encode_audio = lambda item: (torch.zeros(2, 32, latent_frames), torch.ones(latent_frames, dtype=torch.bool))
    encoder._encode_references = lambda item: {f"varlen_{H3_REFERENCE_KINDS_KEY}_int64": torch.tensor([0])}
    item = SimpleNamespace(
        item_key="tone.wav",
        h3_target_mode="audio",
        original_size=(512, 512),
        h3_media_assets=(MediaAsset(Path("tone.wav"), MediaModality.AUDIO, "target", metadata={"frame_count": 124}),),
    )

    (tensors,) = encoder.encode_latents([item])

    assert f"varlen_{H3_REFERENCE_KINDS_KEY}_int64" in tensors
    assert "mmh3_video_geometry_int64" in tensors
    assert f"latents_audio_2x32x{latent_frames}_float32" in tensors


def test_reference_modality_variants_keep_text_and_latent_geometry_aligned():
    image = H3PreparedReference(kind=H3ReferenceKind.IMAGE, image=Image.new("RGB", (8, 8)))
    video = H3PreparedReference(
        kind=H3ReferenceKind.VIDEO,
        frames=np.zeros((5, 8, 8, 3), dtype=np.uint8),
        waveform=torch.zeros(2, 100),
    )
    audio = H3PreparedReference(kind=H3ReferenceKind.AUDIO, waveform=torch.zeros(2, 100))

    video_only = reference_modality_variant((image, video, audio), "video")
    audio_only = reference_modality_variant((image, video, audio), "audio")

    assert [reference.kind for reference in video_only] == [0, 1]
    assert all(reference.waveform is None for reference in video_only)
    assert [reference.kind for reference in audio_only] == [0, 2, 2]


def test_h3_image_dataset_uses_existing_musubi_fields_and_needs_no_audio_vae(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    target = images / "target.png"
    target.write_bytes(b"image fixture")
    config = {
        "general": {"resolution": [512, 512], "batch_size": 1},
        "datasets": [
            {
                "image_directory": str(images),
                "cache_directory": str(tmp_path / "cache"),
            }
        ],
    }

    group, adapter = create_h3_dataset_group(config, Namespace(debug_dataset=False))
    item = ItemInfo(str(target), "prompt", (512, 512), (512, 512), frame_count=1)
    assets = adapter.attach(item)

    assert group.datasets[0].architecture == ARCHITECTURE_MINIMAX_H3
    assert adapter.requires_audio is False
    assert [(asset.modality, asset.role) for asset in assets] == [(MediaModality.IMAGE, "target")]
    assert assets[0].metadata == {"frame_count": 1}


def test_h3_mixed_image_video_dataset_requires_audio_only_for_video_items(tmp_path):
    images = tmp_path / "images"
    videos = tmp_path / "videos"
    images.mkdir()
    videos.mkdir()
    image = images / "image.png"
    video = videos / "video.mp4"
    image.write_bytes(b"image fixture")
    video.write_bytes(b"video fixture")
    config = {
        "general": {"resolution": [512, 512], "batch_size": 1},
        "datasets": [
            {"image_directory": str(images), "cache_directory": str(tmp_path / "image_cache")},
            {
                "video_directory": str(videos),
                "cache_directory": str(tmp_path / "video_cache"),
                "target_frames": [22],
            },
        ],
    }

    group, adapter = create_h3_dataset_group(config, Namespace(debug_dataset=False))
    image_item = ItemInfo(str(image), "image", (512, 512), (512, 512), frame_count=1)
    video_item = ItemInfo(str(video), "video", (512, 512), (512, 512, 22), frame_count=22)

    assert len(group.datasets) == 2
    assert adapter.requires_audio is True
    assert adapter.attach(image_item)[0].modality is MediaModality.IMAGE
    assert adapter.attach(video_item)[0].modality is MediaModality.VIDEO


def test_h3_allows_same_video_directory_for_different_resolution_caches(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    target = videos / "target.mp4"
    target.write_bytes(b"video fixture")
    config = {
        "general": {"batch_size": 1},
        "datasets": [
            {
                "video_directory": str(videos),
                "cache_directory": str(tmp_path / "cache_512"),
                "resolution": [512, 512],
                "target_frames": [22],
            },
            {
                "video_directory": str(videos),
                "cache_directory": str(tmp_path / "cache_768"),
                "resolution": [768, 768],
                "target_frames": [22],
            },
        ],
    }

    group, adapter = create_h3_dataset_group(config, Namespace(debug_dataset=False))
    item = ItemInfo(str(target), "video", (768, 768), (768, 768, 22), frame_count=22)

    assert len(group.datasets) == 2
    assert [dataset.resolution for dataset in group.datasets] == [(512, 512), (768, 768)]
    assert adapter.attach(item)[0].path == target


def test_h3_rejects_conflicting_modes_for_duplicate_target_path(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "target.mp4").write_bytes(b"video fixture")
    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "video_directory": str(videos),
                "cache_directory": str(tmp_path / "cache_av"),
                "target_frames": [22],
                "h3_target_mode": "av",
            },
            {
                "video_directory": str(videos),
                "cache_directory": str(tmp_path / "cache_video"),
                "target_frames": [22],
                "h3_target_mode": "video",
            },
        ],
    }

    with pytest.raises(ValueError, match="conflicting h3_target_mode"):
        create_h3_dataset_group(config, Namespace(debug_dataset=False))


def test_h3_dataset_adapter_does_not_add_shared_config_fields(tmp_path):
    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "video_jsonl_file": str(tmp_path / "videos.jsonl"),
                "cache_directory": str(tmp_path / "cache"),
                "target_frames": [22],
            }
        ],
    }
    params = (
        BlueprintGenerator(ConfigSanitizer())
        .generate(
            config,
            Namespace(debug_dataset=False),
            architecture=ARCHITECTURE_MINIMAX_H3,
        )
        .dataset_group.datasets[0]
        .params
    )
    assert not hasattr(params, "reference_image_directory")
    assert not hasattr(params, "audio_directory")
    assert not hasattr(params, "target_fps")


def test_h3_dataset_adapter_rejects_off_grid_frame_counts(tmp_path):
    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "video_directory": str(tmp_path),
                "cache_directory": str(tmp_path / "cache"),
                "target_frames": [121],
            }
        ],
    }
    with pytest.raises(ValueError, match="frame_count % 17 == 5"):
        create_h3_dataset_group(config, Namespace(debug_dataset=False))


def test_audio_length_policy_returns_valid_mask():
    waveform = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    cropped, mask, offset = fit_audio_length(waveform, 4, crop_mode=CropMode.END, pad_mode=PadMode.ERROR)
    assert cropped.tolist() == [[2.0, 3.0, 4.0, 5.0]]
    assert mask.all()
    assert offset == 2

    padded, mask, offset = fit_audio_length(waveform[:, :2], 4, crop_mode=CropMode.BEGINNING, pad_mode=PadMode.ZERO)
    assert padded.tolist() == [[0.0, 1.0, 0.0, 0.0]]
    assert mask.tolist() == [True, True, False, False]
    assert offset == 0


def test_audio_sample_mask_downsamples_conservatively_to_h3_latents():
    sample_mask = torch.ones(1600, dtype=torch.bool)
    sample_mask[-1] = False
    assert audio_valid_mask_to_latent_mask(sample_mask).tolist() == [True, False]

    with pytest.raises(ValueError, match="divisible"):
        audio_valid_mask_to_latent_mask(sample_mask[:-1])


def test_h3_media_slice_is_architecture_local(tmp_path):
    target = MediaAsset(tmp_path / "target.mp4", MediaModality.VIDEO, "target")
    sliced = slice_media_asset(target, start_seconds=2.0, duration_seconds=3.0)
    assert sliced.start_seconds == 2.0
    assert sliced.duration_seconds == 3.0


def test_ref2va_reference_geometry_matches_released_preprocessing():
    assert resolve_reference_image_size(80, 48) == (2048, 3424)
    assert resolve_reference_image_size(48, 80) == (3424, 2048)
    assert resolve_reference_video_size(1344, 768) == (768, 1344)
    assert trim_reference_frames(1) == 1
    assert trim_reference_frames(4) == 1
    assert trim_reference_frames(5) == 5
    assert trim_reference_frames(21) == 5
    assert trim_reference_frames(25) == 22
    assert trim_reference_frames(124) == 124

    frames = np.arange(30, dtype=np.uint8).reshape(-1, 1, 1, 1) * np.ones((1, 2, 2, 3), dtype=np.uint8)
    resampled = resample_reference_frames(frames, 30.0)
    assert [int(frame[0, 0, 0]) for frame in resampled] == [index for index in range(30) if index not in (2, 7, 12, 17, 22, 27)]


def test_h3_reference_video_is_trimmed_before_text_and_paired_audio_preparation(monkeypatch):
    video_asset = MediaAsset(Path("reference.mp4"), MediaModality.VIDEO, "reference")
    item = SimpleNamespace(h3_media_assets=(video_asset,), frame_count=30)
    audio_frame_counts = []

    monkeypatch.setattr(
        h3_references,
        "_prepare_video",
        lambda _asset, _target_frames, _short_edge, _max_pixels: np.zeros((30, 8, 8, 3), dtype=np.uint8),
    )

    def prepare_audio(_asset, target_frames):
        audio_frame_counts.append(target_frames)
        return torch.zeros(2, target_frames * 10)

    monkeypatch.setattr(h3_references, "_prepare_audio", prepare_audio)

    (reference,) = h3_references.prepare_references(item)

    assert reference.frames.shape[0] == 22
    assert audio_frame_counts == [22]


@pytest.mark.parametrize("source_fps", [12.0, 24.0, 25.0, 30.0, 60.0])
def test_reference_decode_limit_is_the_minimum_needed_for_target_duration(source_fps):
    target_frames = 124
    limit = _source_frame_limit(target_frames, source_fps)
    scale = VIDEO_FPS / source_fps

    assert math.floor(limit * scale + 0.5) >= target_frames
    assert limit == 1 or math.floor((limit - 1) * scale + 0.5) < target_frames


def test_reference_image_short_edge_default_matches_released_preprocessing():
    assert REFERENCE_IMAGE_SHORT_EDGE == 2048
    for width, height in ((512, 512), (80, 48), (48, 80), (1344, 768)):
        assert resolve_reference_image_size(width, height) == resolve_reference_image_size(
            width, height, REFERENCE_IMAGE_SHORT_EDGE
        )
    assert resolve_reference_image_size(512, 512) == (2048, 2048)
    assert resolve_reference_image_size(80, 48) == (2048, 3424)
    assert reference_key_suffix(REFERENCE_IMAGE_SHORT_EDGE) == ""


def test_reference_image_short_edge_scales_and_names_the_cache():
    assert resolve_reference_image_size(512, 512, 768) == (768, 768)
    assert resolve_reference_image_size(80, 48, 768) == (768, 1280)
    assert resolve_reference_image_size(48, 80, 768) == (1280, 768)
    assert resolve_reference_image_size(96, 96, CANVAS_MULTIPLE) == (CANVAS_MULTIPLE, CANVAS_MULTIPLE)
    assert reference_key_suffix(768) == "_se768"


def test_reference_video_sizing_is_configurable_and_names_the_cache():
    assert resolve_reference_video_size(512, 512) == (768, 768)
    assert resolve_reference_video_size(512, 512, 384, 384 * 672) == (384, 384)
    assert resolve_reference_video_size(1024, 512, 384, 384 * 672) == (352, 704)
    assert (
        reference_key_suffix(
            REFERENCE_IMAGE_SHORT_EDGE,
            video_short_edge=384,
            video_max_pixels=384 * 672,
        )
        == "_vse384_vmp258048"
    )


def test_reference_video_sizing_keeps_released_defaults_and_never_overshoots_the_cap():
    # Released defaults must resolve exactly as before the cap was moved after
    # rounding: these are the sizes every existing cache was written with.
    assert resolve_reference_video_size(1344, 768) == (768, 1344)
    assert resolve_reference_video_size(1920, 1080) == (768, 1344)
    assert resolve_reference_video_size(1080, 1920) == (1344, 768)
    assert resolve_reference_video_size(512, 512) == (768, 768)
    assert resolve_reference_video_size(1024, 512, 384, 384 * 672) == (352, 704)

    # A 4:1 source with a small cap used to round back up to 64x32 = 2048 px,
    # twice the requested 1024-pixel budget.
    assert resolve_reference_video_size(4096, 1024, 768, 1024) == (CANVAS_MULTIPLE, CANVAS_MULTIPLE)

    for width, height in [(4096, 1024), (1024, 4096), (1920, 1080), (1080, 1920), (640, 640), (1200, 500)]:
        for short_edge in (CANVAS_MULTIPLE, 96, 384, 768):
            for max_pixels in (CANVAS_MULTIPLE**2, 1024, 65536, 384 * 672, REFERENCE_VIDEO_MAX_PIXELS):
                resolved_height, resolved_width = resolve_reference_video_size(width, height, short_edge, max_pixels)
                assert resolved_height % CANVAS_MULTIPLE == 0 and resolved_width % CANVAS_MULTIPLE == 0
                assert resolved_height >= CANVAS_MULTIPLE and resolved_width >= CANVAS_MULTIPLE
                # The 32x32 floor is the only licensed way to exceed the cap.
                assert resolved_height * resolved_width <= max_pixels or (
                    resolved_height == CANVAS_MULTIPLE and resolved_width == CANVAS_MULTIPLE
                )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        # Both used to pass the trainer's own looser bounds (>= 16 / >= 256) and
        # only fail later inside reference_key_suffix().
        ("reference_video_short_edge", CANVAS_MULTIPLE - 1),
        ("reference_video_max_pixels", CANVAS_MULTIPLE**2 - 1),
    ],
)
def test_h3_reference_video_sizing_argparse_bounds_match_the_canonical_validator(key, value):
    from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser

    args = create_parser().parse_args(["--sdpa"])
    setattr(args, key, value)

    with pytest.raises(ValueError, match="H3 reference video"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)

    args = create_parser().parse_args(["--sdpa"])
    setattr(args, key, CANVAS_MULTIPLE if key == "reference_video_short_edge" else CANVAS_MULTIPLE**2)
    MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_reference_image_target_area_preserves_aspect_and_names_cache_variant():
    assert resolve_reference_image_area_size(512, 512, 512 * 512) == (512, 512)
    assert resolve_reference_image_area_size(1024, 512, 512 * 512) == (352, 736)
    assert reference_key_suffix(REFERENCE_IMAGE_SHORT_EDGE, "target_area", 0) == "_ta"
    assert reference_key_suffix(REFERENCE_IMAGE_SHORT_EDGE, "target_area", 262144) == "_ta262144"


def _reference_latent_cache(path: Path, suffix: str, fingerprint="fingerprint") -> str:
    metadata = {h3_references.REFERENCE_FINGERPRINT_KEY: fingerprint} if fingerprint is not None else None
    save_file(
        {f"varlen_{H3_REFERENCE_KINDS_KEY}{suffix}_int64": torch.tensor([0, 2], dtype=torch.long)},
        str(path),
        metadata=metadata,
    )
    return str(path)


def _latent_cache_predicate(monkeypatch, argv):
    captured = {}

    def encode_datasets(datasets, encode, args, existing_cache_valid=None):
        captured["predicate"] = existing_cache_valid

    adapter = SimpleNamespace(requires_video=False, requires_audio=False)
    monkeypatch.setattr(h3_cache_latents.config_utils, "load_user_config", lambda path: {})
    monkeypatch.setattr(h3_cache_latents, "create_h3_dataset_group", lambda config, args: (SimpleNamespace(datasets=[]), adapter))
    monkeypatch.setattr(h3_cache_latents, "create_latent_encoder", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(h3_cache_latents, "attach_h3_media", lambda batch, dataset_adapter: None)
    monkeypatch.setattr(h3_cache_latents.cache_latents, "encode_datasets", encode_datasets)
    h3_cache_latents.main(["--dataset_config", "dataset.toml", *argv])
    return captured["predicate"]


def test_skip_existing_rejects_reference_caches_written_with_other_sizing(monkeypatch, tmp_path):
    monkeypatch.setattr(h3_cache_latents, "reference_assets", lambda item: item.has_references)
    item = SimpleNamespace(has_references=True, h3_cache_metadata={h3_references.REFERENCE_FINGERPRINT_KEY: "fingerprint"})
    default_cache = _reference_latent_cache(tmp_path / "default.safetensors", "")
    short_edge_cache = _reference_latent_cache(tmp_path / "short_edge.safetensors", "_se512")
    video_cache = _reference_latent_cache(tmp_path / "video.safetensors", "_vse256")
    target_area_cache = _reference_latent_cache(tmp_path / "target_area.safetensors", "_ta")
    capped_area_cache = _reference_latent_cache(tmp_path / "capped_area.safetensors", "_ta262144")

    default_valid = _latent_cache_predicate(monkeypatch, [])
    assert default_valid(item, default_cache) is True
    assert default_valid(item, short_edge_cache) is False
    assert default_valid(item, video_cache) is False
    assert default_valid(item, target_area_cache) is False
    assert default_valid(item, capped_area_cache) is False

    target_area_valid = _latent_cache_predicate(monkeypatch, ["--reference_image_size_mode", "target_area"])
    assert target_area_valid(item, target_area_cache) is True
    assert target_area_valid(item, capped_area_cache) is False
    assert target_area_valid(item, default_cache) is False

    capped_area_valid = _latent_cache_predicate(
        monkeypatch, ["--reference_image_size_mode", "target_area", "--reference_image_max_pixels", "262144"]
    )
    assert capped_area_valid(item, capped_area_cache) is True
    assert capped_area_valid(item, target_area_cache) is False

    assert default_valid(SimpleNamespace(has_references=False), short_edge_cache) is True


def _text_cache_predicate(monkeypatch, argv):
    captured = {}

    def process_text_encoder_batches(*args, existing_cache_valid=None, **kwargs):
        captured["predicate"] = existing_cache_valid

    adapter = SimpleNamespace(requires_video=False, requires_audio=False)
    monkeypatch.setattr(h3_cache_text.config_utils, "load_user_config", lambda path: {})
    monkeypatch.setattr(h3_cache_text, "create_h3_dataset_group", lambda config, args: (SimpleNamespace(datasets=[]), adapter))
    monkeypatch.setattr(
        h3_cache_text,
        "create_conditioning_encoder",
        lambda **kwargs: SimpleNamespace(conditioning_requires_content=False, close=lambda: None),
    )
    monkeypatch.setattr(h3_cache_text, "attach_h3_media", lambda batch, dataset_adapter: None)
    monkeypatch.setattr(h3_cache_text.cache_text_encoder_outputs, "prepare_cache_files_and_paths", lambda datasets: ({}, {}))
    monkeypatch.setattr(h3_cache_text.cache_text_encoder_outputs, "process_text_encoder_batches", process_text_encoder_batches)
    monkeypatch.setattr(h3_cache_text.cache_text_encoder_outputs, "post_process_cache_files", lambda *args, **kwargs: None)
    h3_cache_text.main(["--dataset_config", "dataset.toml", "--text_encoder", "encoder", *argv])
    return captured["predicate"]


def _reference_text_cache(path: Path, fingerprint="fingerprint") -> str:
    metadata = {h3_references.REFERENCE_FINGERPRINT_KEY: fingerprint} if fingerprint is not None else None
    save_file(
        {
            "mmh3_hidden_states_bfloat16": torch.zeros(1),
            H3_CONDITIONING_TASK_KEY: torch.tensor(H3_CONDITIONING_TASK_IDS["ref2va"]),
            H3_REFERENCE_IMAGE_SHORT_EDGE_KEY: torch.tensor(REFERENCE_IMAGE_SHORT_EDGE),
            H3_REFERENCE_IMAGE_SIZE_MODE_KEY: torch.tensor(0),
            H3_REFERENCE_IMAGE_MAX_PIXELS_KEY: torch.tensor(0),
            H3_REFERENCE_VIDEO_SHORT_EDGE_KEY: torch.tensor(REFERENCE_VIDEO_SHORT_EDGE),
            H3_REFERENCE_VIDEO_MAX_PIXELS_KEY: torch.tensor(REFERENCE_VIDEO_MAX_PIXELS),
        },
        str(path),
        metadata=metadata,
    )
    return str(path)


def _identity_text_cache(path: Path, task: str = "fl2va", tensors: dict | None = None) -> str:
    save_file(
        {
            "varlen_mmh3_hidden_states_bfloat16": torch.zeros(1, dtype=torch.bfloat16),
            H3_CONDITIONING_TASK_KEY: torch.tensor(H3_CONDITIONING_TASK_IDS[task]),
            **(tensors or {}),
        },
        str(path),
    )
    return str(path)


def _fingerprint_item(fingerprint: str, *, has_references: bool = True) -> SimpleNamespace:
    return SimpleNamespace(has_references=has_references, h3_cache_metadata={h3_references.REFERENCE_FINGERPRINT_KEY: fingerprint})


def _reference_item(paths, kinds=None, audio_paths=None) -> SimpleNamespace:
    kinds = kinds or [MediaModality.IMAGE] * len(paths)
    audio_paths = audio_paths or [None] * len(paths)
    return SimpleNamespace(
        h3_media_assets=tuple(
            MediaAsset(path, kind, "reference", metadata={"audio_path": str(audio)} if audio else {})
            for path, kind, audio in zip(paths, kinds, audio_paths)
        )
    )


def test_reference_fingerprint_tracks_file_identity_kind_and_order(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    assets = _reference_item([first, second]).h3_media_assets

    baseline = h3_references.reference_fingerprint(assets)
    assert len(baseline) == 64
    assert h3_references.reference_fingerprint(_reference_item([first, second]).h3_media_assets) == baseline
    assert h3_references.reference_fingerprint(_reference_item([second, first]).h3_media_assets) != baseline
    assert (
        h3_references.reference_fingerprint(
            _reference_item([first, second], [MediaModality.IMAGE, MediaModality.VIDEO]).h3_media_assets
        )
        != baseline
    )

    first.write_bytes(b"replaced with other content")
    replaced = h3_references.reference_fingerprint(assets)
    assert replaced != baseline

    os.utime(first, (0, 0))
    assert h3_references.reference_fingerprint(assets) != replaced

    assert h3_references.reference_fingerprint(()) is None
    target = MediaAsset(first, MediaModality.IMAGE, "target")
    assert h3_references.reference_fingerprint((target,)) is None
    assert h3_references.reference_fingerprint((target, *assets)) == h3_references.reference_fingerprint(assets)


def test_reference_fingerprint_tracks_paired_reference_soundtracks(tmp_path):
    video = tmp_path / "clip.mp4"
    audio = tmp_path / "clip.wav"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    assets = _reference_item([video], [MediaModality.VIDEO], [audio]).h3_media_assets

    baseline = h3_references.reference_fingerprint(assets)
    assert baseline != h3_references.reference_fingerprint(_reference_item([video], [MediaModality.VIDEO]).h3_media_assets)
    audio.write_bytes(b"other soundtrack")
    assert h3_references.reference_fingerprint(assets) != baseline


def test_h3_dataset_records_reference_fingerprint_in_cache_metadata(tmp_path):
    images = tmp_path / "images"
    controls = tmp_path / "controls"
    images.mkdir()
    controls.mkdir()
    target = images / "monster.png"
    reference = controls / "monster.png"
    Image.new("RGB", (64, 64)).save(target)
    Image.new("RGB", (64, 64), color=(1, 2, 3)).save(reference)
    config = {
        "general": {"resolution": [64, 64], "batch_size": 1},
        "datasets": [
            {
                "image_directory": str(images),
                "control_directory": str(controls),
                "cache_directory": str(tmp_path / "cache"),
            }
        ],
    }
    _, adapter = create_h3_dataset_group(config, Namespace(debug_dataset=False))
    item = ItemInfo(str(target), "monster", (64, 64), (64, 64))

    assets = adapter.attach(item)

    fingerprint = item.h3_cache_metadata[h3_references.REFERENCE_FINGERPRINT_KEY]
    assert fingerprint == h3_references.reference_fingerprint(assets)

    item.latent_cache_path = str(tmp_path / "cache" / "monster_mmh3.safetensors")
    save_latent_cache_minimax_h3(
        item,
        {
            "latents_1x8x8_float32": torch.zeros(24, 1, 8, 8),
            f"varlen_{H3_REFERENCE_KINDS_KEY}_int64": torch.tensor([0], dtype=torch.long),
        },
    )
    with safe_open(item.latent_cache_path, framework="pt", device="cpu") as handle:
        assert handle.metadata()[h3_references.REFERENCE_FINGERPRINT_KEY] == fingerprint


def test_skip_existing_rejects_reference_caches_encoded_from_other_files(monkeypatch, tmp_path):
    monkeypatch.setattr(h3_cache_latents, "reference_assets", lambda item: item.has_references)
    monkeypatch.setattr(h3_cache_text, "reference_assets", lambda item: item.has_references)
    latent_cache = _reference_latent_cache(tmp_path / "latent.safetensors", "")
    legacy_latent_cache = _reference_latent_cache(tmp_path / "legacy_latent.safetensors", "", fingerprint=None)
    text_cache = _reference_text_cache(tmp_path / "text.safetensors")
    legacy_text_cache = _reference_text_cache(tmp_path / "legacy_text.safetensors", fingerprint=None)

    latent_valid = _latent_cache_predicate(monkeypatch, [])
    text_valid = _text_cache_predicate(monkeypatch, ["--task", "ref2va"])

    assert latent_valid(_fingerprint_item("fingerprint"), latent_cache) is True
    assert latent_valid(_fingerprint_item("other"), latent_cache) is False
    assert latent_valid(_fingerprint_item("fingerprint"), legacy_latent_cache) is False
    assert latent_valid(SimpleNamespace(has_references=False), legacy_latent_cache) is True

    assert text_valid(_fingerprint_item("fingerprint"), text_cache) is True
    assert text_valid(_fingerprint_item("other"), text_cache) is False
    assert text_valid(_fingerprint_item("fingerprint"), legacy_text_cache) is False
    assert text_valid(SimpleNamespace(has_references=False), legacy_text_cache) is True


def test_skip_existing_rejects_text_caches_encoded_for_another_task(monkeypatch, tmp_path):
    cache = _identity_text_cache(tmp_path / "text.safetensors")
    legacy = str(tmp_path / "legacy.safetensors")
    save_file({"varlen_mmh3_hidden_states_bfloat16": torch.zeros(1, dtype=torch.bfloat16)}, legacy)
    item = SimpleNamespace()

    assert _text_cache_predicate(monkeypatch, ["--task", "fl2va"])(item, cache) is True
    assert _text_cache_predicate(monkeypatch, ["--task", "i2va"])(item, cache) is False
    assert _text_cache_predicate(monkeypatch, ["--task", "t2va"])(item, cache) is False
    assert _text_cache_predicate(monkeypatch, ["--task", "t2va"])(item, legacy) is False


def test_skip_existing_rejects_text_caches_with_another_qwen_visual_cap(monkeypatch, tmp_path):
    capped = _identity_text_cache(tmp_path / "capped.safetensors", tensors={H3_TEXT_VISUAL_MAX_PIXELS_KEY: torch.tensor(65_536)})
    uncapped = _identity_text_cache(tmp_path / "uncapped.safetensors")
    item = SimpleNamespace()

    assert _text_cache_predicate(monkeypatch, ["--task", "fl2va", "--h3_text_visual_max_pixels", "65536"])(item, capped) is True
    assert _text_cache_predicate(monkeypatch, ["--task", "fl2va", "--h3_text_visual_max_pixels", "131072"])(item, capped) is False
    assert _text_cache_predicate(monkeypatch, ["--task", "fl2va"])(item, capped) is False
    assert _text_cache_predicate(monkeypatch, ["--task", "fl2va"])(item, uncapped) is True
    assert _text_cache_predicate(monkeypatch, ["--task", "fl2va", "--h3_text_visual_max_pixels", "65536"])(item, uncapped) is False


def test_skip_existing_rejects_reference_caches_sized_with_another_image_strategy(monkeypatch, tmp_path):
    cache = _reference_text_cache(tmp_path / "text.safetensors")
    legacy = _identity_text_cache(
        tmp_path / "legacy.safetensors",
        task="ref2va",
        tensors={
            H3_REFERENCE_VIDEO_SHORT_EDGE_KEY: torch.tensor(REFERENCE_VIDEO_SHORT_EDGE),
            H3_REFERENCE_VIDEO_MAX_PIXELS_KEY: torch.tensor(REFERENCE_VIDEO_MAX_PIXELS),
        },
    )
    item = _fingerprint_item("fingerprint")

    assert _text_cache_predicate(monkeypatch, ["--task", "ref2va"])(item, cache) is True
    assert _text_cache_predicate(monkeypatch, ["--task", "ref2va", "--reference_image_short_edge", "1024"])(item, cache) is False
    assert (
        _text_cache_predicate(monkeypatch, ["--task", "ref2va", "--reference_image_size_mode", "target_area"])(item, cache) is False
    )
    assert _text_cache_predicate(monkeypatch, ["--task", "ref2va", "--reference_image_max_pixels", "262144"])(item, cache) is False
    assert _text_cache_predicate(monkeypatch, ["--task", "ref2va"])(item, legacy) is False


def test_skip_existing_rejects_text_caches_without_the_requested_empty_pair(monkeypatch, tmp_path):
    prompt_only = _identity_text_cache(tmp_path / "prompt.safetensors")
    with_empty = _identity_text_cache(
        tmp_path / "empty.safetensors",
        tensors={
            "varlen_mmh3_empty_hidden_states_bfloat16": torch.zeros(1, dtype=torch.bfloat16),
            "varlen_mmh3_empty_token_tags_int64": torch.zeros(1, dtype=torch.long),
        },
    )
    item = SimpleNamespace()

    assert _text_cache_predicate(monkeypatch, ["--task", "fl2va"])(item, prompt_only) is True
    assert _text_cache_predicate(monkeypatch, ["--task", "fl2va", "--cache_guidance_empty"])(item, prompt_only) is False
    assert _text_cache_predicate(monkeypatch, ["--task", "fl2va", "--cache_guidance_empty"])(item, with_empty) is True


def test_prepare_reference_target_area_uses_bucket_area_and_optional_cap(tmp_path):
    path = tmp_path / "reference.png"
    Image.new("RGB", (1024, 512)).save(path)
    item = SimpleNamespace(
        h3_media_assets=(MediaAsset(path, MediaModality.IMAGE, "reference"),),
        frame_count=5,
        original_size=(1280, 720),
        bucket_size=(768, 512),
    )
    (bucket_sized,) = h3_references.prepare_references(item, image_size_mode="target_area")
    (capped,) = h3_references.prepare_references(item, image_size_mode="target_area", image_max_pixels=262144)
    assert bucket_sized.image.size == (896, 448)
    assert capped.image.size == (736, 352)


@pytest.mark.parametrize("short_edge", (0, -1, -768, CANVAS_MULTIPLE - 1))
def test_reference_image_short_edge_rejects_unusable_values(short_edge):
    with pytest.raises(ValueError, match="short edge"):
        validate_reference_image_short_edge(short_edge)
    with pytest.raises(ValueError, match="short edge"):
        resolve_reference_image_size(512, 512, short_edge)
    with pytest.raises(ValueError, match="short edge"):
        reference_key_suffix(short_edge)


def test_reference_image_aspect_guard_precedes_the_short_edge():
    with pytest.raises(ValueError, match="1:4 to 4:1"):
        resolve_reference_image_size(400, 80)
    with pytest.raises(ValueError, match="1:4 to 4:1"):
        resolve_reference_image_size(80, 400, 768)
    with pytest.raises(ValueError, match="1:4 to 4:1"):
        resolve_reference_image_size(0, 512, 768)


def test_reference_image_short_edge_flag_is_registered_on_every_entrypoint():
    for parser in (create_cache_latents_parser(), create_cache_text_parser(), create_parser()):
        action = next(a for a in parser._actions if a.dest == "reference_image_short_edge")
        assert action.type is int
        assert action.default == REFERENCE_IMAGE_SHORT_EDGE


def test_reference_video_sizing_flags_are_registered_on_every_entrypoint():
    for parser in (create_cache_latents_parser(), create_cache_text_parser(), create_parser()):
        short_edge = next(a for a in parser._actions if a.dest == "reference_video_short_edge")
        max_pixels = next(a for a in parser._actions if a.dest == "reference_video_max_pixels")
        assert short_edge.default == REFERENCE_VIDEO_SHORT_EDGE
        assert max_pixels.default == REFERENCE_VIDEO_MAX_PIXELS


def test_audio_file_decode_resample_and_mask(tmp_path):
    path = tmp_path / "tone.wav"
    samples = (np.sin(np.arange(8000) * 2 * np.pi * 220 / 8000) * 16000).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(samples.tobytes())

    clip = load_audio_asset(
        MediaAsset(path, MediaModality.AUDIO, "target"),
        AudioProcessingSpec(16000, 2, clip_duration_seconds=0.5, crop_mode=CropMode.BEGINNING),
    )

    assert clip is not None
    assert clip.waveform.shape == (2, 8000)
    assert clip.waveform.dtype == torch.float32
    assert clip.valid_mask.all()
    assert clip.source_start_seconds == 0

    target = MediaAsset(
        path,
        MediaModality.VIDEO,
        "target",
        duration_seconds=22 / 24,
        metadata={"frame_count": 22, "fps": 24.0},
    )
    target_clip = load_audio_asset(target, target_audio_processing_spec(target))
    assert target_clip is not None
    assert target_clip.waveform.shape == (2, temporal_shape(22).audio_samples)
    assert int(target_clip.valid_mask.sum()) == round(target.duration_seconds * AUDIO_SAMPLE_RATE)
    assert not target_clip.valid_mask[-1]


def test_target_video_without_audio_stream_becomes_fully_masked_silence(monkeypatch, tmp_path):
    import av

    path = tmp_path / "silent.mp4"
    path.write_bytes(b"container placeholder")

    class NoAudioContainer:
        streams = SimpleNamespace(audio=())

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(av, "open", lambda source: NoAudioContainer())
    target = MediaAsset(path, MediaModality.VIDEO, "target", metadata={"frame_count": 22, "fps": 24.0})
    spec = target_audio_processing_spec(target)
    clip = load_audio_asset(target, spec)

    assert spec.missing is MissingMediaPolicy.ZERO
    assert clip is not None
    assert clip.waveform.shape == (2, temporal_shape(22).audio_samples)
    assert not clip.waveform.any()
    assert not clip.valid_mask.any()
    assert not audio_valid_mask_to_latent_mask(clip.valid_mask).any()

    reference = MediaAsset(path, MediaModality.VIDEO, "reference")
    reference_spec = AudioProcessingSpec(AUDIO_SAMPLE_RATE, 2, missing=MissingMediaPolicy.DROP)
    assert load_audio_asset(reference, reference_spec) is None


def test_missing_policy_does_not_hide_corrupt_audio(tmp_path):
    path = tmp_path / "corrupt.mp4"
    path.write_bytes(b"not a media container")
    target = MediaAsset(path, MediaModality.VIDEO, "target", metadata={"frame_count": 22, "fps": 24.0})

    with pytest.raises(AudioDecodeError, match="cannot decode audio"):
        load_audio_asset(target, target_audio_processing_spec(target))

    drop_spec = AudioProcessingSpec(AUDIO_SAMPLE_RATE, 2, missing=MissingMediaPolicy.DROP)
    with pytest.raises(AudioDecodeError, match="cannot decode audio"):
        load_audio_asset(target, drop_spec)


def test_native_cache_io_records_audio_tensor_and_architecture(tmp_path):
    cache = tmp_path / "sample_mmh3.safetensors"
    item = ItemInfo("sample", "caption", (1280, 720), (1280, 720), frame_count=22, latent_cache_path=str(cache))
    save_latent_cache_minimax_h3(
        item,
        {
            "latents_2x2x3_float32": torch.ones(24, 2, 2, 3),
            "latents_audio_2x32x4_float32": torch.zeros(2, 32, 4),
            "audio_loss_mask": torch.zeros(4, dtype=torch.bool),
        },
    )

    with safe_open(cache, framework="pt") as handle:
        assert set(handle.keys()) == {"latents_audio_2x32x4_float32", "audio_loss_mask", "latents_2x2x3_float32"}
        assert not handle.get_tensor("audio_loss_mask").any()
        assert handle.metadata()["architecture"] == "minimax_h3"
        assert handle.metadata()["frame_count"] == "22"


def test_h3_conditioned_image_cache_pair_requires_matching_sample_fingerprint(tmp_path):
    latent = tmp_path / "latent.safetensors"
    text = tmp_path / "text.safetensors"
    save_file({"x": torch.zeros(1)}, latent, metadata={"sample_fingerprint": "same"})
    save_file({"x": torch.zeros(1)}, text, metadata={"sample_fingerprint": "same"})
    _validate_h3_cache_pair(str(latent), str(text))

    save_file({"x": torch.zeros(1)}, text, metadata={"sample_fingerprint": "different"})
    with pytest.raises(ValueError, match="do not describe the same sample"):
        _validate_h3_cache_pair(str(latent), str(text))


def test_h3_cache_pair_rejects_stale_reference_video_preprocessing(tmp_path):
    latent = tmp_path / "latent.safetensors"
    text = tmp_path / "text.safetensors"
    kinds_key = "varlen_mmh3_reference_kinds_int64"
    save_file({kinds_key: torch.tensor([1], dtype=torch.long)}, latent)
    save_file({"x": torch.zeros(1)}, text)

    with pytest.raises(ValueError, match="reference-video preprocessing changed"):
        _validate_h3_cache_pair(str(latent), str(text))

    contract = torch.tensor(H3_REFERENCE_TEMPORAL_CONTRACT_VERSION, dtype=torch.long)
    save_file({kinds_key: torch.tensor([1], dtype=torch.long), H3_REFERENCE_TEMPORAL_CONTRACT_KEY: contract}, latent)
    save_file({H3_REFERENCE_TEMPORAL_CONTRACT_KEY: contract}, text)
    _validate_h3_cache_pair(str(latent), str(text))


def test_h3_cache_pair_accepts_legacy_non_reference_cache(tmp_path):
    latent = tmp_path / "latent.safetensors"
    text = tmp_path / "text.safetensors"
    save_file({"x": torch.zeros(1)}, latent)
    save_file({"x": torch.zeros(1)}, text)

    _validate_h3_cache_pair(str(latent), str(text))


def test_h3_reference_video_latents_record_temporal_contract(monkeypatch):
    reference = H3PreparedReference(
        kind=H3ReferenceKind.VIDEO,
        frames=np.zeros((5, 4, 4, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(h3_integration, "prepare_references", lambda *_args, **_kwargs: (reference,))
    encoder = h3_integration._NativeLatentEncoder(None, None, torch.float32)
    monkeypatch.setattr(encoder, "_encode_reference_video", lambda *_args, **_kwargs: torch.zeros(24, 1, 2, 2))

    cached = encoder._encode_references(SimpleNamespace())

    assert int(cached[H3_REFERENCE_TEMPORAL_CONTRACT_KEY]) == H3_REFERENCE_TEMPORAL_CONTRACT_VERSION


def test_native_cache_io_accepts_one_frame_image_without_audio(tmp_path):
    cache = tmp_path / "image_mmh3.safetensors"
    item = ItemInfo("image.png", "caption", (512, 512), (512, 512), frame_count=1, latent_cache_path=str(cache))
    item.h3_media_assets = (MediaAsset(Path("image.png"), MediaModality.IMAGE, "target"),)

    save_latent_cache_minimax_h3(item, {"latents_1x2x3_float32": torch.ones(24, 1, 2, 3)})

    with safe_open(cache, framework="pt") as handle:
        assert set(handle.keys()) == {"latents_1x2x3_float32"}
        assert handle.metadata()["frame_count"] == "1"


def test_native_cache_io_accepts_video_only_without_silent_audio(tmp_path):
    cache = tmp_path / "video_only_mmh3.safetensors"
    item = ItemInfo("clip.mp4", "caption", (512, 512), (512, 512), frame_count=22, latent_cache_path=str(cache))
    item.h3_target_mode = "video"
    save_latent_cache_minimax_h3(item, {"latents_7x2x3_float32": torch.ones(24, 7, 2, 3)})

    with safe_open(cache, framework="pt") as handle:
        assert set(handle.keys()) == {"latents_7x2x3_float32"}


def test_native_cache_io_accepts_audio_only_without_dummy_video(tmp_path):
    cache = tmp_path / "audio_only_mmh3.safetensors"
    item = ItemInfo("clip.wav", "caption", (512, 512), (512, 512), frame_count=22, latent_cache_path=str(cache))
    item.h3_target_mode = "audio"
    save_latent_cache_minimax_h3(
        item,
        {
            "latents_audio_2x32x37_float32": torch.ones(2, 32, 37),
            "audio_loss_mask": torch.ones(37, dtype=torch.bool),
            "mmh3_video_geometry_int64": torch.tensor([32, 32], dtype=torch.long),
        },
    )

    with safe_open(cache, framework="pt") as handle:
        assert "latents" not in {key.rsplit("_", 1)[0] for key in handle.keys()}
        assert "latents_audio_2x32x37_float32" in handle.keys()


def test_h3_audio_dataset_builds_cache_paths_and_duration_contract(tmp_path):
    audio = tmp_path / "tone.wav"
    audio.write_bytes(b"fixture")
    audio.with_suffix(".txt").write_text("a clean tone", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    dataset = H3AudioDataset(
        {
            "audio_directory": str(tmp_path),
            "cache_directory": str(cache),
            "h3_target_mode": "audio",
            "target_frames": [124],
            "resolution": [832, 480],
        },
        {},
    )

    _, batch = next(iter(dataset.retrieve_latent_cache_batches(1)))
    item = batch[0]
    assert item.caption == "a clean tone"
    assert item.frame_count == 124
    assert item.original_size == (832, 480)
    assert re.fullmatch(r"tone_audio[0-9a-f]{8}_00000-124_0832x0480_mmh3\.safetensors", Path(item.latent_cache_path).name)
    assert re.fullmatch(r"tone_audio[0-9a-f]{8}_00000-124_mmh3_te\.safetensors", Path(item.text_encoder_output_cache_path).name)


def test_h3_audio_cache_names_disambiguate_sources_sharing_a_stem(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()

    def dataset_for(directory: str) -> H3AudioDataset:
        root = tmp_path / directory
        root.mkdir(exist_ok=True)
        (root / "a.wav").write_bytes(b"fixture")
        return H3AudioDataset(
            {
                "audio_directory": str(root),
                "cache_directory": str(cache),
                "h3_target_mode": "audio",
                "target_frames": [124],
                "resolution": [832, 480],
            },
            {},
        )

    songs, others = dataset_for("songs"), dataset_for("others")
    (_, [song]), (_, [other]) = (
        next(iter(songs.retrieve_latent_cache_batches(1))),
        next(iter(others.retrieve_latent_cache_batches(1))),
    )
    # A same-stem video target caches as `a_0064x0064_mmh3.safetensors`; the `_audio<hash>` marker
    # keeps audio caches out of that name space and apart from each other.
    assert Path(song.latent_cache_path).name != Path(other.latent_cache_path).name
    assert Path(song.text_encoder_output_cache_path).name != Path(other.text_encoder_output_cache_path).name
    assert "_audio" in Path(song.latent_cache_path).name
    assert not Path(song.latent_cache_path).name.startswith("a_0")
    # Deterministic for the same source path.
    assert dataset_for("songs").get_latent_cache_path(song) == song.latent_cache_path


def _write_tone(path: Path, samples: int, sample_rate: int = AUDIO_SAMPLE_RATE) -> Path:
    tone = (np.sin(np.arange(samples) * 2 * np.pi * 220 / sample_rate) * 16000 + 1).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(tone.tobytes())
    return path


def test_reference_audio_follows_the_canonical_sample_grid(tmp_path):
    asset = MediaAsset(_write_tone(tmp_path / "long.wav", AUDIO_SAMPLE_RATE), MediaModality.AUDIO, "reference")

    waveform = h3_references._prepare_audio(asset, 5)

    shape = temporal_shape(5)
    assert waveform.shape == (2, shape.audio_samples)
    assert math.ceil(waveform.shape[1] / AUDIO_HOP_LENGTH) == shape.audio_latent_frames
    legacy_samples = round(5 / VIDEO_FPS * AUDIO_SAMPLE_RATE)
    assert math.ceil(legacy_samples / AUDIO_HOP_LENGTH) == shape.audio_latent_frames + 1


def test_short_reference_audio_is_zero_padded_to_the_target_span(tmp_path):
    asset = MediaAsset(_write_tone(tmp_path / "short.wav", 1600), MediaModality.AUDIO, "reference")

    waveform = h3_references._prepare_audio(asset, 5)

    assert waveform.shape == (2, temporal_shape(5).audio_samples)
    assert waveform[:, :1600].abs().sum() > 0
    assert not waveform[:, 1600:].any()


def test_native_latent_encoder_caches_fl2va_keyframes_for_video_only_target():
    class VideoEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor_values = []
            self.marker = torch.nn.Parameter(torch.zeros((), dtype=torch.float32), requires_grad=False)

        def encode(self, pixels):
            del pixels
            return torch.zeros(1, 24, 2, 2, 2)

        def encode_reference(self, pixels, *, image):
            assert image
            value = float(pixels.mean())
            self.anchor_values.append(value)
            return torch.full((1, 24, 1, 2, 2), value)

    video_encoder = VideoEncoder()
    audio_encoder = torch.nn.Linear(1, 1, bias=False)
    encoder = h3_integration._NativeLatentEncoder(video_encoder, audio_encoder, torch.float32)
    encoder._encode_audio = lambda item: (_ for _ in ()).throw(AssertionError("video-only target must not encode audio"))
    encoder._encode_references = lambda item: {}
    content = np.zeros((5, 32, 32, 3), dtype=np.uint8)
    content[-1] = 255
    item = SimpleNamespace(
        content=content,
        item_key="sample",
        h3_target_mode="video",
        h3_media_assets=(MediaAsset(Path("sample.mp4"), MediaModality.VIDEO, "target"),),
    )

    (tensors,) = encoder.encode_latents([item])

    key = f"varlen_{H3_KEYFRAME_VIDEO_ROWS_KEY}_float32"
    assert tensors[key].shape == (2, 96)
    assert len(video_encoder.anchor_values) == 2
    assert video_encoder.anchor_values[0] != video_encoder.anchor_values[1]
    assert not any(key.startswith(H3_AUDIO_LATENTS_KEY) for key in tensors)


def test_native_latent_encoder_uses_direct_image_vae_path_and_omits_audio():
    class VideoEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.image_calls = 0

        def encode_image(self, pixels):
            assert pixels.shape == (1, 3, 1, 32, 32)
            self.image_calls += 1
            return torch.zeros(1, 24, 1, 2, 2)

        def encode(self, pixels):
            raise AssertionError("image target must not use padded video encoding")

    video_encoder = VideoEncoder()
    encoder = h3_integration._NativeLatentEncoder(video_encoder, None, torch.float32)
    encoder._encode_references = lambda item: {}
    item = SimpleNamespace(
        content=np.zeros((32, 32, 3), dtype=np.uint8),
        item_key="image.png",
        h3_media_assets=(MediaAsset(Path("image.png"), MediaModality.IMAGE, "target"),),
    )

    (tensors,) = encoder.encode_latents([item])

    assert video_encoder.image_calls == 1
    assert set(tensors) == {"latents_1x2x2_float32"}


def test_native_latent_encoder_uses_temporal_vae_and_external_controls_for_conditioned_image(tmp_path):
    class VideoEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.video_calls = 0
            self.control_values = []

        def encode(self, pixels):
            assert pixels.shape == (1, 3, 5, 32, 32)
            self.video_calls += 1
            return torch.zeros(1, 24, 2, 2, 2)

        def encode_image(self, pixels):
            raise AssertionError("conditioned image must use temporal video encoding")

        def encode_reference(self, pixels, *, image):
            assert image
            self.control_values.append(float(pixels.mean()))
            return torch.zeros(1, 24, 1, 2, 2)

    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(first)
    Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8)).save(last)
    item = SimpleNamespace(
        content=np.full((5, 32, 32, 3), 127, dtype=np.uint8),
        item_key="image.png",
        h3_image_mode="first_last",
        h3_condition_paths=(first, last),
        h3_media_assets=(MediaAsset(Path("image.png"), MediaModality.IMAGE, "target"),),
    )
    video_encoder = VideoEncoder()
    encoder = h3_integration._NativeLatentEncoder(video_encoder, None, torch.float32)
    encoder._encode_references = lambda item: {}

    (tensors,) = encoder.encode_latents([item])

    assert video_encoder.video_calls == 1
    assert video_encoder.control_values[0] != video_encoder.control_values[1]
    assert "latents_2x2x2_float32" in tensors
    assert f"varlen_{H3_KEYFRAME_VIDEO_ROWS_KEY}_float32" in tensors
    assert not any(key.startswith(H3_AUDIO_LATENTS_KEY) for key in tensors)


def test_native_latent_encoder_pools_pixel_loss_masks_to_h3_latent_windows():
    class VideoEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def encode(self, pixels):
            assert pixels.shape[2] == 39
            return torch.zeros(1, 24, 12, 2, 2)

        def encode_reference(self, pixels, *, image):
            del pixels
            assert image
            return torch.zeros(1, 24, 1, 2, 2)

    mask = np.zeros((39, 32, 32), dtype=np.uint8)
    mask[0, 0, 0] = 255
    mask[5, -1, -1] = 255
    mask[22, 16, 16] = 255
    item = SimpleNamespace(
        content=np.zeros((39, 32, 32, 3), dtype=np.uint8),
        loss_mask_content=mask,
        item_key="masked.mp4",
        h3_target_mode="video",
        h3_media_assets=(MediaAsset(Path("masked.mp4"), MediaModality.VIDEO, "target"),),
    )
    encoder = h3_integration._NativeLatentEncoder(VideoEncoder(), None, torch.float32)
    encoder._encode_references = lambda item: {}

    (tensors,) = encoder.encode_latents([item])

    cached = tensors["video_loss_mask"]
    assert cached.shape == (12, 2, 2) and cached.dtype == torch.bool
    assert cached[0, 0, 0]
    assert cached[2:7, 1, 1].any()
    assert cached[7:].any()
    assert int(cached.sum()) == 3


def test_h3_training_uses_crop_specific_text_cache_identity(tmp_path):
    video_directory = tmp_path / "videos"
    cache_directory = tmp_path / "cache"
    video_directory.mkdir()
    cache_directory.mkdir()
    (video_directory / "sample.mp4").write_bytes(b"placeholder")
    config = {
        "general": {"resolution": [256, 256], "batch_size": 1},
        "datasets": [
            {
                "video_directory": str(video_directory),
                "cache_directory": str(cache_directory),
                "target_frames": [5],
            }
        ],
    }
    dataset_group, _ = create_h3_dataset_group(config, Namespace(debug_dataset=False))
    latent_path = cache_directory / "sample_00000-005_0256x0256_mmh3.safetensors"
    text_path = cache_directory / "sample_00000-005_mmh3_te.safetensors"
    save_file({"latents_2x16x16_float32": torch.zeros(24, 2, 16, 16)}, latent_path)
    save_file({"varlen_mmh3_hidden_states_float32": torch.zeros(1, 5120)}, text_path)

    dataset = dataset_group.datasets[0]
    dataset.prepare_for_training()
    cached_item = next(iter(next(iter(dataset.batch_manager.buckets.values()))))

    assert cached_item.text_encoder_output_cache_path == str(text_path)


def test_h3_latent_cache_requires_native_primary_latent_key(tmp_path):
    item = ItemInfo("sample", "caption", (16, 16), (16, 16), latent_cache_path=str(tmp_path / "cache.safetensors"))
    with pytest.raises(ValueError, match="latents_FxHxW"):
        save_latent_cache_minimax_h3(item, {"video_latents_float32": torch.ones(1)})

    with pytest.raises(ValueError, match="latents_audio"):
        save_latent_cache_minimax_h3(item, {"latents_2x2x2_float32": torch.zeros(24, 2, 2, 2)})

    with pytest.raises(ValueError, match="audio_loss_mask"):
        save_latent_cache_minimax_h3(
            item,
            {
                "latents_2x2x2_float32": torch.zeros(24, 2, 2, 2),
                "latents_audio_2x32x4_float32": torch.zeros(2, 32, 4),
            },
        )


def test_h3_dataset_accepts_loss_mask_sources(tmp_path):
    manifest = tmp_path / "videos.jsonl"
    manifest.write_text(json.dumps({"video_path": "target.mp4", "caption": "prompt"}), encoding="utf-8")
    config = {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "video_jsonl_file": str(manifest),
                "cache_directory": str(tmp_path / "cache"),
                "target_frames": [22],
                "frame_extraction": "uniform",
                "loss_mask_directory": str(tmp_path / "masks"),
                "default_loss_mask_path": str(tmp_path / "default.png"),
                "loss_mask_use_alpha": True,
                "loss_mask_invert": True,
            }
        ],
    }
    dataset_group, _ = create_h3_dataset_group(config, Namespace(debug_dataset=False))
    dataset = dataset_group.datasets[0]
    assert dataset.loss_mask_directory == str(tmp_path / "masks")
    assert dataset.default_loss_mask_path == str(tmp_path / "default.png")
    assert dataset.loss_mask_use_alpha and dataset.loss_mask_invert
    assert dataset.source_fps is None
    assert dataset.target_fps == 24.0
    assert dataset.vae_frame_stride == 17
    assert dataset.vae_frame_base == 5
    assert dataset.target_frames == (22,)


def test_image_datasource_loads_stem_matched_alpha_loss_mask(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.new("RGB", (2, 2), "black").save(images / "sample.png")
    (images / "sample.txt").write_text("caption", encoding="utf-8")
    alpha = np.array([[0, 255], [64, 128]], dtype=np.uint8)
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[..., 3] = alpha
    Image.fromarray(rgba).save(masks / "sample.png")
    datasource = ImageDirectoryDatasource(
        str(images),
        ".txt",
        loss_mask_directory=str(masks),
        loss_mask_invert=True,
    )

    _, _, _, _, loss_mask = datasource.get_image_data(0)

    np.testing.assert_array_equal(np.asarray(loss_mask), 255 - alpha)


def test_image_datasource_uses_target_alpha_as_loss_mask_fallback(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    alpha = np.array([[0, 255], [64, 128]], dtype=np.uint8)
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[..., 3] = alpha
    Image.fromarray(rgba).save(images / "sample.png")
    (images / "sample.txt").write_text("caption", encoding="utf-8")
    datasource = ImageDirectoryDatasource(str(images), ".txt", loss_mask_use_alpha=True)

    _, _, _, _, loss_mask = datasource.get_image_data(0)

    np.testing.assert_array_equal(np.asarray(loss_mask), alpha)


def test_video_jsonl_datasource_aligns_and_extends_loss_mask_frames(tmp_path):
    video_frames = tmp_path / "video"
    mask_frames = tmp_path / "mask"
    video_frames.mkdir()
    mask_frames.mkdir()
    for index in range(5):
        Image.new("RGB", (2, 2), "black").save(video_frames / f"{index:03}.png")
    Image.new("L", (2, 2), 0).save(mask_frames / "000.png")
    Image.new("L", (2, 2), 255).save(mask_frames / "001.png")
    manifest = tmp_path / "videos.jsonl"
    manifest.write_text(
        json.dumps({"video_path": str(video_frames), "caption": "caption", "loss_mask_path": str(mask_frames)}),
        encoding="utf-8",
    )

    datasource = VideoJsonlDatasource(str(manifest))
    _, video, _, _, loss_mask = datasource.get_video_data(0)

    assert len(video) == len(loss_mask) == 5
    assert not np.asarray(loss_mask[0]).any()
    assert all(np.asarray(frame).all() for frame in loss_mask[1:])


def test_cache_cli_uses_native_musubi_dataset_config(tmp_path):
    parser = create_cache_latents_parser()
    args = parser.parse_args(
        [
            "--dataset_config",
            str(tmp_path / "dataset.toml"),
            "--vae",
            str(tmp_path / "video_vae.safetensors"),
            "--audio_vae",
            str(tmp_path / "audio_vae.safetensors"),
        ]
    )

    assert args.dataset_config.endswith("dataset.toml")
    assert args.vae_dtype == "float32"
    assert args.vae.endswith("video_vae.safetensors")
    assert args.audio_vae.name == "audio_vae.safetensors"
    assert not hasattr(args, "dataset_manifest")

    text_args = create_cache_text_parser().parse_args(
        [
            "--dataset_config",
            str(tmp_path / "dataset.toml"),
            "--text_encoder",
            str(tmp_path / "text_encoder.safetensors"),
            "--tokenizer",
            str(tmp_path / "processor"),
        ]
    )
    assert text_args.task == "t2va"
    assert text_args.text_encoder_quantization == "none"

    bundled_text_args = create_cache_text_parser().parse_args(
        [
            "--dataset_config",
            str(tmp_path / "dataset.toml"),
            "--text_encoder",
            str(tmp_path / "text_encoder.safetensors"),
        ]
    )
    assert bundled_text_args.tokenizer == default_text_encoder_assets()

    quantized_text_args = create_cache_text_parser().parse_args(
        [
            "--dataset_config",
            str(tmp_path / "dataset.toml"),
            "--text_encoder",
            str(tmp_path / "text_encoder.safetensors"),
            "--tokenizer",
            str(tmp_path / "processor"),
            "--text_encoder_quantization",
            "nf4",
        ]
    )
    assert quantized_text_args.text_encoder_quantization == "nf4"

    nvfp4_text_args = create_cache_text_parser().parse_args(
        [
            "--dataset_config",
            str(tmp_path / "dataset.toml"),
            "--text_encoder",
            str(tmp_path / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
            "--tokenizer",
            str(tmp_path / "processor"),
            "--text_encoder_quantization",
            "nvfp4_awq",
        ]
    )
    assert nvfp4_text_args.text_encoder_quantization == "nvfp4_awq"


def test_h3_cache_parsers_do_not_advertise_transformer_loading_modes():
    unsupported = {"--fp8", "--fp8_scaled", "--fp8_text_encoder", "--int8", "--allow_prequantized_fp8", "--blocks_to_swap"}
    for parser in (create_cache_latents_parser(), create_cache_text_parser()):
        assert unsupported.isdisjoint(parser._option_string_actions)


def test_h3_generation_parser_exposes_native_inference_controls():
    parser = create_parser()
    for option in (
        "--text_encoder",
        "--tokenizer",
        "--vae",
        "--audio_vae",
        "--fp8_base",
        "--int8_convrot_base",
        "--blocks_to_swap",
        "--block_swap_granularity",
        "--lora_weight",
        "--compile",
        "--compile_fallback_to_eager",
        "--inductor_config",
        "--h3_fused_qk_norm_rope",
    ):
        assert option in parser._option_string_actions
    assert "--fp8_scaled" not in parser._option_string_actions
    defaults = parser.parse_args(["--model", "model", "--prompt", "prompt", "--output", "out.mp4"])
    assert defaults.tokenizer == default_text_encoder_assets()
    assert defaults.h3_image_mode == "none"
    assert defaults.h3_text_visual_max_pixels == 0


def test_h3_bundled_text_encoder_assets_are_complete():
    asset_dir = default_text_encoder_assets()
    assert {path.name for path in asset_dir.iterdir() if path.suffix == ".json"} >= {
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "video_preprocessor_config.json",
    }


@pytest.mark.parametrize(
    ("factory_name", "inputs", "extra"),
    [
        ("create_latent_encoder", {"video_vae": Path("video"), "audio_vae": Path("audio")}, {}),
        (
            "create_conditioning_encoder",
            {"text_encoder": Path("text"), "tokenizer": Path("tokenizer")},
            {"task": "t2va", "quantization": "none"},
        ),
        (
            "create_training_backend",
            {"model": Path("model")},
            {
                "mode": "ref2va",
                "attention_mode": "torch",
                "split_attention": False,
                "fp8_scaled": False,
                "quantization_device": None,
                "int8_convrot": False,
                "adaln_rank": None,
            },
        ),
    ],
)
def test_h3_component_factories_route_only_explicit_loading_inputs(monkeypatch, factory_name, inputs, extra):
    sentinel = object()
    captured = {}

    def create_component(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(h3_integration, factory_name, create_component)
    factory = getattr(h3_backend, factory_name)
    result = factory(device="cpu", dtype="float32", **inputs, **extra)

    assert result is sentinel
    assert captured == {**inputs, "device": "cpu", "dtype": "float32", **extra}


def test_h3_generator_factory_routes_all_native_components(monkeypatch):
    sentinel = object()
    captured = {}

    def create_component(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(h3_integration, "create_generator", create_component)
    request = H3GenerationRequest("prompt", Path("out.mp4"))
    result = h3_backend.create_generator(
        model=Path("model"),
        text_encoder=Path("text"),
        tokenizer=Path("tokenizer"),
        video_vae=Path("video"),
        audio_vae=Path("audio"),
        device="cpu",
        dtype="bfloat16",
        request=request,
    )

    assert result is sentinel
    assert captured["model"] == Path("model")
    assert captured["text_encoder"] == Path("text")
    assert captured["tokenizer"] == Path("tokenizer")
    assert captured["video_vae"] == Path("video")
    assert captured["audio_vae"] == Path("audio")
    assert captured["request"] is request
    assert captured["num_inference_steps"] == 20
    assert captured["fp8_scaled"] is False
    assert captured["int8_convrot"] is False
    assert captured["blocks_to_swap"] == 0


def test_native_generator_compiles_all_inference_block_regions(monkeypatch, tmp_path):
    from musubi_tuner.minimax_h3 import model_loader
    from musubi_tuner.utils import model_utils

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
            self.token_refiner = SimpleNamespace(blocks=torch.nn.ModuleList([torch.nn.Identity()]))
            self.fused_enabled = False

        def enable_fused_qk_norm_rope(self):
            self.fused_enabled = True

    transformer = Transformer()
    captured = {}

    monkeypatch.setattr(model_loader, "load_transformer", lambda *args, **kwargs: transformer)

    def compile_transformer(args, compiled_transformer, targets, disable_linear):
        captured["args"] = args
        captured["transformer"] = compiled_transformer
        captured["targets"] = targets
        captured["disable_linear"] = disable_linear
        return compiled_transformer

    monkeypatch.setattr(model_utils, "compile_transformer", compile_transformer)
    generator = h3_integration._NativeGenerator(
        model=tmp_path / "model.safetensors",
        text_encoder=tmp_path / "text.safetensors",
        tokenizer=tmp_path / "tokenizer",
        video_vae=tmp_path / "video.safetensors",
        audio_vae=tmp_path / "audio.safetensors",
        device=torch.device("cpu"),
        num_inference_steps=20,
        height=None,
        width=None,
        fp8_scaled=False,
        int8_convrot=False,
        text_encoder_quantization="none",
        blocks_to_swap=0,
        block_swap_h2d_only=False,
        block_swap_ring_size=2,
        block_swap_granularity="block",
        use_pinned_memory_for_block_swap=False,
        lora_weights=(),
        lora_multipliers=(),
        compile_model=True,
        compile_backend="eager",
        compile_mode="default",
        compile_dynamic="false",
        compile_fullgraph=True,
        compile_cache_size_limit=123,
        compile_auto_cache_size_limit=True,
        compile_fallback_to_eager=True,
        inductor_config=("max_autotune=false",),
        fused_qk_norm_rope=True,
        mode="fl2va",
    )

    loaded, networks = generator._load_transformer()

    assert loaded is transformer
    assert networks == []
    assert transformer.fused_enabled
    assert [len(blocks) for blocks in captured["targets"]] == [2, 1]
    assert captured["disable_linear"] is False
    assert captured["args"].compile_backend == "eager"
    assert captured["args"].compile_dynamic == "false"
    assert captured["args"].compile_fullgraph is True
    assert captured["args"].compile_cache_size_limit == 123
    assert captured["args"].compile_fallback_to_eager is True


@pytest.mark.parametrize(
    ("images", "anchors", "expected"),
    (
        ((), (), "t2va"),
        ((object(),), ("first",), "i2va"),
        ((object(),), ("last",), "l2va"),
        ((object(),), (11,), "fl2va"),
        ((object(), object()), ("first", "last"), "fl2va"),
    ),
)
def test_native_generator_routes_endpoint_conditioning_task(images, anchors, expected):
    assert h3_integration._NativeGenerator._conditioning_task(images=images, keyframe_anchors=anchors) == expected


def test_native_generator_selects_ref2va_checkpoint_contract(tmp_path):
    request = H3GenerationRequest(
        "prompt",
        tmp_path / "out.mp4",
        references=(H3Reference(tmp_path / "reference.png", ReferenceKind.IMAGE),),
    )

    generator = h3_integration.create_generator(
        model=tmp_path / "model.safetensors",
        text_encoder=tmp_path / "text.safetensors",
        tokenizer=tmp_path / "tokenizer",
        video_vae=tmp_path / "video_vae.safetensors",
        audio_vae=tmp_path / "audio_vae.safetensors",
        device="cpu",
        dtype="bfloat16",
        request=request,
    )

    assert generator.mode == "ref2va"


def test_h3_online_convrot_excludes_the_reduced_adaln():
    # The reduced AdaLN is tiny and its error is systematic across the whole
    # modulation curve, so it must stay out of the quantized set exactly as it
    # does on the FP8 path.
    from musubi_tuner.minimax_h3.model_loader import H3_FP8_OPTIMIZATION_EXCLUDE_KEYS
    from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer

    quantizer = ConvRotInt8Quantizer(["blocks."], H3_FP8_OPTIMIZATION_EXCLUDE_KEYS + ["adaln_proj"])

    assert not quantizer.is_target_key("blocks.0.adaln_proj.linear.weight")
    assert quantizer.is_target_key("blocks.0.attn.qkv_proj.weight")


def test_h3_online_convrot_targets_only_transformer_weights():
    from musubi_tuner.minimax_h3.model_loader import (
        H3_FP8_OPTIMIZATION_EXCLUDE_KEYS,
        H3_FP8_OPTIMIZATION_TARGET_KEYS,
    )
    from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer

    quantizer = ConvRotInt8Quantizer(H3_FP8_OPTIMIZATION_TARGET_KEYS, H3_FP8_OPTIMIZATION_EXCLUDE_KEYS)

    # Biases and non-weight tensors are never quantized.
    assert not quantizer.is_target_key("blocks.0.attn.qkv_proj.bias")
    assert not quantizer.is_target_key("adaln_t_table")


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ({"h3_convrot_int8": True, "fp8_base": True}, "drop --fp8_base"),
        ({"h3_convrot_int8": True, "int8_convrot_base": True}, "drop --fp8_base"),
        ({"h3_convrot_int8_bwd": "int8"}, "requires --h3_convrot_int8"),
        ({"h3_convrot_int8_fwd": "bf16"}, "requires --h3_convrot_int8"),
        (
            {"h3_convrot_int8": True, "h3_convrot_int8_fwd": "bf16", "h3_convrot_int8_bwd": "int8"},
            "no rotated activations",
        ),
    ],
)
def test_h3_online_convrot_rejects_conflicting_quantization(flags, message):
    from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser

    args = create_parser().parse_args([])
    for key, value in flags.items():
        setattr(args, key, value)

    with pytest.raises(ValueError, match=message):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def _convrot_linear(in_features=256, out_features=128, bias=True, seed=0):
    import torch
    from torch import nn

    from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight
    from musubi_tuner.modules.convrot_int8_utils import CONVROT_GROUPSIZE

    torch.manual_seed(seed)
    reference = nn.Linear(in_features, out_features, bias=bias)
    quantized, scale = quantize_int8_convrot_weight(reference.weight.detach(), CONVROT_GROUPSIZE)
    return reference, quantized, scale


def _patched_convrot_linear(reference, quantized, scale, *, fwd_mode):
    from torch import nn

    from musubi_tuner.modules.convrot_int8_utils import apply_convrot_int8_monkey_patch

    layer = nn.Linear(reference.in_features, reference.out_features, bias=reference.bias is not None)
    model = nn.Module()
    model.inner = layer
    apply_convrot_int8_monkey_patch(model, {"inner.scale_weight": scale}, fwd_mode=fwd_mode)
    layer.weight = nn.Parameter(quantized, requires_grad=False)
    layer.scale_weight = scale
    if reference.bias is not None:
        layer.bias = nn.Parameter(reference.bias.detach().clone())
    return layer


def _convrot_int8_kernel_available() -> bool:
    import torch

    from musubi_tuner.modules.convrot_int8_kernels import HAS_TRITON

    return HAS_TRITON and torch.cuda.is_available()


@pytest.mark.skipif(not _convrot_int8_kernel_available(), reason="the fused ConvRot kernel needs CUDA and triton")
def test_h3_convrot_bf16_forward_is_at_least_as_accurate_as_the_fused_kernel():
    # The fused kernel quantizes the activations as well as the weights; undoing
    # the rotation on the weight instead leaves them in BF16. Both share the same
    # weight error, so the BF16 route can only be the closer of the two, and this
    # is the assertion the CPU tests cannot make: without CUDA the INT8 branch
    # takes an eager fallback that does no activation quantization either.
    import torch

    reference, quantized, scale = _convrot_linear(in_features=512, out_features=512)
    reference = reference.cuda().to(torch.bfloat16)
    quantized, scale = quantized.cuda(), scale.cuda()
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)

    expected = reference(x).detach().float()
    error = {
        mode: float(
            (_patched_convrot_linear(reference, quantized, scale, fwd_mode=mode).cuda()(x).detach().float() - expected).norm()
        )
        for mode in ("int8", "bf16")
    }

    assert error["bf16"] <= error["int8"]


def test_h3_convrot_bf16_forward_matches_the_rotated_path():
    # Undoing an orthogonal rotation on the weight and rotating the activations
    # into it are the same arithmetic, so the two forward modes must agree. The
    # quantization error was fixed when the weight was stored, not here.
    #
    # Without CUDA the rotated mode takes its eager fallback rather than the fused
    # kernel, so this pins the identity rather than the kernel; the kernel is
    # covered by the on-device fidelity ladder in the docs.
    import torch

    reference, quantized, scale = _convrot_linear()
    x = torch.randn(4, 256)

    rotated = _patched_convrot_linear(reference, quantized, scale, fwd_mode="int8")(x)
    unrotated = _patched_convrot_linear(reference, quantized, scale, fwd_mode="bf16")(x)

    assert torch.allclose(rotated, unrotated, atol=2e-3, rtol=2e-3)


def test_h3_convrot_bf16_forward_stays_close_to_the_unquantized_weight():
    import torch

    reference, quantized, scale = _convrot_linear()
    x = torch.randn(4, 256)

    unrotated = _patched_convrot_linear(reference, quantized, scale, fwd_mode="bf16")(x)
    expected = reference(x)
    assert (unrotated - expected).norm() / expected.norm() < 0.05


def test_h3_convrot_bf16_forward_carries_gradients():
    import torch

    reference, quantized, scale = _convrot_linear()
    layer = _patched_convrot_linear(reference, quantized, scale, fwd_mode="bf16")
    x = torch.randn(4, 256, requires_grad=True)

    layer(x).sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()


def _prequantized_convrot_linear(
    in_features=16,
    out_features=8,
    group_size=4,
    bias=True,
    seed=0,
    **modes,
):
    """Build the module layout the pre-quantized Comfy checkpoint loader produces."""
    import torch
    from torch import nn

    from musubi_tuner.minimax_h3.int8_convrot import enable_int8_convrot, rotate_activation

    torch.manual_seed(seed)
    reference = nn.Linear(in_features, out_features, bias=bias)
    rotated = rotate_activation(reference.weight.detach().float(), group_size)
    scale = (rotated.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    quantized = (rotated / scale).round().clamp(-127, 127).to(torch.int8)

    layer = nn.Linear(in_features, out_features, bias=bias)
    layer.weight = nn.Parameter(quantized, requires_grad=False)
    if bias:
        layer.bias = nn.Parameter(reference.bias.detach().clone(), requires_grad=False)
    layer.register_buffer("scale_weight", scale.float())
    layer.register_buffer("int8_convrot_groupsize", torch.tensor(group_size, dtype=torch.int32))

    model = nn.Module()
    model.inner = layer
    assert enable_int8_convrot(model, **modes) == 1
    return reference, layer, model


def test_h3_prequantized_convrot_honors_the_forward_mode():
    # --int8_convrot_base used to ignore --h3_convrot_int8_fwd entirely. The stored
    # weights are the same rotated INT8 in both paths, so the BF16 forward is the same
    # arithmetic through a different route and must agree with the rotated one.
    import torch

    reference, int8_layer, _ = _prequantized_convrot_linear(fwd_mode="int8")
    _, bf16_layer, _ = _prequantized_convrot_linear(fwd_mode="bf16")
    assert int8_layer._convrot_fwd_mode == "int8"
    assert bf16_layer._convrot_fwd_mode == "bf16"

    x = torch.randn(6, 16)
    rotated = int8_layer(x)
    unrotated = bf16_layer(x)
    expected = reference(x)

    assert torch.allclose(rotated, unrotated, atol=2e-3, rtol=2e-3)
    assert (unrotated - expected).norm() / expected.norm() < 0.05


def test_h3_prequantized_convrot_bf16_forward_carries_gradients():
    import torch

    _, layer, _ = _prequantized_convrot_linear(fwd_mode="bf16")
    x = torch.randn(6, 16, requires_grad=True)

    layer(x).sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_h3_prequantized_convrot_rejects_bf16_forward_with_int8_backward():
    from torch import nn

    from musubi_tuner.minimax_h3.int8_convrot import enable_int8_convrot

    with pytest.raises(ValueError, match="no rotated activations"):
        enable_int8_convrot(nn.Module(), fwd_mode="bf16", bwd_mode="int8")


def test_h3_prequantized_convrot_stashes_the_group_size_off_device():
    # int8_convrot_groupsize is a CUDA buffer at training time; reading it with .item()
    # in the forward is a device sync per patched Linear per forward.
    import torch

    _, layer, _ = _prequantized_convrot_linear()
    assert isinstance(layer._convrot_groupsize, int) and layer._convrot_groupsize == 4

    del layer.int8_convrot_groupsize
    assert torch.isfinite(layer(torch.randn(6, 16))).all()


def test_h3_prequantized_convrot_falls_back_to_the_group_size_buffer():
    import torch

    _, layer, _ = _prequantized_convrot_linear()
    reference_output = layer(torch.randn(6, 16))

    del layer._convrot_groupsize
    torch.manual_seed(0)
    assert torch.isfinite(layer(torch.randn(6, 16))).all()
    assert reference_output.shape == (6, 8)


@pytest.mark.parametrize("bwd_mode", ["bf16", "int8"])
def test_h3_prequantized_convrot_backward_leaves_grad_output_untouched(monkeypatch, bwd_mode):
    # The INT8 backward folds the weight scale into the incoming gradient in place;
    # when grad_output is already fp32 the fold used to land on the caller's tensor.
    import torch

    from musubi_tuner.minimax_h3 import int8_convrot

    monkeypatch.setattr(int8_convrot, "_int8_available", lambda tensor: True)
    monkeypatch.setattr(int8_convrot, "_int_mm", lambda left, right: left.to(torch.int32) @ right.to(torch.int32))

    _, layer, _ = _prequantized_convrot_linear(bwd_mode=bwd_mode)
    x = torch.randn(6, 16, requires_grad=True)
    output = layer(x)
    grad_output = torch.randn_like(output)
    baseline = grad_output.clone()

    output.backward(grad_output)

    assert torch.equal(grad_output, baseline)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_h3_convrot_row_quantization_leaves_its_input_untouched():
    import torch

    from musubi_tuner.minimax_h3.int8_convrot import _quantize_rows

    value = torch.randn(4, 8)
    baseline = value.clone()

    _quantize_rows(value)

    assert torch.equal(value, baseline)


def test_h3_prequantized_convrot_modules_are_visible_to_the_lora_fusion(monkeypatch):
    # --h3_convrot_int8_lora_fused selects modules by _convrot_groupsize; the
    # pre-quantized loader used to register only the int8_convrot_groupsize buffer,
    # so the combination always raised "found no ConvRot INT8 Linear layers".
    import torch

    from musubi_tuner.modules import convrot_int8_utils

    monkeypatch.setattr(convrot_int8_utils, "HAS_TRITON", True)
    _, layer, model = _prequantized_convrot_linear()
    assert layer._convrot_lora_fused is False

    assert convrot_int8_utils.enable_convrot_int8_lora_fusion(model) == 1

    assert layer._convrot_lora_fused is True
    # what ConvRotInt8LoRAFn reads off the module
    assert layer.weight.dtype is torch.int8
    assert layer.scale_weight.shape == (8, 1) and layer.scale_weight.dtype is torch.float32
    assert isinstance(layer._convrot_groupsize, int)


def test_h3_convrot_lora_fusion_requires_triton(monkeypatch):
    from musubi_tuner.modules import convrot_int8_utils

    monkeypatch.setattr(convrot_int8_utils, "HAS_TRITON", False)
    _, _, model = _prequantized_convrot_linear()

    with pytest.raises(ValueError, match="requires triton"):
        convrot_int8_utils.enable_convrot_int8_lora_fusion(model)


# Every area-normalized grid spans an interval centered on _ROPE_SPATIAL_SCALE / 2,
# whatever the latent shape or the density scale, so scaling the area normalization
# contracts each coordinate toward that center by exactly the same factor.
_SPATIAL_GRID_CENTER = 16.0


def _centered_rescale(coordinates, scale):
    return _SPATIAL_GRID_CENTER + (coordinates - _SPATIAL_GRID_CENTER) / scale


def _t2va_density_layout(scale=None):
    return build_t2va_packed_sequence(
        torch.ones(4, dtype=torch.long),
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=3,
        patch_size=(1, 2, 2),
        **({} if scale is None else {"spatial_density_scale": scale}),
    )


def _ref2va_density_layout(scale=None):
    return build_ref2va_packed_sequence(
        torch.ones(4, dtype=torch.long),
        (MiniMaxH3ReferenceGeometry(kind=0, num_latent_frames=1, latent_height=8, latent_width=8),),
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=2,
        patch_size=(1, 2, 2),
        **({} if scale is None else {"spatial_density_scale": scale}),
    )


def test_h3_ref2va_packs_an_audio_only_target_behind_its_reference_prefix():
    # An audio target contributes no video rows at all: the sequence is
    # [text | references | target audio] and the target video block is empty.
    layout = build_ref2va_packed_sequence(
        torch.ones(4, dtype=torch.long),
        (
            MiniMaxH3ReferenceGeometry(kind=0, num_latent_frames=1, latent_height=4, latent_width=4),
            MiniMaxH3ReferenceGeometry(kind=2, num_audio_latents=3),
        ),
        num_latent_frames=0,
        latent_height=4,
        latent_width=4,
        num_audio_latents=2,
        patch_size=(1, 2, 2),
    )

    assert layout.num_condition_video_rows == 4
    assert layout.num_condition_audio_rows == 6
    assert layout.sequence_length == 4 + 4 + 6 + 2 * 2
    # Every video row in the sequence is a reference row; nothing follows the target audio.
    assert torch.equal(layout.video_indices, torch.arange(4, 8))
    assert torch.equal(layout.audio_indices, torch.cat((torch.arange(8, 14), torch.arange(14, 18))))
    assert int(layout.token_tags[layout.video_indices].unique()) == int(MiniMaxH3TokenTag.VIDEO)
    assert int(layout.token_tags[layout.audio_indices].unique()) == int(MiniMaxH3TokenTag.AUDIO)
    # The target audio starts after the reference timeline: one image row (+1) then
    # three audio latents (+3) past the text origin of four rows.
    target_audio = layout.position_ids[14:, 0]
    torch.testing.assert_close(target_audio, torch.tensor([8.0, 9.0, 8.0, 9.0], dtype=torch.float64))


def test_h3_temporal_position_grid_is_empty_without_video_frames():
    assert h3_packing._temporal_position_grid(0, 3.0).numel() == 0
    assert h3_packing._temporal_position_grid(1, 3.0).tolist() == [3.0]


def test_h3_spatial_density_scale_is_inert_at_its_default():
    assert torch.equal(_t2va_density_layout().position_ids, _t2va_density_layout(1.0).position_ids)
    assert torch.equal(_ref2va_density_layout().position_ids, _ref2va_density_layout(1.0).position_ids)


def test_h3_spatial_density_scale_contracts_only_the_spatial_grids():
    baseline = _t2va_density_layout()
    dense = _t2va_density_layout(2.0)

    assert torch.equal(dense.position_ids[:, 0], baseline.position_ids[:, 0])
    assert torch.equal(dense.position_ids[dense.text_indices], baseline.position_ids[baseline.text_indices])
    torch.testing.assert_close(
        dense.position_ids[dense.video_indices, 1:],
        _centered_rescale(baseline.position_ids[baseline.video_indices, 1:], 2.0),
    )
    # Audio rows sit at the outer coordinates of the video width grid, so they
    # follow the same contraction; their own timeline stays where it was.
    torch.testing.assert_close(
        dense.position_ids[dense.audio_indices, 2],
        _centered_rescale(baseline.position_ids[baseline.audio_indices, 2], 2.0),
    )
    assert torch.equal(dense.position_ids[dense.audio_indices, 1], baseline.position_ids[baseline.audio_indices, 1])


def test_h3_spatial_density_scale_moves_ref2va_references_with_their_target():
    baseline = _ref2va_density_layout()
    dense = _ref2va_density_layout(2.0)

    # Reference and target rows carry different latent areas, so a shared factor
    # is the only thing that contracts both grids by the same amount.
    assert baseline.num_condition_video_rows == 16
    assert torch.equal(dense.position_ids[:, 0], baseline.position_ids[:, 0])
    torch.testing.assert_close(
        dense.position_ids[dense.video_indices, 1:],
        _centered_rescale(baseline.position_ids[baseline.video_indices, 1:], 2.0),
    )


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan")])
def test_h3_spatial_density_scale_rejects_values_that_are_not_positive(scale):
    with pytest.raises(ValueError, match="spatial density scale"):
        _t2va_density_layout(scale)
    with pytest.raises(ValueError, match="spatial density scale"):
        _ref2va_density_layout(scale)


def test_h3_spatial_density_jitter_rejects_negative_values():
    from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser

    args = create_parser().parse_args([])
    args.h3_spatial_density_jitter = -0.1

    with pytest.raises(ValueError, match="h3_spatial_density_jitter"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_h3_spatial_density_jitter_draws_log_uniformly_inside_its_bounds():
    from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer

    trainer = MiniMaxH3NetworkTrainer()
    assert trainer._draw_spatial_density_scale() is None

    trainer._spatial_density_jitter = 0.5
    draws = [trainer._draw_spatial_density_scale() for _ in range(64)]

    assert all(abs(math.log(draw)) <= math.log(1.5) + 1e-12 for draw in draws)
    assert len(set(draws)) > 1
