import json
import struct
from argparse import Namespace
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open

from musubi_tuner.minimax_h3.architecture import (
    AUDIO_LATENT_FPS,
    AUDIO_SAMPLE_RATE,
    CANVAS_MULTIPLE,
    VIDEO_FLOW_SHIFT,
    AUDIO_FLOW_SHIFT,
    temporal_shape,
)
from musubi_tuner.minimax_h3.audio import audio_valid_mask_to_latent_mask, load_audio_asset, target_audio_processing_spec
from musubi_tuner.minimax_h3.media import (
    AudioProcessingSpec,
    CropMode,
    MediaModality,
    PadMode,
    MediaAsset,
    fit_audio_length,
    slice_media_asset,
)
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.config_utils import (
    BlueprintGenerator,
    ConfigSanitizer,
)
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3 import backend as h3_backend
from musubi_tuner.minimax_h3 import integration as h3_integration
from musubi_tuner.minimax_h3.cache import save_latent_cache_minimax_h3
from musubi_tuner.minimax_h3.dataset import create_h3_dataset_group
from musubi_tuner.minimax_h3_cache_latents import create_parser as create_cache_latents_parser
from musubi_tuner.minimax_h3_cache_text_encoder_outputs import create_parser as create_cache_text_parser
from musubi_tuner.minimax_h3.request import H3GenerationRequest, H3Reference, ReferenceKind, ReferenceRole
from musubi_tuner.minimax_h3.weights import CheckpointInspectionError, inspect_checkpoint
from musubi_tuner.minimax_h3_generate_video import create_parser, request_from_args


def _write_safetensors_header(path: Path, tensors: dict) -> None:
    header = json.dumps(tensors).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header)


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


def test_public_request_requires_paired_audio_and_caps_total_references(tmp_path):
    audio = H3Reference(tmp_path / "voice.wav", ReferenceKind.AUDIO)
    with pytest.raises(ValueError, match="paired"):
        H3GenerationRequest("prompt", tmp_path / "out.mp4", references=(audio,))

    references = tuple(H3Reference(tmp_path / f"image_{index}.png", ReferenceKind.IMAGE) for index in range(9))
    references += tuple(H3Reference(tmp_path / f"video_{index}.mp4", ReferenceKind.VIDEO) for index in range(3))
    references += (audio,)
    with pytest.raises(ValueError, match="12 ordinary references"):
        H3GenerationRequest("prompt", tmp_path / "out.mp4", references=references)


def test_verified_h3_temporal_contract():
    shape = temporal_shape(124)
    assert shape.video_latent_frames == 37
    assert shape.audio_latent_frames == 207
    assert shape.audio_samples == 207 * 800
    assert AUDIO_SAMPLE_RATE == 32000
    assert AUDIO_LATENT_FPS == 40
    assert CANVAS_MULTIPLE == 32
    assert (VIDEO_FLOW_SHIFT, AUDIO_FLOW_SHIFT) == (12.0, 3.0)

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
    assert not target_clip.valid_mask[-1]


def test_native_cache_io_records_audio_tensor_and_architecture(tmp_path):
    cache = tmp_path / "sample_h3.safetensors"
    item = ItemInfo("sample", "caption", (1280, 720), (1280, 720), frame_count=22, latent_cache_path=str(cache))
    save_latent_cache_minimax_h3(
        item,
        {
            "latents_2x2x3_float32": torch.ones(24, 2, 2, 3),
            "audio_latents_float32": torch.zeros(2, 32, 4),
            "audio_loss_mask": torch.ones(4, dtype=torch.bool),
        },
    )

    with safe_open(cache, framework="pt") as handle:
        assert set(handle.keys()) == {"audio_latents_float32", "audio_loss_mask", "latents_2x2x3_float32"}
        assert handle.metadata()["architecture"] == "minimax_h3"
        assert handle.metadata()["frame_count"] == "22"


def test_h3_latent_cache_requires_native_primary_latent_key(tmp_path):
    item = ItemInfo("sample", "caption", (16, 16), (16, 16), latent_cache_path=str(tmp_path / "cache.safetensors"))
    with pytest.raises(ValueError, match="latents_FxHxW"):
        save_latent_cache_minimax_h3(item, {"video_latents_float32": torch.ones(1)})

    with pytest.raises(ValueError, match="audio_latents"):
        save_latent_cache_minimax_h3(item, {"latents_2x2x2_float32": torch.zeros(24, 2, 2, 2)})

    with pytest.raises(ValueError, match="audio_loss_mask"):
        save_latent_cache_minimax_h3(
            item,
            {
                "latents_2x2x2_float32": torch.zeros(24, 2, 2, 2),
                "audio_latents_float32": torch.zeros(2, 32, 4),
            },
        )


def test_h3_uses_existing_video_dataset_fields_only(tmp_path):
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
            }
        ],
    }
    dataset_group, _ = create_h3_dataset_group(config, Namespace(debug_dataset=False))
    dataset = dataset_group.datasets[0]
    assert dataset.source_fps is None
    assert dataset.target_fps == 24.0
    assert dataset.vae_frame_stride == 17
    assert dataset.vae_frame_base == 5
    assert dataset.target_frames == (22,)


def test_cache_cli_uses_native_musubi_dataset_config(tmp_path):
    parser = create_cache_latents_parser()
    args = parser.parse_args(
        [
            "--dataset_config",
            str(tmp_path / "dataset.toml"),
            "--model",
            str(tmp_path / "model"),
        ]
    )

    assert args.dataset_config.endswith("dataset.toml")
    assert args.vae_dtype == "float32"
    assert not hasattr(args, "dataset_manifest")


def test_h3_cache_and_generation_parsers_do_not_advertise_unimplemented_loading_modes():
    unsupported = {"--fp8", "--fp8_scaled", "--fp8_text_encoder", "--int8", "--allow_prequantized_fp8", "--blocks_to_swap"}
    for parser in (create_cache_latents_parser(), create_cache_text_parser(), create_parser()):
        assert unsupported.isdisjoint(parser._option_string_actions)


@pytest.mark.parametrize(
    ("factory_name", "extra"),
    [
        ("create_latent_encoder", {}),
        ("create_conditioning_encoder", {}),
        ("create_generator", {"request": H3GenerationRequest("prompt", Path("out.mp4"))}),
        ("create_training_backend", {"mode": "ref2va", "attention_mode": "torch", "split_attention": False}),
    ],
)
def test_h3_component_factories_route_only_explicit_loading_inputs(monkeypatch, tmp_path, factory_name, extra):
    sentinel = object()
    captured = {}

    def create_component(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(h3_integration, factory_name, create_component)
    factory = getattr(h3_backend, factory_name)
    result = factory(model=tmp_path, device="cpu", dtype="float32", **extra)

    assert result is sentinel
    assert captured == {"model": tmp_path, "device": "cpu", "dtype": "float32", **extra}
