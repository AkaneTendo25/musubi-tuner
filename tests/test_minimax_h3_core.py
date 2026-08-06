import json
import struct
import wave
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.config_utils import (
    BlueprintGenerator,
    ConfigSanitizer,
)
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3 import backend as h3_backend
from musubi_tuner.minimax_h3 import integration as h3_integration
from musubi_tuner.minimax_h3.architecture import (
    AUDIO_FLOW_SHIFT,
    AUDIO_LATENT_FPS,
    AUDIO_SAMPLE_RATE,
    CANVAS_MULTIPLE,
    VIDEO_FLOW_SHIFT,
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
from musubi_tuner.minimax_h3.cache import H3_AUDIO_LATENTS_KEY, H3_KEYFRAME_VIDEO_ROWS_KEY, save_latent_cache_minimax_h3
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
from musubi_tuner.minimax_h3.references import (
    resample_reference_frames,
    resolve_reference_image_size,
    resolve_reference_video_size,
    trim_reference_frames,
)
from musubi_tuner.minimax_h3.request import H3GenerationRequest, H3Reference, ReferenceKind, ReferenceRole
from musubi_tuner.minimax_h3.weights import CheckpointInspectionError, inspect_checkpoint
from musubi_tuner.minimax_h3_cache_latents import create_parser as create_cache_latents_parser
from musubi_tuner.minimax_h3_cache_text_encoder_outputs import create_parser as create_cache_text_parser
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
    assert trim_reference_frames(1) == 22
    assert trim_reference_frames(25) == 22
    assert trim_reference_frames(124) == 124

    frames = np.arange(30, dtype=np.uint8).reshape(-1, 1, 1, 1) * np.ones((1, 2, 2, 3), dtype=np.uint8)
    resampled = resample_reference_frames(frames, 30.0)
    assert [int(frame[0, 0, 0]) for frame in resampled] == [index for index in range(30) if index not in (2, 7, 12, 17, 22, 27)]


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
    assert Path(item.latent_cache_path).name == "tone_00000-124_0832x0480_mmh3.safetensors"
    assert Path(item.text_encoder_output_cache_path).name == "tone_00000-124_mmh3_te.safetensors"


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
    ):
        assert option in parser._option_string_actions
    assert "--fp8_scaled" not in parser._option_string_actions
    assert parser.parse_args(["--model", "model", "--prompt", "prompt", "--output", "out.mp4"]).tokenizer == (
        default_text_encoder_assets()
    )


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


def test_h3_reference_scale_shrinks_the_prepared_size():
    from musubi_tuner.minimax_h3.references import resolve_reference_video_size

    full_h, full_w = resolve_reference_video_size(1920, 1080)
    half_h, half_w = resolve_reference_video_size(1920, 1080, scale=0.5)

    # Rows fall with the square of the scale, so halving the edge quarters them.
    assert half_h * half_w == pytest.approx(full_h * full_w / 4, rel=0.05)
    assert half_h % 32 == 0 and half_w % 32 == 0


def test_h3_reference_scale_preserves_aspect_ratio():
    from musubi_tuner.minimax_h3.references import resolve_reference_video_size

    full_h, full_w = resolve_reference_video_size(1920, 1080)
    half_h, half_w = resolve_reference_video_size(1920, 1080, scale=0.5)

    assert (full_w / full_h) == pytest.approx(half_w / half_h, rel=0.05)


@pytest.mark.parametrize("scale", [0.1, 1.5, 0.0])
def test_h3_reference_scale_is_bounded(scale):
    from musubi_tuner.minimax_h3.references import resolve_reference_video_size

    with pytest.raises(ValueError, match=r"\[0.25, 1.0\]"):
        resolve_reference_video_size(1920, 1080, scale=scale)


def test_h3_reference_scale_defaults_to_the_released_size():
    from musubi_tuner.minimax_h3.references import REFERENCE_VIDEO_SHORT_EDGE, resolve_reference_video_size

    height, width = resolve_reference_video_size(1920, 1080)

    assert min(height, width) == pytest.approx(REFERENCE_VIDEO_SHORT_EDGE, abs=32)


def test_h3_reference_scale_spec_parsing():
    from musubi_tuner.minimax_h3_cache_latents import parse_reference_scales

    assert parse_reference_scales(None) is None
    assert parse_reference_scales("1.0,0.75,0.5") == (1.0, 0.75, 0.5)
    with pytest.raises(ValueError, match="at least one scale"):
        parse_reference_scales(",")


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
