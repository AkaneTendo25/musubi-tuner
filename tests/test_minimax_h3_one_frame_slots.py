"""One-frame target slots, signed placement, slot noise coupling and the reference route."""

import inspect
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3 import inference as h3_inference
from musubi_tuner.minimax_h3 import integration as h3_integration
from musubi_tuner.minimax_h3.cache import (
    H3_CONDITIONING_TASK_IDS,
    H3_CONDITIONING_TASK_KEY,
    H3_ONE_FRAME_CONTROL_INDICES_KEY,
    H3_ONE_FRAME_TARGET_INDEX_KEY,
    H3_REFERENCE_AUDIO_LENGTHS_KEY,
    H3_REFERENCE_AUDIO_ROWS_KEY,
    H3_REFERENCE_KINDS_KEY,
    H3_REFERENCE_ROUTE_KEY,
    H3_REFERENCE_VIDEO_ROWS_KEY,
    H3_REFERENCE_VIDEO_SHAPES_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
    save_latent_cache_minimax_h3,
)
from musubi_tuner.minimax_h3.conditioning import MiniMaxH3ConditioningEncoder
from musubi_tuner.minimax_h3.dataset import H3DatasetAdapter, create_h3_dataset_group
from musubi_tuner.minimax_h3.image_training import (
    H3_ONE_FRAME_CONTENT_FINGERPRINT_KEY,
    H3_ONE_FRAME_LATENT_FINGERPRINT_KEY,
    H3_ONE_FRAME_TARGET_INDICES_KEY,
)
from musubi_tuner.minimax_h3.integration import _NativeGenerator, _NativeLatentEncoder, _NativeTrainingBackend
from musubi_tuner.minimax_h3.media import MediaAsset, MediaModality
from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig
from musubi_tuner.minimax_h3.one_frame import (
    H3_REFERENCE_ROUTE_IDS,
    couple_target_slot_noise,
    one_frame_origin_shift,
    parse_one_frame_options,
    target_slot_output_paths,
)
from musubi_tuner.minimax_h3.packing import (
    MiniMaxH3ReferenceGeometry,
    build_ref2va_packed_sequence,
    build_t2va_packed_sequence,
)
from musubi_tuner.minimax_h3.references import H3ReferenceKind
from musubi_tuner.minimax_h3.request import H3GenerationRequest, H3Reference, ReferenceKind
from musubi_tuner.minimax_h3.training import target_slot_losses
from musubi_tuner.minimax_h3_generate_video import create_parser as create_generate_parser
from musubi_tuner.minimax_h3_generate_video import expected_outputs, request_from_args
from musubi_tuner.minimax_h3_train_network import (
    MiniMaxH3NetworkTrainer,
    _one_frame_target_slot_count,
    _require_one_frame_opt_in,
    _require_slot_layouts_for_coupling,
    _target_slot_metrics,
)
from musubi_tuner.minimax_h3_train_network import (
    create_parser as create_training_parser,
)

FRAME = 5.0 / 3.0


# --- placement parsing -------------------------------------------------------


def test_target_indices_option_parses_signed_slots_and_controls():
    assert parse_one_frame_options("target_indices=-1;0;1,control_index=5") == ((-1, 0, 1), (5,))
    assert parse_one_frame_options("target_indices=-1;1,control_index=-3;4") == ((-1, 1), (-3, 4))
    assert parse_one_frame_options("target_indices=7") == ((7,), None)
    # The scalar form keeps its unsigned contract and int return.
    assert parse_one_frame_options("target_index=24,control_index=0;48") == (24, (0, 48))


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("target_indices=-1;0;1,control_index=0", "distinct"),
        ("target_indices=1;1", "distinct"),
        ("target_indices=1,control_index=2;2", "distinct"),
        ("target_index=1,target_indices=2;3", "not both"),
        ("target_indices=a;1", "integer"),
        ("target_index=2,control_index=-1", "non-negative"),
    ],
)
def test_target_indices_option_rejects_invalid_placement(spec, match):
    with pytest.raises(ValueError, match=match):
        parse_one_frame_options(spec)


def test_origin_shift_moves_only_negative_placements():
    assert one_frame_origin_shift((0, 24), (48,)) == 0
    assert one_frame_origin_shift((-1, 0, 1)) == 1
    assert one_frame_origin_shift((2, 5), (-4,)) == 4


# --- packed layout ----------------------------------------------------------


def test_t2va_slots_place_each_slot_at_its_signed_time_after_the_text_origin():
    tags = torch.tensor([1, 1, 1], dtype=torch.long)
    layout = build_t2va_packed_sequence(
        tags,
        num_latent_frames=3,
        latent_height=4,
        latent_width=4,
        num_audio_latents=2,
        patch_size=(1, 2, 2),
        one_frame_target_indices=(-1, 0, 1),
        one_frame_control_indices=(-3,),
    )
    rows_per_frame = 4
    text_rows = 3
    times = layout.position_ids[:, 0]
    # Shift 3 puts the earliest index (-3, the control) on the text-defined origin.
    control = times[text_rows : text_rows + rows_per_frame]
    assert torch.equal(control, torch.full((rows_per_frame,), 3.0, dtype=torch.float64))
    target_rows = times[-3 * rows_per_frame :].reshape(3, rows_per_frame)
    expected = torch.tensor([3 + FRAME * 2, 3 + FRAME * 3, 3 + FRAME * 4], dtype=torch.float64)
    assert torch.equal(target_rows, expected[:, None].expand(3, rows_per_frame))
    audio = layout.audio_indices
    assert torch.equal(layout.position_ids[audio[0], 0], expected[0])
    # No media row takes a temporal coordinate inside the text block.
    media = torch.cat((layout.video_indices, layout.audio_indices))
    assert float(layout.position_ids[media, 0].min()) >= text_rows
    # Every slot shares the one frame's spatial grid.
    spatial = layout.position_ids[-3 * rows_per_frame :, 1:].reshape(3, rows_per_frame, 2)
    assert torch.equal(spatial[0], spatial[1]) and torch.equal(spatial[1], spatial[2])


def test_single_non_negative_slot_packs_exactly_like_the_scalar_target_index():
    tags = torch.tensor([1, 0, 1], dtype=torch.long)
    common = {"num_latent_frames": 1, "latent_height": 4, "latent_width": 4, "num_audio_latents": 2, "patch_size": (1, 2, 2)}
    scalar = build_t2va_packed_sequence(tags, one_frame_target_index=24, one_frame_control_indices=(0, 48), **common)
    slots = build_t2va_packed_sequence(tags, one_frame_target_indices=(24,), one_frame_control_indices=(0, 48), **common)
    for name in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"):
        assert torch.equal(getattr(scalar, name), getattr(slots, name))


@pytest.mark.parametrize(
    ("targets", "controls", "frames", "match"),
    [
        ((0, 1), None, 3, "one latent frame per slot"),
        ((0, 0), None, 2, "distinct"),
        ((0, 1), (1,), 2, "distinct"),
    ],
)
def test_t2va_slots_reject_mismatched_or_duplicate_placement(targets, controls, frames, match):
    with pytest.raises(ValueError, match=match):
        build_t2va_packed_sequence(
            torch.tensor([1], dtype=torch.long),
            num_latent_frames=frames,
            latent_height=4,
            latent_width=4,
            num_audio_latents=2,
            patch_size=(1, 2, 2),
            one_frame_target_indices=targets,
            one_frame_control_indices=controls,
        )


def test_ref2va_slots_are_measured_from_the_end_of_the_reference_block():
    tags = torch.tensor([1, 0, 1], dtype=torch.long)
    reference = MiniMaxH3ReferenceGeometry(kind=0, num_latent_frames=1, latent_height=2, latent_width=2)
    layout = build_ref2va_packed_sequence(
        tags,
        (reference,),
        num_latent_frames=2,
        latent_height=2,
        latent_width=2,
        num_audio_latents=2,
        patch_size=(1, 2, 2),
        one_frame_target_indices=(-2, 2),
    )
    times = layout.position_ids[:, 0]
    assert float(times[3]) == 3.0  # the image reference
    reference_end = 4.0
    assert torch.equal(times[-2:], torch.tensor([reference_end, reference_end + FRAME * 4], dtype=torch.float64))


# --- noise coupling ---------------------------------------------------------


def test_shared_coupling_repeats_slot_zero_and_keeps_a_standard_normal_marginal():
    torch.manual_seed(0)
    noise = torch.randn(1, 4, 3, 64, 64)
    shared = couple_target_slot_noise(noise, "shared")
    assert torch.equal(shared[:, :, 1], shared[:, :, 0]) and torch.equal(shared[:, :, 2], shared[:, :, 0])
    assert torch.equal(shared[:, :, 0], noise[:, :, 0])
    for slot in range(3):
        values = shared[:, :, slot]
        assert abs(float(values.mean())) < 0.05
        assert abs(float(values.std()) - 1.0) < 0.05


def test_independent_coupling_and_single_slots_are_untouched():
    noise = torch.randn(1, 4, 3, 8, 8)
    assert couple_target_slot_noise(noise, "independent") is noise
    assert not torch.equal(noise[:, :, 0], noise[:, :, 1])
    single = torch.randn(1, 4, 1, 8, 8)
    assert couple_target_slot_noise(single, "shared") is single
    with pytest.raises(ValueError, match="coupling"):
        couple_target_slot_noise(noise, "mixed")


# --- dataset ----------------------------------------------------------------


def _slot_config(tmp_path, *, target_indices=(-1, 0, 1), control_indices=(5,), slot_images=3, multiple_target=True):
    targets = tmp_path / "targets"
    controls = tmp_path / "controls"
    targets.mkdir(parents=True)
    controls.mkdir(parents=True)
    Image.new("RGB", (80, 64), color="white").save(targets / "sample.png")
    for slot in range(1, slot_images):
        Image.new("RGB", (80, 64), color=(slot * 40, 0, 0)).save(targets / f"sample_{slot}.png")
    (targets / "sample.txt").write_text("three lighting layers", encoding="utf-8")
    for index in range(len(control_indices)):
        Image.new("RGB", (48, 48), color="blue").save(controls / f"sample_{index}.png")
    dataset = {
        "target_image_directory": str(targets),
        "target_modalities": ["image"],
        "cache_directory": str(tmp_path / "cache"),
        "multiple_target": multiple_target,
        "fp_1f_target_indices": list(target_indices),
    }
    if control_indices:
        dataset.update(source_image_directory=str(controls), source_modalities=["image"], fp_1f_clean_indices=list(control_indices))
    config = {"general": {"resolution": [64, 64], "batch_size": 1, "caption_extension": ".txt"}, "datasets": [dataset]}
    return config, targets / "sample.png"


def _args(task="fl2va"):
    return Namespace(one_frame=True, task=task, h3_image_mode="none")


def _slot_item(target, slots=3):
    content = [np.full((64, 64, 3), slot * 10, dtype=np.uint8) for slot in range(slots)]
    return ItemInfo(str(target), "caption", (80, 64), (64, 64), content=content if slots > 1 else content[0])


def test_slot_dataset_attaches_signed_slots_and_stacks_targets_in_order(tmp_path):
    config, target = _slot_config(tmp_path)
    adapter = H3DatasetAdapter(config, _args())
    item = _slot_item(target)

    assets = adapter.attach(item)

    assert item.h3_one_frame is True
    assert item.h3_one_frame_target_index is None
    assert item.h3_one_frame_target_indices == (-1, 0, 1)
    assert item.h3_one_frame_control_indices == (5,)
    assert item.content.shape == (3, 64, 64, 3)
    assert [int(frame[0, 0, 0]) for frame in item.content] == [0, 10, 20]
    assert [path.name for path in item.h3_target_paths] == ["sample.png", "sample_1.png", "sample_2.png"]
    assert len(assets) == 1
    assert adapter.one_frame_target_slot_layouts() == ((-1, 0, 1),)

    item.latent_cache_path = str(tmp_path / "slots_mmh3.safetensors")
    save_latent_cache_minimax_h3(
        item,
        {
            "latents_3x4x4_float32": torch.zeros(24, 3, 4, 4),
            "latents_audio_2x32x2_float32": torch.zeros(2, 32, 2),
            "audio_loss_mask": torch.zeros(2, dtype=torch.bool),
            "latents_cond_000_1x4x4_float32": torch.zeros(24, 1, 4, 4),
        },
    )
    with safe_open(item.latent_cache_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        assert f"{H3_ONE_FRAME_TARGET_INDEX_KEY}_int64" not in keys
        assert handle.get_tensor(f"{H3_ONE_FRAME_TARGET_INDICES_KEY}_int64").tolist() == [-1, 0, 1]
        assert handle.get_tensor(f"{H3_ONE_FRAME_CONTROL_INDICES_KEY}_int64").tolist() == [5]


def test_slot_indices_change_only_the_latent_fingerprint(tmp_path):
    config, target = _slot_config(tmp_path, target_indices=(-1, 0, 1))
    first = _slot_item(target)
    H3DatasetAdapter(config, _args()).attach(first)
    repeated = _slot_item(target)
    H3DatasetAdapter(config, _args()).attach(repeated)
    changed = dict(config)
    changed["datasets"] = [dict(config["datasets"][0], fp_1f_target_indices=[-2, 0, 2])]
    second = _slot_item(target)
    H3DatasetAdapter(changed, _args()).attach(second)
    assert first.h3_cache_metadata == repeated.h3_cache_metadata
    assert (
        first.h3_cache_metadata[H3_ONE_FRAME_CONTENT_FINGERPRINT_KEY]
        == second.h3_cache_metadata[H3_ONE_FRAME_CONTENT_FINGERPRINT_KEY]
    )
    assert (
        first.h3_cache_metadata[H3_ONE_FRAME_LATENT_FINGERPRINT_KEY]
        != second.h3_cache_metadata[H3_ONE_FRAME_LATENT_FINGERPRINT_KEY]
    )


def test_slot_list_form_is_not_the_scalar_fingerprint(tmp_path):
    config, target = _slot_config(
        tmp_path / "list", target_indices=(24,), control_indices=(0,), slot_images=1, multiple_target=False
    )
    listed = _slot_item(target, slots=1)
    H3DatasetAdapter(config, _args()).attach(listed)
    scalar_config = dict(config)
    dataset = dict(config["datasets"][0])
    del dataset["fp_1f_target_indices"]
    scalar_config["datasets"] = [dict(dataset, fp_1f_target_index=24)]
    scalar = _slot_item(target, slots=1)
    H3DatasetAdapter(scalar_config, _args()).attach(scalar)
    assert listed.h3_one_frame_target_indices == (24,)
    assert scalar.h3_one_frame_target_index == 24
    assert (
        listed.h3_cache_metadata[H3_ONE_FRAME_LATENT_FINGERPRINT_KEY]
        != scalar.h3_cache_metadata[H3_ONE_FRAME_LATENT_FINGERPRINT_KEY]
    )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"target_indices": (-1, 0, 1), "control_indices": (0,)}, "distinct"),
        ({"target_indices": (2, 2, 3)}, "distinct"),
        ({"target_indices": (-1, 1)}, "has 3 target image"),
        ({"multiple_target": False}, "requires multiple_target"),
    ],
)
def test_slot_dataset_validation(tmp_path, changes, match):
    config, _ = _slot_config(tmp_path, **changes)
    with pytest.raises(ValueError, match=match):
        H3DatasetAdapter(config, _args())


def test_slot_dataset_rejects_scalar_and_list_forms_together(tmp_path):
    config, _ = _slot_config(tmp_path)
    config["datasets"][0]["fp_1f_target_index"] = 0
    with pytest.raises(ValueError, match="not both"):
        H3DatasetAdapter(config, _args())


def test_slot_dataset_rejects_slots_with_different_resolutions(tmp_path):
    config, target = _slot_config(tmp_path)
    Image.new("RGB", (96, 64)).save(target.with_name("sample_2.png"))
    with pytest.raises(ValueError, match="share one resolution"):
        H3DatasetAdapter(config, _args())


def test_scalar_form_keeps_its_unsigned_contract(tmp_path):
    config, _ = _slot_config(tmp_path, slot_images=1, multiple_target=False)
    dataset = config["datasets"][0]
    del dataset["fp_1f_target_indices"]
    dataset["fp_1f_target_index"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        H3DatasetAdapter(config, _args())


def test_slot_jsonl_passes_the_real_dataset_group(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name, color in (("a0.png", "red"), ("a1.png", "green")):
        Image.new("RGB", (64, 64), color=color).save(images / name)
    jsonl = tmp_path / "targets.jsonl"
    jsonl.write_text(
        json.dumps({"image_path_0": "images/a0.png", "image_path_1": "images/a1.png", "caption": "two slots"}) + "\n",
        encoding="utf-8",
    )
    config = {
        "general": {"resolution": [64, 64], "batch_size": 1, "enable_bucket": True},
        "datasets": [
            {
                "image_jsonl_file": str(jsonl),
                "cache_directory": str(tmp_path / "cache"),
                "multiple_target": True,
                "fp_1f_target_indices": [-2, 2],
            }
        ],
    }
    args = Namespace(one_frame=True, task="t2va", h3_image_mode="none", debug_dataset=False)

    group, adapter = create_h3_dataset_group(config, args)
    _, batch = next(group.datasets[0].retrieve_latent_cache_batches(1))
    adapter.attach(batch[0])

    assert batch[0].h3_one_frame_target_indices == (-2, 2)
    assert batch[0].content.shape == (2, 64, 64, 3)
    assert tuple(batch[0].content[0, 0, 0]) == (255, 0, 0)
    assert tuple(batch[0].content[1, 0, 0])[1] > 100


def _padded_jsonl_config(tmp_path, record):
    images = tmp_path / "images"
    images.mkdir()
    for name, color in (("a0.png", "red"), ("a1.png", "green")):
        Image.new("RGB", (64, 64), color=color).save(images / name)
    jsonl = tmp_path / "targets.jsonl"
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return {
        "general": {"resolution": [64, 64], "batch_size": 1, "enable_bucket": True},
        "datasets": [
            {
                "image_jsonl_file": str(jsonl),
                "cache_directory": str(tmp_path / "cache"),
                "multiple_target": True,
                "fp_1f_target_indices": [-2, 2],
            }
        ],
    }


def test_slot_jsonl_accepts_zero_padded_image_path_keys(tmp_path):
    config = _padded_jsonl_config(
        tmp_path, {"image_path_0000": "images/a0.png", "image_path_0001": "images/a1.png", "caption": "two slots"}
    )
    args = Namespace(one_frame=True, task="t2va", h3_image_mode="none", debug_dataset=False)

    group, adapter = create_h3_dataset_group(config, args)
    _, batch = next(group.datasets[0].retrieve_latent_cache_batches(1))
    adapter.attach(batch[0])

    assert [path.name for path in batch[0].h3_target_paths] == ["a0.png", "a1.png"]
    assert batch[0].content.shape == (2, 64, 64, 3)
    assert tuple(batch[0].content[0, 0, 0]) == (255, 0, 0)


def test_slot_jsonl_rejects_two_keys_for_one_index(tmp_path):
    config = _padded_jsonl_config(
        tmp_path,
        {"image_path_0": "images/a0.png", "image_path_00": "images/a1.png", "image_path_1": "images/a1.png", "caption": "c"},
    )
    with pytest.raises(ValueError, match="duplicate image_path index 0"):
        H3DatasetAdapter(config, Namespace(one_frame=True, task="t2va", h3_image_mode="none"))


def test_slot_directory_with_indexed_primary_passes_the_real_dataset_group(tmp_path):
    targets = tmp_path / "targets"
    targets.mkdir()
    for slot, color in enumerate(("red", "green", "blue")):
        Image.new("RGB", (64, 64), color=color).save(targets / f"scene_{slot}.png")
    (targets / "scene.txt").write_text("three slots", encoding="utf-8")
    config = {
        "general": {"resolution": [64, 64], "batch_size": 1, "enable_bucket": True, "caption_extension": ".txt"},
        "datasets": [
            {
                "image_directory": str(targets),
                "cache_directory": str(tmp_path / "cache"),
                "multiple_target": True,
                "fp_1f_target_indices": [0, 12, 24],
            }
        ],
    }
    args = Namespace(one_frame=True, task="t2va", h3_image_mode="none", debug_dataset=False)

    group, adapter = create_h3_dataset_group(config, args)
    _, batch = next(group.datasets[0].retrieve_latent_cache_batches(1))
    adapter.attach(batch[0])

    assert Path(batch[0].item_key).name == "scene_0.png"
    assert batch[0].h3_one_frame_target_indices == (0, 12, 24)
    assert [tuple(frame[0, 0]) for frame in batch[0].content] == [(255, 0, 0), (0, 128, 0), (0, 0, 255)]


def test_user_config_cannot_set_the_private_slot_carrier(tmp_path):
    config, _ = _slot_config(tmp_path)
    config["datasets"][0]["h3_indexed_targets"] = True
    with pytest.raises(ValueError, match="h3_indexed_targets"):
        H3DatasetAdapter(config, _args())


# --- latent caching ---------------------------------------------------------


class _FakeVideoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.image_calls = []

    def encode_image(self, pixels):
        assert pixels.shape[2] == 1
        self.image_calls.append(float(pixels.mean()))
        return torch.full((pixels.shape[0], 24, 1, pixels.shape[-2] // 16, pixels.shape[-1] // 16), float(len(self.image_calls)))

    def encode_reference(self, pixels, image=False):
        return torch.zeros(1, 24, 1, pixels.shape[-2] // 16, pixels.shape[-1] // 16)


class _FakeAudioEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def encode(self, waveform):
        return torch.zeros(2, 32, 2)


def test_latent_encoder_encodes_every_slot_as_its_own_single_image(tmp_path):
    config, target = _slot_config(tmp_path)
    adapter = H3DatasetAdapter(config, _args())
    item = _slot_item(target)
    item.loss_mask_content = np.full((64, 64), 255, dtype=np.uint8)
    adapter.attach(item)
    video_encoder = _FakeVideoEncoder()
    encoder = _NativeLatentEncoder(video_encoder, _FakeAudioEncoder(), torch.float32)

    (tensors,) = encoder.encode_latents([item])

    assert len(video_encoder.image_calls) == 3
    latents = tensors["latents_3x4x4_float32"]
    assert latents.shape == (24, 3, 4, 4)
    assert [float(latents[0, slot, 0, 0]) for slot in range(3)] == [1.0, 2.0, 3.0]
    assert tensors["latents_cond_000_1x4x4_float32"].shape == (24, 1, 4, 4)
    assert tensors["video_loss_mask"].shape == (3, 4, 4)
    assert tensors["latents_audio_2x32x2_float32"].shape == (2, 32, 2)


# --- training forward and reference route ----------------------------------


def _model():
    return SimpleNamespace(config=SimpleNamespace(in_channels=24, audio_in_channels=6, text_dim=8, patch_size=(1, 2, 2)))


def _text_batch(task, tags, route=None):
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.zeros(len(tags), 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.tensor(tags, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS[task])],
    }
    if route is not None:
        batch[H3_REFERENCE_ROUTE_KEY] = [torch.tensor(H3_REFERENCE_ROUTE_IDS[route])]
    return batch


def _fl2va_slot_batch(route=None, tags=(1, 0, 1)):
    batch = _text_batch("fl2va", list(tags), route)
    batch.update(
        {
            H3_ONE_FRAME_TARGET_INDICES_KEY: [torch.tensor([-1, 0, 1], dtype=torch.long)],
            H3_ONE_FRAME_CONTROL_INDICES_KEY: [torch.tensor([4], dtype=torch.long)],
            "latents_cond_000": [torch.ones(24, 1, 2, 2)],
        }
    )
    return batch


def _plan(backend, batch, frames=3):
    video = torch.zeros(1, 24, frames, 2, 2)
    audio = torch.zeros(1, 2, 6, 2)
    torch.manual_seed(0)
    return backend._prepare_training_forward(_model(), batch, video, audio, torch.tensor([0.4]), torch.tensor([0.7]))


def test_training_forward_packs_slots_and_signed_controls_at_their_times():
    plan = _plan(_NativeTrainingBackend(SimpleNamespace(), mode="fl2va"), _fl2va_slot_batch())
    times = plan.kwargs["position_ids"][:, 0]
    assert plan.kwargs["video_hidden_states"].shape[1] == 4  # one control row + three slots
    assert float(times[3]) == pytest.approx(3 + FRAME * 5)  # control at 4, shift 1
    torch.testing.assert_close(times[-3:], torch.tensor([3.0, 3 + FRAME, 3 + 2 * FRAME], dtype=torch.float64))


def test_default_route_is_dual_and_leaves_the_forward_unchanged():
    backend = _NativeTrainingBackend(SimpleNamespace(), mode="fl2va")
    assert backend.reference_route == "dual"
    implicit = _plan(backend, _fl2va_slot_batch())
    backend.reference_route = "dual"
    explicit = _plan(backend, _fl2va_slot_batch())
    for name, value in implicit.kwargs.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, explicit.kwargs[name]), name


@pytest.mark.parametrize("route", ["qwen_image_only", "text_only"])
def test_routes_without_dit_rows_drop_exactly_the_control_rows(route):
    dual = _plan(_NativeTrainingBackend(SimpleNamespace(), mode="fl2va"), _fl2va_slot_batch())
    backend = _NativeTrainingBackend(SimpleNamespace(), mode="fl2va")
    backend.reference_route = route
    tags = (1, 0, 1) if route == "qwen_image_only" else (1, 1, 1)
    routed = _plan(backend, _fl2va_slot_batch(route, tags))
    assert dual.kwargs["video_hidden_states"].shape[1] - routed.kwargs["video_hidden_states"].shape[1] == 1
    torch.testing.assert_close(routed.kwargs["video_hidden_states"], dual.kwargs["video_hidden_states"][:, 1:])
    assert torch.equal(routed.kwargs["position_ids"][-3:], dual.kwargs["position_ids"][-3:])
    assert torch.equal(routed.kwargs["position_ids"][:3], dual.kwargs["position_ids"][:3])


def test_dit_latent_only_route_keeps_control_rows_without_qwen_vision_rows():
    backend = _NativeTrainingBackend(SimpleNamespace(), mode="fl2va")
    backend.reference_route = "dit_latent_only"
    plan = _plan(backend, _fl2va_slot_batch("dit_latent_only", (1, 1, 1)))
    assert plan.kwargs["video_hidden_states"].shape[1] == 4


def test_training_route_must_match_the_text_cache():
    backend = _NativeTrainingBackend(SimpleNamespace(), mode="fl2va")
    backend.reference_route = "text_only"
    with pytest.raises(ValueError, match="written with --h3_reference_route dual"):
        _plan(backend, _fl2va_slot_batch())
    dual = _NativeTrainingBackend(SimpleNamespace(), mode="fl2va")
    with pytest.raises(ValueError, match="written with --h3_reference_route text_only"):
        _plan(dual, _fl2va_slot_batch("text_only", (1, 1, 1)))


def _ref2va_slot_batch(route=None, tags=(1, 0, 1)):
    batch = _text_batch("ref2va", list(tags), route)
    batch.update(
        {
            H3_ONE_FRAME_TARGET_INDICES_KEY: [torch.tensor([-2, 2], dtype=torch.long)],
            H3_REFERENCE_KINDS_KEY: [torch.tensor([int(H3ReferenceKind.IMAGE)])],
            H3_REFERENCE_VIDEO_SHAPES_KEY: [torch.tensor([[1, 2, 2]])],
            H3_REFERENCE_AUDIO_LENGTHS_KEY: [torch.tensor([0])],
            H3_REFERENCE_VIDEO_ROWS_KEY: [torch.ones(1, 96)],
            H3_REFERENCE_AUDIO_ROWS_KEY: [torch.empty(0, 6)],
        }
    )
    return batch


def test_ref2va_route_drops_exactly_the_dit_reference_rows():
    dual = _plan(_NativeTrainingBackend(SimpleNamespace(), mode="ref2va"), _ref2va_slot_batch(), frames=2)
    assert dual.kwargs["video_hidden_states"].shape[1] == 3
    torch.testing.assert_close(dual.kwargs["position_ids"][-2:, 0], torch.tensor([4.0, 4.0 + FRAME * 4], dtype=torch.float64))
    backend = _NativeTrainingBackend(SimpleNamespace(), mode="ref2va")
    backend.reference_route = "qwen_image_only"
    routed = _plan(backend, _ref2va_slot_batch("qwen_image_only"), frames=2)
    assert routed.kwargs["video_hidden_states"].shape[1] == 2
    # The dropped reference keeps its rotary span, so the slots do not move.
    torch.testing.assert_close(routed.kwargs["position_ids"][-2:], dual.kwargs["position_ids"][-2:])
    backend.reference_route = "text_only"
    text_only = _plan(backend, _ref2va_slot_batch("text_only", (1, 1, 1)), frames=2)
    assert text_only.kwargs["video_hidden_states"].shape[1] == 2
    torch.testing.assert_close(text_only.kwargs["position_ids"][-2:], dual.kwargs["position_ids"][-2:])


@pytest.mark.parametrize("route", ["dual", "qwen_image_only", "dit_latent_only", "text_only"])
def test_target_times_are_identical_under_every_route(route):
    """A route only decides which control rows are emitted; placement uses every configured index."""

    def batch(route_name):
        tags = (1, 0, 1) if route_name in ("dual", "qwen_image_only") else (1, 1, 1)
        value = _text_batch("fl2va", list(tags), None if route_name == "dual" else route_name)
        value.update(
            {
                H3_ONE_FRAME_TARGET_INDICES_KEY: [torch.tensor([2, 5], dtype=torch.long)],
                H3_ONE_FRAME_CONTROL_INDICES_KEY: [torch.tensor([-4], dtype=torch.long)],
                "latents_cond_000": [torch.ones(24, 1, 2, 2)],
            }
        )
        return value

    dual = _plan(_NativeTrainingBackend(SimpleNamespace(), mode="fl2va"), batch("dual"), frames=2)
    backend = _NativeTrainingBackend(SimpleNamespace(), mode="fl2va")
    backend.reference_route = route
    routed = _plan(backend, batch(route), frames=2)
    # Shift 4 from the control at -4: targets at 3 + 5/3 * 6 and 3 + 5/3 * 9.
    expected = torch.tensor([3 + FRAME * 6, 3 + FRAME * 9], dtype=torch.float64)
    torch.testing.assert_close(dual.kwargs["position_ids"][-2:, 0], expected)
    torch.testing.assert_close(routed.kwargs["position_ids"][-2:], dual.kwargs["position_ids"][-2:])
    rows = routed.kwargs["video_hidden_states"].shape[1]
    assert rows == (3 if route in ("dual", "dit_latent_only") else 2)


def test_placement_only_controls_emit_no_rows_but_fix_the_shift():
    tags = torch.tensor([1, 1, 1], dtype=torch.long)
    common = {"num_latent_frames": 2, "latent_height": 2, "latent_width": 2, "num_audio_latents": 2, "patch_size": (1, 2, 2)}
    emitted = build_t2va_packed_sequence(tags, one_frame_target_indices=(2, 5), one_frame_control_indices=(-4,), **common)
    placed = build_t2va_packed_sequence(tags, one_frame_target_indices=(2, 5), one_frame_placement_indices=(-4,), **common)
    assert placed.num_condition_video_rows == 0 and emitted.num_condition_video_rows == 1
    assert torch.equal(placed.position_ids[-2:], emitted.position_ids[-2:])
    with pytest.raises(ValueError, match="distinct"):
        build_t2va_packed_sequence(tags, one_frame_target_indices=(2, 5), one_frame_placement_indices=(5,), **common)


def test_denoise_keeps_slot_times_when_a_route_drops_the_controls(monkeypatch):
    layouts = []
    original = h3_inference.build_t2va_packed_sequence

    def spy(*args, **kwargs):
        layout = original(*args, **kwargs)
        layouts.append(layout)
        return layout

    monkeypatch.setattr(h3_inference, "build_t2va_packed_sequence", spy)
    conditioning = {H3_TEXT_HIDDEN_KEY: torch.randn(3, 8), H3_TEXT_TOKEN_TAGS_KEY: torch.ones(3, dtype=torch.long)}
    common = {
        "height": 32,
        "width": 32,
        "frame_count": 1,
        "num_inference_steps": 2,
        "device": torch.device("cpu"),
        "show_progress": False,
        "one_frame_target_indices": (2, 5),
    }
    transformer = _tiny_transformer()
    h3_inference.denoise_fl2va(
        transformer,
        conditioning,
        generator=torch.Generator().manual_seed(0),
        keyframe_rows=torch.zeros(1, 96),
        one_frame_control_indices=(-4,),
        **common,
    )
    h3_inference.denoise_fl2va(
        transformer, conditioning, generator=torch.Generator().manual_seed(0), one_frame_placement_indices=(-4,), **common
    )
    assert torch.equal(layouts[0].position_ids[-6:], layouts[1].position_ids[-6:])
    assert layouts[1].num_condition_video_rows == 0


def test_native_backend_runs_a_slot_forward_and_backward():
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=1,
        attention_head_dim=8,
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=16,
        in_channels=24,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=8,
        time_embed_dim=8,
        rope_freq_dim=1,
    )
    transformer = MiniMaxH3Transformer(config)
    backend = _NativeTrainingBackend(transformer, mode="fl2va")
    batch = _fl2va_slot_batch()
    batch[H3_TEXT_HIDDEN_KEY] = [torch.randn(3, 8)]
    video = torch.randn(1, 24, 3, 2, 2)
    audio = torch.randn(1, 2, 6, 2)
    prediction = backend.predict_training(transformer, batch, video, audio, torch.tensor([0.4]), torch.tensor([0.7]))
    assert prediction.video.shape == video.shape
    prediction.video.square().mean().backward()
    assert transformer.blocks[0].attn.qkv_proj.weight.grad is not None


# --- trainer helpers and flags ---------------------------------------------


def test_slot_batches_require_one_frame_and_one_latent_per_slot():
    batch = {H3_ONE_FRAME_TARGET_INDICES_KEY: [torch.tensor([-1, 0, 1])]}
    assert _one_frame_target_slot_count(batch) == 3
    assert _one_frame_target_slot_count({}) == 0
    with pytest.raises(ValueError, match="requires --one_frame"):
        _require_one_frame_opt_in(Namespace(one_frame=False), batch, torch.zeros(1, 24, 3, 2, 2))
    with pytest.raises(ValueError, match="one target latent frame per slot"):
        _require_one_frame_opt_in(Namespace(one_frame=True), batch, torch.zeros(1, 24, 2, 2, 2))
    _require_one_frame_opt_in(Namespace(one_frame=True), batch, torch.zeros(1, 24, 3, 2, 2))


def test_per_slot_losses_match_a_manual_masked_mean():
    torch.manual_seed(1)
    prediction = torch.randn(1, 4, 3, 2, 2)
    target = torch.randn(1, 4, 3, 2, 2)
    mask = torch.zeros(1, 3, 2, 2)
    mask[:, :, 0, 0] = 1.0
    mask[:, 2] = 1.0
    losses = target_slot_losses(prediction, target, mask)
    for slot in range(3):
        weights = mask[:, slot][:, None].expand(1, 4, 2, 2)
        expected = ((prediction[:, :, slot] - target[:, :, slot]).square() * weights).sum() / weights.sum()
        torch.testing.assert_close(losses[slot], expected)
    metrics = _target_slot_metrics(prediction, target, None, 3, "weighted")
    assert sorted(metrics) == ["loss/target_slot0", "loss/target_slot1", "loss/target_slot2"]
    assert _target_slot_metrics(prediction[:, :, :1], target[:, :, :1], None, 1, "weighted") == {}


def test_training_flags_default_to_the_released_behaviour():
    args = create_training_parser().parse_args([])
    assert args.h3_target_noise_coupling == "independent"
    assert args.h3_reference_route == "dual"


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["--sdpa", "--h3_target_noise_coupling", "shared"], "requires --one_frame"),
        (["--sdpa", "--h3_reference_route", "text_only"], "requires --one_frame or"),
    ],
)
def test_training_flag_validation(argv, match):
    args = create_training_parser().parse_args(argv)
    with pytest.raises(ValueError, match=match):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_route_rejects_training_time_sampling():
    args = create_training_parser().parse_args(["--sdpa", "--one_frame", "--h3_reference_route", "qwen_image_only"])
    args.sample_prompts = "samples.txt"
    with pytest.raises(ValueError, match="training-time sampling"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_slot_flags_pass_validation_with_one_frame():
    args = create_training_parser().parse_args(
        ["--sdpa", "--one_frame", "--h3_target_noise_coupling", "shared", "--h3_reference_route", "dit_latent_only"]
    )
    MiniMaxH3NetworkTrainer().handle_model_specific_args(args)


def test_shared_coupling_is_rejected_without_a_slot_layout():
    args = Namespace(h3_target_noise_coupling="shared")
    with pytest.raises(ValueError, match="no training dataset declares fp_1f_target_indices"):
        _require_slot_layouts_for_coupling(args, ())
    _require_slot_layouts_for_coupling(args, ((0, 1),))
    _require_slot_layouts_for_coupling(Namespace(h3_target_noise_coupling="independent"), ())


def test_every_slot_noise_site_applies_the_coupling():
    batch = {H3_ONE_FRAME_TARGET_INDICES_KEY: [torch.tensor([0, 1, 2])]}
    noise = torch.randn(1, 4, 3, 2, 2)
    shared = MiniMaxH3NetworkTrainer._slot_coupled_noise(Namespace(h3_target_noise_coupling="shared"), batch, noise)
    assert torch.equal(shared[:, :, 2], noise[:, :, 0])
    assert MiniMaxH3NetworkTrainer._slot_coupled_noise(Namespace(h3_target_noise_coupling="shared"), {}, noise) is noise
    # Rollout probes and rollout supervision draw their own noise; both go through the coupling.
    for method in (MiniMaxH3NetworkTrainer._probe_rollout_field, MiniMaxH3NetworkTrainer._rollout_supervision):
        assert "self._slot_coupled_noise(args, batch, self._rollout_noise(video_latents))" in inspect.getsource(method)


def test_rollout_supervision_starts_slot_batches_from_coupled_noise(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    trainer._rollout_noise_generator = torch.Generator().manual_seed(0)
    video = torch.zeros(1, 4, 3, 2, 2)
    batch = {H3_ONE_FRAME_TARGET_INDICES_KEY: [torch.tensor([0, 1, 2])]}
    args = Namespace(h3_target_noise_coupling="shared")
    state = trainer._slot_coupled_noise(args, batch, trainer._rollout_noise(video))
    independent = torch.Generator().manual_seed(0)
    raw = torch.randn(video.shape, generator=independent)
    # The full draw is taken first, so the stream advances exactly as without coupling.
    assert torch.equal(state[:, :, 0], raw[:, :, 0]) and torch.equal(state[:, :, 1], raw[:, :, 0])
    assert torch.equal(trainer._rollout_noise_generator.get_state(), independent.get_state())


def test_slot_metadata_records_layouts_coupling_and_route():
    trainer = MiniMaxH3NetworkTrainer()
    args = Namespace(h3_target_noise_coupling="shared", h3_reference_route="qwen_image_only")
    trainer._target_slot_layouts = ((-1, 0, 1), (0, 24))
    assert trainer._target_slot_metadata(args) == {
        "ss_h3_target_slots": "2,3",
        "ss_h3_target_indices": "-1;0;1,0;24",
        "ss_h3_target_noise_coupling": "shared",
        "ss_h3_reference_route": "qwen_image_only",
    }
    trainer._target_slot_layouts = ()
    assert trainer._target_slot_metadata(Namespace(h3_target_noise_coupling="independent", h3_reference_route="dual")) == {}


# --- text conditioning route ------------------------------------------------


def _route_encoder(task, route):
    encoder = object.__new__(MiniMaxH3ConditioningEncoder)
    encoder.task = task
    encoder.reference_route = route
    encoder.keyframe_visuals = ()
    encoder.max_caption_tokens = 0
    encoder.text_visual_max_pixels = 0
    encoder.reference_image_short_edge = 64
    encoder.reference_image_size_mode = "short_edge"
    encoder.reference_image_max_pixels = 0
    encoder.reference_video_short_edge = 768
    encoder.reference_video_max_pixels = 768 * 1344
    encoder.reference_video_fps = 0.0
    encoder.output_dtype = torch.float32
    calls = []

    def encode_prompt(prompt, images=None, references=None, null_instruction=False, qwen_controls=None):
        calls.append((images, references))
        return torch.zeros(2, 8), torch.ones(2, dtype=torch.long)

    encoder._encode_prompt = encode_prompt
    return encoder, calls


def _one_frame_control_item(tmp_path):
    control = tmp_path / "control.png"
    Image.new("RGB", (32, 32), color="red").save(control)
    return SimpleNamespace(
        item_key="sample",
        caption="caption",
        h3_one_frame=True,
        h3_condition_paths=(control,),
        h3_media_assets=(),
        bucket_size=(32, 32),
    )


@pytest.mark.parametrize(("route", "presents"), [("dual", True), ("qwen_image_only", True), ("dit_latent_only", False)])
def test_text_cache_route_decides_the_qwen_presentation_of_timed_controls(tmp_path, route, presents):
    encoder, calls = _route_encoder("fl2va", route)
    (tensors,) = encoder.encode_conditioning([_one_frame_control_item(tmp_path)])
    images, _ = calls[0]
    assert (images is not None and len(images) == 1) is presents
    if route == "dual":
        assert H3_REFERENCE_ROUTE_KEY not in tensors
    else:
        assert int(tensors[H3_REFERENCE_ROUTE_KEY]) == H3_REFERENCE_ROUTE_IDS[route]


@pytest.mark.parametrize(("route", "presented"), [("qwen_image_only", 1), ("text_only", 0)])
def test_text_cache_route_decides_the_qwen_presentation_of_image_references(tmp_path, route, presented):
    reference = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), color="green").save(reference)
    item = SimpleNamespace(
        item_key="sample",
        caption="caption",
        h3_media_assets=(MediaAsset(reference, MediaModality.IMAGE, "reference"),),
        frame_count=1,
        bucket_size=(32, 32),
    )
    encoder, calls = _route_encoder("ref2va", route)
    (tensors,) = encoder.encode_conditioning([item])
    _, references = calls[0]
    assert len(references) == presented
    assert int(tensors[H3_REFERENCE_ROUTE_KEY]) == H3_REFERENCE_ROUTE_IDS[route]


def test_text_cache_route_needs_timed_controls_under_fl2va(tmp_path):
    encoder, _ = _route_encoder("fl2va", "text_only")
    item = _one_frame_control_item(tmp_path)
    item.h3_condition_paths = ()
    item.content = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    item.h3_one_frame = False
    with pytest.raises(ValueError, match="requires one-frame timed controls"):
        encoder.encode_conditioning([item])


# --- inference --------------------------------------------------------------


def test_generation_cli_parses_slots_coupling_and_route(tmp_path):
    args = create_generate_parser().parse_args(
        [
            "--model",
            "model",
            "--prompt",
            "three layers",
            "--output",
            str(tmp_path / "layers.png"),
            "--frame_count",
            "1",
            "--condition_image",
            "control.png",
            "--one_frame",
            "target_indices=-1;0;1,control_index=5",
            "--h3_target_noise_coupling",
            "shared",
            "--h3_reference_route",
            "dit_latent_only",
        ]
    )
    request = request_from_args(args)
    assert request.one_frame_target_index is None
    assert request.one_frame_target_indices == (-1, 0, 1)
    assert request.one_frame_control_indices == (5,)
    assert request.target_noise_coupling == "shared"
    assert request.reference_route == "dit_latent_only"
    assert [path.name for path in expected_outputs(request)] == [
        "layers_000_index-1.png",
        "layers_001_index0.png",
        "layers_002_index1.png",
    ]


def test_generation_defaults_keep_the_single_target_request():
    args = create_generate_parser().parse_args(
        ["--model", "model", "--prompt", "p", "--output", "out.png", "--frame_count", "1", "--one_frame", "target_index=24"]
    )
    request = request_from_args(args)
    assert request.one_frame_target_index == 24
    assert request.one_frame_target_indices is None
    assert (request.target_noise_coupling, request.reference_route) == ("independent", "dual")
    assert expected_outputs(request) == (Path("out.png"),)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"one_frame_target_indices": (0, 1), "output": Path("out.mp4")}, "image extension"),
        ({"target_noise_coupling": "shared", "one_frame_target_index": 0}, "target slots only"),
        ({"one_frame_target_indices": (0, 1), "one_frame_target_index": 0}, "not both"),
        ({"one_frame_target_indices": (0, 1), "condition_images": (Path("c.png"),), "one_frame_control_indices": (1,)}, "distinct"),
        ({"one_frame_target_indices": (0, 1), "reference_route": "text_only"}, "reference route"),
    ],
)
def test_generation_request_validation(kwargs, match):
    values = {"prompt": "p", "output": Path("out.png"), "frame_count_override": 1, **kwargs}
    with pytest.raises(ValueError, match=match):
        H3GenerationRequest(**values)


def test_generation_request_accepts_route_with_image_references():
    reference = H3Reference(Path("ref.png"), ReferenceKind.IMAGE)
    request = H3GenerationRequest(
        "p",
        Path("out.png"),
        references=(reference,),
        frame_count_override=1,
        one_frame_target_indices=(-1, 1),
        reference_route="qwen_image_only",
    )
    assert request.mode == "reference"
    assert target_slot_output_paths(request.output, (-1, 1))[1].name == "out_001_index1.png"


def _tiny_transformer():
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=1,
        attention_head_dim=8,
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=16,
        in_channels=24,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=8,
        time_embed_dim=8,
        rope_freq_dim=1,
    )
    return MiniMaxH3Transformer(config).eval()


@pytest.mark.parametrize("coupling", ["independent", "shared"])
def test_denoise_generates_one_latent_per_slot_with_the_requested_noise(monkeypatch, coupling):
    drawn = []
    original = h3_inference.couple_target_slot_noise

    def spy(noise, value):
        coupled = original(noise, value)
        drawn.append(coupled.clone())
        return coupled

    monkeypatch.setattr(h3_inference, "couple_target_slot_noise", spy)
    conditioning = {H3_TEXT_HIDDEN_KEY: torch.randn(3, 8), H3_TEXT_TOKEN_TAGS_KEY: torch.ones(3, dtype=torch.long)}
    video, audio = h3_inference.denoise_fl2va(
        _tiny_transformer(),
        conditioning,
        height=32,
        width=32,
        frame_count=1,
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
        show_progress=False,
        one_frame_target_indices=(-1, 0, 1),
        target_noise_coupling=coupling,
    )
    assert video.shape == (1, 24, 3, 2, 2)
    assert audio.shape[-1] == 2
    initial = drawn[0]
    assert torch.equal(initial[:, :, 0], initial[:, :, 1]) is (coupling == "shared")


def test_slot_outputs_are_decoded_one_slot_at_a_time(monkeypatch, tmp_path):
    decoded_frames = []

    class FakeDecoder:
        def to(self, device):
            return self

        def eval(self):
            return self

        def decode(self, latents):
            decoded_frames.append(latents.shape[2])
            return torch.full((1, 3, 1, 8, 8), float(latents[0, 0, 0, 0, 0]))

    monkeypatch.setattr(h3_integration, "load_video_vae_decoder", lambda path, device: FakeDecoder())
    generator = object.__new__(_NativeGenerator)
    generator.video_vae = Path("vae.safetensors")
    generator.device = torch.device("cpu")
    request = H3GenerationRequest("p", tmp_path / "slots.png", frame_count_override=1, one_frame_target_indices=(-1, 0, 1))
    latents = torch.arange(3.0).view(1, 1, 3, 1, 1).expand(1, 24, 3, 2, 2) / 4

    generator._save_target_slots(request, latents)

    assert decoded_frames == [1, 1, 1]
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == ["slots_000_index-1.png", "slots_001_index0.png", "slots_002_index1.png"]
    assert not (tmp_path / "slots.png").exists()


def test_lora_metadata_mismatch_warns(tmp_path):
    path = tmp_path / "adapter.safetensors"
    save_file(
        {"weight": torch.zeros(1)},
        str(path),
        metadata={
            "ss_h3_target_indices": "-1;0;1",
            "ss_h3_target_noise_coupling": "shared",
            "ss_h3_reference_route": "qwen_image_only",
        },
    )
    matching = H3GenerationRequest(
        "p",
        Path("out.png"),
        frame_count_override=1,
        one_frame_target_indices=(-1, 0, 1),
        target_noise_coupling="shared",
        reference_route="qwen_image_only",
        references=(H3Reference(Path("r.png"), ReferenceKind.IMAGE),),
    )
    assert h3_integration.warn_target_slot_metadata(path, matching) == []
    mismatched = H3GenerationRequest("p", Path("out.png"), frame_count_override=1, one_frame_target_indices=(0, 2))
    warnings = h3_integration.warn_target_slot_metadata(path, mismatched)
    # Route, slot indices, and coupling; no ss_h3_target_slots key in this file.
    assert len(warnings) == 3


def _metadata_file(path, metadata):
    save_file({"weight": torch.zeros(1)}, str(path), metadata=metadata)
    return path


def test_slot_lora_used_without_slots_warns(tmp_path):
    path = _metadata_file(
        tmp_path / "slots.safetensors",
        {"ss_h3_target_slots": "3", "ss_h3_target_indices": "-1;0;1", "ss_h3_target_noise_coupling": "shared"},
    )
    scalar = H3GenerationRequest("p", Path("out.png"), frame_count_override=1, one_frame_target_index=0)
    warnings = h3_integration.warn_target_slot_metadata(path, scalar)
    assert any("target slots -1;0;1, inference uses no target slots" in message for message in warnings)
    assert any("target noise coupling shared" in message for message in warnings)


def test_no_slot_lora_used_with_slots_warns(tmp_path):
    path = _metadata_file(tmp_path / "plain.safetensors", {"ss_h3_training_mode": "fl2va"})
    slots = H3GenerationRequest(
        "p", Path("out.png"), frame_count_override=1, one_frame_target_indices=(0, 1), target_noise_coupling="shared"
    )
    warnings = h3_integration.warn_target_slot_metadata(path, slots)
    assert any("no target slots, inference uses 0;1" in message for message in warnings)
    assert any("target noise coupling independent, inference uses shared" in message for message in warnings)
    plain = H3GenerationRequest("p", Path("out.png"), frame_count_override=1, one_frame_target_index=0)
    assert h3_integration.warn_target_slot_metadata(path, plain) == []


def test_slot_count_mismatch_warns_from_the_slot_count_key(tmp_path):
    path = _metadata_file(tmp_path / "count.safetensors", {"ss_h3_target_slots": "2"})
    request = H3GenerationRequest("p", Path("out.png"), frame_count_override=1, one_frame_target_indices=(0, 1, 2))
    warnings = h3_integration.warn_target_slot_metadata(path, request)
    assert warnings == ["2 target slot(s), inference uses 3"]


# --- batched micro-batch per-slot metrics ------------------------------------


class _Accelerator:
    device = torch.device("cpu")

    @staticmethod
    def autocast():
        from contextlib import nullcontext

        return nullcontext()

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def backward(loss):
        loss.backward()


def test_batched_microbatch_slot_metrics_use_the_item_loss_mask():
    torch.manual_seed(3)
    config = MiniMaxH3TransformerConfig(
        num_attention_heads=1,
        attention_head_dim=8,
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=16,
        in_channels=4,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=8,
        time_embed_dim=8,
        rope_freq_dim=1,
    )
    transformer = MiniMaxH3Transformer(config)
    video = torch.randn(2, 4, 3, 2, 2)
    mask = torch.ones(2, 3, 2, 2)
    mask[:, 1] = 0.0  # slot 1 carries no loss at all
    task_id = H3_CONDITIONING_TASK_IDS["t2va"]
    batch = {
        "latents": video,
        "latents_audio": torch.zeros(2, 2, 6, 2),
        "audio_loss_mask": torch.zeros(2, 2, dtype=torch.bool),
        "video_loss_mask": mask,
        "timesteps": torch.tensor([500.0, 500.0]),
        H3_ONE_FRAME_TARGET_INDICES_KEY: torch.tensor([[-1, 0, 1], [-1, 0, 1]]),
        H3_TEXT_HIDDEN_KEY: [torch.randn(3, 8), torch.randn(3, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(3, dtype=torch.long), torch.ones(3, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(task_id), torch.tensor(task_id)],
    }
    args = create_training_parser().parse_args(["--sdpa", "--one_frame", "--h3_batched_microbatch"])
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    trainer.dit_dtype = torch.float32
    trainer.backend = _NativeTrainingBackend(transformer)
    loss, metrics = trainer.process_batch(
        args,
        _Accelerator(),
        transformer,
        SimpleNamespace(set_enabled=lambda enabled: None),
        batch,
        video,
        torch.randn_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    assert metrics.get("h3/batched_microbatch") == 1.0
    assert float(metrics["loss/target_slot1"]) == 0.0
    assert float(metrics["loss/target_slot0"]) > 0.0 and float(metrics["loss/target_slot2"]) > 0.0
    assert torch.isfinite(loss)
