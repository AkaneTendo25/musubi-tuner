from argparse import Namespace
import copy

import numpy as np
from PIL import Image
import pytest
from safetensors import safe_open
import torch

from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.cache import save_latent_cache_minimax_h3
from musubi_tuner.minimax_h3.dataset import H3DatasetAdapter, create_h3_dataset_group
from musubi_tuner.minimax_h3.image_training import (
    H3_ONE_FRAME_CACHE_FORMAT,
    H3_ONE_FRAME_CONTROL_INDICES_KEY,
    H3_ONE_FRAME_CONTENT_FINGERPRINT_KEY,
    H3_ONE_FRAME_LATENT_FINGERPRINT_KEY,
    H3_ONE_FRAME_TARGET_INDEX_KEY,
)


def _config(tmp_path, *, target_index=24, control_indices=(0, 48)):
    targets = tmp_path / "targets"
    controls = tmp_path / "controls"
    targets.mkdir(parents=True)
    controls.mkdir(parents=True)
    Image.new("RGB", (80, 64), color="white").save(targets / "sample.png")
    sizes = ((32, 96), (96, 32), (48, 80), (80, 48))
    colors = ("red", "blue", "green", "yellow")
    for index in range(len(control_indices)):
        Image.new("RGB", sizes[index], color=colors[index]).save(controls / f"sample_{index}.png")
    return {
        "general": {"resolution": [64, 64], "batch_size": 1},
        "datasets": [
            {
                "target_image_directory": str(targets),
                "target_modalities": ["image"],
                "source_image_directory": str(controls),
                "source_modalities": ["image"],
                "cache_directory": str(tmp_path / "cache"),
                "fp_1f_clean_indices": list(control_indices),
                "fp_1f_target_index": target_index,
            }
        ],
    }, targets / "sample.png"


def test_timed_one_frame_controls_attach_without_resampling_target(tmp_path):
    config, target = _config(tmp_path)
    adapter = H3DatasetAdapter(config, Namespace(one_frame=True, task="fl2va", h3_image_mode="none"))
    content = np.zeros((64, 80, 3), dtype=np.uint8)
    item = ItemInfo(str(target), "caption", (80, 64), (64, 64), content=content)

    assets = adapter.attach(item)

    assert item.content is content
    assert item.frame_count == 1
    assert item.h3_one_frame is True
    assert item.h3_one_frame_target_index == 24
    assert item.h3_one_frame_control_indices == (0, 48)
    assert len(item.h3_condition_paths) == 2
    assert [control.shape for control in item.control_content] == [(64, 64, 3), (64, 64, 3)]
    assert len(assets) == 1  # timed controls are condition slots, not Ref2VA references
    assert adapter.attach(item) is assets

    item.latent_cache_path = str(tmp_path / "one_frame_mmh3.safetensors")
    save_latent_cache_minimax_h3(
        item,
        {
            "latents_1x4x4_float32": torch.zeros(24, 1, 4, 4),
            "latents_audio_2x32x2_float32": torch.zeros(2, 32, 2),
            "audio_loss_mask": torch.zeros(2, dtype=torch.bool),
            "latents_cond_000_1x4x4_float32": torch.zeros(24, 1, 4, 4),
            "latents_cond_001_1x4x4_float32": torch.zeros(24, 1, 4, 4),
        },
    )
    with safe_open(item.latent_cache_path, framework="pt", device="cpu") as handle:
        assert (handle.metadata() or {})["h3_cache_format"] == H3_ONE_FRAME_CACHE_FORMAT
        assert int(handle.get_tensor(f"{H3_ONE_FRAME_TARGET_INDEX_KEY}_int64")) == 24
        assert tuple(handle.get_tensor(f"{H3_ONE_FRAME_CONTROL_INDICES_KEY}_int64").tolist()) == (0, 48)


def test_one_frame_time_changes_only_latent_fingerprint(tmp_path):
    config, target = _config(tmp_path)
    first = H3DatasetAdapter(config, Namespace(one_frame=True, task="fl2va", h3_image_mode="none"))
    item = ItemInfo(str(target), "caption", (80, 64), (64, 64), content=np.zeros((64, 80, 3), dtype=np.uint8))
    first.attach(item)

    changed = dict(config)
    changed["datasets"] = [dict(config["datasets"][0], fp_1f_target_index=12)]
    second = H3DatasetAdapter(changed, Namespace(one_frame=True, task="fl2va", h3_image_mode="none"))
    other = ItemInfo(str(target), "caption", (80, 64), (64, 64), content=np.zeros((64, 80, 3), dtype=np.uint8))
    second.attach(other)

    assert (
        item.h3_cache_metadata[H3_ONE_FRAME_CONTENT_FINGERPRINT_KEY]
        == other.h3_cache_metadata[H3_ONE_FRAME_CONTENT_FINGERPRINT_KEY]
    )
    assert (
        item.h3_cache_metadata[H3_ONE_FRAME_LATENT_FINGERPRINT_KEY] != other.h3_cache_metadata[H3_ONE_FRAME_LATENT_FINGERPRINT_KEY]
    )


def test_three_timed_controls_persist_in_order(tmp_path):
    config, target = _config(tmp_path, control_indices=(0, 48, 96))
    adapter = H3DatasetAdapter(config, Namespace(one_frame=True, task="fl2va", h3_image_mode="none"))
    item = ItemInfo(str(target), "caption", (80, 64), (64, 64), content=np.zeros((64, 80, 3), dtype=np.uint8))
    adapter.attach(item)
    item.latent_cache_path = str(tmp_path / "three_controls_mmh3.safetensors")
    tensors = {
        "latents_1x4x4_float32": torch.zeros(24, 1, 4, 4),
        "latents_audio_2x32x2_float32": torch.zeros(2, 32, 2),
        "audio_loss_mask": torch.zeros(2, dtype=torch.bool),
    }
    tensors.update({f"latents_cond_{index:03d}_1x4x4_float32": torch.zeros(24, 1, 4, 4) for index in range(3)})
    save_latent_cache_minimax_h3(item, tensors)

    with safe_open(item.latent_cache_path, framework="pt", device="cpu") as handle:
        assert tuple(handle.get_tensor(f"{H3_ONE_FRAME_CONTROL_INDICES_KEY}_int64").tolist()) == (0, 48, 96)
        assert [key for key in handle.keys() if key.startswith("latents_cond_")] == [
            "latents_cond_000_1x4x4_float32",
            "latents_cond_001_1x4x4_float32",
            "latents_cond_002_1x4x4_float32",
        ]


def test_timed_controls_require_explicit_target_and_matching_count(tmp_path):
    config, _ = _config(tmp_path)
    del config["datasets"][0]["fp_1f_target_index"]
    with pytest.raises(ValueError, match="explicit fp_1f_target_index"):
        H3DatasetAdapter(config, Namespace(one_frame=True, task="fl2va", h3_image_mode="none"))

    config, _ = _config(tmp_path / "mismatch", control_indices=(0,))
    Image.new("RGB", (80, 48), color="green").save(tmp_path / "mismatch" / "controls" / "sample_1.png")
    with pytest.raises(ValueError, match="has 2 controls"):
        H3DatasetAdapter(config, Namespace(one_frame=True, task="fl2va", h3_image_mode="none"))


def test_untimed_image_controls_remain_ref2va_references(tmp_path):
    config, target = _config(tmp_path)
    del config["datasets"][0]["fp_1f_clean_indices"]
    del config["datasets"][0]["fp_1f_target_index"]
    adapter = H3DatasetAdapter(config, Namespace(one_frame=True, task="ref2va", h3_image_mode="none"))
    item = ItemInfo(str(target), "caption", (80, 64), (64, 64), content=np.zeros((64, 80, 3), dtype=np.uint8))

    assets = adapter.attach(item)

    assert len(assets) == 3
    assert item.h3_condition_paths == ()
    assert item.h3_one_frame_control_indices == ()


def test_documented_one_frame_image_aliases_pass_real_dataset_group(tmp_path):
    targets = tmp_path / "targets"
    controls = tmp_path / "controls"
    targets.mkdir()
    controls.mkdir()
    Image.new("RGB", (64, 64)).save(targets / "sample.png")
    Image.new("RGB", (96, 48)).save(controls / "sample.png")
    (targets / "sample.txt").write_text("an edited frame", encoding="utf-8")
    config = {
        "general": {"resolution": [64, 64], "batch_size": 1, "enable_bucket": True, "caption_extension": ".txt"},
        "datasets": [
            {
                "image_directory": str(targets),
                "control_directory": str(controls),
                "cache_directory": str(tmp_path / "cache"),
                "fp_1f_clean_indices": [0],
                "fp_1f_target_index": 24,
            }
        ],
    }
    args = Namespace(one_frame=True, task="fl2va", h3_image_mode="none", debug_dataset=False)

    group, adapter = create_h3_dataset_group(config, args)
    _, batch = next(group.datasets[0].retrieve_latent_cache_batches(1))
    assets = adapter.attach(batch[0])

    assert batch[0].h3_one_frame_control_indices == (0,)
    assert batch[0].control_content[0].shape == (64, 64, 3)
    assert len(assets) == 1


def test_one_frame_aliases_reject_mixed_explicit_schema(tmp_path):
    config, _ = _config(tmp_path)
    config["datasets"][0]["image_directory"] = config["datasets"][0]["target_image_directory"]
    with pytest.raises(ValueError, match="cannot mix one-frame image_directory"):
        H3DatasetAdapter(config, Namespace(one_frame=True, task="fl2va", h3_image_mode="none"))


def test_repeated_target_uses_cache_directory_to_select_time(tmp_path):
    config, target = _config(tmp_path)
    second = copy.deepcopy(config["datasets"][0])
    second["cache_directory"] = str(tmp_path / "cache_other")
    second["fp_1f_target_index"] = 12
    config["datasets"].append(second)
    adapter = H3DatasetAdapter(config, Namespace(one_frame=True, task="fl2va", h3_image_mode="none"))

    items = []
    for cache_name in ("cache", "cache_other"):
        item = ItemInfo(str(target), "caption", (80, 64), (64, 64), content=np.zeros((64, 80, 3), dtype=np.uint8))
        item.latent_cache_path = str(tmp_path / cache_name / "sample_mmh3.safetensors")
        adapter.attach(item)
        items.append(item)
    assert [item.h3_one_frame_target_index for item in items] == [24, 12]

    conflicting = copy.deepcopy(config)
    conflicting["datasets"][1]["cache_directory"] = str(tmp_path / "cache")
    with pytest.raises(ValueError, match="share cache_directory"):
        H3DatasetAdapter(conflicting, Namespace(one_frame=True, task="fl2va", h3_image_mode="none"))
