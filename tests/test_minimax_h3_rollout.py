"""Truncated on-policy rollout supervision (D-OPSD) for MiniMax H3."""

import json
import logging
from contextlib import nullcontext

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.architecture import AUDIO_FLOW_SHIFT, VIDEO_FLOW_SHIFT
from musubi_tuner.minimax_h3.cache import (
    H3_AUDIO_LATENTS_KEY,
    H3_AUDIO_LOSS_MASK_KEY,
    H3_EMPTY_TEXT_HIDDEN_KEY,
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY,
    H3_CONDITIONING_TASK_IDS,
    H3_CONDITIONING_TASK_KEY,
    H3_QWEN_CONTROL_VISUALS_KEY,
    H3_REFERENCE_AUDIO_LENGTHS_KEY,
    H3_REFERENCE_AUDIO_ROWS_KEY,
    H3_REFERENCE_KINDS_KEY,
    H3_REFERENCE_VIDEO_ROWS_KEY,
    H3_REFERENCE_VIDEO_SHAPES_KEY,
    H3_TEXT_HIDDEN_KEY,
    H3_TEXT_TOKEN_TAGS_KEY,
    save_text_encoder_output_cache_minimax_h3,
)
from musubi_tuner.minimax_h3.rollout import (
    H3_ROLLOUT_ITEM_KEYS_BATCH_KEY,
    MAX_ROLLOUT_WINDOW,
    H3RolloutTeacherCache,
    batch_item_key,
    enable_item_keys,
    euler_advance,
    item_key_latent_caches,
    item_key_stem,
    item_key_text_caches,
    reference_bundle_size,
    rollout_base_sigmas,
    rollout_item_key,
    teacher_batch,
)
from musubi_tuner.minimax_h3.training import H3ModelPrediction, shift_sigma
from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser
from musubi_tuner.modules.custom_offloading_utils import FusedArmBackwardGate
from musubi_tuner.training.trainer_base import LOSS_FOR_AVERAGE_KEY


# ---------------------------------------------------------------------------
# Schedule and Euler convention
# ---------------------------------------------------------------------------


def test_rollout_schedule_descends_from_noise_to_the_drawn_stop_and_continues_the_window():
    sigmas = rollout_base_sigmas(0.6, 4, 2)

    # Four steps to reach the stop, then one more state in the window.
    assert len(sigmas) == 6
    assert sigmas[0] == 1.0
    assert sigmas[4] == pytest.approx(0.6)
    step = (1.0 - 0.6) / 4
    assert sigmas[1] == pytest.approx(1.0 - step)
    # The window continues on the SAME step size the rollout used, so a sub-step
    # is the step the sampler would have taken next rather than a second schedule.
    assert sigmas[5] == pytest.approx(0.6 - step)
    assert all(later <= earlier for earlier, later in zip(sigmas, sigmas[1:]))


def test_rollout_schedule_floors_a_window_that_would_reach_a_clean_latent():
    sigmas = rollout_base_sigmas(0.02, 1, MAX_ROLLOUT_WINDOW)

    assert sigmas[1] == pytest.approx(0.02)
    # A single step from 1.0 leaves a stride far larger than the stop itself; the
    # window clamps instead of walking the state past sigma 0, where the flow
    # field is undefined for the next advance.
    assert all(value > 0 for value in sigmas)
    assert sigmas[-1] == pytest.approx(1e-3)


@pytest.mark.parametrize(
    ("stop", "steps", "window", "message"),
    [
        (0.6, 0, 2, "at least one Euler step"),
        (0.6, 4, 0, r"\[1, 3\]"),
        (0.6, 4, 4, r"\[1, 3\]"),
        (1.0, 4, 2, "must lie in"),
        (0.0, 4, 2, "must lie in"),
    ],
)
def test_rollout_schedule_rejects_incoherent_requests(stop, steps, window, message):
    with pytest.raises(ValueError, match=message):
        rollout_base_sigmas(stop, steps, window)


def test_euler_advance_recovers_x0_exactly_at_sigma_zero():
    """The step must be the same convention the trainer states: x0 = x_t + s*v."""
    x0 = torch.randn(1, 4, 2, 2, 2)
    noise = torch.randn_like(x0)
    sigma = 0.7
    state = (1.0 - sigma) * x0 + sigma * noise
    velocity = x0 - noise

    torch.testing.assert_close(euler_advance(state, velocity, sigma, 0.0), x0)
    # And a partial step lands exactly on the true state at the lower sigma.
    torch.testing.assert_close(
        euler_advance(state, velocity, sigma, 0.3),
        (1.0 - 0.3) * x0 + 0.3 * noise,
    )


def test_euler_advance_refuses_to_run_the_schedule_backwards():
    state = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="descend"):
        euler_advance(state, torch.zeros(1, 2), 0.3, 0.7)
    with pytest.raises(ValueError, match="must match"):
        euler_advance(state, torch.zeros(1, 3), 0.7, 0.3)


# ---------------------------------------------------------------------------
# Teacher pairing
# ---------------------------------------------------------------------------


def _write_teacher_cache(directory, item_key: str, *, controls: bool, fill: float = 0.5, control_payload=None, task: str = "t2va"):
    path = directory / f"{item_key}_mmh3_te.safetensors"
    tensors = {
        f"varlen_{H3_TEXT_HIDDEN_KEY}_float32": torch.full((4, 8), fill),
        f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64": torch.ones(4, dtype=torch.long),
        H3_CONDITIONING_TASK_KEY: torch.tensor(H3_CONDITIONING_TASK_IDS[task]),
    }
    if controls:
        # The production marker: how many visuals the conditioner was shown.
        tensors[H3_QWEN_CONTROL_VISUALS_KEY] = torch.tensor(2, dtype=torch.long) if control_payload is None else control_payload
    save_file(tensors, str(path))
    return path


class _StubDataset:
    def __init__(self, items):
        self.batch_manager = type("_Manager", (), {"buckets": {(64, 64): list(items)}})()


class _StubGroup:
    def __init__(self, datasets):
        self.datasets = list(datasets)


def _write_reference_latents(directory, item_key: str, *, references: int, suffix: str = "_se384", fill: float = 1.0):
    """A latent cache holding ``references`` Ref2VA reference entries.

    Only the bundle matters here: the rollout reads the kinds vector to count the
    references and the packed rows to substitute them, and never looks at the
    target latents beside them.
    """
    path = directory / f"{item_key}_mmh3.safetensors"
    tensors = {"latents_2x2x2_float32": torch.zeros(24, 2, 2, 2)}
    if references:
        tensors[f"varlen_{H3_REFERENCE_KINDS_KEY}{suffix}_int64"] = torch.zeros(references, dtype=torch.long)
        tensors[f"varlen_{H3_REFERENCE_VIDEO_SHAPES_KEY}{suffix}_int64"] = torch.ones(references, 3, dtype=torch.long)
        tensors[f"varlen_{H3_REFERENCE_AUDIO_LENGTHS_KEY}{suffix}_int64"] = torch.zeros(references, dtype=torch.long)
        tensors[f"varlen_{H3_REFERENCE_VIDEO_ROWS_KEY}{suffix}_float32"] = torch.full((references, 4), fill)
        tensors[f"varlen_{H3_REFERENCE_AUDIO_ROWS_KEY}{suffix}_float32"] = torch.zeros(0, 4)
    save_file(tensors, str(path))
    return path


def _stub_item(key: str, path, latent_path=None, window: str | None = None) -> ItemInfo:
    item = ItemInfo(key, "", (64, 64), (64, 64))
    item.text_encoder_output_cache_path = str(path)
    if latent_path is not None:
        item.latent_cache_path = str(latent_path)
    item.cache_window = window
    return item


def test_teacher_map_keys_by_item_and_collapses_num_repeats(tmp_path):
    first = _write_teacher_cache(tmp_path, "clip_a", controls=True)
    second = _write_teacher_cache(tmp_path, "clip_b", controls=True, fill=0.25)
    item_a = _stub_item("clip_a", first)
    # num_repeats puts the very same ItemInfo in a bucket several times.
    group = _StubGroup([_StubDataset([item_a, item_a, _stub_item("clip_b", second)])])

    paths = item_key_text_caches(group)

    assert set(paths) == {"clip_a", "clip_b"}
    cache = H3RolloutTeacherCache(paths)
    assert len(cache) == 2
    assert cache.validate_privilege(student_text_paths=_student_text_cache(tmp_path, "clip_a", "clip_b")) == ("qwen", 2)
    entries = cache.entries("clip_b")
    torch.testing.assert_close(entries[H3_TEXT_HIDDEN_KEY], torch.full((4, 8), 0.25))
    assert int(entries[H3_QWEN_CONTROL_VISUALS_KEY]) == 2


def test_teacher_map_names_the_items_it_cannot_pair(tmp_path):
    cache = H3RolloutTeacherCache({"clip_a": str(_write_teacher_cache(tmp_path, "clip_a", controls=True))})

    cache.require(["clip_a"])
    with pytest.raises(ValueError, match="clip_missing"):
        cache.require(["clip_a", "clip_missing"])
    with pytest.raises(KeyError, match="clip_missing"):
        cache.entries("clip_missing")


def test_item_key_stem_strips_only_the_window_token():
    assert item_key_stem("5s_46_00000-073") == "5s_46"
    assert item_key_stem("5s_46_00000-073-01") == "5s_46"
    # A clip that never carried a window keeps its whole name, and a name that
    # merely looks numeric is not mistaken for a window.
    assert item_key_stem("5s_46") == "5s_46"
    assert item_key_stem("clip_2024-01") == "clip_2024-01"


def test_several_windows_of_one_clip_are_kept_apart(tmp_path):
    # target_frames = [22, 73] cuts two windows out of one clip; both keep the
    # clip's item_key and only the window token tells their caches apart.
    short = _write_teacher_cache(tmp_path, "clip_a_00000-022", controls=True, fill=0.5)
    long = _write_teacher_cache(tmp_path, "clip_a_00000-073", controls=True, fill=0.25)
    short_latents = _write_reference_latents(tmp_path, "clip_a_00000-022", references=2)
    long_latents = _write_reference_latents(tmp_path, "clip_a_00000-073", references=2)
    group = _StubGroup(
        [
            _StubDataset(
                [
                    _stub_item("clip_a", short, short_latents, window="00000-022"),
                    _stub_item("clip_a", long, long_latents, window="00000-073"),
                ]
            )
        ]
    )

    paths = item_key_text_caches(group)

    assert set(paths) == {"clip_a_00000-022", "clip_a_00000-073"}
    assert set(item_key_latent_caches(group)) == set(paths)
    cache = H3RolloutTeacherCache(paths, latent_paths=item_key_latent_caches(group))
    cache.require(paths)
    # Each window reads its own cache: pairing by clip would have served one of
    # these tensors for both.
    torch.testing.assert_close(cache.entries("clip_a_00000-022")[H3_TEXT_HIDDEN_KEY], torch.full((4, 8), 0.5))
    torch.testing.assert_close(cache.entries("clip_a_00000-073")[H3_TEXT_HIDDEN_KEY], torch.full((4, 8), 0.25))


def test_a_teacher_cached_at_one_window_serves_the_clips_other_windows(tmp_path, caplog):
    text = _write_teacher_cache(tmp_path, "clip_a_00000-022", controls=False)
    latents = _write_reference_latents(tmp_path, "clip_a_00000-022", references=3, fill=0.75)
    student_dir = tmp_path / "student"
    student_dir.mkdir()
    student_latents = _write_reference_latents(student_dir, "clip_a_00000-073", references=1)
    cache = H3RolloutTeacherCache({"clip_a_00000-022": str(text)}, latent_paths={"clip_a_00000-022": str(latents)})

    with caplog.at_level(logging.WARNING, logger="musubi_tuner.minimax_h3.rollout"):
        cache.require(["clip_a_00000-022", "clip_a_00000-073"])

    assert "clip_a_00000-073" in caplog.text
    # One line per clip, not one per window and certainly not one per step.
    assert caplog.text.count("using") == 1

    assert cache.validate_privilege({"clip_a_00000-073": str(student_latents)}) == ("reference", 1)
    torch.testing.assert_close(
        cache.entries("clip_a_00000-073")[H3_TEXT_HIDDEN_KEY],
        cache.entries("clip_a_00000-022")[H3_TEXT_HIDDEN_KEY],
    )
    assert cache.reference_entries("clip_a_00000-073")[f"{H3_REFERENCE_KINDS_KEY}_se384"].numel() == 3

    # A clip that is absent altogether is still a hard failure: the fallback is
    # across windows of one clip, never across clips.
    with pytest.raises(ValueError, match="clip_b_00000-073"):
        cache.require(["clip_b_00000-073"])


def test_single_window_corpora_keep_their_pre_window_keys(tmp_path):
    # Regression: an item without a window (an image, or a teacher map built
    # before windows were distinguished) keys by the bare clip name, exactly as
    # before, and a windowed student still pairs with it.
    text = _write_teacher_cache(tmp_path, "clip_a", controls=True)
    group = _StubGroup([_StubDataset([_stub_item("clip_a", text)])])

    assert set(item_key_text_caches(group)) == {"clip_a"}
    assert rollout_item_key(_stub_item("clip_a", text)) == "clip_a"

    legacy = H3RolloutTeacherCache({"clip_a": str(text)})
    legacy.require(["clip_a", "clip_a_00000-073"])
    torch.testing.assert_close(legacy.entries("clip_a_00000-073")[H3_TEXT_HIDDEN_KEY], torch.full((4, 8), 0.5))


def test_teacher_without_any_privilege_is_rejected_before_the_model_loads(tmp_path):
    _write_teacher_cache(tmp_path, "clip_a", controls=True)
    plain = _write_teacher_cache(tmp_path, "clip_b", controls=False)
    cache = H3RolloutTeacherCache({"clip_a": str(tmp_path / "clip_a_mmh3_te.safetensors"), "clip_b": str(plain)})

    with pytest.raises(ValueError, match="no privilege over the student") as failure:
        cache.validate_privilege()

    # Both escape routes are named, because auto measured both channels and
    # reporting only the one it fell back to would hide half the diagnosis.
    assert "qwen_control_*" in str(failure.value)
    assert "--ref2va_variant_b" in str(failure.value)
    assert cache.channel is None


def test_teacher_cache_is_bounded_and_reused(tmp_path):
    paths = {f"clip_{index}": str(_write_teacher_cache(tmp_path, f"clip_{index}", controls=True)) for index in range(4)}
    cache = H3RolloutTeacherCache(paths, cache_size=2)

    for key in paths:
        cache.entries(key)

    assert len(cache._cache) == 2
    assert cache.entries("clip_3") is cache.entries("clip_3")


def test_teacher_batch_replaces_only_the_text_side(tmp_path):
    batch = {
        "latents": torch.zeros(1, 4, 2, 2, 2),
        H3_TEXT_HIDDEN_KEY: [torch.zeros(4, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(4, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: torch.tensor([2]),
    }
    entries = {
        H3_TEXT_HIDDEN_KEY: torch.full((6, 8), 0.5),
        H3_TEXT_TOKEN_TAGS_KEY: torch.ones(6, dtype=torch.long),
        H3_QWEN_CONTROL_VISUALS_KEY: torch.tensor(2, dtype=torch.long),
    }

    teacher = teacher_batch(batch, entries)

    assert teacher["latents"] is batch["latents"]
    assert teacher[H3_CONDITIONING_TASK_KEY] is batch[H3_CONDITIONING_TASK_KEY]
    assert teacher[H3_TEXT_HIDDEN_KEY][0].shape == (6, 8)
    assert int(teacher[H3_QWEN_CONTROL_VISUALS_KEY][0]) == 2
    # The student's own batch is untouched, so a step that also runs a student
    # forward is not conditioned on the teacher by accident.
    assert batch[H3_TEXT_HIDDEN_KEY][0].shape == (4, 8)


# ---------------------------------------------------------------------------
# The reference channel (Ref2VA variant B)
# ---------------------------------------------------------------------------


def _variant_b_cache(tmp_path, *, teacher_references=3, student_references=1, controls=False):
    """A teacher/student pair whose privilege lives in the latent cache."""
    teacher_dir = tmp_path / "teacher"
    student_dir = tmp_path / "student"
    teacher_dir.mkdir(exist_ok=True)
    student_dir.mkdir(exist_ok=True)
    text = _write_teacher_cache(teacher_dir, "clip_a", controls=controls)
    teacher_latents = _write_reference_latents(teacher_dir, "clip_a", references=teacher_references, fill=0.75)
    student_latents = _write_reference_latents(student_dir, "clip_a", references=student_references, fill=0.25)
    cache = H3RolloutTeacherCache({"clip_a": str(text)}, latent_paths={"clip_a": str(teacher_latents)})
    return cache, {"clip_a": str(student_latents)}, teacher_latents, student_latents


def test_reference_bundle_size_counts_the_kinds_vector(tmp_path):
    assert reference_bundle_size(None) == 0
    # A T2VA cache legitimately holds no bundle, and that is a count of zero
    # rather than an error: the question is only whether the teacher has more.
    assert reference_bundle_size(str(_write_reference_latents(tmp_path, "plain", references=0))) == 0
    assert reference_bundle_size(str(_write_reference_latents(tmp_path, "three", references=3))) == 3


def test_latent_caches_are_mapped_by_the_same_item_key_as_the_text_ones(tmp_path):
    latents = _write_reference_latents(tmp_path, "clip_a", references=2)
    text = _write_teacher_cache(tmp_path, "clip_a", controls=False)
    group = _StubGroup([_StubDataset([_stub_item("clip_a", text, latents)])])

    assert set(item_key_latent_caches(group)) == set(item_key_text_caches(group)) == {"clip_a"}
    assert item_key_latent_caches(group)["clip_a"].endswith("clip_a_mmh3.safetensors")


def test_latent_privilege_passes_validation_without_any_qwen_control_visuals(tmp_path):
    cache, student_latents, _, _ = _variant_b_cache(tmp_path)

    assert cache.validate_privilege(student_latents) == ("reference", 1)
    assert cache.channel == "reference"


def test_a_teacher_that_does_not_extend_the_student_is_not_privileged(tmp_path):
    # Same reference count in both arms and no control visuals: the two forwards
    # would be conditioned identically, which is self-distillation.
    cache, student_latents, _, _ = _variant_b_cache(tmp_path, teacher_references=2, student_references=2)

    with pytest.raises(ValueError, match="no privilege over the student"):
        cache.validate_privilege(student_latents)


def test_auto_prefers_the_measured_qwen_channel_and_a_pin_overrides_it(tmp_path):
    cache, student_latents, _, _ = _variant_b_cache(tmp_path, controls=True)

    student_text = _student_text_cache(tmp_path, "clip_a")
    assert cache.validate_privilege(student_latents, student_text_paths=student_text)[0] == "qwen"
    assert cache.validate_privilege(student_latents, channel="reference", student_text_paths=student_text)[0] == "reference"
    assert cache.reference_entries("clip_a")


@pytest.mark.parametrize(
    ("payload", "why"),
    [
        (torch.zeros(0, dtype=torch.long), "empty"),
        (torch.tensor(0, dtype=torch.long), "zero"),
    ],
)
def test_control_visuals_that_carry_nothing_are_refused(tmp_path, payload, why):
    """Declared is not informative, and only the first is checkable by key name.

    The channel resolves on key names, so a corpus whose control visuals are
    present but empty or flat would otherwise resolve to qwen, cache, and train
    to completion against a teacher no better informed than the student. The
    symptom would be a training curve that went nowhere, read as a weak method.
    """
    teacher_dir = tmp_path / "teacher"
    student_dir = tmp_path / "student"
    teacher_dir.mkdir()
    student_dir.mkdir()
    _write_teacher_cache(teacher_dir, "clip_a", controls=True, control_payload=payload)
    _write_reference_latents(teacher_dir, "clip_a", references=1, fill=0.75)
    student_latents = _write_reference_latents(student_dir, "clip_a", references=1, fill=0.25)
    cache = H3RolloutTeacherCache({"clip_a": str(teacher_dir / "clip_a_mmh3_te.safetensors")})

    with pytest.raises(ValueError, match="carry no information") as failure:
        cache.validate_privilege({"clip_a": str(student_latents)}, student_text_paths=_student_text_cache(tmp_path, "clip_a"))
    assert why in str(failure.value)


def test_a_pinned_channel_names_the_channel_that_would_have_worked(tmp_path):
    cache, student_latents, _, _ = _variant_b_cache(tmp_path, teacher_references=1, student_references=1, controls=True)

    with pytest.raises(ValueError, match="unprivileged teacher item") as failure:
        cache.validate_privilege(student_latents, channel="reference")

    assert "qwen_control_*" in str(failure.value)


def test_reference_entries_are_empty_until_the_reference_channel_is_resolved(tmp_path):
    cache, student_latents, _, _ = _variant_b_cache(tmp_path, controls=True)

    # Unvalidated, and on the Qwen channel, the teacher's references are the
    # student's: substituting them would be a no-op that can only disagree about
    # the sizing suffix.
    assert cache.reference_entries("clip_a") == {}
    cache.validate_privilege(student_latents, student_text_paths=_student_text_cache(tmp_path, "clip_a"))
    assert cache.reference_entries("clip_a") == {}


def test_reference_entries_read_the_teachers_own_bundle(tmp_path):
    cache, student_latents, _, _ = _variant_b_cache(tmp_path)
    cache.validate_privilege(student_latents)

    entries = cache.reference_entries("clip_a")

    # Suffixed keys survive intact: the sizing suffix is part of the cache
    # identity the conditioning reader looks the bundle up by.
    assert set(entries) == {
        f"{H3_REFERENCE_KINDS_KEY}_se384",
        f"{H3_REFERENCE_VIDEO_SHAPES_KEY}_se384",
        f"{H3_REFERENCE_AUDIO_LENGTHS_KEY}_se384",
        f"{H3_REFERENCE_VIDEO_ROWS_KEY}_se384",
        f"{H3_REFERENCE_AUDIO_ROWS_KEY}_se384",
    }
    assert entries[f"{H3_REFERENCE_KINDS_KEY}_se384"].numel() == 3
    assert cache.reference_entries("clip_a") is entries


def test_teacher_batch_takes_the_teachers_references_and_the_students_target_latents():
    kinds_key = f"{H3_REFERENCE_KINDS_KEY}_se384"
    rows_key = f"{H3_REFERENCE_VIDEO_ROWS_KEY}_se384"
    latents = torch.zeros(1, 4, 2, 2, 2)
    batch = {
        "latents": latents,
        H3_TEXT_HIDDEN_KEY: [torch.zeros(4, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(4, dtype=torch.long)],
        kinds_key: [torch.zeros(1, dtype=torch.long)],
        rows_key: [torch.full((1, 4), 0.25)],
    }
    entries = {
        H3_TEXT_HIDDEN_KEY: torch.full((4, 8), 0.5),
        H3_TEXT_TOKEN_TAGS_KEY: torch.ones(4, dtype=torch.long),
    }
    references = {
        kinds_key: torch.zeros(3, dtype=torch.long),
        rows_key: torch.full((3, 4), 0.75),
    }

    teacher = teacher_batch(batch, entries, references)

    # The rollout state is shared -- the two arms must be compared at one state.
    assert teacher["latents"] is latents
    assert teacher[kinds_key][0].numel() == 3
    torch.testing.assert_close(teacher[rows_key][0], torch.full((3, 4), 0.75))
    # No control marker leaks in from the student on this channel.
    assert H3_QWEN_CONTROL_VISUALS_KEY not in teacher
    assert batch[kinds_key][0].numel() == 1


def test_teacher_batch_replaces_the_bundle_wholesale_across_sizing_suffixes():
    stale_key = f"{H3_REFERENCE_VIDEO_ROWS_KEY}_se512"
    fresh_key = f"{H3_REFERENCE_VIDEO_ROWS_KEY}_se384"
    batch = {
        H3_TEXT_HIDDEN_KEY: [torch.zeros(4, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(4, dtype=torch.long)],
        stale_key: [torch.zeros(1, 4)],
    }
    entries = {H3_TEXT_HIDDEN_KEY: torch.zeros(4, 8), H3_TEXT_TOKEN_TAGS_KEY: torch.ones(4, dtype=torch.long)}

    teacher = teacher_batch(batch, entries, {fresh_key: torch.ones(2, 4)})

    # A student row set left behind under another suffix would be a second,
    # inconsistent bundle the conditioning reader could still find.
    assert stale_key not in teacher
    assert teacher[fresh_key][0].shape == (2, 4)


def test_variant_a_still_substitutes_only_the_text_side(tmp_path):
    first = _write_teacher_cache(tmp_path, "clip_a", controls=True)
    latents = _write_reference_latents(tmp_path, "clip_a", references=2)
    cache = H3RolloutTeacherCache({"clip_a": str(first)}, latent_paths={"clip_a": str(latents)})
    cache.validate_privilege({"clip_a": str(latents)}, student_text_paths=_student_text_cache(tmp_path, "clip_a"))

    kinds_key = f"{H3_REFERENCE_KINDS_KEY}_se384"
    batch = {kinds_key: [torch.zeros(2, dtype=torch.long)], H3_TEXT_HIDDEN_KEY: [torch.zeros(4, 8)]}
    teacher = teacher_batch(batch, cache.entries("clip_a"), cache.reference_entries("clip_a"))

    assert cache.channel == "qwen"
    assert teacher[kinds_key] is batch[kinds_key]
    assert int(teacher[H3_QWEN_CONTROL_VISUALS_KEY][0]) == 2


def test_batch_manager_names_its_items_only_when_asked(tmp_path):
    latent_path = tmp_path / "clip_00000-002_64x64_mmh3.safetensors"
    text_path = tmp_path / "clip_00000-002_mmh3_te.safetensors"
    save_file({"latents_2x2x2_float32": torch.zeros(24, 2, 2, 2)}, str(latent_path))
    item = ItemInfo("clip", "", (64, 64), (64, 64), latent_cache_path=str(latent_path))
    item.text_encoder_output_cache_path = str(text_path)
    save_text_encoder_output_cache_minimax_h3(
        item,
        {
            f"varlen_{H3_TEXT_HIDDEN_KEY}_float32": torch.zeros(2, 5120),
            f"varlen_{H3_TEXT_TOKEN_TAGS_KEY}_int64": torch.ones(2, dtype=torch.long),
            H3_CONDITIONING_TASK_KEY: torch.tensor(H3_CONDITIONING_TASK_IDS["t2va"]),
        },
    )
    manager = BucketBatchManager({(64, 64): [item]}, batch_size=1)

    assert H3_ROLLOUT_ITEM_KEYS_BATCH_KEY not in manager[0]

    enable_item_keys(_StubGroup([type("_D", (), {"batch_manager": manager})()]))
    batch = manager[0]

    assert batch[H3_ROLLOUT_ITEM_KEYS_BATCH_KEY] == ["clip"]
    assert batch_item_key(batch) == "clip"

    # The window, once the dataset knows one, so the name the step looks a
    # teacher up by is the name the teacher map is keyed by.
    item.cache_window = "00000-002"
    assert batch_item_key(manager[0]) == "clip_00000-002"


def test_batch_item_key_requires_the_dataset_to_name_its_items():
    with pytest.raises(KeyError, match="name its items"):
        batch_item_key({})
    with pytest.raises(ValueError, match="one item per step"):
        batch_item_key({H3_ROLLOUT_ITEM_KEYS_BATCH_KEY: ["a", "b"]})


# ---------------------------------------------------------------------------
# The trainer step
# ---------------------------------------------------------------------------


class _FakeAccelerator:
    device = torch.device("cpu")

    @staticmethod
    def autocast():
        return nullcontext()

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def backward(loss):
        loss.backward()


class _ScaleTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))


class _ToggleNetwork:
    def __init__(self, transformer):
        self.transformer = transformer
        self.events = []

    def set_enabled(self, enabled):
        self.events.append(enabled)
        self.transformer.adapter_enabled = enabled

    def is_enabled(self):
        return getattr(self.transformer, "adapter_enabled", True)

    def named_parameters(self):
        return self.transformer.named_parameters()


class _RolloutBackend:
    """A field that depends on the text presented and on the adapter being on.

    Every recorded call carries enough to reconstruct the rollout: which text was
    presented, whether the adapter was live, whether a graph was being built, and
    the timestep the state sat at.
    """

    def __init__(self, velocity=0.25):
        self.calls = []
        self.velocity = velocity

    def predict_training(
        self,
        transformer,
        batch,
        video_hidden_states,
        audio_hidden_states,
        video_timestep,
        audio_timestep,
        *,
        conditioning="prompt",
    ):
        del audio_timestep
        key = H3_EMPTY_TEXT_HIDDEN_KEY if conditioning == "empty" else H3_TEXT_HIDDEN_KEY
        signature = float(batch[key][0].float().mean())
        enabled = getattr(transformer, "adapter_enabled", True)
        self.calls.append(
            {
                "conditioning": conditioning,
                "grad": torch.is_grad_enabled(),
                "adapter": enabled,
                "signature": signature,
                "video_timestep": float(video_timestep.reshape(-1)[0]),
                "video_state": None if video_hidden_states is None else video_hidden_states.detach().clone(),
            }
        )
        scale = transformer.scale if enabled else transformer.scale.detach() * 0 + 1.0
        return H3ModelPrediction(
            None if video_hidden_states is None else torch.ones_like(video_hidden_states) * scale * self.velocity + signature,
            None if audio_hidden_states is None else torch.ones_like(audio_hidden_states) * scale * self.velocity + signature,
        )


_TEACHER_ENTRIES = {
    H3_TEXT_HIDDEN_KEY: torch.full((4, 8), 0.75),
    H3_TEXT_TOKEN_TAGS_KEY: torch.ones(4, dtype=torch.long),
    H3_QWEN_CONTROL_VISUALS_KEY: torch.arange(4, dtype=torch.float32),
}


# The teacher's OWN null twins. Chosen so the teacher's prompted-minus-empty
# field (0.5) differs from the student's (0.25): with the two equal the gap term
# would be zero whatever it supervised, which is the one setting that cannot tell
# a difference-matching objective from a branch-matching one.
_TEACHER_EMPTY_ENTRIES = {
    H3_EMPTY_TEXT_HIDDEN_KEY: torch.full((4, 8), 0.25),
    H3_EMPTY_TEXT_TOKEN_TAGS_KEY: torch.ones(4, dtype=torch.long),
}


class _StubTeacherCache:
    def __init__(self, entries=None, references=None, empty=None):
        self.entries_by_key = {"clip": dict(_TEACHER_ENTRIES) if entries is None else entries}
        self.references_by_key = {"clip": dict(references or {})}
        self.empty_by_key = {"clip": dict(_TEACHER_EMPTY_ENTRIES) if empty is None else empty}
        self.requested = []

    def entries(self, item_key):
        self.requested.append(item_key)
        return self.entries_by_key[item_key]

    def empty_entries(self, item_key):
        entries = self.empty_by_key[item_key]
        if not entries:
            raise ValueError("teacher text cache is missing mmh3_empty_hidden_states; re-cache with --cache_guidance_empty")
        return entries

    def reference_entries(self, item_key):
        return self.references_by_key[item_key]


def _rollout_batch():
    video = torch.zeros(1, 24, 2, 2, 2)
    batch = {
        H3_AUDIO_LATENTS_KEY: torch.zeros(1, 2, 32, 3),
        H3_AUDIO_LOSS_MASK_KEY: torch.ones(1, 3, dtype=torch.bool),
        "timesteps": [0.5],
        H3_TEXT_HIDDEN_KEY: [torch.full((4, 8), 0.25)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(4, dtype=torch.long)],
        H3_EMPTY_TEXT_HIDDEN_KEY: [torch.zeros(4, 8)],
        H3_EMPTY_TEXT_TOKEN_TAGS_KEY: [torch.ones(4, dtype=torch.long)],
        H3_ROLLOUT_ITEM_KEYS_BATCH_KEY: ["clip"],
    }
    return video, batch


_ROLLOUT_FLAGS = ("--h3_rollout_supervision", "--h3_rollout_teacher_config", "TEACHER")


def _flag_args(*extra, teacher=None):
    argv = ["--sdpa"]
    for value in extra:
        argv.append(str(teacher) if value == "TEACHER" else value)
    return create_parser().parse_args(argv)


def _run_step(*flags, teacher=None, active=True, guidance_active=True, seed=0, backend=None, batch=None, teacher_cache=None):
    args = _flag_args(*flags, teacher=teacher)
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    trainer.dit_dtype = torch.float32
    backend = _RolloutBackend() if backend is None else backend
    trainer.backend = backend
    trainer._rollout_teacher = _StubTeacherCache() if teacher_cache is None else teacher_cache
    trainer._rollout_supervision_active = lambda accelerator, probability: active
    trainer._guidance_distillation_active = lambda accelerator, probability: guidance_active
    transformer = _ScaleTransformer()
    network = _ToggleNetwork(transformer)
    video, step_batch = _rollout_batch() if batch is None else batch

    torch.manual_seed(seed)
    loss, metrics = trainer.process_batch(
        args,
        _FakeAccelerator(),
        transformer,
        network,
        step_batch,
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )
    return backend, loss, metrics, trainer


def test_rollout_flag_off_leaves_the_step_bit_identical(tmp_path):
    """The regression guard: the ordinary objective, forward count and metric set."""
    baseline_backend, baseline_loss, baseline_metrics, _ = _run_step()
    teacher = tmp_path / "teacher.toml"
    teacher.write_text("[general]\n", encoding="utf-8")
    inactive_backend, inactive_loss, inactive_metrics, _ = _run_step(*_ROLLOUT_FLAGS, teacher=teacher, active=False)

    assert float(baseline_loss.detach()) == float(inactive_loss.detach())
    assert len(baseline_backend.calls) == len(inactive_backend.calls) == 1
    assert not [key for key in baseline_metrics if "rollout" in key]
    # With the flag on but the step inactive, the loss is untouched and only the
    # reporting keys appear.
    assert inactive_metrics["h3/rollout_active"] == 0.0
    assert inactive_metrics["loss/rollout_video"] == 0.0
    assert LOSS_FOR_AVERAGE_KEY not in inactive_metrics
    assert {key for key in inactive_metrics if "rollout" not in key} == set(baseline_metrics)


def test_rollout_active_step_runs_j_no_grad_then_m_graded_pairs(tmp_path):
    teacher = tmp_path / "teacher.toml"
    teacher.write_text("[general]\n", encoding="utf-8")
    backend, loss, metrics, _ = _run_step(*_ROLLOUT_FLAGS, "--h3_rollout_steps", "3", "--h3_rollout_window", "2", teacher=teacher)
    loss.backward()

    # One data forward, three no-grad rollout steps, then two (student, teacher)
    # pairs: the documented j + 1 + 2m forwards per active step.
    assert len(backend.calls) == 1 + 3 + 4
    data, *rollout = backend.calls
    assert data["grad"] and data["adapter"]
    assert [call["grad"] for call in rollout] == [False, False, False, True, False, True, False]
    # The rollout policy is the current model with the adapter live; only the
    # teacher forwards disable it.
    assert [call["adapter"] for call in rollout] == [True, True, True, True, False, True, False]
    student_signature = float(torch.full((4, 8), 0.25).mean())
    teacher_signature = float(torch.full((4, 8), 0.75).mean())
    assert [call["signature"] for call in rollout[3:]] == [
        student_signature,
        teacher_signature,
        student_signature,
        teacher_signature,
    ]
    assert metrics["h3/rollout_active"] == 1.0
    assert metrics["loss/rollout_video"] > 0.0


def test_rollout_state_starts_at_noise_and_advances_by_the_velocity_convention(tmp_path):
    teacher = tmp_path / "teacher.toml"
    teacher.write_text("[general]\n", encoding="utf-8")
    backend, _, metrics, _ = _run_step(*_ROLLOUT_FLAGS, "--h3_rollout_steps", "2", "--h3_rollout_window", "1", teacher=teacher)

    rollout = backend.calls[1:]
    stop = metrics["h3/rollout_stop_sigma"]
    base = rollout_base_sigmas(stop, 2, 1)
    video_sigmas = shift_sigma(torch.tensor(base, dtype=torch.float32), VIDEO_FLOW_SHIFT)
    audio_sigmas = shift_sigma(torch.tensor(base, dtype=torch.float32), AUDIO_FLOW_SHIFT)
    # Video and audio ride different shifts off one unshifted coordinate, exactly
    # as a data step does, so the two schedules stay synchronized.
    assert not torch.allclose(video_sigmas, audio_sigmas)

    # The model is told the timestep of the shifted sigma at each state.
    assert [call["video_timestep"] for call in rollout[:3]] == pytest.approx(
        [float(1.0 - value) for value in video_sigmas[:3]], abs=1e-6
    )
    # The first state is pure noise: sigma 1 maps to 1 under every shift.
    assert float(video_sigmas[0]) == pytest.approx(1.0)
    # And each state is the previous one advanced by (sigma - next_sigma) * v.
    for index in range(2):
        expected = euler_advance(
            rollout[index]["video_state"],
            torch.full_like(rollout[index]["video_state"], 0.5 * 0.25 + float(torch.full((4, 8), 0.25).mean())),
            float(video_sigmas[index]),
            float(video_sigmas[index + 1]),
        )
        torch.testing.assert_close(rollout[index + 1]["video_state"], expected)


def test_audio_keeps_its_data_loss_on_an_active_step(tmp_path):
    teacher = tmp_path / "teacher.toml"
    teacher.write_text("[general]\n", encoding="utf-8")
    # Audio silenced by weight: only the rollout video term survives.
    _, video_only_loss, video_only_metrics, _ = _run_step(
        *_ROLLOUT_FLAGS, "--h3_audio_loss_weight", "0.0", "--h3_loss_balance", "modality", teacher=teacher
    )
    _, joint_loss, joint_metrics, _ = _run_step(*_ROLLOUT_FLAGS, "--h3_loss_balance", "modality", teacher=teacher)

    # The audio half is the ordinary data loss, unchanged by the rollout, and it
    # enters the joint loss beside the rollout video term.
    assert joint_metrics["loss/audio"] > 0.0
    assert video_only_metrics["loss/audio"] == 0.0
    assert float(joint_loss.detach()) == pytest.approx(
        0.5 * (video_only_metrics["loss/rollout_video"] + joint_metrics["loss/audio"]), rel=1e-6
    )
    assert joint_metrics["loss/rollout_video"] == pytest.approx(video_only_metrics["loss/rollout_video"], rel=1e-6)


def test_active_step_reports_the_ordinary_data_objective_as_the_average(tmp_path):
    teacher = tmp_path / "teacher.toml"
    teacher.write_text("[general]\n", encoding="utf-8")
    _, dense_loss, dense_metrics, _ = _run_step(*_ROLLOUT_FLAGS, teacher=teacher, active=False)
    _, active_loss, active_metrics, _ = _run_step(*_ROLLOUT_FLAGS, teacher=teacher, active=True)

    # The rollout swaps the objective rather than estimating one sparsely, so the
    # optimized loss differs while the reported average stays the loss a run
    # without the flag would have logged.
    assert float(active_loss.detach()) != pytest.approx(float(dense_loss.detach()))
    assert active_metrics[LOSS_FOR_AVERAGE_KEY] == pytest.approx(float(dense_loss.detach()), rel=1e-6)


def test_supervision_masks_the_video_loss_like_every_other_h3_objective(tmp_path):
    teacher = tmp_path / "teacher.toml"
    teacher.write_text("[general]\n", encoding="utf-8")
    video, batch = _rollout_batch()
    masked = dict(batch)
    masked["video_loss_mask"] = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    _, _, unmasked_metrics, _ = _run_step(*_ROLLOUT_FLAGS, teacher=teacher)
    _, _, masked_metrics, _ = _run_step(*_ROLLOUT_FLAGS, teacher=teacher, batch=(video, masked))

    assert unmasked_metrics["loss/rollout_video"] > 0.0
    # Every video element masked out leaves the rollout term with nothing to say.
    assert masked_metrics["loss/rollout_video"] == 0.0


def test_a_caption_dropout_step_keeps_the_data_objective(tmp_path):
    teacher = tmp_path / "teacher.toml"
    teacher.write_text("[general]\n", encoding="utf-8")
    backend, _, metrics, _ = _run_step(*_ROLLOUT_FLAGS, "--h3_caption_dropout_rate", "1.0", teacher=teacher, active=True)

    # The teacher's privilege lives in its prompted presentation; there is no
    # privileged null field, so a dropped step stands down.
    assert metrics["h3/rollout_active"] == 0.0
    assert len(backend.calls) == 1
    assert backend.calls[0]["conditioning"] == "empty"


# ---------------------------------------------------------------------------
# Hybrid rollout + guidance
# ---------------------------------------------------------------------------


_GUIDANCE_FLAGS = ("--h3_guidance_distillation_scale", "3.0")


def _recorded_audio_pair(monkeypatch, symbol, index, run):
    """Run one step with ``symbol`` spied on, returning its first call's audio pair.

    ``joint_velocity_loss(prediction, inputs)`` and ``joint_prediction_loss(
    prediction, reference)`` carry the same two positional slots, so the audio
    half of the objective is read out of both by the same helper.
    """
    import musubi_tuner.minimax_h3_train_network as module

    original = getattr(module, symbol)
    recorded = []

    def spy(prediction, reference, *args, **kwargs):
        # ``H3JointNoisyInputs`` also carries an ``audio`` field -- the noisy state,
        # not the target -- so the target slot is read first.
        target = reference.audio_target if hasattr(reference, "audio_target") else reference.audio
        recorded.append((prediction.audio, target))
        return original(prediction, reference, *args, **kwargs)

    monkeypatch.setattr(module, symbol, spy)
    run()
    monkeypatch.undo()
    return recorded[index]


@pytest.mark.parametrize("form", ["normalized", "contrastive"])
def test_hybrid_inactive_step_is_exactly_a_guidance_step(tmp_path, form):
    """The whole point of lifting the exclusion: 1 - p of the steps must run the
    guidance objective unchanged, schedules, cfg-zero and all."""
    teacher = tmp_path / "teacher.toml"
    teacher.write_text("[general]\n", encoding="utf-8")
    extra = (*_GUIDANCE_FLAGS, "--h3_guidance_loss_form", form, "--h3_guidance_cfg_zero", "--h3_guidance_loss_schedule", "sigma")
    guidance_backend, guidance_loss, guidance_metrics, _ = _run_step(*extra)
    hybrid_backend, hybrid_loss, hybrid_metrics, _ = _run_step(*_ROLLOUT_FLAGS, *extra, teacher=teacher, active=False)

    assert float(hybrid_loss.detach()) == float(guidance_loss.detach())
    # One empty forward plus one trainable forward, on both sides.
    assert [call["conditioning"] for call in hybrid_backend.calls] == [call["conditioning"] for call in guidance_backend.calls]
    assert hybrid_metrics["h3/rollout_active"] == 0.0
    assert {key for key in hybrid_metrics if "rollout" not in key} == set(guidance_metrics)


def test_hybrid_active_step_keeps_the_pure_rollout_video_term(tmp_path):
    """Guidance touches audio only on an active step: the video half is the
    teacher-MSE it is without the guidance flags, to the last bit."""
    teacher = tmp_path / "teacher.toml"
    teacher.write_text("[general]\n", encoding="utf-8")
    _, pure_loss, pure_metrics, _ = _run_step(
        *_ROLLOUT_FLAGS, "--h3_audio_loss_weight", "0.0", "--h3_loss_balance", "modality", teacher=teacher
    )
    _, hybrid_loss, hybrid_metrics, _ = _run_step(
        *_ROLLOUT_FLAGS,
        *_GUIDANCE_FLAGS,
        "--h3_audio_loss_weight",
        "0.0",
        "--h3_loss_balance",
        "modality",
        teacher=teacher,
    )

    assert hybrid_metrics["loss/rollout_video"] == pytest.approx(pure_metrics["loss/rollout_video"], rel=1e-6)
    # With audio silenced the two objectives coincide entirely.
    assert float(hybrid_loss.detach()) == pytest.approx(float(pure_loss.detach()), rel=1e-6)


@pytest.mark.parametrize("form", ["normalized", "contrastive"])
def test_hybrid_active_step_applies_guidance_to_the_audio_half(tmp_path, monkeypatch, form):
    """The audio half of an active step is the data-forward objective *including*
    its guidance modification: the same (prediction, target) pair a pure guidance
    step would have scored, whichever half of the pair that form rewrites."""
    teacher = tmp_path / "teacher.toml"
    teacher.write_text("[general]\n", encoding="utf-8")
    extra = (*_GUIDANCE_FLAGS, "--h3_guidance_loss_form", form)

    guided = _recorded_audio_pair(monkeypatch, "joint_velocity_loss", 0, lambda: _run_step(*extra))
    hybrid = _recorded_audio_pair(
        monkeypatch,
        "joint_prediction_loss",
        0,
        lambda: _run_step(*_ROLLOUT_FLAGS, *extra, teacher=teacher, active=True),
    )
    plain = _recorded_audio_pair(
        monkeypatch,
        "joint_prediction_loss",
        0,
        lambda: _run_step(*_ROLLOUT_FLAGS, teacher=teacher, active=True),
    )

    torch.testing.assert_close(hybrid[0], guided[0])
    torch.testing.assert_close(hybrid[1], guided[1])
    # And the guidance really did move something, so the assertion above is not
    # comparing two copies of the plain data objective.
    assert not torch.allclose(hybrid[0], plain[0]) or not torch.allclose(hybrid[1], plain[1])


def test_hybrid_active_step_does_not_rescale_by_the_guidance_probability(tmp_path):
    """A swapped objective is never divided by a probability -- and rescaling here
    would rebuild the loss from the plain velocity term and drop the rollout."""
    teacher = tmp_path / "teacher.toml"
    teacher.write_text("[general]\n", encoding="utf-8")
    _, dense_loss, dense_metrics, _ = _run_step(*_ROLLOUT_FLAGS, *_GUIDANCE_FLAGS, teacher=teacher, active=True)
    _, sparse_loss, sparse_metrics, _ = _run_step(
        *_ROLLOUT_FLAGS, *_GUIDANCE_FLAGS, "--h3_guidance_distillation_probability", "0.25", teacher=teacher, active=True
    )

    assert float(sparse_loss.detach()) == pytest.approx(float(dense_loss.detach()), rel=1e-6)
    assert sparse_metrics["h3/guidance_distillation_active"] == 1.0
    assert dense_metrics["loss/rollout_video"] == pytest.approx(sparse_metrics["loss/rollout_video"], rel=1e-6)


def test_guidance_alone_is_untouched_by_the_lifted_exclusion(tmp_path):
    """The regression guard for the other arm: with the rollout flag absent the
    sparse guidance estimator still rescales exactly as it did."""
    _, active_loss, _, _ = _run_step(*_GUIDANCE_FLAGS, "--h3_guidance_distillation_probability", "0.25")
    _, dense_loss, _, _ = _run_step(*_GUIDANCE_FLAGS)
    _, plain_loss, _, _ = _run_step()

    assert float(active_loss.detach()) == pytest.approx(
        float(plain_loss.detach()) + 4.0 * (float(dense_loss.detach()) - float(plain_loss.detach())), rel=1e-6
    )


class _SwapAwareScaleTransformer(_ScaleTransformer):
    def __init__(self):
        super().__init__()
        self.swap_mode = "training"
        self.swap_events = []

    def switch_block_swap_for_inference(self):
        self.swap_mode = "inference"
        self.swap_events.append("inference")

    def switch_block_swap_for_training(self):
        self.swap_mode = "training"
        self.swap_events.append("training")


class _SwapRecordingBackend(_RolloutBackend):
    def predict_training(self, transformer, *args, **kwargs):
        prediction = super().predict_training(transformer, *args, **kwargs)
        self.calls[-1]["swap_mode"] = getattr(transformer, "swap_mode", "training")
        return prediction


def test_rollout_brackets_every_no_grad_forward_on_the_forward_only_swap(tmp_path):
    teacher = _teacher_file(tmp_path)
    args = _flag_args(
        "--h3_rollout_supervision",
        "--h3_rollout_teacher_config",
        str(teacher),
        "--h3_rollout_steps",
        "2",
        "--h3_rollout_window",
        "2",
    )
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    trainer.dit_dtype = torch.float32
    trainer.blocks_to_swap = 4
    backend = _SwapRecordingBackend()
    trainer.backend = backend
    trainer._rollout_teacher = _StubTeacherCache()
    trainer._rollout_supervision_active = lambda accelerator, probability: True
    transformer = _SwapAwareScaleTransformer()
    video, batch = _rollout_batch()

    torch.manual_seed(0)
    trainer.process_batch(
        args,
        _FakeAccelerator(),
        transformer,
        _ToggleNetwork(transformer),
        batch,
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )

    modes = [call["swap_mode"] for call in backend.calls]
    grads = [call["grad"] for call in backend.calls]
    # Every no-grad forward runs on the cyclic forward-only schedule; every
    # graph-carrying one runs on the training layout that its backward restores.
    assert [mode == "inference" for mode in modes] == [not grad for grad in grads]
    # The contiguous rollout prefix shares one bracket rather than paying two
    # switches per step; each teacher forward gets its own.
    assert transformer.swap_events == ["inference", "training"] * 3
    assert transformer.swap_mode == "training"


def test_rollout_draws_come_off_dedicated_streams(tmp_path):
    trainer = MiniMaxH3NetworkTrainer()
    trainer._rollout_probability_generator = torch.Generator()
    trainer._rollout_probability_generator.manual_seed(11)
    trainer._rollout_noise_generator = torch.Generator()
    trainer._rollout_noise_generator.manual_seed(13)
    args = _flag_args()

    torch.manual_seed(0)
    expected_global = torch.rand(4)
    torch.manual_seed(0)
    decisions = [trainer._rollout_supervision_active(_FakeAccelerator(), 0.5) for _ in range(8)]
    sigmas = [trainer._draw_rollout_stop_sigma(args) for _ in range(8)]
    noise = trainer._rollout_noise(torch.zeros(1, 4, 2, 2, 2))
    after_global = torch.rand(4)

    assert set(decisions) == {True, False}
    assert all(0.0 < value < 1.0 for value in sigmas) and len(set(sigmas)) > 1
    assert noise.shape == (1, 4, 2, 2, 2) and float(noise.std()) > 0
    # Enabling the rollout must not shift any other branch decision in the run.
    torch.testing.assert_close(after_global, expected_global)


def test_rollout_generators_are_seeded_once_from_the_global_stream():
    trainer = MiniMaxH3NetworkTrainer()

    torch.manual_seed(5)
    first = trainer._draw_rollout_stop_sigma(_flag_args())
    generator = trainer._rollout_noise_generator
    second = trainer._draw_rollout_stop_sigma(_flag_args())

    assert generator is not None and trainer._rollout_noise_generator is generator
    assert first != second


def test_rollout_stop_sigma_is_stratified_when_bucketing_is_on():
    trainer = MiniMaxH3NetworkTrainer()
    args = _flag_args()
    args.num_timestep_buckets = 4

    torch.manual_seed(0)
    draws = [trainer._draw_rollout_stop_sigma(args) for _ in range(200)]

    assert all(0.0 < value < 1.0 for value in draws)
    # Stratification must still cover the whole schedule rather than one bucket.
    assert min(draws) < 0.25 and max(draws) > 0.75


def test_rollout_stop_shifted_covers_the_mid_video_band():
    """The point of the flag. A shift of 12 compresses the unshifted grid toward
    sigma 1, so a uniform base draw puts three stops in five above shifted sigma
    0.9 and barely one in thirteen below 0.5. Drawing on the shifted coordinate
    spreads them evenly, which is what reaches the mid band the D-OPSD preflight
    measured the teacher strongest in."""
    trainer = MiniMaxH3NetworkTrainer()
    base_args = _flag_args()
    shifted_args = _flag_args("--h3_rollout_supervision", "--h3_rollout_stop_shifted", "--h3_rollout_teacher_config", "x")

    torch.manual_seed(0)
    base_stops = [trainer._draw_rollout_stop_sigma(base_args, VIDEO_FLOW_SHIFT) for _ in range(400)]
    shifted_stops = [trainer._draw_rollout_stop_sigma(shifted_args, VIDEO_FLOW_SHIFT) for _ in range(400)]

    def shifted(values):
        return shift_sigma(torch.tensor(values, dtype=torch.float64), VIDEO_FLOW_SHIFT)

    base_band = shifted(base_stops)
    wide_band = shifted(shifted_stops)
    # Without the flag the supervision piles up at the noisy end and the low half
    # of the schedule is all but unvisited.
    assert float((base_band > 0.9).float().mean()) > 0.5
    assert float((base_band < 0.5).float().mean()) < 0.15
    # With it the stops are uniform on the shifted coordinate, so the mid band
    # gets its proportional share and the low half gets half the steps.
    assert 0.30 < float(((wide_band > 0.5) & (wide_band < 0.9)).float().mean()) < 0.50
    assert 0.40 < float((wide_band < 0.5).float().mean()) < 0.60
    assert all(0.0 < float(value) < 1.0 for value in wide_band)


def test_rollout_stop_shifted_is_inert_on_an_unshifted_schedule():
    """An image step runs shift 1, where the round trip is the identity: the draw
    must be the same float the flag-off run made, not a re-derived approximation."""
    trainer = MiniMaxH3NetworkTrainer()
    off = _flag_args()
    on = _flag_args("--h3_rollout_supervision", "--h3_rollout_stop_shifted", "--h3_rollout_teacher_config", "x")

    torch.manual_seed(3)
    trainer._rollout_noise_generator = torch.Generator()
    trainer._rollout_noise_generator.manual_seed(21)
    baseline = [trainer._draw_rollout_stop_sigma(off, 1.0) for _ in range(8)]
    trainer._rollout_noise_generator.manual_seed(21)
    flagged = [trainer._draw_rollout_stop_sigma(on, 1.0) for _ in range(8)]

    assert baseline == flagged
    # And with the flag off, a shifted schedule is untouched too.
    trainer._rollout_noise_generator.manual_seed(21)
    assert [trainer._draw_rollout_stop_sigma(off, VIDEO_FLOW_SHIFT) for _ in range(8)] == baseline


# ---------------------------------------------------------------------------
# Validation matrix
# ---------------------------------------------------------------------------


def _teacher_file(tmp_path):
    path = tmp_path / "teacher.toml"
    path.write_text("[general]\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ((), "requires --h3_rollout_teacher_config"),
        (("--h3_rollout_steps", "0"), "must be at least 1"),
        (("--h3_rollout_window", "0"), r"\[1, 3\]"),
        (("--h3_rollout_window", "4"), r"\[1, 3\]"),
        (("--h3_rollout_probability", "0"), r"\(0, 1\]"),
        (("--h3_rollout_probability", "1.5"), r"\(0, 1\]"),
        (("--h3_video_loss_weight", "0"), "needs --h3_video_loss_weight"),
        (("--h3_observed_modality", "video"), "no observed side"),
        (("--h3_mask_mode", "box"), "observed"),
        (("--h3_extension_video_frames", "2"), "observed"),
        (("--h3_keyframe_anchors", "first"), "observed"),
        (("--h3_frame_sigma_jitter", "0.05"), "no room for"),
        (("--crepa",), "cannot be combined"),
    ],
)
def test_rollout_validation_matrix(tmp_path, flags, message):
    argv = ["--h3_rollout_supervision", *flags]
    if flags:
        argv = ["--h3_rollout_supervision", "--h3_rollout_teacher_config", str(_teacher_file(tmp_path)), *flags]

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_flag_args(*argv))


@pytest.mark.parametrize(
    "flag",
    [
        "--h3_rollout_probability",
        "--h3_rollout_steps",
        "--h3_rollout_window",
        "--h3_rollout_teacher_config",
        "--h3_rollout_stop_shifted",
    ],
)
def test_rollout_flags_require_the_switch(tmp_path, flag):
    # "3" rather than "1": the check reads "did the user write this flag" as a
    # comparison against the default, so a probe value equal to any dial's default
    # would test nothing. 3 is not the default of any of the three.
    argv = (flag,) if flag == "--h3_rollout_stop_shifted" else (flag, str(_teacher_file(tmp_path)) if "teacher" in flag else "3")

    with pytest.raises(ValueError, match="requires --h3_rollout_supervision"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_flag_args(*argv))


def test_rollout_accepts_the_guidance_family_since_round_two(tmp_path):
    """The hybrid: the guidance loss holds the field on inactive steps, which is
    the band round 1 measured the rollout eroding."""
    trainer = MiniMaxH3NetworkTrainer()

    trainer.handle_model_specific_args(
        _flag_args(
            *_ROLLOUT_FLAGS,
            "--h3_guidance_distillation_scale",
            "3.0",
            "--h3_guidance_loss_form",
            "contrastive",
            "--h3_guidance_cfg_zero",
            "--h3_guidance_distillation_probability",
            "0.5",
            teacher=_teacher_file(tmp_path),
        )
    )


def test_rollout_steps_are_not_capped_below_the_wide_band_requirement(tmp_path):
    """A shifted-band stop reaches much lower base sigmas, so the state needs more
    Euler strides to stay on-policy; 8 must validate."""
    MiniMaxH3NetworkTrainer().handle_model_specific_args(
        _flag_args(*_ROLLOUT_FLAGS, "--h3_rollout_stop_shifted", "--h3_rollout_steps", "8", teacher=_teacher_file(tmp_path))
    )


def test_rollout_teacher_config_must_exist():
    with pytest.raises(FileNotFoundError, match="h3_rollout_teacher_config"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(
            _flag_args("--h3_rollout_supervision", "--h3_rollout_teacher_config", "does_not_exist.toml")
        )


def test_rollout_is_recorded_in_adapter_metadata(tmp_path):
    teacher = _teacher_file(tmp_path)
    args = _flag_args(
        "--h3_rollout_supervision",
        "--h3_rollout_teacher_config",
        str(teacher),
        "--h3_rollout_probability",
        "0.25",
        "--h3_rollout_steps",
        "6",
        "--h3_rollout_window",
        "3",
        "--h3_rollout_stop_shifted",
    )
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)

    metadata = trainer.extra_metadata(args)
    assert metadata["ss_h3_rollout_stop_shifted"] == "True"
    assert MiniMaxH3NetworkTrainer().extra_metadata(_flag_args())["ss_h3_rollout_stop_shifted"] == "False"
    assert metadata["ss_h3_rollout_supervision"] == "True"
    assert metadata["ss_h3_rollout_probability"] == "0.25"
    assert metadata["ss_h3_rollout_steps"] == "6"
    assert metadata["ss_h3_rollout_window"] == "3"
    assert metadata["ss_h3_rollout_teacher"] == str(teacher)
    assert MiniMaxH3NetworkTrainer().extra_metadata(_flag_args())["ss_h3_rollout_supervision"] == "False"


def test_full_finetune_rejects_rollout_supervision(tmp_path):
    from musubi_tuner.minimax_h3_train import MiniMaxH3Trainer, create_parser as create_full_parser

    args = create_full_parser().parse_args(
        [
            "--sdpa",
            "--mixed_precision",
            "bf16",
            "--full_bf16",
            "--h3_rollout_supervision",
            "--h3_rollout_teacher_config",
            str(_teacher_file(tmp_path)),
        ]
    )

    with pytest.raises(ValueError, match="needs an adapter it can disable"):
        MiniMaxH3Trainer()._validate_full_finetune_args(args)


# ---------------------------------------------------------------------------
# Fused teacher: one pass over the blocks per supervised sub-step
# ---------------------------------------------------------------------------


def _tiny_transformer():
    from musubi_tuner.minimax_h3.model import MiniMaxH3Transformer, MiniMaxH3TransformerConfig

    config = MiniMaxH3TransformerConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=24,
        num_layers=3,
        num_refiner_layers=2,
        ffn_dim=32,
        in_channels=4,
        audio_in_channels=6,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=24,
        time_embed_dim=16,
        rope_freq_dim=2,
    )
    torch.manual_seed(11)
    return MiniMaxH3Transformer(config)


def _tiny_adapter(transformer):
    """A real LoRA on the tiny transformer, with a delta that is not zero.

    ``lora_up`` is zero-initialised, so an untouched adapter is the identity and
    would make the teacher's disabled forward agree with the student's for the
    wrong reason.
    """
    from musubi_tuner.networks import lora_minimax_h3

    network = lora_minimax_h3.create_arch_network(1.0, 4, 4.0, None, [], transformer)
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    torch.manual_seed(12)
    for module in network.unet_loras:
        torch.nn.init.normal_(module.lora_up.weight, std=0.05)
    return network


def _fused_arm_batches():
    """A student presentation and the longer, privileged teacher presentation.

    The teacher's text is longer on purpose: that is what a ``qwen_control_*``
    cache produces, and it is the reason the two arms pack sequences of different
    lengths and cannot be stacked on a batch axis.
    """
    torch.manual_seed(13)
    student = {
        H3_TEXT_HIDDEN_KEY: [torch.randn(4, 8)],
        H3_TEXT_TOKEN_TAGS_KEY: [torch.ones(4, dtype=torch.long)],
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["t2va"])],
    }
    teacher = dict(student)
    teacher[H3_TEXT_HIDDEN_KEY] = [torch.randn(7, 8)]
    teacher[H3_TEXT_TOKEN_TAGS_KEY] = [torch.ones(7, dtype=torch.long)]
    return student, teacher


def _fused_call(batch, video, audio, timestep):
    return {
        "batch": batch,
        "video_hidden_states": video,
        "audio_hidden_states": audio,
        "video_timestep": timestep,
        "audio_timestep": timestep,
        "conditioning": "prompt",
    }


def _fused_arms(network, student_call, teacher_call):
    from musubi_tuner.minimax_h3.training import H3FusedArm
    from musubi_tuner.minimax_h3_train_network import _forked_frozen_build

    return [
        H3FusedArm(call=student_call),
        H3FusedArm(
            call=teacher_call,
            build=lambda: _forked_frozen_build([], network.set_enabled),
            run=MiniMaxH3NetworkTrainer._frozen_arm_context(network.set_enabled),
        ),
    ]


def test_fused_arms_reproduce_the_separate_forwards_exactly():
    """The correctness gate: fusing changes the schedule, never the arithmetic."""
    from musubi_tuner.minimax_h3.integration import _NativeTrainingBackend

    transformer = _tiny_transformer()
    network = _tiny_adapter(transformer)
    backend = _NativeTrainingBackend(transformer)
    student_batch, privileged_batch = _fused_arm_batches()
    video = torch.randn(1, 4, 2, 4, 4)
    audio = torch.randn(1, 2, 6, 3)
    timestep = torch.tensor([0.4])
    student_call = _fused_call(student_batch, video, audio, timestep)
    teacher_call = _fused_call(privileged_batch, video, audio, timestep)

    # The naive path, exactly as the unfused rollout runs it.
    separate_student = backend.predict_training(transformer, **student_call)
    with torch.no_grad():
        network.set_enabled(False)
        try:
            separate_teacher = backend.predict_training(transformer, **teacher_call)
        finally:
            network.set_enabled(True)

    fused_student, fused_teacher = backend.predict_training_fused(transformer, _fused_arms(network, student_call, teacher_call))

    torch.testing.assert_close(fused_student.video, separate_student.video, rtol=0, atol=0)
    torch.testing.assert_close(fused_student.audio, separate_student.audio, rtol=0, atol=0)
    torch.testing.assert_close(fused_teacher.video, separate_teacher.video, rtol=0, atol=0)
    torch.testing.assert_close(fused_teacher.audio, separate_teacher.audio, rtol=0, atol=0)
    # The privileged arm really is a different, longer packed sequence, so the
    # parity above is not the trivial one of two identical presentations.
    assert not torch.allclose(separate_student.video, separate_teacher.video)


def test_fused_teacher_arm_carries_no_graph_and_the_student_arm_does():
    from musubi_tuner.minimax_h3.integration import _NativeTrainingBackend

    transformer = _tiny_transformer()
    network = _tiny_adapter(transformer)
    backend = _NativeTrainingBackend(transformer)
    student_batch, privileged_batch = _fused_arm_batches()
    video = torch.randn(1, 4, 2, 4, 4)
    audio = torch.randn(1, 2, 6, 3)
    timestep = torch.tensor([0.4])

    student, teacher = backend.predict_training_fused(
        transformer,
        _fused_arms(
            network,
            _fused_call(student_batch, video, audio, timestep),
            _fused_call(privileged_batch, video, audio, timestep),
        ),
    )

    assert student.video.requires_grad and student.audio.requires_grad
    assert not teacher.video.requires_grad and not teacher.audio.requires_grad
    # The adapter is live again on the way out, and only the student's arm feeds
    # it gradients.
    student.video.sum().backward()
    assert all(module.enabled for module in network.unet_loras)
    assert any(module.lora_up.weight.grad is not None for module in network.unet_loras)


class _RecordingOffloader:
    """Counts the block-ring traffic a forward asks for."""

    recompute_requires_wait = False

    def __init__(self):
        self.waits = []
        self.submits = []

    def wait_for_block(self, index):
        self.waits.append(index)

    def submit_move_blocks_forward(self, blocks, index):
        self.submits.append(index)


def test_a_fused_forward_streams_each_swapped_block_once_for_every_arm():
    from musubi_tuner.minimax_h3.integration import _NativeTrainingBackend
    from musubi_tuner.minimax_h3.training import H3FusedArm

    transformer = _tiny_transformer()
    backend = _NativeTrainingBackend(transformer)
    offloader = _RecordingOffloader()
    transformer.blocks_to_swap = 1
    transformer.offloader = offloader
    student_batch, privileged_batch = _fused_arm_batches()
    video = torch.randn(1, 4, 2, 4, 4)
    audio = torch.randn(1, 2, 6, 3)
    timestep = torch.tensor([0.4])

    with torch.no_grad():
        backend.predict_training_fused(
            transformer,
            [
                H3FusedArm(call=_fused_call(student_batch, video, audio, timestep)),
                H3FusedArm(call=_fused_call(privileged_batch, video, audio, timestep)),
            ],
        )
        fused_waits = list(offloader.waits)
        offloader.waits.clear()
        offloader.submits.clear()
        backend.predict_training(transformer, **_fused_call(student_batch, video, audio, timestep))

    # Two arms cost one traversal of the ring, exactly what one arm costs.
    assert fused_waits == list(range(len(transformer.blocks)))
    assert offloader.waits == list(range(len(transformer.blocks)))
    assert offloader.submits == list(range(len(transformer.blocks)))


class _FusedRolloutBackend(_RolloutBackend):
    """The stub field, taught to answer several arms in one call."""

    def __init__(self, velocity=0.25):
        super().__init__(velocity)
        self.fused_calls = 0

    def predict_training_fused(self, transformer, arms):
        self.fused_calls += 1
        predictions = []
        for arm in arms:
            with (arm.build or nullcontext)():
                pass
            with (arm.run or nullcontext)():
                call = dict(arm.call)
                predictions.append(
                    self.predict_training(
                        transformer,
                        call.pop("batch"),
                        call.pop("video_hidden_states"),
                        call.pop("audio_hidden_states"),
                        call.pop("video_timestep"),
                        call.pop("audio_timestep"),
                        **call,
                    )
                )
        return predictions


def test_the_fused_rollout_step_produces_the_same_loss_as_the_unfused_one(tmp_path):
    teacher = tmp_path / "teacher.toml"
    teacher.write_text("[general]\n", encoding="utf-8")
    naive_backend, naive_loss, naive_metrics, _ = _run_step(
        *_ROLLOUT_FLAGS,
        "--h3_rollout_steps",
        "2",
        "--h3_rollout_window",
        "2",
        teacher=teacher,
        backend=_FusedRolloutBackend(),
    )
    fused_backend, fused_loss, fused_metrics, _ = _run_step(
        *_ROLLOUT_FLAGS,
        "--h3_rollout_steps",
        "2",
        "--h3_rollout_window",
        "2",
        "--h3_rollout_fused_teacher",
        teacher=teacher,
        backend=_FusedRolloutBackend(),
    )

    assert naive_backend.fused_calls == 0
    # One fused call per supervised sub-step, replacing two separate forwards.
    assert fused_backend.fused_calls == 2
    torch.testing.assert_close(fused_loss, naive_loss, rtol=0, atol=0)
    assert fused_metrics == naive_metrics
    # Same forwards, same order, same adapter and grad state on each of them.
    assert [call["signature"] for call in fused_backend.calls] == [call["signature"] for call in naive_backend.calls]
    assert [call["grad"] for call in fused_backend.calls] == [call["grad"] for call in naive_backend.calls]
    assert [call["adapter"] for call in fused_backend.calls] == [call["adapter"] for call in naive_backend.calls]


def test_the_fused_teacher_needs_a_backend_that_can_fuse(tmp_path):
    teacher = tmp_path / "teacher.toml"
    teacher.write_text("[general]\n", encoding="utf-8")
    with pytest.raises(TypeError, match="predict_training_fused"):
        _run_step(*_ROLLOUT_FLAGS, "--h3_rollout_fused_teacher", teacher=teacher, backend=_RolloutBackend())


def test_the_fused_teacher_flag_requires_the_switch():
    with pytest.raises(ValueError, match="requires --h3_rollout_supervision"):
        MiniMaxH3NetworkTrainer()._validate_rollout_args(_flag_args("--h3_rollout_fused_teacher"))


def test_the_fused_teacher_cannot_select_int8_for_the_teacher_arm_alone(tmp_path):
    args = _flag_args(
        *_ROLLOUT_FLAGS,
        "--h3_rollout_fused_teacher",
        "--h3_int8_attention",
        "aux",
        teacher=_teacher_file(tmp_path),
    )
    with pytest.raises(ValueError, match="INT8 for the teacher arm alone"):
        MiniMaxH3NetworkTrainer()._validate_rollout_args(args)
    # The modes that apply to both arms alike compose with fusion unchanged.
    for mode in ("off", "train"):
        pinned = _flag_args(
            *_ROLLOUT_FLAGS,
            "--h3_rollout_fused_teacher",
            "--h3_int8_attention",
            mode,
            teacher=_teacher_file(tmp_path),
        )
        MiniMaxH3NetworkTrainer()._validate_rollout_args(pinned)


def test_the_fused_rollout_is_recorded_in_adapter_metadata(tmp_path):
    args = _flag_args(*_ROLLOUT_FLAGS, "--h3_rollout_fused_teacher", teacher=_teacher_file(tmp_path))
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    assert trainer.extra_metadata(args)["ss_h3_rollout_fused_teacher"] == "True"
    assert MiniMaxH3NetworkTrainer().extra_metadata(_flag_args())["ss_h3_rollout_fused_teacher"] == "False"


# ---------------------------------------------------------------------------
# Micro-rollout: depth-0 privileged-teacher supervision on inactive steps
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Micro-rollout: depth-0 teacher-GAP contrast distillation on inactive steps
# ---------------------------------------------------------------------------


# What the stub field answers at each of the four evaluations, given a student
# prompt of 0.25, a student null of 0.0, a teacher prompt of 0.75 and a teacher
# null of 0.25, with the adapter contributing 0.5 * 0.25 and the frozen base
# 1.0 * 0.25.
_STUDENT_PROMPT = 0.5 * 0.25 + 0.25
_STUDENT_EMPTY = 0.5 * 0.25 + 0.0
_TEACHER_PROMPT = 1.0 * 0.25 + 0.75
_TEACHER_EMPTY = 1.0 * 0.25 + 0.25


# ---------------------------------------------------------------------------
# Fused arms and the block-swap ring: retiring a block that another arm still needs
# ---------------------------------------------------------------------------


class _RetiringOffloader(FusedArmBackwardGate):
    """The block-swap accounting that a fused pass has to survive, and no more.

    Real block swap needs CUDA, so what is modelled here is only the invariant the
    H100 run violated: a swapped block's weights are on the GPU between its
    ``wait_for_block`` and the backward-hook firing that RETIRES it -- frees its
    ring slot and repoints its weights back at the CPU master -- and any use of
    the block after that reads a CPU weight, which is what raised ``Expected all
    tensors to be on the same device ... mat2 is on cpu`` at the block's first
    matmul.

    Residency is modelled the way ``LoRAStreamOffloader`` really moves it: a
    ``wait_for_block`` makes the block resident, the ring wrapping past it in the
    forward takes it away again, and so does the backward-hook firing that RETIRES
    it. Entering a block that is not resident is the crash.

    ``gated=False`` restores the pre-fix behaviour (retire on every firing) and
    ``pre_wait=False`` drops the backward pre-hook, so the tests can show that
    both halves of the guard are load-bearing.
    """

    recompute_requires_wait = False

    def __init__(self, blocks, *, gated=True, ring=2, pre_wait=True):
        self.reset_backward_arms()
        self.gated = gated
        self.block_count = len(blocks)
        # What a forward leaves behind: the ring holds the last ``ring`` blocks it
        # streamed, every earlier one has been repointed at its CPU master.
        self.ring = ring
        self.resident = set()
        self.offenders = []
        self.events = []
        self.handles = []
        for index, block in enumerate(blocks):
            if pre_wait:
                self.handles.append(block.register_full_backward_pre_hook(self._pre(index)))
            self.handles.append(block.register_full_backward_hook(self._retire(index)))
            self.handles.append(block.register_forward_pre_hook(self._enter(index)))

    def _enter(self, index):
        def hook(module, inputs):
            # Fires on the pass's own forward AND on every gradient-checkpoint
            # recomputation, which is exactly the set of moments the block's
            # weights have to be resident.
            self.events.append(("enter", index))
            if index not in self.resident:
                self.offenders.append(index)

        return hook

    def _pre(self, index):
        def hook(module, grad_output):
            # The offloader's backward PRE-hook: residency for this invocation,
            # whatever the ring schedule did.
            self._wait(index, "wait_bwd")

        return hook

    def _retire(self, index):
        def hook(module, grad_input, grad_output):
            if self.gated and not self.retire_backward_arm(index):
                return
            # Retiring frees the ring slot: the block's weights go back to their
            # CPU master, and the block behind it is pulled in and waited for.
            self.events.append(("retire", index))
            self.resident.discard(index)
            if index - 1 >= 0:
                self._wait(index - 1, "wait_bwd")

        return hook

    def _wait(self, index, kind):
        self.events.append((kind, index))
        self.resident.add(index)

    def wait_for_block(self, index):
        self._wait(index, "wait")

    def submit_move_blocks_forward(self, blocks, index):
        self.events.append(("submit", index))
        # The ring wraps: consuming this block pulls in the one ``ring`` slots
        # ahead, which evicts this one unless it is in the tail the forward
        # leaves resident for the backward to turn around on.
        if index < self.block_count - self.ring:
            self.resident.discard(index)

    def remove(self):
        for handle in self.handles:
            handle.remove()

    def entries_after_retirement(self):
        """Blocks entered while their weights sat on the CPU -- the crash, as a list."""
        return self.offenders


def _swapped_fused_arms(network, transformer, *, trainable_arms, gated=True, pre_wait=True):
    """Run one fused pass of ``trainable_arms`` student arms plus two frozen ones."""
    from musubi_tuner.minimax_h3.integration import _NativeTrainingBackend
    from musubi_tuner.minimax_h3.training import H3FusedArm
    from musubi_tuner.minimax_h3_train_network import _forked_frozen_build

    backend = _NativeTrainingBackend(transformer)
    student_batch, privileged_batch = _fused_arm_batches()
    video = torch.randn(1, 4, 2, 4, 4)
    audio = torch.randn(1, 2, 6, 3)
    timestep = torch.tensor([0.4])

    # Gradient checkpointing on: the failure is a RECOMPUTATION reading a weight
    # that a sibling arm's backward hook already sent back to the CPU, so a run
    # without checkpointing would never reach it.
    transformer.gradient_checkpointing = True
    transformer.blocks_to_swap = 1
    offloader = _RetiringOffloader(transformer.blocks, gated=gated, pre_wait=pre_wait)
    transformer.offloader = offloader

    frozen = [
        H3FusedArm(
            call=_fused_call(privileged_batch, video, audio, timestep),
            build=lambda: _forked_frozen_build([], network.set_enabled),
            run=MiniMaxH3NetworkTrainer._frozen_arm_context(network.set_enabled),
        )
        for _ in range(2)
    ]
    arms = [H3FusedArm(call=_fused_call(student_batch, video, audio, timestep)) for _ in range(trainable_arms)] + frozen
    try:
        predictions = backend.predict_training_fused(transformer, arms)
        loss = sum(prediction.video.sum() for prediction in predictions[:trainable_arms])
        loss.backward()
    finally:
        offloader.remove()
    return offloader


def test_a_fused_pass_retires_each_swapped_block_after_its_last_trainable_arm():
    """The H100 regression: four arms, two of them trainable.

    The block-swap ring is driven by a per-block backward hook, and that hook
    fires once per graph-carrying invocation. A fused pass invokes each block once
    per ARM, so two trainable arms fire it twice -- and retiring the block on the
    first firing pulls its weights back to the CPU while the second arm has not
    yet recomputed it under gradient checkpointing.
    """
    transformer = _tiny_transformer()
    network = _tiny_adapter(transformer)
    offloader = _swapped_fused_arms(network, transformer, trainable_arms=2)

    block_count = len(transformer.blocks)
    # Every block retired, exactly once, however many arms asked for it.
    retirements = [index for kind, index in offloader.events if kind == "retire"]
    assert sorted(retirements) == list(range(block_count))
    # And nothing touched a block after it was retired: that is the crash.
    assert offloader.entries_after_retirement() == []
    # The ring itself is still traversed once for the whole pass, not once per arm.
    assert [index for kind, index in offloader.events if kind == "wait"] == list(range(block_count))
    assert [index for kind, index in offloader.events if kind == "submit"] == list(range(block_count))


def test_the_ungated_hook_really_does_retire_a_block_another_arm_still_needs():
    """The same pass with the pre-fix accounting, to show the guard discriminates.

    Without the gate the first of the two firings retires the block, and the
    sibling arm's checkpoint recomputation then reads it -- on CUDA that read is
    the reported ``mat2 is on cpu``.
    """
    transformer = _tiny_transformer()
    network = _tiny_adapter(transformer)
    offloader = _swapped_fused_arms(network, transformer, trainable_arms=2, gated=False, pre_wait=False)

    assert offloader.entries_after_retirement() != []
    # Retired once per trainable arm rather than once per block.
    retirements = [index for kind, index in offloader.events if kind == "retire"]
    assert len(retirements) == 2 * len(transformer.blocks)


def test_a_single_trainable_arm_keeps_the_accounting_it_always_had():
    """Why teacher_velocity's two-arm fusion never hit this.

    One trainable arm means one firing per block, so the gate retires on that
    firing and the gated and ungated paths agree event for event.
    """
    events = []
    for gated in (True, False):
        transformer = _tiny_transformer()
        network = _tiny_adapter(transformer)
        offloader = _swapped_fused_arms(network, transformer, trainable_arms=1, gated=gated)
        assert offloader.entries_after_retirement() == []
        events.append(offloader.events)

    assert events[0] == events[1]


def test_the_gate_counts_only_arms_that_actually_build_a_graph():
    """A frozen arm registers no backward node, so it must not be announced.

    Counting arms instead of graph-carrying invocations would leave every block
    holding an announcement no hook will ever consume, and the block would never
    be retired at all.
    """
    transformer = _tiny_transformer()
    network = _tiny_adapter(transformer)
    offloader = _swapped_fused_arms(network, transformer, trainable_arms=2)

    # Two trainable arms among four; nothing is left pending once the pass ends.
    assert offloader._backward_arms == {}


def _swapped_fused_step(network, transformer, *, passes, trainable_arms=1, gated=True, pre_wait=True, ring=1):
    """One optimizer step that traverses the blocks ``passes`` times before backward.

    The shape of an ACTIVE rollout step: ``--h3_rollout_window`` supervised
    sub-steps, each a fused (student, frozen teacher) pass over the whole ring,
    all of them backwarded together at the end of the step.
    """
    from musubi_tuner.minimax_h3.integration import _NativeTrainingBackend
    from musubi_tuner.minimax_h3.training import H3FusedArm
    from musubi_tuner.minimax_h3_train_network import _forked_frozen_build

    backend = _NativeTrainingBackend(transformer)
    student_batch, privileged_batch = _fused_arm_batches()
    video = torch.randn(1, 4, 2, 4, 4)
    audio = torch.randn(1, 2, 6, 3)
    timestep = torch.tensor([0.4])

    transformer.gradient_checkpointing = True
    transformer.blocks_to_swap = 1
    offloader = _RetiringOffloader(transformer.blocks, gated=gated, ring=ring, pre_wait=pre_wait)
    transformer.offloader = offloader

    losses = []
    try:
        for _ in range(passes):
            arms = [H3FusedArm(call=_fused_call(student_batch, video, audio, timestep)) for _ in range(trainable_arms)]
            arms.append(
                H3FusedArm(
                    call=_fused_call(privileged_batch, video, audio, timestep),
                    build=lambda: _forked_frozen_build([], network.set_enabled),
                    run=MiniMaxH3NetworkTrainer._frozen_arm_context(network.set_enabled),
                )
            )
            predictions = backend.predict_training_fused(transformer, arms)
            losses.append(sum(prediction.video.sum() for prediction in predictions[:trainable_arms]))
        # One backward for the whole step, which is what the rollout does: autograd
        # unwinds the sub-step passes newest-first, as many sweeps over the ring as
        # the step made.
        sum(losses).backward()
    finally:
        offloader.remove()
    return offloader


def test_an_active_rollout_steps_fused_sub_steps_each_drive_the_ring_on_their_own():
    """The remaining H100 crash: ``--h3_rollout_window 2`` is TWO fused passes.

    A step may traverse the blocks several times before its single backward, and
    autograd unwinds those traversals newest-first. Summing the announcements of
    all of them lets the newest sweep's firings consume the older sweep's count
    and be suppressed wholesale -- including the part of the hook that makes the
    NEXT block resident -- so the sweep runs on with the ring standing still and
    the first evicted block it recomputes reads a CPU weight.
    """
    transformer = _tiny_transformer()
    network = _tiny_adapter(transformer)
    offloader = _swapped_fused_step(network, transformer, passes=2)

    block_count = len(transformer.blocks)
    retirements = [index for kind, index in offloader.events if kind == "retire"]
    # Two sweeps down the ring, each retiring every block once.
    assert retirements == list(reversed(range(block_count))) * 2
    assert offloader.entries_after_retirement() == []
    # Nothing left pending: the announcements balance the firings exactly.
    assert offloader._backward_arms == {}


def _swapped_plain_passes(network, transformer, *, passes, pre_wait=True, ring=1):
    """A step of ``passes`` ORDINARY trainable forwards, backwarded together.

    What an active rollout step looks like WITHOUT ``--h3_rollout_fused_teacher``:
    the window's student forwards are plain ones, the teachers are separate no-grad
    forwards, and the step still ends with one backward over several graphs.
    """
    from musubi_tuner.minimax_h3.integration import _NativeTrainingBackend

    backend = _NativeTrainingBackend(transformer)
    student_batch, privileged_batch = _fused_arm_batches()
    video = torch.randn(1, 4, 2, 4, 4)
    audio = torch.randn(1, 2, 6, 3)
    timestep = torch.tensor([0.4])

    transformer.gradient_checkpointing = True
    transformer.blocks_to_swap = 1
    offloader = _RetiringOffloader(transformer.blocks, ring=ring, pre_wait=pre_wait)
    transformer.offloader = offloader

    losses = []
    try:
        for _ in range(passes):
            losses.append(backend.predict_training(transformer, **_fused_call(student_batch, video, audio, timestep)).video.sum())
            with torch.no_grad():
                network.set_enabled(False)
                try:
                    backend.predict_training(transformer, **_fused_call(privileged_batch, video, audio, timestep))
                finally:
                    network.set_enabled(True)
        sum(losses).backward()
    finally:
        offloader.remove()
    return offloader


def test_the_naive_rollout_window_is_several_backward_sweeps_too():
    """The same crash without ``--h3_rollout_fused_teacher``.

    Fusion changes only how many arms ride one traversal; the window still costs
    one graph-carrying traversal per sub-step, so the backward still sweeps the
    ring once per sub-step and still has to turn around between them.
    """
    transformer = _tiny_transformer()
    network = _tiny_adapter(transformer)
    offloader = _swapped_plain_passes(network, transformer, passes=2)

    block_count = len(transformer.blocks)
    assert [index for kind, index in offloader.events if kind == "retire"] == list(reversed(range(block_count))) * 2
    assert offloader.entries_after_retirement() == []


def test_the_naive_rollout_window_needs_the_turn_around_reload_as_well():
    """And without the pre-hook it reads a CPU weight, gate or no gate: the naive
    path never announces an arm, so no accounting could have saved it."""
    transformer = _tiny_transformer()
    network = _tiny_adapter(transformer)
    offloader = _swapped_plain_passes(network, transformer, passes=2, pre_wait=False)

    assert offloader.entries_after_retirement() != []


class _ReplayRing(FusedArmBackwardGate):
    """``LoRAStreamOffloader``'s ring with the transfers modelled, not performed.

    Every rule is transcribed from the real class -- the ``rank % B`` slot map,
    ``_load``'s eviction of a slot's previous owner, ``wait_for_block``'s
    self-heal on a slot holding someone else, the forward prefetch ``B`` slots
    ahead, and the rule that gives a block a backward hook only when it streams
    or its predecessor does. What it models instead of copying bytes is the
    BINDING: a block is usable exactly while its weights point at a ring slot,
    and reading one that has been repointed at its CPU master is the reported
    ``mat2 is on cpu``.

    The backward pre-hook is taken from the real class, so this harness sees
    whatever the offloader actually registers.
    """

    recompute_requires_wait = False
    _wait_ctx = "fwd"

    def __init__(self, blocks, stream_idx, *, ring=1, gated=True):
        from musubi_tuner.modules.custom_offloading_utils import LoRAStreamOffloader

        self.reset_backward_arms()
        self.gated = gated
        self.stream_idx = list(stream_idx)
        self.S = len(self.stream_idx)
        self.B = min(ring, self.S)
        self.rank = {block: index for index, block in enumerate(self.stream_idx)}
        self.is_stream = [index in self.rank for index in range(len(blocks))]
        self.in_slot = [None] * self.B
        self.bound = set()
        self.events = []
        self.offenders = []
        self.handles = []
        pre_hook = getattr(LoRAStreamOffloader, "_create_backward_pre_hook", None)
        for index, block in enumerate(blocks):
            if pre_hook is not None and self.is_stream[index]:
                self.handles.append(block.register_full_backward_pre_hook(pre_hook(self, index)))
            hook = self._hook(index)
            if hook is not None:
                self.handles.append(block.register_full_backward_hook(hook))
            self.handles.append(block.register_forward_pre_hook(self._enter(index)))

    def _load(self, rank, slot):
        block = self.stream_idx[rank]
        if self.in_slot[slot] == block:
            self.bound.add(block)
            return
        if self.in_slot[slot] is not None:
            self.bound.discard(self.in_slot[slot])  # repointed at its CPU master
        self.bound.add(block)
        self.in_slot[slot] = block

    def wait_for_block(self, index):
        if not self.is_stream[index]:
            return
        rank = self.rank[index]
        self.events.append(("wait", index))
        if self.in_slot[rank % self.B] != index:
            self._load(rank, rank % self.B)

    def submit_move_blocks_forward(self, blocks, index):
        if not self.is_stream[index]:
            return
        rank = self.rank[index]
        self.events.append(("submit", index))
        if rank + self.B < self.S:
            self._load(rank + self.B, (rank + self.B) % self.B)

    def _hook(self, index):
        prefetch = self.is_stream[index]
        wait_prev = index - 1 >= 0 and self.is_stream[index - 1]
        if not prefetch and not wait_prev:
            return None

        def hook(module, grad_input, grad_output):
            if self.gated and not self.retire_backward_arm(index):
                return
            self.events.append(("retire", index))
            if prefetch and self.rank[index] - self.B >= 0:
                self._load(self.rank[index] - self.B, (self.rank[index] - self.B) % self.B)
            if wait_prev:
                self.wait_for_block(index - 1)

        return hook

    def _enter(self, index):
        def hook(module, inputs):
            self.events.append(("enter", index))
            if self.is_stream[index] and index not in self.bound:
                self.offenders.append(index)

        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()


def _replay_transformer(stream_idx):
    transformer = _tiny_transformer()
    network = _tiny_adapter(transformer)
    transformer.gradient_checkpointing = True
    transformer.blocks_to_swap = 1
    ring = _ReplayRing(transformer.blocks, stream_idx)
    transformer.offloader = ring
    return transformer, network, ring


def _naive_rollout_step(transformer, network, *, graph_traversals, prefix_traversals=2):
    """One NAIVE step: trainable forwards with no-grad teacher forwards between them.

    ``graph_traversals=1`` is an inactive teacher_match step -- the data forward
    plus its depth-0 teacher. More than one is an active rollout step, whose
    window costs one trainable traversal per sub-step and backwards them all at
    the end of the step.
    """
    from musubi_tuner.minimax_h3.integration import _NativeTrainingBackend

    backend = _NativeTrainingBackend(transformer)
    student_batch, privileged_batch = _fused_arm_batches()
    video = torch.randn(1, 4, 2, 4, 4)
    audio = torch.randn(1, 2, 6, 3)
    timestep = torch.tensor([0.4])

    def frozen():
        with torch.no_grad():
            network.set_enabled(False)
            try:
                backend.predict_training(transformer, **_fused_call(privileged_batch, video, audio, timestep))
            finally:
                network.set_enabled(True)

    losses = []
    for _ in range(prefix_traversals):
        frozen()
    for _ in range(graph_traversals):
        losses.append(backend.predict_training(transformer, **_fused_call(student_batch, video, audio, timestep)).video.sum())
        frozen()
    sum(losses).backward()


def test_one_traversal_per_step_never_needs_the_turn_around():
    """Why a run with exactly one graph traversal per step ran clean for 1500 steps.

    One graph-carrying traversal is one sweep, and a single sweep starts exactly
    where the forward left the ring -- interleaved no-grad teacher traversals
    included, since each of them leaves it in the same place.
    """
    transformer, network, ring = _replay_transformer(stream_idx=[1, 2])
    try:
        _naive_rollout_step(transformer, network, graph_traversals=1)
    finally:
        ring.remove()

    assert ring.offenders == []


def test_the_top_block_streams_exactly_when_the_swap_count_reaches_half():
    """Which layouts can reach the turn-around at all, as a closed form.

    ``compute_offload_block_indices`` puts the streamed blocks at the midpoints
    of ``blocks_to_swap`` equal spans, so the highest of them lands on the LAST
    block exactly when ``num_blocks <= 2 * blocks_to_swap``. Below the line the
    last block is resident and its hook covers the top of every backward sweep;
    at or above it the last block streams and nothing but the backward pre-hook
    does. The released 50-block transformer crosses at ``--blocks_to_swap 25``;
    a reduced-layer build of 48 blocks or fewer already crosses it at 24.
    """
    from musubi_tuner.modules.custom_offloading_utils import compute_offload_block_indices

    for num_blocks in range(2, 64):
        for swap in range(1, num_blocks - 1):
            top_streams = (num_blocks - 1) in compute_offload_block_indices(num_blocks, swap, h2d_only=True)
            assert top_streams == (num_blocks <= 2 * swap), (num_blocks, swap)


def test_the_top_streamed_block_survives_a_naive_multi_sweep_backward():
    """The naive crash: a layout whose LAST block streams, per the rule above.

    The hook chain covers every streamed block through its ``+1`` neighbour's
    ``wait_for_block``, but the last block has no neighbour above it. Its
    residency comes from the forward alone, which is enough for a backward of ONE
    sweep -- the forward ended there -- and not for the second sweep of a step
    whose rollout window traversed the blocks more than once: by then the first
    sweep has retired it and walked the ring back down to rank 0, and the sweep
    opens on its checkpoint recomputation against a weight on the CPU master.

    It needs no fused pass and no arm accounting: the naive window is several
    sweeps too.
    """
    transformer, network, ring = _replay_transformer(stream_idx=[1, 2])
    try:
        _naive_rollout_step(transformer, network, graph_traversals=3)
    finally:
        ring.remove()

    assert ring.offenders == []


def test_a_resident_top_block_hid_the_turn_around_for_smaller_swap_counts():
    """And why the same window was clean below 25 swapped blocks.

    With the top block resident its hook still fires and still waits for the
    highest streamed block, so the chain re-enters the ring on every sweep. That
    is the layout the earlier fused benchmark ran on, which is why only the ARM
    ACCOUNTING broke it there.
    """
    transformer, network, ring = _replay_transformer(stream_idx=[0, 1])
    try:
        _naive_rollout_step(transformer, network, graph_traversals=3)
    finally:
        ring.remove()

    assert ring.offenders == []


def test_the_h2d_only_ring_waits_for_a_block_before_its_own_backward():
    """The offloader's own guarantee, independent of any arm accounting."""
    from musubi_tuner.modules.custom_offloading_utils import LoRAStreamOffloader

    offloader = LoRAStreamOffloader.__new__(LoRAStreamOffloader)
    offloader._wait_ctx = "fwd"
    waited = []
    offloader.wait_for_block = lambda index: waited.append((index, offloader._wait_ctx))

    LoRAStreamOffloader._create_backward_pre_hook(offloader, 7)(None, None)

    # Waited for, as a backward-context load, and the context tag is restored.
    assert waited == [(7, "bwd")]
    assert offloader._wait_ctx == "fwd"


def test_a_later_sweep_reloads_the_block_the_previous_sweep_retired():
    """Why the ring alone cannot carry a multi-sweep backward.

    The hook chain is a schedule -- finish block i, make i-1 resident -- and it has
    no step across a sweep boundary: the sweep that just ended left the ring
    holding the FIRST blocks, and the next one starts at the last. Only the
    backward pre-hook covers that turn-around.
    """
    transformer = _tiny_transformer()
    network = _tiny_adapter(transformer)
    offloader = _swapped_fused_step(network, transformer, passes=2, pre_wait=False)

    assert offloader.entries_after_retirement() != []


def test_a_step_that_mixes_a_fused_pass_and_a_plain_one_keeps_them_apart():
    """Fused and unfused traversals in one step, each announcing its own group.

    A traversal that announced nothing would have its firings consume the group
    the pass beside it announced, which is the same failure by another route --
    so the plain forward opens a group of its own.
    """
    from musubi_tuner.minimax_h3.integration import _NativeTrainingBackend
    from musubi_tuner.minimax_h3.training import H3FusedArm
    from musubi_tuner.minimax_h3_train_network import _forked_frozen_build

    transformer = _tiny_transformer()
    network = _tiny_adapter(transformer)
    backend = _NativeTrainingBackend(transformer)
    student_batch, privileged_batch = _fused_arm_batches()
    video = torch.randn(1, 4, 2, 4, 4)
    audio = torch.randn(1, 2, 6, 3)
    timestep = torch.tensor([0.4])

    transformer.gradient_checkpointing = True
    transformer.blocks_to_swap = 1
    offloader = _RetiringOffloader(transformer.blocks, ring=1)
    transformer.offloader = offloader

    try:
        fused = backend.predict_training_fused(
            transformer,
            [
                H3FusedArm(call=_fused_call(student_batch, video, audio, timestep)),
                H3FusedArm(call=_fused_call(student_batch, video, audio, timestep)),
                H3FusedArm(
                    call=_fused_call(privileged_batch, video, audio, timestep),
                    build=lambda: _forked_frozen_build([], network.set_enabled),
                    run=MiniMaxH3NetworkTrainer._frozen_arm_context(network.set_enabled),
                ),
            ],
        )
        plain = backend.predict_training(transformer, **_fused_call(student_batch, video, audio, timestep))
        (sum(prediction.video.sum() for prediction in fused[:2]) + plain.video.sum()).backward()
    finally:
        offloader.remove()

    block_count = len(transformer.blocks)
    retirements = [index for kind, index in offloader.events if kind == "retire"]
    # One sweep down the ring per traversal, the plain one first because autograd
    # unwinds newest-first -- not one interleaved sweep retiring twice per block.
    assert retirements == list(reversed(range(block_count))) * 2
    assert offloader.entries_after_retirement() == []
    assert offloader._backward_arms == {}


def test_the_backward_arm_gate_is_inert_until_a_pass_announces_something():
    """Every non-fused forward in the codebase announces nothing, and must keep
    the hook behaviour it has always had."""
    gate = FusedArmBackwardGate()
    gate.reset_backward_arms()

    # Unannounced: every firing retires, which is the single-invocation contract.
    assert gate.retire_backward_arm(0) is True
    assert gate.retire_backward_arm(0) is True

    gate.note_backward_arm(3)
    gate.note_backward_arm(3)
    assert gate.retire_backward_arm(3) is False
    assert gate.retire_backward_arm(3) is True
    # Consumed, so the block is back to the unannounced contract.
    assert gate.retire_backward_arm(3) is True
    # A block that was never announced is unaffected by its neighbour's count.
    gate.note_backward_arm(1)
    assert gate.retire_backward_arm(2) is True
    assert gate.retire_backward_arm(1) is True


def test_an_abandoned_graphs_announcements_do_not_survive_into_the_next_pass():
    """A step that raises between forward and backward would otherwise leave a
    count no hook consumes, and suppress the next pass's first retirement."""
    gate = FusedArmBackwardGate()
    gate.reset_backward_arms()
    gate.note_backward_arm(0)
    gate.note_backward_arm(0)

    gate.reset_backward_arms()
    assert gate.retire_backward_arm(0) is True


# ---------------------------------------------------------------------------
# Depth-0 teacher MATCHING: kohya-ss/musubi-tuner PR #1047, on our teacher
# ---------------------------------------------------------------------------


# Above this the teacher drops the privileged presentation, so a stub step whose
# base sigma is 0.5 is a TEACHING step at the default 0.75 and an ANCHOR step here.


# ---------------------------------------------------------------------------
# The dial: --h3_rollout_inactive_probability mixes supervised and plain steps
# ---------------------------------------------------------------------------


def test_the_keyframe_channel_is_the_endpoint_teacher_of_the_upstream_pr():
    """A teacher declaring fl2va over a t2va student reads the shared keyframe rows."""
    batch = {
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["t2va"])],
        H3_TEXT_HIDDEN_KEY: [torch.zeros(2, 4)],
    }
    entries = {
        H3_TEXT_HIDDEN_KEY: torch.ones(2, 4),
        H3_CONDITIONING_TASK_KEY: torch.tensor(H3_CONDITIONING_TASK_IDS["fl2va"]),
    }

    clone = teacher_batch(batch, entries)

    # The task is the whole difference: same latents, same everything else, but
    # the teacher's forward now reads the endpoints as conditioning.
    assert int(clone[H3_CONDITIONING_TASK_KEY][0]) == H3_CONDITIONING_TASK_IDS["fl2va"]
    assert int(batch[H3_CONDITIONING_TASK_KEY][0]) == H3_CONDITIONING_TASK_IDS["t2va"]


def test_a_teacher_may_not_differ_from_its_student_by_a_non_endpoint_task():
    """Any other task mismatch is a mispaired cache, not a privilege."""
    batch = {
        H3_CONDITIONING_TASK_KEY: [torch.tensor(H3_CONDITIONING_TASK_IDS["t2va"])],
        H3_TEXT_HIDDEN_KEY: [torch.zeros(2, 4)],
    }
    entries = {
        H3_TEXT_HIDDEN_KEY: torch.ones(2, 4),
        H3_CONDITIONING_TASK_KEY: torch.tensor(H3_CONDITIONING_TASK_IDS["ref2va"]),
    }

    with pytest.raises(ValueError, match="only an endpoint task"):
        teacher_batch(batch, entries)


# --------------------------------------------------------------------------
# Privilege is a difference between the two arms, and the teacher carries only
# the channel that was resolved.


def _student_text_cache(directory, *item_keys: str, controls: bool = False, task: str = "t2va"):
    """Plain student text caches for the items, the baseline every channel is judged against."""
    student_dir = directory / "student_text"
    student_dir.mkdir(exist_ok=True)
    return {key: str(_write_teacher_cache(student_dir, key, controls=controls, task=task)) for key in item_keys}


def test_qwen_controls_the_student_also_carries_are_not_a_privilege(tmp_path):
    teacher = _write_teacher_cache(tmp_path, "clip_a", controls=True)
    cache = H3RolloutTeacherCache({"clip_a": str(teacher)})

    # The same assets on both sides: the teacher knows nothing the student does not.
    with pytest.raises(ValueError, match="no privilege over the student"):
        cache.validate_privilege(student_text_paths=_student_text_cache(tmp_path, "clip_a", controls=True))
    # And the plain student is what makes the very same teacher a Qwen teacher.
    assert cache.validate_privilege(student_text_paths=_student_text_cache(tmp_path, "clip_a"))[0] == "qwen"
    # No student cache, no baseline: the channel is absent, not granted.
    with pytest.raises(ValueError, match="no privilege over the student"):
        cache.validate_privilege()


def test_an_endpoint_teacher_over_an_endpoint_student_is_not_a_keyframe_teacher(tmp_path):
    teacher = _write_teacher_cache(tmp_path, "clip_a", controls=False, task="fl2va")
    cache = H3RolloutTeacherCache({"clip_a": str(teacher)})

    # Both arms read the same keyframe rows: this is the self-distillation that
    # once trained to completion looking like a nine-fold better fit.
    with pytest.raises(ValueError, match="no privilege over the student"):
        cache.validate_privilege(student_text_paths=_student_text_cache(tmp_path, "clip_a", task="fl2va"))
    assert cache.validate_privilege(student_text_paths=_student_text_cache(tmp_path, "clip_a"))[0] == "keyframe"
    # Without the student's caches the channel cannot be assessed, and an
    # unassessable channel is an absent one rather than a granted one.
    with pytest.raises(ValueError, match="no privilege over the student"):
        cache.validate_privilege()


def test_the_teacher_carries_only_the_resolved_channels_presentation(tmp_path):
    # One cache that could serve two channels: Qwen assets AND an endpoint task.
    teacher = _write_teacher_cache(tmp_path, "clip_a", controls=True, task="fl2va")
    cache = H3RolloutTeacherCache({"clip_a": str(teacher)})
    student = _student_text_cache(tmp_path, "clip_a")

    assert cache.validate_privilege(student_text_paths=student, channel="qwen")[0] == "qwen"
    qwen_entries = cache.entries("clip_a")
    assert H3_QWEN_CONTROL_VISUALS_KEY in qwen_entries
    assert H3_CONDITIONING_TASK_KEY not in qwen_entries

    assert cache.validate_privilege(student_text_paths=student, channel="keyframe")[0] == "keyframe"
    keyframe_entries = cache.entries("clip_a")
    assert H3_CONDITIONING_TASK_KEY in keyframe_entries
    assert H3_QWEN_CONTROL_VISUALS_KEY not in keyframe_entries


def test_the_fused_teacher_refuses_block_sparse_attention(tmp_path):
    trainer = MiniMaxH3NetworkTrainer()
    with pytest.raises(ValueError, match="block-sparse attention plan"):
        trainer.handle_model_specific_args(
            _flag_args(
                *_ROLLOUT_FLAGS,
                "--h3_rollout_fused_teacher",
                "--h3_block_sparse_kv_fraction",
                "0.5",
                teacher=_teacher_file(tmp_path),
            )
        )


def test_the_null_anchor_is_left_out_of_the_reported_average(tmp_path):
    teacher = _teacher_file(tmp_path)
    _, plain_loss, _, _ = _run_step(*_ROLLOUT_FLAGS, teacher=teacher, active=False)
    _, anchored_loss, anchored_metrics, _ = _run_step(
        *_ROLLOUT_FLAGS, "--h3_guidance_null_anchor_weight", "1.0", teacher=teacher, active=False
    )

    # The anchor changes what is optimized and not what is reported, exactly as
    # the preservation term does, so two runs stay comparable across the switch.
    assert anchored_metrics["loss/guidance_null_anchor"] >= 0.0
    assert anchored_metrics[LOSS_FOR_AVERAGE_KEY] == pytest.approx(float(plain_loss.detach()), rel=1e-6)
    assert float(anchored_loss.detach()) == pytest.approx(
        anchored_metrics[LOSS_FOR_AVERAGE_KEY] + anchored_metrics["loss/guidance_null_anchor"], rel=1e-5
    )


def test_the_null_anchor_alone_keeps_classic_block_swap_on_its_contract(tmp_path):
    """A run with nothing but the anchor: the frozen reference forward is a
    no-grad teacher pass and must run on the forward-only schedule, while the
    anchor's own student forward carries a graph and must run on the training
    layout its backward restores."""
    teacher = _teacher_file(tmp_path)
    args = _flag_args(*_ROLLOUT_FLAGS, "--h3_guidance_null_anchor_weight", "1.0", teacher=teacher)
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    trainer.dit_dtype = torch.float32
    trainer.blocks_to_swap = 4
    backend = _SwapRecordingBackend()
    trainer.backend = backend
    trainer._rollout_teacher = _StubTeacherCache()
    trainer._rollout_supervision_active = lambda accelerator, probability: False
    transformer = _SwapAwareScaleTransformer()
    video, batch = _rollout_batch()

    torch.manual_seed(0)
    trainer.process_batch(
        args,
        _FakeAccelerator(),
        transformer,
        _ToggleNetwork(transformer),
        batch,
        video,
        torch.ones_like(video),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )

    modes = [call["swap_mode"] for call in backend.calls]
    grads = [call["grad"] for call in backend.calls]
    assert any(grads) and not all(grads), "the step must run both a frozen and a graph-carrying forward"
    assert [mode == "inference" for mode in modes] == [not grad for grad in grads]
    # One bracket for the anchor section, with the student forward stepped back
    # into it, and the training layout restored before the main forward.
    assert transformer.swap_events == ["inference", "training", "inference", "training"]
    assert transformer.swap_mode == "training"


# ---------------------------------------------------------------------------
# --h3_rollout_field_floor: a LENGTH floor on the guidance field at the
# supervised states, taken against the FROZEN empty branch
# ---------------------------------------------------------------------------

_FLOOR_FLAGS = (*_ROLLOUT_FLAGS, "--h3_rollout_field_floor", "1.0")

# With the stub field the adapter scales the 0.25 velocity by 0.5 and the frozen
# base by 1.0, so at every supervised state the student's prompted-minus-frozen-
# empty field is (0.125 + 0.25) - (0.25 + 0.0) = 0.125 against the base's
# (0.25 + 0.25) - (0.25 + 0.0) = 0.25: a length ratio of one half.
_STUB_FIELD_RATIO = ((0.5 * 0.25 + 0.25) - (1.0 * 0.25 + 0.0)) / ((1.0 * 0.25 + 0.25) - (1.0 * 0.25 + 0.0))


def _graph_parameters(loss):
    """The leaf parameters a loss depends on, found by walking its graph."""
    seen, leaves, stack = set(), [], [loss.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        variable = getattr(node, "variable", None)
        if variable is not None:
            leaves.append(variable)
        stack.extend(child for child, _ in node.next_functions)
    return leaves


def test_field_floor_requires_the_switch():
    with pytest.raises(ValueError, match="requires --h3_rollout_supervision"):
        MiniMaxH3NetworkTrainer()._validate_rollout_args(_flag_args("--h3_rollout_field_floor", "1.0"))


def test_field_floor_rejects_a_negative_weight(tmp_path):
    args = _flag_args(*_ROLLOUT_FLAGS, "--h3_rollout_field_floor", "-1", teacher=_teacher_file(tmp_path))
    with pytest.raises(ValueError, match="finite and non-negative"):
        MiniMaxH3NetworkTrainer()._validate_rollout_args(args)


def test_field_floor_adds_two_frozen_forwards_per_supervised_state(tmp_path):
    backend, loss, _, _ = _run_step(
        *_FLOOR_FLAGS, "--h3_rollout_steps", "2", "--h3_rollout_window", "2", teacher=_teacher_file(tmp_path)
    )
    loss.backward()

    # One data forward, two no-grad rollout steps, then per supervised state the
    # graded student, the teacher, and the frozen base's prompted and empty arms.
    assert len(backend.calls) == 1 + 2 + 2 * 4
    window = backend.calls[3:]
    assert [call["grad"] for call in window] == [True, False, False, False] * 2
    assert [call["adapter"] for call in window] == [True, False, False, False] * 2
    assert [call["conditioning"] for call in window] == ["prompt", "prompt", "prompt", "empty"] * 2
    student_signature = float(torch.full((4, 8), 0.25).mean())
    teacher_signature = float(torch.full((4, 8), 0.75).mean())
    assert [call["signature"] for call in window[:4]] == [student_signature, teacher_signature, student_signature, 0.0]
    # All four arms of a sub-step describe one and the same state.
    for sub_step in (window[:4], window[4:]):
        assert all(torch.equal(call["video_state"], sub_step[0]["video_state"]) for call in sub_step)


def test_field_floor_penalises_a_shortened_field_and_leaves_the_reported_loss_alone(tmp_path):
    teacher = _teacher_file(tmp_path)
    _, plain_loss, plain_metrics, _ = _run_step(*_ROLLOUT_FLAGS, teacher=teacher)
    _, floored_loss, metrics, _ = _run_step(*_FLOOR_FLAGS, teacher=teacher)

    assert metrics["h3/rollout_field_ratio"] == pytest.approx(_STUB_FIELD_RATIO)
    assert metrics["loss/rollout_field_floor"] == pytest.approx((1.0 - _STUB_FIELD_RATIO) ** 2)
    assert float(floored_loss.detach()) == pytest.approx(float(plain_loss.detach()) + (1.0 - _STUB_FIELD_RATIO) ** 2)
    # An auxiliary term: the running average is still the data loss of a run
    # without the flag.
    assert metrics[LOSS_FOR_AVERAGE_KEY] == pytest.approx(plain_metrics[LOSS_FOR_AVERAGE_KEY])


def test_field_floor_pulls_the_field_longer_not_shorter(tmp_path):
    """The adapter's scale is its only parameter and a larger scale is a longer
    field, so the floor's gradient on it must point down: more scale, less loss."""
    teacher = _teacher_file(tmp_path)
    _, plain_loss, _, _ = _run_step(*_ROLLOUT_FLAGS, teacher=teacher)
    _, floored_loss, _, _ = _run_step(*_FLOOR_FLAGS, teacher=teacher)
    (plain_scale,) = _graph_parameters(plain_loss)
    (floored_scale,) = _graph_parameters(floored_loss)
    plain_loss.backward()
    floored_loss.backward()
    assert float(floored_scale.grad) < float(plain_scale.grad)


def test_field_floor_is_zero_when_the_field_is_no_shorter(tmp_path, monkeypatch):
    class _LongFieldTransformer(_ScaleTransformer):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.5))

    monkeypatch.setitem(globals(), "_ScaleTransformer", _LongFieldTransformer)
    teacher = _teacher_file(tmp_path)
    _, plain_loss, _, _ = _run_step(*_ROLLOUT_FLAGS, teacher=teacher)
    _, floored_loss, metrics, _ = _run_step(*_FLOOR_FLAGS, teacher=teacher)

    assert metrics["h3/rollout_field_ratio"] == pytest.approx(1.5)
    assert metrics["loss/rollout_field_floor"] == 0.0
    assert float(floored_loss.detach()) == pytest.approx(float(plain_loss.detach()))


def test_field_floor_rides_the_fused_pass(tmp_path):
    teacher = _teacher_file(tmp_path)
    naive_backend, naive_loss, naive_metrics, _ = _run_step(*_FLOOR_FLAGS, teacher=teacher, backend=_FusedRolloutBackend())
    fused_backend, fused_loss, fused_metrics, _ = _run_step(
        *_FLOOR_FLAGS, "--h3_rollout_fused_teacher", teacher=teacher, backend=_FusedRolloutBackend()
    )

    assert naive_backend.fused_calls == 0
    # One fused call per supervised state carries all four arms.
    assert fused_backend.fused_calls == 1
    torch.testing.assert_close(fused_loss, naive_loss, rtol=0, atol=0)
    assert fused_metrics == naive_metrics
    for key in ("signature", "grad", "adapter", "conditioning"):
        assert [call[key] for call in fused_backend.calls] == [call[key] for call in naive_backend.calls]


def test_field_floor_reports_zero_on_an_inactive_step(tmp_path):
    _, _, metrics, _ = _run_step(*_FLOOR_FLAGS, teacher=_teacher_file(tmp_path), active=False)
    assert metrics["loss/rollout_field_floor"] == 0.0
    assert metrics["h3/rollout_field_ratio"] == 0.0


def test_field_floor_needs_the_empty_presentation(tmp_path):
    video, batch = _rollout_batch()
    batch.pop(H3_EMPTY_TEXT_HIDDEN_KEY)
    batch.pop(H3_EMPTY_TEXT_TOKEN_TAGS_KEY)
    with pytest.raises(ValueError, match="cache_guidance_empty"):
        _run_step(*_FLOOR_FLAGS, teacher=_teacher_file(tmp_path), batch=(video, batch))


def test_field_floor_is_recorded_in_adapter_metadata(tmp_path):
    args = _flag_args(*_FLOOR_FLAGS, teacher=_teacher_file(tmp_path))
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    assert trainer.extra_metadata(args)["ss_h3_rollout_field_floor"] == "1.0"
    assert MiniMaxH3NetworkTrainer().extra_metadata(_flag_args())["ss_h3_rollout_field_floor"] == "0.0"


def test_field_floor_has_a_finite_gradient_at_an_exactly_zero_field(tmp_path, monkeypatch):
    """The floor bites hardest where the student's field has vanished, and that is
    where sqrt(mean(x^2)) would have handed back NaN."""

    class _ZeroFieldTransformer(_ScaleTransformer):
        def __init__(self):
            super().__init__()
            # 0 * 0.25 + 0.25 (prompted) == 1.0 * 0.25 + 0.0 (frozen empty): g' == e.
            self.scale = torch.nn.Parameter(torch.tensor(0.0))

    monkeypatch.setitem(globals(), "_ScaleTransformer", _ZeroFieldTransformer)
    _, loss, metrics, _ = _run_step(*_FLOOR_FLAGS, teacher=_teacher_file(tmp_path))
    assert metrics["h3/rollout_field_ratio"] == 0.0
    assert metrics["loss/rollout_field_floor"] == pytest.approx(1.0)
    (scale,) = _graph_parameters(loss)
    loss.backward()
    assert torch.isfinite(scale.grad)


def test_field_length_ratio_drops_a_sample_with_no_authored_element():
    adapted = torch.ones(2, 3, 4)
    base = torch.full((2, 3, 4), 2.0)
    mask = torch.stack([torch.ones(3, 4), torch.zeros(3, 4)])
    ratio, valid = MiniMaxH3NetworkTrainer._field_length_ratio(adapted, base, mask)
    assert ratio[0].item() == pytest.approx(0.5)
    assert valid.tolist() == [True, False]
    # And with no mask every sample is scored.
    _, all_valid = MiniMaxH3NetworkTrainer._field_length_ratio(adapted, base, None)
    assert all_valid.tolist() == [True, True]


def test_field_floor_warns_without_the_null_anchor(tmp_path, caplog):
    args = _flag_args(*_FLOOR_FLAGS, teacher=_teacher_file(tmp_path))
    with caplog.at_level(logging.WARNING):
        MiniMaxH3NetworkTrainer()._validate_rollout_args(args)
    assert any("null_anchor" in record.getMessage() for record in caplog.records)
    caplog.clear()
    anchored = _flag_args(*_FLOOR_FLAGS, "--h3_guidance_null_anchor_weight", "1.0", teacher=_teacher_file(tmp_path))
    with caplog.at_level(logging.WARNING):
        MiniMaxH3NetworkTrainer()._validate_rollout_args(anchored)
    assert not any("null_anchor" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# The floor's direction and sigma ceiling
# ---------------------------------------------------------------------------


def test_field_projection_ratio_scores_a_misdirected_field_short():
    base = torch.zeros(1, 4)
    base[0, 0] = 2.0
    teacher = torch.zeros(1, 4)
    teacher[0, 0] = 5.0  # same direction as the base, different length
    aligned = torch.zeros(1, 4)
    aligned[0, 0] = 1.0
    orthogonal = torch.zeros(1, 4)
    orthogonal[0, 1] = 1.0
    ratio, valid = MiniMaxH3NetworkTrainer._field_projection_ratio(aligned, base, teacher, None)
    assert ratio.item() == pytest.approx(0.5) and valid.item()
    ratio, valid = MiniMaxH3NetworkTrainer._field_projection_ratio(orthogonal, base, teacher, None)
    # As long as the aligned one, and worth nothing along the teacher.
    assert ratio.item() == pytest.approx(0.0) and valid.item()
    # And the gradient is the unit teacher direction, not the student's own.
    adapted = orthogonal.clone().requires_grad_(True)
    ratio, _ = MiniMaxH3NetworkTrainer._field_projection_ratio(adapted, base, teacher, None)
    ratio.sum().backward()
    torch.testing.assert_close(adapted.grad, torch.tensor([[0.5, 0.0, 0.0, 0.0]]))


def test_field_projection_ratio_is_invalid_where_a_reference_vanishes():
    adapted = torch.ones(2, 4)
    base = torch.ones(2, 4)
    teacher = torch.stack([torch.ones(4), torch.zeros(4)])
    _, valid = MiniMaxH3NetworkTrainer._field_projection_ratio(adapted, base, teacher, None)
    assert valid.tolist() == [True, False]


def test_teacher_direction_matches_self_on_an_aligned_stub(tmp_path):
    """The stub field is a multiple of the ones vector for every arm, so the
    projection onto the teacher IS the length: both directions give one number."""
    teacher = _teacher_file(tmp_path)
    _, self_loss, self_metrics, _ = _run_step(*_FLOOR_FLAGS, teacher=teacher)
    _, teacher_loss, teacher_metrics, _ = _run_step(*_FLOOR_FLAGS, "--h3_rollout_field_floor_direction", "teacher", teacher=teacher)
    assert teacher_metrics["h3/rollout_field_ratio"] == pytest.approx(self_metrics["h3/rollout_field_ratio"])
    assert float(teacher_loss.detach()) == pytest.approx(float(self_loss.detach()))
    teacher_loss.backward()


def test_field_floor_sigma_ceiling_drops_the_states_above_it(tmp_path):
    teacher = _teacher_file(tmp_path)
    # A rollout state sits at a shifted video sigma near 1; a ceiling of 0.05 leaves
    # nothing to score, so the floor is zero and the ratio reports zero.
    _, loss, metrics, _ = _run_step(*_FLOOR_FLAGS, "--h3_rollout_field_floor_sigma_max", "0.05", teacher=teacher)
    _, plain_loss, _, _ = _run_step(*_ROLLOUT_FLAGS, teacher=teacher)
    assert metrics["loss/rollout_field_floor"] == 0.0
    assert metrics["h3/rollout_field_ratio"] == 0.0
    assert float(loss.detach()) == pytest.approx(float(plain_loss.detach()))


def test_field_floor_shape_flags_require_the_floor(tmp_path):
    for flag in (("--h3_rollout_field_floor_direction", "teacher"), ("--h3_rollout_field_floor_sigma_max", "0.9")):
        args = _flag_args(*_ROLLOUT_FLAGS, *flag, teacher=_teacher_file(tmp_path))
        with pytest.raises(ValueError, match="need --h3_rollout_field_floor above 0"):
            MiniMaxH3NetworkTrainer()._validate_rollout_args(args)
    with pytest.raises(ValueError, match="requires --h3_rollout_supervision"):
        MiniMaxH3NetworkTrainer()._validate_rollout_args(_flag_args("--h3_rollout_field_floor_direction", "teacher"))
    bad = _flag_args(*_FLOOR_FLAGS, "--h3_rollout_field_floor_sigma_max", "0", teacher=_teacher_file(tmp_path))
    with pytest.raises(ValueError, match="sigma_max"):
        MiniMaxH3NetworkTrainer()._validate_rollout_args(bad)


def test_field_floor_shape_is_recorded_in_adapter_metadata(tmp_path):
    args = _flag_args(
        *_FLOOR_FLAGS,
        "--h3_rollout_field_floor_direction",
        "teacher",
        "--h3_rollout_field_floor_sigma_max",
        "0.9",
        teacher=_teacher_file(tmp_path),
    )
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    metadata = trainer.extra_metadata(args)
    assert metadata["ss_h3_rollout_field_floor_direction"] == "teacher"
    assert metadata["ss_h3_rollout_field_floor_sigma_max"] == "0.9"


# ---------------------------------------------------------------------------
# Stability trio: prompt-only adapter, measured-variance weighting, stop band
# ---------------------------------------------------------------------------

_NORMALIZED_FLAGS = (
    "--h3_guidance_distillation_scale",
    "3.0",
    "--h3_guidance_loss_form",
    "normalized",
    "--h3_guidance_null_source",
    "live",
)


def test_prompt_only_adapter_runs_the_empty_forward_frozen():
    """With the flag, the 'live' empty branch IS the frozen one: the empty forward
    is made with the adapter off and the adapter is back on afterwards."""
    live_backend, live_loss, _, _ = _run_step(*_NORMALIZED_FLAGS)
    only_backend, only_loss, _, trainer = _run_step(*_NORMALIZED_FLAGS, "--h3_adapter_prompt_only")
    live_empty = [c for c in live_backend.calls if c["conditioning"] == "empty"]
    only_empty = [c for c in only_backend.calls if c["conditioning"] == "empty"]
    assert len(live_empty) == len(only_empty) == 1
    assert live_empty[0]["adapter"] is True
    assert only_empty[0]["adapter"] is False
    # The prompted forward keeps the adapter, and it is live again on the way out.
    assert [c["adapter"] for c in only_backend.calls if c["conditioning"] == "prompt"] == [True]
    assert getattr(trainer._adapter_network.transformer, "adapter_enabled", True) is True
    # And the objective changed, because the empty branch did.
    assert float(only_loss.detach()) != pytest.approx(float(live_loss.detach()))


def test_prompt_only_adapter_equals_the_frozen_null_source():
    _, frozen_loss, _, _ = _run_step(*_NORMALIZED_FLAGS[:4], "--h3_guidance_null_source", "frozen")
    _, only_loss, _, _ = _run_step(*_NORMALIZED_FLAGS, "--h3_adapter_prompt_only")
    assert float(only_loss.detach()) == pytest.approx(float(frozen_loss.detach()))


def test_prompt_only_adapter_leaves_an_already_frozen_bracket_alone(tmp_path):
    """Inside a frozen bracket (the teacher, the floor's base_empty arm) the
    adapter is already off; the wrapper must not switch it back on in between."""
    backend, loss, _, _ = _run_step(*_FLOOR_FLAGS, "--h3_adapter_prompt_only", teacher=_teacher_file(tmp_path))
    loss.backward()
    window = backend.calls[3:]
    # student (prompt, on), teacher (prompt, off), base prompt (off), base empty (off)
    assert [c["adapter"] for c in window] == [True, False, False, False]
    assert [c["conditioning"] for c in window] == ["prompt", "prompt", "prompt", "empty"]


def test_prompt_only_adapter_rejects_the_anchor_and_caption_dropout():
    with pytest.raises(ValueError, match="identically zero"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(
            _flag_args("--h3_adapter_prompt_only", "--h3_guidance_null_anchor_weight", "1.0")
        )
    with pytest.raises(ValueError, match="caption_dropout_rate 0"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(
            _flag_args("--h3_adapter_prompt_only", "--h3_caption_dropout_rate", "0.1")
        )


def _curve_file(tmp_path):
    rows = []
    for modality, points in (
        ("video", [(0.2, 0.30, 0.20), (0.6, 0.60, 0.40), (0.9, 1.00, 0.60)]),
        ("audio", [(0.3, 0.50, 0.40), (0.8, 0.80, 0.50)]),
    ):
        for sigma, raw, g in points:
            rows.append({"modality": modality, "sigma": sigma, "raw_mean": raw, "g_mean": g, "base_sigma": sigma})
    path = tmp_path / "gcurve.json"
    path.write_text(json.dumps({"summary": rows}), encoding="utf-8")
    return path


def test_measured_variance_weights_are_inverse_variance_normalised_and_capped(tmp_path):
    from musubi_tuner.minimax_h3_train_network import interpolate_curve, load_measured_variance_curve

    curve = load_measured_variance_curve(str(_curve_file(tmp_path)), weight_max=4.0)
    sigmas, weights = curve["video"]
    variance = torch.tensor([0.30**2 - 0.20**2, 0.60**2 - 0.40**2, 1.00**2 - 0.60**2])
    expected = 1.0 / variance
    expected = expected / expected.mean()
    torch.testing.assert_close(weights, expected)
    assert weights.mean().item() == pytest.approx(1.0)
    # Flat beyond the ends, linear between.
    torch.testing.assert_close(interpolate_curve(sigmas, weights, torch.tensor([0.0, 0.2])), weights[:1].repeat(2))
    mid = interpolate_curve(sigmas, weights, torch.tensor([0.4]))
    torch.testing.assert_close(mid, ((weights[0] + weights[1]) / 2).reshape(1))
    # A bucket that would dominate is capped and the rest renormalised to mean 1.
    tight = load_measured_variance_curve(str(_curve_file(tmp_path)), weight_max=1.2)
    assert tight["video"][1].max().item() <= 1.2 + 1e-6
    assert tight["video"][1].mean().item() < 1.0


def test_measured_variance_weighting_reaches_the_sample_weight_per_modality(tmp_path):
    args = _flag_args("--h3_measured_variance_weighting", str(_curve_file(tmp_path)))
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    video = trainer._sample_weight(args, torch.tensor([0.2, 0.9]))
    audio = trainer._sample_weight(args, torch.tensor([0.3, 0.8]), modality="audio")
    assert video is not None and audio is not None
    # Video: the least noisy bucket weighs most.
    assert video[0] > video[1]
    # Audio reads its own rows, which order the other way round here.
    assert audio[0] > audio[1]
    # Composes with a scheme instead of replacing it.
    cosmap = _flag_args("--h3_measured_variance_weighting", str(_curve_file(tmp_path)), "--weighting_scheme", "cosmap")
    trainer.handle_model_specific_args(cosmap)
    plain = MiniMaxH3NetworkTrainer()._sample_weight(_flag_args("--weighting_scheme", "cosmap"), torch.tensor([0.2, 0.9]))
    torch.testing.assert_close(trainer._sample_weight(cosmap, torch.tensor([0.2, 0.9])), plain * video)


def test_measured_variance_weighting_rejects_a_missing_or_videoless_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(
            _flag_args("--h3_measured_variance_weighting", str(tmp_path / "nope.json"))
        )
    bad = tmp_path / "audio_only.json"
    bad.write_text(json.dumps({"summary": [{"modality": "audio", "sigma": 0.5, "raw_mean": 1.0, "g_mean": 0.5}]}))
    with pytest.raises(ValueError, match="no video rows"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_flag_args("--h3_measured_variance_weighting", str(bad)))


def test_rollout_stop_min_bounds_the_stop_draw(tmp_path):
    args = _flag_args(*_ROLLOUT_FLAGS, "--h3_rollout_stop_min", "0.7", teacher=_teacher_file(tmp_path))
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    torch.manual_seed(0)
    draws = [trainer._draw_rollout_stop_sigma(args) for _ in range(200)]
    assert min(draws) >= 0.7
    assert max(draws) > 0.9
    plain = _flag_args(*_ROLLOUT_FLAGS, teacher=_teacher_file(tmp_path))
    other = MiniMaxH3NetworkTrainer()
    other.handle_model_specific_args(plain)
    torch.manual_seed(0)
    assert min(other._draw_rollout_stop_sigma(plain) for _ in range(200)) < 0.7


def test_rollout_stop_min_is_validated_and_recorded(tmp_path):
    with pytest.raises(ValueError, match="requires --h3_rollout_supervision"):
        MiniMaxH3NetworkTrainer()._validate_rollout_args(_flag_args("--h3_rollout_stop_min", "0.5"))
    with pytest.raises(ValueError, match="stop_min"):
        MiniMaxH3NetworkTrainer()._validate_rollout_args(
            _flag_args(*_ROLLOUT_FLAGS, "--h3_rollout_stop_min", "1.0", teacher=_teacher_file(tmp_path))
        )
    args = _flag_args(*_ROLLOUT_FLAGS, "--h3_rollout_stop_min", "0.7", "--h3_adapter_prompt_only", teacher=_teacher_file(tmp_path))
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    metadata = trainer.extra_metadata(args)
    assert metadata["ss_h3_rollout_stop_min"] == "0.7"
    assert metadata["ss_h3_adapter_prompt_only"] == "True"
    assert metadata["ss_h3_measured_variance_weighting"] == "none"


def test_prompt_only_adapter_respects_a_raw_set_enabled_bracket():
    """A frozen bracket made with the network's own set_enabled(False), not the
    tracking toggle, must still be left alone by the wrapper."""
    _, _, _, trainer = _run_step(*_NORMALIZED_FLAGS, "--h3_adapter_prompt_only")
    network = trainer._adapter_network
    backend = trainer.backend
    video, batch = _rollout_batch()
    inputs = trainer._rollout_state(
        video=video,
        audio=None,
        video_sigma=torch.tensor([0.5]),
        audio_sigma=torch.tensor([0.5]),
    )
    calls_before = len(backend.calls)
    network.set_enabled(False)
    try:
        with torch.no_grad():
            trainer._predict(_FakeAccelerator(), network.transformer, batch, inputs, conditioning="empty")
        # Still off after the empty forward: the wrapper did not restore it.
        assert network.is_enabled() is False
    finally:
        network.set_enabled(True)
    assert backend.calls[calls_before]["adapter"] is False


def test_measured_variance_weight_max_is_recorded(tmp_path):
    args = _flag_args("--h3_measured_variance_weighting", str(_curve_file(tmp_path)), "--h3_measured_variance_weight_max", "2.5")
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    assert trainer.extra_metadata(args)["ss_h3_measured_variance_weight_max"] == "2.5"


# ---------------------------------------------------------------------------
# Teacher-walked prefix and the field cap
# ---------------------------------------------------------------------------


def test_teacher_prefix_walks_the_no_grad_steps_frozen_and_privileged(tmp_path):
    teacher = _teacher_file(tmp_path)
    backend, loss, _, _ = _run_step(
        *_ROLLOUT_FLAGS, "--h3_rollout_steps", "2", "--h3_rollout_window", "1", "--h3_rollout_prefix", "teacher", teacher=teacher
    )
    loss.backward()
    prefix = backend.calls[1:3]
    teacher_signature = float(torch.full((4, 8), 0.75).mean())
    # Frozen, privileged, no graph -- and the adapter is live again for the window.
    assert [c["adapter"] for c in prefix] == [False, False]
    assert [c["signature"] for c in prefix] == [teacher_signature, teacher_signature]
    assert [c["grad"] for c in prefix] == [False, False]
    window = backend.calls[3:]
    assert [c["adapter"] for c in window] == [True, False]
    assert len(backend.calls) == 1 + 2 + 2


def test_student_prefix_is_the_default_and_unchanged(tmp_path):
    teacher = _teacher_file(tmp_path)
    backend, _, _, _ = _run_step(*_ROLLOUT_FLAGS, "--h3_rollout_steps", "2", teacher=teacher)
    student_signature = float(torch.full((4, 8), 0.25).mean())
    assert [c["adapter"] for c in backend.calls[1:3]] == [True, True]
    assert [c["signature"] for c in backend.calls[1:3]] == [student_signature, student_signature]


def test_field_cap_penalises_an_over_long_field(tmp_path, monkeypatch):
    class _LongFieldTransformer(_ScaleTransformer):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.5))

    monkeypatch.setitem(globals(), "_ScaleTransformer", _LongFieldTransformer)
    teacher = _teacher_file(tmp_path)
    _, uncapped_loss, uncapped, _ = _run_step(*_FLOOR_FLAGS, teacher=teacher)
    _, capped_loss, capped, _ = _run_step(*_FLOOR_FLAGS, "--h3_rollout_field_cap", "1.2", teacher=teacher)
    assert uncapped["loss/rollout_field_floor"] == 0.0
    assert capped["h3/rollout_field_ratio"] == pytest.approx(1.5)
    assert capped["loss/rollout_field_floor"] == pytest.approx((1.5 - 1.2) ** 2)
    assert float(capped_loss.detach()) == pytest.approx(float(uncapped_loss.detach()) + (1.5 - 1.2) ** 2)


def test_field_cap_is_silent_inside_the_band(tmp_path):
    teacher = _teacher_file(tmp_path)
    _, plain_loss, plain, _ = _run_step(*_FLOOR_FLAGS, teacher=teacher)
    _, capped_loss, capped, _ = _run_step(*_FLOOR_FLAGS, "--h3_rollout_field_cap", "1.2", teacher=teacher)
    # The stub field sits at ratio one half: below the floor, inside the cap.
    assert capped["loss/rollout_field_floor"] == pytest.approx(plain["loss/rollout_field_floor"])
    assert float(capped_loss.detach()) == pytest.approx(float(plain_loss.detach()))


def test_field_cap_and_prefix_are_validated_and_recorded(tmp_path):
    teacher = _teacher_file(tmp_path)
    with pytest.raises(ValueError, match="need --h3_rollout_field_floor above 0"):
        MiniMaxH3NetworkTrainer()._validate_rollout_args(
            _flag_args(*_ROLLOUT_FLAGS, "--h3_rollout_field_cap", "1.2", teacher=teacher)
        )
    with pytest.raises(ValueError, match="above 1"):
        MiniMaxH3NetworkTrainer()._validate_rollout_args(
            _flag_args(*_FLOOR_FLAGS, "--h3_rollout_field_cap", "0.9", teacher=teacher)
        )
    with pytest.raises(ValueError, match="requires --h3_rollout_supervision"):
        MiniMaxH3NetworkTrainer()._validate_rollout_args(_flag_args("--h3_rollout_prefix", "teacher"))
    args = _flag_args(*_FLOOR_FLAGS, "--h3_rollout_field_cap", "1.25", "--h3_rollout_prefix", "teacher", teacher=teacher)
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    metadata = trainer.extra_metadata(args)
    assert metadata["ss_h3_rollout_field_cap"] == "1.25"
    assert metadata["ss_h3_rollout_prefix"] == "teacher"


# ---------------------------------------------------------------------------
# Sigma cap on the guidance target, adapter EMA, multiplier sweep
# ---------------------------------------------------------------------------


def test_guidance_scale_sigma_max_caps_the_schedule():
    from musubi_tuner.minimax_h3.training import guidance_scale_for_sigma

    sigma = torch.tensor([0.5, 0.95])
    torch.testing.assert_close(guidance_scale_for_sigma(4.0, sigma, "constant"), torch.tensor([4.0, 4.0]))
    torch.testing.assert_close(guidance_scale_for_sigma(4.0, sigma, "constant", sigma_max=0.9), torch.tensor([4.0, 1.0]))
    torch.testing.assert_close(guidance_scale_for_sigma(4.0, sigma, "sigma", sigma_max=0.9), torch.tensor([1.0 + 3.0 * 0.5, 1.0]))
    per_sample = guidance_scale_for_sigma(torch.tensor([2.0, 6.0]), sigma, "constant", sigma_max=0.9)
    torch.testing.assert_close(per_sample, torch.tensor([2.0, 1.0]))


def test_guidance_scale_sigma_max_is_validated_and_recorded():
    with pytest.raises(ValueError, match="sigma_max"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_flag_args("--h3_guidance_scale_sigma_max", "0"))
    trainer = MiniMaxH3NetworkTrainer()
    args = _flag_args("--h3_guidance_scale_sigma_max", "0.9")
    trainer.handle_model_specific_args(args)
    assert trainer.extra_metadata(args)["ss_h3_guidance_scale_sigma_max"] == "0.9"


class _EmaNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.saved = []

    def save_weights(self, path, dtype, metadata):
        self.saved.append((path, dtype, dict(metadata), self.weight.detach().clone()))


def test_adapter_ema_tracks_the_parameters_and_saves_on_schedule(tmp_path):
    args = _flag_args(
        "--h3_adapter_ema_decay", "0.5", "--save_every_n_steps", "2", "--output_dir", str(tmp_path), "--output_name", "arm"
    )
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    network = _EmaNetwork()
    accelerator = _FakeAccelerator()
    accelerator.is_main_process = True
    # Step 0 (global_step 0 -> step 1): the EMA is initialised at the live value.
    with torch.no_grad():
        network.weight.fill_(1.0)
    trainer.on_post_optimizer_step(args, accelerator, network, None, True, 0)
    torch.testing.assert_close(trainer._adapter_ema["weight"], torch.tensor([1.0, 1.0]))
    # Step 1 (-> step 2): decay 0.5 between 1.0 and the new value 3.0 -> 2.0, and a save.
    with torch.no_grad():
        network.weight.fill_(3.0)
    trainer.on_post_optimizer_step(args, accelerator, network, None, True, 1)
    torch.testing.assert_close(trainer._adapter_ema["weight"], torch.tensor([2.0, 2.0]))
    assert len(network.saved) == 1
    path, _dtype, metadata, weights_at_save = network.saved[0]
    assert path.endswith("arm-ema-step00000002.safetensors")
    assert metadata["ss_h3_adapter_ema"] == "True" and metadata["ss_steps"] == "2"
    # Saved with the EMA swapped in, and the live weights put back afterwards.
    torch.testing.assert_close(weights_at_save, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(network.weight.detach(), torch.tensor([3.0, 3.0]))
    # No update without a synchronised gradient step.
    trainer.on_post_optimizer_step(args, accelerator, network, None, False, 2)
    torch.testing.assert_close(trainer._adapter_ema["weight"], torch.tensor([2.0, 2.0]))


def test_adapter_ema_flags_are_validated():
    with pytest.raises(ValueError, match="ema_decay"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_flag_args("--h3_adapter_ema_decay", "1.0"))
    with pytest.raises(ValueError, match="requires --h3_adapter_ema_decay"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_flag_args("--h3_validate_ema"))


def test_validation_multipliers_are_parsed_and_validated():
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(_flag_args("--h3_validation_multipliers", "0.75, 1.0,1.25"))
    assert trainer._validation_multipliers == [0.75, 1.0, 1.25]
    with pytest.raises(ValueError, match="comma-separated numbers"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_flag_args("--h3_validation_multipliers", "a,b"))
    with pytest.raises(ValueError, match="non-negative"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_flag_args("--h3_validation_multipliers", "-1"))


def test_multiplier_sweep_scales_the_modules_pools_per_multiplier_and_restores(monkeypatch):
    """The sweep sets every LoRA module's multiplier, runs the batch pass, pools
    the data-state ratios under val/m<multiplier>/..., and leaves the modules and
    the live pools exactly as it found them."""

    class _Module:
        def __init__(self):
            self.multiplier = 1.0

    class _Network:
        def __init__(self):
            self.unet_loras = [_Module(), _Module()]
            self.text_encoder_loras = []

    trainer = MiniMaxH3NetworkTrainer()
    args = _flag_args("--h3_validation_multipliers", "0.5,1.0,2", "--h3_validation_rollout_probe", "3")
    trainer.handle_model_specific_args(args)
    trainer._validation_dataloader = [(0, {"clip": 1})]
    seen = []

    def fake_validate_batch(accelerator, a, transformer, dataset_index, batch, bins, observed, tasks, accumulators, seed):
        multiplier = network.unet_loras[0].multiplier
        seen.append((multiplier, a.h3_validation_rollout_probe, trainer._rollout_probe_done))
        trainer._velocity_error_ratios.setdefault(0, []).append(multiplier * 0.1)
        trainer._field_ratios.setdefault(0, []).append(multiplier)
        trainer._field_cosines.setdefault(0, []).append(0.9)
        trainer._prompted_drift_ratios.setdefault(0, []).append(multiplier * 0.2)

    monkeypatch.setattr(trainer, "_validate_batch", fake_validate_batch)
    network = _Network()
    trainer._reset_validation_pools()
    trainer._field_ratios = {0: [0.42]}  # a live-pass value that must survive the sweep
    metrics = trainer._validate_multiplier_sweep(_FakeAccelerator(), args, None, network, [0.5], (None,), ((None, "av"),), 0)

    assert [m for m, _, _ in seen] == [0.5, 2.0]
    # The rollout probe is off during the sweep and back afterwards.
    assert all(probe == 0 and done is None for _, probe, done in seen)
    assert args.h3_validation_rollout_probe == 3
    assert metrics["val/m0.5/field"] == pytest.approx(0.5)
    assert metrics["val/m2/field"] == pytest.approx(2.0)
    assert metrics["val/m0.5/velocity_err_rel"] == pytest.approx(0.05)
    assert metrics["val/m2/drift/prompted_rel"] == pytest.approx(0.4)
    assert [module.multiplier for module in network.unet_loras] == [1.0, 1.0]
    assert trainer._field_ratios == {0: [0.42]}


# ---------------------------------------------------------------------------
# Inference null guidance
# ---------------------------------------------------------------------------


def test_null_guidance_combines_two_forwards_with_the_adapter_off_for_the_null():
    from musubi_tuner.minimax_h3.inference import H3NullGuidance, _guided_transformer_call
    from musubi_tuner.minimax_h3.model import MiniMaxH3TransformerOutput

    events = []

    class _Transformer:
        adapter = True

        def __call__(self, **kwargs):
            hidden = kwargs["encoder_hidden_states"]
            events.append((float(hidden.mean()), self.adapter))
            # prompted (mean 1) with adapter -> 3; null (mean 0) without -> 1
            value = 3.0 if self.adapter else 1.0
            return MiniMaxH3TransformerOutput(video=torch.full((1, 2), value), audio=torch.full((1, 2), value * 2))

    transformer = _Transformer()

    def set_adapter(on):
        transformer.adapter = on
        events.append(("adapter", on))

    prompted = torch.ones(1, 4, 8)
    guidance = H3NullGuidance(scale=2.0, null_text_hidden=torch.zeros(4, 8), set_adapter=set_adapter)
    out = _guided_transformer_call(transformer, guidance, encoder_hidden_states=prompted, timestep=torch.tensor([0.5]))
    # null + 2 * (prompted - null) = 1 + 2 * (3 - 1) = 5; audio 2 + 2 * (6 - 2) = 10
    torch.testing.assert_close(out.video, torch.full((1, 2), 5.0))
    torch.testing.assert_close(out.audio, torch.full((1, 2), 10.0))
    assert events == [(1.0, True), ("adapter", False), (0.0, False), ("adapter", True)]
    # Without guidance: one forward, untouched.
    events.clear()
    out = _guided_transformer_call(transformer, None, encoder_hidden_states=prompted)
    torch.testing.assert_close(out.video, torch.full((1, 2), 3.0))
    assert events == [(1.0, True)]


def test_null_guidance_rejects_a_null_presentation_of_another_layout():
    from musubi_tuner.minimax_h3.inference import H3NullGuidance, _guided_transformer_call
    from musubi_tuner.minimax_h3.model import MiniMaxH3TransformerOutput

    class _Transformer:
        def __call__(self, **kwargs):
            return MiniMaxH3TransformerOutput(video=torch.zeros(1, 2), audio=torch.zeros(1, 2))

    guidance = H3NullGuidance(scale=2.0, null_text_hidden=torch.zeros(3, 8))
    with pytest.raises(ValueError, match="layout-preserving"):
        _guided_transformer_call(_Transformer(), guidance, encoder_hidden_states=torch.ones(1, 4, 8))


def test_generator_exposes_the_null_guidance_flag():
    from musubi_tuner.minimax_h3_generate_video import create_parser

    required = ["--model", "m.safetensors", "--prompt", "p", "--output", "o.mp4"]
    args = create_parser().parse_args([*required, "--h3_null_guidance_scale", "4.0"])
    assert args.h3_null_guidance_scale == 4.0
    assert create_parser().parse_args(required).h3_null_guidance_scale == 0.0


def test_adapter_ema_saves_on_the_initialising_step_when_scheduled(tmp_path):
    trainer = MiniMaxH3NetworkTrainer()
    args = _flag_args(
        "--h3_adapter_ema_decay", "0.5", "--save_every_n_steps", "1", "--output_dir", str(tmp_path), "--output_name", "eager"
    )
    trainer.handle_model_specific_args(args)
    network = _EmaNetwork()
    accelerator = _FakeAccelerator()
    accelerator.is_main_process = True
    trainer.on_post_optimizer_step(args, accelerator, network, None, True, 0)
    assert network.saved and network.saved[0][0].endswith("eager-ema-step00000001.safetensors")


# ---------------------------------------------------------------------------
# Sparse null anchor
# ---------------------------------------------------------------------------

_ANCHOR_FLAGS = (*_NORMALIZED_FLAGS[:4], "--h3_guidance_null_source", "frozen", "--h3_guidance_null_anchor_weight", "1.0")


def test_sparse_null_anchor_skips_the_forwards_on_an_inactive_step_and_rescales_an_active_one():
    _, dense_loss, dense_metrics, _ = _run_step(*_ANCHOR_FLAGS)
    trainer_off = None

    def run(active):
        nonlocal trainer_off
        args = _flag_args(*_ANCHOR_FLAGS, "--h3_guidance_null_anchor_probability", "0.25")
        trainer = MiniMaxH3NetworkTrainer()
        trainer.handle_model_specific_args(args)
        trainer.dit_dtype = torch.float32
        backend = _RolloutBackend()
        trainer.backend = backend
        trainer._null_anchor_probability_active = lambda accelerator, probability: active
        transformer = _ScaleTransformer()
        network = _ToggleNetwork(transformer)
        video, batch = _rollout_batch()
        torch.manual_seed(0)
        loss, metrics = trainer.process_batch(
            args,
            _FakeAccelerator(),
            transformer,
            network,
            batch,
            video,
            torch.ones_like(video),
            None,
            torch.float32,
            torch.float32,
            None,
            0,
        )
        return backend, loss, metrics

    off_backend, off_loss, off_metrics = run(False)
    on_backend, on_loss, on_metrics = run(True)
    # Inactive: no anchor forwards at all (data + frozen empty only), term reported as zero.
    assert off_metrics["loss/guidance_null_anchor"] == 0.0
    assert off_metrics["h3/null_anchor_active"] == 0.0
    assert len(off_backend.calls) == 2
    # Active: the two anchor forwards run and the term is the dense one divided by p.
    assert on_metrics["h3/null_anchor_active"] == 1.0
    assert len(on_backend.calls) == 4
    assert on_metrics["loss/guidance_null_anchor"] == pytest.approx(dense_metrics["loss/guidance_null_anchor"] / 0.25)
    assert "h3/null_anchor_active" not in dense_metrics


def test_sparse_null_anchor_is_validated_and_recorded():
    with pytest.raises(ValueError, match="anchor_probability"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(
            _flag_args(*_ANCHOR_FLAGS, "--h3_guidance_null_anchor_probability", "0")
        )
    with pytest.raises(ValueError, match="requires --h3_guidance_null_anchor_weight"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_flag_args("--h3_guidance_null_anchor_probability", "0.5"))
    args = _flag_args(*_ANCHOR_FLAGS, "--h3_guidance_null_anchor_probability", "0.5")
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    assert trainer.extra_metadata(args)["ss_h3_guidance_null_anchor_probability"] == "0.5"


# ---------------------------------------------------------------------------
# Null anchor at rollout states
# ---------------------------------------------------------------------------

_ROLLOUT_ANCHOR_FLAGS = (*_ROLLOUT_FLAGS, "--h3_rollout_null_anchor_weight", "1.0", "--h3_audio_loss_weight", "0.0")


def test_rollout_null_anchor_adds_a_graded_empty_arm_and_a_frozen_empty_arm(tmp_path):
    backend, loss, metrics, _ = _run_step(*_ROLLOUT_ANCHOR_FLAGS, teacher=_teacher_file(tmp_path))
    loss.backward()
    window = backend.calls[3:]
    # student (prompt, on, grad), student empty (empty, on, grad), teacher (off), base empty (off)
    assert [c["conditioning"] for c in window] == ["prompt", "empty", "prompt", "empty"]
    assert [c["adapter"] for c in window] == [True, True, False, False]
    assert [c["grad"] for c in window] == [True, True, False, False]
    assert all(torch.equal(c["video_state"], window[0]["video_state"]) for c in window)
    # Stub: student empty = 0.5 * 0.25 + 0 = 0.125, frozen empty = 0.25: MSE 0.015625 (audio silenced).
    assert metrics["loss/rollout_null_anchor"] == pytest.approx(0.125**2)


def test_rollout_null_anchor_is_an_auxiliary_term(tmp_path):
    teacher = _teacher_file(tmp_path)
    _, plain_loss, plain_metrics, _ = _run_step(*_ROLLOUT_FLAGS, "--h3_audio_loss_weight", "0.0", teacher=teacher)
    _, anchored_loss, metrics, _ = _run_step(*_ROLLOUT_ANCHOR_FLAGS, teacher=teacher)
    assert float(anchored_loss.detach()) == pytest.approx(float(plain_loss.detach()) + metrics["loss/rollout_null_anchor"])
    assert metrics[LOSS_FOR_AVERAGE_KEY] == pytest.approx(plain_metrics[LOSS_FOR_AVERAGE_KEY])


def test_rollout_null_anchor_rides_the_fused_pass_with_and_without_the_floor(tmp_path):
    teacher = _teacher_file(tmp_path)
    for extra in ((), ("--h3_rollout_field_floor", "0.3")):
        naive_backend, naive_loss, naive_metrics, _ = _run_step(
            *_ROLLOUT_ANCHOR_FLAGS, *extra, teacher=teacher, backend=_FusedRolloutBackend()
        )
        fused_backend, fused_loss, fused_metrics, _ = _run_step(
            *_ROLLOUT_ANCHOR_FLAGS, *extra, "--h3_rollout_fused_teacher", teacher=teacher, backend=_FusedRolloutBackend()
        )
        assert fused_backend.fused_calls == 1
        torch.testing.assert_close(fused_loss, naive_loss, rtol=0, atol=0)
        assert fused_metrics == naive_metrics
        assert sorted((c["conditioning"], c["adapter"], c["grad"]) for c in fused_backend.calls) == sorted(
            (c["conditioning"], c["adapter"], c["grad"]) for c in naive_backend.calls
        )


def test_rollout_null_anchor_reports_zero_when_inactive_and_is_validated(tmp_path):
    teacher = _teacher_file(tmp_path)
    _, _, metrics, _ = _run_step(*_ROLLOUT_ANCHOR_FLAGS, teacher=teacher, active=False)
    assert metrics["loss/rollout_null_anchor"] == 0.0
    with pytest.raises(ValueError, match="requires --h3_rollout_supervision"):
        MiniMaxH3NetworkTrainer()._validate_rollout_args(_flag_args("--h3_rollout_null_anchor_weight", "1.0"))
    with pytest.raises(ValueError, match="drop one of the two"):
        MiniMaxH3NetworkTrainer()._validate_rollout_args(
            _flag_args(*_ROLLOUT_ANCHOR_FLAGS, "--h3_adapter_prompt_only", teacher=teacher)
        )
    args = _flag_args(*_ROLLOUT_ANCHOR_FLAGS, teacher=teacher)
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    assert trainer.extra_metadata(args)["ss_h3_rollout_null_anchor_weight"] == "1.0"


# ---------------------------------------------------------------------------
# Audio guidance scale
# ---------------------------------------------------------------------------


def test_audio_guidance_scale_reaches_the_audio_target_only(monkeypatch):
    """With the audio scale at 1 the audio half is the plain data target while the
    video half keeps the guided form; the video pair is bit-identical either way."""
    from musubi_tuner import minimax_h3_train_network as module

    recorded = []
    original = module.contrastive_guidance_target

    def spy(target, empty, guidance_scale, *, audio_guidance_scale=None):
        recorded.append((guidance_scale, audio_guidance_scale))
        return original(target, empty, guidance_scale, audio_guidance_scale=audio_guidance_scale)

    monkeypatch.setattr(module, "contrastive_guidance_target", spy)
    flags = (
        "--h3_guidance_distillation_scale",
        "3.0",
        "--h3_guidance_loss_form",
        "contrastive",
        "--h3_guidance_loss_schedule",
        "constant",
    )
    _run_step(*flags)
    _run_step(*flags, "--h3_guidance_audio_scale", "1.0")
    same_video, same_audio = recorded[0]
    plain_video, plain_audio = recorded[1]
    torch.testing.assert_close(torch.as_tensor(same_video), torch.as_tensor(plain_video))
    assert float(torch.as_tensor(same_audio).reshape(-1)[0]) == pytest.approx(3.0)
    assert float(torch.as_tensor(plain_audio).reshape(-1)[0]) == pytest.approx(1.0)


def test_audio_guidance_scale_is_validated_and_recorded():
    with pytest.raises(ValueError, match="audio_scale"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_flag_args("--h3_guidance_audio_scale", "0.5"))
    trainer = MiniMaxH3NetworkTrainer()
    args = _flag_args("--h3_guidance_distillation_scale", "3.0", "--h3_guidance_audio_scale", "1.0")
    trainer.handle_model_specific_args(args)
    assert trainer.extra_metadata(args)["ss_h3_guidance_audio_scale"] == "1.0"
    assert MiniMaxH3NetworkTrainer().extra_metadata(_flag_args())["ss_h3_guidance_audio_scale"] == "same"


def test_contrastive_target_at_scale_one_is_the_plain_target_bit_for_bit():
    from musubi_tuner.minimax_h3.training import contrastive_guidance_target

    target = H3ModelPrediction(video=torch.randn(1, 4, 2, 3, 3), audio=torch.randn(1, 2, 6, 3))
    empty = H3ModelPrediction(video=torch.randn(1, 4, 2, 3, 3), audio=torch.randn(1, 2, 6, 3))
    guided = contrastive_guidance_target(target, empty, 3.0, audio_guidance_scale=1.0)
    assert guided.audio is target.audio
    torch.testing.assert_close(guided.video, empty.video + 3.0 * (target.video - empty.video))
    plain = contrastive_guidance_target(target, empty, 1.0)
    assert plain.video is target.video and plain.audio is target.audio


# ---------------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------------


def test_term_gradients_report_norms_and_cosines_and_leave_the_graph_usable(tmp_path):
    teacher = _teacher_file(tmp_path)
    _, loss, metrics, _ = _run_step(
        *_FLOOR_FLAGS,
        "--h3_guidance_null_anchor_weight",
        "1.0",
        *_NORMALIZED_FLAGS[:4],
        "--h3_guidance_null_source",
        "frozen",
        "--h3_term_grad_every",
        "1",
        teacher=teacher,
    )
    assert metrics["h3/grad/main_norm"] > 0
    assert "h3/grad/floor_norm" in metrics and "h3/grad/anchor_norm" in metrics
    assert -1.0 <= metrics["h3/grad/cos_main_anchor"] <= 1.0
    assert "h3/grad/cos_main_floor" in metrics
    # The step's own backward still works after the per-term grads.
    loss.backward()


def test_term_gradients_are_off_by_default_and_on_schedule(tmp_path):
    teacher = _teacher_file(tmp_path)
    _, _, metrics, _ = _run_step(*_FLOOR_FLAGS, teacher=teacher)
    assert not [k for k in metrics if k.startswith("h3/grad/")]


def test_adapter_stats_report_delta_norm_and_ema_distance():
    class _Lin(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_down = torch.nn.Linear(4, 2, bias=False)
            self.lora_up = torch.nn.Linear(2, 3, bias=False)
            self.scale = 0.5
            self.multiplier = 1.0
            torch.nn.init.ones_(self.lora_down.weight)
            torch.nn.init.ones_(self.lora_up.weight)

    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unet_loras = torch.nn.ModuleList([_Lin()])
            self.text_encoder_loras = []

    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(_flag_args())
    net = _Net()
    metrics = trainer._adapter_stat_metrics(_FakeAccelerator(), net)
    # up @ down = 2 everywhere on a 3x4 matrix, times scale 0.5 -> 1 everywhere: Frobenius sqrt(12).
    assert metrics["h3/adapter/delta_norm"] == pytest.approx(12**0.5)
    assert "h3/adapter/ema_rel_dist" not in metrics
    trainer._adapter_ema = {name: torch.zeros_like(p) for name, p in net.named_parameters()}
    metrics = trainer._adapter_stat_metrics(_FakeAccelerator(), net)
    assert metrics["h3/adapter/ema_rel_dist"] == pytest.approx(1.0)


def test_train_sigma_bins_report_the_step_bin_only():
    _, _, off, _ = _run_step()
    assert not [k for k in off if k.startswith("loss/video/bin")]
    _, _, on, _ = _run_step("--h3_train_sigma_bins")
    bins = [k for k in on if k.startswith("loss/video/bin")]
    assert len(bins) == 1
    assert on[bins[0]] == pytest.approx(on["loss/video"])


def test_validation_std_flag_is_parsed_and_diagnostic_ints_validated():
    args = _flag_args("--h3_validation_std", "--h3_term_grad_every", "50", "--h3_adapter_stats_every", "10")
    MiniMaxH3NetworkTrainer().handle_model_specific_args(args)
    with pytest.raises(ValueError, match="non-negative"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_flag_args("--h3_term_grad_every", "-1"))


def test_rollout_teacher_value_metrics_appear_with_the_floor(tmp_path):
    _, _, metrics, _ = _run_step(*_FLOOR_FLAGS, teacher=_teacher_file(tmp_path))
    # Stub: teacher field 0.75 and base field 0.25 are parallel -> cos 1; the
    # teacher is three times as far from the student as the base prompted branch.
    assert metrics["h3/rollout_teacher_cos_base"] == pytest.approx(1.0)
    assert metrics["h3/rollout_teacher_vs_base"] == pytest.approx((0.75 + 0.25 - 0.375) / (0.25 + 0.25 - 0.375))


def test_term_gradients_main_is_the_rollout_objective_alone_on_a_rollout_step(tmp_path):
    """On a rollout step the floor is folded into the loss before the auxiliaries;
    the reported main term must still be the rollout objective without it."""
    teacher = _teacher_file(tmp_path)
    _, loss, metrics, _ = _run_step(*_FLOOR_FLAGS, "--h3_term_grad_every", "1", teacher=teacher)
    # main and floor are different terms with different gradients on the scale.
    assert metrics["h3/grad/main_norm"] > 0 and metrics["h3/grad/floor_norm"] > 0
    assert metrics["h3/grad/main_norm"] != pytest.approx(metrics["h3/grad/floor_norm"])
    loss.backward()


def test_adapter_stats_use_the_module_effective_down_weight():
    class _Nora(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_down = torch.nn.Linear(4, 2, bias=False)
            self.lora_up = torch.nn.Linear(2, 3, bias=False)
            self.scale = 1.0
            self.multiplier = 1.0
            torch.nn.init.ones_(self.lora_down.weight)
            torch.nn.init.ones_(self.lora_up.weight)

        def effective_down_weight(self, down):
            return down.weight * 0.5

    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unet_loras = torch.nn.ModuleList([_Nora()])
            self.text_encoder_loras = []

    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(_flag_args())
    metrics = trainer._adapter_stat_metrics(_FakeAccelerator(), _Net())
    # up @ (0.5 * down) = 1 everywhere on 3x4: Frobenius sqrt(12).
    assert metrics["h3/adapter/delta_norm"] == pytest.approx(12**0.5)


def test_term_gradients_are_skipped_under_block_swap(tmp_path, caplog):
    teacher = _teacher_file(tmp_path)
    args = _flag_args(*_FLOOR_FLAGS, "--h3_term_grad_every", "1", teacher=teacher)
    trainer = MiniMaxH3NetworkTrainer()
    trainer.handle_model_specific_args(args)
    trainer.dit_dtype = torch.float32
    trainer.backend = _RolloutBackend()
    trainer._rollout_teacher = _StubTeacherCache()
    trainer._rollout_supervision_active = lambda accelerator, probability: True
    trainer.blocks_to_swap = 4
    transformer = _SwapAwareScaleTransformer()
    network = _ToggleNetwork(transformer)
    video, batch = _rollout_batch()
    with caplog.at_level(logging.WARNING):
        _, metrics = trainer.process_batch(
            args,
            _FakeAccelerator(),
            transformer,
            network,
            batch,
            video,
            torch.ones_like(video),
            None,
            torch.float32,
            torch.float32,
            None,
            0,
        )
    assert not [k for k in metrics if k.startswith("h3/grad/")]
    assert any("blocks_to_swap 0" in r.getMessage() for r in caplog.records)
