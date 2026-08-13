import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from accelerate.data_loader import SeedableRandomSampler

from musubi_tuner.training import accelerator_setup
from musubi_tuner.training.resume_utils import (
    capture_training_rng_state,
    configure_dataloader_for_epoch,
    find_latest_state_dir,
    recover_global_step,
    restore_training_rng_state,
    validate_resume_state_dir,
)
from musubi_tuner.utils import train_utils


def _save_args(tmp_path):
    return SimpleNamespace(
        output_name="dataset",
        output_dir=str(tmp_path),
        save_state_to_huggingface=False,
        save_last_n_epochs_state=None,
        save_last_n_epochs=None,
        save_every_n_epochs=1,
        save_last_n_steps_state=None,
        save_last_n_steps=None,
        save_every_n_steps=1000,
        seed=1234,
    )


def test_resume_metadata_round_trip_and_corrupt_file(tmp_path):
    train_utils.save_resume_metadata(str(tmp_path), global_step=10000, step_in_epoch=24, epoch=87)

    assert train_utils.load_resume_metadata(str(tmp_path)) == {
        "global_step": 10000,
        "step_in_epoch": 24,
        "epoch": 87,
    }
    assert not (tmp_path / f"{train_utils.RESUME_METADATA_NAME}.tmp").exists()

    (tmp_path / train_utils.RESUME_METADATA_NAME).write_text("{broken", encoding="utf-8")
    assert train_utils.load_resume_metadata(str(tmp_path)) is None


def test_all_state_directory_types_save_resume_metadata(tmp_path):
    args = _save_args(tmp_path)

    # Create directories the same way Accelerate does before writing its payload.
    class Accelerator:
        is_main_process = True

        def save_state(self, state_dir):
            from pathlib import Path

            Path(state_dir).mkdir(parents=True, exist_ok=True)

        def wait_for_everyone(self):
            pass

    accelerator = Accelerator()
    train_utils.save_and_remove_state_stepwise(args, accelerator, 10000, epoch=87, step_in_epoch=24)
    train_utils.save_and_remove_state_on_epoch_end(args, accelerator, 87, global_step=10092, step_in_epoch=0)
    train_utils.save_state_on_train_end(args, accelerator, global_step=10100, epoch=88, step_in_epoch=8)

    expected = {
        "dataset-step00010000-state": {
            "global_step": 10000,
            "step_in_epoch": 24,
            "epoch": 87,
            "data_seed": 1234,
        },
        "dataset-000087-state": {
            "global_step": 10092,
            "step_in_epoch": 0,
            "epoch": 87,
            "data_seed": 1234,
        },
        "dataset-state": {"global_step": 10100, "step_in_epoch": 8, "epoch": 88, "data_seed": 1234},
    }
    for dirname, metadata in expected.items():
        path = tmp_path / dirname / train_utils.RESUME_METADATA_NAME
        assert json.loads(path.read_text(encoding="utf-8")) == metadata


def test_recover_global_step_prefers_metadata(tmp_path):
    torch.save({"last_epoch": 9000}, tmp_path / "scheduler.bin")
    train_utils.save_resume_metadata(str(tmp_path), global_step=10000, step_in_epoch=24, epoch=87)

    assert recover_global_step(str(tmp_path)) == 10000


def test_recover_global_step_from_legacy_scheduler(tmp_path):
    torch.save({"last_epoch": 10000, "_step_count": 10001}, tmp_path / "scheduler.bin")

    assert recover_global_step(str(tmp_path)) == 10000


def test_strict_recovery_refuses_silent_step_zero(tmp_path):
    torch.save({"last_epoch": 0}, tmp_path / "scheduler.bin")

    with pytest.raises(ValueError, match="Refusing to silently restart at step 0"):
        recover_global_step(str(tmp_path), strict=True)


def test_resume_validation_distinguishes_weights_from_state(tmp_path):
    weights = tmp_path / "lora.safetensors"
    weights.write_bytes(b"weights")

    with pytest.raises(ValueError, match="Use --network_weights"):
        validate_resume_state_dir(str(weights))

    state_dir = tmp_path / "run-step00000005-state"
    state_dir.mkdir()
    (state_dir / "scheduler.bin").write_bytes(b"scheduler")
    (state_dir / "model.safetensors").write_bytes(b"model")
    (state_dir / "optimizer.bin").write_bytes(b"optimizer")
    (state_dir / "random_states_0.pkl").write_bytes(b"rng")
    with pytest.raises(ValueError, match="incomplete for process 1"):
        validate_resume_state_dir(str(state_dir), process_index=1, num_processes=2)

    (state_dir / "random_states_1.pkl").write_bytes(b"rng")
    validate_resume_state_dir(str(state_dir), process_index=1, num_processes=2)


def test_training_rng_snapshot_restores_python_numpy_torch_and_cuda(monkeypatch):
    random.seed(10)
    np.random.seed(11)
    torch.manual_seed(12)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    state = capture_training_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))

    random.seed(20)
    np.random.seed(21)
    torch.manual_seed(22)
    restore_training_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_non_main_process_still_writes_its_accelerate_state(tmp_path):
    args = _save_args(tmp_path)

    class Accelerator:
        is_main_process = False

        def __init__(self):
            self.saved = []
            self.waits = 0

        def save_state(self, state_dir):
            self.saved.append(state_dir)

        def wait_for_everyone(self):
            self.waits += 1

    accelerator = Accelerator()
    train_utils.save_and_remove_state_stepwise(args, accelerator, 5, epoch=1, step_in_epoch=10)

    assert accelerator.saved == [str(tmp_path / "dataset-step00000005-state")]
    assert accelerator.waits == 3
    assert not (tmp_path / "dataset-step00000005-state" / train_utils.RESUME_METADATA_NAME).exists()


def test_autoresume_selects_highest_complete_matching_state(tmp_path):
    def make_state(name, step, *, complete=True):
        state_dir = tmp_path / name
        state_dir.mkdir()
        torch.save({"last_epoch": step}, state_dir / "scheduler.bin")
        (state_dir / "model.safetensors").write_bytes(b"model")
        (state_dir / "optimizer.bin").write_bytes(b"optimizer")
        if complete:
            (state_dir / "random_states_0.pkl").write_bytes(b"rng")
        return state_dir

    make_state("h3-step00000010-state", 10)
    expected = make_state("h3-step00000020-state", 20)
    make_state("h3-step00000030-state", 30, complete=False)
    make_state("other-step00000040-state", 40)

    args = SimpleNamespace(output_dir=str(tmp_path), output_name="h3")
    assert find_latest_state_dir(args) == str(expected)


def test_state_manifest_marks_completed_save(tmp_path):
    args = _save_args(tmp_path)

    class Accelerator:
        is_main_process = True

        def save_state(self, state_dir):
            from pathlib import Path

            path = Path(state_dir)
            path.mkdir(parents=True, exist_ok=True)
            for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
                (path / name).write_bytes(b"state")

        def wait_for_everyone(self):
            pass

    train_utils.save_and_remove_state_stepwise(args, Accelerator(), 5, epoch=1, step_in_epoch=10)
    state_dir = tmp_path / "dataset-step00000005-state"
    manifest = train_utils.load_state_manifest(str(state_dir))

    assert manifest["complete"] is True
    assert manifest["global_step"] == 5
    assert train_utils.is_complete_state_dir(str(state_dir), process_index=0)


def test_resume_position_uses_exact_metadata_for_mid_epoch():
    metadata = {"global_step": 10000, "epoch": 87, "step_in_epoch": 24}

    assert train_utils.get_resume_position(10000, 116, metadata) == (86, 24)


def test_fresh_training_starts_at_zero():
    assert train_utils.get_resume_position(0, 116, None) == (0, 0)


def test_resume_position_starts_after_epoch_end():
    metadata = {"global_step": 10092, "epoch": 87, "step_in_epoch": 0}

    assert train_utils.get_resume_position(10092, 116, metadata) == (87, 0)


def test_resume_position_normalizes_last_batch_checkpoint():
    metadata = {"global_step": 10092, "epoch": 87, "step_in_epoch": 116}

    assert train_utils.get_resume_position(10092, 116, metadata, num_batches_per_epoch=116) == (87, 0)
    assert train_utils.normalize_step_in_epoch(116, 116) == 0


def test_valid_non_object_metadata_is_rejected(tmp_path):
    path = tmp_path / train_utils.RESUME_METADATA_NAME
    path.write_text("[]", encoding="utf-8")

    assert train_utils.load_resume_metadata(str(tmp_path)) is None


def test_invalid_position_fields_fall_back_to_legacy_epoch_inference():
    metadata = {"global_step": 10000, "epoch": "invalid", "step_in_epoch": 24}

    assert train_utils.get_resume_position(10000, 116, metadata) == (86, 0)


def test_seedable_sampler_reconstructs_exact_resumed_order():
    class Dataset:
        def __init__(self):
            self.seed = None
            self.shared_epoch = object()

        def __len__(self):
            return 32

        def set_seed(self, seed, shared_epoch):
            assert shared_epoch is self.shared_epoch
            self.seed = seed

    class DataLoader:
        def __init__(self):
            self.dataset = Dataset()
            self.sampler = SeedableRandomSampler(self.dataset, data_seed=0)
            self.epoch = None

        def get_sampler(self):
            return self.sampler

        def set_epoch(self, epoch):
            self.epoch = epoch
            self.sampler.set_epoch(epoch)

    uninterrupted = DataLoader()
    configure_dataloader_for_epoch(uninterrupted, epoch=6, data_seed=1234)
    uninterrupted_order = list(uninterrupted.sampler)

    resumed = DataLoader()
    configure_dataloader_for_epoch(resumed, epoch=6, data_seed=1234)
    resumed_order = list(resumed.sampler)

    assert resumed.dataset.seed == 1234
    assert resumed.epoch == 6
    assert resumed_order[11:] == uninterrupted_order[11:]


def test_accelerator_uses_training_seed_for_seedable_sampler(monkeypatch):
    captured_kwargs = {}

    class Accelerator:
        device = torch.device("cpu")

        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(accelerator_setup, "Accelerator", Accelerator)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    args = SimpleNamespace(
        logging_dir=None,
        log_prefix=None,
        log_with=None,
        gradient_accumulation_steps=2,
        mixed_precision=None,
        dynamo_backend="NO",
        ddp_gradient_as_bucket_view=False,
        ddp_static_graph=False,
        seed=1234,
    )

    accelerator_setup.prepare_accelerator(args)

    dataloader_config = captured_kwargs["dataloader_config"]
    assert dataloader_config.use_seedable_sampler is True
    assert dataloader_config.data_seed == 1234
    assert dataloader_config.non_blocking is False


def test_accelerator_enables_non_blocking_transfers_with_pinned_dataloader(monkeypatch):
    captured_kwargs = {}

    class Accelerator:
        device = torch.device("cpu")

        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(accelerator_setup, "Accelerator", Accelerator)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    args = SimpleNamespace(
        logging_dir=None,
        log_prefix=None,
        log_with=None,
        gradient_accumulation_steps=1,
        mixed_precision=None,
        dynamo_backend="NO",
        ddp_gradient_as_bucket_view=False,
        ddp_static_graph=False,
        seed=1234,
        dataloader_pin_memory=True,
    )

    accelerator_setup.prepare_accelerator(args)

    assert captured_kwargs["dataloader_config"].non_blocking is True


def test_dataloader_extra_kwargs_are_opt_in_and_worker_aware(caplog):
    args = SimpleNamespace(dataloader_pin_memory=True, dataloader_prefetch_factor=3)

    assert accelerator_setup.dataloader_extra_kwargs(args, 2) == {"pin_memory": True, "prefetch_factor": 3}
    assert accelerator_setup.dataloader_extra_kwargs(args, 0) == {"pin_memory": True}
    assert "--max_data_loader_n_workers is 0" in caplog.text


def test_dataloader_extra_kwargs_reject_invalid_prefetch_factor():
    args = SimpleNamespace(dataloader_pin_memory=False, dataloader_prefetch_factor=0)

    with pytest.raises(ValueError, match="at least 1"):
        accelerator_setup.dataloader_extra_kwargs(args, 1)


def test_legacy_resume_infers_epoch_without_claiming_batch_position():
    assert train_utils.get_resume_position(10000, 116, None) == (86, 0)
