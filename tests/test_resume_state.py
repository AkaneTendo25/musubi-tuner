import json
from types import SimpleNamespace

import torch
from accelerate.data_loader import SeedableRandomSampler

from musubi_tuner.training import accelerator_setup
from musubi_tuner.training.resume_utils import configure_dataloader_for_epoch, recover_global_step
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
        def save_state(self, state_dir):
            from pathlib import Path

            Path(state_dir).mkdir(parents=True, exist_ok=True)

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


def test_legacy_resume_infers_epoch_without_claiming_batch_position():
    assert train_utils.get_resume_position(10000, 116, None) == (86, 0)
