import json
from types import SimpleNamespace

import torch

from musubi_tuner.training.resume_utils import recover_global_step
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
        "dataset-step00010000-state": {"global_step": 10000, "step_in_epoch": 24, "epoch": 87},
        "dataset-000087-state": {"global_step": 10092, "step_in_epoch": 0, "epoch": 87},
        "dataset-state": {"global_step": 10100, "step_in_epoch": 8, "epoch": 88},
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


def test_legacy_resume_infers_epoch_without_claiming_batch_position():
    assert train_utils.get_resume_position(10000, 116, None) == (86, 0)
