"""CPU checks for loading training state onto an explicitly new H3 shard."""

import copy
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator

from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser
from musubi_tuner.training.resume_utils import restore_training_rng_state
from musubi_tuner.training.trainer_base import NetworkTrainer
from musubi_tuner.utils import train_utils


@pytest.mark.parametrize("new_shard", [False, True])
def test_real_accelerate_resume_preserves_state_and_resets_only_new_shard_cursor(tmp_path, new_shard):
    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(model(torch.ones(1, 2)).square().mean())
        optimizer.step()
        scheduler.step()
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    state_dir = tmp_path / "previous-state"
    accelerator.save_state(str(state_dir))
    train_utils.save_resume_metadata(str(state_dir), global_step=3, step_in_epoch=3, epoch=0, data_seed=99)
    expected_draw = torch.rand(4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100)
    optimizer.param_groups[0]["lr"] = 12
    scheduler.step()
    torch.manual_seed(999)
    trainer = NetworkTrainer()
    initial_step = trainer.resume_from_local_or_hf_if_specified(
        accelerator,
        SimpleNamespace(resume=str(state_dir), resume_from_huggingface=False, resume_as_new_shard=new_shard),
    )
    assert initial_step == (0 if new_shard else 3)
    for key, value in expected_model.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    actual_optimizer = optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for index, expected in expected_optimizer["state"].items():
        for key, value in expected.items():
            torch.testing.assert_close(actual_optimizer["state"][index][key], value, rtol=0, atol=0)
    assert scheduler.state_dict() == expected_scheduler
    metadata = train_utils.load_resume_metadata(trainer._resume_state_dir) if initial_step > 0 else None
    epoch, skipped = train_utils.get_resume_position(initial_step, 5, metadata, num_batches_per_epoch=5)
    assert (epoch, skipped) == ((0, 0) if new_shard else (0, 3))
    # The actual loop restores the captured state after new-shard setup draws.
    torch.rand(20)
    restore_training_rng_state(trainer._resume_rng_state)
    torch.testing.assert_close(torch.rand(4), expected_draw, rtol=0, atol=0)
    accelerator.free_memory()


def test_new_shard_requires_explicit_resume_and_is_default_off():
    parser = create_parser()
    assert parser.parse_args([]).resume_as_new_shard is False
    args = parser.parse_args(["--resume_as_new_shard"])
    with pytest.raises(ValueError, match="requires --resume"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(args)
    with pytest.raises(ValueError, match="requires --resume"):
        NetworkTrainer().resume_from_local_or_hf_if_specified(None, SimpleNamespace(resume=None, resume_as_new_shard=True))
