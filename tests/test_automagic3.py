"""Tests for the Automagic3 optimizer and its trainer integration."""

import argparse

import pytest
import torch
import torch.nn as nn

from musubi_tuner.optimizers import Automagic3
from musubi_tuner.training.trainer_base import NetworkTrainer


def _args(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        optimizer_type="Automagic3",
        optimizer_args=None,
        learning_rate=1e-4,
        lr_scheduler="constant",
        lr_warmup_steps=0,
        lr_decay_steps=0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _train(model: nn.Module, opt: torch.optim.Optimizer, steps: int, dtype=torch.float32) -> tuple[float, float]:
    torch.manual_seed(0)
    x = torch.randn(32, 8, dtype=dtype)
    y = torch.randn(32, 8, dtype=dtype)
    first = last = 0.0
    for step in range(steps):
        loss = torch.nn.functional.mse_loss(model(x).float(), y.float())
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        last = loss.item()
        if step == 0:
            first = last
    return first, last


def _model(dtype=torch.float32) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8)).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_optimizes_and_adapts_the_rate(dtype):
    model = _model(dtype)
    # Railed as the trainer rails it: the optimizer's own bounds are numerical
    # guards, and an unrailed controller walks the rate off this problem
    # (1e-3 to ~3e-1 within 16 steps) until the loss climbs.
    opt = Automagic3(model.parameters(), lr=1e-3, min_lr=1e-5, max_lr=1e-1, fused=False, stochastic_grad_accumulation=False)
    first, last = _train(model, opt, 40, dtype)
    assert last < first
    # The window is 8 steps, so the controller has had room to move off its start.
    assert opt.get_learning_rates()[0] != pytest.approx(1e-3)


def test_adapted_rate_stays_inside_the_rails():
    model = _model()
    opt = Automagic3(model.parameters(), lr=1e-4, min_lr=5e-5, max_lr=2e-4, fused=False, stochastic_grad_accumulation=False)
    _train(model, opt, 40)
    assert 5e-5 <= opt.get_learning_rates()[0] <= 2e-4


def test_each_group_votes_separately():
    """Each param group keeps its own pooled rate."""
    torch.manual_seed(0)
    first, second = nn.Linear(8, 8), nn.Linear(8, 8)
    opt = Automagic3(
        [{"params": list(first.parameters()), "lr": 1e-4}, {"params": list(second.parameters()), "lr": 1e-2}],
        fused=False,
        stochastic_grad_accumulation=False,
    )
    for _ in range(12):
        second(first(torch.randn(4, 8))).pow(2).mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    rates = opt.get_learning_rates()
    assert len(rates) == 2
    assert rates[0] < rates[1]


def test_state_dict_round_trip_restores_the_rate_and_history():
    model = _model()
    opt = Automagic3(model.parameters(), lr=1e-3, fused=False, stochastic_grad_accumulation=False)
    _train(model, opt, 20)
    saved = opt.get_learning_rates()[0]

    restored_model = _model()
    restored = Automagic3(restored_model.parameters(), lr=1e-3, fused=False, stochastic_grad_accumulation=False)
    restored.load_state_dict(opt.state_dict())

    assert restored.get_learning_rates()[0] == pytest.approx(saved)
    for p in restored_model.parameters():
        state = restored.state[p]
        assert state["lr"].dtype is torch.float32
        assert state["sign_history"].dtype is torch.uint8
        assert state["sign_history"].shape == (8, (p.numel() + 7) // 8)


def test_history_is_reset_when_the_window_length_changes():
    model = _model()
    opt = Automagic3(model.parameters(), lr=1e-3, polarity_history=8, fused=False, stochastic_grad_accumulation=False)
    _train(model, opt, 12)

    other_model = _model()
    other = Automagic3(other_model.parameters(), lr=1e-3, polarity_history=16, fused=False, stochastic_grad_accumulation=False)
    other.load_state_dict(opt.state_dict())
    for p in other_model.parameters():
        assert other.state[p]["sign_history"].shape == (16, (p.numel() + 7) // 8)
        assert other.state[p]["hist_fill"] == 0


def test_trainer_selects_automagic3_and_rails_the_rate():
    trainer = NetworkTrainer()
    params = list(_model().parameters())
    _, _, optimizer, _, _ = trainer.get_optimizer(_args(), params)
    assert isinstance(optimizer, Automagic3)
    group = optimizer.param_groups[0]
    assert group["min_lr"] == pytest.approx(1e-6)
    assert group["max_lr"] == pytest.approx(1e-2)
    assert group["weight_decay"] == pytest.approx(1e-4)


def test_trainer_keeps_user_supplied_weight_decay():
    trainer = NetworkTrainer()
    params = list(_model().parameters())
    _, _, optimizer, _, _ = trainer.get_optimizer(_args(optimizer_args=["weight_decay=0.0"]), params)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.0)


def test_trainer_keeps_user_supplied_rails():
    trainer = NetworkTrainer()
    params = list(_model().parameters())
    args = _args(optimizer_args=["min_lr=1e-5", "max_lr=1e-3", "weight_decay=0.0001"])
    _, _, optimizer, _, _ = trainer.get_optimizer(args, params)
    group = optimizer.param_groups[0]
    assert group["min_lr"] == pytest.approx(1e-5)
    assert group["max_lr"] == pytest.approx(1e-3)
    assert group["weight_decay"] == pytest.approx(1e-4)


@pytest.mark.parametrize("hook_arg", ["fused", "stochastic_grad_accumulation"])
def test_trainer_rejects_backward_hook_modes(hook_arg):
    """Both hook modes consume gradients before musubi reduces and clips them."""
    trainer = NetworkTrainer()
    params = list(_model().parameters())
    with pytest.raises(ValueError, match=hook_arg):
        trainer.get_optimizer(_args(optimizer_args=[f"{hook_arg}=True"]), params)


def test_trainer_defaults_hook_modes_off():
    trainer = NetworkTrainer()
    params = list(_model().parameters())
    _, _, optimizer, _, _ = trainer.get_optimizer(_args(), params)
    assert optimizer.fused is False
    assert optimizer.stochastic_grad_accumulation is False
    for p in params:
        assert not hasattr(p, "_accum_grad")


def test_trainer_warns_when_an_external_schedule_is_requested(caplog):
    trainer = NetworkTrainer()
    params = list(_model().parameters())
    with caplog.at_level("WARNING"):
        trainer.get_optimizer(_args(lr_scheduler="cosine"), params)
    assert any("no effect" in record.message for record in caplog.records)


def test_dummy_scheduler_reports_the_adapted_rate():
    trainer = NetworkTrainer()
    model = _model()
    args = _args()
    _, _, optimizer, _, _ = trainer.get_optimizer(args, list(model.parameters()))
    assert trainer.is_schedulefree_optimizer(optimizer, args) is True
    scheduler = trainer.get_lr_scheduler(args, optimizer, num_processes=1)
    _train(model, optimizer, 12)
    scheduler.step()
    assert scheduler.get_last_lr() == optimizer.get_learning_rates()
