import copy
import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from musubi_tuner.training.trainer_base import NetworkTrainer


class _Trainer(NetworkTrainer):
    def __init__(self) -> None:
        self.post_steps = 0

    def on_post_optimizer_step(self, *args, **kwargs) -> None:
        self.post_steps += 1


class _Scheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def test_read_only_optimizer_step_clears_gradients_without_mutating_training_state() -> None:
    trainer = _Trainer()
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.5)
    scheduler = _Scheduler()
    accelerator = SimpleNamespace(sync_gradients=True, optimizer_step_was_skipped=False)
    fused_steps = []
    parameter.grad = torch.tensor(1.5)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state_before = copy.deepcopy(optimizer.state_dict())
    parameter.grad = torch.tensor(3.0)
    before = parameter.detach().clone()

    trainer._run_optimizer_step(
        SimpleNamespace(),
        accelerator,
        None,
        None,
        optimizer,
        scheduler,
        lambda actual: fused_steps.append(actual),
        7,
        skip=True,
    )

    torch.testing.assert_close(parameter, before)
    assert parameter.grad is None
    actual_state = optimizer.state_dict()
    assert actual_state["param_groups"] == state_before["param_groups"]
    for key, expected in state_before["state"][0].items():
        torch.testing.assert_close(actual_state["state"][0][key], expected, rtol=0, atol=0)
    assert scheduler.steps == 0
    assert trainer.post_steps == 0
    assert fused_steps == []


def test_ordinary_optimizer_step_preserves_existing_update_contract() -> None:
    trainer = _Trainer()
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = _Scheduler()
    accelerator = SimpleNamespace(sync_gradients=True, optimizer_step_was_skipped=False)
    fused_steps = []
    parameter.grad = torch.tensor(3.0)

    trainer._run_optimizer_step(
        SimpleNamespace(),
        accelerator,
        None,
        None,
        optimizer,
        scheduler,
        lambda actual: fused_steps.append(actual),
        7,
        skip=False,
    )

    torch.testing.assert_close(parameter, torch.tensor(1.7))
    assert parameter.grad is None
    assert scheduler.steps == 1
    assert trainer.post_steps == 1
    assert fused_steps == [optimizer]


def test_read_only_step_guards_max_norm_and_progress_postfix() -> None:
    source = (Path(__file__).parents[1] / "src/musubi_tuner/training/trainer_base.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    guarded_calls = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        condition = ast.unparse(node.test)
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if child.func.attr in {"apply_max_norm_regularization", "set_postfix"}:
                    guarded_calls[child.func.attr] = condition

    assert "not skip_optimizer_step" in guarded_calls["apply_max_norm_regularization"]
    assert "not skip_optimizer_step" in guarded_calls["set_postfix"]
