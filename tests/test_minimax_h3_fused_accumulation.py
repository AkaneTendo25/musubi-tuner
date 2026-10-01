"""Regression for dense H3 fused Adafactor under Accelerate accumulation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


def run_regression(tuner_root: Path) -> None:
    sys.path.insert(0, str(tuner_root / "src"))

    import torch
    from accelerate import Accelerator
    from transformers import Adafactor

    from musubi_tuner.minimax_h3_train import MiniMaxH3Trainer

    model = torch.nn.Linear(1, 1, bias=False)
    reference = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
        reference.weight.copy_(model.weight)

    optimizer = Adafactor(model.parameters(), lr=0.1, scale_parameter=False, relative_step=False, warmup_init=False)
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    model, optimizer = accelerator.prepare(model, optimizer)
    assert hasattr(optimizer, "optimizer"), "test requires AcceleratedOptimizer, not a raw optimizer"
    raw_optimizer = optimizer.optimizer
    MiniMaxH3Trainer._install_fused_optimizer(
        SimpleNamespace(fused_backward_pass=True, adafactor_triton=False), accelerator, optimizer
    )
    assert optimizer.step.__self__ is optimizer, "AcceleratedOptimizer.step was replaced"
    assert raw_optimizer.step.__self__ is raw_optimizer, "raw Adafactor did not receive fused step"
    installed_step_param = optimizer.step_param
    observed_boundaries = []

    def observed_step_param(parameter, group):
        observed_boundaries.append(accelerator.sync_gradients)
        installed_step_param(parameter, group)

    # This is the same late wrapper used by diagnose_gradients.py. Registered
    # hooks must resolve it dynamically rather than capture the raw method.
    optimizer.step_param = observed_step_param

    initial = model.weight.detach().clone()
    after_backward = []
    sync_states = []
    for _ in range(2):
        with accelerator.accumulate(model):
            loss = model(torch.ones(1, 1)).square().sum()
            accelerator.backward(loss)
            sync_states.append(accelerator.sync_gradients)
            after_backward.append(model.weight.detach().clone())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    assert sync_states == [False, True], sync_states
    assert observed_boundaries == [True], observed_boundaries
    assert torch.equal(after_backward[0], initial), "fused optimizer updated on the first accumulation microstep"
    assert not torch.equal(after_backward[1], initial), "fused optimizer did not update at the accumulation boundary"

    # Two identical microgradients, each divided by accumulation_steps inside
    # Accelerator.backward, must equal one ordinary full gradient.
    reference_optimizer = Adafactor(reference.parameters(), lr=0.1, scale_parameter=False, relative_step=False, warmup_init=False)
    reference(torch.ones(1, 1)).square().sum().backward()
    reference_optimizer.step()
    torch.testing.assert_close(model.weight, reference.weight, rtol=0, atol=0)


def test_dense_fused_accumulation_preserves_accelerate_boundary():
    run_regression(Path(__file__).resolve().parents[1])
