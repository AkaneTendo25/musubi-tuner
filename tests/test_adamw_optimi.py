import argparse
import importlib.util
import sys

import pytest
import torch

from musubi_tuner.training.trainer_base import NetworkTrainer


def _optimizer_args(**overrides):
    base = {
        "optimizer_type": "adamw_optimi",
        "optimizer_args": None,
        "learning_rate": 1e-4,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_adamw_optimi_missing_package_raises_a_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "optimi", None)
    with pytest.raises(ImportError, match="torch-optimi"):
        NetworkTrainer().get_optimizer(_optimizer_args(), [torch.nn.Parameter(torch.zeros(2))])


@pytest.mark.skipif(importlib.util.find_spec("optimi") is None, reason="torch-optimi is not installed")
def test_adamw_optimi_constructs_and_steps_when_available():
    parameter_a = torch.nn.Parameter(torch.randn(4, 4))
    parameter_b = torch.nn.Parameter(torch.randn(8))
    optimizer = NetworkTrainer().get_optimizer(_optimizer_args(optimizer_type="stableadamw_optimi"), [parameter_a, parameter_b])
    assert type(optimizer).__name__ == "StableAdamW"
    (parameter_a.square().sum() + parameter_b.square().sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert torch.isfinite(parameter_a).all()
