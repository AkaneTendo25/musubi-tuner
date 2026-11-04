"""CPU coverage of the dense trainer's shared profiler lifecycle."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Adafactor

from musubi_tuner.minimax_h3_train import MiniMaxH3FullFinetuneModule, MiniMaxH3Trainer, create_parser
from musubi_tuner.minimax_h3_train_network import H3_PROFILE_WAIT_STEPS, H3_PROFILE_WARMUP_STEPS, MiniMaxH3NetworkTrainer
from musubi_tuner.training.trainer_base import NetworkTrainer


def _setup(tmp_path, steps=1, main=True):
    args = create_parser().parse_args(
        ["--sdpa", "--output_dir", str(tmp_path), "--output_name", "dense", "--h3_profile_steps", str(steps)]
    )
    trainer = MiniMaxH3Trainer()
    accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=main, sync_gradients=True, num_processes=1)
    return args, trainer, accelerator


def test_dense_profiler_runs_without_reinstalling_fused_hooks_or_adapter_callbacks(tmp_path, monkeypatch, caplog):
    args, trainer, accelerator = _setup(tmp_path)
    transformer = torch.nn.Linear(2, 2, bias=False)
    network = MiniMaxH3FullFinetuneModule(transformer)
    optimizer = Adafactor(transformer.parameters(), lr=0.01, relative_step=False, scale_parameter=False)
    accelerator.prepare = lambda *objects, **kwargs: objects[0] if len(objects) == 1 else objects
    accelerator.unwrap_model = lambda model: model
    trainer._prepare_with_accelerator(
        args, accelerator, transformer, network, optimizer, None, None, torch.float32, torch.float32, torch.float32
    )
    parameter = transformer.weight
    assert len(parameter._post_accumulate_grad_hooks) == 1

    def forbidden(*args, **kwargs):
        pytest.fail("Dense profiling must not invoke the parent adapter/teacher lifecycle")

    monkeypatch.setattr(MiniMaxH3NetworkTrainer, "on_train_start", forbidden)
    monkeypatch.setattr(MiniMaxH3NetworkTrainer, "on_post_optimizer_step", forbidden)
    with caplog.at_level("INFO"):
        trainer.on_train_start(args, accelerator, network, transformer, optimizer)
    assert "validation, sampling, saves, logging and data waits" in caplog.text
    assert len(parameter._post_accumulate_grad_hooks) == 1
    profiler = trainer._h3_profiler
    calls = []
    original_step_param = optimizer.step_param

    def observed_step_param(parameter, group):
        calls.append(parameter)
        original_step_param(parameter, group)

    optimizer.step_param = observed_step_param
    total = H3_PROFILE_WAIT_STEPS + H3_PROFILE_WARMUP_STEPS + 1
    try:
        for step in range(total):
            # Accumulation microsteps must leave the profiler schedule untouched.
            trainer.on_post_optimizer_step(args, accelerator, network, transformer, False, step)
            assert trainer._h3_profiler is profiler and not profiler.finished
            transformer(torch.ones(1, 2)).square().sum().backward()
            assert parameter.grad is None
            trainer.on_post_optimizer_step(args, accelerator, network, transformer, True, step)
    finally:
        trainer._finalize_h3_profiler()

    assert len(calls) == total  # one fused update, not two hooks, per backward
    assert profiler.finished and trainer._h3_profiler is None
    text = (tmp_path / "dense_profile.txt").read_text(encoding="utf-8")
    assert text.startswith("profile of 1 optimizer step(s), wall ")
    assert "aten::mm" in text
    assert len(parameter._post_accumulate_grad_hooks) == 1


@pytest.mark.parametrize("fail", [False, True])
def test_dense_inherited_train_cleanup_reports_early_stop(tmp_path, monkeypatch, fail):
    args, trainer, accelerator = _setup(tmp_path)

    def train(self, args):
        self.on_train_start(args, accelerator, None, None, None)
        self.on_post_optimizer_step(args, accelerator, None, None, True, 0)
        if fail:
            raise RuntimeError("interrupted training")
        return "stopped"

    monkeypatch.setattr(NetworkTrainer, "train", train)
    if fail:
        with pytest.raises(RuntimeError, match="interrupted training"):
            trainer.train(args)
    else:
        assert trainer.train(args) == "stopped"
    assert trainer._h3_profiler is None
    text = (tmp_path / "dense_profile.txt").read_text(encoding="utf-8")
    assert "profiling window never opened; training ended after 1 optimizer step(s)" in text


def test_dense_profiler_checks_final_run_window(tmp_path):
    args, trainer, accelerator = _setup(tmp_path)
    args.max_train_steps = H3_PROFILE_WAIT_STEPS + H3_PROFILE_WARMUP_STEPS
    with pytest.raises(ValueError, match="raise --max_train_steps or lower --h3_profile_steps"):
        trainer.on_train_start(args, accelerator, None, None, None)
    assert trainer._h3_profiler is None


@pytest.mark.parametrize("steps,main", [(0, True), (1, False)])
def test_dense_profiler_is_absent_when_disabled_or_not_main(tmp_path, steps, main):
    args, trainer, accelerator = _setup(tmp_path, steps, main)
    trainer.on_train_start(args, accelerator, None, None, None)
    trainer.on_post_optimizer_step(args, accelerator, None, None, True, 0)
    trainer._finalize_h3_profiler()
    assert trainer._h3_profiler is None
    assert not list(tmp_path.iterdir())
