"""CPU lifecycle coverage for dense H3's automatic-final-save opt-out."""

import threading
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from safetensors.torch import load_file

from musubi_tuner.minimax_h3_train import MiniMaxH3Trainer, create_parser
from musubi_tuner.minimax_h3_train_network import setup_parser as setup_lora_parser
from musubi_tuner.training import trainer_base
from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.training.trainer_base import NetworkTrainer
from musubi_tuner.utils import async_save, train_utils


class _TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))

    def on_epoch_start(self, transformer):
        pass

    def on_step_start(self):
        pass

    def snapshot_weights(self, dtype):
        return async_save.snapshot_state_dict(self.state_dict(), dtype)

    def save_weights(self, path, dtype, metadata):
        async_save.write_state_dict_file(self.snapshot_weights(dtype), path, metadata)


class _TinyTrainer(NetworkTrainer):
    # Exercise the production H3 policy with the actual common training loop,
    # replacing only the expensive model/dataset construction and H3 objective.
    should_save_final_checkpoint = MiniMaxH3Trainer.should_save_final_checkpoint
    architecture = "mmh3"
    architecture_full_name = "tiny_h3_test"

    def _train_impl(self, args):
        self.accelerator = Accelerator(cpu=True)
        network = _TinyNetwork()
        optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        network, optimizer, scheduler = self.accelerator.prepare(network, optimizer, scheduler)
        dataset = SimpleNamespace(batch_size=1, get_metadata=lambda: {})
        group = SimpleNamespace(num_train_items=1, datasets=[dataset])
        batches = torch.utils.data.DataLoader([{"latents": torch.ones(1)}], batch_size=1)
        self._run_training_loop(
            args,
            self.accelerator,
            1,
            0.0,
            group,
            batches,
            SimpleNamespace(value=0),
            network,
            network,
            network,
            optimizer,
            "SGD",
            {},
            lambda: None,
            lambda: None,
            scheduler,
            [],
            None,
            None,
            torch.float32,
            torch.float32,
        )

    def scale_shift_latents(self, latents):
        return latents

    def process_batch(self, args, accelerator, transformer, network, *unused):
        return network.weight.square().mean(), {}


def test_flag_is_dense_only_and_defaults_to_existing_behavior():
    parser = create_parser()
    assert parser.parse_args([]).skip_final_save is False
    assert parser.parse_args(["--skip_final_save"]).skip_final_save is True
    with pytest.raises(SystemExit):
        setup_lora_parser(setup_parser_common()).parse_args(["--skip_final_save"])
    assert NetworkTrainer().should_save_final_checkpoint(SimpleNamespace(skip_final_save=True))


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("state_mode", ["none", "periodic", "final"])
@pytest.mark.parametrize("stop_mode", ["complete", "dashboard", "save_and_stop"])
def test_training_lifecycle(tmp_path, monkeypatch, skip, state_mode, stop_mode):
    monkeypatch.delenv("MUSUBI_DASHBOARD_METRICS", raising=False)
    args = create_parser().parse_args([])
    args.output_dir = str(tmp_path)
    args.output_name = "tiny"
    args.max_train_steps = 3
    args.save_every_n_steps = 1
    args.save_every_n_epochs = 1
    args.skip_final_save = skip
    args.async_checkpoint_save = True
    args.save_state = state_mode == "periodic"
    args.save_state_on_train_end = state_mode == "final"
    args.full_bf16 = False
    args.mixed_precision = "no"
    args.max_grad_norm = 0.0
    args.seed = 1
    stop_file = tmp_path / "stop.flag"
    if stop_mode == "save_and_stop":
        stop_file.touch()
        args.save_and_stop_request_file = str(stop_file)

    # A real writer is held pending until the production flush calls wait().
    # This proves completion drains writes even when final output is skipped.
    savers = []

    class GatedSaver(async_save.AsyncCheckpointSaver):
        def __init__(self):
            super().__init__()
            self.release = threading.Event()
            self.waits = 0
            savers.append(self)

        def submit(self, job):
            self.release.set()
            release = threading.Event()

            def gated():
                assert release.wait(10), "production loop did not drain the writer"
                job()

            super().submit(gated)
            self.release = release

        def wait(self):
            self.waits += 1
            self.release.set()
            super().wait()

        def join(self, timeout=None):
            self.release.set()
            super().join(timeout)

    monkeypatch.setattr(trainer_base, "AsyncCheckpointSaver", GatedSaver)
    trainer = _TinyTrainer()
    monkeypatch.setattr(
        train_utils,
        "is_dashboard_stop_requested",
        lambda: stop_mode == "dashboard" and bool(savers),
    )
    trainer.train(args)

    should_save_final = not skip or stop_mode == "save_and_stop"
    assert (tmp_path / "tiny.safetensors").exists() == should_save_final
    assert (tmp_path / "tiny-state").exists() == (should_save_final and state_mode != "none")
    periodic = tmp_path / "tiny-step00000001.safetensors"
    assert load_file(str(periodic))["weight"].shape == (1,)
    assert (tmp_path / "tiny-step00000001-state").exists() == (state_mode == "periodic")
    if stop_mode == "complete":
        assert (tmp_path / "tiny-000001.safetensors").exists()
    if stop_mode == "save_and_stop":
        assert not stop_file.exists()
    assert savers and savers[0].waits > 0
    assert not savers[0]._thread.is_alive()
    assert trainer._async_checkpoint_saver is None
