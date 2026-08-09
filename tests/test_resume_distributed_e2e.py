"""Opt-in two-GPU regression for process-local Accelerate RNG states."""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from accelerate import Accelerator

from musubi_tuner.training.resume_utils import validate_resume_state_dir
from musubi_tuner.utils import train_utils


def _run_worker(root):
    accelerator = Accelerator()
    root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    rank = accelerator.process_index
    random.seed(100 + rank)
    np.random.seed(200 + rank)
    torch.manual_seed(300 + rank)
    torch.cuda.manual_seed_all(400 + rank)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    args = SimpleNamespace(
        output_name="ddp",
        output_dir=str(root),
        save_state_to_huggingface=False,
        save_last_n_steps_state=None,
        save_last_n_steps=None,
        save_every_n_steps=5,
        seed=1234,
    )
    train_utils.save_and_remove_state_stepwise(args, accelerator, 5, epoch=1, step_in_epoch=10)
    expected = (random.random(), float(np.random.rand()), torch.rand(3), torch.rand(3, device=accelerator.device).cpu())

    random.seed(900 + rank)
    np.random.seed(900 + rank)
    torch.manual_seed(900 + rank)
    torch.cuda.manual_seed_all(900 + rank)
    state_dir = root / "ddp-step00000005-state"
    validate_resume_state_dir(str(state_dir), rank, accelerator.num_processes)
    accelerator.load_state(str(state_dir))
    actual = (random.random(), float(np.random.rand()), torch.rand(3), torch.rand(3, device=accelerator.device).cpu())
    passed = (
        expected[0] == actual[0]
        and expected[1] == actual[1]
        and torch.equal(expected[2], actual[2])
        and torch.equal(expected[3], actual[3])
    )
    torch.save(passed, root / f"rank-{rank}.pt")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        files = {path.name for path in state_dir.iterdir()}
        assert {"random_states_0.pkl", "random_states_1.pkl"} <= files
        assert all(torch.load(root / f"rank-{index}.pt", weights_only=False) for index in range(accelerator.num_processes))


@pytest.mark.skipif(
    os.environ.get("MUSUBI_RUN_RESUME_DDP_E2E") != "1" or torch.cuda.device_count() < 2,
    reason="set MUSUBI_RUN_RESUME_DDP_E2E=1 with two visible workbox GPUs",
)
def test_two_rank_rng_state_is_bit_exact(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            __file__,
            "--worker",
            "--root",
            str(tmp_path),
        ],
        check=True,
        env=os.environ.copy(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--root", required=True)
    cli_args = parser.parse_args()
    _run_worker(Path(cli_args.root))
