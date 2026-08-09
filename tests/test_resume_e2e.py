"""Opt-in real Accelerate save/load parity tests.

Run on a GPU workbox with MUSUBI_RUN_RESUME_E2E=1. The subprocesses isolate
Accelerate's singleton state and reproduce Musubi's zero-worker epoch shuffle.
"""

from __future__ import annotations

import argparse
import multiprocessing
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
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Dataset

from musubi_tuner.training.resume_utils import (
    capture_training_rng_state,
    configure_dataloader_for_epoch,
    recover_global_step,
    restore_training_rng_state,
    validate_resume_state_dir,
)
from musubi_tuner.utils import train_utils


class EpochShuffleDataset(Dataset):
    def __init__(self):
        self.seed = 0
        self.shared_epoch = multiprocessing.Value("i", 1)
        self.current_epoch = 0
        self.order = list(range(32))

    def set_seed(self, seed, shared_epoch):
        self.seed = seed

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        epoch = self.shared_epoch.value
        if epoch > self.current_epoch:
            random.seed(self.seed + epoch)
            random.shuffle(self.order)
            self.current_epoch = epoch
        return torch.tensor([float(self.order[index])])


def _state_args(root, workers):
    return SimpleNamespace(
        output_name=f"toy{workers}",
        output_dir=str(root),
        save_state_to_huggingface=False,
        save_last_n_steps_state=None,
        save_last_n_steps=None,
        save_every_n_steps=3,
        seed=1234,
    )


def _run_stage(mode, root, workers):
    set_seed(91)
    accelerator = Accelerator(gradient_accumulation_steps=2)
    dataset = EpochShuffleDataset()
    generator = torch.Generator().manual_seed(1234)
    loader = DataLoader(dataset, batch_size=2, num_workers=workers, generator=generator)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 - step / 20)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    global_step = 0
    steps_to_skip = 0
    resume_rng = None
    state_dir = Path(root) / f"toy{workers}-step00000003-state"
    if mode == "resume":
        validate_resume_state_dir(str(state_dir), accelerator.process_index)
        accelerator.load_state(str(state_dir))
        resume_rng = capture_training_rng_state()
        global_step = recover_global_step(str(state_dir), strict=True)
        metadata = train_utils.load_resume_metadata(str(state_dir))
        _, steps_to_skip = train_utils.get_resume_position(global_step, 8, metadata, num_batches_per_epoch=len(loader))

    dataset.shared_epoch.value = 1
    configure_dataloader_for_epoch(loader, 0, 1234)
    if resume_rng is not None and steps_to_skip == 0:
        restore_training_rng_state(resume_rng)
        resume_rng = None

    trace = []
    for step, batch in enumerate(loader):
        if steps_to_skip:
            steps_to_skip -= 1
            if steps_to_skip == 0:
                restore_training_rng_state(resume_rng)
                resume_rng = None
            continue

        py_draw = random.random()
        np_draw = float(np.random.rand())
        torch_draw = torch.rand((), device=accelerator.device)
        if global_step >= 3:
            trace.append((batch.cpu(), py_draw, np_draw, torch_draw.cpu()))

        with accelerator.accumulate(model):
            prediction = model(batch.to(accelerator.device))
            loss = prediction.square().mean() + torch_draw * 0.001 + py_draw * 0.001 + np_draw * 0.001
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if accelerator.sync_gradients:
            global_step += 1
            if global_step == 3 and mode != "resume":
                train_utils.save_and_remove_state_stepwise(
                    _state_args(root, workers), accelerator, 3, epoch=1, step_in_epoch=step + 1
                )
                if mode == "part":
                    return
            if global_step >= 7:
                break

    torch.save(
        {
            "model": accelerator.get_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "trace": trace,
            "global_step": global_step,
        },
        Path(root) / f"{mode}-{workers}.pt",
    )


def _equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


@pytest.mark.skipif(os.environ.get("MUSUBI_RUN_RESUME_E2E") != "1", reason="set MUSUBI_RUN_RESUME_E2E=1 on a workbox")
@pytest.mark.parametrize("workers", [0, 2])
def test_interrupted_resume_is_bit_exact(tmp_path, workers):
    for mode in ("baseline", "part", "resume"):
        subprocess.run(
            [sys.executable, __file__, "--mode", mode, "--root", str(tmp_path), "--workers", str(workers)],
            check=True,
            env=os.environ.copy(),
        )
    baseline = torch.load(tmp_path / f"baseline-{workers}.pt", weights_only=False)
    resumed = torch.load(tmp_path / f"resume-{workers}.pt", weights_only=False)
    assert _equal(baseline, resumed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--workers", type=int, required=True)
    cli_args = parser.parse_args()
    _run_stage(cli_args.mode, cli_args.root, cli_args.workers)
