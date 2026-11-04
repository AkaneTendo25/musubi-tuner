"""CPU mechanism regressions; CUDA transfer/autograd acceptance is separate."""

import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from safetensors.torch import load_file, save_file
from transformers import Adafactor

from musubi_tuner.minimax_h3_train import MiniMaxH3FullFinetuneModule, MiniMaxH3Trainer
from musubi_tuner.modules.custom_offloading_utils import TrainableBlockRingOffloader
from musubi_tuner.training.h3_dense_state import (
    INCOMPLETE,
    MANIFEST,
    install_adafactor_state_restore,
    logical_state,
    register_dense_state_hooks,
)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3, 4, dtype=torch.bfloat16))
        self.bias = torch.nn.Parameter(torch.randn(3, dtype=torch.bfloat16))
        self.register_buffer("frozen_table", torch.randn(2, 3))

    def forward(self, value):
        return torch.nn.functional.linear(value, self.weight.float(), self.bias.float())


class Transformer(torch.nn.Module):
    def __init__(self, compiled=False):
        super().__init__()
        blocks = [Block(), Block()]
        if compiled:
            wrappers = []
            for block in blocks:
                wrapper = torch.nn.Module()
                wrapper.add_module("_orig_mod", block)
                wrappers.append(wrapper)
            blocks = wrappers
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, value):
        return sum(getattr(block, "_orig_mod", block)(value) for block in self.blocks)


def packed_model(compiled=False):
    transformer = Transformer(compiled)
    ring = TrainableBlockRingOffloader.__new__(TrainableBlockRingOffloader)
    ring._entries = {}
    ring.cpu_flat = {}
    for index, block in enumerate(transformer.blocks):
        entries = ring._block_tensors(block)
        ring._layout, size = ring._make_layout([entry[3] for entry in entries])
        flat = torch.zeros(size, dtype=torch.uint8)
        for entry, view in zip(entries, ring._views(flat)):
            view.copy_(entry[3])
        ring._entries[index] = [(module, name, parameter) for module, name, parameter, _ in entries]
        ring.cpu_flat[index] = flat
        ring._bind(index, flat)
    ring.ring_flat = [ring.cpu_flat[0].clone()]
    ring.in_slot = [0]
    ring._bind(0, ring.ring_flat[0])
    ring.free_event = [None]
    ring.load_event = {}
    ring.B = 1
    ring.S = 2
    ring.stream_idx = [0, 1]
    ring.rank = {0: 0, 1: 1}
    ring.device = torch.device("cpu")
    ring._phase = "backward"
    ring._last_wait_rank = 0
    ring.copy_stream = SimpleNamespace(synchronize=lambda: None, wait_event=lambda event: None)
    transformer.offloader = ring
    return MiniMaxH3FullFinetuneModule(transformer)


@pytest.fixture(autouse=True)
def cpu_ring_scheduling(monkeypatch):
    """Emulate only CUDA scheduling; real ring packing/copy/owner methods run."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda stream: None))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: SimpleNamespace(wait_stream=lambda stream: None))


def setup_run(compiled=False, lr=0.01, packed=True):
    accelerator = Accelerator(cpu=True)
    model = packed_model(compiled) if packed else MiniMaxH3FullFinetuneModule(Transformer(compiled))
    raw = Adafactor(model.parameters(), lr=lr, relative_step=False, scale_parameter=False)
    install_adafactor_state_restore(raw)
    model, optimizer = accelerator.prepare(model, raw)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(raw, gamma=0.8)
    scheduler = accelerator.prepare(scheduler)
    MiniMaxH3Trainer._install_fused_optimizer(
        SimpleNamespace(fused_backward_pass=True, adafactor_triton=False), accelerator, optimizer
    )
    register_dense_state_hooks(accelerator, model, optimizer)
    return accelerator, model, optimizer, scheduler


def update(model, scheduler):
    value = torch.randn(2, 4)
    loss = model(value).square().mean()
    loss.backward()
    scheduler.step()


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_complete_accelerate_roundtrip_and_next_update(tmp_path):
    torch.manual_seed(514)
    accelerator, model, optimizer, scheduler = setup_run(compiled=True)
    for _ in range(3):
        update(model, scheduler)
    expected = {name: tensor.clone() for name, tensor in logical_state(model.transformer).items()}
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    # Owner payload is dirty relative to its CPU master: save must flush it.
    assert not torch.equal(model.transformer.offloader.cpu_flat[0], model.transformer.offloader.ring_flat[0])
    path = tmp_path / "standalone-state"
    before_save_rng = torch.get_rng_state().clone()
    accelerator.save_state(str(path))
    assert torch.equal(before_save_rng, torch.get_rng_state())
    assert_nested_equal(load_file(str(path / "model.safetensors")), expected)
    assert_nested_equal(logical_state(model.transformer), expected)
    assert model.transformer.offloader.in_slot == [0]
    rng = torch.get_rng_state().clone()
    update(model, scheduler)
    continued = {name: tensor.clone() for name, tensor in logical_state(model.transformer).items()}
    continued_optimizer = copy.deepcopy(optimizer.state_dict())
    continued_scheduler = copy.deepcopy(scheduler.state_dict())
    accelerator.free_memory()

    # Compile naming may change, and constructor LR deliberately differs.
    resumed, fresh, fresh_optimizer, fresh_scheduler = setup_run(compiled=False, lr=0.9)
    identities = [id(parameter) for parameter in fresh.parameters()]
    resumed.load_state(str(path))
    assert identities == [id(parameter) for parameter in fresh.parameters()]
    assert_nested_equal(logical_state(fresh.transformer), expected)
    assert_nested_equal(fresh_optimizer.state_dict(), expected_optimizer)
    # Accelerate 1.6 emits a CPU-only CUDA-seed warning *after* restoring
    # CPU state. Verify the actual continuation instead of trusting that log.
    assert torch.equal(rng, torch.get_rng_state())
    update(fresh, fresh_scheduler)
    assert_nested_equal(logical_state(fresh.transformer), continued)
    assert_nested_equal(fresh_optimizer.state_dict(), continued_optimizer)
    assert_nested_equal(fresh_scheduler.state_dict(), continued_scheduler)
    resumed.free_memory()


def test_unpatched_packed_state_loses_keys_and_rejects_complete_file(tmp_path):
    from accelerate.utils.other import clean_state_dict_for_safetensors
    from safetensors.torch import load_model

    model = packed_model()
    state = model.state_dict()
    cleaned = clean_state_dict_for_safetensors(state.copy())
    assert len(cleaned) < len(state)
    complete = tmp_path / "complete.safetensors"
    save_file({name: tensor.clone() for name, tensor in state.items()}, str(complete))
    with pytest.raises(RuntimeError, match="None is covering the entire storage"):
        load_model(model, str(complete))


def test_dense_no_swap_uses_same_standalone_state_format(tmp_path):
    accelerator, model, optimizer, _ = setup_run(packed=False)
    expected = {name: tensor.clone() for name, tensor in logical_state(model.transformer).items()}
    accelerator.save_state(str(tmp_path))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    accelerator.load_state(str(tmp_path))
    assert_nested_equal(logical_state(model.transformer), expected)
    accelerator.free_memory()


@pytest.mark.parametrize("packed", [False, True])
def test_legacy_accelerate_state_requires_complete_logical_payload(tmp_path, packed):
    old_accelerator = Accelerator(cpu=True)
    old_model = packed_model(compiled=True) if packed else MiniMaxH3FullFinetuneModule(Transformer(compiled=True))
    old_raw = Adafactor(old_model.parameters(), lr=0.0123, relative_step=False, scale_parameter=False)
    old_model, old_optimizer = old_accelerator.prepare(old_model, old_raw)
    old_accelerator.prepare(torch.optim.lr_scheduler.ExponentialLR(old_raw, gamma=0.8))
    old_model(torch.randn(2, 4)).square().mean().backward()
    old_optimizer.step()
    expected = {name: tensor.clone() for name, tensor in logical_state(old_model.transformer).items()}
    expected_optimizer = copy.deepcopy(old_optimizer.state_dict())
    # This is the real pre-fix serialization, including storage deduplication.
    old_accelerator.save_state(str(tmp_path))
    assert not (tmp_path / MANIFEST).exists()
    old_accelerator.free_memory()

    resumed, fresh, optimizer, _ = setup_run(packed=packed)
    before = {name: tensor.clone() for name, tensor in logical_state(fresh.transformer).items()}
    if packed:
        with pytest.raises(ValueError, match="Legacy packed-ring states may have omitted tensors"):
            resumed.load_state(str(tmp_path))
        assert_nested_equal(logical_state(fresh.transformer), before)
        assert fresh.transformer.offloader.in_slot == [0]
    else:
        resumed.load_state(str(tmp_path))
        assert_nested_equal(logical_state(fresh.transformer), expected)
        assert_nested_equal(optimizer.state_dict(), expected_optimizer)
    resumed.free_memory()


@pytest.mark.parametrize("damage", ["missing", "shape", "dtype", "manifest", "groups", "truncated"])
def test_incompatible_state_rejected_before_tensor_writes(tmp_path, damage):
    accelerator, model, optimizer, _ = setup_run()
    accelerator.save_state(str(tmp_path))
    weights = tmp_path / "model.safetensors"
    manifest = tmp_path / MANIFEST
    if damage in {"missing", "shape", "dtype"}:
        tensors = {name: tensor.clone() for name, tensor in load_file(str(weights)).items()}
        key = next(name for name in tensors if name.endswith("weight"))
        if damage == "missing":
            tensors.pop(key)
        elif damage == "shape":
            tensors[key] = tensors[key].flatten()
        else:
            tensors[key] = tensors[key].double()
        save_file(tensors, str(weights))
    elif damage == "manifest":
        manifest.unlink()
    elif damage == "groups":
        data = json.loads(manifest.read_text())
        data["optimizer_groups"][0].reverse()
        manifest.write_text(json.dumps(data))
    else:
        weights.write_bytes(weights.read_bytes()[:32])
    before = {name: tensor.clone() for name, tensor in logical_state(model.transformer).items()}
    owners = list(model.transformer.offloader.in_slot)
    with pytest.raises(Exception):
        accelerator.load_state(str(tmp_path))
    assert_nested_equal(logical_state(model.transformer), before)
    assert model.transformer.offloader.in_slot == owners
    accelerator.free_memory()


def test_fp32_adafactor_moments_survive_without_bf16_roundtrip():
    parameter = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.bfloat16))
    optimizer = Adafactor([parameter], lr=0.01, relative_step=False, scale_parameter=False)
    source = {
        "step": 19,
        "exp_avg_sq_row": torch.tensor([0.1234567, 0.2345678]),
        "exp_avg_sq_col": torch.tensor([0.3456789, 0.4567891, 0.5678912]),
        "RMS": torch.tensor(0.1234567),
        "sr_seed": 192837,
    }
    optimizer.state[parameter] = source
    saved = copy.deepcopy(optimizer.state_dict())
    optimizer.load_state_dict(saved)
    assert optimizer.state[parameter]["exp_avg_sq_row"].dtype == torch.bfloat16
    install_adafactor_state_restore(optimizer)
    optimizer.load_state_dict(saved)
    assert_nested_equal(optimizer.state[parameter], source)


def test_failed_overwrite_stays_incomplete_until_entire_save_succeeds(tmp_path, monkeypatch):
    accelerator, model, optimizer, _ = setup_run()
    accelerator.save_state(str(tmp_path))
    assert not (tmp_path / INCOMPLETE).exists()
    raw = optimizer.optimizer
    original_state_dict = raw.state_dict

    def fail_optimizer_save():
        assert (tmp_path / INCOMPLETE).exists()
        # The model pre-hook has already published a full payload + manifest,
        # and previous optimizer/RNG files still exist in this same directory.
        assert (tmp_path / MANIFEST).is_file()
        assert (tmp_path / "optimizer.bin").is_file()
        raise RuntimeError("injected optimizer serialization failure")

    monkeypatch.setattr(raw, "state_dict", fail_optimizer_save)
    with pytest.raises(RuntimeError, match="injected optimizer serialization failure"):
        accelerator.save_state(str(tmp_path))
    assert (tmp_path / INCOMPLETE).is_file()
    before = {name: tensor.clone() for name, tensor in logical_state(model.transformer).items()}
    with pytest.raises(ValueError, match="previous state save did not finish"):
        accelerator.load_state(str(tmp_path))
    assert_nested_equal(logical_state(model.transformer), before)
    monkeypatch.setattr(raw, "state_dict", original_state_dict)

    class LastCustomState:
        def state_dict(self):
            assert (tmp_path / INCOMPLETE).is_file()
            return {"completed": True}

        def load_state_dict(self, state):
            assert state == {"completed": True}

    accelerator.register_for_checkpointing(LastCustomState())
    accelerator.save_state(output_dir=str(tmp_path))
    assert not (tmp_path / INCOMPLETE).exists()
    accelerator.load_state(str(tmp_path))
    accelerator.free_memory()


def test_outer_metadata_failure_defers_completion_and_retry_clears_marker(tmp_path):
    accelerator, model, _, _ = setup_run()
    hooks = accelerator._h3_dense_state_hooks
    accelerator.save_state(str(tmp_path))
    with pytest.raises(RuntimeError, match="injected resume metadata failure"):
        with hooks.defer_completion():
            accelerator.save_state(str(tmp_path))
            assert (tmp_path / INCOMPLETE).is_file()
            raise RuntimeError("injected resume metadata failure")
    assert (tmp_path / INCOMPLETE).is_file()
    with pytest.raises(ValueError, match="previous state save did not finish"):
        accelerator.load_state(str(tmp_path))
    with hooks.defer_completion():
        with hooks.defer_completion():
            accelerator.save_state(str(tmp_path))
        assert (tmp_path / INCOMPLETE).is_file()
        (tmp_path / "test_resume_cursor.json").write_text('{"global_step": 3}')
    assert not (tmp_path / INCOMPLETE).exists()
    accelerator.load_state(str(tmp_path))
    accelerator.free_memory()


@pytest.mark.parametrize("mode", ["step", "epoch", "final"])
@pytest.mark.parametrize("failing_writer", ["save_resume_metadata", "save_state_manifest"])
def test_real_training_save_helpers_keep_failed_overwrites_out_of_resume(tmp_path, monkeypatch, mode, failing_writer):
    from musubi_tuner.training.resume_utils import find_latest_state_dir, validate_resume_state_dir
    from musubi_tuner.utils import train_utils

    accelerator, model, _, _ = setup_run()
    args = SimpleNamespace(
        output_name="dense",
        output_dir=str(tmp_path),
        save_state_to_huggingface=False,
        save_every_n_epochs=1,
        save_every_n_steps=1,
        seed=123,
    )

    def save(cursor):
        if mode == "step":
            train_utils.save_and_remove_state_stepwise(args, accelerator, 10, epoch=2, step_in_epoch=cursor)
            return tmp_path / train_utils.STEP_STATE_NAME.format("dense", 10)
        if mode == "epoch":
            train_utils.save_and_remove_state_on_epoch_end(args, accelerator, 2, global_step=10, step_in_epoch=cursor)
            return tmp_path / train_utils.EPOCH_STATE_NAME.format("dense", 2)
        train_utils.save_state_on_train_end(args, accelerator, global_step=10, epoch=2, step_in_epoch=cursor)
        return tmp_path / train_utils.LAST_STATE_NAME.format("dense")

    directory = save(1)
    assert train_utils.is_complete_state_dir(str(directory), process_index=0)
    validate_resume_state_dir(str(directory), num_processes=1)
    assert find_latest_state_dir(args) == str(directory)
    original_writer = getattr(train_utils, failing_writer)

    def fail(*args, **kwargs):
        assert (directory / INCOMPLETE).is_file()
        raise RuntimeError("injected outer metadata failure")

    monkeypatch.setattr(train_utils, failing_writer, fail)
    with pytest.raises(RuntimeError, match="injected outer metadata failure"):
        save(2)
    assert (directory / INCOMPLETE).is_file()
    assert not train_utils.is_complete_state_dir(str(directory), process_index=0)
    assert find_latest_state_dir(args) is None
    with pytest.raises(ValueError):
        validate_resume_state_dir(str(directory), num_processes=1)
    with pytest.raises(ValueError, match="previous state save did not finish"):
        accelerator.load_state(str(directory))
    monkeypatch.setattr(train_utils, failing_writer, original_writer)
    save(2)
    assert not (directory / INCOMPLETE).exists()
    assert train_utils.is_complete_state_dir(str(directory), process_index=0)
    validate_resume_state_dir(str(directory), num_processes=1)
    assert find_latest_state_dir(args) == str(directory)
    metadata = train_utils.load_resume_metadata(str(directory))
    assert metadata == {"global_step": 10, "step_in_epoch": 2, "epoch": 2, "data_seed": 123}
    manifest = train_utils.load_state_manifest(str(directory))
    assert (manifest["global_step"], manifest["step_in_epoch"], manifest["epoch"]) == (10, 2, 2)
    accelerator.load_state(str(directory))
    accelerator.free_memory()


def test_dense_retention_and_stop_save_leave_relocated_states_self_contained(tmp_path):
    from musubi_tuner.training.resume_utils import find_latest_state_dir, validate_resume_state_dir
    from musubi_tuner.utils import train_utils

    output = tmp_path / "run"
    args = SimpleNamespace(
        output_name="dense",
        output_dir=str(output),
        save_state_to_huggingface=False,
        save_every_n_steps=2,
        save_last_n_checkpoints=2,
        seed=123,
    )
    accelerator, model, optimizer, scheduler = setup_run()
    expected = {}

    def directory(step):
        return output / train_utils.STEP_STATE_NAME.format("dense", step)

    # Three periodic checkpoints (2/4/6) retain two. An additional forced
    # checkpoint at 3 proves that a later unscheduled save does not prune it.
    for step in range(1, 8):
        update(model, scheduler)
        if step not in {2, 3, 4, 6, 7}:
            continue
        train_utils.save_and_remove_state_stepwise(
            args, accelerator, step, epoch=1, step_in_epoch=step, apply_retention=step not in {3, 7}
        )
        if step in {4, 6}:
            expected[step] = (
                {name: tensor.clone() for name, tensor in logical_state(model.transformer).items()},
                copy.deepcopy(optimizer.state_dict()),
                copy.deepcopy(scheduler.state_dict()),
                torch.get_rng_state().clone(),
            )
        if step == 6:
            assert not directory(2).exists()
            assert directory(4).is_dir() and directory(6).is_dir()
    # At step 7 the retention arithmetic would delete step 3 (7 - 2*2).
    # The stop-style request explicitly disables that pruning.
    assert {path.name for path in output.iterdir()} == {directory(step).name for step in (3, 4, 6, 7)}
    assert find_latest_state_dir(args) == str(directory(7))
    accelerator.free_memory()

    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for step in (4, 6):
        source, target = directory(step), relocated / directory(step).name
        assert source.resolve().is_relative_to(tmp_path.resolve())
        assert target.resolve().is_relative_to(tmp_path.resolve())
        source.rename(target)
        validate_resume_state_dir(str(target), num_processes=1)
        # Fresh model/optimizer with different constructor LR; source output no
        # longer contains this state, and no ordinary step weight file exists.
        resumed, fresh, fresh_optimizer, fresh_scheduler = setup_run(lr=0.9)
        resumed.load_state(str(target))
        weights, optimizer_state, scheduler_state, rng = expected[step]
        assert_nested_equal(logical_state(fresh.transformer), weights)
        assert_nested_equal(fresh_optimizer.state_dict(), optimizer_state)
        assert_nested_equal(fresh_scheduler.state_dict(), scheduler_state)
        assert torch.equal(torch.get_rng_state(), rng)
        assert train_utils.load_resume_metadata(str(target))["global_step"] == step
        resumed.free_memory()
