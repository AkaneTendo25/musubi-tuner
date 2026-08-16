"""Tests for the background checkpoint writer (--async_checkpoint_save)."""

import os
import threading
import time

import pytest
import torch
from safetensors.torch import load_file, safe_open

from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.async_save import AsyncCheckpointSaver, snapshot_state_dict, write_state_dict_file


def _state_dict(value: float = 1.0):
    return {"lora_unet_a.lora_down.weight": torch.full((4, 4), value), "lora_unet_a.alpha": torch.tensor(2.0)}


def test_parser_flag_defaults_off():
    parser = setup_parser_common()
    args = parser.parse_args(["--dataset_config", "x.toml", "--dit", "x.safetensors"])
    assert args.async_checkpoint_save is False
    args = parser.parse_args(["--dataset_config", "x.toml", "--dit", "x.safetensors", "--async_checkpoint_save"])
    assert args.async_checkpoint_save is True


def test_snapshot_is_detached_cpu_copy():
    sd = _state_dict()
    sd["lora_unet_a.lora_down.weight"].requires_grad_(True)
    snapshot = snapshot_state_dict(sd, torch.float16)
    assert snapshot["lora_unet_a.lora_down.weight"].dtype == torch.float16
    assert snapshot["lora_unet_a.lora_down.weight"].device.type == "cpu"
    assert not snapshot["lora_unet_a.lora_down.weight"].requires_grad

    with torch.no_grad():
        sd["lora_unet_a.lora_down.weight"].mul_(0.0)
    # snapshot must not follow later mutation of the live tensors
    assert torch.all(snapshot["lora_unet_a.lora_down.weight"] == 1.0)


def test_write_state_dict_file_content_and_hashes(tmp_path):
    sd = snapshot_state_dict(_state_dict(), torch.float32)
    file = str(tmp_path / "ckpt.safetensors")
    metadata = {"ss_steps": "100"}
    expected_model_hash, expected_legacy_hash = model_utils.precalculate_safetensors_hashes(sd, dict(metadata))

    write_state_dict_file(sd, file, metadata)

    loaded = load_file(file)
    assert set(loaded.keys()) == set(sd.keys())
    assert torch.equal(loaded["lora_unet_a.lora_down.weight"], sd["lora_unet_a.lora_down.weight"])
    with safe_open(file, framework="pt") as f:
        written = f.metadata()
    assert written["ss_steps"] == "100"
    assert written["sshs_model_hash"] == expected_model_hash
    assert written["sshs_legacy_hash"] == expected_legacy_hash
    # no temporary leftovers
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt.safetensors"]


def test_async_save_writes_file(tmp_path):
    sd = snapshot_state_dict(_state_dict(3.0), torch.float32)
    file = str(tmp_path / "ckpt.safetensors")
    saver = AsyncCheckpointSaver()
    try:
        saver.submit_state_dict(sd, file, {"ss_steps": "1"})
        saver.wait()
        assert torch.all(load_file(file)["lora_unet_a.lora_down.weight"] == 3.0)
    finally:
        saver.join()


def test_second_submit_blocks_until_first_finished(tmp_path):
    release_first = threading.Event()
    first_done = threading.Event()
    order = []

    def first():
        release_first.wait(5)
        order.append("first")
        write_state_dict_file(snapshot_state_dict(_state_dict(1.0), None), str(tmp_path / "a.safetensors"), {})
        first_done.set()

    def second():
        order.append("second")
        write_state_dict_file(snapshot_state_dict(_state_dict(2.0), None), str(tmp_path / "b.safetensors"), {})

    saver = AsyncCheckpointSaver()
    try:
        saver.submit(first)

        submitted = threading.Event()

        def do_submit():
            saver.submit(second)
            submitted.set()

        t = threading.Thread(target=do_submit)
        t.start()
        # the second submit must not get through while the first job is still running
        assert not submitted.wait(0.3)
        release_first.set()
        t.join(5)
        assert submitted.is_set()
        saver.wait()
    finally:
        saver.join()

    assert order == ["first", "second"]
    assert torch.all(load_file(str(tmp_path / "a.safetensors"))["lora_unet_a.lora_down.weight"] == 1.0)
    assert torch.all(load_file(str(tmp_path / "b.safetensors"))["lora_unet_a.lora_down.weight"] == 2.0)


def test_worker_exception_surfaces_at_join():
    def boom():
        raise ValueError("disk on fire")

    saver = AsyncCheckpointSaver()
    saver.submit(boom)
    with pytest.raises(RuntimeError, match="disk on fire"):
        saver.join()


def test_worker_exception_surfaces_at_next_submit(tmp_path):
    def boom():
        raise ValueError("disk on fire")

    saver = AsyncCheckpointSaver()
    try:
        saver.submit(boom)
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                saver.submit(lambda: None)
            except RuntimeError as exc:
                assert "disk on fire" in str(exc)
                break
            time.sleep(0.02)
        else:
            pytest.fail("worker failure was never reported")
    finally:
        saver.join()


def test_write_is_atomic_temp_then_replace(tmp_path):
    """While the write is in flight the real path must not exist yet."""
    file = tmp_path / "ckpt.safetensors"
    started = threading.Event()
    finish = threading.Event()
    sd = snapshot_state_dict(_state_dict(), None)

    real_save = model_utils.precalculate_safetensors_hashes

    def slow_hash(tensors, metadata):
        started.set()
        finish.wait(5)
        return real_save(tensors, metadata)

    saver = AsyncCheckpointSaver()
    try:
        model_utils.precalculate_safetensors_hashes = slow_hash
        saver.submit_state_dict(sd, str(file), {"ss_steps": "1"})
        assert started.wait(5)
        assert not file.exists()  # nothing at the final path until os.replace
        finish.set()
        saver.wait()
    finally:
        model_utils.precalculate_safetensors_hashes = real_save
        saver.join()

    assert file.exists()
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt.safetensors"]


def test_failed_write_leaves_no_partial_file(tmp_path):
    file = tmp_path / "ckpt.safetensors"
    real = model_utils.precalculate_safetensors_hashes

    def boom(tensors, metadata):
        raise RuntimeError("hash failed")

    try:
        model_utils.precalculate_safetensors_hashes = boom
        with pytest.raises(RuntimeError):
            write_state_dict_file(snapshot_state_dict(_state_dict(), None), str(file), {})
    finally:
        model_utils.precalculate_safetensors_hashes = real

    assert not file.exists()
    assert list(tmp_path.iterdir()) == []


def _trainer_with_saver(impl):
    from musubi_tuner.training.trainer_base import NetworkTrainer

    class T(NetworkTrainer):
        def _train_impl(self, args):
            return impl(self)

    return T()


def test_train_joins_saver_and_reports_write_failure(tmp_path):
    """A save queued just before training ends is waited for and its failure raised."""

    def impl(trainer):
        saver = AsyncCheckpointSaver()
        trainer._async_checkpoint_saver = saver
        saver.submit(lambda: (_ for _ in ()).throw(ValueError("write failed")))

    trainer = _trainer_with_saver(impl)
    with pytest.raises(RuntimeError, match="write failed"):
        trainer.train(object())
    assert trainer._async_checkpoint_saver is None


def test_train_does_not_mask_training_exception(tmp_path):
    def impl(trainer):
        saver = AsyncCheckpointSaver()
        trainer._async_checkpoint_saver = saver
        saver.submit(lambda: (_ for _ in ()).throw(ValueError("write failed")))
        raise KeyError("training blew up")

    trainer = _trainer_with_saver(impl)
    with pytest.raises(KeyError, match="training blew up"):
        trainer.train(object())
    assert trainer._async_checkpoint_saver is None


def test_train_drains_pending_save_on_exit(tmp_path):
    file = tmp_path / "late.safetensors"

    def impl(trainer):
        saver = AsyncCheckpointSaver()
        trainer._async_checkpoint_saver = saver

        def slow_write():
            time.sleep(0.2)
            write_state_dict_file(snapshot_state_dict(_state_dict(7.0), None), str(file), {})

        saver.submit(slow_write)

    trainer = _trainer_with_saver(impl)
    trainer.train(object())
    assert file.exists()
    assert torch.all(load_file(str(file))["lora_unet_a.lora_down.weight"] == 7.0)


def test_lora_network_save_weights_roundtrip(tmp_path):
    import musubi_tuner.networks.lora as lora_module

    class Dummy(torch.nn.Module):
        snapshot_weights = lora_module.LoRANetwork.snapshot_weights
        save_weights = lora_module.LoRANetwork.save_weights

        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(2, 2))

    net = Dummy()
    file = str(tmp_path / "lora.safetensors")
    net.save_weights(file, torch.float16, {"ss_steps": "5"})
    loaded = load_file(file)
    assert loaded["w"].dtype == torch.float16
    with safe_open(file, framework="pt") as f:
        md = f.metadata()
    assert "sshs_model_hash" in md and md["ss_steps"] == "5"
    assert os.path.exists(file)
