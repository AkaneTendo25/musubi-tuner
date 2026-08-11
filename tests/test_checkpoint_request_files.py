from types import SimpleNamespace

import torch

from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.utils import train_utils


def test_checkpoint_request_parser_defaults_are_disabled():
    args = setup_parser_common().parse_args([])

    assert args.save_request_file is None
    assert args.save_and_stop_request_file is None


def test_checkpoint_request_files_are_detected_and_consumed(tmp_path):
    save_file = tmp_path / "save.flag"
    stop_file = tmp_path / "stop.flag"
    save_file.touch()
    stop_file.touch()
    accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)

    requested = train_utils.poll_checkpoint_request_files(str(save_file), str(stop_file), accelerator)

    assert requested == (True, True)
    train_utils.consume_checkpoint_request_file(str(save_file))
    train_utils.consume_checkpoint_request_file(str(stop_file))
    assert not save_file.exists() and not stop_file.exists()


def test_checkpoint_request_is_broadcast_from_global_rank_zero(monkeypatch, tmp_path):
    accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def broadcast(flags, src):
        assert src == 0
        flags.copy_(torch.tensor([1, 0], dtype=torch.uint8))

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)

    assert train_utils.poll_checkpoint_request_files(str(tmp_path / "not-visible-on-worker.flag"), None, accelerator) == (
        True,
        False,
    )


def test_checkpoint_request_poll_is_noop_when_disabled(monkeypatch):
    accelerator = SimpleNamespace(device=torch.device("cpu"), is_main_process=True)
    monkeypatch.setattr(torch, "zeros", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("allocated")))

    assert train_utils.poll_checkpoint_request_files(None, None, accelerator) == (False, False)
