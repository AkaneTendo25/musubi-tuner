import logging

import torch

from musubi_tuner.minimax_h3.activation_offload import _OffloadedTensor, ReusableActivationOffloader


class _FakeEvent:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


def _offloader_without_cuda() -> ReusableActivationOffloader:
    offloader = object.__new__(ReusableActivationOffloader)
    offloader._handles = {}
    return offloader


def _handle(block_index: int, *, consumed: bool, gpu: torch.Tensor | None = None, h2d_event=None) -> _OffloadedTensor:
    return _OffloadedTensor(
        block_index=block_index,
        ordinal=0,
        cpu=torch.zeros(1),
        device=torch.device("cpu"),
        d2h_event=None,
        gpu=gpu,
        h2d_event=h2d_event,
        consumed=consumed,
    )


def test_reusable_activation_prefetch_skips_consumed_handle(monkeypatch) -> None:
    offloader = _offloader_without_cuda()
    current = _handle(2, consumed=True)
    previous = _handle(1, consumed=True)
    offloader._handles[(1, 0)] = previous
    scheduled = []
    monkeypatch.setattr(offloader, "_schedule", scheduled.append)

    offloader._prefetch_previous(current)

    assert scheduled == []


def test_reusable_activation_begin_forward_retires_stale_prefetch() -> None:
    offloader = _offloader_without_cuda()
    event = _FakeEvent()
    offloader._handles[(1, 0)] = _handle(1, consumed=True, gpu=torch.zeros(1), h2d_event=event)

    offloader.begin_forward()

    assert event.synchronize_calls == 1
    assert offloader._handles == {}


def test_reusable_activation_begin_forward_recovers_from_an_aborted_backward(caplog) -> None:
    # A forward whose backward never ran (skipped non-finite step, caught OOM,
    # partial autograd.grad) used to raise here forever after.
    offloader = _offloader_without_cuda()
    event = _FakeEvent()
    offloader._handles[(1, 0)] = _handle(1, consumed=False, gpu=torch.zeros(1), h2d_event=event)

    with caplog.at_level(logging.WARNING, logger="musubi_tuner.minimax_h3.activation_offload"):
        offloader.begin_forward()

    assert offloader._handles == {}
    assert event.synchronize_calls == 1
    assert "backward never ran" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="musubi_tuner.minimax_h3.activation_offload"):
        offloader.begin_forward()
    assert caplog.text == ""


def test_reusable_activation_reset_is_callable_without_a_pending_forward() -> None:
    offloader = _offloader_without_cuda()
    offloader.reset()
    assert offloader._handles == {}
