import logging

import pytest
import torch

from musubi_tuner.minimax_h3.activation_offload import _OffloadedTensor, ReusableActivationOffloader

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="reusable activation offload requires CUDA")


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


@requires_cuda
def test_reusable_activation_round_trips_saved_tensors_through_the_side_streams() -> None:
    # The pack path copies D2H on a dedicated stream; the saved activation must
    # still arrive intact for the recomputed backward.
    offloader = ReusableActivationOffloader()
    device = torch.device("cuda")
    weight = torch.randn(8, 8, device=device, dtype=torch.float32, requires_grad=True)
    inputs = torch.randn(4, 8, device=device, dtype=torch.float32, requires_grad=True)

    reference = (inputs @ weight).tanh().sum()
    reference.backward()
    expected_input_grad = inputs.grad.clone()
    expected_weight_grad = weight.grad.clone()
    inputs.grad = None
    weight.grad = None

    with offloader.context(0):
        hidden = (inputs @ weight).tanh()
    hidden.sum().backward()

    torch.testing.assert_close(inputs.grad, expected_input_grad)
    torch.testing.assert_close(weight.grad, expected_weight_grad)


@requires_cuda
def test_reusable_activation_d2h_and_h2d_use_distinct_side_streams() -> None:
    offloader = ReusableActivationOffloader()
    compute = torch.cuda.current_stream()

    assert offloader._d2h_stream != compute
    assert offloader._stream != compute
    assert offloader._d2h_stream != offloader._stream
