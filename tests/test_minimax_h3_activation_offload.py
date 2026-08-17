import logging

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from musubi_tuner.minimax_h3.activation_offload import (
    COMPRESSION_MIN_BYTES,
    ReusableActivationOffloader,
    _fp8_payload_layout,
    _OffloadedTensor,
    _pinned_capacity,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="reusable activation offload requires CUDA")


class _FakeEvent:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


def _offloader_without_cuda(offload_dtype: str = "none") -> ReusableActivationOffloader:
    offloader = object.__new__(ReusableActivationOffloader)
    offloader._handles = {}
    offloader._pool = {}
    offloader._offload_dtype = offload_dtype
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


def test_fp8_payload_layout_reserves_an_aligned_scale_slot() -> None:
    # One byte per element, then the float32 scale at a four-byte-aligned offset.
    assert _fp8_payload_layout(1024) == (1024, 1028)
    assert _fp8_payload_layout(1023) == (1024, 1028)
    assert _fp8_payload_layout(1) == (4, 8)


def test_pinned_capacity_keeps_the_scale_slot_out_of_the_power_of_two() -> None:
    numel = 1 << 20
    # Uncompressed sizing is unchanged: a pure power of two in element units.
    assert _pinned_capacity(numel, numel) == numel
    assert _pinned_capacity(numel - 1, numel - 1) == numel
    # Compressed sizing is one byte per element plus the scale, not a full
    # doubling of the pinned block.
    _, required = _fp8_payload_layout(numel)
    assert _pinned_capacity(numel, required) == numel + 4
    # Half the bf16 payload the same activation would otherwise pin.
    assert _pinned_capacity(numel, required) < numel * 2


def test_offload_dtype_rejects_unknown_values() -> None:
    offloader = _offloader_without_cuda()
    with pytest.raises(ValueError, match="unsupported activation offload dtype"):
        offloader.set_offload_dtype("int8")


def test_offload_dtype_default_never_compresses() -> None:
    offloader = _offloader_without_cuda()
    assert offloader.offload_dtype == "none"
    assert not offloader._should_compress(torch.zeros(1 << 21, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    ("dtype", "numel", "expected"),
    [
        (torch.bfloat16, COMPRESSION_MIN_BYTES, True),
        (torch.float16, COMPRESSION_MIN_BYTES, True),
        # Below the byte threshold the payload contributes nothing to bandwidth,
        # so exactness is kept for free.
        (torch.bfloat16, 16, False),
        # fp32 is never silently squashed to three mantissa bits.
        (torch.float32, COMPRESSION_MIN_BYTES, False),
        (torch.int64, COMPRESSION_MIN_BYTES, False),
        (torch.bool, COMPRESSION_MIN_BYTES, False),
    ],
)
def test_fp8_compression_selects_only_large_half_precision_tensors(dtype, numel, expected) -> None:
    offloader = _offloader_without_cuda("fp8_e4m3")
    tensor = torch.zeros(numel, dtype=dtype)

    assert offloader._should_compress(tensor) is expected


def test_fp8_round_trip_stays_within_e4m3_resolution() -> None:
    torch.manual_seed(0)
    tensor = torch.randn(512, 768, dtype=torch.bfloat16)
    handle = _handle(0, consumed=False)
    handle.compressed = True
    handle.shape = tensor.shape
    handle.dtype = tensor.dtype
    handle.numel = tensor.numel()

    buffer = ReusableActivationOffloader._quantize(tensor)
    restored = ReusableActivationOffloader._dequantize(buffer, handle)

    assert buffer.dtype is torch.uint8
    assert buffer.numel() == _fp8_payload_layout(tensor.numel())[1]
    assert restored.dtype is tensor.dtype
    assert restored.shape == tensor.shape
    reference = tensor.float()
    error = (restored.float() - reference).abs()
    # e4m3 keeps three mantissa bits, so a well-scaled value carries at most a
    # 2^-4 relative error; the tail is measured against the per-tensor amax.
    assert (error <= 0.07 * reference.abs() + 1e-3 * reference.abs().amax()).all()
    assert error.max() < reference.abs().amax()


def test_fp8_round_trip_is_deterministic_and_preserves_zero_and_nan() -> None:
    tensor = torch.randn(64, 64, dtype=torch.bfloat16)
    first = ReusableActivationOffloader._quantize(tensor)
    second = ReusableActivationOffloader._quantize(tensor)
    assert torch.equal(first, second)

    handle = _handle(0, consumed=False)
    handle.compressed, handle.shape, handle.dtype, handle.numel = True, tensor.shape, tensor.dtype, tensor.numel()

    zeros = torch.zeros_like(tensor)
    assert not ReusableActivationOffloader._dequantize(ReusableActivationOffloader._quantize(zeros), handle).any()

    poisoned = tensor.clone()
    poisoned[0, 0] = float("nan")
    restored = ReusableActivationOffloader._dequantize(ReusableActivationOffloader._quantize(poisoned), handle)
    assert torch.isnan(restored[0, 0])
    assert torch.isfinite(restored[1:]).all()


def _simulated_offload_hooks(compress: bool, seen: list[torch.Tensor]):
    """Drive the real pack-side writer and unpack-side reader over a memcpy.

    This is the compressed path with the CUDA plumbing removed: ``_quantize``
    lays the bytes out, a plain copy stands in for the D2H/H2D round trip, and
    ``_dequantize`` reads them back through the same offset arithmetic. Any
    disagreement between the writer's and the reader's view of the buffer shows
    up here without a GPU.
    """
    offloader = _offloader_without_cuda("fp8_e4m3" if compress else "none")

    def pack(tensor: torch.Tensor):
        seen.append(tensor)
        if not offloader._should_compress(tensor):
            return tensor
        handle = _handle(0, consumed=False)
        handle.compressed = True
        handle.shape, handle.dtype, handle.numel = tensor.shape, tensor.dtype, tensor.numel()
        packed = ReusableActivationOffloader._quantize(tensor)
        staged = torch.empty(packed.shape, dtype=packed.dtype)
        staged.copy_(packed)
        return (staged, handle)

    def unpack(value):
        if isinstance(value, tuple):
            return ReusableActivationOffloader._dequantize(*value)
        return value

    return pack, unpack


def _checkpointed_block(hidden: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    def block(value: torch.Tensor) -> torch.Tensor:
        normed = value * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(value.dtype)
        return value + torch.nn.functional.silu(normed @ w1) @ w2

    return checkpoint(block, hidden, use_reentrant=False)


def test_checkpoint_offloads_only_the_block_boundary_hidden_states() -> None:
    # The 1 MiB threshold is chosen for the population this hook actually sees.
    # Non-reentrant checkpoint installs its own inner saved-tensor hooks, so
    # every tensor saved inside the block goes to the recomputation and the only
    # thing reaching our pack hook is the block input that _NoopSaveInputs saves.
    # If a torch upgrade ever changes that, the threshold and the whole "one big
    # well-conditioned tensor per block" premise need revisiting.
    rows = COMPRESSION_MIN_BYTES // 2 // 256
    w1 = (torch.randn(256, 256, dtype=torch.bfloat16) / 16).requires_grad_(True)
    w2 = (torch.randn(256, 256, dtype=torch.bfloat16) / 16).requires_grad_(True)
    hidden = torch.randn(rows, 256, dtype=torch.bfloat16, requires_grad=True)
    seen: list[torch.Tensor] = []
    pack, unpack = _simulated_offload_hooks(False, seen)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = _checkpointed_block(hidden, w1, w2)
    output.float().pow(2).sum().backward()

    # Newer torch versions add zero-size bookkeeping saves alongside the block
    # input; only the boundary hidden states may be large enough to compress.
    large = [tensor for tensor in seen if tensor.numel() * tensor.element_size() >= COMPRESSION_MIN_BYTES]
    assert [tuple(tensor.shape) for tensor in large] == [tuple(hidden.shape)]
    assert all(tensor.numel() == 0 for tensor in seen if tensor.numel() * tensor.element_size() < COMPRESSION_MIN_BYTES)


def test_fp8_layout_survives_a_round_trip_through_a_flat_byte_buffer() -> None:
    # Writer and reader must agree on where the payload ends and the scale
    # begins. A disagreement reads a garbage scale and shows up as a gross
    # error here, distinguishing a layout bug from ordinary fp8 noise.
    for numel in (COMPRESSION_MIN_BYTES, COMPRESSION_MIN_BYTES + 2, COMPRESSION_MIN_BYTES + 3):
        tensor = torch.randn(numel, dtype=torch.bfloat16) * 7.5
        handle = _handle(0, consumed=False)
        handle.compressed = True
        handle.shape, handle.dtype, handle.numel = tensor.shape, tensor.dtype, tensor.numel()

        packed = ReusableActivationOffloader._quantize(tensor)
        staged = torch.empty(packed.shape, dtype=packed.dtype)
        staged.copy_(packed)
        restored = ReusableActivationOffloader._dequantize(staged, handle)

        scale_offset, total = _fp8_payload_layout(numel)
        assert packed.numel() == total
        # The recovered scale is the one the writer applied, so the magnitudes
        # match to fp8 resolution rather than being off by orders of magnitude.
        assert restored.float().abs().max() == pytest.approx(tensor.float().abs().max(), rel=0.05)
        error = (restored.float() - tensor.float()).abs().max()
        assert error < 0.05 * tensor.float().abs().max()
        assert staged[scale_offset : scale_offset + 4].view(torch.float32).item() > 0


def test_fp8_recompute_error_stays_at_fp8_noise_for_a_realistic_block() -> None:
    # Measured against the whole-tensor scale, not elementwise: gradient entries
    # near zero have no meaningful relative tolerance, and it is the size of the
    # perturbation relative to the gradient that matters for training.
    rows = COMPRESSION_MIN_BYTES // 2 // 256

    def run(compress: bool):
        torch.manual_seed(0)
        w1 = (torch.randn(256, 256, dtype=torch.bfloat16) / 16).requires_grad_(True)
        w2 = (torch.randn(256, 256, dtype=torch.bfloat16) / 16).requires_grad_(True)
        hidden = torch.randn(rows, 256, dtype=torch.bfloat16, requires_grad=True)
        pack, unpack = _simulated_offload_hooks(compress, [])
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            output = _checkpointed_block(hidden, w1, w2)
        output.float().pow(2).sum().backward()
        return hidden.grad.float(), w1.grad.float(), w2.grad.float()

    exact_grads, fp8_grads = run(False), run(True)
    # Guard against a vacuous pass: the quantization must really have happened.
    assert any(not torch.equal(exact, approximate) for exact, approximate in zip(exact_grads, fp8_grads))
    for exact, approximate in zip(exact_grads, fp8_grads):
        assert (approximate - exact).abs().max() < 0.05 * exact.abs().max()


@requires_cuda
def test_fp8_offload_halves_the_pinned_pool_and_matches_the_exact_recompute() -> None:
    # Same block as the CPU accuracy test, run through the real pinned buffers
    # and side streams, so this covers the plumbing rather than the numerics.
    device = torch.device("cuda")
    rows = COMPRESSION_MIN_BYTES // 2 // 256  # bf16 block input exactly at the threshold

    def run(offload_dtype: str):
        torch.manual_seed(0)
        w1 = (torch.randn(256, 256, device=device, dtype=torch.bfloat16) / 16).requires_grad_(True)
        w2 = (torch.randn(256, 256, device=device, dtype=torch.bfloat16) / 16).requires_grad_(True)
        hidden = torch.randn(rows, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
        offloader = ReusableActivationOffloader(offload_dtype)
        with offloader.context(0):
            output = _checkpointed_block(hidden, w1, w2)
        output.float().pow(2).sum().backward()
        return offloader.pooled_bytes, hidden.grad.float(), w1.grad.float(), w2.grad.float()

    exact_bytes, *exact_grads = run("none")
    fp8_bytes, *fp8_grads = run("fp8_e4m3")

    # The single offloaded tensor is a large bf16 activation, so the pinned
    # payload drops to one byte per element plus a four-byte scale.
    assert fp8_bytes < exact_bytes * 0.6
    for exact, approximate in zip(exact_grads, fp8_grads):
        assert (approximate - exact).abs().max() < 0.05 * exact.abs().max()


@requires_cuda
def test_fp8_offload_leaves_small_saved_tensors_bit_exact() -> None:
    offloader = ReusableActivationOffloader("fp8_e4m3")
    device = torch.device("cuda")
    weight = torch.randn(8, 8, device=device, dtype=torch.bfloat16, requires_grad=True)
    inputs = torch.randn(4, 8, device=device, dtype=torch.bfloat16, requires_grad=True)

    with offloader.context(0):
        hidden = (inputs @ weight).tanh()
    hidden.sum().backward()

    assert all(not handle.compressed for handle in offloader._handles.values())
    assert all(buffer.dtype is not torch.uint8 for buffer in offloader._pool.values())


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


def test_cpu_offload_dtype_defaults_to_none_and_parses_fp8() -> None:
    from musubi_tuner.minimax_h3_train_network import create_parser

    assert create_parser().parse_args(["--sdpa"]).gradient_checkpointing_cpu_offload_dtype == "none"
    parsed = create_parser().parse_args(["--sdpa", "--gradient_checkpointing_cpu_offload_dtype", "fp8_e4m3"])
    assert parsed.gradient_checkpointing_cpu_offload_dtype == "fp8_e4m3"


def test_cpu_offload_dtype_requires_the_reusable_cpu_offload_path() -> None:
    from musubi_tuner.minimax_h3_train_network import MiniMaxH3NetworkTrainer, create_parser

    base = ["--sdpa", "--gradient_checkpointing_cpu_offload_dtype", "fp8_e4m3"]
    valid = create_parser().parse_args(
        base + ["--gradient_checkpointing", "--gradient_checkpointing_cpu_offload", "--h3_reusable_activation_offload"]
    )
    MiniMaxH3NetworkTrainer().handle_model_specific_args(valid)

    without_offload = create_parser().parse_args(base)
    with pytest.raises(ValueError, match="requires --gradient_checkpointing"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(without_offload)

    without_reusable = create_parser().parse_args(base + ["--gradient_checkpointing", "--gradient_checkpointing_cpu_offload"])
    with pytest.raises(ValueError, match="requires --h3_reusable_activation_offload"):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(without_reusable)
