"""Reusable checkpoint-activation offload for MiniMax H3."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

OFFLOAD_DTYPES = ("none", "fp8_e4m3")

# Below this payload size a tensor contributes nothing measurable to PCIe
# traffic or pinned residency, so it is never worth trading accuracy for.
# Normalization statistics and other small per-block saves stay bit-exact.
COMPRESSION_MIN_BYTES = 1 << 20

# Mirrors the scale convention of ``modules/fp8_optimization_utils``:
# ``scale = amax / finfo(fp8).max``, clamped away from zero, quantized value is
# ``clamp(x / scale)`` and the dequantized value is ``q * scale``.
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0
_SCALE_MIN = 1e-8
_SCALE_BYTES = 4


def _fp8_payload_layout(numel: int) -> tuple[int, int]:
    """Return ``(scale_offset, total_bytes)`` for a packed fp8 payload.

    The scale is carried in the same buffer as the payload so a single copy
    moves both. Its offset is rounded up to four bytes because ``Tensor.view``
    to ``float32`` requires a four-byte-aligned storage offset.
    """
    scale_offset = (numel + _SCALE_BYTES - 1) // _SCALE_BYTES * _SCALE_BYTES
    return scale_offset, scale_offset + _SCALE_BYTES


def _pinned_capacity(numel: int, required: int) -> int:
    """Size the pooled pinned buffer for an activation of ``numel`` elements.

    Power-of-two growth avoids repeatedly leaving differently sized allocations
    in PyTorch's pinned-memory allocator while bounding retained capacity to
    less than 2x the largest input. The fp8 scale slot is added on top of that
    power of two rather than folded into it: activation payloads sit on or just
    under a power of two, and rounding four extra bytes up would double the very
    allocation this compression exists to halve.
    """
    return max(1 << max(numel - 1, 0).bit_length(), numel) + max(required - numel, 0)


@dataclass
class _OffloadedTensor:
    block_index: int
    ordinal: int
    cpu: torch.Tensor
    device: torch.device
    d2h_event: torch.cuda.Event
    gpu: torch.Tensor | None = None
    h2d_event: torch.cuda.Event | None = None
    consumed: bool = False
    # Set only for fp8-compressed payloads; ``cpu`` is then a flat uint8 view
    # holding the fp8 bytes followed by the float32 per-tensor scale.
    compressed: bool = False
    shape: torch.Size | None = None
    dtype: torch.dtype | None = None
    numel: int = 0


class ReusableActivationOffloader:
    """Reuse pinned checkpoint-input buffers and prefetch reverse-order inputs."""

    def __init__(self, offload_dtype: str = "none") -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("reusable H3 activation offload requires CUDA")
        # Keep one geometrically grown allocation per saved-tensor position.
        # Keying the pool by the exact shape retains a complete set of buffers
        # for every sequence length encountered by a multi-bucket dataset.  A
        # real H3 buffer is large enough that this can exhaust pinned host RAM
        # after only a handful of distinct buckets (and once per DDP process).
        self._pool: dict[tuple[int, int, torch.dtype, bool], torch.Tensor] = {}
        self._handles: dict[tuple[int, int], _OffloadedTensor] = {}
        self._offload_dtype = "none"
        self.set_offload_dtype(offload_dtype)
        self._stream = torch.cuda.Stream()
        # A dedicated D2H stream keeps the save-side copy off the compute stream,
        # so offloading a block's activations overlaps the next block's forward
        # instead of serializing with it. It stays separate from the H2D stream:
        # the two directions are only ever active in different phases (D2H during
        # forward, H2D prefetch during backward), but a shared stream would still
        # order a cold-start ``_schedule`` behind whatever D2H work was queued.
        self._d2h_stream = torch.cuda.Stream()

    def set_offload_dtype(self, offload_dtype: str) -> None:
        """Select the wire dtype of the CPU-resident copy.

        ``none`` keeps today's behaviour: the activation is stored in its own
        dtype and the recomputation sees a bit-exact copy. ``fp8_e4m3`` casts
        large floating payloads to ``torch.float8_e4m3fn`` on the GPU before the
        D2H copy, halving both PCIe traffic and pinned residency for bf16/fp16
        activations at the cost of a slightly lossy recomputation.
        """
        if offload_dtype not in OFFLOAD_DTYPES:
            raise ValueError(f"unsupported activation offload dtype {offload_dtype!r}; expected one of {OFFLOAD_DTYPES}")
        if offload_dtype != "none" and not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("fp8 checkpoint-activation offload requires a PyTorch build with torch.float8_e4m3fn")
        self._offload_dtype = offload_dtype
        # Buffers are keyed by their storage layout, so a switch never reuses a
        # pinned block sized for the other representation.
        self._pool.clear()

    def _should_compress(self, tensor: torch.Tensor) -> bool:
        if self._offload_dtype != "fp8_e4m3":
            return False
        # Integer and boolean saves (index tensors, masks) carry no scale and
        # must survive exactly; anything already at or below one byte per
        # element has nothing left to compress.
        if tensor.dtype not in (torch.bfloat16, torch.float16):
            # fp32 activations are reachable (an fp32 model, or a test harness),
            # and squashing 24 mantissa bits to 3 is not a trade this flag
            # promises, so they pass through in their own dtype.
            return False
        return tensor.numel() * tensor.element_size() >= COMPRESSION_MIN_BYTES

    def begin_forward(self) -> None:
        unconsumed = sum(1 for handle in self._handles.values() if not handle.consumed)
        if unconsumed:
            # Raising here used to wedge training permanently: any forward whose
            # backward never ran (a skipped non-finite step, a caught OOM, a
            # partial ``autograd.grad``) left the handles behind and every later
            # step failed with the same misleading message. Recover instead, but
            # warn loudly -- a genuine double forward before backward shows up
            # here too, and the discarded activations would be its symptom.
            logger.warning(
                "reusable H3 activation offload is discarding %d saved activation handle(s) from a previous "
                "grad-enabled forward whose backward never ran (skipped step, caught OOM, or partial autograd.grad); "
                "if you did not expect an aborted backward, a second forward was started before the first was "
                "backpropagated and its gradients will be silently wrong: the pooled buffers are reused, so the "
                "recomputation succeeds on the second forward's data",
                unconsumed,
            )
        self.reset()

    def reset(self) -> None:
        """Retire every outstanding handle so the next forward starts clean."""
        # A saved tensor can be unpacked more than once, and autograd is not
        # required to consume equal ordinals in strict block-reverse order.
        # Retire any speculative copy left behind before its pinned source is
        # reused by the next forward.
        for handle in self._handles.values():
            if handle.gpu is not None:
                assert handle.h2d_event is not None
                handle.h2d_event.synchronize()
        self._handles.clear()

    @contextmanager
    def context(self, block_index: int):
        ordinal = 0

        def pack(tensor: torch.Tensor):
            nonlocal ordinal
            if tensor.device.type != "cuda" or tensor.numel() == 0:
                return tensor
            current_ordinal = ordinal
            ordinal += 1
            compressed = self._should_compress(tensor)
            key = (block_index, current_ordinal, tensor.dtype, compressed)
            storage = self._pool.get(key)
            if compressed:
                # One byte per element plus the four-byte per-tensor scale.
                _, required = _fp8_payload_layout(tensor.numel())
                storage_dtype = torch.uint8
            else:
                required = tensor.numel()
                storage_dtype = tensor.dtype
            if storage is None or storage.numel() < required:
                capacity = _pinned_capacity(tensor.numel(), required)
                storage = torch.empty(capacity, dtype=storage_dtype, device="cpu", pin_memory=True)
                self._pool[key] = storage
            cpu = storage[:required] if compressed else storage[:required].view(tensor.shape)
            compute_stream = torch.cuda.current_stream(tensor.device)
            # The side stream may not read the activation before compute has
            # produced it.
            produced = torch.cuda.Event()
            produced.record(compute_stream)
            event = torch.cuda.Event()
            with torch.cuda.stream(self._d2h_stream):
                self._d2h_stream.wait_event(produced)
                # Quantizing here rather than before ``produced`` keeps the cast
                # off the compute stream -- the whole point of the dedicated D2H
                # stream is that offloading a block overlaps the next block's
                # forward -- and the wait above already orders it after the
                # producer, so it cannot read a half-written activation.
                payload = self._quantize(tensor) if compressed else tensor
                cpu.copy_(payload, non_blocking=True)
                event.record(self._d2h_stream)
            # The GPU activation is now read by a stream other than the one that
            # allocated it; without this the caching allocator may hand its
            # memory to a later compute-stream allocation while the copy is in
            # flight. Same hand-off the H2D side registers in _materialize.
            tensor.record_stream(self._d2h_stream)
            handle = _OffloadedTensor(
                block_index,
                current_ordinal,
                cpu,
                tensor.device,
                event,
                compressed=compressed,
                shape=tensor.shape,
                dtype=tensor.dtype,
                numel=tensor.numel(),
            )
            self._handles[(block_index, current_ordinal)] = handle
            return handle

        def unpack(value):
            if not isinstance(value, _OffloadedTensor):
                return value
            result = self._materialize(value)
            value.consumed = True
            self._prefetch_previous(value)
            return result

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield

    @staticmethod
    def _quantize(tensor: torch.Tensor) -> torch.Tensor:
        """Pack ``tensor`` into a flat uint8 buffer of fp8 bytes plus its scale.

        Round-to-nearest-even is the only rounding involved, so the packed bytes
        are a deterministic function of the input.
        """
        numel = tensor.numel()
        scale_offset, total = _fp8_payload_layout(numel)
        buffer = torch.empty(total, dtype=torch.uint8, device=tensor.device)
        payload = buffer[:numel].view(_FP8_DTYPE).view(tensor.shape)
        scale_slot = buffer[scale_offset : scale_offset + _SCALE_BYTES].view(torch.float32)

        amax = tensor.detach().abs().amax().to(torch.float32)
        # A non-finite amax would scale every finite value to zero and hide the
        # very NaN the non-finite-step guard is watching for; fall back to a
        # unit scale so the NaN survives the round trip.
        amax = torch.where(torch.isfinite(amax), amax, torch.ones_like(amax))
        scale = (amax / _FP8_MAX).clamp_min(_SCALE_MIN)
        # Dividing in the activation's own dtype avoids a transient float32 copy
        # four times the size of the payload we are trying to shrink. The stored
        # scale is the rounded one that was actually applied, so dequantization
        # is the exact inverse operation.
        narrow_scale = scale.to(tensor.dtype)
        scale_slot.copy_(narrow_scale.to(torch.float32).reshape(1))
        payload.copy_((tensor.detach() / narrow_scale).clamp_(-_FP8_MAX, _FP8_MAX))
        return buffer

    @staticmethod
    def _dequantize(buffer: torch.Tensor, handle: _OffloadedTensor) -> torch.Tensor:
        assert handle.shape is not None and handle.dtype is not None
        scale_offset, _ = _fp8_payload_layout(handle.numel)
        payload = buffer[: handle.numel].view(_FP8_DTYPE).view(handle.shape)
        scale = buffer[scale_offset : scale_offset + _SCALE_BYTES].view(torch.float32)
        return payload.to(handle.dtype).mul_(scale.to(handle.dtype))

    def _prefetch_previous(self, handle: _OffloadedTensor) -> None:
        next_handle = self._handles.get((handle.block_index - 1, handle.ordinal))
        if next_handle is not None and not next_handle.consumed and next_handle.gpu is None:
            self._schedule(next_handle)

    def _schedule(self, handle: _OffloadedTensor) -> None:
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(handle.d2h_event)
            # Allocation and the asynchronous writer share one stream.  The
            # consumer is registered in _materialize after it waits for this
            # copy, giving the caching allocator both sides of the hand-off.
            staged = torch.empty(handle.cpu.shape, dtype=handle.cpu.dtype, device=handle.device)
            staged.copy_(handle.cpu, non_blocking=True)
            # Dequantization stays on the prefetch stream: it is ordered after
            # the copy it consumes, and the compute stream waits on the event
            # recorded below, so the recomputation never sees fp8 bytes.
            handle.gpu = self._dequantize(staged, handle) if handle.compressed else staged
            handle.h2d_event = torch.cuda.Event()
            handle.h2d_event.record(self._stream)

    def _materialize(self, handle: _OffloadedTensor) -> torch.Tensor:
        if handle.gpu is None:
            self._schedule(handle)
        assert handle.gpu is not None and handle.h2d_event is not None
        compute_stream = torch.cuda.current_stream(handle.device)
        compute_stream.wait_event(handle.h2d_event)
        result = handle.gpu
        result.record_stream(compute_stream)
        handle.gpu = None
        handle.h2d_event = None
        return result

    @property
    def offload_dtype(self) -> str:
        return self._offload_dtype

    @property
    def pooled_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self._pool.values())
