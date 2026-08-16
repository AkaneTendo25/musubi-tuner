"""Reusable checkpoint-activation offload for MiniMax H3."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


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


class ReusableActivationOffloader:
    """Reuse pinned checkpoint-input buffers and prefetch reverse-order inputs."""

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("reusable H3 activation offload requires CUDA")
        # Keep one geometrically grown allocation per saved-tensor position.
        # Keying the pool by the exact shape retains a complete set of buffers
        # for every sequence length encountered by a multi-bucket dataset.  A
        # real H3 buffer is large enough that this can exhaust pinned host RAM
        # after only a handful of distinct buckets (and once per DDP process).
        self._pool: dict[tuple[int, int, torch.dtype], torch.Tensor] = {}
        self._handles: dict[tuple[int, int], _OffloadedTensor] = {}
        self._stream = torch.cuda.Stream()
        # A dedicated D2H stream keeps the save-side copy off the compute stream,
        # so offloading a block's activations overlaps the next block's forward
        # instead of serializing with it. It stays separate from the H2D stream:
        # the two directions are only ever active in different phases (D2H during
        # forward, H2D prefetch during backward), but a shared stream would still
        # order a cold-start ``_schedule`` behind whatever D2H work was queued.
        self._d2h_stream = torch.cuda.Stream()

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
                "backpropagated and its recomputation will fail",
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
            key = (block_index, current_ordinal, tensor.dtype)
            storage = self._pool.get(key)
            required = tensor.numel()
            if storage is None or storage.numel() < required:
                # Power-of-two growth avoids repeatedly leaving differently
                # sized allocations in PyTorch's pinned-memory allocator while
                # bounding retained capacity to less than 2x the largest input.
                capacity = 1 << max(required - 1, 0).bit_length()
                storage = torch.empty(capacity, dtype=tensor.dtype, device="cpu", pin_memory=True)
                self._pool[key] = storage
            cpu = storage[:required].view(tensor.shape)
            compute_stream = torch.cuda.current_stream(tensor.device)
            # The side stream may not read the activation before compute has
            # produced it.
            produced = torch.cuda.Event()
            produced.record(compute_stream)
            event = torch.cuda.Event()
            with torch.cuda.stream(self._d2h_stream):
                self._d2h_stream.wait_event(produced)
                cpu.copy_(tensor, non_blocking=True)
                event.record(self._d2h_stream)
            # The GPU activation is now read by a stream other than the one that
            # allocated it; without this the caching allocator may hand its
            # memory to a later compute-stream allocation while the copy is in
            # flight. Same hand-off the H2D side registers in _materialize.
            tensor.record_stream(self._d2h_stream)
            handle = _OffloadedTensor(block_index, current_ordinal, cpu, tensor.device, event)
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
            handle.gpu = torch.empty(handle.cpu.shape, dtype=handle.cpu.dtype, device=handle.device)
            handle.gpu.copy_(handle.cpu, non_blocking=True)
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
    def pooled_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self._pool.values())
