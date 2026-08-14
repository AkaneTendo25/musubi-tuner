"""Reusable checkpoint-activation offload for MiniMax H3."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


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

    def begin_forward(self) -> None:
        if any(not handle.consumed for handle in self._handles.values()):
            raise RuntimeError("reusable H3 activation buffers cannot start another grad-enabled forward before backward")
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
            cpu.copy_(tensor, non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(tensor.device))
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
