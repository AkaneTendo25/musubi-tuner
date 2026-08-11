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
            next_handle = self._handles.get((value.block_index - 1, value.ordinal))
            if next_handle is not None and next_handle.gpu is None:
                self._schedule(next_handle)
            return result

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield

    def _schedule(self, handle: _OffloadedTensor) -> None:
        # Allocate on the consuming compute stream. The private stream only
        # performs the copy; after its event is waited below, allocation, use,
        # free, and allocator reuse are all ordered on the compute stream.
        # Allocating here inside ``self._stream`` without record_stream() lets
        # the caching allocator recycle the block while recomputation is still
        # reading it on the compute stream.
        handle.gpu = torch.empty(handle.cpu.shape, dtype=handle.cpu.dtype, device=handle.device)
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(handle.d2h_event)
            handle.gpu.copy_(handle.cpu, non_blocking=True)
            handle.h2d_event = torch.cuda.Event()
            handle.h2d_event.record(self._stream)

    def _materialize(self, handle: _OffloadedTensor) -> torch.Tensor:
        if handle.gpu is None:
            self._schedule(handle)
        assert handle.gpu is not None and handle.h2d_event is not None
        torch.cuda.current_stream(handle.device).wait_event(handle.h2d_event)
        result = handle.gpu
        handle.gpu = None
        handle.h2d_event = None
        return result

    @property
    def pooled_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self._pool.values())
