"""Off-thread checkpoint serialization.

Periodic LoRA saves stall the training loop for seconds: the state dict is
cloned to CPU, then :func:`model_utils.precalculate_safetensors_hashes`
serializes the whole tensor set into memory to hash it, and finally
``save_file`` serializes it a second time onto disk. Only the clone needs to
happen on the training thread (it is what makes the snapshot consistent with
the current step); hashing and writing can run on a background thread.

:class:`AsyncCheckpointSaver` is a single-worker, single-slot queue: saves are
written in submission order, never concurrently, and a submit that arrives
while an earlier save is still writing blocks until that save finishes. Worker
failures are re-raised on the next :meth:`submit` or on :meth:`join`, so an
aborted write is always reported rather than silently dropped.
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Any, Callable, Dict, Optional

import torch


def write_state_dict_file(state_dict: Dict[str, Any], file: str, metadata: Optional[Dict[str, str]]) -> None:
    """Serialize ``state_dict`` to ``file`` atomically.

    ``.safetensors`` targets get the sd-webui-additional-networks hashes added
    to ``metadata`` first. The bytes always land on a temporary sibling file
    that is then :func:`os.replace`-d into place, so a reader never observes a
    half-written checkpoint and an aborted write leaves no truncated file at
    the real path.
    """
    if metadata is not None and len(metadata) == 0:
        metadata = None

    directory = os.path.dirname(os.path.abspath(file))
    os.makedirs(directory, exist_ok=True)
    tmp_file = os.path.join(directory, f".{os.path.basename(file)}.tmp{os.getpid()}")

    try:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            from musubi_tuner.utils import model_utils

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, tmp_file, metadata)
        else:
            torch.save(state_dict, tmp_file)

        os.replace(tmp_file, file)
    except BaseException:
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except OSError:
            pass
        raise


def snapshot_state_dict(state_dict: Dict[str, Any], dtype: Optional[torch.dtype]) -> Dict[str, Any]:
    """Detach/clone every tensor onto the CPU so later mutation cannot race the write."""
    snapshot: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().to("cpu", copy=True)
            if dtype is not None:
                value = value.to(dtype)
            if not value.is_contiguous():
                value = value.contiguous()
        snapshot[key] = value
    return snapshot


class AsyncCheckpointSaver:
    """One worker thread, one queued job. Saves stay ordered and never overlap."""

    def __init__(self, name: str = "checkpoint-saver") -> None:
        self._queue: "queue.Queue[Optional[Callable[[], None]]]" = queue.Queue()
        # One outstanding job at a time: the slot is released only once the job has
        # finished writing, so submit() blocks while an earlier save is in flight.
        self._slot = threading.Semaphore(1)
        self._error: Optional[BaseException] = None
        self._error_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    # -- worker ---------------------------------------------------------
    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                job()
            except BaseException as exc:  # surfaced at the next submit()/join()
                with self._error_lock:
                    if self._error is None:
                        self._error = exc
            finally:
                self._slot.release()

    # -- api ------------------------------------------------------------
    def raise_if_failed(self) -> None:
        with self._error_lock:
            error = self._error
            self._error = None
        if error is not None:
            raise RuntimeError(f"asynchronous checkpoint save failed: {error}") from error

    def submit(self, job: Callable[[], None]) -> None:
        """Queue ``job``; blocks while a previously submitted save is still writing."""
        if self._closed:
            raise RuntimeError("AsyncCheckpointSaver is closed")
        self._slot.acquire()  # blocks until the previous save has finished writing
        try:
            self.raise_if_failed()
        except BaseException:
            self._slot.release()
            raise
        self._queue.put(job)

    def submit_state_dict(self, state_dict: Dict[str, Any], file: str, metadata: Optional[Dict[str, str]]) -> None:
        self.submit(lambda: write_state_dict_file(state_dict, file, metadata))

    def wait(self) -> None:
        """Block until every submitted save has been written, then re-raise failures."""
        self._slot.acquire()
        self._slot.release()
        self.raise_if_failed()

    def join(self, timeout: Optional[float] = None) -> None:
        """Drain, stop the worker and re-raise the first worker failure."""
        if self._closed:
            self.raise_if_failed()
            return
        self._closed = True
        try:
            self._queue.put(None)
            self._thread.join(timeout)
        finally:
            self.raise_if_failed()

    # convenience for ``with`` usage in tests
    def __enter__(self) -> "AsyncCheckpointSaver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.join()
