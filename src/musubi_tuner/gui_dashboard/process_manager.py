"""Subprocess manager for caching and training processes."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from collections import deque
from enum import Enum
from typing import Literal, Optional

logger = logging.getLogger(__name__)

ProcessType = Literal["cache_latents", "cache_text", "cache_dino", "training", "inference", "slider_training"]

# Windows-specific flags for clean subprocess shutdown
_CREATION_FLAGS = 0
if sys.platform == "win32":
    _CREATION_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP


class ProcessState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    FINISHED = "finished"
    ERROR = "error"


class ManagedProcess:
    """Wraps a subprocess with state tracking and log buffering."""

    def __init__(self, cmd: list[str], cwd: Optional[str] = None):
        self.cmd = cmd
        self.cwd = cwd
        self.state = ProcessState.IDLE
        self.exit_code: Optional[int] = None
        self.logs: deque[str] = deque(maxlen=5000)
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self.state == ProcessState.RUNNING:
                raise RuntimeError("Process already running")

            self.state = ProcessState.RUNNING
            self.exit_code = None
            self.logs.clear()
            self.logs.append(f"$ {' '.join(self.cmd)}\n")

            self._proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.cwd,
                creationflags=_CREATION_FLAGS,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            self._reader_thread = threading.Thread(
                target=self._read_output, daemon=True
            )
            self._reader_thread.start()

    def _read_output(self):
        try:
            assert self._proc and self._proc.stdout
            for line in self._proc.stdout:
                self.logs.append(line)
            self._proc.wait()
        except Exception as e:
            self.logs.append(f"\n[Process reader error: {e}]\n")

        with self._lock:
            self.exit_code = self._proc.returncode if self._proc else -1
            if self.state == ProcessState.STOPPING:
                self.state = ProcessState.FINISHED
            elif self.exit_code == 0:
                self.state = ProcessState.FINISHED
            else:
                self.state = ProcessState.ERROR
            self.logs.append(f"\n[Process exited with code {self.exit_code}]\n")

    def terminate(self):
        with self._lock:
            if self.state != ProcessState.RUNNING:
                return
            self.state = ProcessState.STOPPING
            self.logs.append("\n[Stopping process...]\n")

        if self._proc:
            self._proc.terminate()
            # Wait up to 10s then force kill
            t = threading.Thread(target=self._force_kill, daemon=True)
            t.start()

    def _force_kill(self):
        if self._proc:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logs.append("\n[Force killing process...]\n")
                self._proc.kill()

    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "exit_code": self.exit_code,
        }

    def get_logs(self, last_n: Optional[int] = None) -> list[str]:
        if last_n is None:
            return list(self.logs)
        return list(self.logs)[-last_n:]


class ProcessManager:
    """Manages up to 3 concurrent subprocess slots."""

    def __init__(self):
        self._processes: dict[str, ManagedProcess] = {}
        self._lock = threading.Lock()

    def start(self, proc_type: ProcessType, cmd: list[str], cwd: Optional[str] = None):
        with self._lock:
            existing = self._processes.get(proc_type)
            if existing and existing.state == ProcessState.RUNNING:
                raise RuntimeError(f"{proc_type} is already running")

            mp = ManagedProcess(cmd, cwd=cwd)
            self._processes[proc_type] = mp

        mp.start()
        logger.info(f"Started {proc_type}: {' '.join(cmd[:5])}...")

    def stop(self, proc_type: ProcessType):
        with self._lock:
            mp = self._processes.get(proc_type)
            if not mp:
                return
        mp.terminate()

    def get_status(self, proc_type: ProcessType) -> dict:
        mp = self._processes.get(proc_type)
        if not mp:
            return {"state": ProcessState.IDLE.value, "exit_code": None}
        return mp.get_status()

    def get_logs(self, proc_type: ProcessType, last_n: Optional[int] = None) -> list[str]:
        mp = self._processes.get(proc_type)
        if not mp:
            return []
        return mp.get_logs(last_n)

    def get_all_statuses(self) -> dict[str, dict]:
        result = {}
        for pt in ("cache_latents", "cache_text", "cache_dino", "training", "inference", "slider_training"):
            result[pt] = self.get_status(pt)
        return result
