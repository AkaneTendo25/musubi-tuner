"""Subprocess manager for caching and training processes."""

from __future__ import annotations

import logging
import os
import locale
import re
import signal
import subprocess
import sys
import threading
from collections import deque
from enum import Enum
from typing import Literal, Optional

logger = logging.getLogger(__name__)

ProcessType = Literal["cache_latents", "cache_text", "cache_dino", "training", "inference", "slider_training"]

# Windows-specific flags for clean subprocess shutdown
_CREATION_FLAGS = 0
if sys.platform == "win32":
    _CREATION_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP


def _decode_output(data: bytes) -> str:
    for encoding in ("utf-8", locale.getpreferredencoding(False), "cp1251", "cp866"):
        if not encoding:
            continue
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


_LINE_BREAK_RE = re.compile(br"[\r\n]+")


_TQDM_PROGRESS_RE = re.compile(
    r"^(?P<label>.*?)(?::)?\s*(?P<pct>\d+)%\|.*?\|\s*(?P<current>\d+)/(?P<total>\d+)",
    re.IGNORECASE,
)


def _progress_signature(line: str) -> tuple[str, str, int] | None:
    match = _TQDM_PROGRESS_RE.match(line.strip())
    if not match:
        return None

    label = (match.group("label") or "").strip().lower()
    pct = int(match.group("pct"))
    current = int(match.group("current"))

    if label == "steps":
        return ("steps", match.group("total"), current)
    return ("progress", label, pct)


class ProcessState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    FINISHED = "finished"
    ERROR = "error"


class ManagedProcess:
    """Wraps a subprocess with state tracking and log buffering."""

    def __init__(self, cmd: list[str], cwd: Optional[str] = None, env: Optional[dict[str, str]] = None):
        self.cmd = cmd
        self.cwd = cwd
        self.env = env
        self.state = ProcessState.IDLE
        self.exit_code: Optional[int] = None
        self.logs: deque[str] = deque(maxlen=5000)
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_progress_signature: tuple[str, str, int] | None = None

    def start(self):
        with self._lock:
            if self.state == ProcessState.RUNNING:
                raise RuntimeError("Process already running")

            self.state = ProcessState.RUNNING
            self.exit_code = None
            self.logs.clear()
            self.logs.append(f"$ {' '.join(self.cmd)}\n")
            self._last_progress_signature = None

            self._proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.cwd,
                env=self.env,
                creationflags=_CREATION_FLAGS,
                bufsize=0,
            )

            self._reader_thread = threading.Thread(
                target=self._read_output, daemon=True
            )
            self._reader_thread.start()

    def _read_output(self):
        try:
            assert self._proc and self._proc.stdout
            pending = b""
            while True:
                chunk = self._proc.stdout.read(4096)
                if not chunk:
                    break

                pending += chunk
                ended_with_break = pending.endswith((b"\r", b"\n"))
                parts = _LINE_BREAK_RE.split(pending)
                complete_parts = parts if ended_with_break else parts[:-1]
                pending = b"" if ended_with_break else parts[-1]

                for part in complete_parts:
                    if not part:
                        continue
                    self._append_log_line(_decode_output(part))

            if pending:
                self._append_log_line(_decode_output(pending))
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

    def _append_log_line(self, line: str) -> bool:
        signature = _progress_signature(line)
        if signature is not None:
            if signature == self._last_progress_signature:
                return False
            self._last_progress_signature = signature
        else:
            self._last_progress_signature = None

        clean = line.rstrip("\r\n")
        if not clean.strip():
            return False

        self.logs.append(f"{clean}\n")
        return True

    def terminate(self):
        with self._lock:
            if self.state != ProcessState.RUNNING:
                return
            self.state = ProcessState.STOPPING
            self.logs.append("\n[Stopping process...]\n")

        if self._proc:
            if sys.platform == "win32":
                try:
                    self._proc.send_signal(signal.CTRL_BREAK_EVENT)
                except Exception:
                    self._proc.terminate()
            else:
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
                if sys.platform == "win32":
                    try:
                        subprocess.run(
                            ["taskkill", "/PID", str(self._proc.pid), "/T", "/F"],
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except Exception:
                        self._proc.kill()
                else:
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

            env = os.environ.copy()
            # Force UTF-8 stdio for dashboard-launched Python subprocesses so
            # trainer logs with Japanese text do not crash on localized Windows.
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            if proc_type in ("training", "slider_training"):
                env["MUSUBI_DASHBOARD_METRICS"] = "1"

            mp = ManagedProcess(cmd, cwd=cwd, env=env)
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
