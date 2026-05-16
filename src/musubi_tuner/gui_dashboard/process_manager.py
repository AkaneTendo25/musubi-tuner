"""Subprocess manager for caching and training processes."""

from __future__ import annotations

import logging
import json
import os
import locale
import re
import signal
import subprocess
import sys
import threading
import tempfile
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

ProcessType = Literal["cache_latents", "cache_text", "cache_dino", "training", "inference", "slider_training"]
ProcessRef = tuple[int, float | None]

# Windows-specific flags for clean subprocess shutdown
_CREATION_FLAGS = 0
if sys.platform == "win32":
    _CREATION_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP

_STOP_GRACE_SECONDS = 5
_TASKKILL_TIMEOUT_SECONDS = 10


def _collect_windows_process_tree_refs(pid: int) -> list[ProcessRef]:
    if sys.platform != "win32":
        return []

    try:
        import ctypes
        from ctypes import wintypes

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_snapshot = kernel32.CreateToolhelp32Snapshot
        create_snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        create_snapshot.restype = wintypes.HANDLE
        process_first = kernel32.Process32FirstW
        process_first.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        process_first.restype = wintypes.BOOL
        process_next = kernel32.Process32NextW
        process_next.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        process_next.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        snapshot = create_snapshot(0x00000002, 0)
        if snapshot == wintypes.HANDLE(-1).value:
            return [(pid, None)]

        children_by_parent: dict[int, list[int]] = {}
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            if not process_first(snapshot, ctypes.byref(entry)):
                return [(pid, None)]

            while True:
                child_pid = int(entry.th32ProcessID)
                parent_pid = int(entry.th32ParentProcessID)
                children_by_parent.setdefault(parent_pid, []).append(child_pid)
                if not process_next(snapshot, ctypes.byref(entry)):
                    break
        finally:
            close_handle(snapshot)

        refs: list[ProcessRef] = []
        seen: set[int] = set()
        stack = [pid]
        while stack:
            current_pid = stack.pop()
            if current_pid in seen:
                continue
            seen.add(current_pid)
            refs.append((current_pid, None))
            stack.extend(children_by_parent.get(current_pid, []))
        return refs
    except Exception as exc:
        logger.debug("Could not collect Windows process tree for PID %s: %s", pid, exc)
        return [(pid, None)]


def _windows_pid_exists(pid: int) -> bool:
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        open_process.restype = wintypes.HANDLE
        get_exit_code_process = kernel32.GetExitCodeProcess
        get_exit_code_process.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        get_exit_code_process.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL

        handle = open_process(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not get_exit_code_process(handle, ctypes.byref(exit_code)):
                return True
            return int(exit_code.value) == 259
        finally:
            close_handle(handle)
    except Exception:
        return True


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


def _dedupe_process_refs(refs: list[ProcessRef]) -> list[ProcessRef]:
    result: list[ProcessRef] = []
    seen: set[int] = set()
    for pid, create_time in refs:
        if pid <= 0 or pid in seen or pid == os.getpid():
            continue
        seen.add(pid)
        result.append((pid, create_time))
    return result


def _collect_process_tree_refs(pid: int) -> list[ProcessRef]:
    """Capture child PIDs before termination can orphan them."""
    if sys.platform == "win32":
        return _dedupe_process_refs(_collect_windows_process_tree_refs(pid))

    refs: list[ProcessRef] = []
    try:
        import psutil

        parent = psutil.Process(pid)
        processes = parent.children(recursive=True)
        processes.append(parent)
        for process in processes:
            try:
                refs.append((int(process.pid), float(process.create_time())))
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                refs.append((int(process.pid), None))
    except Exception as exc:
        logger.debug("Could not collect process tree for PID %s: %s", pid, exc)
        refs.append((pid, None))
    return _dedupe_process_refs(refs)


def _process_ref_still_matches(ref: ProcessRef) -> bool:
    pid, create_time = ref
    if pid <= 0 or pid == os.getpid():
        return False
    if create_time is None:
        if sys.platform == "win32":
            return _windows_pid_exists(pid)
        try:
            import psutil

            return bool(psutil.pid_exists(pid))
        except ImportError:
            return True
        except Exception:
            return True
    try:
        import psutil

        return abs(float(psutil.Process(pid).create_time()) - create_time) < 0.1
    except ImportError:
        return True
    except Exception as exc:
        if exc.__class__.__name__ == "AccessDenied":
            return True
        return False


def _kill_process_refs(refs: list[ProcessRef]):
    for ref in _dedupe_process_refs(refs):
        if not _process_ref_still_matches(ref):
            continue

        pid = ref[0]
        if sys.platform == "win32":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=_TASKKILL_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                logger.debug("Could not taskkill PID %s: %s", pid, exc)
            continue

        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception as exc:
                logger.debug("Could not kill PID %s: %s", pid, exc)


class ProcessState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    FINISHED = "finished"
    ERROR = "error"


class ManagedProcess:
    """Wraps a subprocess with state tracking and log buffering."""

    def __init__(
        self,
        cmd: list[str],
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        proc_type: Optional[str] = None,
    ):
        self.cmd = cmd
        self.cwd = cwd
        self.env = env
        self.proc_type = proc_type
        self.state = ProcessState.IDLE
        self.exit_code: Optional[int] = None
        self.logs: deque[str] = deque(maxlen=5000)
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_progress_signature: tuple[str, str, int] | None = None
        self._graceful_stop_requested = False
        self._stop_requested = False
        self._stop_file: Optional[Path] = None
        self._force_kill_delay_seconds = _STOP_GRACE_SECONDS
        self._stop_process_refs: list[ProcessRef] = []

    def start(self):
        with self._lock:
            if self.state == ProcessState.RUNNING:
                raise RuntimeError("Process already running")

            self.state = ProcessState.RUNNING
            self.exit_code = None
            self.logs.clear()
            self.logs.append(f"$ {' '.join(self.cmd)}\n")
            self._last_progress_signature = None
            self._graceful_stop_requested = False
            self._stop_requested = False
            self._force_kill_delay_seconds = _STOP_GRACE_SECONDS
            self._stop_process_refs = []

            self._proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.cwd,
                env=self.env,
                creationflags=_CREATION_FLAGS,
                start_new_session=sys.platform != "win32",
                bufsize=0,
            )

            self._reader_thread = threading.Thread(
                target=self._read_output, daemon=True
            )
            self._reader_thread.start()

    def _supports_graceful_stop(self) -> bool:
        return self.proc_type in {"training", "slider_training"}

    def _request_graceful_training_stop(self) -> bool:
        if not self._supports_graceful_stop() or not self._stop_file:
            return False
        try:
            self._stop_file.parent.mkdir(parents=True, exist_ok=True)
            self._stop_file.write_text(json.dumps({"mode": "graceful", "time": time.time()}), encoding="utf-8")
            self._graceful_stop_requested = True
            self.logs.append("\n[Graceful training stop requested. Waiting for final checkpoint/state save...]\n")
            return True
        except OSError as exc:
            self.logs.append(f"\n[Failed to request graceful training stop: {exc}. Sending interrupt instead.]\n")
            return False

    def _finalize_process_state(self):
        with self._lock:
            exit_code = self._proc.returncode if self._proc else -1
            already_reported = self.state in (ProcessState.FINISHED, ProcessState.ERROR) and self.exit_code == exit_code
            self.exit_code = exit_code
            if self.state in (ProcessState.STOPPING, ProcessState.FINISHED):
                self.state = ProcessState.FINISHED
            elif self.exit_code == 0:
                self.state = ProcessState.FINISHED
            else:
                self.state = ProcessState.ERROR
            if not already_reported:
                self.logs.append(f"\n[Process exited with code {self.exit_code}]\n")

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

        self._finalize_process_state()

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
            if self.state == ProcessState.STOPPING:
                self._stop_requested = True
                process_refs = list(self._stop_process_refs)
                if self._graceful_stop_requested:
                    self.logs.append("\n[Force stop requested. Skipping graceful final save...]\n")
                    if self._stop_file:
                        try:
                            self._stop_file.parent.mkdir(parents=True, exist_ok=True)
                            self._stop_file.write_text(json.dumps({"mode": "force", "time": time.time()}), encoding="utf-8")
                        except OSError:
                            pass
                else:
                    self.logs.append("\n[Force stop requested...]\n")
            elif self.state == ProcessState.RUNNING:
                self.state = ProcessState.STOPPING
                self._stop_requested = True
                process_refs = []
            else:
                return

        if not self._proc:
            return

        force_stop = self._graceful_stop_requested
        if force_stop:
            process_refs = _dedupe_process_refs(process_refs + _collect_process_tree_refs(self._proc.pid))
            self._force_kill_delay_seconds = 0
            with self._lock:
                self._stop_process_refs = list(process_refs)
        elif not process_refs:
            process_refs = _collect_process_tree_refs(self._proc.pid)
            with self._lock:
                self._stop_process_refs = list(process_refs)

            if self._request_graceful_training_stop():
                self._force_kill_delay_seconds = 600
                t = threading.Thread(target=self._force_kill, args=(process_refs,), daemon=True)
                t.start()
                return

            self.logs.append("\n[Stopping process...]\n")

            if sys.platform == "win32":
                try:
                    self._proc.send_signal(signal.CTRL_BREAK_EVENT)
                except Exception:
                    self._proc.terminate()
            else:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                except Exception:
                    self._proc.terminate()

        t = threading.Thread(target=self._force_kill, args=(process_refs,), daemon=True)
        t.start()

    def _force_kill(self, process_refs: list[ProcessRef]):
        if self._proc:
            try:
                self._proc.wait(timeout=self._force_kill_delay_seconds)
            except subprocess.TimeoutExpired:
                self.logs.append("\n[Force killing process tree...]\n")
                _kill_process_refs(_dedupe_process_refs(process_refs + _collect_process_tree_refs(self._proc.pid)))
                try:
                    self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    return
                self._finalize_process_state()
                return

            orphan_refs = [ref for ref in process_refs if ref[0] != self._proc.pid and _process_ref_still_matches(ref)]
            if orphan_refs:
                self.logs.append("\n[Force killing remaining child processes...]\n")
                _kill_process_refs(orphan_refs)
                self._finalize_process_state()

    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "exit_code": self.exit_code,
            "stop_requested": self._stop_requested,
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
            cmd_executable = Path(cmd[0]) if cmd else Path(sys.executable)
            python_bin_dir = str(cmd_executable.parent) if cmd_executable.exists() else os.path.dirname(sys.executable)
            path_key = "Path" if "Path" in env else "PATH"
            env[path_key] = python_bin_dir + os.pathsep + env.get(path_key, "")
            repo_src = Path(__file__).resolve().parents[2]
            env["PYTHONPATH"] = str(repo_src) + os.pathsep + env.get("PYTHONPATH", "")
            # Force UTF-8 stdio for dashboard-launched Python subprocesses so
            # trainer logs with Japanese text do not crash on localized Windows.
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            if proc_type in ("training", "slider_training"):
                env["MUSUBI_DASHBOARD_METRICS"] = "1"
                stop_file = Path(tempfile.gettempdir()) / f"musubi_dashboard_stop_{os.getpid()}_{proc_type}.flag"
                try:
                    stop_file.unlink(missing_ok=True)
                except OSError:
                    pass
                env["MUSUBI_DASHBOARD_STOP_FILE"] = str(stop_file)

            mp = ManagedProcess(cmd, cwd=cwd, env=env, proc_type=proc_type)
            if proc_type in ("training", "slider_training"):
                mp._stop_file = Path(env["MUSUBI_DASHBOARD_STOP_FILE"])
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
            return {"state": ProcessState.IDLE.value, "exit_code": None, "stop_requested": False}
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
