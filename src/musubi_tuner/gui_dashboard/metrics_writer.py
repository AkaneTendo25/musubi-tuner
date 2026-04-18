from collections import deque
import json
import os
import threading
import time
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = pa.schema(
    [
        ("step", pa.int64()),
        ("epoch", pa.int32()),
        ("loss", pa.float32()),
        ("avr_loss", pa.float32()),
        ("loss_v", pa.float32()),
        ("loss_a", pa.float32()),
        ("lr", pa.float64()),
        ("step_time", pa.float32()),
    ]
)


class MetricsWriter:
    """Buffered Parquet writer for training metrics.

    Accumulates rows in memory and flushes to a Parquet file periodically
    via a background daemon thread.  Also manages status.json and events.json.
    """

    def __init__(self, run_dir: str, flush_every: int = 10, reset: bool = False):
        self.run_dir = run_dir
        self.flush_every = flush_every

        self.metrics_path = os.path.join(run_dir, "dashboard", "metrics.parquet")
        self.status_path = os.path.join(run_dir, "dashboard", "status.json")
        self.events_path = os.path.join(run_dir, "dashboard", "events.json")

        os.makedirs(os.path.join(run_dir, "dashboard"), exist_ok=True)
        if reset:
            for path in (self.metrics_path, self.status_path, self.events_path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass

        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._events_lock = threading.Lock()
        self._start_time = time.monotonic()
        self._step_count = 0
        self._training_started_at: Optional[float] = None
        self._recent_step_times: deque[float] = deque(maxlen=20)

        # Initialize events file
        if not os.path.exists(self.events_path):
            self._write_json(self.events_path, [], self._events_lock)

        # Initialize status
        self.update_status(step=0, max_steps=0, epoch=0, max_epochs=0, status="initializing")

    # -- public API --

    def log(
        self,
        step: int,
        epoch: int = 0,
        loss: float = 0.0,
        avr_loss: float = 0.0,
        loss_v: Optional[float] = None,
        loss_a: Optional[float] = None,
        lr: float = 0.0,
        step_time: float = 0.0,
    ):
        row = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "avr_loss": avr_loss,
            "loss_v": loss_v if loss_v is not None else float("nan"),
            "loss_a": loss_a if loss_a is not None else float("nan"),
            "lr": lr,
            "step_time": step_time,
        }
        with self._lock:
            self._buffer.append(row)
            self._step_count += 1
            if step_time and step_time > 0:
                if self._training_started_at is None:
                    self._training_started_at = time.monotonic() - step_time
                self._recent_step_times.append(step_time)
            if len(self._buffer) >= self.flush_every:
                self._flush_background()

    def log_event(self, event_type: str, step: int, **extra):
        entry = {"type": event_type, "step": step, "time": time.time(), **extra}
        self._append_event(entry)

    def update_status(self, **kw):
        if self._training_started_at is None:
            elapsed = 0.0
            speed = 0.0
        else:
            elapsed = time.monotonic() - self._training_started_at
            if self._recent_step_times:
                avg_step_time = sum(self._recent_step_times) / len(self._recent_step_times)
                speed = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
            else:
                speed = self._step_count / elapsed if elapsed > 0 and self._step_count > 0 else 0.0
        status = {
            "elapsed_sec": round(elapsed, 1),
            "speed_steps_per_sec": round(speed, 4),
            "time": time.time(),
        }
        status.update(kw)
        self._write_json(self.status_path, status, self._status_lock)

    def flush(self):
        with self._lock:
            if self._buffer:
                self._do_flush(list(self._buffer))
                self._buffer.clear()

    def close(self):
        self.flush()

    # -- internals --

    def _flush_background(self):
        rows = list(self._buffer)
        self._buffer.clear()
        t = threading.Thread(target=self._do_flush, args=(rows,), daemon=True)
        t.start()

    def _do_flush(self, rows: list[dict]):
        with self._metrics_lock:
            table = pa.table({col: [r[col] for r in rows] for col in SCHEMA.names}, schema=SCHEMA)
            try:
                if os.path.exists(self.metrics_path):
                    existing = pq.read_table(self.metrics_path, schema=SCHEMA)
                    table = pa.concat_tables([existing, table])
            except Exception:
                pass

            tmp = self.metrics_path + f".{threading.get_ident()}.tmp"
            pq.write_table(table, tmp)
            self._replace_with_retry(tmp, self.metrics_path)

    def _append_event(self, entry: dict):
        with self._events_lock:
            try:
                events = []
                if os.path.exists(self.events_path):
                    with open(self.events_path, "r", encoding="utf-8") as f:
                        events = json.load(f)
                events.append(entry)
                self._write_json(self.events_path, events, None)
            except Exception:
                pass

    def _write_json(self, path: str, data, lock: Optional[threading.Lock]):
        if lock is None:
            self._write_json_unlocked(path, data)
            return
        with lock:
            self._write_json_unlocked(path, data)

    def _write_json_unlocked(self, path: str, data):
        tmp = path + f".{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        self._replace_with_retry(tmp, path)

    @staticmethod
    def _replace_with_retry(tmp: str, path: str, attempts: int = 8, delay: float = 0.05):
        last_error = None
        for _ in range(attempts):
            try:
                os.replace(tmp, path)
                return
            except PermissionError as e:
                last_error = e
                time.sleep(delay)
        if last_error is not None:
            raise last_error
