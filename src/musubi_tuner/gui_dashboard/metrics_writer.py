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

    def __init__(self, run_dir: str, flush_every: int = 10):
        self.run_dir = run_dir
        self.flush_every = flush_every

        self.metrics_path = os.path.join(run_dir, "dashboard", "metrics.parquet")
        self.status_path = os.path.join(run_dir, "dashboard", "status.json")
        self.events_path = os.path.join(run_dir, "dashboard", "events.json")

        os.makedirs(os.path.join(run_dir, "dashboard"), exist_ok=True)

        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._step_count = 0

        # Initialize events file
        if not os.path.exists(self.events_path):
            self._write_json(self.events_path, [])

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
            if len(self._buffer) >= self.flush_every:
                self._flush_background()

    def log_event(self, event_type: str, step: int, **extra):
        entry = {"type": event_type, "step": step, "time": time.time(), **extra}
        t = threading.Thread(target=self._append_event, args=(entry,), daemon=True)
        t.start()

    def update_status(self, **kw):
        elapsed = time.monotonic() - self._start_time
        speed = self._step_count / elapsed if elapsed > 0 and self._step_count > 0 else 0.0
        status = {
            "elapsed_sec": round(elapsed, 1),
            "speed_steps_per_sec": round(speed, 4),
            "time": time.time(),
        }
        status.update(kw)
        t = threading.Thread(target=self._write_json, args=(self.status_path, status), daemon=True)
        t.start()

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
        table = pa.table({col: [r[col] for r in rows] for col in SCHEMA.names}, schema=SCHEMA)
        try:
            if os.path.exists(self.metrics_path):
                existing = pq.read_table(self.metrics_path, schema=SCHEMA)
                table = pa.concat_tables([existing, table])
            pq.write_table(table, self.metrics_path)
        except Exception:
            # If read fails (corrupt file), overwrite
            pq.write_table(table, self.metrics_path)

    def _append_event(self, entry: dict):
        try:
            events = []
            if os.path.exists(self.events_path):
                with open(self.events_path, "r") as f:
                    events = json.load(f)
            events.append(entry)
            self._write_json(self.events_path, events)
        except Exception:
            pass

    @staticmethod
    def _write_json(path: str, data):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
