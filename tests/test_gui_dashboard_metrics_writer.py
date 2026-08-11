import threading

import pyarrow.parquet as pq

from musubi_tuner.gui_dashboard.metrics_writer import MetricsWriter


def test_metrics_writer_uses_one_background_drain_worker(tmp_path, monkeypatch):
    writer = MetricsWriter(str(tmp_path), flush_every=1, reset=True)
    first_write_started = threading.Event()
    allow_first_write = threading.Event()
    calls = []
    active_writes = 0
    maximum_active_writes = 0
    counter_lock = threading.Lock()

    def slow_flush(rows):
        nonlocal active_writes, maximum_active_writes
        with counter_lock:
            active_writes += 1
            maximum_active_writes = max(maximum_active_writes, active_writes)
        try:
            calls.append([row["step"] for row in rows])
            if not first_write_started.is_set():
                first_write_started.set()
                assert allow_first_write.wait(timeout=5)
        finally:
            with counter_lock:
                active_writes -= 1

    monkeypatch.setattr(writer, "_do_flush", slow_flush)
    writer.log(step=0)
    assert first_write_started.wait(timeout=5)
    first_worker = writer._flush_thread

    for step in range(1, 21):
        writer.log(step=step)

    assert writer._flush_thread is first_worker
    assert first_worker is not None and first_worker.is_alive()
    allow_first_write.set()
    writer.close()

    assert maximum_active_writes == 1
    assert [step for batch in calls for step in batch] == list(range(21))
    assert writer._flush_thread is None


def test_metrics_writer_close_persists_background_and_buffered_rows(tmp_path):
    writer = MetricsWriter(str(tmp_path), flush_every=3, reset=True)
    for step in range(25):
        writer.log(step=step, loss=float(step))
    writer.close()

    table = pq.read_table(writer.metrics_path)
    assert table.column("step").to_pylist() == list(range(25))
    assert writer._flush_thread is None
