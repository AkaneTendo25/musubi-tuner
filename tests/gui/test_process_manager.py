"""Tests for ManagedProcess and ProcessManager."""

import sys
import time

import pytest

from musubi_tuner.gui_dashboard.process_manager import ManagedProcess, ProcessManager, ProcessState


class TestProcessManager:
    def test_initial_state(self):
        pm = ProcessManager()
        status = pm.get_status("training")
        assert status["state"] == "idle"
        assert status["exit_code"] is None

    def test_all_statuses(self):
        pm = ProcessManager()
        statuses = pm.get_all_statuses()
        assert "cache_latents" in statuses
        assert "cache_text" in statuses
        assert "training" in statuses
        for s in statuses.values():
            assert s["state"] == "idle"

    def test_empty_logs(self):
        pm = ProcessManager()
        logs = pm.get_logs("training")
        assert logs == []

    def test_start_simple_command(self):
        pm = ProcessManager()
        # Run a simple command that exits quickly
        pm.start("training", [sys.executable, "-c", "print('hello')"])
        status = pm.get_status("training")
        assert status["state"] in ("running", "finished")

    def test_cannot_start_twice(self):
        pm = ProcessManager()
        # Use a command that takes a bit of time
        pm.start("training", [sys.executable, "-c", "import time; time.sleep(5)"])
        with pytest.raises(RuntimeError, match="already running"):
            pm.start("training", [sys.executable, "-c", "print('second')"])
        pm.stop("training")

    def test_stop_idle_is_noop(self):
        pm = ProcessManager()
        pm.stop("training")  # Should not raise

    def test_managed_process_state_machine(self):
        mp = ManagedProcess([sys.executable, "-c", "print('test')"])
        assert mp.state == ProcessState.IDLE
        mp.start()
        # Wait for completion
        for _ in range(50):
            if mp.state != ProcessState.RUNNING:
                break
            time.sleep(0.1)
        assert mp.state == ProcessState.FINISHED
        assert mp.exit_code == 0
        logs = mp.get_logs()
        assert any("test" in line for line in logs)

    def test_managed_process_logs_command(self):
        mp = ManagedProcess([sys.executable, "-c", "print('hello world')"])
        mp.start()
        for _ in range(50):
            if mp.state != ProcessState.RUNNING:
                break
            time.sleep(0.1)
        logs = mp.get_logs()
        # First line should be the command
        assert logs[0].startswith("$")
