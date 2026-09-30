import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from autosim.experiment_system.executor import lease
from autosim.experiment_validation.priority_guard import process_identity, release_reason, run
from autosim.research.common import atomic_json, digest, read_json


class PriorityGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        source = self.root / "scheduler.py"
        source.write_text("# test source\n")
        self.queue = self.root / "queue.json"
        atomic_json(self.queue, {"status": "running"})
        self.gate = self.root / "continuation/controller.lock"
        self.request = {"queue_pid": os.getpid(), "queue_process_start": process_identity(os.getpid()),
                        "queue_state": str(self.queue), "scheduler_gate": str(self.gate),
                        "sources": {str(source): digest(source)},
                        "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()}

    def tearDown(self):
        self.temp.cleanup()

    def test_pinned_pid_identity(self):
        self.assertIsNotNone(process_identity(os.getpid()))
        self.request["queue_process_start"] = "wrong"
        self.assertEqual(release_reason(self.request), "queue_process_exited_or_replaced")

    def test_expiry_is_absolute(self):
        future = datetime.now(timezone.utc) + timedelta(seconds=6)
        self.assertEqual(release_reason(self.request, current_time=future), "reservation_budget_exhausted")

    def test_changed_source_releases_fail_closed(self):
        Path(next(iter(self.request["sources"]))).write_text("changed")
        with self.assertRaises(ValueError):
            release_reason(self.request)

    def test_reservation_does_not_touch_gpu_and_releases_at_queue_gate(self):
        path = self.root / "input.json"
        atomic_json(path, self.request)
        output = self.root / "guard"
        thread = threading.Thread(target=run, args=(path, output), kwargs={"poll_seconds": .02})
        # An active experiment's GPU lease remains owned throughout this test.
        with lease(self.root / "gpu.lock"):
            thread.start()
            try:
                deadline = time.monotonic() + 3
                while not (output / "state.json").exists():
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.01)
                self.assertEqual(read_json(output / "state.json")["status"], "reserving_next_window")
                with self.assertRaises(BlockingIOError):
                    with lease(self.gate):
                        pass
                atomic_json(self.queue, {"status": "integration_gate_reached"})
            finally:
                atomic_json(self.queue, {"status": "integration_gate_reached"})
                thread.join(timeout=6)
        self.assertFalse(thread.is_alive())
        with lease(self.gate):
            pass
        state = read_json(output / "state.json")
        self.assertFalse(state["gpu_lock_acquired"])
        self.assertEqual(state["status"], "released")

    def test_existing_gate_owner_is_not_preempted(self):
        path = self.root / "input.json"
        self.request["expires_at"] = (datetime.now(timezone.utc) + timedelta(seconds=.1)).isoformat()
        atomic_json(path, self.request)
        with lease(self.gate):
            state = run(path, self.root / "guard", poll_seconds=.02)
        self.assertFalse(state["reservation_acquired"])
        self.assertEqual(state["reason"], "reservation_budget_exhausted")


if __name__ == "__main__":
    unittest.main()
