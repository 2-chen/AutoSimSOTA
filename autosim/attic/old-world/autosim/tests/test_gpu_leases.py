import tempfile
import threading
import unittest
from pathlib import Path

from autosim.research.common import atomic_json, read_json
from autosim.research.leases import (LeaseUnavailable, device_lease, device_leases,
                                     occupancy, pid_alive, sweep_stale)

DEVICES = [{"index": 0, "uuid": "GPU-aaa"}, {"index": 1, "uuid": "GPU-bbb"}]


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)

    def receipts(self):
        return (self.root / "lease_receipts.jsonl").read_text().splitlines()

    def test_lease_is_keyed_by_uuid_and_records_process_identity(self):
        with device_lease(DEVICES[0], job="job1", run_id="run1", root=self.root) as lease:
            self.assertEqual(lease.lock_path.name, "gpu-GPU-aaa.lock")
            holder = read_json(lease.owner_path)
            self.assertEqual(holder["job"], "job1")
            self.assertEqual(holder["run_id"], "run1")
            self.assertEqual(holder["pid"], __import__("os").getpid())
            self.assertEqual(holder["index"], 0)

    def test_release_frees_the_device_and_removes_only_the_sidecar(self):
        with device_lease(DEVICES[0], job="job1", run_id="run1", root=self.root) as lease:
            owner = lease.owner_path
            self.assertTrue(owner.is_file())
        self.assertFalse(owner.is_file())
        with device_lease(DEVICES[0], job="job2", run_id="run1", root=self.root) as lease:
            self.assertEqual(read_json(lease.owner_path)["job"], "job2")

    def test_second_acquire_on_a_held_uuid_waits_and_never_steals(self):
        with device_lease(DEVICES[0], job="first", run_id="run1", root=self.root):
            with self.assertRaises(LeaseUnavailable) as caught:
                with device_lease(DEVICES[0], job="second", run_id="run1", root=self.root,
                                  wait_seconds=0.3, poll_seconds=0.05):
                    self.fail("must not acquire a held device")
            self.assertIn("GPU-aaa", str(caught.exception))
            self.assertEqual(read_json(self.root / "gpu-GPU-aaa.owner.json")["job"], "first")

    def test_receipts_record_lease_release_and_the_waiting_reason(self):
        with device_lease(DEVICES[0], job="first", run_id="run1", root=self.root):
            try:
                with device_lease(DEVICES[0], job="second", run_id="run1", root=self.root):
                    pass
            except LeaseUnavailable:
                pass
        kinds = [__import__("json").loads(line)["event"] for line in self.receipts()]
        self.assertEqual(kinds, ["device_leased", "device_released"])

    def test_multi_device_acquire_is_all_or_nothing(self):
        with device_lease(DEVICES[1], job="holder", run_id="run1", root=self.root):
            with self.assertRaises(LeaseUnavailable):
                with device_leases(DEVICES, job="batch", run_id="run1", root=self.root):
                    self.fail("must not hold a partial set")
            # The device it *did* take first must have been handed back.
            with device_lease(DEVICES[0], job="after", run_id="run1", root=self.root):
                self.assertEqual(read_json(self.root / "gpu-GPU-aaa.owner.json")["job"], "after")

    def test_requests_in_either_order_are_serialised_and_leave_nothing_held(self):
        """Acquisition is UUID-sorted, so request order can never invert the lock order."""
        with device_lease(DEVICES[1], job="holder", run_id="run1", root=self.root):
            for order in (DEVICES, list(reversed(DEVICES))):
                with self.assertRaises(LeaseUnavailable):
                    with device_leases(order, job="batch", run_id="run1", root=self.root):
                        self.fail("must not hold a partial set")
            with device_lease(DEVICES[0], job="after", run_id="run1", root=self.root):
                self.assertEqual(read_json(self.root / "gpu-GPU-aaa.owner.json")["job"], "after")

    def test_a_waiter_progresses_once_the_holder_releases(self):
        import time
        holding = threading.Event()

        def holder():
            with device_leases(DEVICES, job="holder", run_id="run1", root=self.root):
                holding.set()
                time.sleep(0.4)

        thread = threading.Thread(target=holder)
        thread.start()
        self.assertTrue(holding.wait(5))
        with device_leases(DEVICES, job="waiter", run_id="run1", root=self.root,
                           wait_seconds=10, poll_seconds=0.05) as leases:
            self.assertEqual(len(leases), 2)
        thread.join(timeout=5)

    def test_stale_sidecar_from_a_dead_holder_is_reclaimed_and_recorded(self):
        atomic_json(self.root / "gpu-GPU-aaa.owner.json",
                    {"uuid": "GPU-aaa", "job": "dead", "pid": 999999, "index": 0})
        with device_lease(DEVICES[0], job="live", run_id="run1", root=self.root):
            self.assertEqual(read_json(self.root / "gpu-GPU-aaa.owner.json")["job"], "live")
        kinds = [__import__("json").loads(line)["event"] for line in self.receipts()]
        self.assertIn("stale_lease_reclaimed", kinds)

    def test_release_leaves_a_foreign_sidecar_alone(self):
        with device_lease(DEVICES[0], job="mine", run_id="run1", root=self.root) as lease:
            atomic_json(lease.owner_path, {"uuid": "GPU-aaa", "job": "other", "pid": 12345})
        self.assertTrue((self.root / "gpu-GPU-aaa.owner.json").is_file())

    def test_sweep_reports_only_dead_holders(self):
        atomic_json(self.root / "gpu-GPU-aaa.owner.json", {"uuid": "GPU-aaa", "pid": 999999})
        atomic_json(self.root / "gpu-GPU-bbb.owner.json", {"uuid": "GPU-bbb", "pid": 1})
        reclaimed = sweep_stale(self.root, alive=lambda pid: pid == 1)
        self.assertEqual([r["uuid"] for r in reclaimed], ["GPU-aaa"])

    def test_occupancy_reports_current_holders(self):
        self.assertEqual(occupancy(self.root), [])
        with device_leases(DEVICES, job="batch", run_id="run1", root=self.root):
            self.assertEqual(sorted(h["uuid"] for h in occupancy(self.root)), ["GPU-aaa", "GPU-bbb"])

    def test_pid_alive_answers_for_self_and_for_a_dead_pid(self):
        import os
        self.assertTrue(pid_alive(os.getpid()))
        self.assertFalse(pid_alive(999999))
        self.assertFalse(pid_alive(0))


if __name__ == "__main__":
    unittest.main()
