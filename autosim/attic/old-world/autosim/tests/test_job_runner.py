"""What a phase does with several devices, and what it does with one.

The two halves matter equally: the multi-device half is new behaviour that has to be
evidenced (different devices, genuinely overlapping, leases held, records written), and the
single-device half is the *old* behaviour that must not acquire a single new artifact --
a legacy run's output tree is what every completed run in this repository is read against.
"""

import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from autosim.research.accounting import BudgetLedger
from autosim.research.job_runner import (PhaseJob, episodes_of, is_scheduled, run_phase,
                                         schedule_limit)
from autosim.research.leases import LeaseUnavailable, device_lease, occupancy


def device(index, *, uuid=None, selection=True):
    row = {"index": index, "uuid": uuid or f"GPU-{index}", "class_name": "rtx5090",
           "model": "NVIDIA GeForce RTX 5090"}
    if selection:
        row["selection"] = {"mode": "pinned_index", "index": index, "uuid": row["uuid"],
                            "vulkan_gpu_id": index, "torch_index": 0, "cuda_visible": str(index),
                            "renderer": "hybrid", "extra_env": {}}
    return row


def plan(devices=4, **overrides):
    row = {"schema_version": 1, "kind": "autosim_device_plan", "mode": "pinned_index",
           "usable": [device(index) for index in range(devices)],
           "max_parallel_jobs": devices, "legacy_equivalence": False,
           "gpu_hours_limit": 96.0, "hours": 24.0}
    row.update(overrides)
    return row


class FakeBound:
    def __init__(self, runtime, selection, job, output):
        self.runtime = runtime
        self.selection = selection
        self.job = job
        self.output = output


class FakeRuntime:
    """Just the surface ``run_phase`` uses: ``for_job`` and (for the ledger) nothing else."""

    def __init__(self):
        self.plan = None
        self.bound = []

    def for_job(self, selection, *, job, output=None):
        bound = FakeBound(self, selection, job, output)
        self.bound.append(bound)
        return bound


class PhaseTestCase(unittest.TestCase):
    def setUp(self):
        self._temp = TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.runtime = FakeRuntime()
        self.leases = self.root / "leases"
        self.in_flight = 0
        self.peak = 0
        self.gate = threading.Lock()
        self.seen = []

    def job(self, name, *, category="evaluation", delay=0.0, fails=False, result=None):
        def run(bound):
            with self.gate:
                self.in_flight += 1
                self.peak = max(self.peak, self.in_flight)
            try:
                self.seen.append((name, getattr(bound, "job", None), bound))
                if delay:
                    time.sleep(delay)
                if fails:
                    raise RuntimeError(f"{name} failed on purpose")
                return result if result is not None else {"job": name, "episodes": [1, 2, 3]}
            finally:
                with self.gate:
                    self.in_flight -= 1
        return PhaseJob(name, category, run, estimate_seconds=10.0, reserve_seconds=5.0)

    def phase(self, jobs, *, plan_row=None, max_parallel_jobs=None, ledger=None):
        return run_phase(phase="test_phase", jobs=jobs, runtime=self.runtime,
                         plan=plan_row, output=self.root / "schedule",
                         max_parallel_jobs=max_parallel_jobs, ledger=ledger,
                         run_id="run-1", lease_root=self.leases)


class SingleDeviceTests(PhaseTestCase):
    def test_a_legacy_invocation_writes_no_schedule_at_all(self):
        results = self.phase([self.job("a"), self.job("b")], plan_row=None)
        self.assertEqual(sorted(results), ["a", "b"])
        self.assertFalse((self.root / "schedule").exists())
        self.assertEqual([name for name, _, _ in self.seen], ["a", "b"])   # in order
        # Both jobs were handed the *shared* runtime: the sequential path binds nothing, so
        # it cannot leave per-job addressing behind in an artifact.
        self.assertEqual(self.runtime.bound, [])
        self.assertEqual([bound for _, _, bound in self.seen], [self.runtime, self.runtime])
        self.assertEqual(self.peak, 1)

    def test_one_device_or_one_parallel_job_is_the_sequential_path(self):
        self.assertFalse(is_scheduled(None, None))
        self.assertFalse(is_scheduled(plan(4, legacy_equivalence=True), None))
        self.assertFalse(is_scheduled(plan(1), None))
        self.assertFalse(is_scheduled(plan(4), 1))
        self.assertTrue(is_scheduled(plan(4), None))
        self.assertEqual(schedule_limit(plan(4, max_parallel_jobs=2), None), 2)
        self.assertEqual(schedule_limit(plan(4), 3), 3)
        self.assertEqual(schedule_limit(plan(4), 9), 4)          # never wider than the plan
        for plan_row in (None, plan(4, legacy_equivalence=True), plan(1), plan(4)):
            with self.subTest(plan=plan_row):
                results = run_phase(phase="p", jobs=[self.job("a")], runtime=self.runtime,
                                    plan=plan_row, output=self.root / "out",
                                    max_parallel_jobs=1, run_id="r")
                self.assertEqual(sorted(results), ["a"])


class MultiDeviceTests(PhaseTestCase):
    def test_independent_jobs_take_different_devices_and_really_overlap(self):
        results = self.phase([self.job("a", delay=0.25), self.job("b", delay=0.25)],
                           plan_row=plan(4))
        self.assertEqual(sorted(results), ["a", "b"])
        self.assertGreaterEqual(self.peak, 2, "the two jobs never overlapped")
        taken = {bound.selection.uuid: bound.job for _, _, bound in self.seen}
        self.assertEqual(taken, {"GPU-0": "a", "GPU-1": "b"})
        for bound in self.runtime.bound:
            self.assertEqual(bound.selection.cuda_visible,
                             str(bound.selection.index))        # each job sees one card
            self.assertTrue(Path(bound.output).is_dir())

    def test_every_job_leaves_a_record_and_the_devices_go_back(self):
        self.phase([self.job("a"), self.job("b")], plan_row=plan(4))
        schedule = json.loads((self.root / "schedule" / "test_phase"
                               / "schedule.json").read_text())
        self.assertEqual(sorted(schedule["completed"]), ["a", "b"])
        self.assertEqual(schedule["running"], {})
        self.assertEqual(occupancy(self.leases), [])             # leases released
        for name in ("a", "b"):
            record = json.loads((self.root / "schedule" / "test_phase" / name
                                 / "outcome.json").read_text())
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["detail"]["phase"], "test_phase")
            self.assertEqual(len(record["devices"]), 1)

    def test_the_devices_are_held_while_the_jobs_run(self):
        holder = {}

        def run(bound):
            holder["occupancy"] = occupancy(self.leases)
            return {}

        results = run_phase(phase="p", jobs=[PhaseJob("a", "evaluation", run, 10.0, 5.0)],
                            runtime=self.runtime, plan=plan(2), output=self.root / "out",
                            run_id="r", lease_root=self.leases)
        self.assertEqual(sorted(results), ["a"])
        self.assertEqual([row["uuid"] for row in holder["occupancy"]], ["GPU-0"])
        self.assertEqual(holder["occupancy"][0]["job"], "a")
        self.assertEqual(occupancy(self.leases), [])

    def test_a_job_that_fails_is_recorded_and_the_phase_fails_as_a_whole(self):
        with self.assertRaises(RuntimeError) as caught:
            self.phase([self.job("a", fails=True), self.job("b")], plan_row=plan(4))
        self.assertIn("test_phase jobs failed", str(caught.exception))
        self.assertIn("a failed on purpose", str(caught.exception))
        failure = json.loads((self.root / "schedule" / "test_phase" / "a"
                              / "failure.json").read_text())
        self.assertEqual(failure["status"], "failed")
        self.assertIn("failed on purpose", failure["detail"]["error"])
        # The healthy job still finished and its own record says so.
        outcome = json.loads((self.root / "schedule" / "test_phase" / "b"
                             / "outcome.json").read_text())
        self.assertEqual(outcome["status"], "completed")
        self.assertEqual(occupancy(self.leases), [])

    def test_a_job_that_cannot_be_placed_names_the_rule_that_stopped_it(self):
        # Two devices, three heavy jobs, and a limit of two: the third has nowhere to go.
        jobs = [self.job("a", delay=0.4), self.job("b", delay=0.4), self.job("c", delay=0.4)]
        results = self.phase(jobs, plan_row=plan(2), max_parallel_jobs=2)
        self.assertEqual(sorted(results), ["a", "b", "c"])
        self.assertEqual(self.peak, 2)                    # it waited rather than oversubscribed
        schedule = json.loads((self.root / "schedule" / "test_phase"
                               / "schedule.json").read_text())
        self.assertEqual(schedule["max_parallel_jobs"], 2)

    def test_a_device_held_by_someone_else_is_not_run_on(self):
        """The lease is the authority; a job that cannot take it must not start anyway."""
        with device_lease(device(0), job="other_run", run_id="other", root=self.leases):
            with self.assertRaises(LeaseUnavailable) as caught:
                self.phase([self.job("a")], plan_row=plan(2))
            self.assertIn("GPU-0", str(caught.exception))
            self.assertEqual(self.seen, [])
            self.assertEqual([row["job"] for row in occupancy(self.leases)], ["other_run"])

    def test_the_budget_is_checked_before_a_job_starts(self):
        ledger = BudgetLedger(self.root / "accounting.json", wall_limit_seconds=8.0,
                              gpu_hours_limit=None, devices=[device(0), device(1)])
        with self.assertRaises(Exception) as caught:
            self.phase([self.job("a")], plan_row=plan(2), ledger=ledger)
        self.assertIn("budget", str(caught.exception).lower())


class ResumeTests(PhaseTestCase):
    def test_a_completed_job_is_answered_from_its_own_cache_not_re_run(self):
        jobs = [self.job("a"), self.job("b")]
        self.phase(jobs, plan_row=plan(4))
        self.seen.clear()
        cached = self.job("a", result={"cached": True})
        fresh = self.job("b")
        results = self.phase([cached, fresh], plan_row=plan(4))
        # "a" is completed in the schedule, so it is asked on the *unbound* runtime -- the
        # cache path every callable here is idempotent through.
        self.assertEqual([name for name, _, _ in self.seen], ["a", "b"])
        self.assertIs(self.seen[0][2], self.runtime)
        self.assertEqual(results["a"], {"cached": True})
        self.assertEqual(results["b"], {"job": "b", "episodes": [1, 2, 3]})

    def test_a_failed_job_is_never_silently_retried(self):
        with self.assertRaises(RuntimeError):
            self.phase([self.job("a", fails=True)], plan_row=plan(4))
        with self.assertRaises(RuntimeError) as caught:
            self.phase([self.job("a")], plan_row=plan(4))
        self.assertIn("recorded failed", str(caught.exception))
        self.assertIn("failure.json", str(caught.exception))

    def test_a_job_that_never_finished_is_simply_pending_again(self):
        (self.root / "schedule" / "test_phase" / "a").mkdir(parents=True)
        (self.root / "schedule" / "test_phase" / "a" / "partial.txt").write_text("half")
        results = self.phase([self.job("a")], plan_row=plan(4))
        self.assertEqual(results["a"], {"job": "a", "episodes": [1, 2, 3]})
        self.assertEqual(len(self.seen), 1)


class EpisodesTests(unittest.TestCase):
    def test_the_charge_is_the_work_the_job_reported(self):
        self.assertEqual(episodes_of({"episodes": [1, 2, 3]}), 3)
        self.assertEqual(episodes_of({"accepted_episodes": 7}), 7)
        self.assertEqual(episodes_of({"accepted": 0}), 0)
        self.assertIsNone(episodes_of({"status": "ok"}))
        self.assertIsNone(episodes_of(None))


if __name__ == "__main__":
    unittest.main()
