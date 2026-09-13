"""What the scheduler places, what it holds back, and how it resumes.

The two rules worth reading the tests for are the ones that would turn a "parallel" run into
a wrong measurement rather than a slow one: never two heavy jobs on one device, and never a
trainer beside a simulator.  Everything else here is bookkeeping that has to survive a
restart without re-running score-bearing work.
"""

import json
import tempfile
import unittest
from pathlib import Path

from autosim.research.accounting import BudgetLedger
from autosim.research.leases import occupancy
from autosim.research.scheduler import Job, ScheduleRefused, Scheduler

DEVICES = [{"uuid": f"GPU-{index}", "index": index, "class_name": "rtx5090",
            "model": "NVIDIA GeForce RTX 5090"} for index in range(4)]


def job(name, **overrides):
    kwargs = dict(category="collection", kind=name, estimate_seconds=600.0, reserve_seconds=60.0)
    kwargs.update(overrides)
    return Job(name, **kwargs)


class SchedulerTestCase(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.leases = self.root / "leases"

    def scheduler(self, jobs, *, devices=DEVICES, ledger=None, **overrides):
        kwargs = dict(jobs=jobs, devices=devices, output=self.root / "jobs", ledger=ledger,
                      run_id="run-1", lease_root=self.leases)
        kwargs.update(overrides)
        return Scheduler(**kwargs)

    def ledger(self, *, wall_limit=86400.0, gpu_hours=96.0):
        return BudgetLedger(self.root / "accounting.json", wall_limit_seconds=wall_limit,
                            gpu_hours_limit=gpu_hours, devices=DEVICES)


class PlacementTests(SchedulerTestCase):
    def test_independent_jobs_are_placed_on_different_devices(self):
        scheduler = self.scheduler([job("collect_targeted"), job("collect_original")])
        step = scheduler.plan_step()
        self.assertEqual([item.job.name for item in step["assignments"]],
                         ["collect_original", "collect_targeted"])       # deterministic order
        self.assertEqual([[str(device["uuid"]) for device in item.devices]
                          for item in step["assignments"]], [["GPU-0"], ["GPU-1"]])
        self.assertEqual(step["waiting"], [])

    def test_a_dependent_job_waits_and_then_becomes_ready(self):
        scheduler = self.scheduler([job("train_r0", category="training"),
                                    job("collect_r0", depends_on=("train_r0",))])
        self.assertEqual(scheduler.plan_step()["waiting"][0].reason, "dependency")
        self.assertEqual([item.job.name for item in scheduler.plan_step()["assignments"]],
                         ["train_r0"])
        scheduler.start(scheduler.plan_step()["assignments"][0])
        self.assertEqual(scheduler.plan_step()["waiting"][0].reason, "dependency")
        scheduler.complete("train_r0")
        self.assertEqual([item.job.name for item in scheduler.plan_step()["assignments"]],
                         ["collect_r0"])
        self.assertEqual(scheduler.state("collect_r0"), "ready")

    def test_one_heavy_job_per_device_is_enforced(self):
        scheduler = self.scheduler([job("a"), job("b"), job("c"), job("d"), job("e")])
        for item in scheduler.plan_step()["assignments"]:
            scheduler.start(item)
        step = scheduler.plan_step()
        self.assertEqual(step["assignments"], [])
        self.assertEqual(step["waiting"][0].reason, "devices_busy")
        self.assertEqual(step["waiting"][0].detail, "needs 1 of 0 free device(s)")
        self.assertEqual(scheduler.state("e"), "ready")          # ready, just not placeable
        self.assertEqual(len(scheduler.free_devices()), 0)

    def test_a_trainer_never_shares_a_device_with_a_simulator(self):
        scheduler = self.scheduler([job("train_r0", category="training"), job("collect_r0")],
                                   devices=DEVICES[:1])
        first = scheduler.plan_step()["assignments"][0]
        self.assertEqual(first.job.name, "collect_r0")           # by name order
        scheduler.start(first)
        step = scheduler.plan_step()
        self.assertEqual(step["assignments"], [])
        self.assertEqual(step["waiting"][0].reason, "devices_busy")
        self.assertEqual([str(device["uuid"]) for device in scheduler.free_devices()], [])

    def test_max_parallel_jobs_caps_the_heavy_jobs_not_the_device_count(self):
        scheduler = self.scheduler([job("a"), job("b"), job("c")], max_parallel_jobs=2)
        self.assertEqual([item.job.name for item in scheduler.plan_step()["assignments"]],
                         ["a", "b"])
        self.assertEqual(scheduler.plan_step()["waiting"][0].detail,
                         "2 heavy job(s) already placed or running, limit 2")

    def test_a_job_needing_two_devices_takes_two_or_waits(self):
        two = job("wide_collect", device_count=2)
        scheduler = self.scheduler([two], devices=DEVICES[:1])
        self.assertEqual(scheduler.plan_step()["waiting"][0].reason, "devices_busy")
        scheduler = self.scheduler([two], devices=DEVICES[:2])
        assignment = scheduler.plan_step()["assignments"][0]
        self.assertEqual([str(device["uuid"]) for device in assignment.devices],
                         ["GPU-0", "GPU-1"])

    def test_a_failed_dependency_blocks_rather_than_starts(self):
        scheduler = self.scheduler([job("collect_r0"), job("eval_r0", category="evaluation",
                                                           depends_on=("collect_r0",))])
        scheduler.start(scheduler.plan_step()["assignments"][0])
        scheduler.complete("collect_r0", status="failed", detail={"why": "fixture"})
        self.assertEqual(scheduler.state("eval_r0"), "blocked")
        step = scheduler.plan_step()
        self.assertEqual(step["assignments"], [])
        self.assertEqual(step["waiting"][0].reason, "failed_dependency")
        self.assertFalse(scheduler.runnable())

    def test_both_budgets_must_afford_estimate_plus_reserve(self):
        ledger = self.ledger(wall_limit=900.0)
        scheduler = self.scheduler([job("long_collect", estimate_seconds=600.0,
                                        reserve_seconds=600.0)], ledger=ledger)
        step = scheduler.plan_step()
        self.assertEqual(step["assignments"], [])
        self.assertEqual(step["waiting"][0].reason, "budget")
        self.assertIn("wall_clock", step["waiting"][0].detail)
        ledger = self.ledger(gpu_hours=0.5)
        scheduler = self.scheduler([job("collect", estimate_seconds=1200.0,
                                        reserve_seconds=1200.0)], ledger=ledger)
        self.assertIn("gpu_hours", scheduler.plan_step()["waiting"][0].detail)


class LeaseTests(SchedulerTestCase):
    def test_starting_a_job_holds_a_lease_per_device(self):
        scheduler = self.scheduler([job("collect_r0")])
        assignment = scheduler.plan_step()["assignments"][0]
        receipt = scheduler.start(assignment)
        self.assertEqual([row["uuid"] for row in receipt["leases"]], ["GPU-0"])
        held = occupancy(self.leases)
        self.assertEqual([row["uuid"] for row in held], ["GPU-0"])
        self.assertEqual(held[0]["job"], "collect_r0")
        scheduler.complete("collect_r0")
        self.assertEqual(occupancy(self.leases), [])          # released on completion

    def test_a_second_scheduler_cannot_take_a_leased_device(self):
        first = self.scheduler([job("collect_r0")])
        first.start(first.plan_step()["assignments"][0])
        second = self.scheduler([job("collect_r0")], lease_root=self.leases)
        assignment = second.plan_step()["assignments"][0]
        with self.assertRaises(Exception) as caught:
            second.start(assignment)
        self.assertIn("held by another process", str(caught.exception))

    def test_a_crashed_job_gives_its_devices_back_without_a_reaper(self):
        scheduler = self.scheduler([job("collect_r0"), job("collect_r1")])
        for item in scheduler.plan_step()["assignments"]:
            scheduler.start(item)
        scheduler.abandon("collect_r0", reason="fixture crash")
        self.assertTrue((scheduler.job_output("collect_r0") / "abandoned.json").is_file())
        free = [str(device["uuid"]) for device in scheduler.free_devices()]
        self.assertIn("GPU-0", free)
        self.assertNotIn("GPU-1", free)                          # the other job still holds it
        self.assertEqual(scheduler.state("collect_r0"), "ready")     # unresolved, not failed

    def test_devices_are_still_held_while_the_job_runs(self):
        scheduler = self.scheduler([job("collect_r0")])
        scheduler.start(scheduler.plan_step()["assignments"][0])
        self.assertEqual([row["uuid"] for row in occupancy(self.leases)], ["GPU-0"])
        self.assertEqual(scheduler.snapshot()["running"]["collect_r0"]["devices"], ["GPU-0"])


class ResumeTests(SchedulerTestCase):
    def test_a_completed_job_is_never_run_again(self):
        first = self.scheduler([job("collect_r0")])
        first.start(first.plan_step()["assignments"][0])
        first.complete("collect_r0", detail={"episodes": 40})
        resumed = self.scheduler([job("collect_r0")])
        self.assertEqual(resumed.restore(), {"completed": 1, "failed": 0})
        self.assertEqual(resumed.state("collect_r0"), "completed")
        self.assertEqual(resumed.plan_step()["assignments"], [])
        self.assertEqual(json.loads(resumed.outcome_path("collect_r0").read_text())["detail"],
                         {"episodes": 40})
        self.assertEqual(resumed.restore(), {"completed": 0, "failed": 0})   # idempotent

    def test_a_failed_job_is_not_silently_retried(self):
        scheduler = self.scheduler([job("eval_final", category="evaluation")])
        scheduler.start(scheduler.plan_step()["assignments"][0])
        scheduler.complete("eval_final", status="failed", detail={"reason": "score-bearing"})
        resumed = self.scheduler([job("eval_final", category="evaluation")])
        resumed.restore()
        self.assertEqual(resumed.state("eval_final"), "failed")
        self.assertEqual(resumed.plan_step()["assignments"], [])
        self.assertIn("score-bearing",
                      json.loads(resumed.failure_path("eval_final").read_text())["detail"]["reason"])

    def test_a_half_finished_job_is_pending_again(self):
        (self.root / "jobs" / "collect_r0").mkdir(parents=True)
        (self.root / "jobs" / "collect_r0" / "partial.txt").write_text("half", encoding="utf-8")
        scheduler = self.scheduler([job("collect_r0")])
        self.assertEqual(scheduler.restore(), {"completed": 0, "failed": 0})
        self.assertEqual(scheduler.state("collect_r0"), "ready")

    def test_a_dependent_job_runs_after_a_completed_dependency_on_resume(self):
        first = self.scheduler([job("collect_r0"), job("eval_r0", category="evaluation",
                                                       depends_on=("collect_r0",))])
        first.start(first.plan_step()["assignments"][0])
        first.complete("collect_r0")
        resumed = self.scheduler([job("collect_r0"),
                                  job("eval_r0", category="evaluation",
                                      depends_on=("collect_r0",))])
        resumed.restore()
        self.assertEqual(resumed.state("collect_r0"), "completed")
        self.assertEqual(resumed.state("eval_r0"), "ready")
        self.assertEqual([item.job.name for item in resumed.plan_step()["assignments"]],
                         ["eval_r0"])


class AccountingIntegrationTests(SchedulerTestCase):
    def test_a_finished_job_is_charged_with_the_devices_it_held(self):
        ledger = self.ledger()
        scheduler = self.scheduler([job("collect_r0", estimate_seconds=600.0)], ledger=ledger)
        scheduler.start(scheduler.plan_step()["assignments"][0])
        scheduler.complete("collect_r0", episodes=40)
        record = json.loads((self.root / "accounting.json").read_text())
        self.assertEqual(record["jobs"][0]["job"], "collect_r0")
        self.assertEqual(record["jobs"][0]["episodes"], 40)
        self.assertEqual(record["gpu_hours"]["by_category"]["collection"],
                         ledger.charged_gpu_hours)
        self.assertGreater(ledger.charged_gpu_hours, 0.0)

    def test_a_failed_job_is_charged_too(self):
        ledger = self.ledger()
        scheduler = self.scheduler([job("eval_final", category="evaluation")], ledger=ledger)
        scheduler.start(scheduler.plan_step()["assignments"][0])
        scheduler.complete("eval_final", status="failed")
        record = json.loads((self.root / "accounting.json").read_text())
        self.assertEqual(record["gpu_hours"]["by_category"]["evaluation"],
                         ledger.charged_gpu_hours)
        self.assertEqual(record["gpu_hours"]["by_category"].get("retry", 0.0), 0.0)

    def test_the_snapshot_names_why_each_job_is_not_running(self):
        scheduler = self.scheduler([job("a"), job("b")], max_parallel_jobs=1)
        scheduler.start(scheduler.plan_step()["assignments"][0])
        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot["next"]["assignments"], [])
        self.assertEqual(snapshot["next"]["waiting"][0]["job"], "b")
        self.assertEqual(snapshot["next"]["waiting"][0]["reason"], "parallel_limit")
        self.assertEqual(snapshot["running"]["a"]["devices"], ["GPU-0"])

    def test_a_scheduler_without_devices_refuses_to_exist(self):
        with self.assertRaises(ScheduleRefused) as caught:
            self.scheduler([job("a")], devices=[])
        self.assertIn("no usable device", str(caught.exception))

    def test_an_unknown_dependency_or_category_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            self.scheduler([job("a", depends_on=("ghost",))])
        with self.assertRaises(ValueError):
            Job("a", category="not_a_category")
        with self.assertRaises(ValueError):
            self.scheduler([job("a"), job("a")])


if __name__ == "__main__":
    unittest.main()
