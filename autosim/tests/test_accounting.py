import os
import tempfile
import unittest
from pathlib import Path

from autosim.research.accounting import (BudgetLedger, Charge, UtilizationSampler,
                                         describe_process)
from autosim.research.common import read_json


def biggest_first(processes):
    """The order the receipt presents holders in: largest memory first."""
    return [pid for pid, _ in sorted(processes.items(), key=lambda item: -item[1]["mib"])]

DEVICES = [{"uuid": "GPU-aaa", "index": 0, "model": "NVIDIA GeForce RTX 5090",
            "class_name": "rtx5090"},
           {"uuid": "GPU-bbb", "index": 1, "model": "NVIDIA GeForce RTX 5090",
            "class_name": "rtx5090"}]


class Clock:
    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.path = Path(self._temp.name) / "accounting.json"
        self.clock = Clock()

    def ledger(self, *, wall=86400.0, gpu_hours=96.0, path=None):
        return BudgetLedger(path or self.path, wall_limit_seconds=wall,
                            gpu_hours_limit=gpu_hours, devices=DEVICES, clock=self.clock)

    def test_two_devices_for_one_hour_is_two_gpu_hours(self):
        ledger = self.ledger()
        ledger.charge(Charge(job="eval", category="evaluation",
                             devices=("GPU-aaa", "GPU-bbb"), wall_seconds=3600.0))
        self.assertEqual(ledger.charged_gpu_hours, 2.0)
        self.assertEqual(ledger.remaining_gpu_hours, 94.0)
        self.assertEqual(sorted(ledger.snapshot()["gpu_hours"]["by_class"]), ["rtx5090"])
        self.assertEqual(ledger.snapshot()["gpu_hours"]["by_class"]["rtx5090"], 2.0)

    def test_one_device_for_one_hour_is_one_gpu_hour(self):
        ledger = self.ledger()
        ledger.charge(Charge(job="eval", category="evaluation",
                             devices=("GPU-aaa",), wall_seconds=3600.0))
        self.assertEqual(ledger.charged_gpu_hours, 1.0)

    def test_both_budgets_can_refuse_a_start_and_name_the_reason(self):
        ledger = self.ledger(wall=100.0, gpu_hours=10.0)
        self.assertEqual(ledger.may_start(devices=1, estimate_seconds=50)["reason"], "affordable")
        self.assertEqual(ledger.may_start(devices=1, estimate_seconds=200)["reason"], "wall_clock")
        refusal = self.ledger(wall=86400.0, gpu_hours=2.0).may_start(
            devices=4, estimate_seconds=3600, category="training")
        self.assertEqual(refusal["reason"], "gpu_hours")
        self.assertEqual(refusal["needed_gpu_hours"], 4.0)

    def test_reserved_verification_cost_is_part_of_the_affordability_check(self):
        ledger = self.ledger(wall=1000.0, gpu_hours=10.0)
        self.assertFalse(ledger.may_start(devices=1, estimate_seconds=500,
                                          reserve_seconds=600)["allowed"])

    def test_wall_clock_elapsed_comes_from_the_budget_start_and_survives_resume(self):
        ledger = self.ledger(wall=1000.0)
        ledger.charge(Charge(job="a", category="evaluation", devices=("GPU-aaa",),
                             wall_seconds=100.0))
        self.clock.advance(400.0)
        ledger.write()
        resumed = self.ledger(path=self.path, wall=1000.0)
        self.assertAlmostEqual(resumed.elapsed_wall, 400.0, places=3)
        self.assertAlmostEqual(resumed.remaining_wall, 600.0, places=3)
        self.assertAlmostEqual(resumed.charged_gpu_hours, 100.0 / 3600.0, places=6)

    def test_waiting_is_recorded_against_wall_clock_only(self):
        ledger = self.ledger()
        ledger.charge(Charge(job="eval", category="evaluation", devices=("GPU-aaa",),
                             wall_seconds=60.0, waiting_seconds=300.0))
        self.assertEqual(ledger.charged_gpu_hours, 60.0 / 3600.0)
        self.assertAlmostEqual(ledger.snapshot()["gpu_hours"]["waiting_gpu_hours"],
                               300.0 / 3600.0, places=6)
        ledger.record_waiting(job="waited", devices=["GPU-bbb"], seconds=120.0)
        self.assertAlmostEqual(ledger.snapshot()["gpu_hours"]["waiting_gpu_hours"],
                               420.0 / 3600.0, places=6)

    def test_categories_are_separate(self):
        ledger = self.ledger()
        for category, seconds in (("collection", 3600.0), ("evaluation", 1800.0),
                                  ("training", 7200.0), ("retry", 600.0), ("api", 60.0)):
            ledger.charge(Charge(job=f"{category}_job", category=category,
                                 devices=("GPU-aaa",), wall_seconds=seconds))
        by_category = ledger.snapshot()["gpu_hours"]["by_category"]
        self.assertAlmostEqual(by_category["collection"], 1.0, places=6)
        self.assertAlmostEqual(by_category["training"], 2.0, places=6)
        self.assertAlmostEqual(by_category["evaluation"], 0.5, places=6)
        self.assertAlmostEqual(by_category["api"], 60.0 / 3600.0, places=6)
        with self.assertRaises(ValueError):
            ledger.charge(Charge(job="x", category="not_a_category"))

    def test_low_utilization_never_offsets_the_charge(self):
        ledger = self.ledger()
        ledger.charge(Charge(job="idle", category="evaluation", devices=("GPU-aaa",),
                             wall_seconds=3600.0, mean_utilization_pct={"GPU-aaa": 3.0}))
        self.assertEqual(ledger.charged_gpu_hours, 1.0)
        self.assertFalse(ledger.snapshot()["utilization"]["sampled"])
        self.assertIn("never offsets budget", ledger.snapshot()["utilization"]["note"])

    def test_heterogeneous_classes_are_reported_separately(self):
        devices = [{"uuid": "GPU-aaa", "class_name": "rtx5090"},
                   {"uuid": "GPU-ccc", "class_name": "h100"}]
        ledger = BudgetLedger(self.path, wall_limit_seconds=86400.0, gpu_hours_limit=96.0,
                              devices=devices, clock=self.clock)
        ledger.charge(Charge(job="mixed", category="evaluation",
                             devices=("GPU-aaa", "GPU-ccc"), wall_seconds=3600.0))
        by_class = ledger.snapshot()["gpu_hours"]["by_class"]
        self.assertEqual(by_class, {"h100": 1.0, "rtx5090": 1.0})

    def test_snapshot_records_occupancy_per_device(self):
        ledger = self.ledger()
        ledger.charge(Charge(job="eval", category="evaluation", devices=("GPU-aaa",),
                             wall_seconds=900.0))
        ledger.charge(Charge(job="train", category="training", devices=("GPU-bbb",),
                             wall_seconds=300.0))
        occupancy = ledger.snapshot()["occupancy"]["device_seconds"]
        self.assertEqual(occupancy, {"GPU-aaa": 900.0, "GPU-bbb": 300.0})

    def test_ledger_written_atomically_and_readable(self):
        ledger = self.ledger()
        written = ledger.write()
        self.assertEqual(written, self.path)
        record = read_json(self.path)
        self.assertEqual(record["kind"], "autosim_run_accounting")
        self.assertEqual(record["wall_clock"]["limit_seconds"], 86400.0)


class SamplerTests(unittest.TestCase):
    @staticmethod
    def runner(*, gpu, apps=""):
        """Answer the two queries a sample makes, in the spelling nvidia-smi uses."""
        def run(command):
            return apps if "--query-compute-apps" in " ".join(command) else gpu
        return run

    def test_sampler_reports_mean_utilization_and_peak_memory_by_uuid(self):
        outputs = iter([
            "GPU-aaa, 40, 1000\nGPU-bbb, 0, 5\n",
            "GPU-aaa, 60, 3000\nGPU-bbb, 0, 5\n",
        ])
        gpu = self.runner(gpu="GPU-aaa, 60, 3000\nGPU-bbb, 0, 5\n")

        def runner(command):
            if "--query-compute-apps" in " ".join(command):
                return ""
            try:
                return next(outputs)
            except StopIteration:
                return gpu(command)

        sampler = UtilizationSampler(runner=runner)
        sampler._sample_once()
        sampler._sample_once()
        summary = sampler.summary()
        self.assertEqual(summary["GPU-aaa"]["mean_utilization_pct"], 50.0)
        self.assertEqual(summary["GPU-aaa"]["peak_memory_mib"], 3000)
        self.assertEqual(summary["GPU-aaa"]["samples"], 2)
        self.assertEqual(summary["GPU-bbb"]["mean_utilization_pct"], 0.0)

    def test_sampler_names_the_process_holding_memory_on_a_card(self):
        """A card that was supposed to stay idle and moved anyway needs a pid, not a shrug.

        The second multi-GPU probe measured 1030 MiB appearing on card 0 while an identity
        arm ran on card 3, and no artifact recorded *which* process held it.  A pid alone is
        not enough either -- the process is gone by the time the receipt is read -- so the
        sampler records the command line while the process is still alive.
        """
        sampler = UtilizationSampler(runner=self.runner(
            gpu="GPU-aaa, 0, 4000\nGPU-bbb, 0, 1300\n",
            apps=f"GPU-bbb, {os.getpid()}, 1030\nGPU-aaa, 99, 3800\nGPU-aaa, 100, 512\n"))
        sampler._sample_once()
        summary = sampler.summary()
        holder = summary["GPU-bbb"]["processes"][str(os.getpid())]
        self.assertEqual(holder["mib"], 1030)
        self.assertIn("python", (holder["cmd"] or "").lower())
        self.assertEqual(holder["parent"], os.getppid())
        self.assertEqual(biggest_first(summary["GPU-aaa"]["processes"]), ["99", "100"])

    def test_a_process_that_cannot_be_read_is_reported_as_unreadable_not_skipped(self):
        self.assertEqual(describe_process(2 ** 22, proc=Path("/proc")),
                         {"cmd": None, "note": "unreadable: No such file or directory"})

    def test_sampler_ignores_a_failing_query(self):
        sampler = UtilizationSampler(runner=lambda command: "__error__: rc=9")
        sampler._sample_once()
        self.assertEqual(sampler.summary(), {})


if __name__ == "__main__":
    unittest.main()
