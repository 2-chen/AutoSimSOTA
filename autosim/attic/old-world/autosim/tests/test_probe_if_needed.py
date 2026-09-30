"""A multi-device run measures its own container when no receipt matches it.

A device receipt is keyed by the node it was measured on, and on this cluster the
container's hostname is job-scoped (``pt-<job uid>-worker-0``), so a receipt published by a
*separate* probe job can never authorize a research run in *this* job.  `--probe-if-needed`
is the answer: the run measures the devices it is actually holding, then plans from that.

What these tests pin down:

* the ladder is run as its own process, in the run's own directory, honouring the mode the
  run asked for;
* the probe is charged to the same GPU-hour account as everything else;
* only a *missing receipt* triggers it -- a busy or incompatible device is a fact about the
  node that measuring again cannot change, and re-probing it would just burn an hour;
* a probe that does not publish (or publishes and still cannot plan) reports the artifact
  rather than retrying forever or proceeding unverified.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autosim.research import repository_autoresearch as ra
from autosim.research.accounting import BudgetLedger
from autosim.research.device_cli import DevicePlanRefused, DeviceReceiptMissing
from autosim.research.devices import NoCompatibleDevice

UUIDS = ["eed8f088-ad1c-3c82-e41f-c8bd3d8e80d8", "aaaa1111-2222-3333-4444-555566667777",
         "cccc1111-2222-3333-4444-555566667777", "bbbb1111-2222-3333-4444-555566667777"]


def fake_plan(*, mode="pinned_index"):
    return {"schema_version": 1, "mode": mode, "requested": "auto",
            "usable": [{"index": i, "uuid": uuid, "model": "NVIDIA GeForce RTX 5090",
                        "class_name": "rtx5090", "renderer": "hybrid",
                        "selection": {"mode": mode, "index": i, "uuid": uuid,
                                      "vulkan_gpu_id": i, "torch_index": 0,
                                      "cuda_visible": str(i), "renderer": "hybrid"}}
                       for i, uuid in enumerate(UUIDS)],
            "waiting": [], "max_parallel_jobs": 4, "gpu_hours_limit": 96.0,
            "legacy_equivalence": False, "plan_digest": "d" * 64,
            "resource_summary": {"node": "node-4gpu"}}


def receipt_document(*, passed=True, mode="pinned_index"):
    return {"schema_version": 1, "kind": "autosim_device_probe", "passed": passed, "mode": mode,
            "verified_devices": sorted(UUIDS) if passed else [],
            "allocation": [{"index": i, "uuid": uuid} for i, uuid in enumerate(UUIDS)],
            "verdict": {"winner": "preferred" if passed else None,
                        "selection_mode": mode if passed else None,
                        "model_discrepancies": [] if passed else ["no_addressing_mode_verified"]}}


class ProbeIfNeededTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)

    def runner(self, **config):
        settings = dict(repo_input="RoboSynChallenge", output_root=self.root, run_id="probe_gate",
                        task="click_bell", gpus="auto", probe_if_needed=True)
        settings.update(config)
        return ra.RepositoryAutoResearch(ra.MilestoneConfig(**settings))

    def stages(self, runner):
        return [row.get("stage") for row in [runner.state]]

    # --- the gate itself -------------------------------------------------------------

    def test_a_missing_receipt_is_refused_when_the_flag_was_not_passed(self):
        runner = self.runner(probe_if_needed=False)
        with mock.patch.object(ra, "resolve_plan", side_effect=DeviceReceiptMissing("no receipt")):
            with mock.patch.object(runner, "_run_device_probe") as probe:
                with self.assertRaises(RuntimeError) as caught:
                    runner._resolve_device_plan(self.root)
        self.assertIn("multi-device plan refused", str(caught.exception))
        probe.assert_not_called()
        self.assertFalse((runner.run_root / "device_plan.json").exists())

    def test_a_missing_receipt_probes_this_container_and_then_plans(self):
        runner = self.runner()
        calls = []

        def resolve_plan(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise DeviceReceiptMissing("no passing device receipt for this node")
            return fake_plan()

        with mock.patch.object(ra, "resolve_plan", side_effect=resolve_plan):
            with mock.patch.object(runner, "_run_device_probe",
                                   return_value={"published": True,
                                                 "artifact": "run/device_probe/device_probe.json",
                                                 "detail": "rc=0, passed=True"}) as probe:
                plan = runner._resolve_device_plan(self.root)

        self.assertEqual(probe.call_count, 1, "the ladder runs exactly once")
        self.assertEqual(len(calls), 2, "resolve_plan is retried once, with the same node")
        self.assertEqual(plan["mode"], "pinned_index")
        self.assertEqual(runner.state["stage"], "device_plan_resolved")
        self.assertTrue((runner.run_root / "device_plan.json").is_file())

    def test_a_busy_or_incompatible_device_is_never_probed(self):
        """Measuring again cannot free a busy card, so it must not be attempted."""
        runner = self.runner()
        with mock.patch.object(ra, "resolve_plan",
                               side_effect=DevicePlanRefused("index 2 is occupied")):
            with mock.patch.object(runner, "_run_device_probe") as probe:
                with self.assertRaises(RuntimeError):
                    runner._resolve_device_plan(self.root)
        probe.assert_not_called()
        with mock.patch.object(ra, "resolve_plan",
                               side_effect=NoCompatibleDevice("mixed node")):
            with mock.patch.object(runner, "_run_device_probe") as probe:
                with self.assertRaises(RuntimeError):
                    runner._resolve_device_plan(self.root)
        probe.assert_not_called()

    def test_a_probe_that_does_not_publish_is_reported_with_its_reason(self):
        runner = self.runner()
        with mock.patch.object(ra, "resolve_plan", side_effect=DeviceReceiptMissing("no receipt")):
            with mock.patch.object(runner, "_run_device_probe",
                                   return_value={"published": False, "artifact": "run/device_probe",
                                                 "detail": "rc=1, passed=False, mode=None"}):
                with self.assertRaises(RuntimeError) as caught:
                    runner._resolve_device_plan(self.root)
        message = str(caught.exception)
        self.assertIn("did not publish a usable receipt", message)
        self.assertIn("rc=1", message)

    def test_a_refusal_after_measuring_names_the_measurement(self):
        runner = self.runner()
        with mock.patch.object(ra, "resolve_plan",
                               side_effect=[DeviceReceiptMissing("no receipt"),
                                            DevicePlanRefused("mode identity was not verified")]):
            with mock.patch.object(runner, "_run_device_probe",
                                   return_value={"published": True,
                                                 "artifact": "run/device_probe/device_probe.json",
                                                 "detail": "rc=0, passed=True"}):
                with self.assertRaises(RuntimeError) as caught:
                    runner._resolve_device_plan(self.root)
        message = str(caught.exception)
        self.assertIn("mode identity was not verified", message)
        self.assertIn("device_probe.json", message)

    # --- running the ladder ----------------------------------------------------------

    def test_the_probe_runs_as_its_own_process_in_the_run_directory(self):
        runner = self.runner(device_mode="identity")
        runner.task = "click_bell"
        seen = {}

        def fake_call(command, *args, **kwargs):
            seen["command"] = command
            output = Path(command[command.index("--output") + 1])
            (output / "device_probe.json").write_text(
                json.dumps(receipt_document(mode="identity")), encoding="utf-8")
            return 0

        outcome = runner._run_device_probe(self.root, runner=fake_call)
        command = seen["command"]
        self.assertEqual(command[1:3], ["-m", "autosim.research.device_probe"])
        self.assertEqual(command[command.index("--task") + 1], "click_bell")
        self.assertEqual(command[command.index("--gpus") + 1], "auto")
        self.assertEqual(command[command.index("--mode") + 1], "identity",
                         "the ladder measures the mode this run asked for")
        self.assertEqual(Path(command[command.index("--output") + 1]),
                         runner.run_root / "device_probe")
        self.assertTrue(outcome["published"])
        self.assertEqual(outcome["artifact"], str(runner.run_root / "device_probe/device_probe.json"))
        self.assertEqual(runner.state["stage"], "device_probe_finished")
        self.assertEqual(runner.state["device_probe"]["verified_devices"], sorted(UUIDS))

    def test_auto_mode_measures_pinned_index(self):
        """'auto' has nothing to measure until a receipt names a mode; 'pinned_index' is
        the shipped default, and the receipt it publishes decides what 'auto' will mean."""
        runner = self.runner()
        seen = {}

        def fake_call(command, *args, **kwargs):
            seen["mode"] = command[command.index("--mode") + 1]
            return 1

        runner._run_device_probe(self.root, runner=fake_call)
        self.assertEqual(seen["mode"], "pinned_index")

    def test_a_failed_ladder_leaves_its_artifact_and_no_receipt(self):
        runner = self.runner()

        def fake_call(command, *args, **kwargs):
            output = Path(command[command.index("--output") + 1])
            (output / "device_probe.json").write_text(
                json.dumps(receipt_document(passed=False)), encoding="utf-8")
            return 1

        outcome = runner._run_device_probe(self.root, runner=fake_call)
        self.assertFalse(outcome["published"])
        self.assertEqual(runner.state["device_probe"]["passed"], False)
        self.assertEqual(runner.state["device_probe"]["verdict"]["selection_mode"], None)

    # --- accounting -------------------------------------------------------------------

    def test_the_probe_is_charged_to_the_gpu_hour_account(self):
        runner = self.runner()
        ledger = BudgetLedger(runner.run_root / "accounting.json", wall_limit_seconds=24 * 3600,
                              gpu_hours_limit=96.0,
                              devices=[{"uuid": uuid, "index": i, "model": "NVIDIA GeForce RTX 5090",
                                        "class_name": "rtx5090"} for i, uuid in enumerate(UUIDS)])
        runner.probe_charge = {"job": "device_probe", "category": "probe",
                               "wall_seconds": 1800.0, "status": "completed",
                               "devices": [str(uuid) for uuid in UUIDS],
                               "detail": {"passed": True, "mode": "pinned_index"}}
        with mock.patch.object(runner, "device_ledger", return_value=ledger):
            runner._charge_device_probe()
        snapshot = ledger.snapshot()
        self.assertAlmostEqual(snapshot["gpu_hours"]["charged"], 2.0, places=6)
        self.assertEqual(snapshot["gpu_hours"]["by_category"]["probe"], 2.0)
        self.assertEqual([row["job"] for row in snapshot["jobs"]], ["device_probe"])
        self.assertEqual(snapshot["jobs"][0]["category"], "probe")
        self.assertEqual(len(snapshot["occupancy"]["device_seconds"]), len(UUIDS))

    def test_nothing_is_charged_when_the_probe_did_not_run(self):
        runner = self.runner()
        ledger = BudgetLedger(runner.run_root / "accounting.json", wall_limit_seconds=24 * 3600,
                              gpu_hours_limit=96.0, devices=[{"uuid": UUIDS[0], "index": 0}])
        with mock.patch.object(runner, "device_ledger", return_value=ledger):
            runner._charge_device_probe()
        self.assertEqual(ledger.snapshot()["gpu_hours"]["charged"], 0.0)

    def test_a_legacy_run_has_no_probe_account_to_charge(self):
        """The legacy path keeps no ledger at all, so the charge is a no-op there."""
        runner = self.runner()
        runner.probe_charge = {"job": "device_probe", "category": "probe", "wall_seconds": 60.0,
                               "status": "completed", "devices": [UUIDS[0]], "detail": {}}
        runner._charge_device_probe()          # no ledger: must not raise
        self.assertIsNone(runner.ledger)


if __name__ == "__main__":
    unittest.main()
