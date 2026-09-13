import json
import tempfile
import unittest
from pathlib import Path

from autosim.research import device_cli
from autosim.research.device_cli import (DevicePlanRefused, plan_digest, plan_identity,
                                         protocol_mismatch, resolve_plan)
from autosim.research.devices import NoCompatibleDevice, discover, probe_cache_dir

IDLE = """0, GPU-eed8f088-ad1c-3c82-e41f-c8bd3d8e80d8, NVIDIA GeForce RTX 5090, 32607, 12, 0, 580.95.05, 12.0
1, GPU-aaaa1111-2222-3333-4444-555566667777, NVIDIA GeForce RTX 5090, 32607, 8, 0, 580.95.05, 12.0
2, GPU-cccc1111-2222-3333-4444-555566667777, NVIDIA GeForce RTX 5090, 32607, 4, 0, 580.95.05, 12.0
3, GPU-bbbb1111-2222-3333-4444-555566667777, NVIDIA GeForce RTX 5090, 32607, 6, 0, 580.95.05, 12.0
"""

BUSY = """0, GPU-eed8f088-ad1c-3c82-e41f-c8bd3d8e80d8, NVIDIA GeForce RTX 5090, 32607, 31120, 97, 580.95.05, 12.0
1, GPU-aaaa1111-2222-3333-4444-555566667777, NVIDIA GeForce RTX 5090, 32607, 8, 0, 580.95.05, 12.0
2, GPU-cccc1111-2222-3333-4444-555566667777, NVIDIA GeForce RTX 5090, 32607, 4, 0, 580.95.05, 12.0
3, GPU-bbbb1111-2222-3333-4444-555566667777, NVIDIA GeForce RTX 5090, 32607, 6, 0, 580.95.05, 12.0
"""

UUIDS = ["eed8f088-ad1c-3c82-e41f-c8bd3d8e80d8", "aaaa1111-2222-3333-4444-555566667777",
         "cccc1111-2222-3333-4444-555566667777", "bbbb1111-2222-3333-4444-555566667777"]
TORCH_ROWS = [{"index": i, "uuid": uuid, "name": "NVIDIA GeForce RTX 5090"}
              for i, uuid in enumerate(UUIDS)]


def make_report(*, table=IDLE, environ=None):
    def runner(command, **kwargs):
        if "--query-compute-apps" in " ".join(command):
            return ""
        return table
    return discover(runner=runner, environ=environ or {}, torch_rows=TORCH_ROWS,
                    disk_path=Path(tempfile.gettempdir()), host="node-4gpu")


def receipt(*, mode="pinned_index", verified=None, digest=None):
    verified = UUIDS if verified is None else verified
    return {"passed": True, "mode": mode, "probe_finished_at": "2026-09-13T16:00:00Z",
            "probe_receipt_key": digest or "0" * 64,
            "devices": {uuid: ("verified_sim" if uuid in verified else "unknown")
                        for uuid in UUIDS}}


class ResolvePlanTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)

    def publish(self, row):
        key = device_cli.receipt_lookup_key(make_report(), image="img", worker_spec="spec")
        path = probe_cache_dir(self.root) / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row), encoding="utf-8")
        return key

    def resolve(self, **overrides):
        kwargs = dict(platform_root=self.root, requested="auto", mode="pinned_index",
                      hours=24.0, image="img", worker_spec="spec", report=make_report())
        kwargs.update(overrides)
        return resolve_plan(**kwargs)

    def test_without_a_receipt_the_plan_is_refused_and_the_probe_is_named(self):
        with self.assertRaises(DevicePlanRefused) as caught:
            self.resolve()
        message = str(caught.exception)
        self.assertIn("autosim.research.device_probe", message)
        self.assertIn("never assumed available", message)

    def test_a_receipt_verified_under_another_mode_is_not_reused(self):
        self.publish(receipt(mode="identity"))
        with self.assertRaises(DevicePlanRefused) as caught:
            self.resolve(mode="pinned_index")
        self.assertIn("identity", str(caught.exception))

    def test_auto_adopts_the_mode_the_node_was_verified_under(self):
        self.publish(receipt(mode="identity"))
        plan = self.resolve(mode="auto")
        self.assertEqual(plan["mode"], "identity")
        self.assertEqual(plan["requested_mode"], "auto")
        self.assertEqual(plan["probe_receipt_mode"], "identity")

    def test_unverified_devices_wait_with_a_reason_instead_of_being_used(self):
        self.publish(receipt(verified=UUIDS[:2]))
        plan = self.resolve(requested="auto", accept_partial=True)
        self.assertEqual([gpu["index"] for gpu in plan["usable"]], [0, 1])
        self.assertEqual([row["index"] for row in plan["waiting"]], [2, 3])
        self.assertTrue(plan["accepted_partial"])
        self.assertIn("unknown", plan["waiting"][0]["reason"])

    def test_a_partial_plan_needs_an_explicit_acknowledgement(self):
        self.publish(receipt(verified=UUIDS[:3]))
        with self.assertRaises(DevicePlanRefused) as caught:
            self.resolve(requested="auto")
        self.assertIn("--accept-device-plan", str(caught.exception))
        plan = self.resolve(requested="auto", accept_partial=True)
        self.assertEqual(len(plan["usable"]), 3)

    def test_a_busy_device_is_allocatable_neither_way(self):
        busy = make_report(table=BUSY)
        self.publish(receipt())
        plan = self.resolve(report=busy, requested="auto", accept_partial=True)
        self.assertEqual([gpu["index"] for gpu in plan["usable"]], [1, 2, 3])
        self.assertIn("31120 MiB in use", " ".join(row["reason"] for row in plan["waiting"]))
        self.assertEqual([row["capability"] for row in plan["waiting"]], ["busy"])

    def test_no_usable_device_raises_with_the_resource_reason(self):
        self.publish(receipt(verified=[]))
        with self.assertRaises(NoCompatibleDevice) as caught:
            self.resolve()
        self.assertIn("0 of 4 allowed devices usable", str(caught.exception))

    def test_devices_outside_the_outer_filter_are_refused(self):
        self.publish(receipt())
        outer = make_report(environ={"CUDA_VISIBLE_DEVICES": "2,3"})
        with self.assertRaises(ValueError) as caught:
            self.resolve(report=outer, requested="0,1", accept_partial=True)
        self.assertIn("outside the allowed range", str(caught.exception))

    def test_only_the_card_legacy_would_take_is_legacy_equivalent(self):
        self.publish(receipt())
        self.assertTrue(self.resolve(requested="0")["legacy_equivalence"])
        self.assertFalse(self.resolve(requested="auto")["legacy_equivalence"])
        self.assertFalse(self.resolve(requested="0,1", accept_partial=True)["legacy_equivalence"])
        self.assertFalse(self.resolve(requested="1")["legacy_equivalence"])

    def test_identity_mode_built_from_a_pinned_receipt_is_refused(self):
        self.publish(receipt(mode="pinned_index"))
        with self.assertRaises(DevicePlanRefused):
            self.resolve(mode="identity")

    def test_the_plan_carries_the_receipt_digest_and_the_resource_summary(self):
        key = self.publish(receipt())
        plan = self.resolve(requested="0,1", accept_partial=True)
        self.assertEqual(len(plan["probe_receipt_sha256"]), 64)
        self.assertEqual(plan["probe_receipt_key"], key)
        self.assertEqual(len(plan["plan_digest"]), 64)
        self.assertIn("usable", plan["resource_summary"])
        self.assertEqual(plan["gpu_hours_limit"], 48.0)      # hours x usable
        self.assertEqual(plan["max_parallel_jobs"], 2)       # clamped to the usable set
        self.assertEqual([gpu["selection"]["cuda_visible"] for gpu in plan["usable"]],
                         ["0", "1"])


class PlanIdentityTests(unittest.TestCase):
    def test_occupancy_is_not_identity_but_the_device_set_is(self):
        def plan(rows, memory):
            table = "".join(
                f"{index}, GPU-{uuid}, NVIDIA GeForce RTX 5090, 32607, {memory}, 0, 580.95.05, 12.0\n"
                for index, uuid in enumerate(rows))
            return make_report(table=table)

        base = {"mode": "pinned_index", "requested": "auto",
                "allowed_uuids": UUIDS, "max_parallel_jobs": 4, "gpu_hours_limit": 96.0}
        moved = dict(base)
        self.assertEqual(plan_digest(base), plan_digest(moved))
        moved["max_parallel_jobs"] = 2
        self.assertNotEqual(plan_digest(base), plan_digest(moved))
        self.assertEqual(plan_identity(base)["requested"], "auto")
        self.assertNotIn("usable", plan_identity(base))   # occupancy lives outside identity

    def test_a_resume_reports_the_field_that_changed(self):
        recorded = {"mode": "pinned_index", "requested": "auto", "allowed_uuids": UUIDS,
                    "max_parallel_jobs": 4, "gpu_hours_limit": 96.0}
        resolved = dict(recorded, max_parallel_jobs=2)
        self.assertEqual(protocol_mismatch(recorded, resolved),
                         {"max_parallel_jobs": {"recorded": 4, "requested": 2}})
        self.assertEqual(protocol_mismatch(recorded, dict(recorded)), {})


class ProtocolIntegrationTests(unittest.TestCase):
    """The invocation surface: what enters the frozen protocol, and what a resume refuses.

    The budget key list below is the one every pre-multi-device run recorded; the point of
    the test is that the new knobs never appear in it, which is what keeps a legacy
    `--gpu i` run byte-identical (and its resume legal).
    """

    LEGACY_BUDGET_KEYS = [
        "attempts_per_round", "controller", "development_episodes", "dry_run",
        "final_episodes", "full_budget", "gpu", "hours", "min_original_fraction",
        "output_root", "probe_only", "repo_input", "rounds", "run_id", "screen_steps",
        "selection_episodes", "task", "train_seed"]

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)

    def config(self, **overrides):
        from autosim.research.repository_autoresearch import MilestoneConfig
        kwargs = dict(repo_input="/tmp/repo", output_root=self.root, run_id="run-1")
        kwargs.update(overrides)
        return MilestoneConfig(**kwargs)

    def runner(self, **overrides):
        from autosim.research.repository_autoresearch import RepositoryAutoResearch
        return RepositoryAutoResearch(self.config(**overrides))

    def test_a_legacy_invocation_keeps_the_recorded_budget_keys(self):
        from autosim.research.repository_autoresearch import protocol_budget
        self.assertEqual(sorted(protocol_budget(self.config())), self.LEGACY_BUDGET_KEYS)
        multi = protocol_budget(self.config(gpus="auto", device_mode="pinned_index",
                                           max_parallel_jobs=2, gpu_hours=48.0,
                                           accept_device_plan=True, multi_gpu_probe=True))
        self.assertEqual(sorted(multi), self.LEGACY_BUDGET_KEYS)   # same keys, same values
        self.assertEqual(multi, protocol_budget(self.config()))

    def test_the_cli_keeps_the_single_device_default_and_rejects_mixed_forms(self):
        from autosim.research.repository_autoresearch import (make_parser,
                                                              _reject_ambiguous_devices)
        parser = make_parser()
        legacy = parser.parse_args(["/tmp/repo", "--gpu", "3"])
        _reject_ambiguous_devices(parser, legacy)
        self.assertEqual((legacy.gpu, legacy.gpus, legacy.device_mode), ("3", None, "auto"))
        mixed = parser.parse_args(["/tmp/repo", "--gpu", "3", "--gpus", "auto"])
        with self.assertRaises(SystemExit):
            _reject_ambiguous_devices(parser, mixed)
        lonely = parser.parse_args(["/tmp/repo", "--gpu-hours", "48"])
        with self.assertRaises(SystemExit):
            _reject_ambiguous_devices(parser, lonely)

    def test_a_resume_must_repeat_the_device_plan_it_was_recorded_under(self):
        runner = self.runner(gpus="auto", max_parallel_jobs=2)
        (runner.run_root / "protocol.json").write_text(json.dumps({"budget": {}}), encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            runner._check_recorded_plan({"budget": {"devices": {"plan_digest": "abc"}}})
        self.assertIn("executed single-device", str(caught.exception))

        recorded = {"plan_digest": "abc", "mode": "pinned_index", "requested": "auto",
                    "max_parallel_jobs": 4, "gpu_hours_limit": 96.0}
        (runner.run_root / "protocol.json").write_text(
            json.dumps({"budget": {"devices": recorded}}), encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            runner._check_recorded_plan({"budget": {"devices": dict(recorded, max_parallel_jobs=2)}})
        self.assertEqual(caught.exception.args[0],
                         "resume device plan differs from the recorded one: "
                         "{'max_parallel_jobs': {'recorded': 4, 'requested': 2}}")
        runner._check_recorded_plan({"budget": {"devices": dict(recorded)}})   # identical: allowed

    def test_a_single_device_run_cannot_be_resumed_with_the_multi_device_flags(self):
        runner = self.runner(gpus="auto")
        (runner.run_root / "protocol.json").write_text(
            json.dumps({"budget": {"devices": {"plan_digest": "abc"}}}), encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            runner._check_recorded_plan({"budget": {}})
        self.assertIn("resuming it without --gpus", str(caught.exception))
        self.assertIn("recorded plan: abc", str(caught.exception))

    def test_a_multi_device_plan_executes_and_takes_no_legacy_index_lock(self):
        """Where the plan used to be refused, it now runs -- under a different lock rule.

        The legacy index lock exists to keep one single-device run off another run's card,
        and it is exactly what the UUID leases replace.  Taking both would be self-defeating:
        the leases read a held legacy lock as occupancy, so a multi-device run holding index
        0 would be blocked from its own first lease.
        """
        from contextlib import nullcontext
        from unittest import mock
        plan = {"mode": "pinned_index",
                "usable": [{"index": 0, "uuid": "GPU-a"}, {"index": 1, "uuid": "GPU-b"}],
                "waiting": [], "max_parallel_jobs": 2, "gpu_hours_limit": 48.0,
                "legacy_equivalence": False}

        class Reached(Exception):
            """Raised where the first device work would begin: far enough to observe."""

        def observe(plan_row, **overrides):
            runner = self.runner(gpus="auto", controller="fixed", **overrides)
            held = []

            def fake_initialize():
                runner.runtime = mock.Mock()
                runner.spec = mock.Mock()

            with mock.patch.object(type(runner), "initialize", side_effect=fake_initialize), \
                    mock.patch.object(type(runner), "_prepare_official_data", side_effect=Reached), \
                    mock.patch("autosim.research.repository_autoresearch.gpu_lock",
                               side_effect=lambda index: held.append(index) or nullcontext()):
                runner.plan = plan_row
                with self.assertRaises(Reached):
                    runner.execute()
            return runner, held

        runner, held = observe(plan)
        self.assertEqual(held, [])                          # no legacy lock was taken
        ledger = runner.device_ledger()                     # the GPU-hour account is open
        self.assertEqual(ledger.gpu_hours_limit, 48.0)      # ... on the plan's budget
        self.assertEqual(ledger.path, runner.run_root / "accounting.json")

        runner, held = observe(None)                        # the legacy invocation still takes it
        self.assertEqual(held, ["0"])
        self.assertIsNone(runner.device_ledger())           # ... and keeps no new artifacts
        self.assertFalse((runner.run_root / "accounting.json").exists())


if __name__ == "__main__":
    unittest.main()
