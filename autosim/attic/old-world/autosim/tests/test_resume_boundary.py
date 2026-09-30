import os
import resource
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path

from autosim.research.artifact_commits import ArtifactStore
from autosim.research.common import atomic_json, digest, object_digest, read_json
from autosim.research.continuation import ContinuationSupervisor, process_identity
from autosim.research.resume_boundary import BoundaryRefused, inspect_resume_boundary, load_resume_receipt


SCIENCE = {"task": "fixture", "recipe": "frozen"}


class ResumeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_root = self.root / "runs/fixture"
        self.evaluation = self.run_root / "evaluations/official_development"
        self.deadline = time.time() + 300
        self.previous = {"attempt": 1, "status": "failed", "execution_revision": "r1",
                         "process": {"pid": 999999999, "host": os.uname().nodename}}
        atomic_json(self.run_root / "run_state.json", {"stage": "official_development", "status": "blocked",
                                                       "final_confirmation_opened": False})
        atomic_json(self.run_root / "protocol.json", {"continuation_scientific_contract": SCIENCE})

    def failure(self, directory=None):
        directory = directory or self.evaluation
        atomic_json(directory / "startup.json", {"phase": "environment_constructing", "policy_has_acted": False})
        atomic_json(directory / "process/process.json", {"status": "failed", "returncode": -11,
            "pid": 999999999, "host": os.uname().nodename})
        return directory

    def inspect(self, **kwargs):
        options = dict(scientific_contract=SCIENCE, execution_revision="r1",
                       previous_attempt=self.previous, deadline_epoch=self.deadline)
        return inspect_resume_boundary(self.root, self.run_root, **(options | kwargs))

    def committed(self, key):
        target = self.run_root / key / "result.json"
        atomic_json(target, {"complete": True})
        store = ArtifactStore(self.run_root / "harness/artifact_commits", artifact_root=self.run_root)
        store.prepare(key, inputs={}, scientific_contract=SCIENCE, execution_revision="r1", attempt_id="one")
        store.transition(key, "running")
        store.transition(key, "validating")
        store.commit(key, outputs={"result": target}, transaction_id=key)
        return target

    def probe(self, *, attempts=1, limit=3, passed=False, root=None):
        root = root or self.run_root / "device_probe"
        atomic_json(root / "probe_contract.json", {"startup_attempts": limit})
        atomic_json(root / "budget.json", {"wall_clock": {"budget_started_at_epoch": time.time(), "limit_seconds": 1800},
            "gpu_hours": {"limit": 2, "charged": .1}})
        for number in range(1, attempts + 1):
            directory = root / "attempts" / f"attempt_{number}"
            generation = f"generation-{number}"
            atomic_json(directory / "attempt.json", {"attempt_id": number, "generation": generation,
                "passed": passed and number == attempts, "retryable": not passed})
            atomic_json(directory / "generation.json", {"generation": generation, "rows": {}})
            atomic_json(directory / "cleanup.json", {"generation": generation,
                                                      "confirmed_reaped": True, "unreaped_workers": []})
            self.failure(directory / "schedule/native_probe/native_0/native")
        atomic_json(root / "device_probe.json", {"passed": passed, "requested_devices": ["GPU-a"],
                                                 "verified_devices": ["GPU-a"] if passed else []})
        return root

    def test_pre_reset_failure_generates_hash_bound_envelope(self):
        self.failure()
        result = self.inspect()
        self.assertTrue(result["allowed"], result["reason"])
        self.assertTrue(result["resume_receipt"]["safe_boundary"])
        self.assertFalse(result["resume_receipt"]["episode_started"])
        row = load_resume_receipt(Path(result["receipt_path"]), expected_sha256=result["receipt_sha256"],
            run_root=self.run_root, scientific_contract=SCIENCE, execution_revision="r1")
        self.assertEqual(row["run_root"], str(self.run_root))
        with self.assertRaises(BoundaryRefused):
            load_resume_receipt(Path(result["receipt_path"]), expected_sha256="wrong",
                run_root=self.run_root, scientific_contract=SCIENCE, execution_revision="r1")

    def test_any_uncommitted_reset_prevents_replay(self):
        self.failure()
        atomic_json(self.evaluation / "reset_started.json", {"seed": 19})
        result = self.inspect()
        self.assertFalse(result["allowed"])
        self.assertIn("reset", result["reason"])

    def test_uncommitted_training_and_api_proposal_are_not_replayed(self):
        self.failure()
        for relative in ("rounds/round_1/candidate/recipe_200.json", "rounds/round_1/api/request.json"):
            path = self.run_root / relative
            atomic_json(path, {"record": "started"})
            self.assertFalse(self.inspect()["allowed"])
            path.unlink()

    def test_committed_round_may_be_reconstructed_without_replaying_its_old_resets(self):
        target = self.committed("rounds/round_1")
        atomic_json(target.parent / "reset_started.json", {"started": True})
        atomic_json(self.run_root / "run_state.json", {"stage": "round_1_completed", "status": "blocked"})
        result = self.inspect()
        self.assertTrue(result["allowed"], result["reason"])
        self.assertEqual(result["resume_receipt"]["recovery_scope"], "committed_phase_boundary")
        self.assertEqual(len(result["progress_receipts"]), 1)
        target.write_text("changed")
        self.assertFalse(self.inspect()["allowed"])

    def test_live_worker_or_unknown_host_blocks_even_with_ready_artifacts(self):
        self.failure()
        atomic_json(self.evaluation / "lifecycle.json", process_identity(os.getpid()))
        self.assertFalse(self.inspect()["allowed"])
        atomic_json(self.evaluation / "lifecycle.json", {"pid": 42, "host": "different-allocation"})
        self.assertFalse(self.inspect()["allowed"])

    def test_final_pre_reset_stays_sealed_and_never_exposes_results(self):
        final_root = self.failure(self.run_root / "evaluations/candidate_final_confirmation")
        atomic_json(self.run_root / "run_state.json", {"stage": "final_confirmation", "status": "blocked",
                                                       "final_confirmation_opened": True})
        result = self.inspect()
        self.assertTrue(result["allowed"], result["reason"])
        self.assertTrue(result["resume_receipt"]["final_opened"])
        (final_root / "initializations.jsonl").write_text('{"episode_seed": 123456789, "success": true}\n')
        refused = self.inspect()
        self.assertFalse(refused["allowed"])
        self.assertNotIn("123456789", str(refused))
        self.assertNotIn("success", str(refused))

    def test_final_opened_cannot_return_to_prior_research_stage(self):
        self.failure()
        atomic_json(self.run_root / "run_state.json", {"stage": "official_development", "status": "blocked",
                                                       "final_confirmation_opened": True})
        self.assertFalse(self.inspect()["allowed"])

    def test_probe_reuses_original_attempt_limit_and_wall_clock(self):
        root = self.probe()
        atomic_json(self.run_root / "run_state.json", {"stage": "device_probe_finished", "status": "blocked"})
        result = self.inspect()
        self.assertTrue(result["allowed"], result["reason"])
        self.assertEqual(result["probe_limits"][0]["next_attempt"], 2)
        self.assertFalse(result["probe_limits"][0]["reset_budget_permitted"])
        self.probe(attempts=3)
        self.assertIn("attempt budget exhausted", self.inspect()["reason"])

    def test_missing_probe_generation_commit_or_cleanup_cannot_reset_history(self):
        root = self.probe()
        atomic_json(self.run_root / "run_state.json", {"stage": "device_probe_finished", "status": "blocked"})
        (root / "attempts/attempt_1/attempt.json").unlink()
        self.assertFalse(self.inspect()["allowed"])

    def test_passed_initial_probe_does_not_hide_a_failed_repetition_reset(self):
        base = self.probe(passed=True)
        repeat = self.probe(root=base / "repetitions/probe_2")
        atomic_json(repeat / "attempts/attempt_1/schedule/native_probe/native_0/native/reset_started.json", {"started": True})
        atomic_json(self.run_root / "run_state.json", {"stage": "device_probe_finished", "status": "blocked"})
        self.assertFalse(self.inspect()["allowed"])

    def test_expired_budget_and_changed_science_are_refused(self):
        self.failure()
        self.assertFalse(self.inspect(deadline_epoch=time.time() - 1)["allowed"])
        self.assertFalse(self.inspect(scientific_contract={"task": "different"})["allowed"])

    def test_runtime_attempt_budget_cannot_be_reset_by_outer_supervisor(self):
        self.failure(self.evaluation / "startup_attempt_3")
        self.assertIn("startup attempt budget exhausted", self.inspect()["reason"])

    def test_lifecycle_compatibility_is_bound_and_reaping_is_host_attested(self):
        self.failure()
        receipt = {"passed": True, "scope": "lifecycle_only", "from_revision": "r1", "to_revision": "r2",
                   "scientific_contract_sha256": object_digest(SCIENCE), "validation_receipt_sha256": "host-receipt"}
        result = self.inspect(execution_revision="r2", compatibility=receipt)
        self.assertTrue(result["allowed"], result["reason"])
        self.assertTrue(result["compatibility"]["previous_workers_reaped"])

    def test_real_process_failure_safe_boundary_then_new_process_completes(self):
        previous_core_limit = resource.getrlimit(resource.RLIMIT_CORE)
        resource.setrlimit(resource.RLIMIT_CORE, (0, previous_core_limit[1]))
        self.addCleanup(resource.setrlimit, resource.RLIMIT_CORE, previous_core_limit)
        script = self.root / "worker.py"
        script.write_text(
            "import os, pathlib, resource, signal\n"
            "root=pathlib.Path('runs/fixture')\n"
            "marker=root/'first_attempt'\n"
            "if not marker.exists():\n"
            "    marker.parent.mkdir(parents=True,exist_ok=True)\n"
            "    marker.write_text(str(os.getpid()))\n"
            "    resource.setrlimit(resource.RLIMIT_CORE,(0,0))\n"
            "    os.kill(os.getpid(),signal.SIGSEGV)\n"
            "(root/'completed').write_text(str(os.getpid()))\n")
        supervisor = ContinuationSupervisor(self.root / "continuation", deadline_epoch=self.deadline)
        validation = {"passed": True, "execution_revision": "r1", "scientific_contract_sha256": object_digest(SCIENCE),
                      "source_files": {"worker.py": digest(script)}}
        options = dict(cwd=self.root, execution_revision="r1", scientific_contract=SCIENCE,
                       stage="official_development", allocation_reconciled=True,
                       validation_receipt=validation, cleanup_seconds=.5)
        first = supervisor.run_once([sys.executable, str(script)], **options)
        self.assertEqual(first["returncode"], -signal.SIGSEGV)
        self.failure()
        atomic_json(self.evaluation / "process/process.json", {"status": "failed", "returncode": first["returncode"],
            "pid": first["worker"]["pid"], "host": first["worker"]["host"]})
        boundary = self.inspect(previous_attempt=first)
        self.assertTrue(boundary["allowed"], boundary["reason"])
        second = supervisor.run_once([sys.executable, str(script)], **options,
            resume_receipt=boundary["resume_receipt"], workers=boundary["workers"])
        self.assertEqual(second["status"], "completed")
        self.assertNotEqual((self.run_root / "first_attempt").read_text(), (self.run_root / "completed").read_text())
        self.assertEqual(read_json(supervisor.path)["deadline_epoch"], self.deadline)


if __name__ == "__main__":
    unittest.main()
