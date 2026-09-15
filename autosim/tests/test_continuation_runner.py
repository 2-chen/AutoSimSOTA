"""Exercise the product outer loop with real guarded children, no API or GPU."""
import copy
import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from autosim.research.common import atomic_json, digest, object_digest, read_json
from autosim.research.continuation_runner import run_bounded_research
from autosim.research.resume_boundary import inspect_resume_boundary


SCIENCE = {"task": "trusted_fixture", "recipe": "fixed"}
BASE, CANDIDATE = "a" * 64, "b" * 64
CHILD = r'''
import os, resource, signal, sys
from pathlib import Path
from autosim.research.artifact_commits import ArtifactStore
from autosim.research.common import atomic_json, digest, read_json
from autosim.research.resume_boundary import load_resume_receipt
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
root = Path(sys.argv[1])
config = read_json(root / 'fixture.json')
science = config['science']
n = int(os.environ['AUTOSIM_CONTINUATION_ATTEMPT'])
revision = os.environ['TEST_EXECUTION_REVISION']
entry = {'attempt': n, 'pid': os.getpid(), 'revision': revision,
         'argv': sys.argv, 'resume': os.environ.get('AUTOSIM_RESUME_RECEIPT')}
if n > 1:
    receipt = Path(os.environ['AUTOSIM_RESUME_RECEIPT'])
    envelope = load_resume_receipt(receipt,
        expected_sha256=os.environ['AUTOSIM_RESUME_RECEIPT_SHA256'],
        run_root=root, scientific_contract=science, execution_revision=revision)
    entry.update(resume_sha256=digest(receipt), stage=envelope['resume_receipt']['stage'])
    compatibility = os.environ.get('AUTOSIM_EXECUTION_COMPATIBILITY')
    if compatibility:
        entry['compatibility'] = read_json(Path(compatibility))
        assert digest(Path(compatibility)) == os.environ['AUTOSIM_EXECUTION_COMPATIBILITY_SHA256']
else:
    assert not entry['resume'], 'inherited stale resume authority'
    assert 'AUTOSIM_EXECUTION_COMPATIBILITY' not in os.environ
atomic_json(root / 'invocations' / f'{n}.json', entry)
mode = config['mode']
failures = config.get('failures', 1)
if n > failures:
    atomic_json(root / 'run_state.json', {'status': 'completed', 'stage': 'complete'})
    sys.exit(0)
atomic_json(root / 'run_state.json', {'status': 'blocked', 'stage': 'official_development',
                                     'final_confirmation_opened': False})
if mode == 'no_boundary':
    sys.exit(7)
native = root / 'evaluations/official_development'
atomic_json(native / 'startup.json', {'phase': 'environment_constructing', 'policy_has_acted': False})
atomic_json(native / 'process/process.json', {'status': 'failed', 'returncode': -signal.SIGSEGV,
                                            'pid': os.getpid(), 'host': os.uname().nodename})
if mode == 'unsafe':
    atomic_json(native / 'reset_started.json', {'started': True})
if mode in {'progress', 'invalid_progress'}:
    key = f'rounds/round_{n}'
    output = root / key / 'result.json'
    atomic_json(output, {'complete': True, 'attempt': n})
    store = ArtifactStore(root / 'harness/artifact_commits', artifact_root=root)
    store.prepare(key, inputs={}, scientific_contract=science, execution_revision=revision, attempt_id=str(n))
    store.transition(key, 'running')
    store.transition(key, 'validating')
    store.commit(key, outputs={'result': output}, transaction_id=key)
    if mode == 'invalid_progress':
        receipt = next((root / 'harness/artifact_commits').glob('*.json'))
        row = read_json(receipt)
        row['commit_sha256'] = 'tampered'
        atomic_json(receipt, row)
if mode == 'repair' and n == 1:
    atomic_json(root / 'repair_activation_request.json', {'candidate': 'fixture',
        'command': ['untrusted-api-command'], 'scientific_contract': {'task': 'changed'}})
os.kill(os.getpid(), signal.SIGSEGV)
'''


class ContinuationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.run_root = self.root / "runs/fixture"
        self.script = self.source / "worker.py"
        self.script.write_text(CHILD)
        self.source_files = {"worker.py": digest(self.script)}
        atomic_json(self.root / "source_manifest.json", self.source_files)
        atomic_json(self.run_root / "protocol.json", {"continuation_scientific_contract": SCIENCE})
        self.deadline = time.time() + 300
        self.manifest = {"run_root": str(self.run_root), "scientific_contract": copy.deepcopy(SCIENCE),
                         "production_code_identity": BASE,
                         "research_command": [sys.executable, str(self.script), str(self.run_root)]}
        self.policy = {"continuation": {"enabled": True, "max_relaunches": 3, "max_no_progress": 2},
                       "lifecycle": {"startup_attempts": 3}}
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith("AUTOSIM_EXECUTION_") or key.startswith("AUTOSIM_RESUME_"):
                self.env.pop(key)
        self.env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        self.manager_patch = patch("autosim.research.continuation_runner.ExecutionActivation")
        self.manager = self.manager_patch.start().return_value
        self.addCleanup(self.manager_patch.stop)
        self.manager.register_baseline.side_effect = lambda: self.plan(BASE)
        self.manager.prepare_activation.side_effect = lambda request, from_revision: self.plan(CANDIDATE, from_revision)
        self.manager.prepare_rollback.side_effect = lambda from_revision: self.plan(BASE, from_revision, rollback=True)
        # Pure-component signing/contract verification is independently tested in
        # test_execution_activation; process supervision and receipts stay real.
        self.validation_patch = patch("autosim.research.continuation_runner.verify_prepared_activation")
        self.validation_patch.start()
        self.addCleanup(self.validation_patch.stop)

    def plan(self, revision, from_revision=None, rollback=False):
        compatibility = ({"passed": True, "scope": "lifecycle_only", "from_revision": from_revision,
                          "to_revision": revision, "scientific_contract_sha256": object_digest(SCIENCE),
                          "validation_receipt_sha256": "trusted-kernel-fixture"} if from_revision else None)
        return {"execution_revision": revision, "env": {"TEST_EXECUTION_REVISION": revision},
                "validation_receipt": {"passed": True, "execution_revision": revision,
                    "scientific_contract_sha256": object_digest(SCIENCE), "source_files": self.source_files},
                "compatibility": compatibility, "rollback": rollback}

    def run_loop(self, mode="failure", failures=1, **kwargs):
        atomic_json(self.run_root / "fixture.json", {"science": SCIENCE, "mode": mode, "failures": failures})
        return run_bounded_research(self.root, manifest=self.manifest, policy=self.policy,
            deadline_epoch=self.deadline, env=self.env, **kwargs)

    def invocations(self):
        return [read_json(path) for path in sorted((self.run_root / "invocations").glob("*.json"))]

    def journal(self):
        return read_json(self.root / "continuation/continuation.json")

    def test_real_first_segv_safe_boundary_then_fresh_process_completes(self):
        result = self.run_loop()
        self.assertEqual((result["status"], result["returncode"]), ("completed", 0), result)
        first, second = self.invocations()
        self.assertNotEqual(first["pid"], second["pid"])
        self.assertEqual(result["attempts"][0]["returncode"], -signal.SIGSEGV)
        self.assertTrue(Path(second["resume"]).is_relative_to(self.root / "continuation/boundaries"))
        self.assertEqual(second["resume_sha256"], digest(Path(second["resume"])))
        self.assertEqual(self.journal()["deadline_epoch"], self.deadline)
        self.assertEqual({attempt["command_sha256"] for attempt in result["attempts"]},
                         {object_digest(self.manifest["research_command"])})

    def test_no_safe_boundary_never_restarts(self):
        result = self.run_loop("no_boundary")
        self.assertEqual(len(self.invocations()), 1)
        self.assertEqual(result["status"], "stopped")
        self.assertIn("no certified", result["reason"])

    def test_reset_already_started_never_restarts(self):
        result = self.run_loop("unsafe")
        self.assertEqual(len(self.invocations()), 1)
        self.assertEqual(result["status"], "stopped")
        self.assertIn("reset", result["reason"])

    def test_budget_check_prevents_initial_child(self):
        result = self.run_loop(budget_check=lambda reserve: False)
        self.assertFalse(self.invocations())
        self.assertEqual(result["status"], "stopped")
        self.assertIn("budget", result["reason"])

    def test_budget_exhausted_after_failure_never_restarts(self):
        result = self.run_loop(budget_check=lambda reserve: reserve < 30)
        self.assertEqual(len(self.invocations()), 1)
        self.assertIn("budget", result["reason"])
        self.manager.prepare_activation.assert_not_called()

    def test_expired_original_deadline_never_starts(self):
        self.deadline = time.time() - 1
        result = self.run_loop()
        self.assertFalse(self.invocations())
        self.assertIn("budget", result["reason"])

    def test_no_progress_limit_stops_before_third_process(self):
        result = self.run_loop(failures=20)
        self.assertEqual(len(self.invocations()), 2)
        self.assertEqual(self.journal()["no_progress_count"], 2)
        self.assertIn("no new committed progress", result["reason"])

    def test_fast_child_commits_are_counted_without_external_heartbeat(self):
        self.policy["continuation"]["max_no_progress"] = 1
        mirror = []
        result = self.run_loop("progress", failures=3, progress_receipts=mirror)
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(len(self.invocations()), 4)
        self.assertEqual(len(mirror), 3)
        self.assertTrue(all(len(attempt["new_progress_commits"]) == 1 for attempt in result["attempts"][:3]))

    def test_invalid_commit_is_terminal_and_not_progress(self):
        result = self.run_loop("invalid_progress")
        self.assertEqual(len(self.invocations()), 1)
        self.assertEqual(result["attempts"][0]["status"], "invalid_progress")
        self.assertEqual(self.journal()["progress_commits"], [])

    def test_restart_cap_stops_even_with_committed_progress(self):
        self.policy["continuation"]["max_relaunches"] = 1
        result = self.run_loop("progress", failures=20)
        self.assertEqual(len(self.invocations()), 2)
        self.assertIn("restart limit", result["reason"])

    def test_disabled_continuation_runs_exactly_once(self):
        self.policy["continuation"]["enabled"] = False
        result = self.run_loop()
        self.assertEqual(len(self.invocations()), 1)
        self.assertEqual(result["status"], "failed")

    def test_stale_parent_resume_environment_is_removed(self):
        self.env.update(AUTOSIM_RESUME_RECEIPT="/wrong-run/receipt.json", AUTOSIM_RESUME_RECEIPT_SHA256="wrong",
                        AUTOSIM_EXECUTION_COMPATIBILITY="/wrong-run/compat.json",
                        AUTOSIM_EXECUTION_COMPATIBILITY_SHA256="wrong")
        result = self.run_loop(failures=0)
        self.assertEqual(result["status"], "completed", result)
        self.assertIsNone(self.invocations()[0]["resume"])

    def test_tampered_persisted_resume_envelope_prevents_second_launch(self):
        def tamper(*args, **kwargs):
            boundary = inspect_resume_boundary(*args, **kwargs)
            if boundary["allowed"]:
                path = Path(boundary["receipt_path"])
                row = read_json(path)
                row["execution_revision"] = "tampered"
                atomic_json(path, row)
            return boundary
        with patch("autosim.research.resume_boundary.inspect_resume_boundary", side_effect=tamper):
            result = self.run_loop()
        self.assertEqual(len(self.invocations()), 1)
        self.assertIn("changed", result["reason"])

    def test_candidate_failure_rolls_back_in_new_process_with_bound_receipts(self):
        self.policy["continuation"]["max_no_progress"] = 3
        result = self.run_loop("repair", failures=2)
        self.assertEqual(result["status"], "completed", result)
        rows = self.invocations()
        self.assertEqual([row["revision"] for row in rows], [BASE, CANDIDATE, BASE])
        self.assertEqual(len({row["pid"] for row in rows}), 3)
        self.assertEqual([attempt["operation"] for attempt in result["attempts"]], ["activate", "activate", "rollback"])
        self.assertEqual(rows[1]["compatibility"]["to_revision"], CANDIDATE)
        self.assertEqual(rows[2]["compatibility"]["to_revision"], BASE)
        self.assertTrue(rows[2]["compatibility"]["previous_workers_reaped"])
        for row in rows[1:]:
            self.assertEqual(read_json(Path(row["resume"]))["execution_revision"], row["revision"])
            self.assertEqual(digest(Path(row["resume"])), row["resume_sha256"])
        self.manager.prepare_activation.assert_called_once()
        self.manager.prepare_rollback.assert_called_once_with(from_revision=CANDIDATE)

    def test_api_request_cannot_replace_frozen_research_command(self):
        frozen = list(self.manifest["research_command"])
        def prepare(request, from_revision):
            self.assertEqual(request["command"], ["untrusted-api-command"])
            self.manifest["research_command"][:] = request["command"]
            self.manifest["scientific_contract"] = request["scientific_contract"]
            return self.plan(CANDIDATE, from_revision)
        self.manager.prepare_activation.side_effect = prepare
        result = self.run_loop("repair")
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(len(self.invocations()), 2)
        self.assertEqual({attempt["command_sha256"] for attempt in result["attempts"]}, {object_digest(frozen)})
        self.assertEqual(self.journal()["scientific_contract_sha256"], object_digest(SCIENCE))

    def test_failed_activation_validation_never_starts_candidate(self):
        self.manager.prepare_activation.side_effect = RuntimeError("candidate contract failed")
        result = self.run_loop("repair")
        self.assertEqual(len(self.invocations()), 1)
        self.assertIn("candidate contract failed", result["reason"])


if __name__ == "__main__":
    unittest.main()
