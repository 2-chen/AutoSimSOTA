import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

from autosim.research.artifact_commits import ArtifactConflict, ArtifactStore
from autosim.research.common import atomic_json, digest, object_digest, read_json
from autosim.research.continuation import ContinuationRefused, ContinuationSupervisor, process_gone, process_identity
from autosim.research.environment_contract import collect_environment_contract, inspect_act_checkpoint, interpreter_contract


SCIENCE = {"task": "fixture", "seeds": [1, 2], "normalization": "inline_v1"}


class TemporaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)


class ArtifactTests(TemporaryTest):
    def setUp(self):
        super().setUp()
        self.store = ArtifactStore(self.root / "commits", artifact_root=self.root)
        self.args = dict(inputs={"checkpoint": "weight_hash"}, scientific_contract=SCIENCE,
                         execution_revision="r1")
        self.file = self.root / "metrics.json"
        self.file.write_text('{"episodes":[1,2]}')

    def prepare(self):
        return self.store.prepare("evaluation", **self.args, attempt_id="a1")

    def commit(self):
        self.prepare()
        self.store.transition("evaluation", "running")
        self.store.transition("evaluation", "validating")
        return self.store.commit("evaluation", outputs={"metrics": self.file}, transaction_id="settle1")

    def test_file_existence_and_uncommitted_attempt_do_not_authorize_reuse(self):
        self.assertIsNone(self.store.reuse("evaluation", **self.args))
        self.prepare()
        self.assertIsNone(self.store.reuse("evaluation", **self.args))
        with self.assertRaises(ArtifactConflict):
            self.store.commit("evaluation", outputs={"metrics": self.file}, transaction_id="t")

    def test_commit_and_settlement_identity_are_idempotent(self):
        row = self.commit()
        repeated = self.store.commit("evaluation", outputs={"metrics": self.file}, transaction_id="settle1")
        self.assertEqual(row, repeated)
        self.assertEqual(row, self.store.reuse("evaluation", **self.args))
        with self.assertRaises(ArtifactConflict):
            self.store.commit("evaluation", outputs={"metrics": self.file}, transaction_id="another")

    def test_science_checkpoint_or_output_mutation_refuses_cache(self):
        self.commit()
        for changed in ({"scientific_contract": {**SCIENCE, "seeds": [3]}},
                        {"inputs": {"checkpoint": "other"}}):
            with self.subTest(changed=changed), self.assertRaises(ArtifactConflict):
                self.store.reuse("evaluation", **(self.args | changed))
        self.file.write_text('{"episodes":[1]}')
        with self.assertRaises(ArtifactConflict):
            self.store.reuse("evaluation", **self.args)

    def test_revision_requires_explicit_science_preserving_compatibility(self):
        self.commit()
        args = self.args | {"execution_revision": "r2"}
        with self.assertRaises(ArtifactConflict):
            self.store.reuse("evaluation", **args)
        receipt = {"passed": True, "scope": "lifecycle_only", "from_revision": "r1", "to_revision": "r2",
                   "scientific_contract_sha256": object_digest(SCIENCE), "validation_receipt_sha256": "host-test"}
        self.assertEqual(self.store.reuse("evaluation", **args, compatibility=receipt)["execution_revision"], "r1")
        with self.assertRaises(ArtifactConflict):
            self.store.reuse("evaluation", **args, compatibility=receipt | {"scope": "model"})

    def test_partial_output_and_symlink_escape_are_rejected(self):
        self.prepare()
        self.store.transition("evaluation", "running")
        self.store.transition("evaluation", "validating")
        self.file.unlink()
        with self.assertRaises(ArtifactConflict):
            self.store.commit("evaluation", outputs={"metrics": self.file}, transaction_id="t")
        self.file.symlink_to(Path(__file__).resolve())
        with self.assertRaises(ArtifactConflict):
            self.store.commit("evaluation", outputs={"metrics": self.file}, transaction_id="t")

    def test_uncommitted_revision_replacement_preserves_prior_attempt_and_requires_reaping(self):
        self.prepare()
        self.store.transition("evaluation", "running")
        receipt = {"passed": True, "scope": "lifecycle_only", "from_revision": "r1", "to_revision": "r2",
                   "scientific_contract_sha256": object_digest(SCIENCE), "validation_receipt_sha256": "host"}
        with self.assertRaises(ArtifactConflict):
            self.store.restart_uncommitted("evaluation", execution_revision="r2", attempt_id="a2",
                                           compatibility=receipt, previous_terminated=False)
        changed = self.store.restart_uncommitted("evaluation", execution_revision="r2", attempt_id="a2",
                                                compatibility=receipt, previous_terminated=True)
        self.assertEqual(changed["state"], "prepared")
        self.assertEqual(changed["prior_attempts"][0]["attempt_id"], "a1")


class EnvironmentTests(TemporaryTest):
    def checkpoint(self, *, std=0.0, unused=True):
        config = {"type": "act", "input_features": {"observation.state": {"type": "STATE", "shape": [2]}},
                  "output_features": {"action": {"type": "ACTION", "shape": [2]}},
                  "normalization_mapping": {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"}}
        if unused:
            config["input_features"]["observation.qvel"] = {"type": "STATE", "shape": [2]}
            config["input_features"]["observation.qf"] = {"type": "STATE", "shape": [2]}
        header, payload = {}, b""
        for prefix, features in (("normalize_inputs", config["input_features"]),
                                 ("normalize_targets", config["output_features"]),
                                 ("unnormalize_outputs", config["output_features"])):
            for feature in features:
                for statistic in ("mean", "std"):
                    name = f"{prefix}.buffer_{feature.replace('.', '_')}.{statistic}"
                    raw = struct.pack("<ff", *(2 * [std if statistic == "std" else 0.0]))
                    header[name] = {"dtype": "F32", "shape": [2], "data_offsets": [len(payload), len(payload) + len(raw)]}
                    payload += raw
        atomic_json(self.root / "config.json", config)
        raw_header = json.dumps(header).encode()
        (self.root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
        return self.root

    def test_zero_std_and_unused_recorded_modalities_are_legal(self):
        result = inspect_act_checkpoint(self.checkpoint())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["checked_statistics"]), 10)
        self.assertFalse(result["prediction_parity_verified"])
        self.assertFalse(result["native_runtime_verified"])

    def test_nonfinite_and_negative_stats_are_not_admitted(self):
        for value in (float("nan"), float("inf"), -1.0):
            with self.subTest(value=value):
                result = inspect_act_checkpoint(self.checkpoint(std=value))
                self.assertEqual(result["status"], "failed")

    def test_trusted_caller_hashes_avoid_rehash_but_never_skip_structure(self):
        self.checkpoint()
        known = {"config_sha256": digest(self.root / "config.json"),
                 "checkpoint_sha256": digest(self.root / "model.safetensors")}
        with mock.patch("autosim.research.environment_contract.digest", side_effect=AssertionError("unexpected rehash")):
            result = inspect_act_checkpoint(self.root, verified_file_hashes=known)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["checkpoint_sha256"], known["checkpoint_sha256"])
            (self.root / "model.safetensors").write_bytes(b"invalid")
            self.assertEqual(inspect_act_checkpoint(self.root, verified_file_hashes=known)["status"], "failed")

    def test_truncated_or_oversized_checkpoint_is_rejected(self):
        self.checkpoint()
        for raw in (b"short", struct.pack("<Q", 10**12)):
            (self.root / "model.safetensors").write_bytes(raw)
            self.assertEqual(inspect_act_checkpoint(self.root)["status"], "failed")

    def test_interpreter_metadata_probe_uses_actual_interpreter_without_gpu_import(self):
        result = interpreter_contract(Path(sys.executable))
        self.assertEqual(result["status"], "passed")
        self.assertIn("torch", result["versions"])
        self.assertEqual(interpreter_contract(self.root / "missing")["status"], "failed")

    def test_inventory_checks_named_assets_without_walking_venv(self):
        code = self.root / "code.py"
        code.write_text("pass\n")
        environment = self.root / ".venv"
        environment.mkdir()
        (environment / "escape").symlink_to("/does-not-exist")
        result = collect_environment_contract(self.root, interpreters={"simulation": Path(sys.executable)},
            assets={"checkpoint": {"path": str(code), "sha256": digest(code)}, "training_env": {"path": str(environment)}},
            source_files=[code])
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["native_runtime_verified"])
        self.assertEqual(len(result["source_files"]), 1)
        bad = collect_environment_contract(self.root, interpreters={"simulation": Path(sys.executable)},
            assets={"checkpoint": {"path": str(code), "sha256": "wrong"}})
        self.assertEqual(bad["status"], "failed")
        with self.assertRaises(ValueError):
            collect_environment_contract(self.root, interpreters={}, source_files=[environment])


class ContinuationTests(TemporaryTest):
    def setUp(self):
        super().setUp()
        self.script = self.root / "worker.py"
        self.script.write_text("import os, pathlib\npathlib.Path('child_pid').write_text(str(os.getpid()))\nraise SystemExit(1)\n")
        self.supervisor = ContinuationSupervisor(self.root / "host", deadline_epoch=time.time() + 60,
                                                 max_restarts=4, max_no_progress=3)

    def validation(self, revision="r1"):
        return {"passed": True, "execution_revision": revision,
                "scientific_contract_sha256": object_digest(SCIENCE),
                "source_files": {"worker.py": digest(self.script)}}

    def resume(self, stage="official_development"):
        return {"safe_boundary": True, "stage": stage, "episode_started": False,
                "scientific_contract_sha256": object_digest(SCIENCE)}

    def run_child(self, **kwargs):
        options = dict(cwd=self.root, execution_revision="r1", scientific_contract=SCIENCE,
                       stage="official_development", allocation_reconciled=True,
                       cleanup_seconds=.5, validation_receipt=self.validation())
        return self.supervisor.run_once([sys.executable, str(self.script)], **(options | kwargs))

    def test_two_attempts_execute_distinct_real_processes_with_persistent_limits(self):
        first = self.run_child()
        first_pid = (self.root / "child_pid").read_text()
        second = self.run_child(resume_receipt=self.resume())
        self.assertEqual(first["status"], "failed")
        self.assertEqual(second["status"], "failed")
        self.assertNotEqual(first_pid, (self.root / "child_pid").read_text())
        original = read_json(self.supervisor.path)
        ContinuationSupervisor(self.supervisor.root, deadline_epoch=time.time() + 10000,
                               max_restarts=99, max_no_progress=99)
        restored = read_json(self.supervisor.path)
        self.assertEqual(restored["deadline_epoch"], original["deadline_epoch"])
        self.assertEqual(restored["max_restarts"], 4)
        self.assertEqual(restored["no_progress_count"], 2)

    def test_no_progress_and_unknown_phase_refuse_continuation(self):
        self.run_child()
        with self.assertRaises(ContinuationRefused):
            self.run_child()
        self.run_child(resume_receipt=self.resume())
        self.run_child(resume_receipt=self.resume())
        with self.assertRaisesRegex(ContinuationRefused, "no new committed"):
            self.run_child(resume_receipt=self.resume())

    def test_live_or_remote_worker_and_unreconciled_allocation_are_refused(self):
        for opts in ({"workers": [process_identity(os.getpid())]},
                     {"workers": [{"pid": 123, "host": "different-host"}]},
                     {"allocation_reconciled": False}):
            with self.subTest(opts=opts), self.assertRaises(ContinuationRefused):
                self.run_child(**opts)
        self.assertFalse(process_gone({"pid": 123, "host": "different-host"}))

    def test_changed_science_or_unvalidated_revision_is_refused(self):
        self.run_child()
        with self.assertRaises(ContinuationRefused):
            self.run_child(scientific_contract={**SCIENCE, "seeds": [9]}, resume_receipt=self.resume())
        with self.assertRaises(ContinuationRefused):
            self.run_child(execution_revision="r2", resume_receipt=self.resume(), validation_receipt=None)

    def test_activation_and_rollback_require_bound_receipts_and_fresh_process(self):
        self.run_child()
        compat = {"passed": True, "scope": "lifecycle_only", "from_revision": "r1", "to_revision": "r2",
                  "scientific_contract_sha256": object_digest(SCIENCE), "validation_receipt_sha256": "trusted-test"}
        second = self.run_child(execution_revision="r2", validation_receipt=self.validation("r2"),
                                compatibility=compat, resume_receipt=self.resume())
        self.assertEqual(second["operation"], "activate")
        third = self.run_child(rollback=True, resume_receipt=self.resume(),
                              compatibility=compat | {"from_revision": "r2", "to_revision": "r1"})
        self.assertEqual(third["operation"], "rollback")
        self.assertEqual(read_json(self.supervisor.path)["active_revision"], "r1")

    def test_final_latch_prevents_research_replay_even_after_host_restart(self):
        state = self.root / "run_state.json"
        self.script.write_text("import json, pathlib\npathlib.Path('run_state.json').write_text(json.dumps({'final_confirmation_opened': True}))\nraise SystemExit(1)\n")
        self.run_child(run_state_path=state)
        with self.assertRaisesRegex(ContinuationRefused, "final opened"):
            self.run_child(run_state_path=state, resume_receipt=self.resume())
        with self.assertRaisesRegex(ContinuationRefused, "pre-reset"):
            self.run_child(stage="final_confirmation", resume_receipt=self.resume("final_confirmation") | {"episode_started": True})

    def test_deadline_and_shared_budget_are_enforced_before_launch(self):
        with self.assertRaisesRegex(ContinuationRefused, "shared"):
            self.run_child(budget_check=lambda seconds: False)
        ContinuationSupervisor(self.supervisor.root, deadline_epoch=time.time() - 1)
        with self.assertRaisesRegex(ContinuationRefused, "budget exhausted"):
            self.run_child()
        self.assertFalse((self.root / "child_pid").exists())

    def test_absolute_deadline_terminates_real_child_without_renewing_budget(self):
        self.script.write_text("import time\ntime.sleep(30)\n")
        ContinuationSupervisor(self.supervisor.root, deadline_epoch=time.time() + 2)
        started = time.monotonic()
        attempt = self.run_child()
        self.assertEqual(attempt["status"], "budget_exhausted")
        self.assertLess(time.monotonic() - started, 4)
        self.assertTrue(process_gone(attempt["process"]))
        self.assertTrue(process_gone(attempt["worker"]))
        with self.assertRaises(ContinuationRefused):
            self.run_child(resume_receipt=self.resume())

    def test_unresolved_launch_is_not_resubmitted_without_termination_evidence(self):
        state = read_json(self.supervisor.path)
        state["attempts"].append({"attempt": 1, "status": "intent", "stage": "official_development"})
        atomic_json(self.supervisor.path, state)
        with self.assertRaisesRegex(ContinuationRefused, "outcome unresolved"):
            self.run_child(resume_receipt=self.resume())
        with self.assertRaises(ContinuationRefused):
            self.supervisor.reconcile_launch(workers=[], termination_evidence={})
        self.supervisor.reconcile_launch(workers=[], termination_evidence={"verified": True, "evidence_sha256": "platform"})
        self.assertEqual(self.run_child(resume_receipt=self.resume())["attempt"], 2)

    def test_only_new_committed_artifacts_count_as_progress(self):
        output = self.root / "metrics.json"
        store = ArtifactStore(self.root / "commits", artifact_root=self.root)
        store.prepare("phase", inputs={}, scientific_contract=SCIENCE, execution_revision="r1", attempt_id="a")
        store.transition("phase", "running")
        store.transition("phase", "validating")
        commit_path = store._path("phase")
        self.script.write_text(
            "from pathlib import Path\nfrom autosim.research.artifact_commits import ArtifactStore\n"
            "Path('metrics.json').write_text('complete')\n"
            "ArtifactStore(Path('commits'), artifact_root=Path('.')).commit('phase', outputs={'metrics':Path('metrics.json')},transaction_id='t')\n"
            "raise SystemExit(1)\n")
        # The receipt exists but is not committed before the child: do not pass it
        # as already completed progress; publish a distinct commit after validation.
        published = self.root / "published.json"
        self.script.write_text(self.script.read_text().replace("raise SystemExit(1)",
            f"import shutil\nshutil.copyfile({str(commit_path)!r}, 'published.json')\nraise SystemExit(1)"))
        result = self.run_child(progress_receipts=[published])
        self.assertEqual(len(result["new_progress_commits"]), 1)
        self.assertEqual(read_json(self.supervisor.path)["no_progress_count"], 0)


if __name__ == "__main__":
    unittest.main()
