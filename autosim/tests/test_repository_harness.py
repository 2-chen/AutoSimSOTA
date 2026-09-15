import copy
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

from autosim.robosyn_data import evaluation_seed_bank
from autosim.research.artifact_commits import ArtifactConflict
from autosim.research.common import atomic_json, digest, object_digest, read_json
from autosim.research.evaluation_merge import recompute_summary
from autosim.research.environment_contract import inspect_act_checkpoint
from autosim.research.repository_harness import HarnessArtifacts, scientific_identity


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.helper = self.root / "policy/act/checkpoint_compat.py"
        self.helper.parent.mkdir(parents=True)
        self.helper.write_text("# trusted ACT compatibility fixture\n")
        self.protocol = {"task": "fixture", "budget": {"development_episodes": 2, "screen_steps": 200},
                         "act_checkpoint_contract": {"normalization": "checkpoint_inline_v1",
                                                     "implementation_sha256": digest(self.helper)},
                         "continuation_scientific_contract": {"recipe": "fixed"}}
        self.runtime = SimpleNamespace(_execution_code_digest=lambda: "r1", repo=self.root)
        self.harness = HarnessArtifacts(self.root, self.runtime, self.protocol)
        self.checkpoint = self.root / "official"
        self.checkpoint.mkdir()
        (self.checkpoint / "model.safetensors").write_bytes(b"fixture-weights")
        atomic_json(self.checkpoint / "config.json", {"shape": [14]})
        self.code = self.root / "evaluator.py"
        self.code.write_text("# trusted fixture code\n")
        self.path = self.root / "evaluations/baseline_development"
        self.master, self.count, self.purpose = 12, 2, "development"
        # Business-proof fixtures deliberately do not contain actual tensor
        # files. Mock only the strong validator, never weaken the production gate.
        def fixture_contract(checkpoint, *, compatibility_file, verified_file_hashes):
            self.assertEqual(compatibility_file, self.helper)
            return {"status": "passed", "scope": "cpu_checkpoint_normalization_structure",
                    "normalization": "checkpoint_inline_v1", **verified_file_hashes}
        patcher = mock.patch("autosim.research.repository_harness.inspect_act_checkpoint",
                             side_effect=fixture_contract)
        self.validator = patcher.start()
        self.addCleanup(patcher.stop)

    def args(self, checkpoint=None, path=None):
        return (checkpoint or self.checkpoint, path or self.path, self.purpose, self.master, self.count)

    def write_native(self, checkpoint=None, path=None):
        checkpoint, path = checkpoint or self.checkpoint, path or self.path
        rows = [{"episode_index": i, "episode_seed": seed, "success": i == 0,
                 "action_steps": 10, "inference_call_count": 2,
                 "total_inference_time_seconds": .02, "average_inference_time_seconds": .01}
                for i, seed in enumerate(evaluation_seed_bank(self.master, self.count))]
        data = {"execution_mode": "real_simulation", "purpose": "development", "harness": "trusted_native",
                "config": {"task": "fixture", "seed": self.master, "episode_count": self.count,
                           "timeout_action_steps": 10, "ranking_eligible": True},
                "episodes": rows, "summary": recompute_summary(rows, timeout_action_steps=10),
                "artifact_directory": str(path)}
        atomic_json(path / "evaluation_metrics.json", data)
        atomic_json(path / "evaluation_request.json", {"weight_sha256": digest(checkpoint / "model.safetensors"),
            "model_config_sha256": digest(checkpoint / "config.json"), "episodes": self.count,
            "master_seed": self.master, "purpose": "development"})
        atomic_json(path / "protocol.json", {"task": {"name": "fixture", "max_episode_steps": 10},
            "checkpoint_sha256": digest(checkpoint / "model.safetensors"), "seed": self.master,
            "episodes": self.count, "purpose": "development", "frozen_files": {str(self.code): digest(self.code)}})
        atomic_json(path / "process/process.json", {"status": "completed", "returncode": 0})
        return data

    def committed_eval(self):
        self.assertIsNone(self.harness.begin_evaluation(*self.args()))
        result = self.write_native()
        self.harness.commit_evaluation(*self.args(), result)
        return result

    def test_business_validated_result_can_be_reused_after_supervisor_restart(self):
        expected = self.committed_eval()
        restored = HarnessArtifacts(self.root, self.runtime, self.protocol)
        self.assertEqual(restored.begin_evaluation(*self.args()), expected)
        self.assertEqual(self.validator.call_count, 1)

    def test_every_checkpoint_origin_is_checked_before_native_evaluation(self):
        for name in ("official", "official_data_continuation", "candidate"):
            with self.subTest(origin=name):
                checkpoint = self.root / name
                checkpoint.mkdir(exist_ok=True)
                (checkpoint / "model.safetensors").write_bytes(name.encode())
                atomic_json(checkpoint / "config.json", {"shape": [14]})
                path = self.root / "evaluations" / name
                self.assertIsNone(self.harness.begin_evaluation(*self.args(checkpoint, path)))
                contract = read_json(path / "checkpoint_contract.json")
                self.assertEqual(contract["status"], "passed")
                self.assertEqual(contract["checkpoint_sha256"], digest(checkpoint / "model.safetensors"))
                self.assertFalse((path / "process/process.json").exists())
        self.assertEqual(self.validator.call_count, 3)

    def test_production_gate_rejects_fake_tensors_and_persists_failure(self):
        self.validator.side_effect = inspect_act_checkpoint
        with self.assertRaisesRegex(ArtifactConflict, "checkpoint contract failed"):
            self.harness.begin_evaluation(*self.args())
        self.assertEqual(read_json(self.path / "checkpoint_contract.json")["status"], "failed")
        self.assertFalse((self.path / "process/process.json").exists())

    def test_checkpoint_contract_is_required_and_committed(self):
        self.committed_eval()
        contract = self.path / "checkpoint_contract.json"
        contract.unlink()
        with self.assertRaises(ArtifactConflict):
            self.harness.begin_evaluation(*self.args())

    def test_changed_helper_is_rejected_even_before_first_native_launch(self):
        self.helper.write_text("# changed compatibility implementation\n")
        with self.assertRaisesRegex(ArtifactConflict, "frozen protocol"):
            self.harness.begin_evaluation(*self.args())
        self.validator.assert_not_called()

    def test_valid_metrics_after_crash_are_committed_only_with_prior_matching_intent(self):
        self.harness.begin_evaluation(*self.args())
        expected = self.write_native()
        restored = HarnessArtifacts(self.root, self.runtime, self.protocol)
        self.assertEqual(restored.begin_evaluation(*self.args()), expected)
        path = restored.store._path(restored._key(self.path))
        self.assertEqual(read_json(path)["state"], "committed")

    def test_historical_metrics_without_intent_are_not_auto_admitted(self):
        self.write_native()
        with self.assertRaisesRegex(ArtifactConflict, "historical"):
            self.harness.begin_evaluation(*self.args())

    def test_reset_before_incomplete_output_cannot_be_replayed(self):
        self.harness.begin_evaluation(*self.args())
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "initializations.jsonl").write_text('{"seed":1}\n')
        with self.assertRaisesRegex(ArtifactConflict, "started reset"):
            self.harness.begin_evaluation(*self.args())

    def test_same_count_wrong_seed_or_duplicate_seed_is_rejected(self):
        for kind in ("wrong", "duplicate"):
            path = self.path.parent / kind
            args = self.args(path=path)
            self.harness.begin_evaluation(*args)
            data = self.write_native(path=path)
            data["episodes"][1]["episode_seed"] = 42 if kind == "wrong" else data["episodes"][0]["episode_seed"]
            atomic_json(path / "evaluation_metrics.json", data)
            with self.subTest(kind=kind), self.assertRaises(ArtifactConflict):
                self.harness.commit_evaluation(*args, data)

    def test_process_success_and_complete_summary_are_both_required(self):
        self.harness.begin_evaluation(*self.args())
        data = self.write_native()
        data["summary"]["success_count"] = 2
        atomic_json(self.path / "evaluation_metrics.json", data)
        with self.assertRaisesRegex(ArtifactConflict, "summary"):
            self.harness.commit_evaluation(*self.args(), data)
        data = self.write_native()
        atomic_json(self.path / "process/process.json", {"status": "failed", "returncode": -11})
        with self.assertRaisesRegex(ArtifactConflict, "process"):
            self.harness.commit_evaluation(*self.args(), data)

    def test_native_frozen_files_and_checkpoint_are_verified(self):
        self.committed_eval()
        self.code.write_text("# changed judge\n")
        with self.assertRaisesRegex(RuntimeError, "frozen protocol"):
            self.harness.begin_evaluation(*self.args())
        (self.checkpoint / "model.safetensors").write_bytes(b"new weights")
        with self.assertRaises(ArtifactConflict):
            self.harness.begin_evaluation(*self.args())

    def test_full_protocol_science_cannot_be_weakened_by_shared_launcher_identity(self):
        self.committed_eval()
        changed = copy.deepcopy(self.protocol)
        changed["budget"]["screen_steps"] += 1
        self.assertEqual(scientific_identity(changed), scientific_identity(self.protocol))
        restored = HarnessArtifacts(self.root, self.runtime, changed)
        with self.assertRaises(ArtifactConflict):
            restored.begin_evaluation(*self.args())

    def round_fixture(self):
        directory = self.root / "rounds/round_1"
        context = {"current_policy_checkpoint": str(self.checkpoint), "development_evidence_id": "evidence"}
        self.harness.begin_round(directory, 1, context, [])
        candidate = directory / "candidate/checkpoint"
        candidate.mkdir(parents=True)
        (candidate / "model.safetensors").write_bytes(b"candidate")
        atomic_json(candidate / "config.json", {"shape": [14]})
        proposal = {"proposal_id": "p1", "decision": "experiment"}
        atomic_json(directory / "proposal.json", {"proposal": proposal, "validation": "passed",
                                                  "provider": {"kind": "fixture"}})
        atomic_json(directory / "candidate/checkpoint_contract.json", {"status": "passed",
            "checkpoint_sha256": digest(candidate / "model.safetensors"),
            "config_sha256": digest(candidate / "config.json"),
            "compatibility_sha256": digest(self.helper)})
        atomic_json(directory / "data_admission.json", {"admitted": ["source"]})
        atomic_json(directory / "mixture.json", {"parts": ["source"]})
        atomic_json(directory / "candidate/training_exposure_audit.json", {"parts": [
            {"source_kind": "research_requested_collection", "yielded_samples": 12}]})
        analysis = {"summary": "fixture"}
        atomic_json(directory / "candidate_failure_analysis.json", analysis)
        dataset = directory / "collection/source"
        atomic_json(dataset / "meta/info.json", {"total_episodes": 1, "total_frames": 10})
        evaluation = self.root / "evaluations/round_1_candidate_development"
        self.harness.begin_evaluation(*self.args(candidate, evaluation))
        metrics = self.write_native(candidate, evaluation)
        self.harness.commit_evaluation(*self.args(candidate, evaluation), metrics)
        result = {"round": 1, "status": "completed", "proposal": proposal, "proposal_id": "p1",
                  "checkpoint": str(candidate), "checkpoint_sha256": digest(candidate / "model.safetensors"),
                  "mixture": str(directory / "mixture.json"),
                  "training_exposure_audit": str(directory / "candidate/training_exposure_audit.json"),
                  "candidate_failure_analysis": str(directory / "candidate_failure_analysis.json"),
                  "candidate_evidence_id": object_digest(analysis), "development_evaluation": str(evaluation),
                  "development_summary": metrics["summary"],
                  "cumulative_data": [{"profile": "targeted", "root": str(dataset)}]}
        atomic_json(directory / "round_result.json", result)
        return directory, context, result

    def test_round_cache_binds_data_metadata_mixture_exposure_checkpoint_and_proposal(self):
        directory, context, result = self.round_fixture()
        self.harness.commit_round(directory / "round_result.json", result)
        self.assertEqual(self.harness.begin_round(directory, 1, context, []), result)
        data_path = Path(result["cumulative_data"][0]["root"]) / "meta/info.json"
        atomic_json(data_path, {"total_episodes": 99})
        with self.assertRaises(ArtifactConflict):
            self.harness.begin_round(directory, 1, context, [])

    def test_round_write_before_commit_can_be_reconciled_after_restart(self):
        directory, context, result = self.round_fixture()
        restored = HarnessArtifacts(self.root, self.runtime, self.protocol)
        self.assertEqual(restored.begin_round(directory, 1, context, []), result)

    def test_final_opened_allows_only_committed_round_reconstruction(self):
        directory, context, result = self.round_fixture()
        self.harness.commit_round(directory / "round_result.json", result)
        atomic_json(self.root / "run_state.json", {"final_confirmation_opened": True})
        self.assertEqual(self.harness.begin_round(directory, 1, context, []), result)
        with self.assertRaisesRegex(ArtifactConflict, "final opened"):
            self.harness.begin_round(self.root / "rounds/round_2", 2, context, [])

    def test_round_with_zero_actual_training_exposure_is_not_committed(self):
        directory, context, result = self.round_fixture()
        atomic_json(Path(result["training_exposure_audit"]), {"parts": []})
        with self.assertRaisesRegex(ArtifactConflict, "training batches"):
            self.harness.commit_round(directory / "round_result.json", result)


if __name__ == "__main__":
    unittest.main()
