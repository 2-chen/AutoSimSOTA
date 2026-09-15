"""Trusted repository-workflow hooks for contract-bound evaluation/round commits."""
from __future__ import annotations

import math
from pathlib import Path

from autosim.robosyn_data import evaluation_seed_bank
from .artifact_commits import ArtifactConflict, ArtifactStore
from .common import assert_frozen, atomic_json, digest, object_digest, read_json
from .environment_contract import collect_environment_contract, inspect_act_checkpoint
from .evaluation_merge import _compare_summary, recompute_summary


def _protocol_science(protocol: dict) -> dict:
    operational = {"continuation_scientific_contract", "execution_revision", "devices",
                   "environment_manifest", "harness_execution", "harness_config"}
    result = {k: v for k, v in protocol.items() if k not in operational}
    if "budget" in result:
        result["budget"] = {k: v for k, v in result["budget"].items() if k not in {
            "hours", "output_root", "run_id", "gpu", "gpu_hours", "max_parallel_jobs",
            "gpus", "device_mode", "compute_controller", "compute_codegen"}}
    return result


def scientific_identity(protocol: dict) -> dict:
    """The trusted launcher may supply the same recipe identity before initialization."""
    value = protocol.get("continuation_scientific_contract")
    if value is not None and (not isinstance(value, dict) or not value):
        raise ValueError("continuation science identity must be a nonempty object")
    return value if value is not None else _protocol_science(protocol)


class HarnessArtifacts:
    def __init__(self, run_root: Path, runtime, protocol: dict, *, execution_revision: str | None = None,
                 compatibility: dict | None = None):
        self.root, self.runtime, self.protocol = Path(run_root).absolute(), runtime, protocol
        self.science = scientific_identity(protocol)
        self.protocol_hash = object_digest(_protocol_science(protocol))
        self.revision = execution_revision or runtime._execution_code_digest()
        self.compatibility = compatibility
        self.store = ArtifactStore(self.root / "harness/artifact_commits", artifact_root=self.root)
        self.pending_rounds: dict[str, dict] = {}

    def _within(self, path: Path) -> Path:
        path = Path(path).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ArtifactConflict("workflow artifact escaped its run root")
        return path

    def _key(self, path: Path) -> str:
        return str(self._within(path).relative_to(self.root.resolve()))

    def _request(self, inputs: dict) -> dict:
        return {"inputs": {"full_protocol_science_sha256": self.protocol_hash, **inputs},
                "scientific_contract": self.science, "execution_revision": self.revision}

    def _prepare(self, key: str, request: dict, *, existing_output: bool) -> None:
        path = self.store._path(key)
        if not path.exists() and existing_output:
            raise ArtifactConflict("unregistered historical output requires a new experiment or explicit audit")
        if path.exists():
            row = read_json(path)
            self.store._check(row, request["inputs"], self.science)
            if row["execution_revision"] != self.revision:
                if existing_output or not self.compatibility or not self.compatibility.get("previous_workers_reaped"):
                    raise ArtifactConflict("incomplete attempt belongs to another executor revision")
                row = self.store.restart_uncommitted(key, execution_revision=self.revision,
                    attempt_id=object_digest({"previous": row["attempt_id"], "revision": self.revision})[:24],
                    compatibility=self.compatibility, previous_terminated=True)
            if row["state"] == "failed":
                raise ArtifactConflict("failed attempt cannot be silently replayed")
            attempt = row["attempt_id"]
        else:
            attempt = "initial"
        self.store.prepare(key, **request, attempt_id=attempt)
        if read_json(path)["state"] == "prepared":
            self.store.transition(key, "running")

    def _commit(self, key: str, outputs: dict) -> dict:
        row = read_json(self.store._path(key))
        if row["state"] == "running":
            self.store.transition(key, "validating")
        return self.store.commit(key, outputs=outputs,
                                 transaction_id=object_digest({"key": key, "attempt": row["attempt_id"],
                                                               "inputs": row["inputs_sha256"]}))

    def _evaluation_input(self, checkpoint: Path, path: Path, purpose: str, master: int, count: int) -> dict:
        if count <= 0:
            raise ValueError("evaluation bank must be nonempty")
        checkpoint = Path(checkpoint)
        return self._request({"kind": "evaluation", "purpose": purpose, "master_seed": master,
            "count": count, "bank_sha256": object_digest(evaluation_seed_bank(master, count)),
            "checkpoint_sha256": digest(checkpoint / "model.safetensors"),
            "checkpoint_config_sha256": digest(checkpoint / "config.json"),
            "checkpoint_compatibility_sha256": digest(self._checkpoint_helper())})

    def _checkpoint_helper(self) -> Path:
        return Path(getattr(self.runtime, "eval_repo", self.runtime.repo)) / "policy/act/checkpoint_compat.py"

    def _checkpoint_contract(self, checkpoint: Path, path: Path, inputs: dict, *, create: bool = False) -> Path:
        contract_path = path / "checkpoint_contract.json"
        expected = {"checkpoint_sha256": inputs["checkpoint_sha256"],
                    "config_sha256": inputs["checkpoint_config_sha256"],
                    "compatibility_sha256": inputs["checkpoint_compatibility_sha256"]}
        frozen_helper = self.protocol.get("act_checkpoint_contract", {}).get("implementation_sha256")
        if frozen_helper and expected["compatibility_sha256"] != frozen_helper:
            raise ArtifactConflict("ACT compatibility implementation changed from the frozen protocol")
        if not contract_path.exists() and create:
            atomic_json(contract_path, inspect_act_checkpoint(Path(checkpoint),
                compatibility_file=self._checkpoint_helper(), verified_file_hashes=expected))
        if not contract_path.is_file():
            raise ArtifactConflict("evaluation lacks its pre-native checkpoint contract")
        contract = read_json(contract_path)
        if (contract.get("status") != "passed" or any(contract.get(k) != v for k, v in expected.items())
                or contract.get("scope") != "cpu_checkpoint_normalization_structure"
                or contract.get("normalization") != "checkpoint_inline_v1"):
            raise ArtifactConflict("evaluation checkpoint contract failed or changed")
        return contract_path

    @staticmethod
    def _started(path: Path) -> bool:
        patterns = ("initializations.jsonl", "startup_attempt_*/initializations.jsonl",
                    "shards/shard_*/initializations.jsonl", "shards/shard_*/startup_attempt_*/initializations.jsonl")
        if any(file.stat().st_size > 0 for pattern in patterns for file in path.glob(pattern)):
            return True
        for pattern in ("startup.json", "startup_attempt_*/startup.json", "shards/shard_*/startup.json",
                        "shards/shard_*/startup_attempt_*/startup.json"):
            for file in path.glob(pattern):
                if read_json(file).get("phase") in {"first_reset_started", "first_reset_finished",
                                                     "rollout_started", "episode_running", "evaluation_reset_started"}:
                    return True
        return False

    def _validate_evaluation(self, checkpoint: Path, path: Path, purpose: str, master: int, count: int,
                             supplied: dict | None = None, *, inputs: dict | None = None) -> tuple[dict, dict]:
        inputs = inputs or self._evaluation_input(checkpoint, path, purpose, master, count)["inputs"]
        checkpoint_contract = self._checkpoint_contract(checkpoint, path, inputs)
        metrics = path / "evaluation_metrics.json"
        data = read_json(metrics)
        if supplied is not None and object_digest(data) != object_digest(supplied):
            raise ArtifactConflict("returned evaluation differs from its persisted metrics")
        rows, config = data.get("episodes", []), data.get("config", {})
        if data.get("execution_mode") != "real_simulation" or data.get("purpose") != purpose:
            raise ArtifactConflict("evaluation lacks a real-simulation purpose certificate")
        bank = evaluation_seed_bank(master, count)
        if [row.get("episode_seed") for row in rows] != bank or len(set(bank)) != count:
            raise ArtifactConflict("evaluation does not cover the exact frozen seed bank in order")
        if (config.get("seed") != master or config.get("episode_count") != count
                or config.get("task") != self.protocol["task"] or not data.get("harness")):
            raise ArtifactConflict("evaluation config or harness differs from its request")
        if config.get("ranking_eligible") is False and purpose != "smoke":
            raise ArtifactConflict("diagnostic metrics cannot become a ranked evaluation")
        horizon = config.get("timeout_action_steps")
        if type(horizon) is not int or horizon <= 0:
            raise ArtifactConflict("invalid native episode horizon")
        for row in rows:
            if type(row.get("success")) is not bool or type(row.get("action_steps")) is not int:
                raise ArtifactConflict("invalid episode outcome types")
            if not 0 < row["action_steps"] <= horizon:
                raise ArtifactConflict("episode action count outside the task horizon")
            for name in ("total_inference_time_seconds", "average_inference_time_seconds"):
                if row.get(name) is not None and not (math.isfinite(float(row[name])) and float(row[name]) >= 0):
                    raise ArtifactConflict("nonfinite episode timing")
        differences = _compare_summary(data.get("summary", {}), recompute_summary(rows, timeout_action_steps=horizon),
                                       where="evaluation")
        if differences:
            raise ArtifactConflict("evaluation summary disagrees with the complete episode rows")
        request_path = path / "evaluation_request.json"
        request = read_json(request_path)
        expected = {"weight_sha256": inputs["checkpoint_sha256"],
                    "model_config_sha256": inputs["checkpoint_config_sha256"],
                    "episodes": count, "master_seed": master, "purpose": purpose}
        if any(request.get(k) != value for k, value in expected.items()):
            raise ArtifactConflict("native evaluation request changed")
        actual = self._within(Path(data.get("artifact_directory", path)))
        if not actual.is_relative_to(path.resolve()):
            raise ArtifactConflict("native proof escaped its logical evaluation")
        protocol_path, process_path = actual / "protocol.json", actual / "process/process.json"
        proof, process = read_json(protocol_path), read_json(process_path)
        if (proof.get("checkpoint_sha256") != expected["weight_sha256"] or proof.get("seed") != master
                or proof.get("episodes") != count or proof.get("purpose") != purpose
                or proof.get("task", {}).get("name") != self.protocol["task"]
                or proof.get("task", {}).get("max_episode_steps") != horizon):
            raise ArtifactConflict("native protocol proof differs from the logical request")
        if not proof.get("frozen_files"):
            raise ArtifactConflict("native protocol lacks frozen code evidence")
        assert_frozen(proof["frozen_files"])
        if process.get("status") != "completed" or process.get("returncode") != 0:
            raise ArtifactConflict("native evaluator process did not complete successfully")
        outputs = {"metrics": metrics, "request": request_path, "native_protocol": protocol_path,
                   "native_process": process_path, "checkpoint_contract": checkpoint_contract}
        for name in ("shard_plan.json", "merge_receipt.json"):
            if (path / name).is_file():
                outputs[name] = path / name
        return data, outputs

    def begin_evaluation(self, checkpoint: Path, path: Path, purpose: str, master: int, count: int) -> dict | None:
        path = self._within(path)
        key, request = self._key(path), self._evaluation_input(checkpoint, path, purpose, master, count)
        cached = self.store.reuse(key, **request, compatibility=self.compatibility)
        if cached is not None:
            return self._validate_evaluation(Path(checkpoint), path, purpose, master, count,
                                             inputs=request["inputs"])[0]
        metrics = path / "evaluation_metrics.json"
        if not metrics.is_file() and self._started(path):
            raise ArtifactConflict("evaluation started reset/rollout but has no complete committed result")
        self._prepare(key, request, existing_output=metrics.exists())
        self._checkpoint_contract(Path(checkpoint), path, request["inputs"], create=not metrics.exists())
        if metrics.is_file():
            # The original matching intent survived; verify every business proof
            # before completing a crash between metric publication and commit.
            return self.commit_evaluation(checkpoint, path, purpose, master, count)
        return None

    def commit_evaluation(self, checkpoint: Path, path: Path, purpose: str, master: int, count: int,
                          result: dict | None = None) -> dict:
        path = self._within(path)
        key = self._key(path)
        request = self._evaluation_input(checkpoint, path, purpose, master, count)
        row = read_json(self.store._path(key))
        self.store._check(row, request["inputs"], self.science)
        data, outputs = self._validate_evaluation(Path(checkpoint), path, purpose, master, count, result,
                                                   inputs=request["inputs"])
        self._commit(key, outputs)
        return data

    @staticmethod
    def _data_identity(rows) -> list[dict]:
        result = []
        for row in rows:
            profile, value = (row["profile"], row["root"]) if isinstance(row, dict) else row
            root = Path(value)
            files = [root / "meta" / name for name in ("info.json", "stats.json", "episodes.jsonl", "tasks.jsonl")]
            if not files[0].is_file():
                raise ArtifactConflict("training dataset metadata is missing")
            result.append({"profile": profile, "root": str(root.absolute()),
                           "metadata": {p.name: digest(p) for p in files if p.is_file()}})
        return result

    def begin_round(self, round_dir: Path, round_index: int, context: dict, cumulative: list) -> dict | None:
        round_dir = self._within(round_dir)
        checkpoint = Path(context["current_policy_checkpoint"])
        inputs = {"kind": "round", "round_index": round_index, "context_sha256": object_digest(context),
                  "current_checkpoint_sha256": digest(checkpoint / "model.safetensors"),
                  "current_checkpoint_config_sha256": digest(checkpoint / "config.json"),
                  "cumulative_data": self._data_identity(cumulative)}
        request, key = self._request(inputs), self._key(round_dir)
        cached = self.store.reuse(key, **request, compatibility=self.compatibility)
        result_path = round_dir / "round_result.json"
        if cached is not None:
            result = read_json(result_path)
            self._round_outputs(result_path, result)
            return result
        state_path = self.root / "run_state.json"
        if state_path.is_file() and read_json(state_path).get("final_confirmation_opened"):
            raise ArtifactConflict("final opened: only previously committed rounds can be reconstructed")
        self._prepare(key, request, existing_output=result_path.exists())
        self.pending_rounds[key] = request
        if result_path.exists():
            return self.commit_round(result_path, read_json(result_path))
        return None

    def _round_outputs(self, result_path: Path, result: dict) -> dict:
        if object_digest(read_json(result_path)) != object_digest(result):
            raise ArtifactConflict("round result differs from its durable record")
        directory = result_path.parent
        proposal_path = directory / "proposal.json"
        proposal_record = read_json(proposal_path)
        if proposal_record.get("validation") != "passed" or not isinstance(proposal_record.get("proposal"), dict):
            raise ArtifactConflict("round proposal lacks the validated controller receipt")
        proposal = proposal_record["proposal"]
        if result.get("proposal") != proposal or result.get("proposal_id") != proposal.get("proposal_id"):
            raise ArtifactConflict("round proposal identity changed")
        outputs = {"round_result": result_path, "proposal": proposal_path}
        status = result.get("status")
        if status == "stopped_by_controller":
            if proposal.get("decision") != "stop":
                raise ArtifactConflict("controller stop lacks a stop proposal")
            return outputs
        if status not in {"completed", "quarantined"}:
            raise ArtifactConflict("round has no valid terminal contract")
        checkpoint = self._within(Path(result["checkpoint"]))
        if digest(checkpoint / "model.safetensors") != result.get("checkpoint_sha256"):
            raise ArtifactConflict("round checkpoint changed")
        outputs.update(checkpoint=checkpoint / "model.safetensors", checkpoint_config=checkpoint / "config.json",
                       admission=directory / "data_admission.json")
        contract_path = directory / "candidate/checkpoint_contract.json"
        contract = read_json(contract_path)
        if (contract.get("status") != "passed" or contract.get("checkpoint_sha256") != result["checkpoint_sha256"]
                or contract.get("config_sha256") != digest(checkpoint / "config.json")):
            raise ArtifactConflict("candidate checkpoint contract is missing, failed or stale")
        helper_hash = self.protocol.get("act_checkpoint_contract", {}).get("implementation_sha256")
        if helper_hash and contract.get("compatibility_sha256") != helper_hash:
            raise ArtifactConflict("candidate inference adapter differs from frozen protocol")
        outputs["checkpoint_contract"] = contract_path
        for key in ("mixture", "training_exposure_audit", "candidate_failure_analysis"):
            outputs[key] = self._within(Path(result[key]))
        exposure = read_json(outputs["training_exposure_audit"])
        if not any(int(p.get("yielded_samples", 0)) > 0 and p.get("source_kind") == "research_requested_collection"
                   for p in exposure.get("parts", [])):
            raise ArtifactConflict("requested collection never entered actual training batches")
        analysis = read_json(outputs["candidate_failure_analysis"])
        if object_digest(analysis) != result.get("candidate_evidence_id"):
            raise ArtifactConflict("candidate evidence identity changed")
        # Explicitly include metadata files as committed dependencies, without
        # walking large environments or hashing all video payloads each resume.
        data = self._data_identity(result["cumulative_data"])
        for index, row in enumerate(data):
            for name in row["metadata"]:
                path = self._within(Path(row["root"]) / "meta" / name)
                outputs[f"data_{index}_{name}"] = path
        if status == "completed":
            evaluation = self._within(Path(result["development_evaluation"]))
            evaluation_key = self._key(evaluation)
            commit_path = self.store._path(evaluation_key)
            record = read_json(commit_path)
            if record.get("state") != "committed":
                raise ArtifactConflict("round development evaluation was not committed")
            metrics = read_json(evaluation / "evaluation_metrics.json")
            master, count = metrics["config"]["seed"], metrics["config"]["episode_count"]
            if count != self.protocol.get("budget", {}).get("development_episodes", count):
                raise ArtifactConflict("round development count differs from frozen research budget")
            request = self._evaluation_input(checkpoint, evaluation, "development", master, count)
            if self.store.reuse(evaluation_key, **request, compatibility=self.compatibility) is None:
                raise ArtifactConflict("round development commit is incomplete")
            self._validate_evaluation(checkpoint, evaluation, "development", master, count,
                                      inputs=request["inputs"])
            if result.get("development_summary") != metrics.get("summary"):
                raise ArtifactConflict("round summary differs from committed development")
            outputs.update(development_commit=commit_path, development_metrics=evaluation / "evaluation_metrics.json")
        else:
            recovery = result.get("recovery", {}).get("decision", {})
            if recovery.get("action") != "quarantine_candidate" or result.get("development_summary") is not None:
                raise ArtifactConflict("quarantined candidate has an invalid recovery contract")
            fallback = Path(result["fallback_policy_checkpoint"])
            if digest(fallback / "model.safetensors") != result.get("fallback_policy_checkpoint_sha256"):
                raise ArtifactConflict("quarantine fallback checkpoint changed")
        return outputs

    def commit_round(self, result_path: Path, result: dict) -> dict:
        result_path = self._within(result_path)
        key = self._key(result_path.parent)
        if key not in self.pending_rounds:
            raise ArtifactConflict("round has no checked begin intent in this supervisor")
        self._commit(key, self._round_outputs(result_path, result))
        return result

    def environment_check(self, *, interpreters: dict, assets: dict | None = None,
                          checkpoint: Path | None = None, inventory: dict | None = None) -> dict:
        return collect_environment_contract(self.runtime.repo, interpreters=interpreters,
            assets=assets, checkpoint=checkpoint, inventory=inventory,
            source_files=[Path(__file__)], output=self.root / "harness/environment_manifest.json")
