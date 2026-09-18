"""Recovery control-flow tests use CPU fixtures, never claim simulator results."""
import json
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from autosim.research.common import atomic_json, digest, object_digest, read_json
from autosim.research.evaluation import TraceEnv
from autosim.research.policy_rpc import policy_observation
from autosim.research.recovery_agent import RecoveryAgent, development_failure_snapshot, validate_decision
from autosim.research import repository_autoresearch as research
from autosim.research.common import find_benchmark
from autosim.research.robosyn_adapter import RoboSynAdapter


@lru_cache(maxsize=1)
def declared_space():
    """The task's real declaration, so fixture proposals face the contract the system runs.

    Read from the adapter rather than restated: these fixtures exercise failure handling,
    and a local copy of the schema would let them keep passing after the real one moved.
    """
    repo = find_benchmark("RoboSynChallenge", Path(__file__).resolve().parents[2],
                          marker="scripts/eval_policy.py")
    if repo is None:
        pytest.skip("RoboSynChallenge checkout not found")
    return RoboSynAdapter(repo).optimization_space("click_bell")


def decision(snapshot, **changes):
    eligible = snapshot.get("quarantine_eligible", False)
    result = {"schema_version": 1, "based_on_snapshot": object_digest(snapshot),
        "action": "quarantine_candidate" if eligible else "stop", "evidence_refs": ["quarantine_eligible"],
        "diagnosis": "Observed invalid state; physical cause remains unproven.",
        "next_trial_training_params": {"optimizer_lr": 5e-6, "n_action_steps": 10} if eligible else {},
        "next_trial_rationale": "Test a conservative training change within the frozen search space."}
    return result | changes


class FakeClient:
    model = "fixture-only"
    base_url = "https://fixture.invalid"
    available = True

    def __init__(self):
        self.calls = []

    def chat_with_metadata(self, system, user, **kwargs):
        self.calls.append(json.loads(user))
        return json.dumps(decision(self.calls[-1]["snapshot"])), {"model": self.model, "usage": {"total_tokens": 300}}


def failed_evaluation(path, message="ValueError: invalid state shape/values: (1, 14)"):
    shard = path / "shards/shard_00"
    atomic_json(shard / "worker_failure.json", {"error": message})
    atomic_json(shard / "process/process.json", {"status": "failed", "returncode": 1})
    return shard


def snapshot(path, **kwargs):
    return development_failure_snapshot(path, state_dim=14, checkpoint_sha256="fixture",
        baseline_verified=True, round_index=1, rounds_remaining=1,
        training_params={}, training_space=research.training_space_view(declared_space()), **kwargs)


def test_legacy_nan_classification_keeps_raw_logs_private(tmp_path):
    root = tmp_path / "round_1_candidate_development"
    shard = failed_evaluation(root)
    (shard / "telemetry.jsonl").write_text(json.dumps({"step": 260, "robot_qpos": [float("nan")] * 14,
                                                    "secret_field": "do-not-send"}) + "\n")
    (shard / "stdout.log").write_text("API_KEY=do-not-send")
    data = snapshot(root)
    assert data["quarantine_eligible"]
    assert data["shards"][0]["nonfinite_samples"][0]["nonfinite_indices"] == list(range(14))
    serialized = json.dumps(data, allow_nan=False)
    assert "do-not-send" not in serialized and str(tmp_path) not in serialized
    failed_evaluation(root / "unused")  # outside the expected shard layout is never read


@pytest.mark.parametrize("message", ["ValueError: invalid state shape/values: (1, 13)",
                                     "RuntimeError: CUDA out of memory", "RuntimeError: unknown fault"])
def test_unrecognized_or_shape_failure_cannot_be_quarantined(tmp_path, message):
    root = tmp_path / "round_1_candidate_development"
    failed_evaluation(root, message)
    data = snapshot(root)
    assert not data["quarantine_eligible"]
    with pytest.raises(ValueError, match="preconditions"):
        validate_decision(decision(data, action="quarantine_candidate"), data)


def test_mixed_failed_workers_and_completed_aggregate_fail_closed(tmp_path):
    root = tmp_path / "round_1_candidate_development"
    failed_evaluation(root)
    atomic_json(root / "shards/shard_01/process/process.json", {"returncode": -9, "status": "failed"})
    assert not snapshot(root)["quarantine_eligible"]
    atomic_json(root / "evaluation_metrics.json", {})
    assert not snapshot(root)["quarantine_eligible"]
    with pytest.raises(ValueError, match="development only"):
        snapshot(tmp_path / "round_1_candidate_final_confirmation")


def test_unstarted_planned_shard_or_changed_checkpoint_cannot_be_hidden(tmp_path):
    root = tmp_path / "round_1_candidate_development"
    failed_evaluation(root)
    atomic_json(root / "shard_plan.json", {"count": 2})
    assert not snapshot(root)["quarantine_eligible"]
    atomic_json(root / "shards/shard_01/evaluation_metrics.json", {})
    assert snapshot(root)["quarantine_eligible"]
    atomic_json(root / "evaluation_request.json", {"weight_sha256": "different checkpoint"})
    assert not snapshot(root)["quarantine_eligible"]


def test_recovery_api_is_cached_and_budgeted(tmp_path):
    root = tmp_path / "round_1_candidate_development"
    failed_evaluation(root)
    data, client = snapshot(root), FakeClient()
    agent = RecoveryAgent(tmp_path / "agent", client, max_calls=1)
    first = agent.decide(data)
    assert agent.decide(data) == first and len(client.calls) == 1
    with pytest.raises(RuntimeError, match="budget exhausted"):
        agent.decide(data | {"round": 2})


@pytest.mark.parametrize("changes", [
    {"action": "run_shell"}, {"shell": "echo arbitrary"}, {"based_on_snapshot": "wrong"},
    {"evidence_refs": ["final_confirmation"]}, {"schema_version": True},
    {"next_trial_training_params": {"optimizer_lr": 0.2}},
    {"next_trial_training_params": {"optimizer_lr": True}},
    {"next_trial_training_params": {"training_steps": 100}},
])
def test_malicious_or_out_of_protocol_decision_is_rejected(tmp_path, changes):
    root = tmp_path / "round_1_candidate_development"
    failed_evaluation(root)
    data = snapshot(root)
    with pytest.raises(ValueError):
        validate_decision(decision(data, **changes), data)


def test_unknown_remote_outcome_is_never_recharged(tmp_path):
    root = tmp_path / "round_1_candidate_development"
    failed_evaluation(root)
    data, client = snapshot(root), FakeClient()
    def lost(*args, **kwargs):
        client.calls.append("one")
        raise TimeoutError("API_KEY=never persist this")
    client.chat_with_metadata = lost
    agent = RecoveryAgent(tmp_path / "agent", client)
    with pytest.raises(RuntimeError, match="request failed"):
        agent.decide(data)
    with pytest.raises(RuntimeError, match="unresolved API request"):
        agent.decide(data)
    assert len(client.calls) == 1
    assert all("never persist" not in p.read_text() for p in (tmp_path / "agent").rglob("*.json"))


def test_first_invalid_step_captures_action_without_changing_observation(tmp_path):
    finite = {"robot": {"qpos": np.zeros((1, 14)), "qvel": np.ones((1, 14))}}
    invalid = {"robot": {"qpos": np.full((1, 14), np.nan)}}
    base = Mock()
    base.step.side_effect = [(finite, 0, False, False, {}), (invalid, 0, False, False, {})]
    proxy = TraceEnv(base, None, tmp_path, None, "development")
    proxy.seed = 42
    proxy.step(np.ones(14))
    assert proxy.step(np.full(14, 2))[0] is invalid
    report = read_json(tmp_path / "nonfinite_dynamics.json")
    assert report["first_observed_step"] == 2
    assert report["recent_steps"][-1]["action"] == [2.0] * 14
    assert report["recent_steps"][-1]["qpos"] == [None] * 14
    assert np.isnan(invalid["robot"]["qpos"]).all()
    with pytest.raises(ValueError, match="non-finite observation.state"):
        policy_observation(invalid, {"state_dim": 14, "cameras": []})
    sealed = TraceEnv(base, None, tmp_path / "sealed", None, "final_confirmation")
    sealed._capture_dynamics(np.zeros(14), invalid)
    assert not (tmp_path / "sealed").exists()


def cpu_runner(tmp_path, monkeypatch, *, rounds=2, all_invalid=False, error=None):
    """Exercise the real supervisor/round/proposal/recovery/selection; GPU leaves are fixtures."""
    runner = research.RepositoryAutoResearch(research.MilestoneConfig(repo_input=str(tmp_path),
        output_root=tmp_path / "runs", run_id="cpu_fixture", task="click_bell", controller="fixed",
        recovery_controller="api", rounds=rounds, development_episodes=2, selection_episodes=2,
        attempts_per_round=10, screen_steps=5))
    root = runner.run_root
    official = root / "fixture_official"
    official.mkdir()
    (official / "model.safetensors").write_bytes(b"fixture, not a real checkpoint")
    data = root / "fixture_data"
    targeted = root / "fixture_targeted"
    for path in (data, targeted):
        atomic_json(path / "meta/info.json", {"total_episodes": 1, "total_frames": 3})
    runner.assets = {"official_checkpoint": str(official), "official_dataset": str(data),
                     "checkpoint_weight_sha256": digest(official / "model.safetensors"),
                     "dataset_info_sha256": digest(data / "meta/info.json")}
    runner.task = "click_bell"
    runner.spec = SimpleNamespace(state_dim=14, correction_supported=True, as_dict=lambda: {"state_dim": 14})
    runner.adapter = SimpleNamespace(challenge_context=lambda spec: {}, capabilities=lambda spec: [],
                                     optimization_space=lambda task: declared_space())
    runner.allowed_profiles = {"targeted_recovery"}
    runtime = SimpleNamespace(deadline=None, cost_model={}, prepare_data=lambda *args: {})
    runner.runtime = runtime
    trained = []
    def train(spec, dataset, output, **kwargs):
        trained.append(kwargs)
        path = output / "checkpoint"
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").write_bytes(b"fixture candidate")
        atomic_json(output / "training_exposure_audit.json", {"parts": [{
            "source_kind": "research_requested_collection", "yielded_samples": 2}]})
        return path
    runtime.train = train
    client = FakeClient()
    monkeypatch.setattr("autosim.research.compute_agent.client_from_file", lambda path: client)
    monkeypatch.setattr(research, "LLMClient", lambda: client)
    monkeypatch.setattr(research, "gpu_lock", lambda gpu: nullcontext())
    monkeypatch.setattr(runner, "initialize", lambda: runner.save(status="running", stage="initialized"))
    monkeypatch.setattr(runner, "start_or_resume_budget", lambda: None)
    monkeypatch.setattr(runner, "update_capability", lambda *args: None)
    monkeypatch.setattr(runner, "_prepare_official_data", lambda: {})
    monkeypatch.setattr(runner, "run_phase", lambda phase, jobs: {job.name: job.run(runtime) for job in jobs})
    acting = []
    def collect(index, name, **kwargs):
        if name == "targeted":
            acting.append(kwargs["checkpoint"])
        if name == "original" or index > 1:
            return None
        return {"requested_factor_readback": "verified", "accepted_episodes": 1,
                "dataset_root": str(targeted), "attempts_consumed": 1}
    monkeypatch.setattr(runner, "_collect_one", collect)
    evaluated = []
    def evaluate(spec, checkpoint, output, *, episodes, master_seed, purpose):
        evaluated.append(output.name)
        if purpose == "development" and ("round_1_candidate" in output.name or
                                          all_invalid and "candidate" in output.name):
            failed_evaluation(output, error or "ValueError: invalid state shape/values: (1, 14)")
            raise RuntimeError("fixture evaluation jobs failed")
        result = {"purpose": purpose, "execution_mode": "real_simulation",  # contract-shaped CPU fixture only
            "config": {"task": "click_bell", "seed": master_seed},
            "summary": {"episode_count": episodes, "success_count": 0, "success_rate": 0,
                        "average_action_steps": 10},
            "episodes": [{"episode_seed": i, "success": False} for i in range(episodes)]}
        atomic_json(output / "evaluation_metrics.json", result)
        return result
    runtime.evaluate = evaluate
    def analyze(task, evaluation, path, repo=None):
        value = {"summary": {"fixture_only": True}, "task_evidence": {"task": task}}
        atomic_json(path, value)
        return value
    monkeypatch.setattr(research, "task_evidence", analyze)
    deployments = []
    def export(deployment, selected, final):
        deployments.append(deployment)
        atomic_json(root / "export_validation.json", {"status": "CPU_fixture", "export_runtime_verified": False})
        return root / "fixture_export"
    monkeypatch.setattr(runner, "_export", export)
    return runner, client, trained, evaluated, acting, deployments


def test_candidate_failure_continues_next_round_and_excludes_invalid_ranking(tmp_path, monkeypatch):
    runner, client, trained, evaluated, acting, deployments = cpu_runner(tmp_path, monkeypatch)
    result = runner.execute()
    assert result["status"] == "completed_without_target_improvement"
    assert len(trained) == 2 and len(client.calls) == 1
    assert trained[1]["params"]["optimizer_lr"] == 5e-6
    assert trained[1]["params"]["n_action_steps"] == 10
    assert trained[1]["pretrained"] == acting[1] == Path(runner.assets["official_checkpoint"])
    assert "round_1_candidate_selection_validation" not in evaluated
    assert "round_2_candidate_selection_validation" in evaluated
    prior = read_json(runner.run_root / "rounds/round_1/round_result.json")
    assert prior["status"] == "quarantined" and prior["development_summary"] is None
    assert not (Path(prior["development_evaluation"]) / "evaluation_metrics.json").exists()
    assert read_json(runner.run_root / "selection.json")["selected_round"] == 2
    # A cached invalid round must not retrain/re-evaluate or charge the LLM again.
    again, _ = runner.run_round(1, {}, [])
    assert again == prior and len(trained) == 2 and len(client.calls) == 1


@pytest.mark.parametrize("rounds", [1, 2])
def test_all_candidates_invalid_retains_official_without_score_claim(tmp_path, monkeypatch, rounds):
    runner, client, trained, evaluated, acting, deployments = cpu_runner(tmp_path, monkeypatch,
                                                                       rounds=rounds, all_invalid=True)
    result = runner.execute()
    assert result["execution_complete"] and not result["result_valid"]
    assert result["selected_checkpoint"] is None and not result["performance_target_achieved"]
    assert result["quarantined_rounds"] == list(range(1, rounds + 1))
    assert not any("selection_validation" in name for name in evaluated)
    assert deployments == [Path(runner.assets["official_checkpoint"])]


def test_unknown_failure_diagnoses_and_preserves_original_exception(tmp_path, monkeypatch):
    runner, client, trained, _, _, _ = cpu_runner(tmp_path, monkeypatch, error="RuntimeError: unknown kernel error")
    with pytest.raises(RuntimeError, match="fixture evaluation jobs failed"):
        runner.execute()
    assert runner.state["status"] == "blocked"
    assert runner.state["stage"] == "round_1_candidate_development"
    assert len(trained) == len(client.calls) == 1
    assert runner.state["recovery"]["decision"]["action"] == "stop"


def test_recovery_does_not_extend_deadline(tmp_path, monkeypatch):
    runner, client, *_ = cpu_runner(tmp_path, monkeypatch)
    runner.runtime.deadline = 1
    with pytest.raises(RuntimeError, match="fixture evaluation jobs failed"):
        runner.execute()
    assert not client.calls and runner.runtime.deadline == 1
    assert runner.state["recovery"]["error_type"] == "TimeoutError"


def test_final_failure_diagnosis_never_receives_evaluation_details(tmp_path, monkeypatch):
    runner, client, *_ = cpu_runner(tmp_path, monkeypatch)
    def fail():
        runner.save(stage="final_confirmation")
        raise ValueError("final seed 12345 API_KEY=private-value")
    monkeypatch.setattr(runner, "initialize", fail)
    with pytest.raises(ValueError):
        runner.execute()
    outgoing = json.dumps(client.calls)
    assert "12345" not in outgoing and "private-value" not in outgoing
    assert client.calls[0]["snapshot"]["research_feedback_permitted"] is False


def test_disabled_recovery_never_calls_api(tmp_path, monkeypatch):
    from dataclasses import replace
    runner, client, trained, *_ = cpu_runner(tmp_path, monkeypatch)
    runner.config = replace(runner.config, recovery_controller="disabled")
    with pytest.raises(RuntimeError, match="fixture evaluation jobs failed"):
        runner.execute()
    assert not client.calls and len(trained) == 1


def test_malformed_next_proposal_remains_a_repairable_validation_error():
    with pytest.raises(ValueError, match="training.params"):
        research.RepositoryAutoResearch._bind_recovery_params({"training": {}},
            {"recovery_training_constraints": {"optimizer_lr": 5e-6}})


def _checkout_looking_like_robosyn(root):
    """The entry point settles identity before it builds anything, so the fixture must be
    a checkout it can identify -- otherwise this tests the identity gate, not exit codes."""
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "robosynchallenge" / "tasks").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text('name = "RoboSynChallenge"\n', encoding="utf-8")
    (root / "scripts" / "eval_policy.py").write_text("# native evaluator\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("status, code", [("completed", 0), ("completed_without_target_improvement", 0),
                                         ("prepared", 0), ("blocked", 3)])
def test_cli_operational_exit_code_is_separate_from_improvement(tmp_path, monkeypatch, status, code):
    configurations = []
    def fake_runner(config):
        configurations.append(config)
        return SimpleNamespace(execute=lambda: {"status": status})
    monkeypatch.setattr(research, "RepositoryAutoResearch", fake_runner)
    repo = _checkout_looking_like_robosyn(tmp_path)
    assert research.main([str(repo), "--recovery-controller", "api", "--recovery-max-calls", "3"]) == code
    assert configurations[0].recovery_controller == "api"
    assert configurations[0].recovery_max_calls == 3
