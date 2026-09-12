import json
from pathlib import Path

import pytest

from autosim.llm_client import LLMClient
from autosim.research.repository_autoresearch import (
    ClickBellAutoResearch,
    MilestoneConfig,
    build_prompt,
    discover_robosyn,
    load_deepseek_environment,
    make_parser,
    PROJECT_ROOT,
    resolve_assets,
    validate_proposal,
)
from autosim.research.decision_controllers import control_proposal


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
ROBOSYN = WORKSPACE / "RoboSynChallenge"


def proposal(**overrides):
    value = {
        "proposal_id": "round_1_contact",
        "parent_checkpoint_sha256": "weights",
        "parent_data_version": "data",
        "development_evidence_id": "evidence",
        "hypothesis": "contact suffixes may improve press depth",
        "primary_intervention": "policy_correction",
        "collection": {"enabled": True, "mode": "policy_correction", "profile": "targeted_recovery",
                       "targeted_attempts": 50, "original_attempts": 50, "target_episodes": 50},
        "training": {"steps": 20_000, "params": {"action_loss_profile": "valid_mean"},
                     "targeted_sampling_mass": 0.5,
                     "phase_weights": {"early": 0.75, "approach": 1.0, "contact_recovery": 3.0},
                     "horizon_floor": 0.25},
        "expected_validation": "improve complete task success on the fixed development bank",
    }
    value.update(overrides)
    return value


def test_real_repository_discovery_and_assets():
    found = discover_robosyn(ROBOSYN)
    assets = resolve_assets(ROBOSYN)
    assert found["benchmark"] == "RoboSynChallenge"
    assert found["task_contract"]["state_dim"] == 14
    assert len(found["task_contract"]["cameras"]) == 3
    assert found["task_contract"]["max_episode_steps"] == 361
    assert assets["official_episodes"] == 1000


def test_repository_misrecognition_is_rejected(tmp_path):
    (tmp_path / "pyproject.toml").write_text("name='something-else'", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        discover_robosyn(tmp_path)


def test_proposal_contract_accepts_bounded_targeted_collection():
    result = validate_proposal(proposal(), round_index=1, evidence_id="evidence",
                               parent_checkpoint_sha256="weights", parent_data_version="data")
    assert result["collection"]["original_attempts"] >= result["collection"]["targeted_attempts"]


@pytest.mark.parametrize("mutation, message", [
    ({"development_evidence_id": "final-bank"}, "wrong development evidence"),
    ({"primary_intervention": "targeted_data",
      "collection": {"enabled": True, "mode": "expert", "profile": "targeted_camera",
                     "targeted_attempts": 60, "original_attempts": 40, "target_episodes": 30}}, "minimum original-distribution"),
    ({"training": {"steps": 20_000, "params": {"optimizer_lr": 0.1},
                   "targeted_sampling_mass": 0.2,
                   "phase_weights": {"early": 1, "approach": 1, "contact_recovery": 1},
                   "horizon_floor": 0.25}}, "outside allowed space"),
])
def test_proposal_contract_rejects_invalid_or_leaking_actions(mutation, message):
    with pytest.raises(ValueError, match=message):
        validate_proposal(proposal(**mutation), round_index=1, evidence_id="evidence",
                          parent_checkpoint_sha256="weights", parent_data_version="data")


def test_round_two_prompt_contains_real_round_one_feedback():
    context = {"development_evidence_id": "feedback-hash", "parent_checkpoint_sha256": "weights",
               "parent_data_version": "data", "round_2_must_use_round_1_feedback": True,
               "development_evidence": {"round_1_development_summary": {"success_rate": 0.7}}}
    _, user = build_prompt(2, context)
    assert "feedback-hash" in user and "round_1_development_summary" in user


def test_round_two_can_stop_without_spending_more_budget():
    value = proposal(
        decision="stop", primary_intervention="data_processing",
        collection={"enabled": False, "mode": "expert", "profile": "targeted_camera",
                    "targeted_attempts": 0, "original_attempts": 0, "target_episodes": 0})
    assert validate_proposal(
        value, round_index=2, evidence_id="evidence",
        parent_checkpoint_sha256="weights", parent_data_version="data")["decision"] == "stop"


def test_deepseek_environment_has_precedence(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.example")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    monkeypatch.setenv("AUTOSIM_LLM_API_KEY", "legacy-secret")
    client = LLMClient()
    assert client.api_key == "deepseek-secret"
    assert client.base_url == "https://api.deepseek.example"
    assert client.model == "deepseek-flash"


def test_dotenv_loader_is_narrow_and_never_returns_secret(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY='secret-value'\nUNRELATED=do-not-load\n", encoding="utf-8")
    env.chmod(0o600)
    monkeypatch.chdir(tmp_path)
    for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL", "UNRELATED"):
        monkeypatch.delenv(key, raising=False)
    result = load_deepseek_environment(project_root=tmp_path)
    assert result["api_key_available"] is True
    assert "secret-value" not in json.dumps(result)
    assert "UNRELATED" not in __import__("os").environ
    assert result["model"] == "deepseek-flash"


def test_dotenv_loader_rejects_unsafe_permissions(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=secret-value\n", encoding="utf-8")
    env.chmod(0o644)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(PermissionError, match="group/world"):
        load_deepseek_environment(project_root=tmp_path)


def test_default_output_root_is_project_scoped(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    args = make_parser().parse_args([str(ROBOSYN)])
    assert args.output_root == PROJECT_ROOT / "autoresearch_runs"


def test_research_seed_derives_independent_purpose_banks(tmp_path):
    legacy = ClickBellAutoResearch(MilestoneConfig(
        repo_input=str(ROBOSYN), output_root=tmp_path, run_id="seed1000", train_seed=1000))
    replicate = ClickBellAutoResearch(MilestoneConfig(
        repo_input=str(ROBOSYN), output_root=tmp_path, run_id="seed1001", train_seed=1001))
    assert legacy._master_seed("development") == 310_911_001
    roles = ("development", "selection_validation", "final_confirmation",
             "collection_round_1_original", "collection_round_1_targeted", "export_smoke_1")
    assert len({replicate._master_seed(role) for role in roles}) == len(roles)
    assert all(replicate._master_seed(role) != legacy._master_seed(role) for role in roles)
    assert replicate._master_seed("development") == replicate._master_seed("development")


def test_non_clickbell_api_egress_requires_explicit_authorization(tmp_path):
    blocked = ClickBellAutoResearch(MilestoneConfig(
        repo_input=str(ROBOSYN), output_root=tmp_path, run_id="water", task="water_pouring"))
    blocked.task = "water_pouring"
    with pytest.raises(PermissionError, match="allow-api-egress"):
        blocked._require_api_egress_authorization()
    allowed = ClickBellAutoResearch(MilestoneConfig(
        repo_input=str(ROBOSYN), output_root=tmp_path, run_id="water-ok", task="water_pouring",
        allow_api_egress=True))
    allowed.task = "water_pouring"
    allowed._require_api_egress_authorization()


def test_task_and_budget_are_explicit_cli_protocol_fields():
    args = make_parser().parse_args([
        str(ROBOSYN), "--task", "click_bell", "--rounds", "3",
        "--attempts-per-round", "40", "--training-steps", "5000",
        "--min-original-fraction", "0.3",
    ])
    assert (args.task, args.rounds, args.attempts_per_round, args.training_steps) == (
        "click_bell", 3, 40, 5000)
    assert args.min_original_fraction == pytest.approx(0.3)


def test_generic_discovery_does_not_require_clickbell_dimensions():
    found = discover_robosyn(ROBOSYN, "water_pouring")
    assert found["selected_task"] == "water_pouring"
    assert found["task_contract"]["instruction"]
    assert len(found["inventory"]) == 10


@pytest.mark.parametrize("strategy", ["fixed", "random", "heuristic"])
def test_matched_control_controllers_return_validated_proposals(strategy):
    context = {
        "development_evidence_id": "evidence", "parent_checkpoint_sha256": "weights",
        "parent_data_version": "data", "allowed_collection_profiles": [
            "composite_hard", "targeted_camera", "targeted_recovery"],
        "allowed_collection_modes": ["expert"], "attempts_per_round": 40,
        "training_steps": 5000, "min_original_fraction": 0.25,
        "development_evidence": {"categories": {"little_object_motion": 3}},
    }
    value = control_proposal(strategy, 1, context, seed=1000)
    result = validate_proposal(
        value, round_index=1, evidence_id="evidence",
        parent_checkpoint_sha256="weights", parent_data_version="data",
        allowed_profiles=set(context["allowed_collection_profiles"]),
        allowed_collection_modes={"expert"}, attempts_per_round=40,
        training_steps=5000, min_original_fraction=0.25)
    assert result["collection"]["targeted_attempts"] + result["collection"]["original_attempts"] == 40


def test_fixed_control_prefers_readable_profile_without_clickbell_composite():
    context = {
        "development_evidence_id": "evidence", "parent_checkpoint_sha256": "weights",
        "parent_data_version": "data", "allowed_collection_profiles": [
            "targeted_appearance", "targeted_camera", "targeted_clutter", "targeted_recovery"],
        "allowed_collection_modes": ["expert"], "attempts_per_round": 10,
        "training_steps": 200, "min_original_fraction": 0.25,
        "development_evidence": {},
    }
    value = control_proposal("fixed", 1, context, seed=1000)
    assert value["collection"]["profile"] == "targeted_recovery"


def test_completed_run_returns_without_initialization(tmp_path, monkeypatch):
    root = tmp_path / "RoboSynChallenge" / "done"
    root.mkdir(parents=True)
    (root / "run_state.json").write_text(json.dumps({
        "schema_version": 1, "status": "completed", "stage": "complete",
        "repo": str(ROBOSYN.absolute()), "run_id": "done", "error": None,
    }), encoding="utf-8")
    runner = ClickBellAutoResearch(MilestoneConfig(
        repo_input=str(ROBOSYN), output_root=tmp_path, run_id="done"))
    monkeypatch.setattr(runner, "initialize", lambda: (_ for _ in ()).throw(AssertionError("called")))
    assert runner.execute()["status"] == "completed"


def test_dry_run_is_repository_in_and_no_gpu_or_api(tmp_path):
    runner = ClickBellAutoResearch(MilestoneConfig(
        repo_input=str(ROBOSYN), output_root=tmp_path, run_id="dry", dry_run=True))
    result = runner.execute()
    assert result["status"] == "prepared"
    assert result["benchmark"] == "RoboSynChallenge"
    assert (tmp_path / "RoboSynChallenge/dry/protocol.json").is_file()
    assert not (tmp_path / "RoboSynChallenge/dry/rounds").exists()
    protocol = json.loads((tmp_path / "RoboSynChallenge/dry/protocol.json").read_text())
    assert protocol["task"] == "click_bell"


def test_probe_only_reports_task_with_missing_assets(tmp_path):
    runner = ClickBellAutoResearch(MilestoneConfig(
        repo_input=str(ROBOSYN), output_root=tmp_path, run_id="sample-probe",
        task="sample_loading", probe_only=True))
    result = runner.execute()
    assert result["status"] == "prepared"
    manifest = json.loads((tmp_path / "RoboSynChallenge/sample-probe/asset_manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    capabilities = json.loads((tmp_path / "RoboSynChallenge/sample-probe/capabilities.json").read_text())
    assert capabilities["task"] == "sample_loading"
