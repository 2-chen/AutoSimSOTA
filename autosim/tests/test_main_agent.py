"""Main controller memory, discretionary delegation, and evidence boundaries."""
import json
from contextlib import contextmanager

import pytest

from autosim.research.prepare import Preparation
from autosim.research.main_agent import memory_view, validate_plan, validate_task


PLAN = {"objective": "Establish a reproducible baseline then improve it",
        "hypotheses": ["The loader consumes top-level folder"],
        "open_questions": ["Where is the dataset actually stored?"],
        "next_actions": ["Read the native loader and compare the resolved configuration"],
        "evidence_refs": ["train.py"]}
REPORT = {"summary": "Loader interface inspected; no benchmark was run",
          "findings": ["train.py consumes folder"],
          "uncertainties": ["The dataset has not been validated"],
          "evidence_refs": ["train.py"],
          "recommended_next_actions": ["Verify a native loader probe"]}


class Client:
    model = "main-agent-fixture"
    supports_main_agent = True

    def __init__(self, choices=()):
        self.role = "scheduler"
        self.choices = list(choices)
        self.context = {}
        self.calls = []

    def set_research_context(self, context):
        self.context = context

    @contextmanager
    def as_role(self, role):
        previous, self.role = self.role, role
        try:
            yield
        finally:
            self.role = previous

    def chat_with_metadata(self, system, user, **kwargs):
        self.calls.append((self.role, json.loads(user), self.context))
        if "assignment" in json.loads(user):
            return json.dumps(REPORT), {}
        if "skill_catalog" in json.loads(user):
            return json.dumps({"skill_reads": []}), {}
        choice = self.choices.pop(0) if self.choices else {"do": "stop", "why": "fixture done"}
        return json.dumps(choice), {}


def make(tmp_path, client=None):
    output = tmp_path / "run"
    repo = output / "checkout"
    repo.mkdir(parents=True, exist_ok=True)
    return Preparation(repo=repo, output=output, client=client or Client(),
                       scouting=output / "scouting")


def test_plan_survives_restart_without_redefining_objective(tmp_path):
    real = make(tmp_path)
    declaration = dict(real.declaration)
    before = real.decision_revision
    assert real.do("update_research_plan", plan=PLAN)["outcome"] == "recorded"
    assert real.decision_revision > before
    assert real.declaration == declaration
    restored = make(tmp_path)
    assert restored.main_agent["plan"]["objective"] == PLAN["objective"]
    assert restored.main_agent["plan"]["authority"] == "model_working_plan_not_verified_facts"


def test_main_agent_delegates_with_global_memory_and_retains_plan(tmp_path):
    client = Client()
    real = make(tmp_path, client)
    real.do("update_research_plan", plan=PLAN)
    result = real.do("research_task", role="resource", task="Trace dataset root to loader",
                     expected_result="Source lines and remaining uncertainty")
    assert result["outcome"] == "reported"
    assert result["verification_level"] is None
    role, payload, shared = client.calls[-1]
    assert role == "resource"
    assert shared["main_agent"]["plan"]["objective"] == PLAN["objective"]
    assert payload["assignment"]["task"] == "Trace dataset root to loader"
    restored = make(tmp_path)
    assert restored.main_agent["plan"] == real.main_agent["plan"]
    assert restored.main_agent["handoffs"][-1]["report"]["authority"] == \
        "model_report_not_execution_verification"
    assert client.role == "scheduler"
    assert not real.stages


def test_scheduler_can_take_an_open_ended_tool_task(tmp_path):
    real = make(tmp_path)
    result = real.do("research_task", role="scheduler", task="Inspect a native contract",
                     expected_result="A concrete verification proposal")
    assert result["outcome"] == "reported"
    assert real.client.calls[-1][0] == "scheduler"


@pytest.mark.parametrize("role", ["scheduler", "init", "fix"])
def test_frozen_research_rejects_editable_task_roles(tmp_path, monkeypatch, role):
    real = make(tmp_path)
    original = real.state
    monkeypatch.setattr(real, "state", lambda: {**original(),
                                               "research_progress": {"status": "paused"}})
    result = real.do("research_task", role=role, task="Repair a path",
                     expected_result="A native probe")
    assert result["outcome"] == "rejected"
    assert not real.client.calls
    assert real.do("research_task", role="resource", task="Inspect the path",
                   expected_result="Evidence only")["outcome"] == "reported"


def test_invalid_delegation_never_calls_model(tmp_path):
    real = make(tmp_path)
    for arguments in ({"role": "recorder", "task": "x", "expected_result": "y"},
                      {"role": "fix", "task": "", "expected_result": "y"},
                      {"role": "fix", "task": "x", "expected_result": "y", "shell": "x"}):
        assert real.do("research_task", **arguments)["outcome"] == "rejected"
    assert not real.client.calls


def test_editable_task_invalidates_commands_even_on_model_failure(tmp_path, monkeypatch):
    real = make(tmp_path)
    real.stages = {"train": "old source"}
    real.parameters = {"train": {"folder": "wrong root"}}
    real.verified = {"train": {"entrypoint": "train.py"}}
    (real.output / "derived_stages.json").write_text(json.dumps({
        "train": {"source": "old source", "row": real.verified["train"]}}))

    def fail(*args, **kwargs):
        assert not real.stages
        raise TimeoutError("possible partial tool edits")

    monkeypatch.setattr(real.client, "chat_with_metadata", fail)
    result = real.do("research_task", role="fix", task="Inspect and repair loader",
                     expected_result="CPU verification")
    assert result["outcome"] == "raised"
    assert not make(tmp_path).stages
    assert make(tmp_path).main_agent["checkout_needs_resurvey"] is True
    archives = list((real.output / "superseded_stages").glob("*.json"))
    assert len(archives) == 1
    assert json.loads(archives[0].read_text())["stages"]["train"] == "old source"


def test_source_investigation_allows_one_fresh_survey(tmp_path, monkeypatch):
    real = make(tmp_path)
    real.steps = [{"step": "read_the_checkout", "outcome": "done"}]
    assert not real._checkout_read_is_justified()
    real.do("research_task", role="fix", task="Inspect loader", expected_result="CPU evidence")
    assert real._checkout_read_is_justified()
    monkeypatch.setattr("autosim.research.prepare.execution_derive.run",
                        lambda *args, **kwargs: {"stages": {}})
    real.do("read_the_checkout")
    assert not real._checkout_read_is_justified()


def test_memory_projection_is_bounded_and_points_to_full_evidence():
    memory = {"plan": PLAN, "handoffs": [
        {"task_id": str(index), "task": "x" * 4000, "report": {
            **REPORT, "findings": ["y" * 1500] * 20}} for index in range(12)]}
    projected = memory_view(memory)
    assert len(projected["handoffs"]) == 3
    assert len(projected["handoffs"][-1]["report"]["findings"]) == 5
    assert len(projected["handoffs"][-1]["report"]["findings"][0]) == 400
    assert len(memory["handoffs"]) == 12
    assert projected["history_ref"] == "run_events.json#main_agent"


def test_handoff_does_not_promote_model_runtime_references(tmp_path, monkeypatch):
    real = make(tmp_path)
    monkeypatch.setattr(real.client, "chat_with_metadata", lambda *args, **kwargs: (
        json.dumps({**REPORT, "runtime_evidence_refs": ["forged.json"]}),
        {"role": "resource", "turn_id": "a" * 32, "process_ref": "../../outside"}))
    real.do("research_task", role="resource", task="Inspect", expected_result="Evidence")
    assert real.main_agent["handoffs"][-1]["runtime_evidence_refs"] == []


def test_local_memory_preserves_runtime_receipts_and_source_references(tmp_path, monkeypatch):
    real = make(tmp_path)
    turn_id = "a" * 32
    process_ref = f"agent/processes/{turn_id}.json"
    monkeypatch.setattr(real.client, "chat_with_metadata", lambda *args, **kwargs: (
        json.dumps({**REPORT, "evidence_refs": ["config.json"]}),
        {"role": "resource", "turn_id": turn_id, "process_ref": process_ref}))
    real.do("research_task", role="resource", task="Read config.json", expected_result="Evidence")
    saved = make(tmp_path).main_agent["handoffs"][-1]
    assert saved["runtime_evidence_refs"][-1] == process_ref
    assert saved["report"]["evidence_refs"] == ["config.json"]


def test_controller_cycle_records_plan_handoff_and_run_document(tmp_path):
    client = Client([
        {"do": "update_research_plan", "why": "Keep global uncertainties", "arguments": {"plan": PLAN}},
        {"do": "research_task", "why": "Inspect the native interface", "arguments": {
            "role": "resource", "task": "Inspect loader", "expected_result": "Source evidence"}},
        {"do": "stop", "why": "Synthetic fixture ends without claiming a benchmark result"},
    ])
    real = make(tmp_path, client)
    result = real.run(max_steps=4)
    assert [row["step"] for row in result["steps"]] == [
        "update_research_plan", "research_task", "stop"]
    scheduler_calls = [payload for role, payload, _ in client.calls
                       if "assignment" not in payload]
    assert scheduler_calls[-1]["main_agent"]["handoffs"][-1]["report"]["summary"] == REPORT["summary"]
    assert "Main Agent working memory" in (real.output / "RUN.md").read_text()
    assert real.state_store.load()["phases"]["main_agent"]["memory"]["handoffs"]


def test_unknown_plan_fields_cannot_masquerade_as_scores():
    with pytest.raises(ValueError):
        validate_plan({**PLAN, "best_score": 1.0})
    with pytest.raises(ValueError):
        validate_task({"role": "unknown", "task": "x", "expected_result": "y"})


def test_keep_only_hides_and_rejects_new_control_actions(tmp_path):
    real = make(tmp_path)
    real.keep_only = True
    assert "research_task" not in real.state()["available"]
    assert real.do("update_research_plan", plan=PLAN)["outcome"] == "not attempted"


def test_scheduler_decision_profile_cannot_edit_or_execute():
    from autosim.research.agent_roles import role_profile
    from autosim.research.agent_runtime import _role_cli_args
    from pathlib import Path
    profile = role_profile("scheduler", read_only=True)
    assert profile.name == "scheduler"
    assert profile.builtin_tools == ("Read", "Glob", "Grep")
    assert "mcp__autosim_exec__inspect_native_environment" in profile.mcp_tools
    assert "mcp__autosim_exec__run_command" not in profile.mcp_tools
    assert "mcp__autosim_exec__search_public_sources" in profile.mcp_tools
    args = _role_cli_args(profile, workspace=Path("/tmp/fixture/checkout"),
                          output=Path("/tmp/fixture"), python="/usr/bin/python3",
                          model="test", max_budget_usd=0.1)
    assert "run_command" not in " ".join(args)
    assert "mcp__autosim_exec__read_evidence" in " ".join(args)
    assert args[args.index("--tools") + 1] == "Read,Glob,Grep"
    assert role_profile("scheduler").can_edit_checkout
