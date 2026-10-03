from __future__ import annotations

import json
from pathlib import Path

import pytest

from autosim.research.agent_client import AgentRuntimeClientError, RoleAwareAgentClient


def client_for(tmp_path: Path, *, total: float = 2.0) -> RoleAwareAgentClient:
    output = tmp_path / "run"
    workspace = output / "checkout"
    workspace.mkdir(parents=True)
    return RoleAwareAgentClient(workspace=workspace, output=output,
                                run_id="research-test", turn_budget_usd=0.25,
                                total_budget_usd=total, timeout=90)


def test_stable_instructions_precede_dynamic_request_and_usage_tracks_bytes(tmp_path, monkeypatch):
    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        return {"status": "completed", "final_text": "{}"}
    monkeypatch.setattr("autosim.research.agent_runtime.run_coding_agent", run)
    client = client_for(tmp_path)
    client.set_research_context({"state_revision": 7, "budget": 2})
    _, metadata = client.chat_with_metadata("stable role", '{"budget":2,"request":"A"}')
    client.chat_with_metadata("stable role", '{"request":"B","budget":2}')
    marker = "## Current request and evidence"
    assert calls[0]["prompt"].split(marker)[0] == calls[1]["prompt"].split(marker)[0]
    assert metadata["prompt_efficiency"]["deduplicated_fields"] == 1
    assert metadata["prompt_efficiency"]["prompt_bytes"] == len(calls[0]["prompt"].encode())


def test_large_role_context_rotates_without_changing_run_budget(tmp_path):
    client = client_for(tmp_path)
    from autosim.research.common import atomic_json
    root = client.output / "agent"
    key = "a" * 32
    atomic_json(root / "roles" / "init.json", {
        "role": "init", "run_id": client.run_id, "session_key": key})
    atomic_json(root / "sessions" / f"{key}.json", {
        "schema_version": 1, "pricing_ref": "agent/pricing/previous.json"})
    atomic_json(root / "pricing" / "previous.json", {"gateway": {"receipts": [
        {"usage": {"input_miss_tokens": 10_000, "input_hit_tokens": 150_000}}]}})
    assert client._resume_role("init") is False
    assert client.total_budget_usd == 2
    assert json.loads((root / "context_rotation.json").read_text())["input_tokens"] == 160_000


def test_only_verified_global_exhaustion_is_terminal():
    from autosim.research.agent_client import run_model_budget_exhausted
    local = AgentRuntimeClientError("local", status="budget_exhausted",
        failure_category="provider_turn_budget", run_budget={"remaining_usd": 2.7})
    assert not run_model_budget_exhausted(local)
    assert run_model_budget_exhausted(AgentRuntimeClientError("total",
        status="budget_exhausted", failure_category="run_model_budget"))


@pytest.mark.parametrize("category", ["missing_resume_session", "provider_or_tool_error"])
def test_failed_role_session_rehydrates_without_erasing_evidence_or_budget(tmp_path, category):
    client = client_for(tmp_path)
    from autosim.research.common import atomic_json
    key = "b" * 32
    root = client.output / "agent"
    atomic_json(root / "roles/fix.json", {
        "role": "fix", "run_id": client.run_id, "session_key": key})
    session = {"schema_version": 1, "status": "failed", "failure_category": category,
               "session_id": "missing-provider-conversation", "total_cost_usd": 0.17}
    atomic_json(root / f"sessions/{key}.json", session)
    assert client._resume_role("fix") is False
    assert client.total_budget_usd == 2.0
    assert json.loads((root / f"sessions/{key}.json").read_text()) == session
    assert json.loads((root / "context_rotation.json").read_text())["failure_category"] == category


def test_agent_client_uses_independent_resumable_sessions_per_role(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        role = str(kwargs["role"])
        roles = tmp_path / "run" / "agent" / "roles"
        roles.mkdir(parents=True, exist_ok=True)
        (roles / f"{role}.json").write_text(json.dumps({
            "role": role, "run_id": "research-test", "session_key": "a" * 32,
        }), encoding="utf-8")
        return {"status": "completed", "final_text": '{"do":"stop"}',
                "session_id": f"session-{role}", "model": "test-model",
                "total_cost_usd": 0.03, "usage": {"input_tokens": 12},
                "run_budget": {"remaining_usd": 1.97}}

    monkeypatch.setattr("autosim.research.agent_runtime.run_coding_agent", fake_run)
    client = client_for(tmp_path)
    first, meta = client.chat_with_metadata("system", "user", timeout=10)
    with client.as_role("ideator"):
        second, ideator_meta = client.chat_with_metadata("ideas", "context")
    third, _ = client.chat_with_metadata("next", "question")

    assert first == second == third == '{"do":"stop"}'
    assert [call["role"] for call in calls] == ["scheduler", "ideator", "scheduler"]
    assert [call["resume"] for call in calls] == [False, False, True]
    assert calls[0]["timeout"] == 10
    assert meta["runtime"] == "claude_code" and meta["total_cost_usd"] == 0.03
    assert ideator_meta["role"] == "ideator"
    assert client._role == "scheduler"


def test_agent_client_returns_runtime_trace_identity_and_forwards_decision_attempt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    turn_id = "a" * 32
    decision_attempt_id = "b" * 32
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"status": "completed", "final_text": '{"do":"read_the_checkout"}',
                "role": "scheduler", "turn_id": turn_id,
                "process_ref": f"agent/processes/{turn_id}.json",
                "decision_attempt_id": decision_attempt_id, "event_count": 7}

    monkeypatch.setattr("autosim.research.agent_runtime.run_coding_agent", fake_run)
    client = client_for(tmp_path)
    text, metadata = client.chat_with_metadata(
        "system", "user", decision_attempt_id=decision_attempt_id)

    assert text == '{"do":"read_the_checkout"}'
    assert calls[0]["decision_attempt_id"] == decision_attempt_id
    assert metadata["turn_id"] == turn_id
    assert metadata["process_ref"] == f"agent/processes/{turn_id}.json"
    assert metadata["decision_attempt_id"] == decision_attempt_id
    assert metadata["event_count"] == 7


def test_agent_client_consumes_execution_not_display_projection(tmp_path, monkeypatch):
    original = json.dumps({'commands': ['except Exception as e:\n    raise\n']})
    monkeypatch.setattr('autosim.research.agent_runtime.run_coding_agent',
        lambda **_: {'status': 'completed', 'execution_text': original,
                     'final_text': '[DISPLAY ONLY]'})
    client = client_for(tmp_path)
    text, metadata = client.chat_with_metadata('system', 'user')
    assert text == original
    assert 'execution_text' not in metadata
    assert 'final_text' not in metadata


def test_agent_client_freezes_runtime_identity_and_budgets(tmp_path: Path) -> None:
    client_for(tmp_path)
    output = tmp_path / "run"
    workspace = output / "checkout"
    with pytest.raises(ValueError, match="inputs changed"):
        RoleAwareAgentClient(workspace=workspace, output=output,
                             run_id="research-test", turn_budget_usd=0.25,
                             total_budget_usd=3.0, timeout=90)


def test_global_context_is_refreshed_across_role_sessions(tmp_path, monkeypatch):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return {"status": "completed", "final_text": "{}"}

    monkeypatch.setattr("autosim.research.agent_runtime.run_coding_agent", run)
    client = client_for(tmp_path)
    client.set_research_context({"state_revision": 7, "main_agent": {"objective": "trace loader"}})
    with client.as_role("resource"):
        client.chat_with_metadata("inspect", "source")
    assert '"state_revision": 7' in calls[-1]["prompt"]
    client.set_research_context({"state_revision": 8})
    with client.as_role("fix"):
        client.chat_with_metadata("repair", "error")
    assert '"state_revision": 8' in calls[-1]["prompt"]
    assert "trace loader" not in calls[-1]["prompt"]
    client.chat_with_metadata("decision", "already contains state", include_research_context=False,
                              read_only=True, auto_skills=False)
    assert "Shared research context" not in calls[-1]["prompt"]
    assert calls[-1]["read_only"] is True
    assert calls[-1]["auto_skills"] is False


def test_agent_client_rejects_noncompleted_or_empty_turn(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("autosim.research.agent_runtime.run_coding_agent",
                        lambda **_: {"status": "interrupted", "final_text": "",
                                     "failure_category": "wall_timeout",
                                     "run_budget": {"remaining_usd": 1.0}})
    client = client_for(tmp_path)
    with pytest.raises(AgentRuntimeClientError, match="interrupted") as interrupted:
        client.chat_with_metadata("system", "user")
    assert interrupted.value.status == "interrupted"
    assert interrupted.value.failure_category == "wall_timeout"
    assert interrupted.value.run_budget == {"remaining_usd": 1.0}

    monkeypatch.setattr("autosim.research.agent_runtime.run_coding_agent",
                        lambda **_: {"status": "completed", "final_text": "  "})
    with pytest.raises(AgentRuntimeClientError, match="without a final response"):
        client.chat_with_metadata("system", "user")


def test_agent_client_scopes_recorder_as_read_only_model_role(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    with client.as_role("recorder"):
        assert client._role == "recorder"
    assert client._role == "scheduler"
