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


def test_agent_client_refuses_pro_before_freezing_a_run(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    with pytest.raises(ValueError, match="must use deepseek-flash"):
        client_for(tmp_path)
    assert not (tmp_path / "run/agent/runtime_spec.json").exists()


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


def test_toolless_review_never_receives_inaccessible_context_pages(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr('autosim.research.agent_runtime.run_coding_agent',
        lambda **kwargs: calls.append(kwargs) or {'status': 'completed', 'final_text': '{}'})
    client = client_for(tmp_path)
    body = 'source-' * 4000
    client.set_research_context({'source_excerpt': body})
    with client.as_role('supervisor'):
        _, metadata = client.chat_with_metadata('review', json.dumps({'source_excerpt': body}),
            decision_only=True, read_only=True)
    assert body in calls[-1]['prompt']
    assert metadata['prompt_efficiency']['request_paged_leaves'] == 0
    assert 'context_paged' not in calls[-1]['prompt']
    with client.as_role('scheduler'):
        _, metadata = client.chat_with_metadata('inspect', json.dumps({'source_excerpt': body}))
    assert body not in calls[-1]['prompt']
    assert 'read_evidence' in calls[-1]['prompt']
    assert metadata['prompt_efficiency']['request_paged_leaves'] == 1
    assert metadata['prompt_efficiency']['deduplicated_fields'] == 1


def test_common_role_contract_precedes_different_task_instructions(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr('autosim.research.agent_runtime.run_coding_agent',
        lambda **kwargs: calls.append(kwargs) or {'status': 'completed', 'final_text': '{}'})
    client = client_for(tmp_path)
    client.chat_with_metadata('first task instructions', '{}')
    client.chat_with_metadata('different task instructions', '{}')
    assert calls[0]['prompt'].split('## Task instructions')[0] == calls[1]['prompt'].split('## Task instructions')[0]


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


def test_warm_session_is_preserved_but_capability_change_still_rotates(tmp_path):
    from autosim.research.common import atomic_json
    from autosim.research.agent_runtime import capability_identity
    from autosim.research.agent_roles import role_profile
    client = client_for(tmp_path)
    root = client.output / 'agent'
    key = 'c' * 32
    atomic_json(root / 'roles/init.json', {'role': 'init', 'run_id': client.run_id, 'session_key': key})
    session = {'schema_version': 1, 'status': 'completed', 'pricing_ref': 'agent/pricing/previous.json',
        'capability_identity': capability_identity(role_profile('init'))}
    from autosim.research.runtime_freshness import controller_identity
    session['controller_runtime_id'] = controller_identity()
    atomic_json(root / f'sessions/{key}.json', session)
    atomic_json(root / 'pricing/previous.json', {'gateway': {'receipts': [
        {'status': 'priced', 'usage': {'input_hit_tokens': 95000, 'input_miss_tokens': 5000,
                                     'output_tokens': 2000}}]}})
    assert client._resume_role('init') is True
    assert client.total_budget_usd == 2
    session['capability_identity'] = 'stale'
    atomic_json(root / f'sessions/{key}.json', session)
    assert client._resume_role('init') is False
    assert 'capability/schema changed' in json.loads((root / 'context_rotation.json').read_text())['reason']


@pytest.mark.parametrize('previous', [None, 'old-controller-runtime'])
def test_controller_deployment_rotates_role_context_without_erasing_memory_or_budget(tmp_path, previous):
    from autosim.research.common import atomic_json, digest
    from autosim.research.agent_runtime import capability_identity
    from autosim.research.agent_roles import role_profile
    client = client_for(tmp_path)
    root, key = client.output / 'agent', 'b'*32
    atomic_json(root / 'roles/scheduler.json', {'role':'scheduler', 'run_id':client.run_id, 'session_key':key})
    path = root / f'sessions/{key}.json'
    session = {'schema_version':1, 'status':'completed',
               'capability_identity':capability_identity(role_profile('scheduler'))}
    if previous is not None:
        session['controller_runtime_id'] = previous
    atomic_json(path, session)
    before = digest(path)
    pointer = digest(root / 'roles/scheduler.json')
    assert client._resume_role('scheduler') is False
    rotation = json.loads((root / 'context_rotation.json').read_text())
    assert 'obsolete operation assumptions' in rotation['reason']
    assert digest(path) == before and digest(root / 'roles/scheduler.json') == pointer
    assert client.total_budget_usd == 2


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
        import hashlib
        from autosim.research.agent_runtime import capability_identity
        from autosim.research.agent_roles import role_profile
        from autosim.research.runtime_freshness import controller_identity
        key = hashlib.sha256(role.encode()).hexdigest()[:32]
        (roles / f"{role}.json").write_text(json.dumps({
            "role": role, "run_id": "research-test", "session_key": key,
        }), encoding="utf-8")
        sessions = roles.parent / 'sessions'
        sessions.mkdir(exist_ok=True)
        (sessions / f'{key}.json').write_text(json.dumps({'schema_version':1,
            'status':'completed', 'controller_runtime_id':controller_identity(),
            'capability_identity':capability_identity(role_profile(role))}), encoding='utf-8')
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
    monkeypatch.setattr(client,'_resume_role',lambda role:pytest.fail('bounded decision resumed unrelated memory'))
    client.chat_with_metadata('audit','only supplied evidence',include_research_context=False,
                             resume_session=False,decision_only=True,read_only=True,auto_skills=False)
    assert calls[-1]['resume'] is False and calls[-1]['decision_only'] is True


def test_generation_controls_are_explicit_only_for_bounded_reviews(tmp_path, monkeypatch):
    calls=[]
    monkeypatch.setattr('autosim.research.agent_runtime.run_coding_agent',
        lambda **kwargs: (calls.append(kwargs) or {'status':'completed','final_text':'{}'}))
    client=client_for(tmp_path)
    client.chat_with_metadata('main','inspect',max_tokens=1800,thinking='disabled')
    assert calls[-1]['generation_policy'] is None
    with client.as_role('supervisor'):
        client.chat_with_metadata('review','source',decision_only=True,
                                 max_tokens=1800,thinking='disabled')
    assert calls[-1]['generation_policy']=={'max_output_tokens':1800,'thinking':'disabled'}


def test_explicit_auxiliary_bound_preserves_tools_without_limiting_main_by_default(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr('autosim.research.agent_runtime.run_coding_agent',
        lambda **kw:calls.append(kw) or {'status':'completed','final_text':'{}'})
    client=client_for(tmp_path)
    with client.as_role('init'):
        client.chat_with_metadata('argv','bounded engineering response',bounded_generation=True,
            max_tokens=3000,thinking='disabled')
    assert calls[-1]['generation_policy']=={'max_output_tokens':3000,'thinking':'disabled'}
    assert calls[-1]['decision_only'] is False
    assert 'mcp__autosim_exec__inspect_native_environment' in calls[-1]['prompt']
    client.chat_with_metadata('main','research decision',max_tokens=6000)
    assert calls[-1]['generation_policy'] is None


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
    with pytest.raises(AgentRuntimeClientError, match="without a final response") as empty:
        client.chat_with_metadata("system", "user")
    assert empty.value.failure_category == 'missing_final_response'
    assert empty.value.status == 'failed'
    assert empty.value.runtime_failure['evidence_id']
    assert empty.value.runtime_failure['native_operation_status'] == 'not_inferred'
    assert (client.output / empty.value.runtime_failure['evidence_ref']).is_file()


def test_empty_final_session_is_rotated_without_changing_old_receipt_or_budget(tmp_path):
    from autosim.research.common import atomic_json
    client=client_for(tmp_path)
    root=client.output/'agent'; key='b'*32
    atomic_json(root/'roles/scheduler.json',{'role':'scheduler','run_id':client.run_id,'session_key':key})
    path=root/'sessions'/f'{key}.json'
    atomic_json(path,{'schema_version':1,'failure_category':'missing_final_response'})
    original=path.read_bytes()
    assert client._resume_role('scheduler') is False
    assert path.read_bytes()==original and client.total_budget_usd==2
    assert json.loads((root/'context_rotation.json').read_text())['failure_category']=='missing_final_response'


def test_agent_client_scopes_recorder_as_read_only_model_role(tmp_path: Path) -> None:
    client = client_for(tmp_path)
    with client.as_role("recorder"):
        assert client._role == "recorder"
    assert client._role == "scheduler"
