import json
import time
import uuid
import shutil
from pathlib import Path

import pytest

from autosim.research.agent_runtime import (_McpExecutor, execute_agent_command,
                                            _turn_status,
                                            _validate_agent_prompt,
                                            _MAX_CODING_AGENT_PROMPT_BYTES,
                                            AgentPromptTooLarge,
                                            _role_cli_args,
                                            _project_claude_event,
                                            _visible_text_delta,
                                            _complete_progress_words,
                                            _claude_configuration,
                                            handle_mcp_message,
                                            network_deny_filter_bytes,
                                            refresh_agent_document)
from autosim.research.process_executor import ProcessAttempt
from autosim.research.agent_roles import role_profile
from autosim.research.agent_runtime import _role_skill_context


def test_missing_resume_session_requires_precise_cli_preexecution_evidence():
    from autosim.research.agent_runtime import _missing_resume_session
    session_id = str(uuid.uuid4())
    attempt = ProcessAttempt(launched=True, returncode=1,
                             stderr=f"No conversation found with session ID: {session_id}\n")
    result = {"is_error": True, "num_turns": 0, "session_id": session_id}
    assert _missing_resume_session(attempt, session_id, result, resume=True)
    assert not _missing_resume_session(attempt, session_id, result, resume=False)
    assert not _missing_resume_session(attempt, str(uuid.uuid4()), result, resume=True)
    assert not _missing_resume_session(attempt, session_id, {**result, "num_turns": 1}, resume=True)
    assert not _missing_resume_session(attempt, session_id, {**result, "is_error": False}, resume=True)
    attempt = ProcessAttempt(launched=True, returncode=1,
                             stderr="tool says: " + attempt.stderr)
    assert not _missing_resume_session(attempt, session_id, result, resume=True)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    output = tmp_path / "run"
    workspace = output / "checkout"
    workspace.mkdir(parents=True)
    (output / "agent_fixture.json").write_text(
        json.dumps({"schema_version": 1, "workspace": str(workspace)}), encoding="utf-8")
    return output, workspace


def test_network_filter_profile_fails_closed_for_unknown_architecture():
    assert network_deny_filter_bytes("x86_64")
    with pytest.raises(RuntimeError, match="not defined"):
        network_deny_filter_bytes("mips64")


def test_gateway_permission_failure_has_terminal_evidence_and_no_model_charge(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from autosim.research import agent_runtime
    output, workspace = _fixture(tmp_path)
    monkeypatch.setattr("autosim.llm_client.load_credential_file", lambda **_: {"source": "fixture"})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-secret")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")

    @contextmanager
    def denied(**kwargs):
        raise PermissionError(1, "Operation not permitted fixture-secret")
        yield

    monkeypatch.setattr(agent_runtime, "deepseek_turn_gateway", denied)
    monkeypatch.setattr(agent_runtime, "run_process_stream",
                        lambda *a, **k: pytest.fail("provider must not start"))
    result = agent_runtime.run_coding_agent(
        workspace=workspace, output=output, prompt="test", run_id="probe",
        max_budget_usd=.1, max_total_budget_usd=1, auto_skills=False,
        cli="/usr/bin/claude")
    assert result["status"] == "infrastructure_blocked"
    assert result["run_budget"]["spent_usd"] == 0
    assert result["run_budget"]["reserved_usd"] == 0
    receipt = json.loads((output / result["process_ref"]).read_text())
    assert receipt["status"] == "not_launched"
    assert receipt["startup_failure"]["errno"] == 1
    assert receipt["startup_failure"]["phase"] == "deepseek_gateway_start"
    assert "fixture-secret" not in json.dumps(receipt)
    assert "fixture-secret" not in (output / "agent/events.jsonl").read_text()
    assert json.loads((output / "agent/session.json").read_text())["status"] == "infrastructure_blocked"


def test_autosota_roles_get_narrow_tool_profiles_and_recorder_is_not_a_code_agent():
    scheduler = role_profile("scheduler")
    supervisor = role_profile("supervisor")
    resource = role_profile("resource")
    assert scheduler.can_edit_checkout and scheduler.can_execute_diagnostics
    assert "Edit" in scheduler.builtin_tools and scheduler.mcp_tools
    assert not supervisor.can_edit_checkout and not supervisor.can_execute_diagnostics
    assert "Edit" not in supervisor.builtin_tools
    assert set(supervisor.mcp_tools) == {"mcp__autosim_exec__read_evidence",
        "mcp__autosim_exec__search_public_sources", "mcp__autosim_exec__read_public_source",
        "mcp__autosim_exec__inspect_workspace_resources"}
    assert not resource.can_edit_checkout and not resource.can_execute_diagnostics
    recorder = role_profile("recorder")
    assert recorder.can_run_agent
    assert not recorder.builtin_tools and not recorder.mcp_tools
    assert not recorder.can_edit_checkout and not recorder.can_execute_diagnostics


def test_read_only_supervisor_cli_cannot_receive_mcp_or_mutation_tools(tmp_path):
    output, workspace = _fixture(tmp_path)
    args = _role_cli_args(role_profile("supervisor"), workspace=workspace,
                          output=output, python="/usr/bin/python3", model="fixture",
                          max_budget_usd=0.1)
    tools_index = args.index("--tools")
    allowed_index = args.index("--allowedTools")
    strict_index = args.index("--strict-mcp-config")
    config = json.loads(args[args.index("--mcp-config") + 1])

    assert args[tools_index + 1] == "Read,Glob,Grep"
    assert "Edit" not in args[allowed_index:strict_index]
    assert "mcp__autosim_exec__run_command" not in args[allowed_index:strict_index]
    assert "mcp__autosim_exec__read_evidence" in args[allowed_index:strict_index]
    assert "--read-only" in config["mcpServers"]["autosim_exec"]["args"]
    assert "--restricted" in args


def test_role_skill_retrieval_is_global_metadata_first_and_advisory():
    metadata, prompt = _role_skill_context("scheduler", "inspect unfamiliar source and test a bug")
    assert metadata["status"] == "reference_only_not_enforced"
    assert metadata["skills"]
    assert all(row["scope"] == "general" for row in metadata["skills"])
    assert all(row["body_sha256"] and row["body_in_prompt"] for row in metadata["skills"])
    assert "advisory only" in prompt
    assert "/home/" not in prompt


def test_autosota_roles_get_narrow_tool_profiles_and_recorder_is_not_a_code_agent():
    scheduler = role_profile("scheduler")
    supervisor = role_profile("supervisor")
    resource = role_profile("resource")
    assert scheduler.can_edit_checkout and scheduler.can_execute_diagnostics
    assert "Edit" in scheduler.builtin_tools and scheduler.mcp_tools
    assert not supervisor.can_edit_checkout and not supervisor.can_execute_diagnostics
    assert "Edit" not in supervisor.builtin_tools
    assert set(supervisor.mcp_tools) == {"mcp__autosim_exec__read_evidence",
        "mcp__autosim_exec__search_public_sources", "mcp__autosim_exec__read_public_source",
        "mcp__autosim_exec__inspect_workspace_resources"}
    assert not resource.can_edit_checkout and not resource.can_execute_diagnostics
    recorder = role_profile("recorder")
    assert recorder.can_run_agent
    assert not recorder.builtin_tools and not recorder.mcp_tools
    assert not recorder.can_edit_checkout and not recorder.can_execute_diagnostics


def test_role_skill_retrieval_is_global_metadata_first_and_advisory():
    metadata, prompt = _role_skill_context("scheduler", "inspect unfamiliar source and test a bug")
    assert metadata["status"] == "reference_only_not_enforced"
    assert metadata["skills"]
    assert all(row["scope"] == "general" for row in metadata["skills"])
    assert all(row["body_sha256"] and row["body_in_prompt"] for row in metadata["skills"])
    assert "advisory only" in prompt
    assert "/home/" not in prompt


def test_provider_budget_error_is_resumable_not_misreported_as_repository_failure():
    attempt = ProcessAttempt(True, 1)
    status = _turn_status(attempt, {"subtype": "error_max_budget_usd", "is_error": True}, [])
    assert status == ("budget_exhausted", "provider_turn_budget", False)


def test_coding_agent_sends_large_prompt_via_stdin(tmp_path, monkeypatch):
    from autosim.research import agent_runtime

    output, workspace = _fixture(tmp_path)
    captured = {}
    monkeypatch.setattr("autosim.llm_client.load_credential_file",
                        lambda **_: {"source": "fixture", "model": "fixture-model"})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-secret")
    monkeypatch.setenv("DEEPSEEK_MODEL", "fixture-model")
    monkeypatch.setattr(agent_runtime, "_role_skill_context",
                        lambda *_: ({"skills": []}, ""))

    def fake_stream(command, **kwargs):
        captured["command"] = command
        captured["input_bytes"] = kwargs["input_bytes"]
        kwargs["on_stdout_line"](json.dumps({
            "type": "result", "total_cost_usd": 0.01, "duration_ms": 12,
            "usage": {"input_tokens": 10000}, "result": "bounded prompt accepted"}))
        return ProcessAttempt(launched=True, returncode=0, stdout="", stderr="",
                              containment_mode="fixture")

    monkeypatch.setattr(agent_runtime, "run_process_stream", fake_stream)
    prompt = "source evidence " * 9000  # Above Linux's per-argument limit.
    _validate_agent_prompt(prompt)
    result = agent_runtime.run_coding_agent(
        workspace=workspace, output=output, prompt=prompt,
        run_id="large-prompt-test", max_budget_usd=0.05,
        max_total_budget_usd=0.05, cli="/usr/bin/claude", role="resource")

    assert result["status"] == "completed"
    assert prompt not in captured["command"]
    assert captured["input_bytes"] == prompt.encode("utf-8")
    assert len(captured["input_bytes"]) > 128 * 1024


def test_executor_text_is_separate_from_redacted_display(tmp_path, monkeypatch):
    from autosim.research import agent_runtime
    output, workspace = _fixture(tmp_path)
    monkeypatch.setattr("autosim.llm_client.load_credential_file",
                        lambda **_: {"source": "fixture", "model": "fixture-model"})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-secret")
    monkeypatch.setenv("DEEPSEEK_MODEL", "fixture-model")
    monkeypatch.setattr(agent_runtime, "_role_skill_context", lambda *_: ({"skills": []}, ""))
    code = 'except BaseException as e:\n    raise\n'
    original = json.dumps({'commands': [code], 'private': '/home/operator/private'})

    def fake_stream(command, **kwargs):
        kwargs['on_stdout_line'](json.dumps({'type': 'result', 'total_cost_usd': 0.01,
            'duration_ms': 12, 'usage': {'input_tokens': 10}, 'result': original}))
        return ProcessAttempt(launched=True, returncode=0, stdout='', stderr='',
                              containment_mode='fixture')

    monkeypatch.setattr(agent_runtime, 'run_process_stream', fake_stream)
    result = agent_runtime.run_coding_agent(workspace=workspace, output=output,
        prompt='Return a proposal', run_id='separation-test', max_budget_usd=0.05,
        max_total_budget_usd=0.05, cli='/usr/bin/claude', role='resource')
    assert result['status'] == 'completed'
    assert result['execution_text'] == original
    assert json.loads(result['final_text'])['commands'] == [code]
    assert '/home/operator' not in result['final_text']
    # Persistent stream projections never receive the raw execution field.
    assert 'execution_text' not in (output/'agent/events.jsonl').read_text()


@pytest.mark.parametrize('truncated', [False, True])
def test_decision_only_json_has_no_tools_and_requires_complete_terminal(tmp_path, monkeypatch, truncated):
    from autosim.research import agent_runtime
    output, workspace = _fixture(tmp_path)
    monkeypatch.setattr('autosim.llm_client.load_credential_file',
                        lambda **_: {'source':'fixture','model':'fixture-model'})
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'fixture-secret')
    monkeypatch.setenv('DEEPSEEK_MODEL', 'fixture-model')
    captured={}
    response=json.dumps({'type':'result','result':'{"approved":true,"reason":"quoted source"}',
                         'total_cost_usd':.01,'usage':{'input_tokens':10}},indent=2)
    def fake_stream(command, **kwargs):
        captured.update(command=command,options=kwargs)
        assert kwargs['on_stdout_line'] is None
        return ProcessAttempt(launched=True,returncode=0,stdout=response[:-5] if truncated else response,
                              stderr='',containment_mode='fixture')
    monkeypatch.setattr(agent_runtime,'run_process_stream',fake_stream)
    result=agent_runtime.run_coding_agent(workspace=workspace,output=output,prompt='review source',
        run_id='decision-test',max_budget_usd=.05,max_total_budget_usd=.05,
        cli='/usr/bin/claude',role='objective',read_only=True,decision_only=True,output_format='json')
    command=captured['command']
    assert command[command.index('--tools')+1]==''
    assert '--allowedTools' not in command
    assert command[command.index('--output-format')+1]=='json'
    assert '--verbose' not in command
    assert captured['options']['decouple_callbacks']
    if truncated:
        assert result['status']=='failed' and result['failure_category']=='stream_protocol'
        assert result['execution_text']==''
    else:
        assert result['status']=='completed'
        assert json.loads(result['execution_text'])['approved'] is True


def test_deepseek_turn_uses_gateway_usage_not_claude_cli_estimate(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from autosim.research import agent_runtime

    output, workspace = _fixture(tmp_path)
    monkeypatch.setattr("autosim.llm_client.load_credential_file",
                        lambda **_: {"source": "fixture", "model": "deepseek-flash"})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-secret")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    monkeypatch.setattr(agent_runtime, "_role_skill_context", lambda *_: ({"skills": []}, ""))
    captured = {}

    class FakeGateway:
        base_url = "http://127.0.0.1:54321"
        local_token = "local-fixture-token"
        gate = None

        def __init__(self):
            self.gate = self

        def snapshot(self):
            return {"cost_usd": .002, "unknown": False, "requests": 1,
                    "denied_count": 0, "held_usd": 0, "receipts": []}

    @contextmanager
    def fake_gateway(**kwargs):
        captured["gateway_args"] = kwargs
        yield FakeGateway()

    def fake_stream(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        kwargs["on_stdout_line"](json.dumps({
            "type": "result", "total_cost_usd": 0.20, "duration_ms": 12,
            "usage": {"input_tokens": 100}, "result": "done"}))
        return ProcessAttempt(launched=True, returncode=0, stdout="", stderr="",
                              containment_mode="fixture")

    monkeypatch.setattr(agent_runtime, "deepseek_turn_gateway", fake_gateway)
    monkeypatch.setattr(agent_runtime, "run_process_stream", fake_stream)
    result = agent_runtime.run_coding_agent(
        workspace=workspace, output=output, prompt="hello", run_id="official-pricing",
        max_budget_usd=.05, max_total_budget_usd=.50,
        cli="/usr/bin/claude", role="resource")
    assert result["status"] == "completed"
    assert result["total_cost_usd"] == pytest.approx(.002)
    assert result["cli_estimate_usd"] == pytest.approx(.20)
    assert result["run_budget"]["spent_usd"] == pytest.approx(.002)
    assert result["run_budget"]["cost_basis"] == "deepseek_official_estimate_v1"
    assert "--max-budget-usd" not in captured["command"]
    assert captured["environment"]["ANTHROPIC_BASE_URL"] == FakeGateway.base_url
    assert captured["environment"]["ANTHROPIC_AUTH_TOKEN"] == FakeGateway.local_token
    assert captured["gateway_args"]["upstream_key"] == "fixture-secret"
    receipt = json.loads((output / result["pricing_ref"]).read_text())
    assert receipt["official_estimate_usd"] == pytest.approx(.002)
    assert receipt["cli_estimate_usd"] == pytest.approx(.20)
    document = (output / "RUN.md").read_text()
    assert "Run model budget ledger is unreadable" not in document
    assert "$0.002000" in document


def test_catalog_driven_main_agent_turn_does_not_auto_inject_skill_bodies(tmp_path,
                                                                            monkeypatch):
    from autosim.research import agent_runtime

    output, workspace = _fixture(tmp_path)
    monkeypatch.setattr("autosim.llm_client.load_credential_file",
                        lambda **_: {"source": "fixture", "model": "fixture-model"})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-secret")
    monkeypatch.setenv("DEEPSEEK_MODEL", "fixture-model")
    monkeypatch.setattr(agent_runtime, "_role_skill_context",
                        lambda *_: pytest.fail("automatic skill retrieval was invoked"))
    captured = {}

    def fake_stream(command, **kwargs):
        captured["prompt"] = kwargs["input_bytes"].decode("utf-8")
        kwargs["on_stdout_line"](json.dumps({
            "type": "result", "total_cost_usd": 0.01,
            "result": '{"skill_reads":[]}'}))
        return ProcessAttempt(launched=True, returncode=0, stdout="", stderr="",
                              containment_mode="fixture")

    monkeypatch.setattr(agent_runtime, "run_process_stream", fake_stream)
    result = agent_runtime.run_coding_agent(
        workspace=workspace, output=output, prompt="short skill catalog",
        run_id="catalog-test", max_budget_usd=0.05, max_total_budget_usd=0.05,
        cli="/usr/bin/claude", role="scheduler", auto_skills=False)

    assert result["status"] == "completed"
    assert captured["prompt"] == "short skill catalog"
    session = json.loads((output / "agent" / "session.json").read_text(encoding="utf-8"))
    assert session["skill_selection"] == {
        "status": "main_agent_catalog_selection", "skills": []}


def test_coding_agent_input_limit_is_independent_of_os_argument_limit():
    assert _MAX_CODING_AGENT_PROMPT_BYTES == 4 * 1024 * 1024
    _validate_agent_prompt("x" * _MAX_CODING_AGENT_PROMPT_BYTES)
    with pytest.raises(AgentPromptTooLarge, match="configured 4 MiB"):
        _validate_agent_prompt("x" * (_MAX_CODING_AGENT_PROMPT_BYTES + 1))


def test_stream_projection_includes_only_visible_text_not_thinking_or_tool_json():
    assert _visible_text_delta({"type": "stream_event", "event": {
        "type": "content_block_delta", "delta": {"type": "text_delta", "text": "visible"}}}) == "visible"
    assert _visible_text_delta({"type": "stream_event", "event": {
        "type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "private"}}}) == ""
    assert _visible_text_delta({"type": "stream_event", "event": {
        "type": "content_block_delta", "delta": {"type": "input_json_delta",
                                                       "partial_json": "secret"}}}) == ""


def test_live_progress_is_flushed_at_word_boundaries():
    text = ("visible progress sentence words " * 8) + "unfinished"
    chunk, suffix = _complete_progress_words(text, limit=240)
    assert chunk
    assert chunk + " " + suffix == text
    assert chunk[-1].isalpha()
    assert suffix[0].isalpha()
    assert not chunk.endswith("unfinishe")


def test_coding_agent_turn_uses_and_settles_persistent_run_budget(tmp_path, monkeypatch):
    from autosim.research import agent_runtime

    output, workspace = _fixture(tmp_path)
    (workspace / "answer.py").write_text("print(42)\n", encoding="utf-8")
    captured = {}
    monkeypatch.setattr("autosim.llm_client.load_credential_file",
                        lambda **_: {"source": "fixture", "model": "fixture-model"})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-secret")
    monkeypatch.setenv("DEEPSEEK_MODEL", "fixture-model")
    monkeypatch.setattr(agent_runtime, "_role_skill_context", lambda *_: ({"skills": []}, ""))

    def fake_stream(command, **kwargs):
        captured["command"] = command
        kwargs["on_stdout_line"](json.dumps({
            "type": "result", "total_cost_usd": 0.03, "duration_ms": 12,
            "usage": {"input_tokens": 10}, "result": "done"}))
        return ProcessAttempt(launched=True, returncode=0, stdout="", stderr="",
                              containment_mode="fixture")

    monkeypatch.setattr(agent_runtime, "run_process_stream", fake_stream)
    decision_attempt_id = "c" * 32
    result = agent_runtime.run_coding_agent(
        workspace=workspace, output=output, prompt="inspect the synthetic fixture",
        run_id="budget-test", max_budget_usd=0.08, max_total_budget_usd=0.10,
        cli="/usr/bin/claude", role="scheduler",
        decision_attempt_id=decision_attempt_id)

    args = captured["command"]
    assert args[args.index("--max-budget-usd") + 1] == "0.08"
    assert result["status"] == "completed"
    assert result["turn_id"] == result["process_ref"].split("/")[-1][:-5]
    assert result["decision_attempt_id"] == decision_attempt_id
    assert result["event_count"] >= 1
    turn_events = [json.loads(line) for line in
                   (output / "agent" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert turn_events
    assert all(row["turn_id"] == result["turn_id"] for row in turn_events)
    assert all(row["process_ref"] == result["process_ref"] for row in turn_events)
    assert all(row["decision_attempt_id"] == decision_attempt_id for row in turn_events)
    assert turn_events[-1]["type"] == "turn_finished"
    assert result["run_budget"]["spent_usd"] == pytest.approx(0.03)
    assert result["run_budget"]["remaining_usd"] == pytest.approx(0.07)
    session = json.loads((output / "agent" / "session.json").read_text(encoding="utf-8"))
    assert session["budget_reservation_id"]
    assert session["run_budget"] == result["run_budget"]

    # A second role inherits the same run ceiling. If the remaining amount cannot reserve
    # its full requested turn, refuse before launching rather than pass a partial provider cap.
    second = agent_runtime.run_coding_agent(
        workspace=workspace, output=output, prompt="review the synthetic result",
        run_id="budget-test", max_budget_usd=0.08, max_total_budget_usd=None,
        cli="/usr/bin/claude", role="monitor")
    assert second["status"] == "budget_exhausted"
    assert second["failure_category"] == "run_model_budget"
    assert captured["command"] == args
    assert second["run_budget"]["limit_usd"] == pytest.approx(0.10)
    assert second["run_budget"]["spent_usd"] == pytest.approx(0.03)
    assert second["run_budget"]["remaining_usd"] == pytest.approx(0.07)

    exhausted = agent_runtime.run_coding_agent(
        workspace=workspace, output=output, prompt="audit the final report",
        run_id="budget-test", max_budget_usd=0.08, max_total_budget_usd=None,
        cli="/usr/bin/claude", role="objective")
    assert exhausted["status"] == "budget_exhausted"
    assert exhausted["run_budget"]["spent_usd"] == pytest.approx(0.03)

    with pytest.raises(agent_runtime.AgentRuntimeError, match="silently change"):
        agent_runtime.run_coding_agent(
            workspace=workspace, output=output, prompt="another role",
            run_id="budget-test", max_budget_usd=0.08,
            max_total_budget_usd=0.05, cli="/usr/bin/claude", role="objective")


def test_role_handoff_keeps_independent_sessions_and_resumes_the_selected_role(
        tmp_path, monkeypatch):
    from autosim.research import agent_runtime
    from autosim.research.research_state import ResearchStateStore
    from types import SimpleNamespace

    output, workspace = _fixture(tmp_path)
    monkeypatch.setattr("autosim.llm_client.load_credential_file",
                        lambda **_: {"source": "fixture", "model": "fixture-model"})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-secret")
    monkeypatch.setenv("DEEPSEEK_MODEL", "fixture-model")
    monkeypatch.setattr(agent_runtime, "_role_skill_context", lambda *_: ({"skills": []}, ""))
    calls = []
    monkeypatch.setattr(agent_runtime, "capture_process_identity",
                        lambda process, *, run_id, attempt_id, argv: {
                            "pid": process.pid, "pgid": process.pid, "start_ticks": 1,
                            "boot_id": "fixture-boot", "command_sha256": "a" * 64,
                            "run_id": run_id, "attempt_id": attempt_id,
                            "argv_sha256": "b" * 64})

    def fake_stream(command, **kwargs):
        calls.append(list(command))
        kwargs["on_start"](SimpleNamespace(pid=40000 + len(calls), args=command))
        kwargs["on_stdout_line"](json.dumps({
            "type": "result", "total_cost_usd": 0.01, "duration_ms": 12,
            "usage": {"input_tokens": 10}, "result": "handoff recorded"}))
        return ProcessAttempt(launched=True, returncode=0, stdout="", stderr="",
                              containment_mode="fixture")

    monkeypatch.setattr(agent_runtime, "run_process_stream", fake_stream)
    common = {"workspace": workspace, "output": output, "run_id": "handoff-test",
              "max_budget_usd": 0.05, "max_total_budget_usd": 0.20,
              "cli": "/usr/bin/claude"}
    scheduler = agent_runtime.run_coding_agent(
        **common, prompt="inspect and propose the next action", role="scheduler")
    monitor = agent_runtime.run_coding_agent(
        **common, prompt="independently assess progress", role="monitor")
    scheduler_resume = agent_runtime.run_coding_agent(
        **common, prompt="continue from the scheduler context", role="scheduler",
        resume=True)

    scheduler_key = scheduler["agent_session_key"]
    monitor_key = monitor["agent_session_key"]
    assert scheduler_key != monitor_key
    assert (output / "agent" / "sessions" / f"{scheduler_key}.json").is_file()
    assert (output / "agent" / "sessions" / f"{monitor_key}.json").is_file()
    scheduler_pointer = json.loads(
        (output / "agent" / "roles" / "scheduler.json").read_text(encoding="utf-8"))
    monitor_pointer = json.loads(
        (output / "agent" / "roles" / "monitor.json").read_text(encoding="utf-8"))
    assert scheduler_pointer["session_key"] == scheduler_key
    assert monitor_pointer["session_key"] == monitor_key
    assert scheduler_resume["agent_session_key"] == scheduler_key
    assert scheduler_resume["session_id"] == scheduler["session_id"]
    resumed_args = calls[-1]
    assert resumed_args[resumed_args.index("--resume") + 1] == scheduler["session_id"]
    latest_scheduler = json.loads((output / "agent" / "sessions" /
                                   f"{scheduler_key}.json").read_text(encoding="utf-8"))
    assert latest_scheduler["role"] == "scheduler"
    scheduler_receipt = json.loads((output / latest_scheduler["process_ref"]).read_text(
        encoding="utf-8"))
    assert scheduler_receipt["status"] == "completed"
    assert scheduler_receipt["process_identity"]["run_id"] == "handoff-test"
    state = ResearchStateStore(output, run_id="handoff-test", repository=workspace).load()
    assert state["phases"]["agent_runtime"]["current_action"] == {}
    assert state["phases"]["agent_runtime"]["process_identity"] is None
    assert json.loads((output / "agent" / "session.json").read_text(
        encoding="utf-8"))["role"] == "scheduler"


def test_only_one_agent_turn_can_own_a_run_at_a_time(tmp_path):
    from autosim.research import agent_runtime

    output, workspace = _fixture(tmp_path)
    with agent_runtime._agent_turn_lock(output):
        with pytest.raises(agent_runtime.AgentRuntimeError, match="currently owns"):
            agent_runtime.run_coding_agent(
                workspace=workspace, output=output, prompt="a competing turn",
                run_id="lock-test", cli="/usr/bin/claude")


def test_role_session_lookup_rejects_symlinked_session_directories(tmp_path):
    from autosim.research import agent_runtime

    output, _ = _fixture(tmp_path)
    target = tmp_path / "external-agent-state"
    target.mkdir()
    (output / "agent").symlink_to(target, target_is_directory=True)
    with pytest.raises(agent_runtime.AgentRuntimeError, match="directory is a symlink"):
        agent_runtime._safe_session(output)

    (output / "agent").unlink()
    (output / "agent").mkdir()
    (output / "agent" / "roles").symlink_to(target, target_is_directory=True)
    with pytest.raises(agent_runtime.AgentRuntimeError, match="directory is a symlink"):
        agent_runtime._latest_role_session(output, role="scheduler")


def test_formal_preparation_reconciles_an_interrupted_coding_agent_process(tmp_path):
    import subprocess

    from autosim.research.common import atomic_json
    from autosim.research.prepare import Preparation
    from autosim.research.process_executor import (capture_process_identity,
                                                   inspect_process_identity)
    from autosim.research.research_state import ResearchStateStore

    output, workspace = _fixture(tmp_path)
    process = subprocess.Popen(["/usr/bin/sleep", "60"], start_new_session=True)
    attempt_id = "a" * 32
    identity = capture_process_identity(
        process, run_id="formal-run", attempt_id=attempt_id, argv=process.args)
    assert inspect_process_identity(identity)["status"] == "matching_running"

    parent_action = {"step": "choose", "status": "running", "decision_id": "decision-1"}
    state = ResearchStateStore(output, run_id="formal-run", repository=workspace)
    state.record("preparation", "choose_started", status="running",
                 phase_state={"current_action": parent_action})
    process_ref = f"agent/processes/{attempt_id}.json"
    action = {"step": "coding_agent_turn", "status": "running",
              "role": "scheduler", "attempt_id": attempt_id,
              "process_ref": process_ref, "process_identity": identity,
              "parent_action": parent_action}
    atomic_json(output / process_ref, {
        "schema_version": 1, "run_id": "formal-run", "attempt_id": attempt_id,
        "status": "running", "process_identity": identity, "action": action})
    state.record("agent_runtime", "agent_process_started", status="running",
                 phase_state={"current_action": action,
                              "process_identity": identity,
                              "parent_action": parent_action})

    try:
        preparation = Preparation(repo=workspace, output=output, client=object(),
                                  scouting=tmp_path / "scouting", run_id="formal-run")
        assert preparation.recovery_after_interruption["parent_action"] == parent_action
        result = preparation.do("reconcile_interrupted_action")

        receipt = json.loads((output / process_ref).read_text(encoding="utf-8"))
        assert result["process_status"] == "terminated"
        assert result["outcome_known"] is False
        assert receipt["status"] == "interrupted"
        assert not preparation.recovery_after_interruption
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize('fault', ['', 'reserved', 'starting', 'uncapped', 'foreign_run'])
def test_unreserved_prelaunch_proof_requires_capped_admission_absence(tmp_path, fault):
    from autosim.research import agent_runtime
    from autosim.research.common import atomic_json
    output, workspace = _fixture(tmp_path)
    session = {'status':'running','run_id':'formal-run',
        'workspace_identity':str(workspace.resolve()),'session_id':str(uuid.uuid4()),
        'agent_session_key':'b'*32,'max_total_budget_usd':10,
        'budget_output':str(output),'budget_reservation_id':None}
    ledger = {'run_id':'formal-run','entries':[]}
    if fault == 'reserved': ledger['entries'] = [{'session_id':session['session_id']}]
    if fault == 'uncapped': session['max_total_budget_usd'] = None
    if fault == 'foreign_run': ledger['run_id'] = 'elsewhere'
    if fault == 'starting':
        atomic_json(output/'agent/processes/start.json', {'session_id':session['session_id']})
    atomic_json(output/'agent/cost_ledger.json', ledger)
    proof = agent_runtime.unreserved_prelaunch_proof(output=output, workspace=workspace,
        session=session, run_id='formal-run')
    assert bool(proof) == (fault == '')


def test_preparation_reconciles_fix_interrupted_before_admission(tmp_path):
    from autosim.research import agent_runtime
    from autosim.research.common import atomic_json, now
    from autosim.research.prepare import Preparation
    output, workspace = _fixture(tmp_path)
    action = {'step':'agent_fix','status':'running','started_at':now()}
    session = {'schema_version':1,'status':'running','run_id':'formal-run','role':'fix',
        'workspace_identity':str(workspace.resolve()),'session_id':str(uuid.uuid4()),
        'agent_session_key':'b'*32,'max_total_budget_usd':10,
        'budget_output':str(output),'budget_reservation_id':None,'started_at':time.time()}
    agent_runtime._persist_agent_session(output, session)
    atomic_json(output/'agent/cost_ledger.json', {'run_id':'formal-run','entries':[]})
    preparation = Preparation(repo=workspace, output=output, client=object(),
        scouting=tmp_path/'scouting', run_id='formal-run')
    preparation.recovery_after_interruption = {'action':action,
        'process_status':{'status':'not_recorded'}}
    result = preparation.do('reconcile_interrupted_action')
    assert result.get('process_status') == 'not_launched', result
    assert result['outcome_known'] is False
    assert not preparation.recovery_after_interruption
    assert agent_runtime._safe_session(output)['status'] == 'interrupted'


def _stale_role_projection(output, workspace):
    from autosim.research import agent_runtime
    from autosim.research.common import atomic_json
    attempt = 'e'*32
    identity = {'pid':12345,'run_id':'formal-run','attempt_id':attempt}
    session = {'schema_version':1,'status':'running','run_id':'formal-run','role':'init',
        'workspace_identity':str(workspace.resolve()),'session_id':str(uuid.uuid4()),
        'agent_session_key':'c'*32,'max_total_budget_usd':10,
        'process_attempt_id':attempt,'process_ref':f'agent/processes/{attempt}.json',
        'process_identity':identity}
    agent_runtime._persist_agent_session(output,session)
    atomic_json(output/session['process_ref'],{'run_id':'formal-run','attempt_id':attempt,
        'process_identity':identity,'status':'interrupted'})
    return session


def test_fresh_role_retires_stopped_prior_role_projection_preserving_costs(tmp_path,monkeypatch):
    from autosim.research import agent_runtime
    from autosim.research.common import atomic_json
    output,workspace = _fixture(tmp_path)
    _stale_role_projection(output,workspace)
    atomic_json(output/'agent/cost_ledger.json',{'run_id':'formal-run','entries':[
        {'status':'unknown','reserved_usd':2}]})
    original = (output/'agent/cost_ledger.json').read_bytes()
    monkeypatch.setattr(agent_runtime,'inspect_process_identity',lambda _: {'status':'not_running'})
    def fresh(**kw):
        assert agent_runtime._safe_session(output)['status'] == 'interrupted'
        return {'status':'completed'}
    monkeypatch.setattr(agent_runtime,'_run_coding_agent_turn',fresh)
    result = agent_runtime.run_coding_agent(workspace=workspace,output=output,prompt='next',
        run_id='formal-run',role='scheduler')
    assert result['status'] == 'completed'
    assert (output/'agent/cost_ledger.json').read_bytes() == original


@pytest.mark.parametrize('fault',['foreign_workspace','receipt_mismatch','lock_owned'])
def test_stale_role_reconciliation_fails_closed_before_any_signal(tmp_path,monkeypatch,fault):
    from autosim.research import agent_runtime
    from autosim.research.common import atomic_json
    output,workspace = _fixture(tmp_path)
    session = _stale_role_projection(output,workspace)
    if fault == 'foreign_workspace':
        session['workspace_identity'] = str(tmp_path/'foreign')
        agent_runtime._persist_agent_session(output,session)
    elif fault == 'receipt_mismatch':
        atomic_json(output/session['process_ref'],{'run_id':'different'})
    signals = []
    monkeypatch.setattr(agent_runtime,'terminate_recorded_process',lambda i:signals.append(i))
    monkeypatch.setattr(agent_runtime,'inspect_process_identity',lambda _: {'status':'matching_running'})
    monkeypatch.setattr(agent_runtime,'_run_coding_agent_turn',lambda **kw:pytest.fail('launched'))
    from contextlib import nullcontext
    lock = agent_runtime._agent_turn_lock(output) if fault == 'lock_owned' else nullcontext()
    with lock, pytest.raises(agent_runtime.AgentRuntimeError):
        agent_runtime.run_coding_agent(workspace=workspace,output=output,prompt='next',run_id='formal-run')
    assert not signals
    assert agent_runtime._safe_session(output)['status'] == 'running'


def test_agent_command_requires_checkout_bound_to_run(tmp_path):
    output, workspace = _fixture(tmp_path)
    marker = output / "agent_fixture.json"
    marker.write_text(json.dumps({"workspace": str(tmp_path / "elsewhere")}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="marker is missing"):
        execute_agent_command(arguments={"argv": ["true"]}, workspace=workspace,
                              output=output)


def test_credential_like_workspace_files_are_refused_before_model_or_tool_access(tmp_path):
    output, workspace = _fixture(tmp_path)
    (workspace / ".env.example").write_text("TOKEN=placeholder", encoding="utf-8")
    (workspace / ".env").write_text("TOKEN=synthetic-secret", encoding="utf-8")
    with pytest.raises(RuntimeError, match="credential-like"):
        _McpExecutor(workspace=workspace, output=output)


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is unavailable")
def test_agent_execution_is_checkout_scoped_and_network_denied(tmp_path):
    output, workspace = _fixture(tmp_path)
    outside = output / "must-not-exist"
    outside.write_text("host-only", encoding="utf-8")
    command = ("from pathlib import Path; import os, socket; "
               "Path('agent-created.txt').write_text('inside'); "
               f"print(Path({str(outside)!r}).exists()); "
               "print(repr(os.environ.get('CUDA_VISIBLE_DEVICES'))); "
               "print(repr(os.environ.get('NVIDIA_VISIBLE_DEVICES'))); "
               "print(repr(os.environ.get('HIP_VISIBLE_DEVICES'))); "
               "print(Path('/dev/nvidia0').exists()); socket.socket()")
    result = execute_agent_command(
        arguments={"argv": ["/usr/bin/python3", "-c", command], "timeout_seconds": 10},
        workspace=workspace, output=output)

    assert result["status"] == "failed"
    assert result["containment"] == "pid_namespace"
    assert "PermissionError" in result["stderr"]
    assert result["stdout"].splitlines() == ["False", "''", "'none'", "''", "False"]
    assert (workspace / "agent-created.txt").read_text(encoding="utf-8") == "inside"
    assert outside.read_text(encoding="utf-8") == "host-only"


def test_mcp_server_exposes_one_command_tool_and_returns_structured_receipt(tmp_path):
    output, workspace = _fixture(tmp_path)
    executor = _McpExecutor(workspace=workspace, output=output)
    listed = handle_mcp_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                                executor)
    assert listed["result"]["tools"][0]["name"] == "run_command"
    reply = handle_mcp_message({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                "params": {"name": "run_command", "arguments": {
                                    "argv": ["/usr/bin/python3", "-c", "print('mcp-ok')"]}}},
                               executor)

    payload = json.loads(reply["result"]["content"][0]["text"])
    assert reply["result"]["isError"] is False
    assert payload["status"] == "completed"
    assert payload["stdout"].strip() == "mcp-ok"


def test_agent_event_projection_keeps_command_output_but_not_file_contents():
    names = {}
    _project_claude_event({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "read-1", "name": "Read", "input": {"file_path": ".env"}},
        {"type": "tool_use", "id": "run-1", "name": "mcp__autosim_exec__run_command",
         "input": {"argv": ["python3", "test.py"]}}]}}, names)
    private_text = "TOKEN=do-not-persist-this"
    read_event = _project_claude_event({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "read-1", "content": private_text}]}}, names)
    command_event = _project_claude_event({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "run-1", "content": json.dumps({
            "status": "completed", "returncode": 0, "stdout": "42\\n"})}]}}, names)

    assert private_text not in json.dumps(read_event)
    assert "42" in command_event[0]["summary"]


def test_mcp_child_launch_uses_clean_environment_and_report_projects_events(tmp_path):
    output, workspace = _fixture(tmp_path)
    config = _claude_configuration(workspace=workspace, output=output,
                                   python="/usr/bin/python3")
    server = config["mcpServers"]["autosim_exec"]
    assert server["command"] == "/usr/bin/env"
    assert server["args"][0] == "-i"
    assert "ANTHROPIC_AUTH_TOKEN" not in server["args"]
    assert "DEEPSEEK_API_KEY" not in server["args"]
    read_only = _claude_configuration(workspace=workspace, output=output,
                                      python="/usr/bin/python3", include_executor=False)
    assert "--read-only" in read_only["mcpServers"]["autosim_exec"]["args"]
    read_only = _claude_configuration(workspace=workspace, output=output,
                                      python="/usr/bin/python3", include_executor=False)
    assert "--read-only" in read_only["mcpServers"]["autosim_exec"]["args"]

    agent = output / "agent"
    agent.mkdir()
    (agent / "session.json").write_text(json.dumps({
        "schema_version": 1, "status": "completed", "session_id": "fixture-session",
        "model": "fixture-model", "total_cost_usd": 0.01}), encoding="utf-8")
    (agent / "events.jsonl").write_text(json.dumps({
        "at": "2026-09-28T00:00:00Z", "type": "tool_result",
        "summary": "command completed · exit 0 · stdout `42` <!-- AUTOSIM_AGENT_ACTIVITY_END -->"}) + "\n",
        encoding="utf-8")

    report = refresh_agent_document(output).read_text(encoding="utf-8")
    assert "fixture-session" in report
    assert "`$0.010000`" in report
    assert "stdout &#96;42&#96;" in report
    assert report.count("<!-- AUTOSIM_AGENT_ACTIVITY_END -->") == 1
    assert "Raw chain-of-thought" in report
