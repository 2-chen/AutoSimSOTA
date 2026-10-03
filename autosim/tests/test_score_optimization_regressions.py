"""Regressions found in the real LIBERO Agent-loop trace; no model/GPU claims."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

from autosim.research.common import atomic_json, digest, object_digest
from autosim.research.evidence_store import read_attempt_evidence
from autosim.research.main_agent import validate_plan
from autosim.research.process_executor import run_process_stream
from autosim.research.workspace_snapshot import create


def bound_workspace(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("# native source")
    resources = source / "training-data"
    resources.mkdir()
    (resources / "scene.hdf5").write_bytes(b"toy not actual demo")
    (resources / "sub").mkdir()
    (resources / "sub" / "episode.hdf5").write_bytes(b"toy")
    output = tmp_path / "run"
    checkout = output / "checkout"
    create(source, checkout, tracked_only=False, max_bytes=1024,
           resources=[{"source": str(resources), "target": "data"}])
    return output, checkout, resources


def test_resource_metadata_inspection_sees_actual_binding_not_empty_placeholder(tmp_path):
    from autosim.research.resource_inventory import inspect
    output, checkout, source = bound_workspace(tmp_path)
    assert not list((checkout / "data").iterdir())
    result = inspect(output, target="data")
    assert {e["name"] for e in result["entries"]} == {"scene.hdf5", "sub"}
    assert str(source) not in json.dumps(result)
    assert "toy not actual demo" not in json.dumps(result)
    assert read_attempt_evidence(output, result["evidence_id"])["termination_reason"] == "metadata_only"
    nested = inspect(output, target="data", directory="sub")
    assert nested["entries"][0]["name"] == "episode.hdf5"


@pytest.mark.parametrize("directory", ["../source", "/etc", "sub/../../source"])
def test_resource_metadata_does_not_grant_arbitrary_path_access(tmp_path, directory):
    from autosim.research.resource_inventory import inspect
    output, _, _ = bound_workspace(tmp_path)
    with pytest.raises(ValueError):
        inspect(output, target="data", directory=directory)
    with pytest.raises(ValueError):
        inspect(output, target="not-authorized")


def test_resource_metadata_does_not_follow_links_or_expose_private_names(tmp_path):
    from autosim.research.resource_inventory import inspect
    output, _, source = bound_workspace(tmp_path)
    (source / ".env").write_text("SECRET=not-a-real-secret")
    (source / "outside").symlink_to(tmp_path)
    result = inspect(output, target="data")
    assert ".env" not in {e["name"] for e in result["entries"]}
    assert "SECRET" not in json.dumps(result)
    with pytest.raises(ValueError, match="symlink"):
        inspect(output, target="data", directory="outside")


def test_readonly_worker_can_inspect_parent_binding_without_execution_access(tmp_path):
    from autosim.research.agent_runtime import _McpExecutor, handle_mcp_message
    output, _, _ = bound_workspace(tmp_path)
    worker = output / "agent_workers" / ("a" * 32)
    checkout = worker / "checkout"
    checkout.mkdir(parents=True)
    atomic_json(worker / "workspace_snapshot.json", {"destination": str(checkout)})
    atomic_json(worker / "worker_parent.json", {"output": str(output), "read_only": True})
    executor = _McpExecutor(workspace=checkout, output=worker, allow_commands=False)
    result = handle_mcp_message({"id": 1, "method": "tools/call", "params": {
        "name": "inspect_workspace_resources", "arguments": {"target": "data"}}}, executor)
    assert not result["result"]["isError"]
    assert "scene.hdf5" in result["result"]["content"][0]["text"]
    tools = handle_mcp_message({"id": 2, "method": "tools/list"}, executor)
    assert "run_command" not in {t["name"] for t in tools["result"]["tools"]}


def test_early_reader_source_identity_does_not_require_derived_commands(tmp_path):
    from autosim.research.native_jobs import source_identity
    output, repo, _ = bound_workspace(tmp_path)
    assert not (output / "derived_stages.json").exists()
    first = source_identity(output, repo)
    assert first
    atomic_json(output / "derived_stages.json", {})
    assert source_identity(output, repo) != first


def test_planning_turn_respects_configured_runtime_instead_of_fixed_300s():
    from autosim.research.provision import _planning_timeout
    class CodingClient:
        supports_main_agent = True
        timeout = 900
    class LegacyClient:
        timeout = 900
    assert _planning_timeout(CodingClient()) == 900
    assert _planning_timeout(LegacyClient()) == 300


def test_actual_environment_plan_call_receives_configured_window(tmp_path, monkeypatch):
    from autosim.research import provision
    received = []
    class Client:
        supports_main_agent = True
        timeout = 900
        def chat_with_metadata(self, system, user, **kwargs):
            received.append(kwargs["timeout"])
            return json.dumps({"python": "3.10", "commands": [], "probes": [],
                               "reasoning": "regression fixture"}), {}
    monkeypatch.setattr(provision, "platform_facts", lambda: {})
    monkeypatch.setattr(provision, "plan_problems", lambda *_: [])
    provision.plan(Client(), tmp_path, manifests={}, attempts=1)
    assert received == [900]


def test_slow_reporting_callback_does_not_timeout_already_exited_child(tmp_path):
    seen = []
    def callback(line):
        seen.append(line)
        time.sleep(.15)
    attempt = run_process_stream([sys.executable, "-c", "print('completed', flush=True)"],
        cwd=tmp_path, env=os.environ.copy(), timeout=.1, on_stdout_line=callback)
    assert not attempt.timed_out
    assert attempt.returncode == 0
    assert seen == ["completed"]


def test_alive_child_is_still_interrupted_after_slow_callback(tmp_path):
    attempt = run_process_stream([sys.executable, "-c",
        "import time; print('running',flush=True); time.sleep(20)"], cwd=tmp_path,
        env=os.environ.copy(), timeout=.1, on_stdout_line=lambda _: time.sleep(.15))
    assert attempt.timed_out


def test_provider_interruption_is_sealed_not_misreported_as_native_failure(tmp_path):
    from autosim.research.runtime_recovery import seal_turn_failure
    output = tmp_path / "run"
    workspace = output / "checkout"
    workspace.mkdir(parents=True)
    turn = "b" * 32
    atomic_json(output / "agent/processes" / f"{turn}.json", {"status": "timed_out"})
    fault = seal_turn_failure(output, workspace, {"status": "interrupted",
        "failure_category": "wall_timeout", "turn_id": turn,
        "process_ref": f"agent/processes/{turn}.json", "timed_out": True,
        "final_text": '{"commands":["valid proposal, not adopted"]}'}, role="init", timeout=900)
    evidence = read_attempt_evidence(output, fault["evidence_id"])
    assert fault["native_operation_status"] == "not_inferred"
    assert fault["provider_final_response_present"]
    assert fault["timeout_seconds"] == 900
    assert evidence["status"] == "provider_turn_failed"
    assert "valid proposal, not adopted" in evidence["text"]
    assert "not a failed native" in evidence["text"]


def test_provider_failure_survives_partial_event_record(tmp_path):
    from autosim.research.runtime_recovery import seal_turn_failure
    output = tmp_path / "run"
    workspace = output / "checkout"
    workspace.mkdir(parents=True)
    events = output / "agent/events.jsonl"
    events.parent.mkdir()
    events.write_text('[]\n{"type":', encoding="utf-8")
    record = seal_turn_failure(output, workspace, {"status": "interrupted",
        "failure_category": "wall_timeout"}, role="init", timeout=900)
    assert read_attempt_evidence(output, record["evidence_id"])["status"] == "provider_turn_failed"


def test_preview_directory_references_are_not_mislabeled_as_escape(tmp_path, monkeypatch):
    from autosim.research import environment_demo, native_context
    repo = tmp_path / "checkout"
    (repo / "envs").mkdir(parents=True)
    class ContextReached(Exception):
        pass
    def stopped(*args):
        raise ContextReached
    monkeypatch.setattr(native_context, "load_context", stopped)
    arguments = dict(code="pass", purpose="real reset preview", timeout=1)
    with pytest.raises(ContextReached):
        environment_demo.capture(tmp_path, repo, source_refs=["envs"], **arguments)
    with pytest.raises(ValueError, match="does not exist"):
        environment_demo.capture(tmp_path, repo, source_refs=["missing"], **arguments)
    with pytest.raises(ValueError, match="escaped"):
        environment_demo.capture(tmp_path, repo, source_refs=[".."], **arguments)


def test_optional_targeted_data_plan_is_memory_not_execution_permission():
    plan = {"objective": "Improve native score", "hypotheses": [], "open_questions": [],
            "next_actions": [], "evidence_refs": [], "data_strategy": {
                "route": "native expert", "why": "development failures show poor recovery",
                "targets": ["legal training states with development-observed recovery needs"],
                "producer_to_loader": ["expert.py", "convert.py", "loader.py"],
                "evidence_refs": ["development receipt"],
                "next_probe": "collect one successful training trajectory and load its batch"}}
    checked = validate_plan(plan)
    assert checked["data_strategy"]["authority"] == "model_data_strategy_not_collection_verification"
    plan["data_strategy"]["grant_gpu"] = True
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_environment_preparation_not_hidden_behind_open_ended_planning(tmp_path, monkeypatch):
    from autosim.research.prepare import Preparation
    class Client:
        supports_main_agent = True
        model = "fixture"
    output = tmp_path / "run"
    repo = output / "checkout"
    repo.mkdir(parents=True)
    controller = Preparation(repo=repo, output=output, client=Client(), scouting=output / "scouting")
    controller.declaration = {"name": "fixture"}
    monkeypatch.setattr(controller, "_surveyed_runnable_stages", lambda: {"train": {"entrypoint": "train.py"}})
    monkeypatch.setattr(controller, "_declaration_refresh_is_justified", lambda **_: False)
    assert not controller.main_agent.get("plan")
    available = controller._available_operations(research_progress={}, research_options={})
    assert "build_the_environment" in available
    assert "run_the_loop" not in available


def test_run_document_explains_provider_stop_and_marks_data_plan_unverified(tmp_path):
    from autosim.research import recorder, run_record
    atomic_json(tmp_path / "agent/runtime_failure.json", {
        "message": "init provider wall_timeout", "native_operation_status": "not_inferred"})
    held = recorder.make_snapshot(tmp_path,
        run_record.build_report_view(tmp_path, "derived", status="adaptation_unresolved"),
        {"actions": [{"step": "stop", "because": "model planning did not complete"}],
         "plan": {"data_strategy": {"route": "training policy success filtering",
                    "why": "development recovery coverage", "targets": ["legal training recovery"],
                    "producer_to_loader": ["collector", "converter", "loader"],
                    "next_probe": "load one valid successful trajectory"}}})
    text = recorder.render(tmp_path, held, {})
    assert "为什么停止" in text and "model planning did not complete" in text
    assert "不是原生安装、reset、训练或 rollout 失败" in text
    assert "legal training recovery" in text and "collector → converter → loader" in text
    assert "尚非采集结果" in text and "以上是 Agent 计划" in text
