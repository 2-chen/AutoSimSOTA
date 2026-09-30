"""Scheduling contracts, including isolated native CPU and read-only worker execution."""
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from autosim.research import agent_tasks, scheduling, screening
from autosim.research.budget import RunBudget
from autosim.research.common import atomic_json, digest, object_digest, read_json
from autosim.research.execution_graph import ExecutionGraph
from autosim.research.ideas import Idea
from autosim.research.task_budget import TaskGPUBudget


def inventory(output, repo):
    source = repo / "train.py"
    source.write_text("# native source\n")
    entries = [["file", "train.py", source.stat().st_size, digest(source), 0o644]]
    atomic_json(output / "workspace_snapshot.json", {"schema_version": 2,
        "destination": str(repo.resolve()), "source_entries": entries,
        "source_tree_fingerprint": object_digest(entries)})
    if not (output / "derived_stages.json").exists():
        atomic_json(output / "derived_stages.json", {})


def test_gpu_windows_reserve_and_extend_atomically(tmp_path, monkeypatch):
    from autosim.research import task_budget
    epoch = [100.0]
    monkeypatch.setattr(task_budget.time, "time", lambda: epoch[0])
    budget = TaskGPUBudget(tmp_path)
    budget.initialize(tmp_path / "repo", cap_seconds=100)
    assert budget.reserve("first", "GPU-1", 20)["reserved_seconds"] == 20
    budget.reserve("second", "GPU-2", 30)
    assert budget.snapshot()["available_gpu_seconds"] == 50
    epoch[0] += 10
    assert budget.extend("first", 50)["reserved_seconds"] == 60
    with pytest.raises(ValueError, match="exceeds task"):
        budget.extend("first", 61)
    budget.finish("first", 12)
    assert budget.snapshot()["available_gpu_seconds"] == 58


def test_host_admission_blocks_oversubscription_and_releases(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduling, "capacity", lambda: {
        "cpu": {"effective_cpus": 4}, "memory": {"total_mib": 1000, "available_mib": 1000}})
    profile = {"cpu": 2, "memory_mib": 100, "gpu": False}
    first = scheduling.HostLease("first", profile, directory=tmp_path / "leases")
    second = scheduling.HostLease("second", profile, directory=tmp_path / "leases")
    try:
        assert first.acquire()
        assert not second.acquire()
        first.release()
        assert second.acquire()
    finally:
        first.release()
        second.release()


def test_host_admission_does_not_reclaim_invisible_foreign_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduling, "capacity", lambda: {
        "cpu": {"effective_cpus": 4}, "memory": {"total_mib": 1000, "available_mib": 1000}})
    monkeypatch.setattr(scheduling, "inspect_process_identity", lambda _: {"status": "not_running"})
    profile = {"cpu": 2, "memory_mib": 100, "gpu": False}
    lease = scheduling.HostLease("new", profile, directory=tmp_path / "leases")
    atomic_json(lease.path, {"foreign": {"resources": profile, "process": {},
        "cores": [0, 1], "pid_namespace": "foreign-namespace"}})
    assert not lease.acquire()
    assert "foreign" in read_json(lease.path)


def test_model_slots_bound_concurrent_provider_admission(tmp_path):
    scheduling.configure(tmp_path)
    atomic_json(tmp_path / "scheduler_policy.json", {**scheduling.DEFAULTS, "model_slots": 1})
    finished = threading.Event()
    errors = []
    def blocked():
        try:
            with scheduling.model_slot(tmp_path, timeout=.15):
                errors.append("provider unexpectedly admitted")
        except TimeoutError:
            finished.set()
    with scheduling.model_slot(tmp_path, timeout=1):
        worker = threading.Thread(target=blocked)
        worker.start()
        assert finished.wait(2)
    worker.join()
    assert not errors


@pytest.mark.parametrize("profile", [{"cpu": True}, {"cpu": 0}, {"priority": 11},
                                      {"gpu": 1}, {"memory_mib": -1}, {"other": 2}])
def test_resource_profiles_reject_malformed_counts(profile):
    with pytest.raises(ValueError):
        scheduling.resource_profile(profile)


REPORT = {"summary": "调查完成，未运行基准", "findings": ["原生源码已读取"],
    "uncertainties": [], "evidence_refs": ["train.py"], "recommended_next_actions": ["复核原生接口"]}


class Reader:
    def __init__(self, gate):
        self.gate = gate
        self.children = []
    def fork_readonly(self, **kwargs):
        child = Reader(self.gate)
        child.scope = kwargs
        self.children.append(child)
        return child
    def chat_with_metadata(self, system, user, **kwargs):
        assert kwargs["read_only"]
        assert self.gate.wait(5)
        return json.dumps(REPORT), {"model": "fixture", "charged": True}


def wait_tasks(output):
    end = time.monotonic()+5
    while time.monotonic() < end:
        rows = agent_tasks.records(output)
        if rows and all(r["status"] not in {"submitted", "running"} for r in rows):
            return rows
        time.sleep(.02)
    pytest.fail("reader did not finish")


def test_readers_are_async_isolated_cancellable_stale_and_acknowledged(tmp_path):
    output, repo = tmp_path / "run", tmp_path / "run/checkout"
    repo.mkdir(parents=True)
    inventory(output, repo)
    scheduling.configure(output)
    gate = threading.Event()
    client = Reader(gate)
    assignment = {"role": "resource", "task": "调查原生入口", "expected_result": "证据与待复核接口"}
    first = agent_tasks.submit(output, repo, client, assignment, {"state_revision": 3}, 4)
    second = agent_tasks.submit(output, repo, client, assignment, {"state_revision": 3}, 4)
    try:
        with pytest.raises(ValueError, match="slots"):
            agent_tasks.submit(output, repo, client, assignment, {}, 4)
        assert client.children[0].scope["output"] != client.children[1].scope["output"]
        assert client.children[0].scope["workspace"].joinpath("train.py").read_text() == "# native source\n"
        agent_tasks.cancel(output, second["id"], "已有足够证据")
        (repo / "train.py").write_text("# changed source\n")
    finally:
        gate.set()
    rows = wait_tasks(output)
    assert {r["id"]: r["status"] for r in rows}[second["id"]] == "cancelled"
    ready = agent_tasks.collect(output, repo)
    assert len(ready) == 1 and ready[0]["stale"]
    assert len(agent_tasks.collect(output, repo)) == 1  # Not lost until parent commits.
    agent_tasks.acknowledge(output, first["id"])
    assert not agent_tasks.collect(output, repo)


def test_lost_reader_is_not_silently_replayed_and_can_be_discarded(tmp_path):
    identity = "a"*32
    atomic_json(tmp_path / "agent_tasks" / identity / "task.json", {
        "id": identity, "status": "running", "cancel_requested": False})
    assert not agent_tasks.collect(tmp_path, tmp_path)
    assert agent_tasks.records(tmp_path)[0]["status"] == "outcome_unknown"
    agent_tasks.cancel(tmp_path, identity, "inspect receipts and discard lost read-only result")
    assert agent_tasks.records(tmp_path)[0]["status"] == "cancelled"


def test_editable_worker_not_admitted(tmp_path):
    scheduling.configure(tmp_path)
    with pytest.raises(ValueError, match="read-only"):
        agent_tasks.submit(tmp_path, tmp_path, Reader(threading.Event()),
            {"role": "fix", "task": "change source", "expected_result": "patch"}, {}, 4)


def test_role_forks_preserve_parent_context_and_share_budget(tmp_path, monkeypatch):
    from autosim.research.agent_client import RoleAwareAgentClient
    output = tmp_path / "run"
    checkout = output / "checkout"
    checkout.mkdir(parents=True)
    parent = RoleAwareAgentClient(workspace=checkout, output=output, run_id="toy",
        turn_budget_usd=1, total_budget_usd=5)
    parent.set_research_context({"owner": "scheduler"})
    root, repo = agent_tasks.snapshot(output, checkout, "b"*32, empty=True)
    child = parent.fork_readonly(output=root, workspace=repo, role="resource")
    child.set_research_context({"owner": "resource"})
    assert parent._role == "scheduler" and parent._research_context["owner"] == "scheduler"
    assert child.budget_output == parent.output
    assert not (child.output / "agent/cost_ledger.json").exists()


def test_worker_runtime_charges_parent_ledger_only(tmp_path, monkeypatch):
    from autosim.research import agent_runtime
    from autosim.research.process_executor import ProcessAttempt
    output, checkout = tmp_path / "run", tmp_path / "run/checkout"
    checkout.mkdir(parents=True)
    root, repo = agent_tasks.snapshot(output, checkout, "d"*32, empty=True)
    scheduling.configure(output)
    monkeypatch.setattr("autosim.llm_client.load_credential_file", lambda **_: {"source": "fixture"})
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-secret")
    monkeypatch.setenv("DEEPSEEK_MODEL", "fixture-model")
    def execute(command, **kwargs):
        kwargs["on_stdout_line"](json.dumps({"type": "result", "total_cost_usd": .01,
            "duration_ms": 3, "usage": {"input_tokens": 10}, "result": "read-only report"}))
        return ProcessAttempt(launched=True, returncode=0, stdout="", stderr="", containment_mode="fixture")
    monkeypatch.setattr(agent_runtime, "run_process_stream", execute)
    result = agent_runtime.run_coding_agent(workspace=repo, output=root, budget_output=output,
        prompt="Inspect source without running experiments", run_id="toy", timeout=5,
        max_budget_usd=.05, max_total_budget_usd=1, role="resource", read_only=True,
        auto_skills=False, cli="/usr/bin/claude")
    assert result["status"] == "completed", result
    assert result["run_budget"]["spent_usd"] == pytest.approx(.01)
    assert not (root / "agent/cost_ledger.json").exists()
    assert read_json(output / "agent/cost_ledger.json")["entries"][0]["actual_usd"] == pytest.approx(.01)


def test_async_recorder_returns_before_writer_and_publishes_later(tmp_path):
    from autosim.research import recorder, run_record
    from tests.test_recorder import Writer
    scheduling.configure(tmp_path)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    gate, started = threading.Event(), threading.Event()
    class AsyncWriter(Writer):
        workspace = checkout
        def fork_readonly(self, **kwargs):
            assert kwargs["role"] == "recorder"
            return self
        def chat_with_metadata(self, *args, **kwargs):
            started.set()
            assert gate.wait(5)
            return super().chat_with_metadata(*args, **kwargs)
    writer = AsyncWriter()
    view = run_record.build_report_view(tmp_path, "derived", status="running")
    text = recorder.refresh(tmp_path, view, context={"plan": {}}, client=writer, timeout=4)
    run_record.publish_document(tmp_path / "RUN.md", text)
    assert started.wait(2) and not writer.calls
    gate.set()
    end = time.monotonic()+5
    while time.monotonic() < end:
        if (tmp_path / "RUN.md").read_text().find("以上为模型解释") >= 0:
            break
        time.sleep(.02)
    assert "以上为模型解释" in (tmp_path / "RUN.md").read_text()
    assert (tmp_path / "RUN.html").is_file()


def test_waiting_segments_do_not_consume_model_relaunches(tmp_path, monkeypatch):
    from tests.test_main_agent import make
    prep = make(tmp_path)
    scheduling.configure(prep.output)
    RunBudget(prep.output, wall_seconds=30)
    prep.budget = RunBudget.existing(prep.output)
    count = [0]
    def cycle(**kwargs):
        count[0] += 1
        prep.steps.append({"step": "wait_for_jobs" if count[0] <= 3 else "stop"})
        return {"status": "action_limit" if count[0] <= 3 else "completed"}
    monkeypatch.setattr(prep, "_run_cycle", cycle)
    report = prep.run(max_steps=1, max_relaunch=0)
    assert count[0] == 4
    assert report["supervision"]["event_wait_segments"] == 3
    assert report["supervision"]["automatic_relaunches"] == 0


def test_generic_detached_cpu_stages_complete_with_independent_outputs(tmp_path):
    from autosim.research import native_jobs
    output, checkout = tmp_path / "run", tmp_path / "run/checkout"
    checkout.mkdir(parents=True)
    (checkout / "README.md").write_text("synthetic native benchmark")
    scheduling.configure(output)
    RunBudget(output, wall_seconds=20)
    atomic_json(output / "environment.json", {"verdict": {"passed": True}, "interpreter": sys.executable})
    sources, stages = {}, {}
    for stage in ("prepare_data", "train"):
        code = ("import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); "
                "p.mkdir(parents=True,exist_ok=True); time.sleep(.4); "
                "(p/'artifact.txt').write_text('native product'); print('global_step=2')")
        sources[stage] = {"source": f"def stage_argv_{stage}(i):\n    return [i['python'], '-c', {code!r}, i['output']]\n", "parameters": {}}
        stages[stage] = {"available": True, "entrypoint": "train.py", "invocation": "python train.py", "artifact": "artifact.txt"}
    atomic_json(output / "derived_stages.json", sources)
    atomic_json(output / "execution.json", {"stages": stages})
    inventory(output, checkout)
    declaration = {"benchmark": "toy_checkout", "evidence": "synthetic test", "repo_markers": ["README.md"],
        "tasks": {"kind": "module_registry", "pattern": "", "names": ["ToyTask"]},
        "task_contract": {"primary_metric": {"name": "success_rate", "direction": "maximize", "unit": "fraction", "source": "log"}},
        "assets": {}, "capabilities": {}, "optimization_space": {"collection": [], "training": [{"name": "epochs", "kind": "integer", "description": "epochs", "low": 1, "high": 2, "default": 1}]}}
    atomic_json(output / "scouting/toy_prepared/declaration.json", {"declaration": declaration})
    jobs = []
    try:
        for stage in stages:
            job = native_jobs.submit(output, repo=checkout, stage=stage, settings={"device": "cpu"},
                requested_seconds=8, reason="independent synthetic stage", resources={"cpu": 1, "memory_mib": 64, "gpu": False})
            jobs.append(job["job_id"])
        end = time.monotonic()+15
        while time.monotonic() < end:
            rows = [native_jobs.status(output, identity) for identity in jobs]
            if all(row["status"] in {"completed", "failed", "cancelled"} for row in rows):
                break
            time.sleep(.1)
        assert all(row["status"] == "completed" for row in rows), rows
        directories = [row["result"]["stage_result"]["output_directory"] for row in rows]
        assert len(set(directories)) == 2
        assert all((Path(path) / "artifact.txt").read_text() == "native product" for path in directories)
        assert not (output / "task_gpu_budget.json").exists()  # CPU work is not billed as GPU time.
        events = read_json(output / "scheduling_metrics.json")["events"]
        assert len([row for row in events if row["kind"] == "resource_wait"]) == 2
    finally:
        for identity in jobs:
            native_jobs.cancel(output, identity, reason="fixture cleanup")


def test_worker_precondition_failure_has_full_evidence_and_visible_handoff(tmp_path):
    from autosim.research import native_jobs
    from autosim.research.evidence_store import read_attempt_evidence
    from tests.test_main_agent import make
    prep = make(tmp_path)
    scheduling.configure(prep.output)
    RunBudget(prep.output, wall_seconds=15)
    atomic_json(prep.output / "environment.json", {"verdict": {"passed": True}, "interpreter": sys.executable})
    atomic_json(prep.output / "derived_stages.json", {"train": {
        "source": "def stage_argv_train(i):\n    return [i['python'], '-c', 'print(1)']\n", "parameters": {}}})
    inventory(prep.output, prep.repo)
    job = native_jobs.submit(prep.output, repo=prep.repo, stage="train", settings={"device": "cpu"},
        requested_seconds=5, reason="intentional fixture without declaration", resources={"cpu": 1, "memory_mib": 64})
    try:
        end = time.monotonic()+10
        while time.monotonic() < end:
            current = native_jobs.status(prep.output, job["job_id"])
            if (current.get("result") or {}).get("evidence_id"):
                break
            time.sleep(.05)
        assert current["status"] == "failed", current
        result = current["result"]
        assert result["evidence_id"] == job["job_id"]
        text = read_attempt_evidence(prep.output, job["job_id"])["text"]
        assert "Traceback" in text and "optimization_space" in text
        view = next(row for row in prep._native_job_view() if row["job_id"] == job["job_id"])
        assert "optimization_space" in view["failure"]
        assert view["evidence_id"] == job["job_id"] and view["result_ref"].endswith("result.json")
    finally:
        native_jobs.cancel(prep.output, job["job_id"], reason="fixture cleanup")


def test_parallel_dag_runs_ready_siblings_and_waits_for_both(tmp_path):
    resource = {"cpu": 1, "memory_mib": 64, "gpu": False}
    graph = ExecutionGraph({"nodes": [
        {"id": "left", "role": "prepare", "resources": resource},
        {"id": "right", "role": "prepare", "resources": resource},
        {"id": "join", "role": "train", "depends_on": ["left", "right"],
         "bindings": {"left_data": "left", "right_data": "right"}, "resources": resource}]})
    barrier = threading.Barrier(2)
    def invoke(name, inputs):
        if name != "join":
            barrier.wait(timeout=3)
        else:
            assert set(inputs) == {"left_data", "right_data"}
        path = tmp_path / name
        path.write_text(name)
        return {"status": "completed", "ran": True, "returncode": 0, "path": path}
    result = graph.execute("join", invoke=invoke, artifact_of=lambda row: row["path"], max_workers=2)
    assert result["status"] == "completed"
    assert [row["id"] for row in result["nodes"]] == ["left", "right", "join"]


def test_parallel_dag_does_not_start_successor_after_failure(tmp_path):
    resource = {"cpu": 1, "memory_mib": 64}
    graph = ExecutionGraph({"nodes": [
        {"id": "broken", "role": "prepare", "resources": resource},
        {"id": "join", "role": "train", "depends_on": ["broken"], "resources": resource}]})
    calls = []
    def invoke(name, inputs):
        calls.append(name)
        return {"status": "failed", "ran": True, "returncode": 1}
    result = graph.execute("join", invoke=invoke, artifact_of=lambda _: None, max_workers=2)
    assert result["status"] == "failed" and calls == ["broken"]


def test_active_native_job_blocks_environment_and_editable_research(tmp_path):
    from tests.test_main_agent import make
    prep = make(tmp_path)
    scheduling.configure(prep.output)
    atomic_json(prep.output / "native_jobs" / ("c"*32) / "request.json", {"stage": "train"})
    assert prep.do("build_the_environment")["outcome"] == "blocked"
    result = prep.do("research_task", role="fix", task="edit", expected_result="patch")
    assert result["outcome"] in {"blocked", "rejected"}


def test_identical_failed_derivation_needs_changed_input_or_fix(tmp_path, monkeypatch):
    from tests.test_main_agent import make
    prep = make(tmp_path)
    scheduling.configure(prep.output)
    inventory(prep.output, prep.repo)
    calls = []
    def derive(step, **args):
        calls.append(step)
        return {"outcome": "no runnable command", "because": "native missing module"}
    monkeypatch.setattr(prep, "_do_operation", derive)
    for _ in range(2):
        prep.do("derive_a_command", stage="train")
    assert prep.do("derive_a_command", stage="train")["outcome"] == "not attempted"
    (prep.repo / "train.py").write_text("# Fix changed source\n")
    assert prep.do("derive_a_command", stage="train")["outcome"] == "no runnable command"
    assert len(calls) == 3


def test_wait_ignores_output_age_and_does_not_call_model(tmp_path, monkeypatch):
    from tests.test_main_agent import make
    prep = make(tmp_path)
    atomic_json(prep.output / "scheduler_policy.json", {**scheduling.DEFAULTS, "wait_seconds": .04})
    monkeypatch.setattr(prep, "_native_job_view", lambda: [{"job_id": "one", "status": "running",
        "progress": {"seconds_since_output": time.time()}}])
    assert prep._step_wait_for_jobs()["outcome"] == "waiting"
    assert not prep.client.calls


def screening_fixture(tmp_path):
    from tests.test_derived_research import research
    real = research(tmp_path)
    checkout = real.output / "checkout"
    checkout.mkdir(parents=True)
    real.repo = checkout
    real.backend.repo = checkout
    real.state_store.repository = checkout
    scheduling.configure(real.output)
    inventory(real.output, real.repo)
    RunBudget(real.output, wall_seconds=30)
    real.budget = RunBudget.existing(real.output)
    real.library.add(Idea(label="loss_low", granularity="param", change={"train.loss_scale": .5},
                          mechanism="lower multiplier", why="controlled toy comparison", status="cleared"))
    doc = {"id": "toy", "budget_axis": "train.n_epochs", "rungs": [1, 4],
           "eta": 2, "min_peers": 2, "evidence": ["train.py"], "reason": "Native epochs control training work"}
    screening.configure(real, doc, {"train.n_epochs": 4, "device": "cpu"})
    return real, doc


def test_screening_runs_native_train_eval_without_formal_selection(tmp_path):
    real, _ = screening_fixture(tmp_path)
    result = screening.trial(real, study_id="toy", idea_label="loss_low", rung=0,
                             window_seconds=20, reason="compare cheap native rollout")
    assert result["status"] == "completed", result
    assert result["metric_value"] == .5
    assert not (real.run_root / "selection.json").exists()
    assert not list((real.run_root / "measurements").glob("*.json"))
    assert not screening.inspect(real, "toy")["promotion_recommendations"]
    assert real.library.get("loss_low").times_tried == 0


def test_screening_protocol_is_frozen_and_requires_native_axis(tmp_path):
    real, doc = screening_fixture(tmp_path)
    with pytest.raises(ValueError, match="declared"):
        screening.configure(real, {**doc, "id": "other", "budget_axis": "epochs"}, {})
    with pytest.raises(ValueError, match="frozen"):
        screening.configure(real, {**doc, "min_peers": 3}, {"train.n_epochs": 4, "device": "cpu"})
    (real.repo / "train.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="changed"):
        screening.trial(real, study_id="toy", idea_label="loss_low", rung=0,
                        window_seconds=10, reason="trial")


def test_screening_same_rung_advice_respects_metric_direction_and_grace(tmp_path):
    real, _ = screening_fixture(tmp_path)
    base = real.run_root / "screening/toy/trials"
    for identity, value in [("a", .2), ("b", .8)]:
        atomic_json(base / identity / "trial.json", {"trial_id": identity, "idea_label": identity,
            "rung": 0, "status": "completed", "metric_value": value})
    result = screening.inspect(real, "toy")
    assert result["promotion_recommendations"] == [{"idea_label": "b", "from_rung": 0, "suggested_rung": 1}]


def test_run_report_exposes_queue_timing_and_separate_screening(tmp_path):
    from autosim.research import recorder, run_record
    scheduling.configure(tmp_path)
    scheduling.note(tmp_path, kind="resource_wait", identity="one", seconds=3, status="admitted")
    atomic_json(tmp_path / "research/derived/screening/toy/trials/one/trial.json",
                {"idea_label": "cheap", "rung": 0, "status": "completed", "metric_value": .5})
    view = run_record.build_report_view(tmp_path, "derived", status="running")
    snapshot = recorder.make_snapshot(tmp_path, view, {"native_jobs": [{"stage": "train",
        "status": "queued", "resources": {"cpu": 2, "memory_mib": 64, "gpu": True},
        "wait_reason": "gpu_busy", "window_seconds": 60}]})
    rendered = recorder.render(tmp_path, snapshot, {})
    assert "后台任务与资源调度" in rendered and "GPU 正忙" in rendered
    assert "低成本候选筛选（不计入正式提升）" in rendered
    assert not snapshot["measurements"]
