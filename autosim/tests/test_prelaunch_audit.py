"""Regression contracts for the LIBERO trajectory audit; no model or GPU required."""
import json
import subprocess
import sys
from pathlib import Path
from contextlib import contextmanager
import pytest

from autosim.research import provision as pv, recorder, native_context, native_execution
from autosim.research.common import atomic_json, isolated_argv, run_local_environment
from autosim.research.evidence_store import read_attempt_evidence


def test_import_inspection_is_not_an_evaluator_launch():
    context = {"stage_paths": {"evaluate": {"entrypoint": "libero/lifelong/evaluate.py"}}}
    assert pv.probe_stage_violation('{python} -c "import libero.lifelong.evaluate as E;print(E.__file__)"', context) == ""
    assert pv.stage_command_violation("env FOO=bar {python} libero/lifelong/evaluate.py", context)
    assert pv.stage_command_violation("{python} -m libero.lifelong.evaluate", context)
    assert pv.probe_stage_violation('{python} -c "print(123);import json"', context)


def test_prelaunch_rejection_has_immutable_evidence_and_never_executes(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "bounded_run", lambda *a, **k: pytest.fail("refused command executed"))
    row = pv.run("python selected_evaluator.py", env={}, cwd=tmp_path, timeout=1,
                 output=tmp_path / "build.log", rejection={"failure_kind": "stage_boundary", "excerpt": "refused"})
    assert not row["launched"]
    assert read_attempt_evidence(tmp_path, row["evidence_id"])["termination_reason"] == "stage_boundary"


def test_cpu_operations_do_not_bind_host_devices(tmp_path):
    repo = tmp_path / "checkout"
    repo.mkdir()
    argv = isolated_argv(["true"], output=tmp_path, repo=repo, allow_gpu=False)
    assert "--dev-bind" not in argv and "--dev" in argv


def test_cooperative_build_returns_each_operation_and_keeps_all_probe_receipts(tmp_path, monkeypatch):
    repo, output = tmp_path / "repo", tmp_path / "run"
    repo.mkdir(); output.mkdir()
    calls = []
    monkeypatch.setattr(pv, "platform_facts", lambda: {})
    monkeypatch.setattr(pv, "isolated_argv", lambda a, **k: a)
    def execute(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "verified", "")
    monkeypatch.setattr(pv, "bounded_run", execute)
    monkeypatch.setattr(pv, "save_recipe", lambda *a, **k: None)
    seed = {"python": sys.executable, "templates": ["echo installed", "echo configured"],
            "probes": ["{python} -c 'import sys;print(sys.version)'", "{python} -c 'import os;print(os.name)'"],
            "reasoning": "fixture"}
    arguments = dict(client=None, prefix=output / "env", output=output,
                     python=sys.executable, manifests={}, assets={}, seed=seed, max_operations=1)
    for _ in range(3):
        result = pv.build(repo, **arguments)
        assert result["status"] == "yielded" and not result["verdict"]["passed"]
    result = pv.build(repo, **arguments)
    assert result["verdict"]["passed"] and len(calls) == 4
    assert len(result["probes"]) == 2
    assert len([r for r in result["record"] if r.get("kind") == "probe"]) == 2
    assert json.loads((output / "provision_cursor.json").read_text()) == {}


def test_configuration_identity_survives_into_later_stages(tmp_path):
    repo = tmp_path / "checkout"; repo.mkdir()
    prefix = tmp_path / "env"; (prefix / "bin").mkdir(parents=True)
    (prefix / "bin/python").symlink_to("/usr/bin/python3")
    env = run_local_environment(tmp_path, {})
    native_context.publish_context(tmp_path, repo, prefix / "bin/python", env)
    native_context.record_command_configuration(tmp_path,
        f'env TASK_CONFIG_PATH="{tmp_path / "config"}" {prefix}/bin/python -c "import sys"', env)
    assert run_local_environment(tmp_path, {})["TASK_CONFIG_PATH"] == str(tmp_path / "config")
    row = native_context.load_context(tmp_path, repo)
    assert row["paths"]["TASK_CONFIG_PATH"] == str(tmp_path / "config")
    with pytest.raises(ValueError, match="different checkout"):
        native_context.load_context(tmp_path, tmp_path)


def test_budget_number_cannot_be_reused_as_performance_claim():
    state = {"event_revision": "v", "plan": {"turn_budget_usd": 4}, "budget": {},
             "measurements": [], "actions": [], "verified_media": [], "telemetry_attempts": []}
    answer = {"event_revision": "v", "sections": {k: {
        "text": "成功率提高了4%。", "evidence_ids": ["plan"]} for k in recorder.FIELDS}}
    with pytest.raises(ValueError, match="typed metric"):
        recorder._validate(answer, state)


def test_typed_metric_claim_must_reference_exact_measurement():
    state = {"event_revision": "v", "plan": {}, "budget": {},
             "measurements": [{"id": "m-a", "metric_value": .5, "metric_unit": "fraction"}],
             "actions": [], "verified_media": [], "telemetry_attempts": []}
    row = {"text": "成功率为0.5。", "evidence_ids": ["m-a"], "metric_claims": [
        {"measurement_id": "m-a", "field": "metric_value", "value": .5, "unit": "fraction"}]}
    answer = {"event_revision": "v", "sections": {k: row for k in recorder.FIELDS}}
    assert recorder._validate(answer, state)["sections"]["finding"]["metric_claims"]
    row["metric_claims"][0]["unit"] = "USD"
    with pytest.raises(ValueError, match="typed measurement"):
        recorder._validate(answer, state)


def test_gpu_native_window_uses_task_budget_and_settles_on_error(tmp_path, monkeypatch):
    from autosim.research.compute_decision import ComputeDecision
    from autosim.research.task_budget import TaskGPUBudget
    monkeypatch.setattr("autosim.research.compute_decision.recheck_gpu", lambda *a, **k: None)
    class Lease:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr("autosim.research.gpu_lease.GPUResourceLease", Lease)
    decision = ComputeDecision("cuda", 0, {"CUDA_VISIBLE_DEVICES": "GPU-fixture"},
                               evidence={"selected_device": {"uuid": "GPU-fixture"}})
    with pytest.raises(RuntimeError, match="probe failed"):
        with native_execution.resource_window(tmp_path, tmp_path, 10, decision) as (seconds, gpu, env):
            assert gpu and seconds == 10 and env["CUDA_VISIBLE_DEVICES"] == "GPU-fixture"
            assert TaskGPUBudget(tmp_path).snapshot()["active_leases"] == 1
            raise RuntimeError("probe failed")
    assert TaskGPUBudget(tmp_path).snapshot()["active_leases"] == 0


def test_guardrails_refuse_secondary_regression_and_missing_values():
    from autosim.research.metric_guardrails import specifications, read_values, compare
    from autosim.research.derived_research import DerivedResearch
    specs = specifications({"research_goal": {"guardrail_metrics": [
        {"name": "latency", "direction": "minimize", "unit": "seconds", "max_regression": .1}]}})
    baseline = read_values(specs, said="latency: 2", artifact=None)
    candidate = read_values(specs, said="latency: 3", artifact=None)
    assert compare(specs, candidate, baseline)["status"] == "violated"
    assert compare(specs, {}, baseline)["status"] == "unknown"
    assert compare(specs, baseline, baseline)["status"] == "passed"
    assert DerivedResearch._metric_utility({"metric_utility": .9, "success_rate": .9,
                                          "guardrails": {"status": "violated"}}) is None


def test_public_search_hits_and_fetched_pages_have_distinct_verified_evidence(tmp_path, monkeypatch):
    from autosim.research import public_sources as ps
    def download(url):
        if "bing.com" in url:
            return b'<rss><channel><item><title>Official doc</title><link>https://example.org/doc</link><description>Snippet</description></item></channel></rss>', "application/rss+xml"
        return b'<html><script>not evidence</script><body>Native collection documentation</body></html>', "text/html"
    monkeypatch.setattr(ps, "_download", download)
    hits = ps.search(tmp_path, "native collection")
    assert hits["results"][0]["verification"] == "search_hit_not_page_verified"
    page = ps.read_source(tmp_path, hits["results"][0]["url"])
    assert page["verification"] == "page_fetched_not_claims_verified"
    assert "not evidence" not in page["text"]
    assert read_attempt_evidence(tmp_path, page["evidence_id"])["text"]


@pytest.mark.parametrize("url", ["http://example.org", "https://user:pass@example.org", "https://example.org:8443", "https://example.org\r\nx: y"])
def test_public_source_url_rejects_unsafe_transports(url):
    from autosim.research.public_sources import validate_url
    with pytest.raises(ValueError): validate_url(url)


def test_public_source_dns_private_addresses_never_open_connection(monkeypatch):
    from autosim.research import public_sources as ps
    monkeypatch.setattr(ps.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, "127.0.0.1 STREAM local\n", ""))
    monkeypatch.setattr(ps.socket, "create_connection", lambda *a, **k: pytest.fail("private address contacted"))
    with pytest.raises(ValueError, match="nonpublic"):
        ps._download("https://example.org/doc")


def test_readonly_scheduler_can_inspect_native_but_not_edit_or_shell(tmp_path):
    from autosim.research.agent_roles import role_profile
    from autosim.research.agent_runtime import _role_cli_args
    profile = role_profile("scheduler", read_only=True)
    assert "mcp__autosim_exec__inspect_native_environment" in profile.mcp_tools
    assert "mcp__autosim_exec__run_command" not in profile.mcp_tools
    assert "Edit" not in profile.builtin_tools
    args = _role_cli_args(profile, workspace=tmp_path, output=tmp_path,
                          python="/usr/bin/python3", model="fixture", max_budget_usd=.1)
    config = json.loads(args[args.index("--mcp-config")+1])
    assert "--native-readonly" in config["mcpServers"]["autosim_exec"]["args"]


def test_native_cpu_inspection_executes_actual_env_readonly_and_seals_evidence(tmp_path):
    from autosim.research.agent_runtime import _McpExecutor
    output = tmp_path / "run"; repo = output / "checkout"; repo.mkdir(parents=True)
    atomic_json(output / "workspace_snapshot.json", {"destination": str(repo)})
    subprocess.run(["/usr/bin/python3", "-m", "venv", "--without-pip", str(output / "env")], check=True)
    source = repo / "source.py"; source.write_text("original")
    env = run_local_environment(output, {})
    ctx = native_context.publish_context(output, repo, output / "env/bin/python", env)
    executor = _McpExecutor(output=output, workspace=repo, allow_commands=False, allow_native=True)
    code = (f"import sys, pathlib; assert sys.prefix == {str(output / 'env')!r}; print('selected-env-ok'); "
            "\ntry: pathlib.Path('source.py').write_text('changed')"
            "\nexcept OSError: print('source readonly')"
            "\nelse: raise RuntimeError('source writable')")
    row = executor.native_probe({"code": code, "purpose": "核验实际解释器与只读边界"})
    assert row["status"] == "completed", row
    assert row["context_identity"] == ctx["identity"] and not row["gpu_access"]
    assert "selected-env-ok" in row["stdout"] and "source readonly" in row["stdout"]
    assert source.read_text() == "original"
    assert read_attempt_evidence(output, row["evidence_id"])["text"]


def test_environment_preview_request_does_not_require_a_measurement(tmp_path):
    state = {"event_revision": "v", "revision": "r", "plan": {}, "budget": {},
             "measurements": [], "actions": [], "verified_media": [], "telemetry_attempts": []}
    answer = {"event_revision": "v", "sections": {k: {
        "text": "尚未完成正式评测，希望先观察环境。", "evidence_ids": ["run"]} for k in recorder.FIELDS},
        "demo_requests": [{"kind": "environment_smoke", "purpose": "观察 reset 与动作响应"}]}
    clean = recorder._validate(answer, state)
    recorder._enqueue(tmp_path, state, clean["demo_requests"])
    assert recorder.pending_requests(tmp_path)[0]["kind"] == "environment_smoke"


def test_initialization_source_is_saved_before_modification_and_not_backfilled(tmp_path):
    from autosim.research.initialization_audit import preserve
    repo = tmp_path / "repo"; repo.mkdir()
    source = repo / "evaluate.py"; source.write_text("success = real_predicate()\n")
    output = tmp_path / "run"
    preserve(output, repo)
    source.write_text("success = True\n")
    preserve(output, repo)
    snapshots = json.loads((output / "initialization_audit/snapshots.json").read_text())
    key = snapshots["snapshots"][0]["files"]["evaluate.py"]
    assert (output / "initialization_audit/blobs" / key).read_text() == "success = real_predicate()\n"


def test_cooperative_fix_preserves_failed_operation_and_remaining_queue(tmp_path, monkeypatch):
    repo, output = tmp_path / "repo", tmp_path / "run"
    repo.mkdir(); output.mkdir()
    calls, fixed = [], False
    monkeypatch.setattr(pv, "platform_facts", lambda: {})
    monkeypatch.setattr(pv, "isolated_argv", lambda a, **k: a)
    monkeypatch.setattr(pv, "save_recipe", lambda *a, **k: None)
    def execute(argv, **kwargs):
        nonlocal fixed
        text = argv[-1]; calls.append(text)
        if text == "echo repair": fixed = True
        return subprocess.CompletedProcess(argv, 1 if text == "echo original" and not fixed else 0,
                                           "missing configuration" if not fixed else "verified", "")
    monkeypatch.setattr(pv, "bounded_run", execute)
    def repair(client, repo, record, failure, **kwargs):
        assert read_attempt_evidence(output, failure["evidence_id"])["text"]
        kwargs["transcript"].append({"kind": "resume", "commands": ["echo repair"]})
        return ["echo repair"], None
    monkeypatch.setattr(pv, "resume", repair)
    seed = {"python": sys.executable, "templates": ["echo original", "echo remaining"],
            "probes": ["{python} -c 'import sys;print(sys.version)'"], "reasoning": "fixture"}
    arguments = dict(client=None, prefix=output / "env", output=output, python=sys.executable,
                     manifests={}, assets={}, seed=seed, max_operations=1)
    result = pv.build(repo, **arguments)
    assert result["status"] == "yielded" and result["latest_failure"]["evidence_id"]
    cursor = json.loads((output / "provision_cursor.json").read_text())
    assert cursor["pending"] == ["echo repair", "echo original", "echo remaining"]
    for _ in range(4): result = pv.build(repo, **arguments)
    assert result["verdict"]["passed"]
    assert calls[:4] == ["echo original", "echo repair", "echo original", "echo remaining"]


def test_probe_correction_requires_independent_exact_source_quotation(tmp_path):
    source = tmp_path / "api.py"; source.write_text("def reset(seed=None): pass\n")
    class Review:
        def chat_with_metadata(self, *a, **k):
            assert k["read_only"]
            return json.dumps({"approved": True, "reason": "same reset capability, corrected signature",
                "citations": [{"file": "api.py", "quote": "def reset(seed=None)"}]}), {}
    claim = {"same_capability": "Use native reset with the supported keyword", "source_refs": ["api.py"]}
    assert pv.review_probe_replacement(Review(), tmp_path, {}, ["corrected reset"], claim)["approved"]
    claim["source_refs"] = [123]
    assert not pv.review_probe_replacement(Review(), tmp_path, {}, [], claim)["approved"]
    assert not pv.review_probe_replacement(Review(), tmp_path, {}, [], None)["approved"]


def test_main_agent_requires_own_plan_before_environment_work(tmp_path):
    from autosim.research.prepare import Preparation
    class Client:
        supports_main_agent = True
        model = "fixture"
    output = tmp_path / "run"; repo = output / "checkout"; repo.mkdir(parents=True)
    controller = Preparation(repo=repo, output=output, client=Client(), scouting=output / "scouting")
    plan = {"objective": "Establish a native baseline", "hypotheses": [], "open_questions": [],
            "next_actions": ["Read native config"], "evidence_refs": []}
    assert "build_the_environment" not in controller.state()["available"]
    assert controller.do("build_the_environment")["outcome"] == "not attempted"
    controller.do("update_research_plan", plan=plan)
    assert controller.main_agent["plan"]["objective"] == plan["objective"]


def test_secondary_metric_cannot_silently_read_different_artifact(tmp_path):
    from autosim.research.metric_guardrails import specifications, read_values
    from autosim.research.metric_contract import MetricSpec
    specs = specifications({"research_goal": {"guardrail_metrics": [{"name": "latency",
        "direction": "minimize", "unit": "seconds", "source": "json", "json_key": "latency",
        "artifact_root": "output", "artifact_pattern": "other.json", "max_regression": 0}]}})
    artifact = tmp_path / "score.json"; artifact.write_text('{"latency":1}')
    reading = read_values(specs, said="", artifact=artifact,
                         primary_spec=MetricSpec(source="json", artifact_root="output", artifact_pattern="score.json"))
    assert reading["latency"]["value"] is None


def test_initialization_audit_exposes_pre_baseline_evaluator_change(tmp_path):
    from autosim.research import initialization_audit, supervisor
    from autosim.research.common import digest
    from autosim.research.snapshot import Snapshots
    output = tmp_path / "run"; repo = output / "checkout"; repo.mkdir(parents=True)
    source = repo / "evaluate.py"; source.write_text("success = real_predicate()\n")
    atomic_json(output / "workspace_snapshot.json", {"source_entries": [
        ["file", "evaluate.py", source.stat().st_size, digest(source), 0]]})
    initialization_audit.preserve(output, repo)
    source.write_text("success = True\n")
    research = output / "research/test"; Snapshots(research).capture(["evaluate.py"], repo=repo, name="baseline")
    audit = supervisor._initialization_changes(output, research)
    assert audit["status"] == "available" and "real_predicate" in audit["diff"] and "success = True" in audit["diff"]


def test_actual_native_preview_is_readonly_unscored_and_projected_into_run_md(tmp_path):
    import shutil
    import base64
    from autosim.research import environment_demo, run_record
    from autosim.research.common import digest
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg: pytest.skip("real preview decoding requires ffmpeg")
    # Explicit synthetic fixture bytes; no benchmark, policy or simulated success claim.
    fixture = tmp_path / "fixture.mp4"
    subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=red:s=64x64:d=0.3", "-threads", "1", "-pix_fmt", "yuv420p",
                    str(fixture)], check=True)
    fixture_bytes = base64.b64encode(fixture.read_bytes()).decode()
    output = tmp_path / "run"; repo = output / "checkout"; repo.mkdir(parents=True)
    source = repo / "api.py"; source.write_text("# synthetic CPU fixture, not simulation evidence\n")
    atomic_json(output / "workspace_snapshot.json", {"destination": str(repo)})
    atomic_json(output / "frozen_protocol.json", {"must": "remain unchanged"})
    original = digest(output / "frozen_protocol.json")
    subprocess.run(["/usr/bin/python3", "-m", "venv", "--without-pip", str(output / "env")], check=True)
    native_context.publish_context(output, repo, output / "env/bin/python", run_local_environment(output, {}))
    code = ("import os, pathlib, base64\n"
        f"for path in [pathlib.Path('api.py'), pathlib.Path({str(output / 'frozen_protocol.json')!r})]:\n"
        "    try: path.write_text('forbidden')\n"
        "    except OSError: pass\n"
        "    else: raise RuntimeError('authoritative path writable')\n"
        f"(pathlib.Path(os.environ['AUTOSIM_DEMO_DIR'])/'fixture.mp4').write_bytes(base64.b64decode({fixture_bytes!r}))\n")
    row = environment_demo.capture(output, repo, code=code, purpose="合成 CPU 预览合同测试，并非仿真 demo",
                                   source_refs=["api.py"], timeout=30)
    assert row["status"] == "recorded_unscored", row
    assert digest(output / "frozen_protocol.json") == original and "synthetic" in source.read_text()
    view = run_record.build_report_view(output, "derived", status="running")
    snap = recorder.make_snapshot(output, view, {})
    text = recorder.render(output, snap, {})
    assert "环境预览（不计分）" in text and "播放环境预览" in text and "实际录制帧" in text
    assert not snap["measurements"]
    frame = output / row["media"][0]["preview"]["path"]
    frame.write_bytes(b"tampered")
    assert recorder.make_snapshot(output, view, {})["environment_demo"]["status"] == "evidence_unavailable"


def test_latest_agent_resource_request_survives_cooperative_resume(tmp_path, monkeypatch):
    from autosim.research.compute_decision import ComputeDecision
    from autosim.research import native_execution
    repo, output = tmp_path / "repo", tmp_path / "run"
    repo.mkdir(); output.mkdir()
    resources = []
    monkeypatch.setattr(pv, "platform_facts", lambda: {})
    monkeypatch.setattr(pv, "isolated_argv", lambda a, **k: a)
    monkeypatch.setattr(pv, "save_recipe", lambda *a, **k: None)
    @contextmanager
    def window(output, repo, timeout, decision=None):
        resources.append("gpu" if decision else "cpu")
        yield timeout, bool(decision), {}
    monkeypatch.setattr(native_execution, "resource_window", window)
    monkeypatch.setattr(pv, "bounded_run", lambda argv, **kwargs:
        subprocess.CompletedProcess(argv, 0, "native import diagnostic", ""))
    probe = "{python} -c 'import sys;print(sys.version)'"
    def repair(client, repo, record, failure, **kwargs):
        kwargs["transcript"].append({"kind": "resume", "resource_requests": [
            {"command": probe, "resource": "cpu", "why": "only inspect interpreter identity"}]})
        return [], None
    monkeypatch.setattr(pv, "resume", repair)
    seed = {"python": sys.executable, "templates": ["echo setup"], "probes": [probe],
            "resource_requests": [{"command": probe, "resource": "gpu", "why": "initial assumption"}]}
    arguments = dict(client=None, prefix=output / "env", output=output, python=sys.executable,
        manifests={}, assets={}, seed=seed, max_operations=1,
        compute=ComputeDecision("cuda", 0, {}, "fixture"))
    # First installation succeeds; then force a failed probe and a revised resource request.
    first = pv.build(repo, **arguments)
    assert first["status"] == "yielded"
    monkeypatch.setattr(pv, "bounded_run", lambda argv, **kwargs:
        subprocess.CompletedProcess(argv, 1 if len(resources) == 2 else 0, "native import diagnostic", ""))
    assert pv.build(repo, **arguments)["status"] == "yielded"
    assert pv.build(repo, **arguments)["verdict"]["passed"]
    assert resources == ["cpu", "gpu", "cpu"]


def test_guardrail_measurement_attachment_blocks_primary_only_win(tmp_path):
    from autosim.research.derived_research import DerivedResearch
    from autosim.research.metric_guardrails import specifications
    from autosim.research.metric_contract import MetricSpec
    real = object.__new__(DerivedResearch)
    real.run_root, real.metric_spec = tmp_path, MetricSpec()
    real.guardrail_specs = specifications({"research_goal": {"guardrail_metrics": [{"name": "latency",
        "direction": "minimize", "unit": "seconds", "source": "log", "max_regression": .1}]}})
    baseline = {"metric_value": .5, "metric_utility": .5}
    real._attach_guardrails(baseline, label="baseline", said="latency: 1.0", artifact=None)
    atomic_json(tmp_path / "measurements/baseline.json", baseline)
    candidate = {"metric_value": .9, "metric_utility": .9}
    real._attach_guardrails(candidate, label="candidate", said="latency: 2.0", artifact=None)
    assert candidate["guardrails"]["status"] == "violated" and not candidate["eligible_for_best"]
    assert real._metric_utility(candidate) is None
    helped, reason = real._did_it_help(idea=None, before_measured=False, measured=True,
                                      result=candidate, baseline=None)
    assert not helped and "secondary" in reason


def test_explicit_gpu_request_cannot_silently_run_on_cpu(tmp_path, monkeypatch):
    from autosim.research.compute_decision import ComputeDecision
    repo, output = tmp_path / "repo", tmp_path / "run"
    repo.mkdir(); output.mkdir()
    calls = []
    monkeypatch.setattr(pv, "platform_facts", lambda: {})
    monkeypatch.setattr(pv, "isolated_argv", lambda a, **k: a)
    monkeypatch.setattr(pv, "save_recipe", lambda *a, **k: None)
    def execute(argv, **kwargs):
        calls.append(argv[-1]); return subprocess.CompletedProcess(argv, 0, "installed", "")
    monkeypatch.setattr(pv, "bounded_run", execute)
    monkeypatch.setattr(pv, "resume", lambda *a, **k: ([], None))
    probe = "{python} -c 'import sys;print(sys.version)'"
    result = pv.build(repo, client=None, prefix=output / "env", output=output, python=sys.executable,
        manifests={}, assets={}, max_operations=2, compute=ComputeDecision("cpu", 0, {}, "no GPU"),
        seed={"python": sys.executable, "templates": ["echo setup"], "probes": [probe],
              "resource_requests": [{"command": probe, "resource": "gpu", "why": "native CUDA check"}]})
    assert result["status"] == "yielded" and not result["verdict"]["passed"]
    assert calls == ["echo setup"] and "unavailable" in result["latest_failure"]["excerpt"]


def test_skill_choice_cache_preserves_agent_reason_and_invalidates_on_new_problem(tmp_path):
    from autosim.research.prepare import Preparation
    chosen = "autosimsota.building-an-environment-that-runs"
    class Client:
        supports_main_agent = True
        model = "fixture"
        calls = 0
        def chat_with_metadata(self, system, user, **kwargs):
            self.calls += 1
            assert "skill_catalog" in json.loads(user)
            return json.dumps({"skill_reads": [{"id": chosen, "why": "inspect actual prefix setup"}]}), {}
    output = tmp_path / "run"; repo = output / "checkout"; repo.mkdir(parents=True)
    client = Client()
    controller = Preparation(repo=repo, output=output, client=client, scouting=output / "scouting")
    facts = controller.state()
    for _ in range(2):
        methods, _, errors = controller._select_controller_methods(facts, decision_attempt_id="a" * 32, traces=[])
        assert not errors
        row = next(r for r in methods["selection"] if r["id"] == chosen)
        assert row["selection_reason"] == "inspect actual prefix setup" and row["selected_by"] == "main_agent"
    assert client.calls == 1
    atomic_json(output / "environment.json", {"latest_failure": {"failure_kind": "new_fault", "excerpt": "new config API"}})
    controller._select_controller_methods(facts, decision_attempt_id="b" * 32, traces=[])
    assert client.calls == 2
