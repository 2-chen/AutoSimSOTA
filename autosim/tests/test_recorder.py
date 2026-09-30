"""The reporting agent chooses presentation, never measurements or execution."""
import json
import hashlib
from contextlib import contextmanager
import xml.etree.ElementTree as ET

import pytest

from autosim.research import recorder, run_record, report_page
from autosim.research.common import atomic_json
from autosim.research.prepare import Preparation
from autosim.research.agent_roles import role_profile
from autosim.research.agent_runtime import _role_cli_args


def test_report_includes_partial_build_facts_and_separate_model_budget(tmp_path):
    atomic_json(tmp_path / "provision_progress.json", {"status": "building", "attempts": [
        {"ok": True, "seconds": 12, "evidence_ref": "evidence/" + "a"*32 + ".json"},
        {"ok": False, "seconds": 3, "evidence_ref": "evidence/" + "b"*32 + ".json"}]})
    atomic_json(tmp_path / "agent" / "cost_ledger.json", {
        "limit_usd": 30, "entries": [{"status": "settled", "actual_usd": .24}]})
    view = run_record.build_report_view(tmp_path, "derived", status="budget_exhausted")
    snapshot = recorder.make_snapshot(tmp_path, view, {"actions": [], "plan": {}})
    assert recorder._evidence(snapshot)["provisioning"]["attempts"][0]["ok"]
    result = recorder.render(tmp_path, snapshot, {})
    assert "环境准备的实际操作" in result
    assert "$0.24000" in result and "$29.76000" in result


def test_report_distinguishes_environment_availability_from_readiness(tmp_path):
    atomic_json(tmp_path / "environment_pool.json", {"enabled": True})
    atomic_json(tmp_path / "environment_selection.json", {
        "id": "opaque-base", "reason": "Python 与现有依赖匹配，尚需探针",
        "result": {"ok": True}})
    atomic_json(tmp_path / "package_cache_view.json", {"links": ["file:///fixture.whl"]})
    atomic_json(tmp_path / "environment_cache_publication.json", {
        "snapshot": {"status": "capacity_skipped"}})
    view = run_record.build_report_view(tmp_path, "derived", status="running")
    state = recorder.make_snapshot(tmp_path, view, {"actions": [], "plan": {}})
    evidence = recorder._evidence(state)["environment_reuse"]
    assert evidence["clone_ok"] and evidence["available_wheels"] == 1
    document = recorder.render(tmp_path, state, {})
    assert "环境复用" in document and "不是实测命中数" in document
    assert "Python 与现有依赖匹配" in document
    assert "仍需原生探针" in document and "空间不足，未发布" in document


def measurement(root, label, value, protocol="a" * 64, *, direction="maximize", confirmation=False):
    attempt = hashlib.sha256(label.encode()).hexdigest()[:32]
    base = root / "research" / "derived"
    atomic_json(base / "attempts" / attempt / "receipt.json", {
        "attempt_id": attempt, "stage": "evaluate", "seconds": 12,
        "comparison_protocol_sha256": protocol})
    path = base / "measurements" / (label + ".json")
    atomic_json(path, {"label": label, "ok": value is not None, "metric_value": value,
                      "metric": {"name": "reward", "unit": "points", "direction": direction},
                      "metric_reading": {"episodes_completed": 20},
                      "evaluate": {"attempt_id": attempt}, "varied": {"chunk": 8},
                      "settings": {"task": "demo"}, "confirmation": confirmation})
    return path


def snapshot(root):
    return recorder.make_snapshot(root, run_record.build_report_view(root, "derived", status="running"),
                                  {"actions": [{"step": "train", "outcome": "done"}], "plan": {}})


class Writer:
    supports_recorder = True
    supports_main_agent = True

    def __init__(self, mode="normal"):
        self.calls, self.roles, self.mode = [], [], mode

    @contextmanager
    def as_role(self, role):
        self.roles.append(role)
        yield

    def chat_with_metadata(self, system, user, **kwargs):
        packet = json.loads(user)
        self.calls.append((packet, kwargs))
        if self.mode == "service_failure":
            raise RuntimeError("fixture service unavailable")
        if "skill_catalog" in packet:
            return json.dumps({"skill_reads": [{"id": "autosimsota.writing-research-updates", "why": "解释当前结果"}]}), {}
        answer = {"event_revision": packet["event_revision"],
                  "sections": {key: {"text": "当前证据支持继续核对训练结果，而非宣称完成研究。",
                                     "evidence_ids": ["run"]} for key in recorder.FIELDS},
                  "charts": [{"kind": "comparison", "why": "比较同协议的开发指标。"},
                             {"kind": "cost", "why": "看清阶段耗时。"}], "demo_requests": []}
        ids = [key for key in packet["evidence"] if key.startswith("m-")]
        if ids:
            answer["demo_requests"] = [{"measurement_id": ids[0], "purpose": "观察接触阶段的行为。"}]
        if self.mode == "fabricated":
            answer["sections"]["finding"]["text"] = "成功率提高了99%。"
        if self.mode == "unknown_ref":
            answer["sections"]["finding"]["evidence_ids"] = ["invented"]
        if self.mode == "command":
            answer["charts"][0]["command"] = "run arbitrary code"
        return json.dumps(answer, ensure_ascii=False), {"role": "recorder"}


def test_recorder_event_to_charts_request_and_cached_narration(tmp_path):
    measurement(tmp_path, "baseline", -3)
    measurement(tmp_path, "candidate", -1)
    held = snapshot(tmp_path)
    client = Writer()
    narrative = recorder.narrate(tmp_path, held, client=client)
    assert narrative["status"] == "written"
    assert narrative["skill_selection"][0]["body_sha256"]
    assert client.roles == ["recorder"]
    assert all(options["read_only"] and not options["auto_skills"] for _, options in client.calls)
    assert len(recorder.pending_requests(tmp_path)) == 1
    doc = recorder.render(tmp_path, held, narrative)
    assert "最新发现" in doc and "相对基线差值" in doc and "同协议" in doc
    assert held["measurements"][1]["delta"] == 2
    assert held["measurements"][0]["samples"] == 20
    charts = list((tmp_path / "report" / "charts").glob("*.svg"))
    assert len(charts) == 2
    for path in charts:
        ET.fromstring(path.read_text())
    recorder.narrate(tmp_path, held, client=client)
    assert len(client.calls) == 2
    # New heartbeat/budget/current-action bytes do not create a new research event.
    view = run_record.build_report_view(tmp_path, "derived", status="running", current_action="poll")
    view["budget"] = {"elapsed_wall_seconds": 999}
    later = recorder.make_snapshot(tmp_path, view, {"actions": held["actions"], "plan": {}})
    assert later["event_revision"] == held["event_revision"]
    recorder.narrate(tmp_path, later, client=client)
    assert len(client.calls) == 2
    assert "已有更新" in recorder.render(tmp_path, later, narrative)


@pytest.mark.parametrize("mode", ["service_failure", "fabricated", "unknown_ref", "command"])
def test_bad_narration_falls_back_and_is_not_retried_on_heartbeat(tmp_path, mode):
    held = snapshot(tmp_path)
    client = Writer(mode)
    assert not recorder.narrate(tmp_path, held, client=client)
    count = len(client.calls)
    recorder.narrate(tmp_path, held, client=client)
    assert len(client.calls) == count
    doc = recorder.render(tmp_path, held, {})
    assert "写作服务本轮未完成" in doc and "没有分数不等于成功率为零" in doc
    assert not recorder.pending_requests(tmp_path)


def test_comparison_rejects_mismatched_protocol_and_hidden_confirmation(tmp_path):
    measurement(tmp_path, "baseline", 1)
    measurement(tmp_path, "candidate", 5, protocol="b" * 64)
    measurement(tmp_path, "hidden", 99, confirmation=True)
    measurement(tmp_path, "failed", None)
    held = snapshot(tmp_path)
    assert all(row["label"] != "hidden" for row in held["measurements"])
    assert not next(row for row in held["measurements"] if row["label"] == "candidate")["comparable"]
    assert "hidden" not in json.dumps(recorder._evidence(held))
    assert "不计算" in recorder.render(tmp_path, held, {})
    assert not list((tmp_path / "report/charts").glob("*-comparison.svg"))


def test_new_research_event_stales_narration_and_generates_a_new_turn(tmp_path):
    measurement(tmp_path, "baseline", 1)
    first = snapshot(tmp_path)
    writer = Writer()
    narrative = recorder.narrate(tmp_path, first, client=writer)
    measurement(tmp_path, "candidate", 2)
    second = snapshot(tmp_path)
    assert "上一份叙述已过期" in recorder.render(tmp_path, second, narrative)
    recorder.narrate(tmp_path, second, client=writer)
    assert len(writer.calls) == 4


def test_recorder_has_no_tools_or_mcp_and_cannot_execute(tmp_path):
    profile = role_profile("recorder")
    args = _role_cli_args(profile, workspace=tmp_path, output=tmp_path,
                          python="python", model="fixture", max_budget_usd=.1)
    assert not profile.builtin_tools and not profile.mcp_tools
    assert args[args.index("--tools") + 1] == ""
    assert json.loads(args[args.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert "--allowedTools" not in args


def test_scheduler_capture_is_explicit_frozen_and_not_replayed(tmp_path, monkeypatch):
    output = tmp_path / "out"
    real = Preparation(repo=tmp_path, output=output, client=Writer(), scouting=tmp_path / "scout")
    path = measurement(output, "baseline", 1)
    held = snapshot(output)
    recorder._enqueue(output, held, [{"measurement_id": held["measurements"][0]["id"], "purpose": "观察失败行为"}])
    request = recorder.pending_requests(output)[0]
    calls = []

    class Research:
        def capture_demo(self, **kwargs):
            calls.append(kwargs)
            assert not recorder.pending_requests(output)  # capturing persisted before side effects
            return {"status": "captured", "attempt_id": "3" * 32}

    monkeypatch.setattr(real, "state", lambda: {"research_progress": {"status": "paused"}})
    monkeypatch.setattr(real, "_research_controller", lambda: (Research(), {}))
    result = real.do("review_report_demo", request_id=request["id"], decision="capture", reason="预留预算内展示")
    assert result["outcome"] == "captured" and len(calls) == 1
    assert calls[0]["settings"] == {"task": "demo"}
    assert real._step_review_report_demo(request_id=request["id"], decision="capture", reason="重复")["outcome"] == "rejected"
    recorder._enqueue(output, held, [{"measurement_id": held["measurements"][0]["id"], "purpose": "换一种措辞"}])
    assert not recorder.pending_requests(output)


def test_scheduler_decline_never_constructs_executor(tmp_path, monkeypatch):
    output = tmp_path / "out"
    real = Preparation(repo=tmp_path, output=output, client=Writer(), scouting=tmp_path / "scout")
    measurement(output, "baseline", 1)
    held = snapshot(output)
    recorder._enqueue(output, held, [{"measurement_id": held["measurements"][0]["id"], "purpose": "观察"}])
    request = recorder.pending_requests(output)[0]
    monkeypatch.setattr(real, "_research_controller", lambda: pytest.fail("must not launch"))
    assert real._step_review_report_demo(request_id=request["id"], decision="decline", reason="保留复评预算")["outcome"] == "declined"


def test_preparation_publishes_narration_and_survives_telemetry_refresh(tmp_path):
    output = tmp_path / "out"
    real = Preparation(repo=tmp_path, output=output, client=Writer(), scouting=tmp_path / "scout")
    real.steps = [{"step": "build_the_environment", "outcome": "done", "because": "fixture"}]
    real._refresh_document(status="running", current="choose")
    path = output / "RUN.md"
    original = path.read_text()
    assert "最新发现" in original and original.index("当前进展") < original.index("技术详情")
    assert "最新发现" in (output / "RUN.html").read_text()
    trigger = recorder.read(output, "report/recorder_trigger.json")
    run_record.refresh_live_status(output, fallback_status="running", fallback_current="train")
    assert recorder.read(output, "report/recorder_trigger.json") == trigger
    assert "最新发现" in path.read_text()
    run_record.refresh_research_projection(output, "derived")
    assert path.read_text().count(recorder.START) == 1


def test_live_html_removes_active_content_and_external_images(tmp_path):
    path = tmp_path / "RUN.html"
    report_page.write_live_page(path,
        '# 中文报告\n\n<script>alert(1)</script>\n\n'
        '<img src="https://external.invalid/pixel" onerror="alert(2)">'
        '<a href="javascript:alert(3)">恶意链接</a>\n\n'
        '![本地曲线](report/charts/real.svg)')
    rendered = path.read_text()
    assert "<script" not in rendered and "onerror" not in rendered
    assert "https://external" not in rendered and "javascript:" not in rendered
    assert 'src="report/charts/real.svg"' in rendered


@pytest.mark.parametrize("block", ["changed", "heldout", "running", "exhausted"])
def test_capture_gates_reject_unsafe_or_unbudgeted_requests(tmp_path, monkeypatch, block):
    output = tmp_path / "out"
    real = Preparation(repo=tmp_path, output=output, client=Writer(), scouting=tmp_path / "scout")
    path = measurement(output, "baseline", 1)
    held = snapshot(output)
    recorder._enqueue(output, held, [{"measurement_id": held["measurements"][0]["id"], "purpose": "观察"}])
    request = recorder.pending_requests(output)[0]
    progress = {"status": "paused"}
    if block == "running":
        progress["status"] = "running"
    elif block == "exhausted":
        class Exhausted:
            def remaining(self):
                return 0
        real.budget = Exhausted()
    else:
        row = json.loads(path.read_text())
        row["confirmation" if block == "heldout" else "metric_value"] = True if block == "heldout" else 20
        atomic_json(path, row)
    monkeypatch.setattr(real, "state", lambda: {"research_progress": progress})
    monkeypatch.setattr(real, "_research_controller", lambda: pytest.fail("must not launch"))
    assert real._step_review_report_demo(request_id=request["id"], decision="capture", reason="观察")["outcome"] in {"blocked", "rejected"}


def test_runtime_report_includes_actual_measurement_table_and_charts(tmp_path):
    output = tmp_path / "out"
    client = Writer()
    real = Preparation(repo=tmp_path, output=output, client=client, scouting=tmp_path / "scout")
    measurement(output, "baseline", .2)
    measurement(output, "candidate", .4)
    real.steps = [{"step": "run_the_loop", "outcome": "done", "because": "synthetic fixture, not real benchmark"}]
    real._refresh_document(status="running", current="choose")
    text = (output / "RUN.md").read_text()
    assert "baseline" in text and "candidate" in text and "report/charts/" in text
    assert recorder.pending_requests(output)
    assert real.state()["report_demo_requests"]
    assert "review_report_demo" in real.state()["available"]
    assert "report/charts/" in (output / "RUN.html").read_text()


def test_generic_negative_metric_is_not_lost_or_mixed_with_other_units():
    source = {"measurements": [{"record": {"label": "a", "ok": True,
               "metric_value": -2, "metric": {"name": "energy", "direction": "minimize"}}}]}
    assert report_page._points(source) == [("a", -2)]
    ET.fromstring(report_page.curve_svg(report_page._points(source)))
    source["measurements"].append({"record": {"label": "b", "ok": True, "metric_value": 8,
                                              "metric": {"name": "reward"}}})
    assert report_page._points(source) == []


def test_training_plot_is_frozen_to_its_snapshot_and_changed_source_is_withheld(tmp_path):
    held = snapshot(tmp_path)
    path = tmp_path / "report/telemetry/train.svg"
    path.parent.mkdir(parents=True)
    content = '<svg xmlns="http://www.w3.org/2000/svg"><text>loss</text></svg>'
    path.write_text(content)
    held["telemetry_attempts"] = [{"chart_ref": "report/telemetry/train.svg"}]
    expected = hashlib.sha256(content.encode()).hexdigest()
    held["source_revision_vector"]["report/telemetry/train.svg"] = expected
    document = recorder.render(tmp_path, held, {})
    assert f"report/charts/telemetry-{expected}.svg" in document
    path.write_text('<svg/>')
    assert "快照暂不一致" in recorder.render(tmp_path, held, {})
    assert (tmp_path / f"report/charts/telemetry-{expected}.svg").read_text() == content
