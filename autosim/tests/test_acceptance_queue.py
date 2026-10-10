import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def queue(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "tools/run_acceptance_queue.py"
    spec = importlib.util.spec_from_file_location("acceptance_queue_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PROJECT", tmp_path)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return module


def evidence(queue, output, repo=None):
    queue.atomic_json(output / "preparation_derived.json", {
        "status": "completed", "evaluation_verdict": {"verdict": "real"}})
    queue.atomic_json(output / "research/derived/research_report.json", {
        "run_status": "completed", "rounds": [
            {"round": 0, "measured": True, "metric_value": 0},
            {"round": 1, "status": "measured", "metric_value": 0}],
        "confirmation": {"status": "taken", "metric_value": 0, "baseline_metric_value": 0}})
    if repo:
        queue.atomic_json(output / "workspace_snapshot.json", {"source": str(repo)})


def test_zero_is_real_measurement_not_improvement(queue, tmp_path):
    evidence(queue, tmp_path)
    assert queue.acceptance(tmp_path)[0]
    assert "不等于 SOTA" in queue.acceptance(tmp_path)[1]


def test_generic_scoring_is_not_data_campaign_acceptance(queue, tmp_path):
    evidence(queue, tmp_path)
    assert not queue.acceptance(tmp_path, require_data_campaign=True)[0]


def test_wait_unit_does_not_launch_or_budget_until_predecessor_exits(queue, tmp_path, monkeypatch):
    path, jobs = manifest(queue, tmp_path)
    evidence(queue, Path(jobs[0]['output']), Path(jobs[0]['repo']))
    evidence(queue, Path(jobs[1]['output']), Path(jobs[1]['repo']))
    counter = []
    def observe(unit):
        if unit == 'autosim-predecessor.service':
            counter.append(1)
            return {'LoadState': 'loaded', 'ActiveState': 'active' if len(counter) == 1 else 'inactive'}
        return {'LoadState': 'loaded', 'ActiveState': 'inactive', 'ExecMainStatus': '0'}
    monkeypatch.setattr(queue, 'service', observe)
    monkeypatch.setattr(queue.subprocess, 'run', lambda *a, **k: pytest.fail('no launch of adopted/completed jobs'))
    assert queue.main([str(path), '--wait-unit', 'autosim-predecessor.service']) == 0
    assert len(counter) == 2


def test_research_brief_is_a_separate_cli_argument(queue, tmp_path):
    path, jobs = manifest(queue, tmp_path)
    brief = tmp_path/'brief.md'; brief.write_text('数据研究任务')
    jobs[0]['research_brief'] = str(brief)
    jobs[0]['require_data_campaign'] = True
    queue.validate({'jobs': jobs}, path)
    argv = queue.command(jobs[0])
    assert argv[-2:] == ['--research-brief', str(brief)]
    assert 'research_brief' not in jobs[0]['settings']


def test_no_report_is_not_acceptance(queue, tmp_path):
    assert queue.acceptance(tmp_path)[0] is False


@pytest.mark.parametrize("change", ["candidate", "supervisor", "confirmation", "boolean"])
def test_strict_gate(queue, tmp_path, change):
    evidence(queue, tmp_path)
    p = queue.record(tmp_path / "preparation_derived.json")
    r = queue.record(tmp_path / "research/derived/research_report.json")
    if change == "candidate":
        r["rounds"] = r["rounds"][:1]
    elif change == "supervisor":
        p["evaluation_verdict"]["verdict"] = "uncertain"
    elif change == "confirmation":
        r["confirmation"]["status"] = "not_declared"
    elif change == "boolean":
        r["rounds"][0]["metric_value"] = True
    queue.atomic_json(tmp_path / "preparation_derived.json", p)
    queue.atomic_json(tmp_path / "research/derived/research_report.json", r)
    assert queue.acceptance(tmp_path)[0] is False


def test_nonfinite_numbers_are_not_scores(queue):
    assert not queue.number(float("nan"))
    assert not queue.number(float("inf"))


def test_gate_does_not_count_a_candidate_overwritten_by_staging(queue,tmp_path):
    from tests.test_candidate_source_guard import history
    evidence(queue,tmp_path)
    root=tmp_path/'research/derived'
    report=queue.record(root/'research_report.json')
    report['rounds'][1]['label']='round_1'
    queue.atomic_json(root/'research_report.json',report)
    history(root)
    assert queue.acceptance(tmp_path)[0] is False
    history(root,mismatch=False)
    assert queue.acceptance(tmp_path)[0] is True


def manifest(queue, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    root = tmp_path / "autoresearch_runs/matrix"
    output = tmp_path / "autoresearch_runs/first"
    queue.atomic_json(output / "workspace_snapshot.json", {"source": str(repo)})
    jobs = [{"id": "first", "repo": str(repo), "output": str(output),
        "unit": "autosim-first.service", "settings": {"task": "task"},
        "interpreter": "/usr/bin/python3", "resources": [], "adopt": True},
        {"id": "second", "repo": str(repo),
         "output": str(tmp_path / "autoresearch_runs/second"),
         "unit": "autosim-second.service", "settings": {},
         "interpreter": "/usr/bin/python3", "resources": [], "adopt": False}]
    queue.atomic_json(root / "manifest.json", {"jobs": jobs})
    return root / "manifest.json", jobs


def test_failed_first_stops_queue_without_launch_or_reset(queue, tmp_path, monkeypatch):
    path, jobs = manifest(queue, tmp_path)
    monkeypatch.setattr(queue, "service", lambda _: {"LoadState": "loaded", "ActiveState": "failed"})
    monkeypatch.setattr(queue.subprocess, "run", lambda *a, **k: pytest.fail("must not launch successor"))
    assert queue.main([str(path)]) == 1
    state = queue.record(path.parent / "queue_state.json")
    assert state["jobs"][0]["status"] == "暂停"
    assert state["jobs"][1]["status"] == "等待"
    assert not Path(jobs[1]["output"]).exists()


def test_serial_advancement_only_after_native_gate(queue, tmp_path, monkeypatch):
    path, jobs = manifest(queue, tmp_path)
    evidence(queue, Path(jobs[0]["output"]), Path(jobs[0]["repo"]))
    second_started = []

    def observe(unit):
        if unit == jobs[1]["unit"] and not second_started:
            return {"LoadState": "not-found", "ActiveState": "inactive"}
        return {"LoadState": "loaded", "ActiveState": "inactive", "ExecMainStatus": "0"}

    def launch(args, **kwargs):
        assert queue.acceptance(Path(jobs[0]["output"]))[0]
        second_started.append(args)
        evidence(queue, Path(jobs[1]["output"]), Path(jobs[1]["repo"]))

    monkeypatch.setattr(queue, "service", observe)
    monkeypatch.setattr(queue.subprocess, "run", launch)
    assert queue.main([str(path)]) == 0
    assert len(second_started) == 1
    assert all(row["status"] == "已验收" for row in queue.record(path.parent / "queue_state.json")["jobs"])
    assert queue.main([str(path)]) == 0
    assert len(second_started) == 1


def test_command_keeps_task_budget_and_readonly_resource(queue):
    args = queue.command({"repo": "/repo", "output": "/run", "settings": {"task": "Lift"},
        "interpreter": "/python", "resources": ["data=/datasets"]})
    assert args[args.index("--budget-scope") + 1] == "task"
    assert args[-2:] == ["--resource", "data=/datasets"]
    assert "--storage-auto-gc" not in args
    assert json.loads(args[5]) == {"task": "Lift"}
