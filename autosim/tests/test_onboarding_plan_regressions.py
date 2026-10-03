"""Real RoboSyn failure shapes; no fake native score or paid API requests."""
import json
from pathlib import Path
import pytest

from autosim.research import provision, agent_tasks
from autosim.research.common import read_json
from autosim.research.main_agent import validate_task
from autosim.research.workspace_snapshot import create
from autosim.research.evidence_store import read_attempt_evidence


def workspace(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("print('fixture')")
    (source / "unrelated_mesh.stl").write_bytes(b"large-unrelated-fixture" * 100)
    data = tmp_path / "data"
    (data / "task").mkdir(parents=True)
    (data / "task/demo.hdf5").write_bytes(b"fixture")
    output = tmp_path / "run"
    repo = output / "checkout"
    create(source, repo, tracked_only=False, resources=[{"source":str(data),"target":"existing_data"}])
    return output, repo, data


def test_json_prose_placeholders_do_not_poison_fenced_plan():
    assert provision._object('Wrong path {repo}/dataset; shell ${PREFIX}.\n```json\n'
        '{"python":"3.10","commands":["echo {repo}"]}\n```') == {
        "python":"3.10","commands":["echo {repo}"]}


def test_json_ambiguity_and_python_dict_are_not_silently_adopted():
    with pytest.raises(ValueError, match="ambiguous"):
        provision._object('{"python":"3.9"}\n{"python":"3.10"}')
    with pytest.raises(ValueError):
        provision._object("{'python': '3.10'}")


def test_scoped_snapshot_excludes_unrelated_binary_and_keeps_parent_bindings(tmp_path, monkeypatch):
    output, repo, _ = workspace(tmp_path)
    monkeypatch.setattr(agent_tasks, "MAX_READER_BYTES", 1000)
    with pytest.raises(ValueError, match="source_paths"):
        agent_tasks.snapshot(output, repo, "a"*32)
    root, copy = agent_tasks.snapshot(output, repo, "b"*32, source_paths=["train.py"])
    assert (copy / "train.py").is_file()
    assert not (copy / "unrelated_mesh.stl").exists()
    assert read_json(root / "workspace_snapshot.json")["source_paths"] == ["train.py"]
    assert read_json(root / "worker_parent.json")["output"] == str(output)


@pytest.mark.parametrize("paths", [["../outside"],["/etc"],[],["missing.py"]])
def test_scope_cannot_escape_or_silently_ignore_nonexistent_source(tmp_path, paths):
    output, repo, _ = workspace(tmp_path)
    with pytest.raises(ValueError):
        agent_tasks.snapshot(output, repo, "c"*32, source_paths=paths)


def test_task_contract_accepts_scope_only_for_asynchronous_reader():
    task = {"role":"resource","task":"read source","expected_result":"citations","source_paths":["train.py"]}
    assert validate_task(task, allow_source_paths=True)["source_paths"] == ["train.py"]
    with pytest.raises(ValueError):
        validate_task(task)


def test_bound_nested_assets_checked_as_metadata_not_placeholder(tmp_path):
    output, repo, data = workspace(tmp_path)
    assert not (repo / "existing_data/task/demo.hdf5").exists()
    assert provision._asset_exists(repo, Path("existing_data/task/demo.hdf5"))
    assert not provision._asset_exists(repo, Path("existing_data/task/missing.hdf5"))
    (data / "escape").symlink_to(tmp_path)
    assert not provision._asset_exists(repo, Path("existing_data/escape/source/train.py"))


def test_failed_plan_seals_proposals_with_framework_repair_owner(tmp_path, monkeypatch):
    output, repo, _ = workspace(tmp_path)
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            return 'Explanation {repo}.\n```json\n{"python":"3.10","commands":["bad proposal"],"probes":["bad"]}\n```', {}
    client = Client(); client.output = output
    monkeypatch.setattr(provision, "platform_facts", lambda: {})
    monkeypatch.setattr(provision, "plan_problems", lambda *_: ["commands may not launch training"])
    with pytest.raises(provision.EnvironmentPlanError) as held:
        provision.plan(client, repo, manifests={}, attempts=1)
    fault = held.value.planning_failure
    assert fault["native_operation_status"] == "not_started"
    assert fault["repair_owner"] == "environment_planner"
    text = read_attempt_evidence(output, fault["evidence_id"])["text"]
    assert "bad proposal" in text and "may not launch training" in text
    class NextClient:
        def chat_with_metadata(self, system, user, **kwargs):
            context = json.loads(user)["previous_environment_plan_failure"]
            assert context["evidence_id"] == fault["evidence_id"]
            assert "bad proposal" in context["proposal_excerpt"]
            return json.dumps({"python":"3.10","commands":["echo fixture"],
                               "probes":["echo fixture"],"reasoning":"fixture"}), {}
    next_client = NextClient(); next_client.output = output
    monkeypatch.setattr(provision, "plan_problems", lambda *_: [])
    provision.plan(next_client, repo, manifests={}, attempts=1)


def test_plan_can_reference_exact_bound_resource_alias(tmp_path, monkeypatch):
    output, repo, _ = workspace(tmp_path)
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            payload = json.loads(user)
            assert payload["explicit_bound_resources"][0]["target"] == "existing_data"
            return json.dumps({"python":"3.10","commands":["{pip} install fixture"],
                "probes":["{python} -c 'import fixture'"],"reasoning":"fixture",
                "assets":[{"what":"data","where":"bound_resource_1","produced_by":"already present"}]}), {}
    monkeypatch.setattr(provision, "platform_facts", lambda: {})
    plan = provision.plan(Client(), repo, manifests={}, attempts=1)
    assert plan["assets"][0]["where"] == str(repo / "existing_data")


def test_plan_failure_never_requests_benchmark_source_fix(tmp_path):
    from autosim.research.prepare import Preparation
    class Client:
        supports_agent_fix = True
        model = "fixture"
        def as_role(self, role): raise AssertionError("must not invoke source Fix")
    repo = tmp_path / "checkout"; repo.mkdir()
    controller = Preparation(repo=repo, output=tmp_path, client=Client(), scouting=tmp_path / "scouting")
    assert controller._fix_failed_action(action_id="fixture", action={"step":"build_the_environment",
        "outcome":"raised","repair_owner":"environment_planner"}) is None


def test_plan_problems_checks_bound_nested_path_not_host_placeholder(tmp_path):
    output, repo, _ = workspace(tmp_path)
    plan = {"python":"3.10", "commands":["{pip} install fixture"],
            "probes":["{python} -c 'import fixture'"], "reasoning":"fixture",
            "assets":[{"what":"demo", "where":"existing_data/task/demo.hdf5",
                       "produced_by":"already present"}]}
    assert not any("declared path is absent" in fault for fault in
        provision.plan_problems(plan, {"repository_checkout":str(repo)}))
    plan["assets"][0]["where"] = "dataset"
    assert any("declared path is absent" in fault for fault in
        provision.plan_problems(plan, {"repository_checkout":str(repo)}))


def test_run_document_distinguishes_proposal_rejection_from_native_failure(tmp_path):
    from autosim.research import recorder, run_record
    snapshot = recorder.make_snapshot(tmp_path,
        run_record.build_report_view(tmp_path, "derived", status="adaptation_unresolved"),
        {"actions":[{"step":"build_the_environment", "failure_domain":"framework_plan",
                     "because":"ambiguous JSON proposal", "evidence_refs":[]}], "plan":{}})
    text = recorder.render(tmp_path, snapshot, {})
    assert "环境方案尚未通过校验" in text
    assert "尚未执行原生安装、训练或仿真" in text


def test_optional_real_robosyn_source_scope(tmp_path):
    import os
    source_run = os.environ.get("AUTOSIM_ROBOSYN_TRACE")
    if not source_run:
        pytest.skip("optional read-only real trace replay")
    original = read_json(Path(source_run) / "workspace_snapshot.json")
    from autosim.research.common import atomic_json
    output = tmp_path / "replay"
    atomic_json(output / "workspace_snapshot.json", original)
    repo = Path(original["destination"])
    scopes = ["launch", "scripts", "policy/act", "robosynchallenge", "pyproject.toml"]
    root, copy = agent_tasks.snapshot(output, repo, "e"*32, source_paths=scopes)
    assert (copy / "policy/act/finetune.sh").exists()
    assert not (copy / "policy/pi0").exists()
    assert not (copy / "policy/pi05").exists()
    assert read_json(root / "workspace_snapshot.json")["source_bytes"] < 64 * 1024**2
