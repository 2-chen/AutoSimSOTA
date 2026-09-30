"""A repository-defined DAG can execute without a four-stage naming template."""

import json
import sys
from pathlib import Path

import pytest

from autosim.research.adapter_protocol import OptimizationSpace
from autosim.research.compute_decision import ComputeDecision
from autosim.research.declarative_backend import DeclarativeBackend
from autosim.research.derived_research import DerivedResearch
from autosim.research.execution_graph import ExecutionGraph
from autosim.research.ideas import Idea
from autosim.research.metric_contract import MetricSpec
from autosim.research.receipt_verifier import verify_measurement


def graph():
    return {"nodes": [
        {"id": "make_data", "role": "data_preparation"},
        {"id": "fit_policy", "role": "policy_update", "depends_on": ["make_data"],
         "bindings": {"dataset": "make_data"}},
        {"id": "score", "role": "evaluate", "depends_on": ["fit_policy"],
         "bindings": {"checkpoint": "fit_policy"}}]}


def test_runtime_backend_coalesces_sparse_overrides_and_binds_budget_placeholders(tmp_path):
    source = ("def stage_argv_train(i):\n"
              "    argv = [i['python'], 'train.py', '--num_envs=8', "
              "'--total_timesteps={steps}']\n"
              "    for key in sorted(i['settings']):\n"
              "        argv = argv + ['--' + key + '=' + str(i['settings'][key])]\n"
              "    for key in sorted(i['extra']):\n"
              "        argv = argv + ['--' + key + '=' + str(i['extra'][key])]\n"
              "    return argv\n")
    backend = DeclarativeBackend(
        repo=tmp_path,
        answer={"stages": {"train": {"available": True, "entrypoint": "train.py",
                                       "artifact": "runs/*/final_ckpt.pt"}}},
        sources={"train": source}, parameters={})

    argv = backend.argv("train", {
        "python": "python", "steps": 1024,
        "settings": {"num_envs": 512},
        "extra": {"num_envs": 256},
    })

    assert argv.count("--num_envs=256") == 1
    assert "--num_envs=8" not in argv and "--num_envs=512" not in argv
    assert "--total_timesteps=1024" in argv


def _research(tmp_path, *, declaration=None, graph_doc=None):
    scripts = {
        "make_data": ("import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                      "p.mkdir(parents=True,exist_ok=True); "
                      "(p/'data.txt').write_text('samples')"),
        "fit_policy": ("import pathlib,sys; source=pathlib.Path(sys.argv[1]); "
                       "p=pathlib.Path(sys.argv[2]); p.mkdir(parents=True,exist_ok=True); "
                       "(p/'policy.pt').write_text(source.read_text()+'-trained')"),
        "score": ("import pathlib,sys; source=pathlib.Path(sys.argv[1]); "
                  "p=pathlib.Path(sys.argv[2]); p.mkdir(parents=True,exist_ok=True); "
                  "(p/'result.txt').write_text(source.read_text()); "
                  "print('mean_reward: 3.0')")}
    args = {"make_data": "i['output']", "fit_policy": "i['dataset'], i['output']",
            "score": "i['checkpoint'], i['output']"}
    sources = {name: (f"def stage_argv_{name}(i):\n"
                      f"    return [i['python'], '-c', {script!r}, {args[name]}]\n")
               for name, script in scripts.items()}
    artifacts = {"make_data": "data.txt", "fit_policy": "policy.pt",
                 "score": "result.txt"}
    answer = {"stages": {name: {"available": True, "entrypoint": "repo_native.py",
                                  "invocation": "python repo_native.py",
                                  "artifact": artifacts[name]}
                         for name in scripts}}
    if graph_doc:
        answer["execution_graph"] = graph_doc
    return DerivedResearch(
        repo=tmp_path, output=tmp_path / "out",
        backend=DeclarativeBackend(repo=tmp_path, answer=answer,
                                   sources=sources, parameters={}),
        interpreter=Path(sys.executable), space=OptimizationSpace(), client=None,
        stages=sources, run_id="graph", declaration=declaration,
        compute=ComputeDecision(device="cpu", device_index=0, environment={},
                                why="synthetic graph test"))


def test_graph_order_and_artifact_bindings_reach_native_commands(tmp_path):
    real = _research(tmp_path)
    result = real.run_graph(graph(), target="score", settings={})
    assert result["status"] == "completed", result
    assert [row["id"] for row in result["nodes"]] == [
        "make_data", "fit_policy", "score"]
    assert result["nodes"][1]["inputs"]["dataset"] == result["artifacts"]["make_data"]
    assert result["nodes"][2]["inputs"]["checkpoint"] == result["artifacts"]["fit_policy"]
    assert Path(result["artifacts"]["score"]).read_text() == "samples-trained"
    assert len(list((real.run_root / "graph_runs").glob("*.json"))) == 1
    assert json.loads((real.run_root / "execution_graph_frozen.json").read_text()) == graph()


def test_graph_with_no_training_node_runs_source_evaluation(tmp_path):
    real = _research(tmp_path, declaration={"task_contract": {
        "policy_representation": "source"}})
    real.backend.sources["score"] = (
        "def stage_argv_score(i):\n"
        "    return [i['python'], '-c', "
        "'import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
        "p.mkdir(parents=True,exist_ok=True); "
        "(p/\"result.txt\").write_text(\"source policy\"); "
        "print(\"mean_reward: 3.0\")', i['output']]\n")
    real.backend = DeclarativeBackend(repo=tmp_path, answer=real.backend.answer,
                                      sources=real.backend.sources, parameters={})
    result = real.run_graph({"nodes": [{"id": "score", "role": "evaluate"}]},
                            target="score", settings={})
    assert result["status"] == "completed"
    assert len(result["nodes"]) == 1
    assert Path(result["artifacts"]["score"]).read_text() == "source policy"


def test_graph_checkpoint_score_uses_frozen_policy_and_explicit_metric(tmp_path):
    real = _research(tmp_path)
    real.metric_spec = MetricSpec.from_declaration({"research_goal": {
        "primary_metric": {"name": "mean_reward", "direction": "maximize"}}})
    measured = real.measure_graph(graph(), target="score", settings={}, label="baseline")
    assert measured["ok"] is True and measured["metric_value"] == 3.0
    assert measured["restorable_policy"] is True
    frozen = measured["policy_artifact"]["path"]
    assert frozen == measured["graph_run"]["nodes"][2]["inputs"]["checkpoint"]
    assert Path(frozen).read_text() == "samples-trained"
    assert json.loads((real.run_root / "measurements" / "baseline.json").read_text())[
        "metric_value"] == 3.0
    assert verify_measurement(real.run_root, "baseline")["status"] == "consistent"


def test_graph_comparison_protocol_rejects_evaluator_change_before_producers(tmp_path):
    real = _research(tmp_path)
    real.metric_spec = MetricSpec.from_declaration({"research_goal": {
        "primary_metric": {"name": "mean_reward", "direction": "maximize"}}})
    assert real.measure_graph(graph(), target="score", settings={"task": "Lift"},
                              label="baseline")["ok"]
    attempts = len(list((real.run_root / "attempts").iterdir()))
    real.backend.sources["score"] += "\n# changed evaluator command\n"
    refused = real.measure_graph(graph(), target="score", settings={"task": "Lift"},
                                  label="changed_evaluator")
    assert refused["where"] == "comparison_protocol"
    assert len(list((real.run_root / "attempts").iterdir())) == attempts


def test_research_loop_baseline_can_use_explicit_graph_stage_names(tmp_path):
    documented = {**graph(), "score_target": "score"}
    declaration = {"research_goal": {"primary_metric": {
        "name": "mean_reward", "direction": "maximize"}}}
    real = _research(tmp_path, declaration=declaration, graph_doc=documented)
    report = real.run(rounds=0)
    assert report["rounds"][0]["measured"] is True
    assert report["rounds"][0]["metric_value"] == 3.0
    assert real.execution_graph == documented


def test_graph_research_round_can_improve_a_source_controller(tmp_path):
    (tmp_path / "controller.py").write_text("VALUE = 1.0\n", encoding="utf-8")
    (tmp_path / "evaluate.py").write_text(
        "import pathlib,sys\n"
        "from controller import VALUE\n"
        "out=pathlib.Path(sys.argv[1])\n"
        "out.mkdir(parents=True,exist_ok=True)\n"
        "(out/'result.txt').write_text(str(VALUE))\n"
        "print(f'mean_reward: {VALUE}')\n", encoding="utf-8")
    documented = {"score_target": "score", "nodes": [
        {"id": "score", "role": "evaluate"}]}
    answer = {"stages": {"score": {"available": True, "entrypoint": "evaluate.py",
                                   "invocation": "python evaluate.py", "artifact": "result.txt"}},
              "execution_graph": documented}
    sources = {"score": "def stage_argv_score(i):\n"
                        "    return [i['python'], i['repo'] + '/evaluate.py', i['output']]\n"}
    declaration = {"task_contract": {"policy_representation": "source"},
                   "research_goal": {"primary_metric": {"name": "mean_reward",
                                                        "direction": "maximize"}}}
    real = DerivedResearch(
        repo=tmp_path, output=tmp_path / "out",
        backend=DeclarativeBackend(repo=tmp_path, answer=answer,
                                   sources=sources, parameters={}),
        interpreter=Path(sys.executable), space=OptimizationSpace(), client=None,
        stages=sources, run_id="graph-code", declaration=declaration,
        compute=ComputeDecision(device="cpu", device_index=0, environment={},
                                why="synthetic graph code test"))
    real.library.add(Idea(label="improve controller", granularity="code",
                          change={"file": "controller.py", "find": "VALUE = 1.0",
                                  "replace": "VALUE = 2.0"},
                          touches=["controller.py"], risk="low", crosses="none",
                          why="a source-defined controller is the candidate"))
    report = real.run(rounds=1)
    assert report["rounds"][0]["metric_value"] == 1.0, report["rounds"]
    assert report["rounds"][1]["metric_value"] == 2.0, report["rounds"]
    assert (tmp_path / "controller.py").read_text() == "VALUE = 2.0\n"


def test_graph_source_policy_can_measure_without_checkpoint(tmp_path):
    real = _research(tmp_path, declaration={"task_contract": {
        "policy_representation": "source"}})
    real.metric_spec = MetricSpec.from_declaration({"research_goal": {
        "primary_metric": {"name": "mean_reward", "direction": "maximize"}}})
    real.backend.sources["score"] = (
        "def stage_argv_score(i):\n"
        "    return [i['python'], '-c', "
        "'import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
        "p.mkdir(parents=True,exist_ok=True); "
        "(p/\"result.txt\").write_text(\"source policy\"); "
        "print(\"mean_reward: 3.0\")', i['output']]\n")
    real.backend = DeclarativeBackend(repo=tmp_path, answer=real.backend.answer,
                                      sources=real.backend.sources, parameters={})
    measured = real.measure_graph({"nodes": [{"id": "score", "role": "evaluate"}]},
                                  target="score", settings={}, label="baseline")
    assert measured["ok"] is True and measured["metric_value"] == 3.0
    assert measured["restorable_policy"] is False
    assert "source-defined controller" in measured["scored"]


@pytest.mark.parametrize("document,reason", [
    ({"nodes": [{"id": "x", "role": "r", "depends_on": ["y"]}]}, "missing"),
    ({"nodes": [{"id": "x", "role": "r", "depends_on": ["y"]},
                {"id": "y", "role": "r", "depends_on": ["x"]}]}, "cycle"),
    ({"nodes": [{"id": "x", "role": "r"},
                {"id": "y", "role": "r", "bindings": {"dataset": "x"}}]}, "outside"),
    ({"nodes": [{"id": "x", "role": "r", "bindings": {"output": "x"}}]}, "reserved"),
])
def test_invalid_dependency_contract_is_rejected(document, reason):
    with pytest.raises(ValueError, match=reason):
        ExecutionGraph(document)


def test_graph_refuses_unavailable_node_before_any_command(tmp_path):
    real = _research(tmp_path)
    real.backend.stages["fit_policy"]["available"] = False
    result = real.run_graph(graph(), target="score", settings={})
    assert result["status"] == "blocked" and result["where"] == "fit_policy"
    assert not (real.run_root / "make_data").exists()


def test_graph_records_synchronous_executor_error_without_losing_run(tmp_path, monkeypatch):
    real = _research(tmp_path)

    def broken(*args, **kwargs):
        raise RuntimeError("a generated command was malformed")

    monkeypatch.setattr(real, "run_stage", broken)
    result = real.run_graph(graph(), target="score", settings={})
    assert result["status"] == "internal_error"
    record = json.loads(next((real.run_root / "graph_runs").glob("*.json")).read_text())
    assert record["status"] == "internal_error"
    assert "malformed" in record["why"]
