"""Bundle records claims separately from executable evidence."""

import json

from autosim.research.benchmark_bundle import build


def test_bundle_preserves_record_fingerprints_without_claiming_a_verified_score(tmp_path):
    repo = tmp_path / "repository"
    repo.mkdir()
    output = tmp_path / "run"
    output.mkdir()
    (output / "execution.json").write_text(json.dumps({"stages": {
        "evaluate": {"entrypoint": "eval.py", "available": True,
                     "artifact": "results.json"}}}), encoding="utf-8")
    (output / "derived_stages.json").write_text(json.dumps({"evaluate": {
        "source": "def stage_argv_evaluate(i): return []",
        "row": {"entrypoint": "eval.py", "artifact": "results.json"}}}), encoding="utf-8")
    declaration = {"task_contract": {"metrics": ["reward"]},
                   "capabilities": {"native_evaluation": {"status": "declared"}}}
    root = build(repo=repo, output=output, declaration=declaration)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    graph = json.loads((root / "execution_graph.json").read_text(encoding="utf-8"))
    assert manifest["fingerprint"]
    assert manifest["status"] == "onboarded_commands_unconfirmed"
    assert graph["nodes"] == [{
        "id": "evaluate", "role": "evaluate", "available": True,
        "entrypoint": "eval.py", "artifact_pattern": "results.json",
        "working_directory": None, "source_ref": "../derived_stages.json",
        "verification": "command_recorded"}]
    assert json.loads((root / "protocol.json").read_text(encoding="utf-8"))[
        "task_contract"] == declaration["task_contract"]


def test_bundle_preserves_explicit_dependencies_without_inventing_them(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "run"
    output.mkdir()
    (output / "execution.json").write_text(json.dumps({
        "stages": {"rollout": {"available": True, "entrypoint": "rollout.py"},
                   "update": {"available": True, "entrypoint": "update.py"}},
        "execution_graph": {"nodes": [
            {"id": "rollout", "role": "collect"},
            {"id": "update", "role": "policy_update", "depends_on": ["rollout"],
             "bindings": {"dataset": "rollout"}}]}}), encoding="utf-8")
    root = build(repo=repo, output=output, declaration={})
    graph = json.loads((root / "execution_graph.json").read_text())
    assert graph["graph_status"] == "declared_valid"
    update = next(row for row in graph["nodes"] if row["id"] == "update")
    assert update["depends_on"] == ["rollout"]
    assert update["bindings"] == {"dataset": "rollout"}
