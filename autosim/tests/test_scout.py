"""The scout reads repositories, so these fixtures are repositories.

No model is called: the drafting half is exercised through a stub client that returns
whatever a test hands it, which is exactly the failure this module has to survive -- a model
that omits a field, invents a capability, types a sentence into a path, or leaves off the
final brace. What is asserted is that each of those becomes a stated fault rather than a
crash or a silent pass.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from autosim.research import scout
from autosim.research.declaration import (CAPABILITIES, DeclarativeAdapter, problems_in,
                                          space_from, verify)
from autosim.research.survey import recorded_provenance, summarise, survey


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path):
    """A miniature benchmark: a task list, a teleop collector, and a dataset next door."""
    root = tmp_path / "ToyBench"
    write(root / "setup.py", "setup(name='toybench', version='1.0')\n")
    write(root / "README.md", "# ToyBench\nA toy manipulation benchmark.\n")
    for index in range(4):
        write(root / "tasks" / f"task_{index}.task", f"task {index}\n")
    write(root / "scripts" / "collect.py",
          "import robosuite\nfrom robosuite.utils.input_utils import input2action\n"
          "def collect_human_trajectory(env, device):\n"
          "    device.start_control()\n    action = input2action(device=device, robot=env.robots[0])\n")
    write(root / "base" / "package" / "__init__.py", "VERSION = '1'\n")
    return root


def declaration(**overrides):
    value = {
        "benchmark": "ToyBench",
        "evidence": "setup.py names it and tasks/ holds one file per task",
        "repo_markers": ["setup.py", "tasks"],
        "tasks": {"kind": "glob", "pattern": "tasks/*.task", "task_id_from": "file stem"},
        "task_contract": {"state_dim": 7, "action_dim": 7,
                          "cameras": {"front": [64, 64, 3]}, "max_episode_steps": 200,
                          "recorded_fps": 20.0, "instruction": "do the thing"},
        "assets": {},
        "capabilities": {name: {"status": "unsupported", "why": "not in this fixture"}
                         for name in CAPABILITIES},
        "optimization_space": {"collection": [], "training": [
            {"name": "steps", "kind": "integer", "description": "training steps",
             "low": 1, "high": 1000, "default": 100}]},
    }
    value.update(overrides)
    return value


# -- the survey names no benchmark ------------------------------------------------------

def test_survey_reports_signals_without_labelling_them(repo):
    report = survey(repo)
    sources = {row["path"]: row["signals"] for row in report["signal_sources"]}
    assert "scripts/collect.py" in sources
    # The file collects trajectory-shaped things *and* waits for a person. Both facts are
    # reported; neither is turned into a verdict here.
    assert "teleoperation" in sources["scripts/collect.py"]
    assert "collection" in sources["scripts/collect.py"]
    # No verdict about what this project *is*: the survey carries the path it was given,
    # which necessarily contains the name, and nothing beyond that.
    assert "benchmark" not in summarise(report)


def test_survey_does_not_claim_a_neighbours_artifacts(tmp_path, repo):
    """A sibling project's checkpoint sits on the same disk and must not be adopted."""
    neighbour = tmp_path / "OtherBench" / "checkpoints"
    neighbour.mkdir(parents=True)
    (neighbour / "model.safetensors").write_bytes(b"x" * 200_000)
    report = survey(repo)
    assert report["asset_provenance"]["inside_repo"] == 0
    assert report["asset_provenance"]["outside_repo"] == 1
    assert "OtherBench" not in json.dumps(summarise(report)["dataset_structure"])


def test_survey_reads_a_recorded_files_own_provenance(tmp_path):
    """Ownership of data stored elsewhere is settled by what the data says about itself."""
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "ToyBench"
    write(root / "setup.py", "setup(name='toybench')\n")
    write(root / "defs" / "task_0.bddl", "(define (problem t))\n")
    data = tmp_path / "datasets" / "toybench"
    data.mkdir(parents=True)
    with h5py.File(data / "task_0_demo.hdf5", "w") as handle:
        group = handle.create_group("data")
        group.attrs["bddl_file_name"] = "defs/task_0.bddl"
        group.create_dataset("padding", data=np.zeros(2_000_000, dtype="uint8"))
        demo = group.create_group("demo_0")
        demo.create_dataset("actions", data=[[0.0] * 7] * 5)
        demo.create_dataset("rewards", data=[0] * 5)
        observe = demo.create_group("obs")
        observe.create_dataset("front_rgb", data=np.zeros((5, 64, 64, 3), dtype="uint8"))
    report = survey(root)
    provenance = report["recorded_provenance"]
    assert provenance["recorded_paths"] == ["defs/task_0.bddl"]
    # The definition it names is inside the checkout, so the file belongs to it even though
    # the file itself does not live there.
    assert provenance["resolve_inside_repo"] == ["defs/task_0.bddl"]
    assert summarise(report)["recorded_provenance"]["resolve_inside_repo"]


# -- what is checked, and what is only asserted ------------------------------------------

def test_verify_confirms_paths_and_refuses_to_promote_behaviour(repo):
    value = declaration()
    value["capabilities"]["native_evaluation"] = {
        "status": "declared", "entrypoint": "scripts/collect.py", "evidence": "this fixture"}
    result = verify(value, repo)
    assert result["identity"] == "confirmed"
    assert result["task_count"] == 4
    # Nothing here ran the benchmark, so nothing about behaviour may be called verified.
    assert result["capabilities"]["native_evaluation"]["status"] == "declared"
    assert "exercises behaviour" in result["capabilities"]["native_evaluation"]["limitation"]


def test_unsupported_is_carried_through_with_its_reason(repo):
    value = declaration()
    value["capabilities"]["new_trajectory_generation"] = {
        "status": "unsupported", "why": "the only collector waits for a human operator"}
    result = verify(value, repo)
    row = result["capabilities"]["new_trajectory_generation"]
    assert row["status"] == "unsupported"
    assert "human operator" in row["limitation"]


def test_identity_survives_a_marker_that_was_written_wrong(repo):
    """One bad marker in a list of good ones is a typo, not a different project."""
    value = declaration(repo_markers=["setup.py", "tasks", "totally/not/here"])
    result = verify(value, repo)
    assert result["identity"] == "confirmed"
    assert result["identity_failures"] == ["totally/not/here"]
    assert result["identity_markers_resolved"] == ["setup.py", "tasks"]


def test_identity_fails_when_no_marker_resolves(repo):
    value = declaration(repo_markers=["nope/one", "nope/two"])
    assert verify(value, repo)["identity"] == "not_found"


def test_a_declared_entry_point_that_does_not_exist_fails(repo):
    value = declaration()
    value["capabilities"]["native_evaluation"] = {
        "status": "declared", "entrypoint": "scripts/evaluate.py", "evidence": "looks right"}
    result = verify(value, repo)
    assert result["capabilities"]["native_evaluation"]["status"] == "failed"
    assert result["state"] == "partly_contradicted"


def test_a_pattern_counts_what_it_matches(repo):
    result = verify(declaration(), repo)
    check = next(row for row in result["checks"] if row["subject"] == "tasks/*.task")
    assert check["kind"] == "pattern" and check["matches"] == 4


def test_a_module_registry_task_list_is_not_reported_as_confirmed(repo):
    value = declaration(tasks={"kind": "module_registry", "pattern": "tasks",
                               "task_id_from": "module name", "names": ["a", "b"]})
    result = verify(value, repo)
    check = next(row for row in result["checks"] if row["check"] == "tasks_enumerated")
    # Names read from a module the system never imported are an assertion, and reporting
    # them as a pass is the one thing this report exists to prevent.
    assert check["passed"] is None


def test_an_asset_outside_the_checkout_is_not_verified(repo, tmp_path):
    outside = tmp_path / "elsewhere" / "model.pt"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"x" * 200_000)
    value = declaration(assets={"checkpoint": {"path": "../elsewhere/model.pt"}})
    result = verify(value, repo)
    row = result["assets"]["checkpoint"]
    assert row["status"] == "declared" and row["inside_repo"] is False
    assert "nothing here establishes" in row["limitation"]


def test_the_declared_contract_is_compared_against_recorded_data(tmp_path):
    h5py = pytest.importorskip("h5py")
    root = tmp_path / "ToyBench"
    write(root / "setup.py", "setup(name='toybench')\n")
    data = tmp_path / "store"
    data.mkdir()
    with h5py.File(data / "x_demo.hdf5", "w") as handle:
        group = handle.create_group("data")
        group.create_dataset("padding", data=np.zeros(2_000_000, dtype="uint8"))
        demo = group.create_group("demo_0")
        demo.create_dataset("actions", data=[[0.0] * 7] * 5)
        observe = demo.create_group("obs")
        observe.create_dataset("front_rgb", data=np.zeros((5, 64, 64, 3), dtype="uint8"))
    report = survey(root)
    wrong = declaration(task_contract={"state_dim": 1, "action_dim": 12,
                                       "cameras": {"back": [32, 32, 3]}, "max_episode_steps": 5})
    check = next(row for row in verify(wrong, root, survey_report=report)["checks"]
                 if row["check"] == "contract_matches_recorded_data")
    assert not check["passed"]
    assert any("action_dim" in row for row in check["disagreements"])


# -- what a malformed draft is told ------------------------------------------------------

@pytest.mark.parametrize("mutation, fragment", [
    ({"assets": {"dataset": {"path": "datasets/x (downloaded by y.py)"}}}, "not a sentence"),
    ({"tasks": {"kind": "glob", "pattern": "tasks/<suite>/*.task", "task_id_from": "stem"}},
     "placeholder"),
    ({"capabilities": {"native_evaluation": {"status": "declared",
                                             "entrypoint": "a.py (the evaluator)",
                                             "evidence": "e"}}}, "not a sentence"),
    ({"decision": "experiment"}, "missing required key"),
])
def test_malformed_drafts_are_rejected_with_a_usable_message(mutation, fragment):
    value = declaration(**mutation)
    if "decision" in mutation:
        value = {k: v for k, v in value.items() if k != "task_contract"}
    faults = problems_in(value)
    assert any(fragment in fault for fault in faults), faults


def test_an_invented_capability_is_rejected_rather_than_ignored():
    value = declaration()
    value["capabilities"]["official_policy_placeholder"] = {"status": "unsupported", "why": "no"}
    faults = problems_in(value)
    assert any("no such entry" in fault for fault in faults)


def test_numeric_capabilities_do_not_drift_into_their_own_contract():
    value = declaration()
    value["optimization_space"]["training"] = [
        {"name": "steps", "kind": "integer", "description": "steps", "low": 10, "high": 1,
         "default": 5}]
    # Both halves must object: the draft is told before it is accepted, and the space
    # refuses rather than quietly dropping the axis it could not build.
    assert any("low < high" in fault for fault in problems_in(value))
    with pytest.raises(ValueError, match="low < high"):
        space_from(value)


# -- drafting survives a model that is careless -------------------------------------------

class StubClient:
    model = "stub"
    base_url = "stub://"
    available = True

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat_with_metadata(self, system, user, **kwargs):
        self.calls.append(user)
        return self.replies.pop(0), {"model": self.model, "finish_reason": "stop"}


def test_a_reply_missing_its_final_brace_is_recovered():
    body = json.dumps(declaration())
    assert body.endswith("}")
    assert scout._balanced_object(body[:-1])["benchmark"] == "ToyBench"


def test_a_genuinely_truncated_reply_is_not_recovered():
    assert scout._balanced_object('{"benchmark": "ToyBench", "evidence": "unfinis') is None


def test_a_reply_wrapped_in_prose_is_read_by_its_braces():
    body = 'Here you go:\n```json\n' + json.dumps(declaration()) + '\n```\nHope that helps.'
    assert scout._balanced_object(body)["benchmark"] == "ToyBench"


def test_drafting_splits_into_two_questions_and_repairs_a_rejected_one(repo):
    """The first identity reply is missing a key; the retry is complete."""
    identity = declaration()
    del identity["assets"]
    replies = [json.dumps(identity), json.dumps(declaration()),
               json.dumps({"capabilities": declaration()["capabilities"],
                           "optimization_space": declaration()["optimization_space"]})]
    client = StubClient(replies)
    result, attempts = scout.propose(survey(repo), [], client)
    assert result["benchmark"] == "ToyBench"
    assert [row["status"] for row in attempts] == ["rejected", "accepted", "accepted"]
    # The retry is told what was wrong, not merely that something was.
    assert "missing required key" in client.calls[1]


def test_a_failed_path_names_the_real_file_it_was_reaching_for(repo):
    """`it does not exist` is misleading when the file exists one directory over."""
    stored = "/store/toybench/task_0_demo.hdf5"
    value = declaration(assets={"dataset": {"path": "datasets/toybench/task_0_demo.hdf5"}})
    report = {"datasets": [{"path": stored}], "model_artifacts": []}
    result = verify(value, repo, survey_report=report)
    assert result["assets"]["dataset"]["status"] == "failed"
    assert result["assets"]["dataset"]["nearest_surveyed"] == [stored]
    faults = scout.path_faults(result)
    assert faults["assets"]["dataset"]["files_the_survey_found"] == [stored]
    assert faults["assets"]["dataset"]["you_wrote"] == "datasets/toybench/task_0_demo.hdf5"


def test_a_status_word_is_not_accepted_as_a_path():
    value = declaration(assets={"checkpoint": {"path": "unsupported"}})
    faults = problems_in(value)
    assert any("does not name a location" in fault for fault in faults), faults


def test_a_declared_benchmark_satisfies_the_adapter_protocol(repo):
    from autosim.research.adapter_protocol import check_adapter
    value = declaration()
    value["capabilities"]["native_evaluation"] = {
        "status": "declared", "entrypoint": "scripts/collect.py", "evidence": "this fixture"}
    adapter = DeclarativeAdapter(declaration=value, repo=repo, verification=verify(value, repo))
    assert check_adapter(adapter) == []
    assert adapter.select_task("auto") == "task_0"
    spec = adapter.task_spec("task_0")
    assert (spec.state_dim, spec.action_dim) == (7, 7)
    # A benchmark that states its setup differently leaves these empty rather than lying.
    assert spec.gym_config == "" and spec.event_families == {}
    assert space_from(value).axis("training", "steps").high == 1000


def test_a_declaration_with_nothing_to_vary_is_reported(repo):
    from autosim.research.adapter_protocol import check_adapter
    value = declaration()
    value["optimization_space"] = {"collection": [], "training": []}
    adapter = DeclarativeAdapter(declaration=value, repo=repo, verification=verify(value, repo))
    assert any("declares nothing" in problem for problem in check_adapter(adapter))


def test_the_ownership_check_prefers_what_the_data_says(tmp_path):
    structures = [{"groups": {"data": {"attrs": {"bddl_file_name": "defs/task.bddl"}}}}]
    root = tmp_path / "bench"
    write(root / "defs" / "task.bddl", "(define)\n")
    assert recorded_provenance(structures, root)["resolve_inside_repo"] == ["defs/task.bddl"]
    assert recorded_provenance(structures, tmp_path / "elsewhere")["resolve_inside_repo"] == []
