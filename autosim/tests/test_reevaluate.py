"""A number that came back, a number that did not, and a benchmark that could not be asked.

The distinction these hold down is between `not_reproduced` and `could_not_reevaluate`, and it
is the whole reason the module exists. An evaluator that runs again and prints a different
number is a finding about the benchmark or the policy. An interpreter that has been uninstalled,
a checkout that has moved, a policy whose bytes changed under us -- those establish nothing
about the number at all. A system that reports the second kind as the first turns every broken
environment into evidence that a result was fabricated, and that is worse than not checking.

The other property is structural rather than asserted: the re-evaluation writes into a fresh
directory, so it has no access to the artifacts the original measurement produced. A check that
compared the recorded number against the recorded number would pass on a benchmark that no
longer runs at all.
"""

import json
import sys
from pathlib import Path

import pytest

from autosim.research import reevaluate

#: An evaluator that prints the number it is told to print. The stub benchmark's evaluate
#: stage reads `--say`, so a test can make the second run agree or disagree with the first
#: without any simulation existing.
ARGV = """\
def stage_argv_evaluate(i):
    argv = [i["python"], "-c",
            'import sys; print("succ: " + sys.argv[1])', str(i["settings"]["say"])]
    return argv
"""


def _run(tmp_path, *, claimed="0.50", say="0.50", policy=b"checkpoint", interpreter=None):
    """A run directory with one scored measurement and the records to rebuild its command."""
    root = tmp_path / "checkout"
    root.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "out"
    run_root = output / "research" / "derived"
    (run_root / "measurements").mkdir(parents=True, exist_ok=True)
    (tmp_path / "policy.pth").write_bytes(policy)
    (run_root / "measurements" / "baseline.json").write_text(json.dumps({
        "label": "baseline", "ok": True, "metric_value": float(claimed),
        "metric_utility": float(claimed), "scored": "a checkpoint this run trained",
        "settings": {"say": say, "episodes": 20},
        "evaluate": {"device": "cpu", "compute_environment": {
            "CUDA_VISIBLE_DEVICES": ""}},
        "policy_artifact": {"path": str(tmp_path / "policy.pth"),
                            "content_sha256": reevaluate.digest(tmp_path / "policy.pth")},
        # The shape a measurement actually stores -- `MetricSpec.as_dict()`, where the bound
        # names are `minimum`/`maximum` and the unit is `fraction`. A fixture in any other
        # shape would exercise a metric contract that no run has ever written.
        "metric": {"name": "success_rate", "direction": "maximize", "unit": "fraction",
                   "minimum": 0.0, "maximum": 1.0, "source": "log", "json_key": "",
                   "csv_column": "", "aggregation": "", "min_samples": 1}}),
        encoding="utf-8")
    (run_root / "comparison_protocol.json").write_text(json.dumps({
        "schema_version": 1, "target": "evaluate", "settings": {"say": say, "episodes": 20}}),
        encoding="utf-8")
    (output / "derived_stages.json").write_text(json.dumps({
        "evaluate": {"source": ARGV, "parameters": {},
                     "row": {"available": True, "entrypoint": "x.py",
                             "invocation": "python x.py", "artifact": "",
                             "working_directory": "{repo}"}}}), encoding="utf-8")
    (output / "execution.json").write_text(json.dumps({
        "repo": str(root),
        "stages": {"evaluate": {"available": True, "entrypoint": "x.py",
                                "invocation": "python x.py", "artifact": ""}}}),
        encoding="utf-8")
    (output / "environment.json").write_text(json.dumps({
        "interpreter": interpreter or sys.executable}), encoding="utf-8")
    return output, run_root


def test_a_number_the_evaluator_prints_again_is_reproduced(tmp_path):
    """The case the mechanism exists for. Note that the second run happens in a directory
    that did not exist when the first one ran -- what it reads was written by itself."""
    output, run_root = _run(tmp_path)
    verdict = reevaluate.reevaluate(run_root, "baseline", output=tmp_path / "again")
    assert verdict["status"] == "reproduced"
    assert verdict["claimed"] == 0.5 and verdict["measured"] == 0.5
    assert "the same value the measurement recorded" in verdict["why"]
    assert verdict["policy_matches"] is True


def test_a_number_the_evaluator_does_not_print_again_is_not_reproduced(tmp_path):
    """A real finding: the evaluator ran, exited zero, and produced a different number."""
    output, run_root = _run(tmp_path, claimed="0.50", say="0.30")
    verdict = reevaluate.reevaluate(run_root, "baseline", output=tmp_path / "again")
    assert verdict["status"] == "not_reproduced"
    assert verdict["claimed"] == 0.5 and verdict["measured"] == 0.3
    assert "where the measurement recorded" in verdict["why"]


def test_a_missing_interpreter_is_not_evidence_about_the_number(tmp_path):
    """The distinction the module is built around. An uninstalled environment establishes
    nothing, and reporting it as a failed reproduction is how a system starts lying."""
    output, run_root = _run(tmp_path, interpreter="/nonexistent/bin/python")
    verdict = reevaluate.reevaluate(run_root, "baseline", output=tmp_path / "again")
    assert verdict["status"] == "could_not_reevaluate"
    assert verdict["measured"] is None
    assert "this is not evidence about the number" in verdict["why"]


def test_an_evaluator_that_fails_establishes_nothing_either_way(tmp_path):
    output, run_root = _run(tmp_path)
    (output / "derived_stages.json").write_text(json.dumps({
        "evaluate": {"source": "def stage_argv_evaluate(i):\n    return [i['python'], '-c', "
                               "'raise SystemExit(3)']\n",
                     "parameters": {}, "row": {"available": True}}}), encoding="utf-8")
    verdict = reevaluate.reevaluate(run_root, "baseline", output=tmp_path / "again")
    assert verdict["status"] == "could_not_reevaluate"
    assert "returncode 3" in verdict["why"]
    assert "either way" in verdict["why"]


def test_a_policy_whose_bytes_changed_is_refused_before_anything_runs(tmp_path):
    """Re-running against overwritten weights measures a different policy and would be
    reported as a failure to reproduce, which is a finding about the wrong thing."""
    output, run_root = _run(tmp_path)
    (tmp_path / "policy.pth").write_bytes(b"a different checkpoint")
    decision = reevaluate.plan(run_root, "baseline")
    assert decision["status"] == "refused"
    assert "would score a different policy" in decision["why"]
    assert decision["policy"]["matches"] is False


def test_the_real_policy_archiver_produces_a_recheckable_identity(tmp_path):
    from autosim.research.experiment_bundle import freeze_artifact

    _, run_root = _run(tmp_path)
    archived = freeze_artifact(tmp_path / "policy.pth",
                               run_root / "experiments" / "baseline" / "policy.pth",
                               max_bytes=1024)
    measurement = run_root / "measurements" / "baseline.json"
    row = json.loads(measurement.read_text(encoding="utf-8"))
    row["policy_artifact"] = archived
    measurement.write_text(json.dumps(row), encoding="utf-8")
    assert archived["content_sha256"] == reevaluate.digest(Path(archived["path"]))
    assert reevaluate.plan(run_root, "baseline")["status"] == "ready"
    assert reevaluate.reevaluate(run_root, "baseline", output=tmp_path / "again")[
        "status"] == "reproduced"
    Path(archived["path"]).write_bytes(b"tampered")
    assert reevaluate.plan(run_root, "baseline")["status"] == "refused"


def _isolated_source_run(tmp_path):
    from autosim.research.snapshot import Snapshots
    from autosim.research.workspace_snapshot import create

    output, run_root = _run(tmp_path)
    source = tmp_path / "original_source"
    source.mkdir()
    (source / "eval.py").write_text("0.50", encoding="utf-8")
    checkout = output / "checkout"
    create(source, checkout, max_bytes=1024)
    execution = output / "execution.json"
    row = json.loads(execution.read_text(encoding="utf-8"))
    row["repo"] = str(checkout)
    execution.write_text(json.dumps(row), encoding="utf-8")
    command = ("def stage_argv_evaluate(i):\n"
               "    return [i['python'], '-c', "
               "'from pathlib import Path; print(\"succ: \" + Path(\"eval.py\").read_text())']\n")
    stages = output / "derived_stages.json"
    row = json.loads(stages.read_text(encoding="utf-8"))
    row["evaluate"]["source"] = command
    stages.write_text(json.dumps(row), encoding="utf-8")
    Snapshots(run_root / "snapshots").capture(["eval.py"], repo=checkout,
                                                name="baseline")
    (checkout / "eval.py").write_text("0.30", encoding="utf-8")
    return source, checkout, run_root


def _isolated_source_controller_run(tmp_path):
    from autosim.research.snapshot import Snapshots
    from autosim.research.workspace_snapshot import capture_state, create

    output, run_root = _run(tmp_path)
    source = tmp_path / "source_controller"
    source.mkdir()
    (source / "score.txt").write_text("0.50", encoding="utf-8")
    checkout = output / "checkout"
    workspace = create(source, checkout, max_bytes=1024)
    execution_path = output / "execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["repo"] = str(checkout)
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    stages_path = output / "derived_stages.json"
    stages = json.loads(stages_path.read_text(encoding="utf-8"))
    stages["evaluate"]["source"] = (
        "def stage_argv_evaluate(i):\n"
        "    return [i['python'], '-c', "
        "'from pathlib import Path; print(\"success_rate: \" + Path(\"score.txt\").read_text())']\n")
    stages_path.write_text(json.dumps(stages), encoding="utf-8")
    snapshots = Snapshots(run_root / "snapshots")
    state = capture_state(checkout, workspace, snapshots, name="source-state-baseline")
    measurement_path = run_root / "measurements" / "baseline.json"
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    measurement.update(scored="a source-defined controller with no weight artifact",
                       policy_artifact={}, restorable_policy=False,
                       source_state=state,
                       source_policy_artifact={"kind": "source_tree", **state})
    measurement_path.write_text(json.dumps(measurement), encoding="utf-8")
    return source, checkout, run_root


def test_recheck_rebuilds_an_isolated_source_from_verified_base_and_snapshot(tmp_path):
    source, checkout, run_root = _isolated_source_run(tmp_path)
    decision = reevaluate.plan(run_root, "baseline")
    assert decision["source_isolation"] == "fresh_copy_planned"
    verdict = reevaluate.reevaluate(run_root, "baseline", output=tmp_path / "again")
    assert verdict["status"] == "reproduced", verdict
    assert verdict["source_isolation"] == "fresh_copy_from_verified_base_and_snapshot"
    assert (tmp_path / "again" / "checkout" / "eval.py").read_text() == "0.50"
    assert (checkout / "eval.py").read_text() == "0.30"


def test_source_defined_controller_rechecks_from_its_frozen_source_tree(tmp_path):
    source, checkout, run_root = _isolated_source_controller_run(tmp_path)
    decision = reevaluate.plan(run_root, "baseline")
    assert decision["status"] == "ready", decision
    assert decision["policy"]["kind"] == "source_tree"
    verdict = reevaluate.reevaluate(run_root, "baseline", output=tmp_path / "again")
    assert verdict["status"] == "reproduced", verdict
    assert verdict["policy_matches"] is True
    assert (tmp_path / "again" / "checkout" / "score.txt").read_text() == "0.50"
    assert (checkout / "score.txt").read_text() == "0.50"


def test_source_defined_controller_recheck_refuses_a_missing_overlay_blob(tmp_path):
    _, checkout, run_root = _isolated_source_controller_run(tmp_path)
    from autosim.research.snapshot import Snapshots
    from autosim.research.workspace_snapshot import capture_state

    path = checkout / "score.txt"
    path.write_text("0.75", encoding="utf-8")
    snapshots = Snapshots(run_root / "snapshots")
    workspace = json.loads((checkout.parent / "workspace_snapshot.json").read_text())
    frozen = capture_state(checkout, workspace, snapshots,
                           name="source-state-tampered-fixture")
    measurement_path = run_root / "measurements" / "baseline.json"
    measurement = json.loads(measurement_path.read_text())
    measurement["source_state"] = frozen
    measurement["source_policy_artifact"] = {"kind": "source_tree", **frozen}
    measurement_path.write_text(json.dumps(measurement))
    snapshot = snapshots.get(frozen["source_snapshot"])
    snapshots.blobs.joinpath(next(key for key in snapshot.files.values() if key)).unlink()
    decision = reevaluate.plan(run_root, "baseline")
    assert decision["status"] == "refused"
    assert "missing or corrupt" in decision["why"]


def test_recheck_refuses_when_isolated_source_base_has_changed(tmp_path):
    source, _, run_root = _isolated_source_run(tmp_path)
    (source / "eval.py").write_text("0.25", encoding="utf-8")
    verdict = reevaluate.reevaluate(run_root, "baseline", output=tmp_path / "again")
    assert verdict["status"] == "could_not_reevaluate"
    assert "source base changed" in verdict["why"]


def test_recheck_does_not_need_the_mutated_isolated_checkout(tmp_path):
    _, checkout, run_root = _isolated_source_run(tmp_path)
    checkout.rename(tmp_path / "moved_checkout")
    verdict = reevaluate.reevaluate(run_root, "baseline", output=tmp_path / "again")
    assert verdict["status"] == "reproduced", verdict


def test_a_frozen_policy_directory_is_verified_before_recheck(tmp_path):
    from autosim.research.experiment_bundle import freeze_artifact

    _, run_root = _run(tmp_path)
    source = tmp_path / "policy_tree"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"weights")
    archived = freeze_artifact(source, run_root / "experiments" / "baseline" / "policy",
                               max_bytes=1024)
    path = run_root / "measurements" / "baseline.json"
    row = json.loads(path.read_text())
    row["policy_artifact"] = archived
    path.write_text(json.dumps(row))
    assert reevaluate.plan(run_root, "baseline")["status"] == "ready"
    (Path(archived["path"]) / "weights.bin").write_bytes(b"tampered")
    assert reevaluate.plan(run_root, "baseline")["status"] == "refused"


def test_recheck_uses_declared_score_stage_and_native_json_result(tmp_path):
    output, run_root = _run(tmp_path)
    path = run_root / "measurements" / "baseline.json"
    row = json.loads(path.read_text())
    row["metric"] = {"name": "reward", "direction": "maximize", "unit": "reward",
                     "minimum": None, "maximum": None, "source": "json",
                     "json_key": "summary.reward", "csv_column": "",
                     "aggregation": "", "min_samples": 1}
    row["metric_value"] = 2.5
    row["metric_utility"] = 2.5
    path.write_text(json.dumps(row))
    protocol = run_root / "comparison_protocol.json"
    protocol_row = json.loads(protocol.read_text())
    protocol_row["target"] = "score_native"
    protocol.write_text(json.dumps(protocol_row))
    source = ("def stage_argv_score_native(i):\n"
              "    return [i['python'], '-c', "
              "'import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True); (p/\"result.json\").write_text(json.dumps({\"summary\": {\"reward\": 2.5}}))', i['output']]\n")
    (output / "derived_stages.json").write_text(json.dumps({
        "score_native": {"source": source, "parameters": {},
                         "row": {"available": True, "artifact": "result.json"}}}))
    (output / "execution.json").write_text(json.dumps({
        "repo": str(tmp_path / "checkout"),
        "stages": {"score_native": {"available": True, "artifact": "result.json"}}}))
    verdict = reevaluate.reevaluate(run_root, "baseline", output=tmp_path / "again")
    assert verdict["status"] == "reproduced", verdict
    assert verdict["measured"] == 2.5


def test_a_checkout_that_has_moved_is_named_rather_than_run_against(tmp_path):
    output, run_root = _run(tmp_path)
    (output / "execution.json").write_text(json.dumps({
        "repo": str(tmp_path / "gone"),
        "stages": {"evaluate": {"available": True}}}), encoding="utf-8")
    decision = reevaluate.plan(run_root, "baseline")
    assert decision["status"] == "refused"
    assert "is not at" in decision["why"]


def test_a_measurement_with_no_number_has_nothing_to_reproduce(tmp_path):
    """The common case in this repository's own runs: eight arms, none of which produced a
    number. Re-running one would be investigating a measurement that does not exist."""
    output, run_root = _run(tmp_path)
    (run_root / "measurements" / "baseline.json").write_text(json.dumps({
        "label": "baseline", "ok": False, "metric_value": None,
        "why": "the evaluate stage did not exit normally"}), encoding="utf-8")
    decision = reevaluate.plan(run_root, "baseline")
    assert decision["status"] == "refused"
    assert "produced no number" in decision["why"]
    assert "its result is the failure it recorded" in decision["why"]


def test_a_recipe_with_no_source_for_the_command_cannot_be_rebuilt(tmp_path):
    output, run_root = _run(tmp_path)
    (output / "derived_stages.json").write_text(json.dumps({
        "evaluate": {"source": "", "parameters": {}, "row": {}}}), encoding="utf-8")
    decision = reevaluate.plan(run_root, "baseline")
    assert decision["status"] == "refused"
    assert "no source for the evaluate command" in decision["why"]


def test_the_plan_says_what_would_be_run_and_what_is_missing(tmp_path):
    """Plan-only exists so the cost of a re-evaluation can be judged before it is paid, and so
    the reasons not to run are readable without a benchmark present."""
    output, run_root = _run(tmp_path)
    (run_root / "comparison_protocol.json").unlink()
    decision = reevaluate.plan(run_root, "baseline")
    assert decision["status"] == "ready"
    assert decision["settings_frozen"] is False
    assert decision["settings"] == {"say": "0.50", "episodes": 20}
    assert "would_run" in decision
    assert any("no frozen comparison protocol" in gap for gap in decision["gaps"])
    # The private keys are inputs to `reevaluate`, not part of the answer a reader reads, and
    # `public` is the single place that strips them -- so the plan a person reads and the plan
    # the runner consumes cannot drift into two dicts.
    assert {"_sources", "_parameters", "_answer", "_repo"} <= set(decision)
    assert not [key for key in reevaluate.public(decision) if key.startswith("_")]
    assert set(reevaluate.public(decision)) < set(decision)


def test_a_reevaluation_will_not_write_into_an_existing_directory(tmp_path):
    """An output directory that already holds a run's artifacts is one the re-evaluation could
    read from, and a check that reads what it is checking is not a check."""
    output, run_root = _run(tmp_path)
    (tmp_path / "again").mkdir()
    verdict = reevaluate.reevaluate(run_root, "baseline", output=tmp_path / "again")
    assert verdict["status"] == "refused"
    assert "already exists" in verdict["why"]


def test_the_verdict_carries_what_it_does_not_establish(tmp_path):
    """`reproduced` is quoted later, away from this code, so its limits travel with it."""
    output, run_root = _run(tmp_path)
    verdict = reevaluate.reevaluate(run_root, "baseline", output=tmp_path / "again")
    assert verdict["status"] == "reproduced"
    assert any("not that it is the number the benchmark's test set would give" in one
               for one in verdict["limitations"])
    assert any("repeated here rather than caught" in one for one in verdict["limitations"])


def test_the_records_one_level_up_are_found_without_being_told_where_they_are(tmp_path):
    """The real layout: a run's measurements live in `output/research/<run_id>/`, and the
    commands that produced them were written one level above by the preparation. A caller
    should not have to know that, and a mechanism that required it would be unusable on the
    runs that already exist."""
    output, run_root = _run(tmp_path)
    assert run_root.parent != output and not (run_root / "derived_stages.json").is_file()
    assert reevaluate._recipe_root(run_root) == output
    assert reevaluate.plan(run_root, "baseline")["status"] == "ready"


def test_the_verb_is_reachable_from_the_command_line(tmp_path, capsys):
    """A mechanism nothing can invoke is a module, not a capability. The check is that the
    plan comes back through the CLI and that its exit code says which way the plan went."""
    from autosim import cli

    output, run_root = _run(tmp_path)
    assert cli.main(["recheck", str(run_root), "baseline",
                     "--output", str(tmp_path / "again"), "--plan-only"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
    (run_root / "measurements" / "baseline.json").write_text(json.dumps({
        "label": "baseline", "ok": False, "metric_value": None}), encoding="utf-8")
    assert cli.main(["recheck", str(run_root), "baseline",
                     "--output", str(tmp_path / "again"), "--plan-only"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "refused"
