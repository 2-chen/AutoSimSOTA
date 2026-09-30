"""Measuring the winner on episodes the search never saw, and being unable to do it twice.

A search that reports the best of its arms reports the highest noise among them as well as the
best candidate, and no arithmetic on the way out can undo that -- the selection has already
happened. The only fix is a set of episodes the search is not allowed to look at, measured
once at the end.

Which episodes those are is a fact about a benchmark that this system cannot read off the
code: a seed for one, a task list for another, a fixed initial-state bank for a third where
the answer is that nothing varies. So the split is *declared*, and the tests below are mostly
about the ways a declared split turns out to be no split at all -- reserving a key the
protocol does not freeze, or a value the search already measures at. Both read like a held-out
set in the declaration and neither reserves an episode.
"""

import json
import sys
from pathlib import Path

from autosim.research.derived_research import _load_confirmation
from autosim.research.declarative_backend import DeclarativeBackend
from autosim.research.experiment_bundle import freeze_artifact
from autosim.research.metric_contract import MetricSpec
from autosim.research.adapter_protocol import OptimizationSpace
from autosim.research.compute_decision import ComputeDecision
from autosim.research.workspace_snapshot import create
from tests.test_derived_research import research


def _declaring(real, confirmation):
    real.declaration = {**real.declaration, "research_goal": {"confirmation": confirmation}}
    return real


def test_the_search_cannot_measure_on_the_held_out_episodes(tmp_path):
    """The whole mechanism. Without this the split is a comment."""
    real = _declaring(research(tmp_path), {"seed": 101})
    assert real.measure(settings={"seed": 0}, label="baseline")["ok"] is True
    refused = real.measure(settings={"seed": 101}, label="candidate")
    assert refused["where"] == "comparison_protocol"
    assert refused["ran"] is False
    assert "held-out seed value" in refused["why"]


def test_the_confirmation_measures_at_the_frozen_settings_plus_the_held_out_values(tmp_path):
    real = _declaring(research(tmp_path), {"seed": 101})
    real.measure(settings={"seed": 0}, label="baseline")
    confirmed = real.confirm()
    assert confirmed["ok"] is True
    record = json.loads((real.run_root / "confirmation.json").read_text(encoding="utf-8"))
    assert record["settings"]["seed"] == 101
    assert record["held_out"] == {"seed": 101}
    assert record["metric_value"] == 0.5


def test_confirmation_evaluates_the_frozen_winner_without_training(tmp_path):
    real = _declaring(research(tmp_path), {"seed": 101})
    first = real.measure(settings={"seed": 0, "train.n_epochs": 3}, label="baseline")
    assert first["ok"] is True
    real._advance_best("baseline", score=0.5, scale="success_rate", why="baseline")
    second = real.measure(settings={"seed": 0, "train.n_epochs": 7}, label="round_1")
    assert second["ok"] is True
    original = real.run_stage
    calls = []

    def observed(stage, **kwargs):
        calls.append((stage, kwargs))
        return original(stage, **kwargs)

    real.run_stage = observed
    confirmed = real.confirm()
    assert confirmed["ok"] is True
    assert confirmed["candidate_label"] == "baseline"
    assert [stage for stage, _ in calls] == ["evaluate"]
    assert calls[0][1]["checkpoint"] == first["policy_artifact"]["path"]
    assert confirmed["policy_artifact"]["content_sha256"] == (
        first["policy_artifact"]["content_sha256"])
    assert confirmed["settings"] == {"seed": 101}
    assert json.loads((real.run_root / "confirmation.json").read_text())[
        "candidate_label"] == "baseline"


def test_confirmation_compares_frozen_baseline_and_candidate_at_one_heldout_protocol(tmp_path):
    real = _declaring(research(tmp_path), {"seed": 101})
    baseline = real.measure(settings={"seed": 0}, label="baseline")
    candidate = real.measure(settings={"seed": 0, "train.n_epochs": 3}, label="round_1")
    assert baseline["ok"] and candidate["ok"]
    real.snapshots.capture([], repo=real.repo, name="round-1")
    real.snapshots.advance("round-1", score=0.5, scale="success_rate")
    original = real.run_stage
    calls = []

    def observed(stage, **kwargs):
        calls.append((stage, kwargs))
        return original(stage, **kwargs)

    real.run_stage = observed
    result = real.confirm()
    assert result["ok"] is True
    assert [stage for stage, _ in calls] == ["evaluate", "evaluate"]
    assert [row["checkpoint"] for _, row in calls] == [
        baseline["policy_artifact"]["path"], candidate["policy_artifact"]["path"]]
    assert all(row["settings"] == {"seed": 101} for _, row in calls)
    assert result["baseline_metric_value"] == result["metric_value"] == 0.5
    assert result["confirmation_comparison"]["verdict"] == "not_established"
    refused = real.measure(settings={"seed": 0}, label="too_late")
    assert refused["where"] == "comparison_protocol"
    assert "held-out set was exposed" in refused["why"]


def test_confirmation_accepts_and_verifies_a_directory_policy(tmp_path):
    real = _declaring(research(tmp_path), {"seed": 101})
    baseline = real.measure(settings={"seed": 0}, label="baseline")
    assert baseline["ok"]
    original = Path(baseline["policy_artifact"]["path"])
    directory = real.run_root / "directory_checkpoint"
    directory.mkdir()
    (directory / "model.pth").write_bytes(original.read_bytes())
    policy = freeze_artifact(directory, real.run_root / "frozen_directory", max_bytes=1024)
    baseline["policy_artifact"] = policy
    (real.run_root / "measurements" / "baseline.json").write_text(
        json.dumps(baseline), encoding="utf-8")
    result = real.confirm()
    assert result["ok"] is True
    assert result["policy_artifact"]["sha256"] == policy["sha256"]


def test_confirmation_rejects_a_mutated_directory_policy(tmp_path):
    real = _declaring(research(tmp_path), {"seed": 101})
    baseline = real.measure(settings={"seed": 0}, label="baseline")
    original = Path(baseline["policy_artifact"]["path"])
    directory = real.run_root / "directory_checkpoint"
    directory.mkdir()
    (directory / "model.pth").write_bytes(original.read_bytes())
    policy = freeze_artifact(directory, real.run_root / "frozen_directory", max_bytes=1024)
    baseline["policy_artifact"] = policy
    (real.run_root / "measurements" / "baseline.json").write_text(
        json.dumps(baseline), encoding="utf-8")
    (Path(policy["path"]) / "model.pth").write_text("mutated", encoding="utf-8")
    result = real.confirm()
    assert result["ok"] is False
    assert result["where"] == "policy_identity"


def test_confirmation_refuses_to_compare_different_source_states_in_one_checkout(tmp_path):
    real = _declaring(research(tmp_path), {"seed": 101})
    source = real.repo / "controller.py"
    source.write_text("version = 1\n", encoding="utf-8")
    assert real.measure(settings={"seed": 0}, label="baseline")["ok"]
    real.snapshots.capture(["controller.py"], repo=real.repo, name="baseline")
    source.write_text("version = 2\n", encoding="utf-8")
    assert real.measure(settings={"seed": 0}, label="round_1")["ok"]
    real.snapshots.capture(["controller.py"], repo=real.repo, name="round-1")
    real.snapshots.advance("round-1", score=1.0, scale="success_rate")
    result = real.confirm()
    assert result["ok"] is False
    assert result["where"] == "baseline_source_identity"


def _isolated_source_controller(tmp_path):
    source = tmp_path / "source-base"
    source.mkdir()
    (source / "controller.py").write_text("baseline\n", encoding="utf-8")
    output = tmp_path / "out"
    checkout = output / "checkout"
    workspace = create(source, checkout)
    code = (
        "from pathlib import Path; text=Path('controller.py').read_text(); "
        "print('success_rate: 0.75' if text.strip() == 'candidate' else "
        "'success_rate: 0.25')")
    sources = {"evaluate": ("def stage_argv_evaluate(i):\n"
                            f"    return [i['python'], '-c', {code!r}]\n")}
    answer = {"stages": {"evaluate": {"available": True, "entrypoint": "score.py",
                                        "invocation": "python score.py", "artifact": "",
                                        "working_directory": "{repo}"}}}
    declaration = {"task_contract": {"policy_representation": "source"},
                   "research_goal": {"primary_metric": {"name": "success_rate",
                                                         "direction": "maximize"},
                                     "confirmation": {"seed": 101}}}
    (output / "derived_stages.json").write_text(json.dumps({
        "evaluate": {"source": sources["evaluate"], "parameters": {},
                     "row": answer["stages"]["evaluate"]}}), encoding="utf-8")
    (output / "execution.json").write_text(json.dumps({
        "repo": str(checkout), "stages": answer["stages"]}), encoding="utf-8")
    (output / "environment.json").write_text(json.dumps({
        "interpreter": sys.executable}), encoding="utf-8")
    from autosim.research.derived_research import DerivedResearch

    real = DerivedResearch(
        repo=checkout, output=output,
        backend=DeclarativeBackend(repo=checkout, answer=answer, sources=sources,
                                   parameters={}),
        interpreter=Path(sys.executable), space=OptimizationSpace(), client=None,
        stages=sources, run_id="source-controller", declaration=declaration,
        compute=ComputeDecision(device="cpu", device_index=0, environment={},
                                why="the test chose it"))
    assert workspace["destination"] == str(checkout)
    return real, checkout


def test_source_controller_measurement_round_trips_through_independent_reevaluation(tmp_path):
    from autosim.research.reevaluate import reevaluate

    real, checkout = _isolated_source_controller(tmp_path)
    measured = real.measure(settings={"seed": 0}, label="baseline")
    assert measured["ok"] and measured["metric_value"] == 0.25
    verdict = reevaluate(real.run_root, "baseline", output=tmp_path / "rechecked-baseline")
    assert verdict["status"] == "reproduced", verdict
    assert verdict["policy_matches"] is True
    assert (checkout / "controller.py").read_text() == "baseline\n"


def test_source_controller_confirmation_restores_baseline_and_candidate_separately(tmp_path):
    real, checkout = _isolated_source_controller(tmp_path)
    baseline = real.measure(settings={"seed": 0}, label="baseline")
    assert baseline["ok"] and baseline["metric_value"] == 0.25, baseline
    real._advance_best("baseline", score=0.25, scale="success_rate", why="baseline")

    (checkout / "controller.py").write_text("candidate\n", encoding="utf-8")
    real.touched.add("controller.py")
    candidate = real.measure(settings={"seed": 0}, label="round_1")
    assert candidate["ok"] and candidate["metric_value"] == 0.75
    real._advance_best("round-1", score=0.75, scale="success_rate", why="candidate")

    result = real.confirm()
    assert result["ok"] is True, result
    assert result["baseline_metric_value"] == 0.25
    assert result["metric_value"] == 0.75
    assert result["source_isolation"]["baseline"] == \
        "fresh_copy_from_verified_base_and_overlay"
    assert result["source_isolation"]["candidate"] == \
        "fresh_copy_from_verified_base_and_overlay"
    assert (checkout / "controller.py").read_text() == "candidate\n"
    assert result["confirmation_comparison"]["verdict"] == "not_established"


def test_confirmation_keeps_paired_statistics_separate_from_an_l4_claim(tmp_path):
    real = _declaring(research(tmp_path), {"seed": 101})
    real.metric_spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "source": "csv",
        "aggregation": "mean", "episode_id_column": "episode_id",
        "task_column": "task", "initial_state_hash_column": "initial_state_sha256"}}})
    real.backend.answer["stages"]["evaluate"]["artifact"] = "results.csv"
    script = (
        "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True); "
        "seed=int(sys.argv[2]); winner='round_1' in sys.argv[3]; "
        "rows=['task,episode_id,initial_state_sha256,success_rate']; "
        "rows += [f'lift,{n},{(str(seed)+str(n)).encode().hex().ljust(64,chr(48))},'"
        "+str(int(winner)) for n in range(20)]; "
        "(p/'results.csv').write_text('\\n'.join(rows)+'\\n')"
    )
    source = ("def stage_argv_evaluate(i):\n"
              f"    return [i['python'], '-c', {script!r}, i['output'], "
              "str(i['seed']), i['checkpoint']]\n")
    real.backend.sources["evaluate"] = source
    real.backend = DeclarativeBackend(repo=real.repo, answer=real.backend.answer,
                                      sources=real.backend.sources, parameters={})
    baseline = real.measure(settings={"seed": 0}, label="baseline")
    candidate = real.measure(settings={"seed": 0}, label="round_1")
    assert baseline["ok"] and candidate["ok"]
    real.snapshots.capture([], repo=real.repo, name="round-1")
    real.snapshots.advance("round-1", score=1.0, scale="success_rate")
    result = real.confirm()
    assert result["ok"] is True
    comparison = result["confirmation_comparison"]
    assert comparison["paired_initial_states"] == 20
    assert comparison["statistical_evidence"]["verdict"] == "better"
    assert comparison["verdict"] == "not_established"


def test_failed_confirmation_attempts_remain_visible_after_retry(tmp_path):
    real = _declaring(research(tmp_path), {"seed": 101})
    real.measure(settings={"seed": 0}, label="baseline")
    sources = dict(real.backend.sources)
    real.backend.sources.pop("evaluate")
    assert real.confirm()["ok"] is False
    real.backend.sources.update(sources)
    assert real.confirm()["ok"] is True
    attempts = list((real.run_root / "confirmation_attempts").glob("*.json"))
    assert len(attempts) == 2
    assert sorted(json.loads(p.read_text())["ok"] for p in attempts) == [False, True]


def test_exposed_heldout_evaluation_cannot_be_retried_after_candidate_failure(tmp_path):
    real = _declaring(research(tmp_path), {"seed": 101})
    assert real.measure(settings={"seed": 0}, label="baseline")["ok"]
    assert real.measure(settings={"seed": 0, "train.n_epochs": 3}, label="round_1")["ok"]
    real.snapshots.capture([], repo=real.repo, name="round-1")
    real.snapshots.advance("round-1", score=1.0, scale="success_rate")
    original = real.run_stage
    calls = []

    def fail_candidate(stage, **kwargs):
        calls.append(stage)
        if len(calls) == 2:
            return {"stage": stage, "ran": False, "status": "failed",
                    "returncode": 1, "attempt_id": "candidate-failed"}
        return original(stage, **kwargs)

    real.run_stage = fail_candidate
    assert real.confirm()["ok"] is False
    assert calls == ["evaluate", "evaluate"]
    real.run_stage = original
    assert real.confirmation_state()["retryable"] is False
    refused = real.confirm()
    assert refused["ok"] is False
    assert "already launched" in refused["why"]


def test_a_confirmation_cannot_be_taken_twice(tmp_path):
    """Measuring the held-out set repeatedly until it looks good is the failure the split
    exists to prevent, and it is indistinguishable in a record from having measured once."""
    real = _declaring(research(tmp_path), {"seed": 101})
    real.measure(settings={"seed": 0}, label="baseline")
    assert real.confirm()["ok"] is True
    second = real.confirm()
    assert second["ok"] is False
    assert "already confirmed" in second["why"]
    assert "a second one would be a search of the held-out set" in second["why"]


def test_a_confirmation_already_on_disk_survives_a_restart(tmp_path):
    """Read from the file rather than from memory: the value of a held-out measurement is that
    it happened once, and a resumed run is still the same run."""
    real = _declaring(research(tmp_path), {"seed": 101})
    real.measure(settings={"seed": 0}, label="baseline")
    real.confirm()
    assert _load_confirmation(real.run_root)
    assert real.confirm()["ok"] is False


def test_reserving_a_value_the_search_already_uses_is_refused(tmp_path):
    """A "held-out" value equal to the search's own is a second measurement of the same
    episodes wearing the word confirmation, and it would be reported as independent of them."""
    real = _declaring(research(tmp_path), {"seed": 0})
    refused = real.measure(settings={"seed": 0}, label="baseline")
    assert refused["where"] == "comparison_protocol"
    assert "which reserves nothing" in refused["why"]


def test_reserving_a_key_the_protocol_does_not_freeze_is_refused(tmp_path):
    """The quiet failure: a key outside the frozen set is silently ignored, so the
    confirmation runs on the search's own episodes while the record says it did not."""
    real = _declaring(research(tmp_path), {"learning_rate": 0.001})
    refused = real.measure(settings={"seed": 0}, label="baseline")
    assert refused["where"] == "comparison_protocol"
    assert "does not freeze" in refused["why"]
    assert "would have no effect" in refused["why"]


def test_a_declared_key_beyond_the_generic_names_is_allowed(tmp_path):
    """Benchmarks name their own evaluation switches, and `protocol_keys` is how a run says
    which ones count. A split on one of those is as real as a split on a seed."""
    real = research(tmp_path)
    real.declaration = {**real.declaration,
                        "research_goal": {"protocol_keys": ["rollout_length"],
                                          "confirmation": {"rollout_length": 500}}}
    assert real.measure(settings={"rollout_length": 100}, label="baseline")["ok"] is True
    assert real.measure(settings={"rollout_length": 500}, label="candidate")["ran"] is False
    confirmed = real.confirm()
    assert confirmed["ok"] is True
    assert confirmed["settings"]["rollout_length"] == 500


def test_a_run_that_declares_no_split_says_so_rather_than_inventing_one(tmp_path):
    """No split is the ordinary case for a benchmark whose evaluation is fixed. The run then
    has nothing to confirm on, and saying that is better than measuring its own episodes again
    and calling the result independent."""
    real = research(tmp_path)
    real.measure(settings={"seed": 0}, label="baseline")
    refused = real.confirm()
    assert refused["ok"] is False
    assert refused["where"] == "confirmation"
    assert "nothing this run can confirm on" in refused["why"]
    assert "stays the maximum of the arms it measured" in refused["why"]
    assert not (real.run_root / "confirmation.json").exists()


def test_a_malformed_split_is_refused_rather_than_read_as_no_split(tmp_path):
    """A declaration the system cannot parse must not silently become 'no split declared' --
    the run would then search freely and report as if it had held something out."""
    real = _declaring(research(tmp_path), ["seed"])
    refused = real.measure(settings={"seed": 0}, label="baseline")
    assert refused["where"] == "comparison_protocol"
    assert "must be an object mapping a setting name" in refused["why"]


def test_the_frozen_protocol_records_the_split(tmp_path):
    """A reader of the run directory has to be able to tell that the search was bounded, and
    which episodes it was bounded away from."""
    real = _declaring(research(tmp_path), {"seed": 101})
    real.measure(settings={"seed": 0}, label="baseline")
    protocol = json.loads((real.run_root / "comparison_protocol.json").read_text(
        encoding="utf-8"))
    assert protocol["settings"] == {"seed": 0}
    assert protocol["confirmation"] == {"seed": 101}


def test_the_split_itself_cannot_change_once_frozen(tmp_path):
    """Moving the held-out episodes after the search ran would make them the search's episodes
    retroactively, which is the one thing the split exists to prevent."""
    real = _declaring(research(tmp_path), {"seed": 101})
    real.measure(settings={"seed": 0}, label="baseline")
    real.declaration = {**real.declaration, "research_goal": {"confirmation": {"seed": 202}}}
    refused = real.measure(settings={"seed": 0}, label="candidate")
    assert refused["where"] == "comparison_protocol"
    assert "held-out split" in refused["why"]


def test_a_failed_confirmation_is_not_recorded_as_one(tmp_path):
    """The held-out set costs what it costs and is spent once. A confirmation whose own
    measurement died has not confirmed anything, and marking it spent would end the run's one
    chance to check its result."""
    real = _declaring(research(tmp_path), {"seed": 101})
    real.measure(settings={"seed": 0}, label="baseline")
    real.backend.sources.pop("evaluate")
    real.backend.answer["stages"]["evaluate"]["available"] = False
    assert real.confirm()["ok"] is False
    assert _load_confirmation(real.run_root) == {}


def test_a_run_that_declared_a_split_and_never_measured_it_says_so(tmp_path):
    """The report is what a reader has. A held-out set that was declared and left unmeasured
    means the best number above it is the maximum of the search's own arms -- and a reader who
    is not told that will assume it was checked."""
    real = _declaring(research(tmp_path), {"seed": 101})
    report = real.run(rounds=0)
    assert report["confirmation"]["status"] == "available_not_taken"
    assert report["confirmation"]["held_out"] == {"seed": 101}
    assert "maximum of the search's own arms" in report["confirmation"]["why"]


def test_a_run_asked_to_confirm_records_the_one_checked_number(tmp_path):
    real = _declaring(research(tmp_path), {"seed": 101})
    report = real.run(rounds=0, confirm=True)
    confirmation = report["confirmation"]
    assert confirmation["status"] == "taken"
    assert confirmation["metric_value"] == 0.5
    assert confirmation["held_out"] == {"seed": 101}


def test_a_run_with_no_declared_split_reports_that_it_has_none(tmp_path):
    """Not a failure and not the same as a split that went unmeasured: the ordinary case of a
    benchmark whose evaluation is fixed, and the report has to distinguish the two."""
    real = research(tmp_path)
    report = real.run(rounds=0)
    assert report["confirmation"]["status"] == "not_declared"
    assert "no episodes the search did not see" in report["confirmation"]["why"]


def test_the_document_states_the_confirmation_rather_than_leaving_it_to_be_inferred(tmp_path):
    from autosim.research import run_record

    real = _declaring(research(tmp_path), {"seed": 101})
    real.run(rounds=0, confirm=True)
    document = run_record.generate(real.run_root).read_text(encoding="utf-8")
    assert "未参与搜索的确认测量" in document
    assert "seed=101" in document
    assert "同协议留出基线" in document
    assert "not_established" in document


def test_the_document_says_when_a_declared_split_went_unmeasured(tmp_path):
    from autosim.research import run_record

    real = _declaring(research(tmp_path), {"seed": 101})
    real.run(rounds=0)
    document = run_record.generate(real.run_root).read_text(encoding="utf-8")
    assert "声明了未参与搜索的设置却没有测" in document
    assert "最大值" in document


def test_a_run_asked_to_confirm_but_declaring_nothing_reports_why_not(tmp_path):
    """`--confirm` on a run with no declared split must not silently do nothing, and must not
    measure the search's own episodes and call the result a confirmation."""
    real = research(tmp_path)
    report = real.run(rounds=0, confirm=True)
    assert report["confirmation"]["status"] == "available_not_taken" or \
        report["confirmation"]["status"] == "not_declared", report["confirmation"]
    assert not (real.run_root / "confirmation.json").exists()


def test_a_failed_confirmation_is_distinct_from_never_having_tried(tmp_path):
    """"The split went unmeasured" and "we measured it and got nothing" are different
    findings, and a record that reports the second as the first has lost the wall clock, the
    GPU hours, and the fact that the run tried."""
    real = _declaring(research(tmp_path), {"seed": 101})
    real.measure(settings={"seed": 0}, label="baseline")
    real.backend.sources.pop("evaluate")
    real.backend.answer["stages"]["evaluate"]["available"] = False
    real.confirm()
    state = real.confirmation_state()
    assert state["status"] == "attempted_and_failed"
    assert state["retryable"] is True
    assert state["why"]


def test_the_document_separates_a_failed_confirmation_from_an_untried_one(tmp_path):
    from autosim.research import run_record

    real = _declaring(research(tmp_path), {"seed": 101})
    real.measure(settings={"seed": 0}, label="baseline")
    real.backend.sources.pop("evaluate")
    real.backend.answer["stages"]["evaluate"]["available"] = False
    real.run(rounds=0, confirm=True)
    document = run_record.generate(real.run_root).read_text(encoding="utf-8")
    assert "未参与搜索的确认测量失败了" in document
    assert "评测尚未启动，修复启动条件后可重试" in document


def test_a_failed_confirmation_can_be_tried_again(tmp_path):
    """It establishes nothing, so it does not spend the run's one chance -- but the record of
    the first attempt stays until a second one succeeds."""
    real = _declaring(research(tmp_path), {"seed": 101})
    real.measure(settings={"seed": 0}, label="baseline")
    sources, answers = dict(real.backend.sources), dict(real.backend.answer["stages"]["evaluate"])
    real.backend.sources.pop("evaluate")
    real.backend.answer["stages"]["evaluate"]["available"] = False
    assert real.confirm()["ok"] is False
    real.backend.sources.update(sources)
    real.backend.answer["stages"]["evaluate"].update(answers)
    second = real.confirm()
    assert second["ok"] is True
    assert real.confirmation_state()["status"] == "taken"
