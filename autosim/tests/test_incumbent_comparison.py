"""A last failed cursor must not erase an adopted development incumbent."""

import pytest

from autosim.research.common import read_json
from autosim.research.derived_research import DerivedResearch
from autosim.research.ideas import Idea
from autosim.research.declarative_backend import DeclarativeBackend
from tests.test_derived_research import research


@pytest.mark.parametrize("before", [False, True])
@pytest.mark.parametrize("score,helped", [(0.21, False), (0.23, False), (0.24, True)])
def test_incumbent_takes_precedence_over_failed_cursor_and_original_baseline(before, score, helped):
    actual, why = DerivedResearch._did_it_help(
        idea=None, before_measured=before, measured=True,
        result={"metric_value": score, "metric_utility": score},
        baseline=0, incumbent=0.23)
    assert actual is helped
    assert "incumbent" in why and "0.23" in why
    assert "none came out" not in why


def test_minimized_metric_compares_normalized_utility():
    helped, _ = DerivedResearch._did_it_help(
        idea=None, before_measured=False, measured=True,
        result={"metric_value": 2, "metric_utility": -2},
        baseline=-10, incumbent=-1)
    assert not helped


@pytest.mark.parametrize("status", ["unknown", "violated"])
def test_guardrails_still_block_primary_improvement(status):
    helped, why = DerivedResearch._did_it_help(
        idea=None, before_measured=False, measured=True,
        result={"metric_utility": 1, "guardrails": {"status": status}},
        baseline=0, incumbent=0.23)
    assert not helped and "secondary" in why


def test_first_valid_measurement_without_incumbent_remains_progress():
    helped, why = DerivedResearch._did_it_help(
        idea=None, before_measured=False, measured=True,
        result={"metric_utility": 0}, baseline=None)
    assert helped and "none came out" in why


def test_proposal_schema_rejection_names_exact_missing_and_extra_fields(tmp_path):
    real = research(tmp_path)
    candidate = {'label': 'new hypothesis', 'granularity': 'code', 'mechanism': 'bounded idea',
        'change': {}, 'risk': 'low', 'crosses': 'none', 'why': 'observed evidence', 'touches': [],
        'hypothesis': 'misnamed extra field'}
    result = real.propose_research_idea(candidate)
    assert result['status'] == 'invalid'
    assert result['missing_fields'] == ['evidence']
    assert result['unexpected_fields'] == ['hypothesis']
    assert "missing=['evidence']" in result['why']
    assert real.library.ideas == []


def test_loop_records_comparison_against_best_without_changing_baseline(tmp_path):
    real = research(tmp_path)
    real.library.add(Idea(label="bounded candidate", granularity="param",
                          change={"train.n_epochs": 2}, risk="low", crosses="none",
                          why="exercise the generic candidate comparison"))
    real.run(rounds=1, yield_after_action=True, max_rounds_per_action=1)
    baseline_path = real.run_root / "measurements/baseline.json"
    baseline_bytes = baseline_path.read_bytes()
    # Model a previously adopted development snapshot without rewriting the
    # baseline/cursor. Native CPU fixture experiments remain the same.
    best = real.snapshots.capture([], repo=real.repo, name="linked-incumbent")
    real.snapshots.advance(best.name, score=0.75, scale="success_rate")
    report = real.run(rounds=1, yield_after_action=True, max_rounds_per_action=1)
    candidate = report["rounds"][-1]
    assert candidate["comparison_reference"] == {
        "name": "linked-incumbent", "utility": 0.75, "scale": "success_rate"}
    assert "incumbent" in candidate["verdict"]
    assert real.snapshots.best().name == "linked-incumbent"
    assert baseline_path.read_bytes() == baseline_bytes
    assert read_json(real.run_root / "controller_session.json")["pending_action"] is None


def test_native_cpu_candidate_above_original_baseline_but_below_incumbent_rolls_back(tmp_path):
    real = research(tmp_path)
    policy = tmp_path / "controller.py"
    policy.write_text("VALUE = 0.5\n")
    sources = dict(real.sources)
    sources["evaluate"] = (
        "def stage_argv_evaluate(i):\n"
        "    return [i['python'], '-c', "
        "'import controller; print(\"success_rate:\", controller.VALUE)']\n")
    real.sources = sources
    real.backend = DeclarativeBackend(repo=tmp_path, answer=real.backend.answer,
                                      sources=sources, parameters={})
    real.library.add(Idea(label="controller candidate", granularity="code",
        touches=["controller.py"], risk="low", crosses="none",
        change={"file": "controller.py", "find": "VALUE = 0.5", "replace": "VALUE = 0.6"},
        why="synthetic CPU controller fixture, not a benchmark improvement"))
    real.run(rounds=1, yield_after_action=True, max_rounds_per_action=1)
    baseline = (real.run_root / "measurements/baseline.json").read_bytes()
    best = real.snapshots.capture(["controller.py"], repo=real.repo, name="linked-incumbent")
    real.snapshots.advance(best.name, score=0.75, scale="success_rate")
    report = real.run(rounds=1, yield_after_action=True, max_rounds_per_action=1)
    candidate = report["rounds"][-1]
    assert candidate["metric_value"] == 0.6
    assert candidate["undone"] is True and "incumbent" in candidate["verdict"]
    assert policy.read_text() == "VALUE = 0.5\n"
    assert real.snapshots.best().name == "linked-incumbent"
    assert (real.run_root / "measurements/baseline.json").read_bytes() == baseline
