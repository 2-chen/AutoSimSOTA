"""The machinery that lets the system write a reader instead of shipping one.

No model is called here. What is tested is everything around the model: the syntax boundary
that decides what may run, the differential that decides whether it is right, and the two
gates' failure messages. A generated function that is wrong has to be rejected with a
message specific enough to correct, or the loop that produces it cannot converge -- so the
messages are the thing worth asserting.
"""

from pathlib import Path

import pytest

from autosim.research.contract_codegen import (FUNCTION, differential, gold_cases,
                                               _disagreement_summary, _offending_call, oracle)

PROJECT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name("fixtures") / "robosyn_task_contracts.json"
CORRECT = Path(__file__).with_name("fixtures") / "task_contract_reference.py"
pytestmark = pytest.mark.skipif(not FIXTURE.is_file(), reason="oracle fixture not captured")

#: What the reference reader is given that cannot be derived from a document: a task's
#: semantic family is a label with no config equivalent, so it is declared rather than
#: guessed, and the event-name-to-family map is a fact about the benchmark's randomizers.
def _declared(task):
    from autosim.research.registry import (CORRECTION_CAPABLE, EVENT_FAMILIES,
                                           EXPERT_ADAPTERS, TASK_ROLES)
    return {"setting": "random", "family": TASK_ROLES[task]["family"],
            "correction_supported": task in CORRECTION_CAPABLE,
            "expert_adapter": EXPERT_ADAPTERS.get(task, "official"),
            "event_families": EVENT_FAMILIES}


def test_a_function_that_reads_no_files_cannot_be_made_to(tmp_path):
    """The boundary is the allow-list, so it is worth asserting it actually holds."""
    from autosim.research.contract_codegen import checked_function
    for source in ('def task_contract(d):\n    import os\n    return {}\n',
                   'def task_contract(d):\n    return open("/etc/passwd").read()\n',
                   'def task_contract(d):\n    return d.get("x")\n',
                   'def task_contract(d):\n    return d["x"].values()\n'):
        with pytest.raises(ValueError):
            checked_function(source, FUNCTION)


def test_the_rejection_names_the_call_it_refused():
    refused = _offending_call('def task_contract(d):\n    return d.get("x")\n')
    assert "d.get('x')" in refused or 'd.get("x")' in refused


def test_the_rejection_survives_source_that_does_not_parse():
    """A candidate that does not parse is a rejected draft, not a crash."""
    assert _offending_call("def task_contract(d):\n    return ]\n") == ""


def test_the_differential_accepts_a_faithful_reader(tmp_path):
    oracle_doc = oracle(FIXTURE)
    repo = _checkout()
    if repo is None:
        pytest.skip("checkout not found")
    cases = gold_cases(repo, oracle_doc,
                       declared={t: _declared(t) for t in oracle_doc["tasks"]})
    report = differential(CORRECT.read_text(encoding="utf-8"), cases)
    assert report["passed"], [r for r in report.get("rows", []) if not r["passed"]][:2]


def test_the_differential_reports_the_field_and_the_task_that_disagree(tmp_path):
    oracle_doc = oracle(FIXTURE)
    repo = _checkout()
    if repo is None:
        pytest.skip("checkout not found")
    cases = gold_cases(repo, oracle_doc,
                       declared={t: _declared(t) for t in oracle_doc["tasks"]})
    # Identical to the reference except that it sorts the cameras, which the benchmark's
    # own reader does not -- the kind of deviation that looks correct and is not.
    source = CORRECT.read_text(encoding="utf-8").replace(
        'cameras = [s["uid"] for s in sensors]',
        'cameras = sorted([s["uid"] for s in sensors])')
    assert source != CORRECT.read_text(encoding="utf-8")
    report = differential(source, cases)
    assert not report["passed"]
    failures = [row for row in report["rows"] if not row["passed"]]
    assert failures, "a sorted camera list must disagree with an unsorted one"
    assert all("cameras" in row["disagreements"][0] for row in failures)
    # And the summary a retry would be given names the task and the field.
    summary = _disagreement_summary(report)
    assert "cameras" in summary and failures[0]["task"] in summary


def test_a_candidate_that_raises_is_evidence_rather_than_a_crash():
    # `raise` is not in the allow-list, so the candidate fails the way a real one does:
    # by reaching for a key the document does not have.
    source = CORRECT.read_text(encoding="utf-8").replace(
        '    gym = documents["gym"]', '    gym = documents["gym"]["absent_key"]')
    report = differential(source, [{"task": "any", "input": {"task": "t"},
                                    "expected": {"name": "t"}}])
    assert not report["passed"]
    assert "KeyError" in report["rows"][0]["disagreements"][0]


def _checkout():
    from autosim.research.common import find_benchmark
    return find_benchmark("RoboSynChallenge", PROJECT.parent, marker="scripts/eval_policy.py")
