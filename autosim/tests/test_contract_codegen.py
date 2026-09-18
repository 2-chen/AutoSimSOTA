"""The machinery that lets the system write a reader instead of shipping one.

No model is called here. What is tested is everything around the model: the syntax boundary
that decides what may run, the differential that decides whether it is right, and the two
gates' failure messages. A generated function that is wrong has to be rejected with a
message specific enough to correct, or the loop that produces it cannot converge -- so the
messages are the thing worth asserting.
"""

from pathlib import Path

import pytest

from autosim.research.contract_codegen import (FIELD_ORDER, _offending_call,
                                               differential_reader, function_name, gold_cases,
                                               oracle)

PROJECT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name("fixtures") / "robosyn_task_contracts.json"
CORRECT = Path(__file__).with_name("fixtures") / "task_contract_fields.py"
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


def test_a_function_that_reads_no_files_cannot_be_made_to():
    """The boundary is the allow-list, so it is worth asserting it actually holds."""
    from autosim.research.contract_codegen import checked_function
    for body in ('import os\n    return {}',
                 'return open("/etc/passwd").read()',
                 'return d.get("x")',
                 'return d["x"].values()'):
        with pytest.raises(ValueError):
            checked_function(f"def field_x(d):\n    {body}\n", "field_x")


def test_the_rejection_names_the_call_it_refused():
    refused = _offending_call('def field_x(d):\n    return d.get("x")\n')
    assert "d.get('x')" in refused or 'd.get("x")' in refused


def test_the_rejection_survives_source_that_does_not_parse():
    """A candidate that does not parse is a rejected draft, not a crash."""
    assert _offending_call("def field_x(d):\n    return ]\n") == ""


def _cases():
    oracle_doc = oracle(FIXTURE)
    repo = _checkout()
    if repo is None:
        pytest.skip("checkout not found")
    return gold_cases(repo, oracle_doc,
                      declared={t: _declared(t) for t in oracle_doc["tasks"]})


NAMES = [function_name(f) for f in FIELD_ORDER]


def test_a_faithful_reader_passes_every_task():
    report = differential_reader(CORRECT.read_text(encoding="utf-8"), NAMES, _cases())
    assert report["passed"], [r for r in report["rows"] if not r["passed"]][:2]


def test_a_deviation_is_localised_to_the_field_that_has_it():
    """The property the decomposition is for: a wrong field names one function to fix."""
    source = CORRECT.read_text(encoding="utf-8").replace(
        'return [s["uid"] for s in d["gym"]["sensor"] if s["sensor_type"] == "Camera"]',
        'return sorted([s["uid"] for s in d["gym"]["sensor"] if s["sensor_type"] == "Camera"])')
    assert source != CORRECT.read_text(encoding="utf-8")
    report = differential_reader(source, NAMES, _cases())
    assert not report["passed"]
    assert list(report["field_failures"]) == ["field_cameras"]
    # And the report names the task and the values, so a repair is specific.
    detail = report["field_failures"]["field_cameras"][0]
    assert "cameras" in detail and "expected" in detail


def test_a_field_that_raises_is_blamed_on_itself():
    """Not on whichever function happens to be examined first."""
    source = CORRECT.read_text(encoding="utf-8").replace(
        '    return d["task"]', '    return d["absent"]["deeper"]')
    report = differential_reader(source, NAMES, _cases())
    assert list(report["field_failures"]) == ["field_name"]
    assert "KeyError" in report["field_failures"]["field_name"][0]


def test_one_broken_field_does_not_hide_the_others():
    """Every function is called, so a repaired field can be verified while another is not."""
    cases = [{"task": "t", "input": {"task": "t"}, "expected": {"name": "t", "setting": "s"}}]
    source = ("def field_name(d):\n    return d['task']\n\n"
              "def field_setting(d):\n    return d['absent']\n")
    report = differential_reader(source, ["field_name", "field_setting"], cases)
    assert list(report["field_failures"]) == ["field_setting"]
    assert report["rows"][0]["disagreements"] == [
        "setting: raised KeyError: 'absent'"]


def _checkout():
    from autosim.research.common import find_benchmark
    return find_benchmark("RoboSynChallenge", PROJECT.parent, marker="scripts/eval_policy.py")
