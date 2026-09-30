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

#: Which family each of this benchmark's randomiser functions belongs to. This is *input*:
#: a fact a benchmark declares about itself, which the reader cannot derive from a config
#: file. It lives here, as test data, because the alternative was a table of one benchmark's
#: task names, families and randomisers kept inside the package -- knowledge no other
#: benchmark could use or replace. Every other value the reader needs is read back out of
#: the frozen contract document this test compares against.
RANDOMISER_FAMILIES = {
    "randomize_light": "appearance", "randomize_visual_material": "appearance",
    "randomize_camera_intrinsics": "camera", "randomize_camera_extrinsics": "camera",
    "randomize_robot_eef_pose": "robot_pose", "randomize_robot_qpos": "robot_pose",
    "replace_distractor_slots_from_library": "clutter",
}


def _declared(row):
    """What the reference reader is given that cannot be derived from a document.

    A task's semantic family is a label with no config equivalent, so it is declared rather
    than guessed. The values come from the frozen contract document -- they are what that
    document records the benchmark declaring, read back rather than restated.
    """
    return {"setting": row.get("setting") or "random",
            "family": (row.get("roles") or {}).get("family") or "task_semantics",
            "correction_supported": bool(row.get("correction_supported")),
            "expert_adapter": row.get("expert_adapter") or "official",
            "event_families": RANDOMISER_FAMILIES}


@pytest.mark.parametrize("body", [
    'import os\n    return {}',
    'return open("/etc/passwd").read()',
    'return d["x"].__class__.__name__',
    'return d.read()',
    'return "{0.__class__}".format(d)',
])
def test_a_function_that_reads_no_files_cannot_be_made_to(body):
    """The boundary is the allow-list, so it is worth asserting it actually holds.

    The cases are the ones that matter -- opening a file, importing, and the three routes
    out of a live object (its class, a dunder, and a format string naming an attribute the
    syntax check cannot see). `d.get("x")` and `d.values()` used to be on this list because
    *every* method call was refused; they are pure operations on a dict the function was
    handed, and refusing them cost a whole attempt every time a generated function iterated
    one. See `test_validated_kernel_surface.py` for where that line now sits.
    """
    from autosim.research.contract_codegen import checked_function
    with pytest.raises(ValueError):
        checked_function(f"def field_x(d):\n    {body}\n", "field_x")


def test_the_rejection_names_the_call_it_refused():
    """A refusal that does not say what it refused leaves the model guessing."""
    refused = _offending_call('def field_x(d):\n    return d["x"].__class__\n')
    assert "__class__" in refused


def test_the_rejection_survives_source_that_does_not_parse():
    """A candidate that does not parse is a rejected draft, not a crash."""
    assert _offending_call("def field_x(d):\n    return ]\n") == ""


def _cases():
    oracle_doc = oracle(FIXTURE)
    repo = _checkout()
    if repo is None:
        pytest.skip("checkout not found")
    return gold_cases(repo, oracle_doc,
                      declared={t: _declared(row)
                                for t, row in oracle_doc["tasks"].items()})


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


def test_a_proposal_filled_in_from_the_skeleton_is_accepted():
    """The skeleton is the starting point the controller is shown, so every proposal built
    from it has to be one the validator accepts. Otherwise the system hands over a form that
    is invalid by construction and refuses it when it comes back.

    Measured on RoboTwin: the scout declared `seed_start` and `seed` as required integer axes
    with no default. The skeleton carried `null` in both, and a proposal copied verbatim was
    refused with "does not accept None; it takes a integer in [0.0, 1000000000.0]" -- which
    is a true statement about the axis and says nothing about what to do next.
    """
    from autosim.research.adapter_protocol import Axis, OptimizationSpace, check_space

    good = OptimizationSpace(
        collection=(Axis("episodes", "integer", "how many", low=1, high=100, default=10),),
        training=(
            Axis("steps", "integer", "how long", low=1, high=1000, default=100),
            Axis("seed", "integer", "the seed", low=0, high=10**9, default=0, optional=True),
            # No default and not optional: required, and nothing to require.
            Axis("start", "integer", "where to start", low=0, high=100),
        ))
    problems = check_space(good)
    assert len(problems) == 1
    message = problems[0]
    assert "start" in message and "no default and is not optional" in message
    # It says what to write, not only what is wrong.
    assert "mark it optional" in message

    # And an axis marked optional with no default is fine -- that is the fix.
    fixed = OptimizationSpace(
        collection=good.collection,
        training=tuple(a for a in good.training if a.name != "start") + (
            Axis("start", "integer", "where to start", low=0, high=100, optional=True),))
    assert check_space(fixed) == []


def test_a_structure_axis_that_accepts_null_is_not_called_unfillable():
    """The rule is "nothing the axis would accept is available", and the axis answers it.
    A structure axis with no validator takes null; one with a validator takes whatever the
    validator says. A kind test here would refuse the first."""
    from autosim.research.adapter_protocol import Axis, OptimizationSpace, check_space

    open_shape = OptimizationSpace(training=(
        Axis("bins", "structure", "any shape", validator=None),))          # accepts null
    assert check_space(open_shape) == []
    strict = OptimizationSpace(training=(
        Axis("bins", "structure", "must be a list", validator=lambda v: isinstance(v, list)),))
    assert len(check_space(strict)) == 1
