"""What the loop can see about its own search that each attempt cannot.

AutoSOTA puts this role outside the loop on purpose and has it emit "high-level corrective
guidance rather than low-level patch instructions". That distinction is the opposite of what a
rule-based system does: a monitor that says "set PYTHONPATH to X" has taken the decision away
from the thing that can read the repository, while one that says "six attempts have changed
the environment and the environment has not changed" has told the agent something about itself
and left the decision where it belongs. Every test below is about that line.
"""

import json

from autosim.research import monitor


def _round(change, state=None, looked=0, failure="it failed"):
    return {"change": change, "state": state or {"working_directory": "{repo}"},
            "failure": failure, "looked": looked}


def test_a_round_that_leaves_the_state_alone_has_moved_nothing():
    """The state is the stage's configuration, and it is what "progress" has to mean. Rounds
    are counted by the loop and would call every one of them a round -- nineteen attempts at a
    package that was never the problem were nineteen rounds."""
    same = {"environment": {"A": "1"}}
    picture = monitor.observe([_round(same), _round({"environment": {"A": "1"}},
                                                   state={"working_directory": "{repo}"})])
    assert picture["moved"] is False
    assert any("has not changed" in one for one in picture["signals"])

    # A round that changes the state has moved it, whatever else it did or did not do.
    moved = monitor.observe([_round({}, state={"working_directory": "{repo}"}),
                             _round({}, state={"working_directory": "{repo}/policy"})])
    assert moved["moved"] is True
    assert not any("has not changed" in one for one in moved["signals"])


def test_the_same_field_changed_again_and_again_is_said_out_loud():
    """Six environment edits in a row that did not fix it is a finding about the approach, and
    a different finding from six argument edits that did not fix it."""
    picture = monitor.observe([_round({"environment": {"N": str(n)}}) for n in range(5)])
    assert picture["changed_most"] == {"environment": 5}
    assert "5 of 5 attempts changed environment" in picture["phase"]
    assert any("the last 5 attempts changed environment" in one for one in picture["signals"])

    # Two is not a pattern.
    quiet = monitor.observe([_round({"environment": {"N": "1"}}), _round({"environment": {"N": "2"}})])
    assert not any("attempts changed" in one for one in quiet["signals"])


def test_never_looking_is_a_thing_the_loop_can_notice_about_itself():
    """The single most expensive habit this loop had: proposing changes from the payload it was
    handed, when the machine held the answer one inspection away."""
    blind = monitor.observe([_round({"parameters": {"a": str(n)}}) for n in range(5)])
    assert any("nothing has been looked at" in one for one in blind["signals"])

    looking = monitor.observe([_round({"parameters": {"a": str(n)}}, looked=1) for n in range(5)])
    assert not any("nothing has been looked at" in one for one in looking["signals"])
    assert looking["attempts_that_looked"] == 5


def test_the_guidance_names_the_shape_and_never_the_fix():
    """The line this module is built on. A monitor that names a fix has made a decision the
    reader of the repository was better placed to make."""
    picture = monitor.observe([_round({"environment": {"N": str(n)}}) for n in range(5)])
    said = monitor.guidance(picture)
    assert "the last 5 attempts changed environment" in said
    assert "not a diagnosis of the program" in said
    # No field name from the vocabulary is proposed as a value to set.
    for forbidden in ("set ", "should be", "try ", "use ", "PYTHONPATH="):
        assert forbidden not in said.lower(), said


def test_an_empty_trace_says_so_rather_than_drawing_a_conclusion():
    assert monitor.observe([]) == {"rounds": 0, "phase": "nothing tried yet", "moved": False,
                                   "signals": []}
    assert monitor.guidance(monitor.observe([])) == ""


def test_the_picture_is_json_serialisable_because_it_travels_in_a_payload():
    picture = monitor.observe([_round({"environment": {"N": "1"}}, looked=2)])
    json.dumps(picture)
    assert set(picture) >= {"rounds", "phase", "moved", "signals", "changed_most"}
