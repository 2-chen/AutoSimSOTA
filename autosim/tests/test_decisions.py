"""The decision record, and the property that makes it a record rather than a diary.

A run leaves stages that ran, numbers that came out, and artifacts that appeared. What it did
not leave was the reasoning -- written once into a model's response and gone. And a decision
whose consequence is never attached answers nothing: the system this borrows its shape from
defined a decision log, connected it to nothing, and left "why did the system do X" unanswerable.
"""

import json

from autosim.research.decisions import Decision, Decisions, outcome_of


def test_a_decision_is_recorded_with_the_grounds_it_was_made_on(tmp_path):
    log = Decisions(tmp_path)
    decision_id = log.record(Decision(
        activity="vary train.loss_scale=2.0", by="model", agent="deepseek",
        why="the loss plateaued at epoch 20 and the scale was the only axis not yet moved",
        used=["evidence/round_1.json"], round=1, produced=["rounds/round_1/proposal.json"]))
    assert decision_id

    rows = json.loads((tmp_path / "decisions.json").read_text())["rows"]
    assert len(rows) == 1
    assert rows[0]["by"] == "model" and "plateaued" in rows[0]["why"]
    assert rows[0]["used"] == ["evidence/round_1.json"]
    # Nothing has been done about it yet, and the record says so rather than implying the
    # decision had no consequences.
    assert rows[0]["outcome"] == {"state": "open"}


def test_closing_a_decision_is_what_makes_it_answer_something(tmp_path):
    log = Decisions(tmp_path)
    decision_id = log.record(Decision(activity="run on cuda:0", by="tool", why="nothing busy"))
    assert log.open_ids() == [decision_id]

    log.resolve(decision_id, {"ran": True, "success_rate": 0.65})
    assert log.open_ids() == []
    row = json.loads((tmp_path / "decisions.json").read_text())["rows"][0]
    assert row["outcome"]["state"] == "known"
    assert row["outcome"]["what"]["success_rate"] == 0.65
    # And the file says who decided what, which a reader weighing the claim needs.
    assert row["by"] == "tool"


def test_a_decision_left_open_is_reported_as_open(tmp_path):
    """A decision nobody looked at afterwards is a finding about the run, not a decision that
    turned out to have no consequences. The two read the same in a record that drops them."""
    log = Decisions(tmp_path)
    log.record(Decision(activity="a", by="tool", why="w"))
    log.record(Decision(activity="b", by="model", why="w"))
    log.resolve(log.rows[0]["id"], {"ran": True})
    assert log.summary() == {"decisions": 2, "by": {"tool": 1, "model": 1}, "unresolved": 1,
                             "outcome_only": 0}


def test_a_decision_whose_maker_is_unrecognised_is_refused(tmp_path):
    """`by` is not decoration: a model's judgement and a rule's application are different
    kinds of claim, and a record that conflates them cannot be weighed."""
    log = Decisions(tmp_path)
    for maker in ("model", "tool", "human"):
        log.record(Decision(activity="a", by=maker, why="w"))
    assert log.summary()["by"] == {"model": 1, "tool": 1, "human": 1}
    try:
        log.record(Decision(activity="a", by="the system", why="w"))
        assert False, "an unrecognised maker was accepted"
    except ValueError as exc:
        assert "the system" in str(exc)


def test_resolving_something_that_was_never_opened_is_kept_and_marked(tmp_path):
    """It means a caller closed a decision it never recorded, which is worth being able to see
    rather than silently creating a decision with no grounds."""
    log = Decisions(tmp_path)
    log.resolve("never-opened", {"ran": True})
    assert log.summary()["outcome_only"] == 1
    assert log.summary()["decisions"] == 0


def test_the_record_survives_being_reopened(tmp_path):
    """A run is resumed, so the record is appended to rather than started again."""
    Decisions(tmp_path).record(Decision(activity="first", by="tool", why="w"))
    reopened = Decisions(tmp_path)
    assert len(reopened.rows) == 1
    reopened.record(Decision(activity="second", by="human", why="w"))
    assert len(Decisions(tmp_path).rows) == 2


def test_a_stage_outcome_is_the_verdict_and_not_a_second_copy_of_the_record(tmp_path):
    """The argv and the output are already in `measurements/`; duplicating them here would be
    the second copy that drifts."""
    ran = outcome_of({"ran": True, "returncode": 0, "seconds": 12.5,
                      "artifact": {"matched": 2}, "readings": {"loss": -2.5}, "argv": ["x"]})
    assert ran["ran"] and ran["readings"] == {"loss": -2.5} and "argv" not in ran
    assert outcome_of({"ran": False, "why": "collect unavailable"}) == {
        "ran": False, "why": "collect unavailable"}
