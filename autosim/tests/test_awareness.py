"""What the system can say about its own state, tested against the faults it missed.

Each case here is a fault that cost real time and was found by a person reading output. The
point of the module is that the material was already in the record; these tests are that the
reading happens now.
"""

import json
import time
from pathlib import Path

from autosim.research import awareness


def measurement(root: Path, **fields) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{fields.get('stage', 'train')}.json"
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


# -- the cost model: what "too long" means, from runs that already happened -----------------

def test_the_expected_duration_comes_from_earlier_runs_of_the_same_command():
    history = [{"stage": "train", "seconds": 3600, "returncode": 0},
               {"stage": "train", "seconds": 7200, "returncode": 0},
               {"stage": "train", "seconds": 3800, "returncode": 0},
               {"stage": "evaluate", "seconds": 40, "returncode": 0}]
    assert awareness.cost_of(history, "train") == 3800      # the median, not the mean
    assert awareness.cost_of(history, "evaluate") is None   # too few to say
    assert awareness.cost_of([], "train") is None


def test_a_single_run_is_not_enough_to_have_an_expectation():
    """`None` is "not enough to say", which is a different answer from "it takes no time"."""
    assert awareness.cost_of([{"stage": "train", "seconds": 9e4, "returncode": 0}],
                             "train") is None


def test_a_run_that_hung_does_not_raise_the_expectation_for_the_next_one():
    """One run sat for ten hours. A mean would make the next hang look normal."""
    history = [{"stage": "train", "seconds": 7200, "returncode": 0},
               {"stage": "train", "seconds": 7600, "returncode": 0},
               {"stage": "train", "seconds": 36000, "returncode": 0}]
    assert awareness.cost_of(history, "train") == 7600


# -- the three things it could not see -----------------------------------------------------

def test_a_run_that_named_no_numbers_while_running_for_hours_is_a_contradiction():
    """Four runs printed losses and success rates for hours and were written down as `{}`.

    A stage that printed `train loss: 5.36` cannot have produced no named numbers, and the
    contradiction was in the record the moment it was written.
    """
    record = {"stage": "train", "seconds": 14400, "readings": {},
              "said": "did not finish within 14400s\n[info] Epoch: 0 | train loss: 5.36"}
    found = awareness.readings_contradiction(record)
    assert found is not None and found.kind == "contradiction"
    assert "no number" in found.detail
    # A short stage that named nothing is not a contradiction -- it may simply print nothing.
    assert awareness.readings_contradiction(
        {"stage": "train", "seconds": 3, "readings": {}, "said": "x"}) is None
    # And a stage that did name numbers is fine.
    assert awareness.readings_contradiction(
        {"stage": "train", "seconds": 9999, "readings": {"loss": 5.36}, "said": "y"}) is None


def test_a_run_past_its_expected_duration_is_overdue(tmp_path):
    awareness.began(tmp_path, stage="train", argv=["python", "main.py"],
                    expected_seconds=3600, device="cuda:0")
    assert awareness.stalled(tmp_path, now=time.time() + 3600) is None       # on time
    found = awareness.stalled(tmp_path, now=time.time() + 3600 * 4)
    assert found is not None and found.kind == "overdue"
    assert "expected" in found.detail
    awareness.ended(tmp_path)
    assert awareness.stalled(tmp_path, now=time.time() + 3600 * 40) is None


def test_a_stage_with_no_expectation_is_not_called_overdue(tmp_path):
    """Inventing an expectation would make the first run of anything an anomaly."""
    awareness.began(tmp_path, stage="train", argv=["x"], expected_seconds=None)
    assert awareness.stalled(tmp_path, now=time.time() + 86400 * 7) is None
    awareness.ended(tmp_path)


def test_a_failure_nothing_read_is_reported_as_undiagnosed(tmp_path):
    """The derivation loop reads failures and revises; the research loop files them.

    The same error got two different responses depending on which loop it landed in, and a
    declaration error that survived two whole attempts landed in the one that only files.
    """
    path = measurement(tmp_path, stage="train", returncode=1, said="AttributeError: ...")
    filed = {"stage": "train", "event": "stage", "returncode": 1}
    found = awareness.observe(tmp_path, measurements=[path], events=[filed])
    assert [row.kind for row in found] == ["undiagnosed"]

    looked_at = {"stage": "train", "event": "device_reconsidered"}
    assert [row.kind for row in awareness.observe(tmp_path, measurements=[path],
                                                  events=[filed, looked_at])] == []
    # A run that succeeded is not undiagnosed.
    ok = measurement(tmp_path / "ok", stage="train", returncode=0)
    assert awareness.observe(tmp_path / "ok", measurements=[ok]) == []


def test_a_check_that_fails_is_reported_as_a_blind_spot_not_raised(tmp_path):
    """A diagnostic that dies on bad input dies at the worst moment.

    A check added to this repository to catch a class of error crashed on a `None` and took
    down a measurement that had already screened two candidates successfully.
    """
    bad = tmp_path / "train.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("this is not json", encoding="utf-8")
    found = awareness.observe(tmp_path, measurements=[bad])       # must not raise
    assert all(row.kind in ("blind", "contradiction", "undiagnosed") for row in found)


def test_the_reading_is_a_sentence_and_not_a_dump():
    nothing = awareness.describe([])
    assert "nothing" in nothing
    said = awareness.describe([awareness.Observation("overdue", "train", "running 10h")])
    assert said == "[overdue] train: running 10h"


def test_a_repair_that_reduces_faults_by_claiming_less_is_refused():
    """Fewer faults is necessary and not sufficient.

    RoboTwin's declaration came back with no optimization space at all: a verification repair
    replaced thirty-seven declared axes with none, and was accepted because the only thing
    compared was the number of path faults -- and a document that declares less has fewer of
    those. The axes were twenty-five collection knobs and twelve training ones, each citing
    the file it had been read from.
    """
    from autosim.research.scout import declared_content

    full = {"optimization_space": {"collection": [{"name": "a"}, {"name": "b"}],
                                   "training": [{"name": "c"}]},
            "capabilities": {"native_evaluation": {"status": "declared"},
                             "training": {"status": "unknown"}}}
    gutted = {"optimization_space": {"collection": [], "training": []},
              "capabilities": {"native_evaluation": {"status": "declared"},
                               "training": {"status": "unknown"}}}
    assert declared_content(full) == (3, 1)
    assert declared_content(gutted) == (0, 1)
    # The rule, as the caller applies it: faults must fall AND nothing declared may shrink.
    for repair, accepted in ((full, True), (gutted, False)):
        before, after = declared_content(full), declared_content(repair)
        assert (all(new >= old for new, old in zip(after, before))) is accepted
