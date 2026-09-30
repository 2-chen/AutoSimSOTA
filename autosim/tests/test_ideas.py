"""What the run could do, thought about once, audited, and chosen from.

The property that makes this module worth having is the one its absence cost: **a loop whose
proposal shape is "a value from the declared space" can only ever tune.** AutoSOTA's numbers
put 33% of its improvements in tuning and 51% in changes to code. Everything here exists to
make the second kind reachable and safe -- a type on the idea, an audit before selection, and
a way to leave the library when one kind of move has been made enough.
"""

import json
from dataclasses import dataclass, field

import pytest

from autosim.research import ideas, redlines


@dataclass
class Idea:
    label: str = "an idea"
    mechanism: str = ""
    change: str = ""
    why: str = ""
    evidence: str = ""
    touches: list = field(default_factory=list)


LINES = redlines.RedLines(benchmark="Toy")


def _library(tmp_path) -> ideas.IdeaLibrary:
    return ideas.IdeaLibrary(tmp_path / "ideas.json")


# -- the library ----------------------------------------------------------------------------

def test_the_library_survives_being_reopened(tmp_path):
    """The run is resumed, and the library is what a person reads to see what it considered.
    An idea never selected exists nowhere else."""
    library = _library(tmp_path)
    library.add(ideas.Idea(label="more mass", granularity="param", risk="low",
                           change={"targeted_sampling_mass": 0.75}, why="the failures are "
                           "all in the targeted profile"))
    reopened = ideas.IdeaLibrary(tmp_path / "ideas.json")
    assert [one.label for one in reopened.ideas] == ["more mass"]
    assert reopened.get("more mass").change == {"targeted_sampling_mass": 0.75}
    assert reopened.get("more mass").why


def test_candidates_must_fit_the_selected_native_stage_contract():
    params = {"--total_timesteps": {"value": "1024"},
              "--num_envs": {"value": "16"}}
    supported = ideas.Idea(label="more environment steps", granularity="param",
                           change={"training": {"total_timesteps": 2048}})
    unrelated = ideas.Idea(label="offline optimizer", granularity="param",
                           change={"total_iters": 2000, "batch_size": 512})
    noop = ideas.Idea(label="noop", granularity="code",
                      change={"file": "train.py", "find": "seed", "replace": "seed"})

    assert ideas.execution_compatibility(supported, stage_parameters=params) == (True, "")
    compatible, reason = ideas.execution_compatibility(unrelated, stage_parameters=params)
    assert not compatible and "batch_size" in reason and "total_iters" in reason
    assert ideas.execution_compatibility(noop, stage_parameters=params)[0] is False


def test_settings_candidates_must_also_fit_declared_axes_and_ranges():
    from autosim.research.adapter_protocol import Axis, OptimizationSpace

    space = OptimizationSpace(training=(Axis(
        "total_iters", "integer", "optimizer iterations", low=1, high=10, default=3),))
    supported = ideas.Idea(label="more declared iterations", granularity="param",
                           change={"total_iters": 6})
    unsupported = ideas.Idea(label="native but undeclared", granularity="param",
                             change={"total_timesteps": 6})
    out_of_range = ideas.Idea(label="outside declaration", granularity="param",
                              change={"total_iters": 11})

    assert ideas.declared_space_compatibility(supported, space=space) == (True, "")
    ok, reason = ideas.declared_space_compatibility(unsupported, space=space)
    assert not ok and "no such axis" in reason
    ok, reason = ideas.declared_space_compatibility(out_of_range, space=space)
    assert not ok and "does not accept" in reason


def test_parameter_change_uses_exact_declared_axis_name_not_a_guessed_stage_prefix():
    from autosim.research.adapter_protocol import Axis, OptimizationSpace

    space = OptimizationSpace(training=(Axis(
        "total_timesteps", "integer", "training steps", low=1024,
        high=10_000_000, default=2_000_000),))
    exact = ideas.Idea(label="longer run", granularity="param",
                       change={"total_timesteps": 4_000_000})
    guessed_prefix = ideas.Idea(label="wrong section prefix", granularity="param",
                                change={"train.total_timesteps": 4_000_000})

    assert ideas.declared_space_compatibility(exact, space=space) == (True, "")
    accepted, why = ideas.declared_space_compatibility(guessed_prefix, space=space)
    assert not accepted and "no such axis" in why


def test_a_label_is_an_identity_not_a_name(tmp_path):
    """Two ideas with one label are one idea. Letting the second replace the first would lose
    what the first was for, and the first is the one something already happened to."""
    library = _library(tmp_path)
    first = library.add(ideas.Idea(label="x", mechanism="the first"))
    library.note("x", worked=False, outcome="it made things worse")
    again = library.add(ideas.Idea(label="x", mechanism="the second"))
    assert again is first
    assert library.get("x").mechanism == "the first"
    assert library.get("x").times_tried == 1


def test_event_bound_outcome_note_is_idempotent_across_reopens(tmp_path):
    library = _library(tmp_path)
    library.add(ideas.Idea(label="x", granularity="code", status=ideas.CLEARED))
    library.note("x", worked=False, outcome="discarded after interruption",
                 event_id="discard-round-1")

    reopened = _library(tmp_path)
    reopened.note("x", worked=False, outcome="discarded after interruption",
                  event_id="discard-round-1")

    final = _library(tmp_path).get("x")
    assert final.times_tried == 1
    assert final.outcome_event_ids == ["discard-round-1"]


def test_the_audit_runs_once_and_never_again(tmp_path):
    """An idea that met a red line did not run and did not fail. Re-auditing it every round
    would let a later, clearer reading of the same idea change the verdict."""
    library = _library(tmp_path)
    library.add(ideas.Idea(label="widen the tolerance", touches=["scripts/eval_policy.py"]))
    library.add(ideas.Idea(label="raise the mass", mechanism="raise the sampling mass"))
    verdicts = library.audit(LINES)
    assert [one.verdict for one in verdicts] == [redlines.REJECTED, redlines.CLEARED]
    assert library.get("widen the tolerance").status == ideas.REJECTED
    assert library.get("raise the mass").status == ideas.CLEARED
    assert library.audit(LINES) == [], "an audited idea was audited again"

    # And a rejected idea is not a candidate.
    assert [one.label for one in library.usable()] == ["raise the mass"]


# -- choosing -------------------------------------------------------------------------------

def test_the_least_risky_untried_idea_is_chosen_first(tmp_path):
    """Deliberately unexciting. A selection that reasoning could improve is a selection that
    needs reasoning about the library rather than about the benchmark, and the reasoning is
    better spent on the ideas -- which is why they are produced in a batch."""
    library = _library(tmp_path)
    for label, risk in (("risky", "high"), ("safe", "low"), ("middling", "medium")):
        library.add(ideas.Idea(label=label, risk=risk))
    library.audit(LINES)
    assert ideas.select(library).label == "safe"
    library.note("safe", worked=False)
    assert ideas.select(library).label == "middling"


def test_three_of_a_kind_stops_being_the_place_to_look(tmp_path):
    """AutoSOTA's Leap Path, and the number is theirs: three of a kind is where a pattern is
    visible, and a fourth of the same kind is the round that was already tried."""
    library = _library(tmp_path)
    library.add(ideas.Idea(label="a", granularity="param", risk="low"))
    library.add(ideas.Idea(label="code-one", granularity="code", risk="high"))
    library.audit(LINES)

    assert not ideas.needs_a_leap(["param"])
    assert not ideas.needs_a_leap(["param", "code", "param"])
    assert ideas.needs_a_leap(["param", "param", "param"])
    # And when the run has been making one kind of move, selection prefers another.
    assert ideas.select(library, history=["param", "param", "param"]).label == "code-one"
    # With nothing else available it still returns something: a library that cannot help is
    # not a reason to stop.
    only_param = _library(tmp_path / "second")
    only_param.add(ideas.Idea(label="p", granularity="param", risk="low"))
    only_param.audit(LINES)
    assert ideas.select(only_param, history=["param", "param", "param"]).label == "p"


def test_an_outcome_is_recorded_because_a_library_without_one_is_a_wish_list(tmp_path):
    library = _library(tmp_path)
    library.add(ideas.Idea(label="x"))
    library.audit(LINES)
    library.note("x", worked=True, outcome="0.42 against 0.31")
    assert library.get("x").status == ideas.WORKED and library.get("x").times_tried == 1
    assert library.summary()["by_status"][ideas.WORKED] == 1
    # An outcome with a secret in it is redacted like every other record.
    library.note("x", worked=False, outcome="failed with key sk-abcdefghijklmnopqrst")
    assert "sk-abcdefghijklmnopqrst" not in library.get("x").outcome


# -- the batch, and the leap ----------------------------------------------------------------

class _Client:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat_with_metadata(self, system, user, **kwargs):
        self.calls += 1
        return self.replies.pop(0), {}


def test_a_batch_that_yields_nothing_is_a_library_that_says_so(tmp_path):
    """Never raises. Losing a run to a parse failure in the step that was supposed to fill the
    library is the failure this module's neighbours keep finding."""
    assert ideas.build(_Client(["not json at all", "still not json"]), {}) == []

    class Broken:
        def chat_with_metadata(self, system, user, **kwargs):
            raise TimeoutError("no answer")

    assert ideas.build(Broken(), {}) == []
    assert ideas.leap(Broken(), {}, avoid="param") is None


def test_the_batch_is_typed_and_the_typing_is_not_guessed():
    """A list that is all `param` is the list the loop already had, and it is why the loop
    could only ever tune. The prompt now prefers a smaller applicable set over a quota; what
    the code guarantees is that an unknown kind becomes `param` rather than crashing."""
    reply = json.dumps({"ideas": [
        {"label": "a", "granularity": "code", "risk": "high", "mechanism": "patch it",
         "change": {"file": "x.py", "find": "a", "replace": "b"}},
        {"label": "b", "granularity": "algo", "risk": "low"},
        {"label": "c", "granularity": "nonsense", "risk": "nonsense"},
        {"label": "", "granularity": "param"},                       # no label: dropped
        "not an object",
    ]})
    built = ideas.build(_Client([reply]), {})
    assert [one.label for one in built] == ["a", "b", "c"]
    assert [one.granularity for one in built] == ["code", "algo", "param"]
    assert built[0].change["file"] == "x.py"
    assert built[1].risk == "low" and built[2].risk == "medium"


def test_generated_parameter_idea_does_not_treat_axis_source_as_a_write(tmp_path):
    reply = json.dumps({"ideas": [
        {"label": "lower learning rate", "granularity": "param",
         "change": {"learning_rate": 1e-5},
         "touches": ["scripts/eval_train.py"], "crosses": "none"},
        {"label": "patch evaluator", "granularity": "code",
         "change": {"file": "scripts/eval_train.py", "find": "a", "replace": "b"},
         "touches": ["scripts/eval_train.py"], "crosses": "none"}
    ]})
    generated = ideas.build(_Client([reply]), {})
    assert generated[0].touches == []
    assert generated[1].touches == ["scripts/eval_train.py"]
    library = _library(tmp_path)
    for idea in generated:
        library.add(idea)
    assert [one.verdict for one in library.audit(LINES)] == [
        redlines.CLEARED, redlines.REJECTED]


def test_a_leap_has_to_be_a_different_kind_of_move():
    """The point of leaving the library is to stop making the move that has not worked. A leap
    that returns another of the same kind is the round that was already tried, wearing a new
    name."""
    same = json.dumps({"label": "another param", "granularity": "param", "mechanism": "m"})
    different = json.dumps({"label": "patch the loader", "granularity": "code",
                            "mechanism": "the loader opens a name the data does not use",
                            "change": {"file": "loader.py", "find": "a", "replace": "b"},
                            "why": "the diagnosis named the filename"})
    found = ideas.leap(_Client([same, different]), {}, avoid="param")
    assert found is not None and found.granularity == "code"
    assert found.mechanism


def test_repeated_action_type_alone_is_not_stagnation():
    improving = {"round_history": [
        {"round": 1, "status": "measured", "metric_utility": 1.0},
        {"round": 2, "status": "measured", "metric_utility": 2.0},
        {"round": 3, "status": "measured", "metric_utility": 3.0}]}
    same_failure = {"round_history": [
        {"round": index, "status": "nothing was measured", "why_not": "same OOM"}
        for index in (1, 2, 3)]}
    assert ideas.needs_a_leap(["param", "param", "param"])
    assert ideas.evidence_stalled(improving) is False
    assert ideas.evidence_stalled(same_failure) is True


def test_the_library_reads_as_a_document_with_the_ideas_that_were_never_chosen(tmp_path):
    """An idea that was considered and not run is a finding about the search: it is the
    difference between a run that tried everything and a run that ran out."""
    library = _library(tmp_path)
    library.add(ideas.Idea(label="tried it", granularity="param", risk="low",
                           mechanism="raise the mass"))
    library.add(ideas.Idea(label="never reached", granularity="code", risk="high",
                           mechanism="patch the guard"))
    library.audit(LINES)
    library.note("tried it", worked=True, outcome="0.42")
    rendered = ideas.render(library)
    assert "never reached" in rendered and "未采用" in rendered
    assert "共 2 条" in rendered and "code 1" in rendered
