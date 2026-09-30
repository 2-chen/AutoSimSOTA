"""The objective is scored, ordered, and honest about what it has not reached."""

import json

from autosim.research.objective import (Check, Rubric, answer_children, decompose, evaluate,
                                        plain, render,
                                        SPINE)


class _Client:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def chat_with_metadata(self, *a, **k):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply, {}


def _total(rubric):
    return sum(one.weight for one in rubric.checks)


def test_the_spine_alone_is_a_hundred():
    """The weights are the whole objective, so a fully met spine is a score of one."""
    assert _total(plain()) == 100.0
    assert plain().score() == 0.0


def test_a_met_spine_scores_one():
    facts = {name: True for name, _, _ in SPINE}
    assert evaluate(plain(), facts).score() == 1.0


def test_not_reached_is_not_the_same_as_no():
    """The distinction the module turns on: a run that has not got as far as measuring has
    not failed to measure, and both score zero -- but only one of them is the frontier."""
    rubric = plain()
    evaluate(rubric, {"environment": {"passed": True, "because": "built"}})
    assert rubric.checks[0].state() == "yes"
    assert rubric.checks[2].state() == "not reached"
    reach = evaluate(plain(), {"environment": {"passed": False}})
    assert reach.checks[2].state() == "not reached"
    assert reach.score() == 0.0


def test_a_run_that_got_further_scores_higher():
    """The whole reason for the module: two runs that both failed to measure have to be
    tellable apart, or `_best` cannot advance and nothing can be preferred."""
    nowhere = evaluate(plain(), {})
    halfway = evaluate(plain(), {"environment": {"passed": True}})
    nearly = evaluate(plain(), {"environment": {"passed": True},
                                "stages": {"passed": True},
                                "measurement": {"passed": True}})
    assert nowhere.score() < halfway.score() < nearly.score()


def test_a_child_cannot_be_worth_more_than_its_parent():
    """Conservation. A benchmark-specific leaf is a way for the top-level question to be
    partly met, not a separate thing to be scored -- otherwise decomposing a rubric would
    raise the run's score without it having done anything."""
    rubric = plain()
    rubric.checks[1].children = [Check(name="seeds", question="own seeds", weight=999.0)]
    assert rubric.score() == 0.0
    rubric.checks[1].children[0].passed = True
    assert rubric.score() == 25.0 / 100.0
    assert rubric.score() <= plain().score() + 25.0 / 100.0


def test_a_parent_that_passed_does_not_need_its_children():
    rubric = plain()
    rubric.checks[1].children = [Check(name="seeds", question="own seeds", weight=1.0)]
    rubric.checks[1].children[0].passed = False
    evaluate(rubric, {"stages": {"passed": True}})
    assert rubric.score() == 25.0 / 100.0


def test_a_rung_nobody_answered_still_earns_what_was_observed_under_it():
    """The case the module exists for. RoboTwin's evaluation never produced a number, and
    the invocation was nevertheless got right; a rubric that scores the second as nothing
    is a ladder with extra words."""
    rubric = plain()
    evaluate(rubric, {"environment": {"passed": True}})
    rubric.checks[2].children = [Check(name="own-states", question="own initial states?",
                                       weight=1.0, passed=True),
                                 Check(name="number", question="a number came out?",
                                       weight=1.0, passed=False)]
    assert rubric.checks[2].state() == "not reached"
    assert rubric.score() == (15.0 + 12.5) / 100.0


def test_the_frontier_is_the_first_thing_not_met():
    rubric = evaluate(plain(), {"environment": {"passed": True}, "stages": {"passed": False}})
    assert rubric.frontier().name == "stages"


def test_the_frontier_descends_only_into_a_rung_something_was_observed_in():
    """`stages` is the first rung not met, so it is the frontier -- and it points at the
    child that failed, because that child says which part of it failed."""
    rubric = evaluate(plain(), {"environment": {"passed": True}, "stages": {"passed": False}})
    rubric.checks[1].children = [Check(name="first", question="one", weight=1.0, passed=True),
                                 Check(name="second", question="two", weight=1.0, passed=False)]
    assert rubric.frontier().name == "second"


def test_an_unanswered_rung_is_the_frontier_not_its_children():
    """A rung nobody answered is not a rung to look inside: until `environment` is settled,
    the question under `stages` is not the next thing to work on."""
    rubric = plain()
    rubric.checks[1].children = [Check(name="seeds", question="own seeds?", weight=1.0,
                                       passed=True)]
    assert rubric.frontier().name == "environment"


def test_a_met_rubric_has_no_frontier():
    rubric = evaluate(plain(), {name: True for name, _, _ in SPINE})
    assert rubric.frontier() is None


def test_render_shows_the_holes():
    """A scoreboard that shows only what passed cannot be told from one whose other rows
    were never looked at."""
    text = render(evaluate(plain(), {"environment": {"passed": True, "because": "built"}}))
    assert "not reached" in text
    assert "下一步在" in text


def test_render_survives_an_empty_rubric():
    assert render(Rubric([]))


def test_decompose_puts_the_leaves_under_their_question():
    client = _Client('{"stages": [{"name": "seeds", "question": "did it use the own seeds?", '
                     '"weight": 2}]}')
    rubric = decompose(client, {"benchmark": "x"})
    assert [one.name for one in rubric.checks[1].children] == ["seeds"]


def test_decompose_ignores_rows_with_no_question():
    """A row with no question is a row that cannot be answered, and an unanswerable row
    would sit in the frontier forever."""
    client = _Client('{"stages": [{"name": "seeds"}, {"question": "real one?", "weight": 1}]}')
    rubric = decompose(client, {})
    assert [one.question for one in rubric.checks[1].children] == ["real one?"]


def test_decompose_falls_back_to_the_spine():
    """Never raises. A run whose rubric could not be decomposed scores less finely, which is
    a worse outcome than a good rubric and a much better one than losing the run."""
    assert len(decompose(_Client("not json at all"), {}).checks) == len(SPINE)
    assert decompose(_Client(RuntimeError("no network")), {}).score() == 0.0


def test_decompose_names_a_weight_it_could_not_read():
    client = _Client('{"stages": [{"question": "q?", "weight": "heavy"}]}')
    assert decompose(client, {}).checks[1].children[0].weight == 1.0


def test_render_carries_the_reason():
    rubric = evaluate(plain(), {"environment": {"passed": False, "because": "no sapien"}})
    assert "no sapien" in render(rubric)


class _AnsweringClient:
    model = "a-test-client"

    def __init__(self, answers):
        self.answers = answers
        self.seen = ""

    def chat_with_metadata(self, system, user, **kwargs):
        self.seen = user
        return json.dumps({"answers": self.answers}), {}


def _with_children():
    rubric = plain()
    rubric.checks[2].children = [
        Check(name="own-states", question="did it use the benchmark's own initial states?",
              weight=1.0),
        Check(name="number", question="did a number come out?", weight=1.0)]
    return rubric


def test_the_run_answers_its_own_questions():
    """The children are the benchmark's own, and no code can answer them: `did the run log
    which config file it loaded` is about a record that exists and a fact only this
    benchmark's reader knows how to find. The run wrote the questions; this is the other half.
    """
    rubric = _with_children()
    client = _AnsweringClient([{"name": "own-states", "passed": True,
                                "because": "round_1 records seed_start=100000000"},
                               {"name": "number", "passed": None,
                                "because": "no measurement produced one"}])
    assert answer_children(client, rubric, material={"rounds": []}) == 2
    assert rubric.checks[2].children[0].passed is True
    assert rubric.checks[2].children[0].by == "model"
    assert rubric.checks[2].children[1].passed is None
    # Null is an answer and it is recorded as one, with the reason it is null.
    assert "no measurement produced one" in render(rubric)


def test_a_question_nobody_answered_stays_not_reached():
    rubric = _with_children()
    client = _AnsweringClient([{"name": "own-states", "passed": True, "because": "x"}])
    assert answer_children(client, rubric, material={}) == 1
    assert rubric.checks[2].children[1].passed is None
    assert rubric.checks[2].children[1].by == ""


def test_answering_never_raises():
    """A run whose scoreboard could not be filled in is a run with a coarser scoreboard."""
    assert answer_children(_AnsweringClient("not a list"), _with_children(),
                           material={}) == 0
    assert answer_children(_AnsweringClient([{"name": "nobody-asked", "passed": True}]),
                           _with_children(), material={}) == 0
    assert answer_children(_AnsweringClient([]), _with_children(), material={}) == 0


def test_a_rubric_with_no_children_asks_nothing():
    assert answer_children(_AnsweringClient([]), plain(), material={}) == 0
