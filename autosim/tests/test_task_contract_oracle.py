"""The task contract a hand-written reader produces, frozen as an acceptance criterion.

Replacing hand-written adapters with ones the system derives itself is only safe if the
replacement can be checked against something. This is that something: the semantic contract
of all ten RoboSynChallenge tasks, captured from the reader that exists today.

It is an *oracle*, not a snapshot. The distinction matters for how it should be maintained:
a failure here does not mean "the fixture is stale, update it". It means the thing producing
task contracts now disagrees with the thing that was verified against real runs, and the
disagreement has to be explained before either side is believed. `state_dim` and `action_dim`
from here drive the observation contract, the trainer's input shape and every evaluation, so
a silent change is not a cosmetic one.

Absolute paths are excluded on purpose: `gym_config`, `action_config` and the keys of
`config_hashes` name the checkout's own location, which differs by installation and says
nothing about the task.
"""

import json
from pathlib import Path

import pytest

from autosim.research.common import find_benchmark

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
ROBOSYN = find_benchmark("RoboSynChallenge", WORKSPACE, marker="scripts/eval_policy.py")
FIXTURE = Path(__file__).with_name("fixtures") / "robosyn_task_contracts.json"

pytestmark = pytest.mark.skipif(ROBOSYN is None,
                                reason="RoboSynChallenge checkout not found")


def oracle() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_the_oracle_covers_every_task_the_registry_declares():
    from autosim.research.registry import TASK_IDS
    captured = oracle()["tasks"]
    assert set(captured) == set(TASK_IDS)


@pytest.mark.parametrize("task", sorted(json.loads(FIXTURE.read_text(encoding="utf-8"))["tasks"]))
def test_the_reader_reproduces_the_frozen_contract(task):
    from autosim.research.registry import load_task
    expected = oracle()["tasks"][task]
    # Compared under JSON semantics, because the oracle is a JSON document and the reader
    # returns tuples where JSON has lists. Both serialize identically into a task signature,
    # which is what the rest of the system keys on, so the distinction carries no meaning
    # here -- and normalising keeps a tuple/list difference from being reported as a
    # contract change.
    actual = json.loads(json.dumps(load_task(ROBOSYN, task).as_dict()))
    for field in oracle()["fields"]:
        assert actual[field] == expected[field], (
            f"{task}.{field}: the reader now says {actual[field]!r} where the frozen "
            f"contract says {expected[field]!r}")


def test_the_frozen_fields_are_the_ones_the_rest_of_the_system_reads():
    """A field the system never reads is not worth freezing; one it reads must be here."""
    from autosim.research.registry import TaskSpec
    import dataclasses
    frozen = set(oracle()["fields"])
    # These carry the checkout's own location rather than the task's meaning.
    location_bound = {"gym_config", "action_config", "config_hashes"}
    everything = {row.name for row in dataclasses.fields(TaskSpec)}
    assert frozen == everything - location_bound


def test_the_oracle_says_what_it_is_for():
    document = oracle()
    assert "oracle" in document["note"]
    assert "must" in document["note"]
