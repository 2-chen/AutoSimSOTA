from pathlib import Path

import pytest

from autosim.research.common import find_benchmark
from autosim.research.registry import load_task
from autosim.research.robosyn_adapter import RoboSynAdapter


PROJECT = Path(__file__).resolve().parents[1]
# Found by content, not by counting parents: where the benchmark checkout lives is the
# operator's choice, and asserting a depth made moving it a test failure.
ROBOSYN = find_benchmark("RoboSynChallenge", PROJECT, marker="scripts/eval_policy.py")
if ROBOSYN is None:
    pytest.skip("no RoboSynChallenge checkout on disk", allow_module_level=True)


def test_composite_hard_is_a_click_bell_only_historical_profile():
    adapter = RoboSynAdapter(ROBOSYN)
    click = adapter.collection_profiles(load_task(ROBOSYN, "click_bell"))
    water = adapter.collection_profiles(load_task(ROBOSYN, "water_pouring"))
    handle = adapter.collection_profiles(load_task(ROBOSYN, "handle_basket"))

    assert "composite_hard" in click
    assert "composite_hard" not in water
    assert "composite_hard" not in handle
    assert "targeted_appearance" not in water
    assert "targeted_camera" not in water
    assert water
    assert handle


def test_unreadable_profiles_are_diagnostic_unknown_not_training_actions():
    adapter = RoboSynAdapter(ROBOSYN)
    spec = load_task(ROBOSYN, "water_pouring")
    records = {row.capability_id: row for row in adapter.capabilities(spec)}
    assert records["targeted_collection.targeted_appearance"].status == "unknown"
    assert records["targeted_collection.targeted_appearance"].evidence["training_admission"] == "forbidden"
    assert records["targeted_collection.targeted_camera"].status == "unknown"
