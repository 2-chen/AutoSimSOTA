import json
from pathlib import Path

import pytest

from autosim.research.learning_probe_analysis import _canonical_train_config, _wilson


def test_wilson_is_bounded_and_contains_observed_rate():
    low, high = _wilson(21, 40)
    assert 0 <= low < 21 / 40 < high <= 1


def test_canonical_config_only_ignores_output_directory():
    left = {"steps": 5000, "seed": 1000, "output_dir": "/a"}
    right = {"steps": 5000, "seed": 1000, "output_dir": "/b"}
    assert _canonical_train_config(left) == _canonical_train_config(right)
    right["seed"] = 1001
    assert _canonical_train_config(left) != _canonical_train_config(right)
