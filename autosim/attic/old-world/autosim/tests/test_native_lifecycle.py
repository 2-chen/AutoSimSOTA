import json
import os
from pathlib import Path

import pytest

from autosim.research.native_lifecycle import NativeLifecycle, process_identity, retryable_startup, startup_attempt_limit
from autosim.research.common import atomic_json, read_json


def test_shared_pre_reset_contract_refuses_a_reset_that_crashed_before_initialization(tmp_path):
    atomic_json(tmp_path / "process/process.json", {"status": "failed", "returncode": -11})
    atomic_json(tmp_path / "startup.json", {"phase": "environment_constructing"})
    assert retryable_startup(tmp_path)
    atomic_json(tmp_path / "reset_started.json", {"pid": os.getpid()})
    assert not retryable_startup(tmp_path)


def test_coordinator_is_the_only_recovery_owner():
    assert startup_attempt_limit(3, {"lifecycle": {"startup_attempts": 3}}, coordinated=True) == 1
    assert startup_attempt_limit(3, {"lifecycle": {"startup_attempts": 2}}) == 2
    with pytest.raises(ValueError):
        startup_attempt_limit(4)


def test_process_start_identity_is_live_and_rejects_missing_process():
    assert process_identity(os.getpid())
    assert process_identity(999999999) is None


def test_isolated_worker_uses_same_durable_reset_and_actual_work_markers(tmp_path):
    worker = NativeLifecycle(tmp_path, config={})
    worker.start()
    worker.constructing()
    worker.ready()
    worker.before_reset()
    assert read_json(tmp_path / "reset_started.json")["pid"] == os.getpid()
    worker.step_started()
    total = sum(i * i for i in range(10000))
    worker.step_completed()
    worker.work_finished()
    worker.completed()
    row = read_json(tmp_path / "lifecycle.json")
    assert row["phase"] == "completed" and row["steps"] == 1 and total > 0
    assert row["work_ended"] > row["work_started"]
