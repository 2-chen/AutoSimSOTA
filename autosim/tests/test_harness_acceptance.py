"""Trusted admission checks reject self-reported success without worker evidence."""
from copy import deepcopy

import pytest

from autosim.research.common import atomic_json
from autosim.research.harness_acceptance import native_receipt_verdict, validate_run


def native_receipt(count=4):
    devices = [f"GPU-{index:08x}-0000-0000-0000-{index:012x}" for index in range(count)]
    workers = [{"worker": f"native_{i}", "uuid": device, "generation": "generation-a",
                "claim": f"claim-{i}", "attempt_id": "1", "pid": 1000 + i, "pgid": 1000 + i,
                "process_start_identity": str(9000 + i), "host": "test-node", "phase": "completed",
                "reset_started": True, "steps": 100, "work_started": 100.0 + i,
                "work_ended": 130.0 + i} for i, device in enumerate(devices)]
    evidence = {"generation": "generation-a", "passed": True, "workers": workers,
                "overlap_seconds": 30.0 - (count - 1), "minimum_overlap_seconds": 5.0,
                "verified_rollout_concurrency": count, "construction_concurrency": count,
                "primary_failure": None}
    return {"schema_version": 3, "passed": True, "requested_devices": devices,
            "allocation": [{"uuid": d, "index": i, "model": "RTX 5090"} for i, d in enumerate(devices)],
            "verified_devices": devices, "verified_concurrent_workers": count,
            "verified_rollout_concurrency": count, "construction_concurrency": count,
            "concurrency_evidence": evidence, "cache_mode": "cold", "native_cache_namespace": "cache-a",
            "startup_attempts": [{"attempt_id": 1, "generation": "generation-a", "passed": True,
                                  "retryable": False, "primary_failure": None}],
            "validation_injection": {"requested_worker": None, "repetition": 0, "applied": False, "evidence": []}}


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_complete_distinct_real_work_evidence_passes(count):
    verdict = native_receipt_verdict(native_receipt(count), count)
    assert verdict["passed"], verdict["failures"]


@pytest.mark.parametrize("field,value", [
    ("steps", 0), ("steps", True), ("phase", "ready"), ("pid", 0),
    ("process_start_identity", None), ("claim", ""), ("attempt_id", ""),
    ("host", ""), ("work_started", True),
])
def test_worker_identity_and_actual_business_work_are_required(field, value):
    receipt = native_receipt()
    receipt["concurrency_evidence"]["workers"][0][field] = value
    assert not native_receipt_verdict(receipt, 4)["passed"]


def test_null_generation_on_every_worker_does_not_count_as_one_valid_generation():
    receipt = native_receipt()
    receipt["concurrency_evidence"]["generation"] = None
    for worker in receipt["concurrency_evidence"]["workers"]:
        worker["generation"] = None
    assert not native_receipt_verdict(receipt, 4)["passed"]


def test_duplicate_logical_worker_is_not_four_workers():
    receipt = native_receipt()
    receipt["concurrency_evidence"]["workers"][1]["worker"] = "native_0"
    assert not native_receipt_verdict(receipt, 4)["passed"]


def test_uuid_claims_must_match_the_actual_allocation_and_request():
    receipt = native_receipt()
    receipt["verified_devices"] = ["GPU-unallocated-" + str(i) for i in range(4)]
    for i, worker in enumerate(receipt["concurrency_evidence"]["workers"]):
        worker["uuid"] = receipt["verified_devices"][i]
    assert not native_receipt_verdict(receipt, 4)["passed"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 1000.0])
def test_reported_overlap_must_be_finite_and_equal_observed_intersection(value):
    receipt = native_receipt()
    receipt["concurrency_evidence"]["overlap_seconds"] = value
    assert not native_receipt_verdict(receipt, 4)["passed"]


def test_four_sequential_workers_do_not_prove_concurrency():
    receipt = native_receipt()
    for i, worker in enumerate(receipt["concurrency_evidence"]["workers"]):
        worker.update(work_started=i * 30.0, work_ended=i * 30.0 + 10)
    receipt["concurrency_evidence"]["overlap_seconds"] = 0.0
    assert not native_receipt_verdict(receipt, 4)["passed"]


def test_failed_generation_cannot_be_marked_success_by_top_level_flag():
    receipt = native_receipt()
    receipt["concurrency_evidence"]["primary_failure"] = {"worker": "native_0", "reason": "worker_exited"}
    assert not native_receipt_verdict(receipt, 4)["passed"]


@pytest.mark.parametrize("replacement", [None, "invalid", [None]])
def test_malformed_worker_evidence_returns_a_failed_verdict(replacement):
    receipt = native_receipt()
    receipt["concurrency_evidence"]["workers"] = replacement
    assert not native_receipt_verdict(receipt, 4)["passed"]


def test_empty_native_run_list_cannot_pass_the_actual_entrypoint_gate(tmp_path):
    atomic_json(tmp_path / "run_state.json", {"stage": "native_admission_complete",
                "native_admission_passed": True, "rounds": [], "final_confirmation_opened": False})
    atomic_json(tmp_path / "environment_manifest.json", {"status": "passed"})
    atomic_json(tmp_path / "native_admission_runs.json", [])
    report = validate_run(tmp_path, expected_gpus=4, stage="native_admission")
    assert not report["passed"]
