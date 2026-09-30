import hashlib
import json

from autosim.research.policy_consumption import (verify_policy_consumption,
                                                 verify_rollout_evidence,
                                                 verify_metric_lineage)


def test_native_policy_load_event_binds_frozen_bytes(tmp_path):
    policy = tmp_path / "policy.pth"
    policy.write_bytes(b"candidate weights")
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    log = tmp_path / "evaluation.log"
    log.write_text("AUTOSIM_POLICY_LOADED " + json.dumps({
        "path": str(policy), "content_sha256": digest}) + "\nsuccess: 0.75\n",
        encoding="utf-8")
    assert verify_policy_consumption(log, policy)["status"] == "verified"
    policy.write_bytes(b"different weights")
    assert verify_policy_consumption(log, policy)["status"] == "mismatch"


def test_missing_or_conflicting_policy_load_event_does_not_verify(tmp_path):
    policy = tmp_path / "policy.pth"
    policy.write_bytes(b"weights")
    log = tmp_path / "evaluation.log"
    log.write_text("success: 0.9\n", encoding="utf-8")
    assert verify_policy_consumption(log, policy)["status"] == "unverified"
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    marker = "AUTOSIM_POLICY_LOADED " + json.dumps({
        "path": str(policy), "content_sha256": digest}) + "\n"
    log.write_text(marker * 2, encoding="utf-8")
    assert verify_policy_consumption(log, policy)["status"] == "verified"
    conflicting = "AUTOSIM_POLICY_LOADED " + json.dumps({
        "path": str(policy), "content_sha256": "0" * 64}) + "\n"
    log.write_text(marker + conflicting, encoding="utf-8")
    assert verify_policy_consumption(log, policy)["status"] == "mismatch"


def test_rollout_witness_requires_same_policy_and_completed_episode(tmp_path):
    log = tmp_path / "evaluation.log"
    identity = "a" * 64
    log.write_text("success: 1.0\n", encoding="utf-8")
    assert verify_rollout_evidence(log, policy_sha256=identity)["status"] == "unverified"
    log.write_text("AUTOSIM_ROLLOUT_COMPLETED " + json.dumps({
        "episode_id": "task-0/seed-3", "policy_sha256": identity}) + "\n",
        encoding="utf-8")
    assert verify_rollout_evidence(log, policy_sha256=identity)["status"] == "verified"
    assert verify_rollout_evidence(log, policy_sha256="b" * 64)["status"] == "mismatch"


def test_native_metric_lineage_binds_policy_episode_set_and_value(tmp_path):
    log = tmp_path / "evaluation.log"
    identity = "a" * 64
    log.write_text("AUTOSIM_METRIC_REPORTED " + json.dumps({
        "policy_sha256": identity, "episode_ids": ["one", "two"], "value": 0.5}) + "\n",
        encoding="utf-8")
    good = verify_metric_lineage(log, policy_sha256=identity,
                                 episode_ids=["two", "one"], value=0.5)
    assert good["status"] == "verified"
    assert verify_metric_lineage(log, policy_sha256=identity,
                                 episode_ids=["one"], value=0.5)["status"] == "mismatch"
    assert verify_metric_lineage(log, policy_sha256=identity,
                                 episode_ids=["one", "two"], value=0.7)["status"] == "mismatch"
