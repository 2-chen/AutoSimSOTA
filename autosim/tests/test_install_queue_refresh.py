import json
import sys
import pytest

from autosim.research import provision as pv, environment_pool as pool
from autosim.research.evidence_store import capture_attempt_evidence


def test_combined_install_and_probe_repair_preserves_both_outputs(tmp_path, monkeypatch):
    class Client:
        output = tmp_path
        def chat_with_metadata(self, *args, **kwargs):
            return json.dumps({"repair_mode": "replace_operation", "commands": ["echo corrected"],
                "probes": ["python corrected_probe.py"],
                "install_replacement_evidence": {}, "probe_replacement_evidence": {}}), {}
    monkeypatch.setattr(pv, "platform_facts", lambda: {})
    monkeypatch.setattr(pv, "review_install_replacement", lambda *a, **kw: {"approved": True})
    monkeypatch.setattr(pv, "review_probe_replacement", lambda *a, **kw: {"approved": True})
    transcript = []
    commands, probes = pv.resume(Client(), tmp_path, [], {"command": "wrong", "evidence_id": "a"*32},
        manifests={}, transcript=transcript, values={}, current_probes=["python original_probe.py"])
    assert commands == ["echo corrected"] and probes == ["python corrected_probe.py"]
    assert transcript[-1]["kind"] == "resume"
    assert transcript[-1]["replacement_review"]["approved"]
    assert transcript[-1]["probe_replacement_review"]["approved"]


def test_source_citation_line_numbers_are_not_host_permissions(tmp_path):
    (tmp_path / "pyproject.toml").write_text('name = "actual-package"\n')
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            payload = json.loads(user)
            assert payload["sources"] == {"pyproject.toml": 'name = "actual-package"\n'}
            return json.dumps({"approved": True, "reason": "same actual package",
                "citations": [{"file": "pyproject.toml", "quote": 'name = "actual-package"'}]}), {}
    evidence = {"same_capability": "same package", "source_refs": ["checkout: pyproject.toml:1-2"]}
    assert pv.review_probe_replacement(Client(), tmp_path, {}, ["python probe.py"], evidence)["approved"]
    evidence["source_refs"] = [{"file": "pyproject.toml", "line": 1}]
    assert pv.review_probe_replacement(Client(), tmp_path, {}, ["python probe.py"], evidence)["approved"]
    evidence["source_refs"] = ["checkout: ../outside.py:1"]
    assert not pv.review_probe_replacement(Client(), tmp_path, {}, [], evidence)["approved"]


def test_replacement_review_sees_all_original_capabilities(tmp_path):
    (tmp_path / "api.py").write_text("def native_action(): pass\n")
    class Client:
        def chat_with_metadata(self, system, user, **kwargs):
            assert json.loads(user)["required_existing_probes"] == ["train --help", "eval --help", "python reset.py"]
            assert "Retain ALL" in system
            return json.dumps({"approved": False, "reason": "replacement loses evaluator",
                "citations": [{"file": "api.py", "quote": "native_action"}]}), {}
    review = pv.review_probe_replacement(Client(), tmp_path, {}, ["python -c 'import module'"],
        {"same_capability": "native API", "source_refs": ["api.py"]},
        required_probes=["train --help", "eval --help", "python reset.py"])
    assert not review["approved"]


def test_refresh_repairs_known_failed_head_before_replaying_it(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"; repo.mkdir()
    seen = []
    def execute(command, **kwargs):
        seen.append(command)
        ok = command != "false"
        log = tmp_path / ("success.log" if ok else "failure.log")
        log.write_text("recovered" if ok else "invalid original operation")
        attempt = ("b" if ok else "a") * 32
        capture_attempt_evidence(tmp_path, attempt_id=attempt, log=log,
            receipt_ref="receipt.json", status="completed" if ok else "failed",
            returncode=0 if ok else 1, termination_reason="completed" if ok else "failed")
        return {"command": command, "ok": ok, "returncode": 0 if ok else 1,
            "seconds": .01, "failure_kind": "ok" if ok else "failed", "evidence_id": attempt}
    calls = []
    def repair(*args, **kwargs):
        calls.append(args[3]["evidence_id"])
        if len(calls) == 1:
            return [], None
        kwargs["transcript"].append({"kind": "resume", "parent_evidence_id": "a"*32,
            "replacement_review": {"approved": True}, "reviewed_same_capability": True})
        return ["echo recovered"], ["{python} -c 'import pathlib'"]
    monkeypatch.setattr(pv, "run", execute)
    monkeypatch.setattr(pv, "resume", repair)
    seed = {"python": sys.executable, "templates": ["false"], "probes": ["{python} -c 'import json'"]}
    pv.build(repo, client=object(), prefix=tmp_path / "env", output=tmp_path,
        python=sys.executable, manifests={}, assets={}, max_operations=1, max_rounds=1, seed=seed)
    pv.build(repo, client=object(), prefix=tmp_path / "env", output=tmp_path,
        python=sys.executable, manifests={}, assets={}, max_operations=1, max_rounds=1, seed=seed)
    assert seen == ["false", "echo recovered"]
    assert calls == ["a"*32, "a"*32]
    cursor = json.loads((tmp_path / "provision_cursor.json").read_text())
    assert cursor["probes"] == ["{python} -c 'import pathlib'"]
    transcript = json.loads((tmp_path / "transcript.json").read_text())["rows"]
    assert any(row.get("kind") == "queue_reconciled" and row["original_retired"] for row in transcript)


def test_declared_package_matches_rank_reusable_environment_not_just_torch():
    rows = [{"id": "generic", "python": "3.11", "kind": "conda", "cloneable": True,
        "packages": {"torch": "2.7", "numpy": "2"}, "matching_package_count": 1},
        {"id": "task-ready-lead", "python": "3.11", "kind": "conda", "cloneable": True,
        "packages": {"native-engine": "1", "planner": "2"}, "matching_package_count": 2,
        "declared_package_matches": {"native-engine": "1", "planner": "2"}}]
    catalog = pool.short_catalog(rows)
    assert catalog[0]["id"] == "task-ready-lead"
    assert catalog[0]["declared_package_matches"] == rows[1]["declared_package_matches"]


@pytest.mark.parametrize("covered,success", [(True, True), (False, True), (True, False)])
def test_related_retirement_requires_success_evidence_and_complete_review(tmp_path, monkeypatch, covered, success):
    log = tmp_path / "capability.log"; log.write_text("NATIVE_CAPABILITY_OK")
    capture_attempt_evidence(tmp_path, attempt_id="b"*32, log=log, receipt_ref="cap.json",
        status="completed" if success else "failed", returncode=0 if success else 1,
        termination_reason="completed" if success else "failed")
    class Client:
        output = tmp_path
        def chat_with_metadata(self, *args, **kwargs):
            payload = json.loads(args[1])
            assert payload["pending_installation_operations"] == ["old source route", "old related installer"]
            return json.dumps({"repair_mode": "replace_operation", "commands": ["python native_probe.py"],
                "retire_operations": ["old related installer"], "capability_evidence_ids": ["b"*32],
                "install_replacement_evidence": {}}), {}
    def review(*args):
        failure = args[2]
        assert failure["capability_evidence"][0]["returncode"] == 0
        return {"approved": True, "covered_operations": ["old related installer"] if covered else []}
    monkeypatch.setattr(pv, "review_install_replacement", review)
    monkeypatch.setattr(pv, "platform_facts", lambda: {})
    transcript = []
    commands, _ = pv.resume(Client(), tmp_path, [], {"command": "old source route", "evidence_id": "a"*32},
        manifests={}, transcript=transcript, values={}, attempts=1,
        pending_commands=["old source route", "old related installer"])
    if covered and success:
        assert commands == ["python native_probe.py"]
        assert transcript[-1]["retired_operations"] == ["old related installer"]
    else:
        assert commands == []
        assert transcript[-1]["kind"] == "repair_proposal_rejected"
