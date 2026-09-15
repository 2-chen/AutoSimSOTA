"""SCO lifecycle tests: every platform query is mocked; no jobs are submitted."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[2] / "tools"


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("harness_sco_test", TOOLS / "harness_sco.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BASE = tmp_path / "base"
    module.PROJECT = module.BASE / "project"
    module.BASE.mkdir()
    campaign = module.BASE / "campaign"
    campaign.mkdir()
    calls, jobs = [], []

    def query(argv):
        calls.append(list(argv))
        if argv[:3] == ["acp", "jobs", "list"]:
            return list(jobs)
        if argv[:3] == ["acp", "jobs", "create"]:
            display = argv[argv.index("--job-name") + 1]
            row = {"name": "pt-test-" + str(len(jobs)), "display_name": display, "state": "PENDING"}
            jobs.append(row)
            return {"name": row["name"]}
        if argv[:3] == ["acp", "jobs", "describe"]:
            return next(row for row in jobs if row["name"] == argv[-1])
        raise AssertionError(f"unexpected mocked platform operation {argv[:3]}")

    monkeypatch.setattr(module, "query", query)
    binary = module.BASE / "harness_validation/dependencies/bwrap"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\x7fELF test-only offline fixture; never executed")
    binary.chmod(0o755)
    for name in ("optix_runtime/nvoptix.bin", "container_libs/lib/libnvoptix.so.1",
                 "compute_validation/dependencies/torch/hub/checkpoints/resnet18-f37072fd.pth"):
        path = module.BASE / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("offline fixture")

    def create_root(gate="G1", attempt="current", g0=True):
        root = campaign / f"{gate}_{attempt}"
        source = root / "source"
        for name in (".venv/bin/python", "RoboSynChallenge/policy/act/.venv/bin/python",
                     "RoboSynChallenge/checkpoints/ACT_sim_click_bell/model.safetensors",
                     "RoboSynChallenge/lerobot_dataset/RoboSynChallenge/cobotmagic_Sim_click_bell/meta/info.json",
                     "autosim/autosim/a.py", "tools/run_full_research_sco.sh"):
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("offline immutable fixture")
        hashes = {"autosim/autosim/a.py": module.sha(source / "autosim/autosim/a.py")}
        module.save(root / "source_manifest.json", hashes)
        module.save(root / "harness_config.json", {"schema_version": 1})
        count, limit = module.GATES[gate]
        seconds = int(limit / count * 3600)
        code = module.code_identity(hashes)
        module.save(root / "launch_manifest.json", {"schema_version": 2, "campaign": str(campaign),
            "gate": gate, "run_id": f"harness_{gate.lower()}_{attempt}", "expected_gpus": count,
            "run_root": str(root / "runs/run"), "gpu_hours_cap": limit,
            "max_container_seconds": seconds - 120, "allocation_cap_seconds": seconds,
            "production_code_identity": code, "harness_config_sha256": module.sha(root / "harness_config.json"),
            "repair_sandbox": {"binary": str(binary), "sha256": module.sha(binary)},
            "research_command": ["autosim", "research", str(source / "RoboSynChallenge"),
                                  "--hours", str((seconds-120)/3600), "--gpu-hours", str(limit)]})
        if g0:
            cpu, api = campaign / "cpu_evidence.json", campaign / "api_evidence.json"
            module.save(cpu, {"tests": 1, "failures": 0, "errors": 0})
            module.save(api, {"api_used": True, "request_ids": ["provider-test-id"]})
            module.save(campaign / "g0_acceptance.json", {"passed": True, "production_code_identity": code,
                "cpu_tests": {"passed": True, "tests": 1, "failures": 0, "errors": 0,
                              "receipt": {"path": str(cpu), "sha256": module.sha(cpu)}},
                "api_coding": {"passed": True, "api_used": True, "activation_verified": True,
                               "request_ids": ["provider-test-id"],
                               "receipt": {"path": str(api), "sha256": module.sha(api)}}})
        return root

    module.test_campaign, module.test_calls, module.test_jobs = campaign, calls, jobs
    module.test_create_root = create_root
    module.test_binary = binary
    return module


def creates(module):
    return [call for call in module.test_calls if call[:3] == ["acp", "jobs", "create"]]


def test_verified_g0_submits_one_bounded_job_and_reconciles_repeated_call(harness):
    root = harness.test_create_root()
    result = harness.submit(root)
    again = harness.submit(root)
    assert result["name"] == again["name"] and again["reconciled"]
    assert len(creates(harness)) == 1
    assert (root / "submission_intent.json").exists()
    manifest = json.loads((root / "launch_manifest.json").read_text())
    assert manifest["max_container_seconds"] <= 2580 and manifest["gpu_hours_cap"] == .75
    assert (harness.test_campaign / "G1_budget.json").exists()


@pytest.mark.parametrize("change", ["missing", "api_unused", "api_no_requests", "not_activated",
                                    "api_evidence_changed", "cpu_failed", "cpu_zero_tests", "wrong_revision"])
def test_g0_cpu_and_real_api_evidence_are_mandatory_before_platform_calls(harness, change):
    root = harness.test_create_root()
    path = harness.test_campaign / "g0_acceptance.json"
    value = json.loads(path.read_text())
    if change == "missing":
        path.unlink()
    else:
        if change == "api_unused": value["api_coding"]["api_used"] = False
        if change == "api_no_requests": value["api_coding"]["request_ids"] = []
        if change == "not_activated": value["api_coding"]["activation_verified"] = False
        if change == "api_evidence_changed": value["api_coding"]["receipt"]["sha256"] = "wrong"
        if change == "cpu_failed": value["cpu_tests"]["failures"] = 1
        if change == "cpu_zero_tests": value["cpu_tests"]["tests"] = 0
        if change == "wrong_revision": value["production_code_identity"] = "unverified-code"
        harness.save(path, value)
    with pytest.raises(RuntimeError, match="G0"):
        harness.submit(root)
    assert harness.test_calls == []


def test_unknown_create_result_never_reissues_the_mutation(harness, monkeypatch):
    root = harness.test_create_root()
    calls = []
    def uncertain(argv):
        calls.append(argv)
        if argv[:3] == ["acp", "jobs", "create"]:
            assert (root / "submission_intent.json").exists()
            raise RuntimeError("transport outcome unknown")
        assert argv[:3] == ["acp", "jobs", "list"]
        return []
    monkeypatch.setattr(harness, "query", uncertain)
    with pytest.raises(RuntimeError, match="unknown"):
        harness.submit(root)
    with pytest.raises(RuntimeError, match="unresolved"):
        harness.submit(root)
    assert sum(call[:3] == ["acp", "jobs", "create"] for call in calls) == 1


def test_pending_intent_reconciles_known_display_without_creating(harness):
    root = harness.test_create_root()
    harness.save(root / "submission_intent.json", {"status": "create_requested"})
    harness.test_jobs.append({"name": "pt-existing", "display_name": "autosim-harness-g1-current", "state": "RUNNING"})
    assert harness.submit(root)["name"] == "pt-existing"
    assert not creates(harness)


def test_display_name_collision_without_local_intent_is_not_adopted(harness):
    root = harness.test_create_root()
    harness.test_jobs.append({"name": "pt-other", "display_name": "autosim-harness-g1-current", "state": "RUNNING"})
    with pytest.raises(RuntimeError, match="no local create intent"):
        harness.submit(root)
    assert not creates(harness)


def test_same_attempt_concurrent_submit_is_exactly_once(harness):
    root = harness.test_create_root()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(harness.submit, root) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert len({row["name"] for row in results}) == 1 and len(creates(harness)) == 1


def test_two_attempts_cannot_reserve_the_same_gate_budget(harness):
    roots = [harness.test_create_root(attempt=name) for name in ("a", "b")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(harness.submit, root) for root in roots]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=10))
            except RuntimeError as exc:
                assert "budget" in str(exc)
                outcomes.append(None)
    assert len([row for row in outcomes if row]) == 1 and len(creates(harness)) == 1


def prior_attempt(harness, *, state="FAILED", expired=False, bad_interval=False):
    previous = harness.test_create_root(attempt="prior")
    current = datetime.now(timezone.utc)
    manifest = json.loads((previous / "launch_manifest.json").read_text())
    deadline = current + timedelta(seconds=-10 if expired else 1200)
    manifest["run_deadline_utc"] = deadline.isoformat()
    harness.save(previous / "launch_manifest.json", manifest)
    harness.save(previous / "submission_intent.json", {"status": "create_requested"})
    harness.save(previous / "job_description.json", {"state": state,
        "start_time": (current - timedelta(seconds=900)).isoformat(),
        "complete_time": (current - timedelta(seconds=1000 if bad_interval else 300)).isoformat()})
    return previous, deadline


def test_prior_terminal_cost_is_charged_and_original_deadline_not_extended(harness):
    _, original_deadline = prior_attempt(harness)
    root = harness.test_create_root()
    harness.submit(root)
    manifest = json.loads((root / "launch_manifest.json").read_text())
    assert manifest["previous_allocated_gpu_hours"] == pytest.approx(600 / 3600)
    assert manifest["gpu_hours_cap"] == pytest.approx(.75 - 600 / 3600)
    assert datetime.fromisoformat(manifest["run_deadline_utc"]) <= original_deadline
    assert manifest["max_container_seconds"] <= 1080


def test_active_or_unknown_allocation_reserves_full_cap_despite_stale_end_fields(harness):
    prior_attempt(harness, state="RUNNING")
    root = harness.test_create_root()
    with pytest.raises(RuntimeError, match="budget"):
        harness.submit(root)
    assert not creates(harness)


def test_expired_original_deadline_cannot_be_reset_by_new_attempt(harness):
    prior_attempt(harness, expired=True)
    root = harness.test_create_root()
    with pytest.raises(RuntimeError, match="original wall-clock"):
        harness.submit(root)
    assert not creates(harness)


def test_negative_platform_duration_cannot_credit_the_budget(harness):
    prior_attempt(harness, bad_interval=True)
    root = harness.test_create_root()
    with pytest.raises(RuntimeError, match="timestamps"):
        harness.submit(root)
    assert not creates(harness)


def test_double_gpu_gate_requires_same_revision_single_gpu_business_acceptance(harness):
    root = harness.test_create_root(gate="G2")
    with pytest.raises(RuntimeError, match="G1"):
        harness.submit(root)
    assert harness.test_calls == []


@pytest.mark.parametrize("broken", ["missing", "changed", "symlink"])
def test_offline_bwrap_is_verified_before_any_platform_query(harness, broken):
    root = harness.test_create_root()
    binary = harness.test_binary
    if broken == "missing": binary.unlink()
    elif broken == "changed": binary.write_bytes(b"\x7fELF changed")
    else:
        target = binary.with_suffix(".real")
        binary.rename(target)
        binary.symlink_to(target)
    with pytest.raises(RuntimeError, match="bubblewrap"):
        harness.submit(root)
    assert harness.test_calls == []


def test_cached_bwrap_is_copied_from_real_bytes_not_a_vendor_symlink(tmp_path, harness, monkeypatch):
    target = harness.test_binary
    target.unlink()
    binary = tmp_path / "vendor-bwrap"
    binary.write_bytes(b"\x7fELF distinct test bytes")
    binary.chmod(0o755)
    alias = tmp_path / "bwrap-alias"
    alias.symlink_to(binary)
    monkeypatch.setenv("AUTOSIM_REPAIR_BWRAP", str(alias))
    result = harness.cache_repair_bwrap()
    assert target.is_file() and not target.is_symlink() and os.access(target, os.X_OK)
    assert target.read_bytes() == binary.read_bytes() and result["sha256"] == harness.sha(binary)


def test_prepare_freezes_cached_sandbox_binary_identity_in_launch_manifest(harness, monkeypatch):
    def freeze(root):
        source = root / "source"
        source.mkdir(parents=True)
        (source / "a.py").write_text("frozen fixture")
        harness.save(root / "source_manifest.json", {"a.py": harness.sha(source / "a.py")})
        return source
    monkeypatch.setattr(harness, "freeze_repository", freeze)
    monkeypatch.setenv("AUTOSIM_REPAIR_BWRAP", str(harness.test_binary))
    root = harness.prepare(harness.test_campaign, "G1", "prepare")
    manifest = json.loads((root / "launch_manifest.json").read_text())
    assert manifest["repair_sandbox"]["binary"] == str(harness.test_binary)
    assert manifest["repair_sandbox"]["sha256"] == harness.sha(harness.test_binary)
    assert harness.test_calls == []


def test_source_manifest_cannot_claim_an_old_g0_revision_for_changed_code(harness):
    root = harness.test_create_root()
    source = root / "source/autosim/autosim/a.py"
    source.write_text("changed production code")
    harness.save(root / "source_manifest.json", {"autosim/autosim/a.py": harness.sha(source)})
    with pytest.raises(RuntimeError, match="verified production revision"):
        harness.submit(root)
    assert harness.test_calls == []


def test_manifest_cannot_request_more_gpus_than_the_registered_gate(harness):
    root = harness.test_create_root()
    manifest = json.loads((root / "launch_manifest.json").read_text())
    manifest["expected_gpus"] = 8
    harness.save(root / "launch_manifest.json", manifest)
    with pytest.raises(RuntimeError, match="GPU count"):
        harness.submit(root)
    assert harness.test_calls == []


def test_unparsed_create_response_keeps_intent_and_never_creates_again(harness, monkeypatch):
    root = harness.test_create_root()
    count = 0
    def unparsed(argv):
        nonlocal count
        if argv[:3] == ["acp", "jobs", "create"]:
            count += 1
            return {"name": None, "create_output": "accepted but identifier unavailable"}
        return []
    monkeypatch.setattr(harness, "query", unparsed)
    with pytest.raises(RuntimeError, match="unique job identity"):
        harness.submit(root)
    with pytest.raises(RuntimeError, match="unresolved"):
        harness.submit(root)
    assert count == 1


def test_submission_lists_all_pages_before_claiming_a_display_name(harness, monkeypatch):
    root = harness.test_create_root()
    original = harness.query
    pages = []
    def paged(argv):
        if argv[:3] == ["acp", "jobs", "list"]:
            page = argv[argv.index("--page-token") + 1]
            pages.append(page)
            if page == "1":
                return [{"name": f"pt-old-{i}", "display_name": f"old-{i}", "state": "SUCCEEDED"}
                        for i in range(500)]
            return []
        return original(argv)
    monkeypatch.setattr(harness, "query", paged)
    harness.submit(root)
    assert pages == ["1", "2"] and len(creates(harness)) == 1


@pytest.mark.parametrize("valid", [True, False])
def test_remote_shell_verifies_frozen_bwrap_and_sets_its_path_before_entrypoint(tmp_path, valid):
    source = tmp_path / "source"
    (source / "tools").mkdir(parents=True)
    (source / ".venv/bin").mkdir(parents=True)
    (source / ".venv/bin/python").symlink_to(sys.executable)
    shutil.copyfile(TOOLS / "run_full_research_sco.sh", source / "tools/run_full_research_sco.sh")
    (source / "tools/run_full_research_sco.py").write_text(
        "import os,sys\nfrom pathlib import Path\nPath(sys.argv[1],'entered.txt').write_text(os.environ['AUTOSIM_REPAIR_BWRAP'])\n")
    binary = Path("/data/AutoResearch/AutoSimSOTA/harness_validation/dependencies/bwrap")
    import hashlib
    (tmp_path / "launch_manifest.json").write_text(json.dumps({"schema_version": 2,
        "repair_sandbox": {"binary": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest() if valid else "invalid"}}))
    process = subprocess.run(["bash", str(source / "tools/run_full_research_sco.sh"), str(tmp_path)],
                             capture_output=True, text=True, timeout=10)
    assert (process.returncode == 0) is valid
    assert (tmp_path / "entered.txt").exists() is valid
    if valid:
        assert (tmp_path / "entered.txt").read_text() == str(binary)
