"""Freeze, budget and submit the ordered harness gates through the real research CLI."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from freeze_sco_repository import freeze_repository
from sco_compute_validation import BASE, PROJECT, IMAGE, MOUNT, TERMINAL, query, save


GATES = {"G1": (1, .75), "G2": (2, 2), "G3": (4, 8), "G4": (4, 16), "G5": (4, 96), "G6": (8, 12)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_repair_bwrap() -> dict:
    """Cache the real executable bytes; SCO never follows a Codex vendor symlink."""
    target = BASE / "harness_validation/dependencies/bwrap"
    discovered = os.environ.get("AUTOSIM_REPAIR_BWRAP") or shutil.which("bwrap")
    source = Path(discovered).resolve() if discovered else target
    if not source.is_file() or source.read_bytes()[:4] != b"\x7fELF":
        raise RuntimeError("a real offline bubblewrap ELF executable is required")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise RuntimeError("cached bubblewrap must be a regular file, not a symlink")
    if target.exists() and sha(target) != sha(source):
        raise RuntimeError("cached bubblewrap differs from the selected binary; version the dependency explicitly")
    if not target.exists():
        temporary = target.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        temporary.chmod(0o755)
        os.replace(temporary, target)
    if not os.access(target, os.X_OK):
        raise RuntimeError("cached bubblewrap is not executable")
    return {"binary": str(target.absolute()), "sha256": sha(target), "format": "ELF",
            "scope": "offline executable; kernel isolation is independently probed"}


def _checked_evidence(value: dict, label: str) -> dict:
    reference = value.get("receipt")
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        raise RuntimeError(f"{label} requires a bound evidence artifact")
    path = Path(reference["path"])
    if not path.is_file() or sha(path) != reference.get("sha256"):
        raise RuntimeError(f"{label} evidence is missing or changed")
    return {"path": str(path), "sha256": reference["sha256"]}


def code_identity(manifest: dict) -> str:
    production = {name: value for name, value in manifest.items()
                  if name.startswith(("autosim/autosim/", "RoboSynChallenge/", "tools/"))}
    return hashlib.sha256(json.dumps(production, sort_keys=True).encode()).hexdigest()


def scientific_recipe(command: list[str], source_manifest: dict) -> dict:
    names = ("--task", "--controller", "--rounds", "--attempts-per-round", "--training-steps",
             "--development-episodes", "--selection-episodes", "--final-episodes", "--train-seed")
    return {"schema_version": 1,
            "recipe": {name: command[command.index(name) + 1] for name in names},
            "benchmark_source": {name: value for name, value in source_manifest.items()
                                 if name.startswith("RoboSynChallenge/")}}


def prepare(campaign: Path, gate: str, attempt_id: str) -> Path:
    gpus, cap = GATES[gate]
    root = campaign / f"{gate}_{attempt_id}"
    if (root / "launch_manifest.json").exists():
        return root
    source = freeze_repository(root)
    sandbox = cache_repair_bwrap()
    native = gate in {"G1", "G2", "G3", "G6"}
    policy = {"schema_version": 1, "validation": {
        "stage": "native_admission" if native else "research", "force_probe": True,
        "probe_repetitions": 3 if gate == "G3" else 2 if native else 1,
        "inject_failure_worker": gpus - 1 if native else None,
        "inject_failure_generation": 1}}
    save(root / "harness_config.json", policy)
    run_id = f"harness_{gate.lower()}_{attempt_id}"
    seconds = int(cap / gpus * 3600)
    # Include platform startup/release in the cap instead of calling it free time.
    execution_seconds = max(120, seconds - 120)
    command = [str(source / ".venv/bin/autosim"), "research", str(source / "RoboSynChallenge"),
               "--task", "click_bell", "--run-id", run_id, "--output-root", str(root / "runs"),
               "--gpus", "auto", "--max-parallel-jobs", str(gpus), "--probe-if-needed",
               "--controller", "api", "--compute-controller", "api",
               "--compute-env-file", str(PROJECT / ".env"), "--recovery-controller", "api",
               "--recovery-max-calls", "8", "--recovery-max-tokens", "80000",
               "--rounds", "2", "--attempts-per-round", "100" if gate == "G5" else "16",
               "--training-steps", "20000" if gate == "G5" else "200",
               "--development-episodes", "40" if gate == "G5" else "8",
               "--selection-episodes", "100" if gate == "G5" else "8",
               "--final-episodes", "200" if gate == "G5" else "8", "--train-seed", "2000",
               "--hours", str(execution_seconds / 3600), "--gpu-hours", str(cap),
               "--harness-config", str(root / "harness_config.json")]
    manifest = json.loads((root / "source_manifest.json").read_text())
    save(root / "launch_manifest.json", {
        "schema_version": 2, "campaign": str(campaign), "gate": gate, "run_id": run_id,
        "run_root": str(root / "runs/RoboSynChallenge" / run_id), "expected_gpus": gpus,
        "expected_model": "5090", "gpu_hours_cap": cap, "max_container_seconds": execution_seconds,
        "allocation_cap_seconds": seconds, "research_command": command,
        "scientific_contract": scientific_recipe(command, manifest),
        "production_code_identity": code_identity(manifest),
        "harness_config_sha256": sha(root / "harness_config.json"),
        "repair_sandbox": sandbox,
        "previous_allocated_gpu_hours": 0, "acceptance_stage": policy["validation"]["stage"],
        "submission_authority": "user requested full execution of harness plan and final four-GPU submission",
        "budget_group": "new_harness_full_research" if gate == "G5" else "harness_validation_20260915"})
    return root


def admitted(campaign: Path, gate: str, code: str) -> dict:
    required = {"G1": [], "G2": ["G1"], "G3": ["G1", "G2"],
                "G4": ["G1", "G2", "G3"], "G5": ["G1", "G2", "G3", "G4"],
                "G6": ["G1", "G2", "G3"]}[gate]
    g0 = campaign / "g0_acceptance.json"
    if not g0.exists() or json.loads(g0.read_text()).get("passed") is not True:
        raise RuntimeError("G0 CPU/real-process/API coding gates have not passed")
    if json.loads(g0.read_text()).get("production_code_identity") != code:
        raise RuntimeError("G0 did not verify this production source revision")
    baseline = json.loads(g0.read_text())
    cpu = baseline.get("cpu_tests", {})
    if (not isinstance(cpu, dict) or cpu.get("passed") is not True
            or type(cpu.get("tests")) is not int or cpu["tests"] <= 0
            or type(cpu.get("failures")) is not int or cpu["failures"] != 0
            or type(cpu.get("errors")) is not int or cpu["errors"] != 0):
        raise RuntimeError("G0 CPU tests are incomplete or failed")
    _checked_evidence(cpu, "G0 CPU tests")
    coding = baseline.get("api_coding", {})
    if (not isinstance(coding, dict) or coding.get("passed") is not True
            or coding.get("api_used") is not True or coding.get("activation_verified") is not True
            or not isinstance(coding.get("request_ids"), list) or not coding["request_ids"]
            or any(not isinstance(request, str) or not request for request in coding["request_ids"])):
        raise RuntimeError("G0 real API coding and verified activation have not passed")
    _checked_evidence(coding, "G0 real API coding")
    receipts = {}
    for previous in required:
        candidates = []
        for run in campaign.glob(f"{previous}_*/launch_manifest.json"):
            manifest = json.loads(run.read_text())
            receipt = Path(manifest["run_root"]) / "harness_acceptance.json"
            terminal = run.parent / "job_description.json"
            if (manifest.get("production_code_identity") == code and receipt.exists() and terminal.exists()
                    and json.loads(terminal.read_text()).get("state") == "SUCCEEDED"
                    and json.loads(receipt.read_text()).get("passed") is True):
                candidates.append(receipt)
        if not candidates:
            raise RuntimeError(f"{previous} has no successful same-revision real-entrypoint acceptance")
        receipts[previous] = {"path": str(candidates[-1]), "sha256": sha(candidates[-1])}
    return {"g0": {"path": str(g0), "sha256": sha(g0)}, "prerequisites": receipts}


def submit(root: Path) -> dict:
    """Serialize a gate's reservations and create intents across all its attempts."""
    root = Path(root).absolute()
    manifest = json.loads((root / "launch_manifest.json").read_text())
    gate = manifest["gate"]
    if gate not in GATES:
        raise ValueError("unknown harness gate")
    campaign = Path(manifest["campaign"])
    with (campaign / f".{gate}_submission.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _submit_locked(root)


def _submit_locked(root: Path) -> dict:
    manifest = json.loads((root / "launch_manifest.json").read_text())
    campaign, gate = Path(manifest["campaign"]), manifest["gate"]
    if type(manifest.get("expected_gpus")) is not int or manifest["expected_gpus"] != GATES[gate][0]:
        raise RuntimeError("frozen GPU count differs from the registered gate")
    source = root / "source"
    files = json.loads((root / "source_manifest.json").read_text())
    if code_identity(files) != manifest.get("production_code_identity"):
        raise RuntimeError("source manifest no longer matches the verified production revision")
    if any(not (source / p).is_file() or sha(source / p) != expected for p, expected in files.items()):
        raise RuntimeError("frozen source changed")
    if sha(root / "harness_config.json") != manifest["harness_config_sha256"]:
        raise RuntimeError("frozen harness policy changed")
    sandbox = manifest.get("repair_sandbox", {})
    binary = Path(sandbox.get("binary", "/missing-offline-bwrap"))
    expected_binary = (BASE / "harness_validation/dependencies/bwrap").absolute()
    if (binary != expected_binary or not binary.is_file() or binary.is_symlink()
            or not os.access(binary, os.X_OK) or sha(binary) != sandbox.get("sha256")):
        raise RuntimeError("frozen offline bubblewrap dependency is missing or changed")
    save(root / "gate_admission.json", admitted(campaign, gate, manifest["production_code_identity"]))
    needed = [source / ".venv/bin/python", source / "RoboSynChallenge/policy/act/.venv/bin/python",
              source / "RoboSynChallenge/checkpoints/ACT_sim_click_bell/model.safetensors",
              source / "RoboSynChallenge/lerobot_dataset/RoboSynChallenge/cobotmagic_Sim_click_bell/meta/info.json",
              BASE / "optix_runtime/nvoptix.bin", BASE / "container_libs/lib/libnvoptix.so.1",
              BASE / "compute_validation/dependencies/torch/hub/checkpoints/resnet18-f37072fd.pth"]
    if any(not path.is_file() for path in needed):
        raise RuntimeError("an offline execution dependency is missing")
    submission_path = root / "submission.json"
    if submission_path.is_file():
        previous_submission = json.loads(submission_path.read_text())
        if previous_submission.get("name"):
            name = previous_submission["name"]
            details = query(["acp", "jobs", "describe", "--workspace-name", "share-space", name])
            if details.get("name") != name:
                raise RuntimeError("stored submission identity did not reconcile")
            save(root / "job_description.json", details)
            return {**previous_submission, "state": details.get("state"), "reconciled": True}
    rows, page = [], 1
    while True:
        batch = query(["acp", "jobs", "list", "--workspace-name", "share-space", "--user-name", "250010008",
                       "--page-size", "500", "--page-token", str(page)])
        rows += batch
        if len(batch) < 500:
            break
        page += 1
    display = "autosim-" + manifest["run_id"].replace("_", "-")
    matches = [row for row in rows if row.get("display_name") == display]
    if matches:
        if len(matches) != 1:
            raise RuntimeError("ambiguous submission identity")
        if not (root / "submission_intent.json").exists():
            raise RuntimeError("matching display name has no local create intent; refusing to adopt another job")
        result = {"name": matches[0]["name"], "reconciled": True}
        save(root / "submission.json", result)
        save(root / "job_description.json", matches[0])
        return result
    if (root / "submission_intent.json").exists():
        raise RuntimeError("prior create outcome unresolved; no duplicate submission")
    active = [{k: row.get(k) for k in ("name", "display_name", "state", "roles")}
              for row in rows if row["state"] not in TERMINAL]
    save(root / "quota_check.json", {"active_jobs": active, "member_remaining_quota": "unknown"})
    reserved = 0.0
    prior_deadlines = []
    for path in campaign.glob(f"{gate}_*/launch_manifest.json"):
        if path.parent == root or not (path.parent / "submission_intent.json").exists():
            continue
        prior = json.loads(path.read_text())
        if prior.get("run_deadline_utc"):
            prior_deadlines.append(datetime.fromisoformat(prior["run_deadline_utc"]))
        details_path = path.parent / "job_description.json"
        details = json.loads(details_path.read_text()) if details_path.exists() else {}
        end = details.get("complete_time") or details.get("suspend_time")
        if details.get("state") in TERMINAL and end and details.get("start_time"):
            elapsed = (datetime.fromisoformat(end.replace("Z", "+00:00")) -
                       datetime.fromisoformat(details["start_time"].replace("Z", "+00:00"))).total_seconds()
            if elapsed < 0:
                raise RuntimeError("invalid prior allocation timestamps")
            reserved += prior["expected_gpus"] * elapsed / 3600
        else:
            reserved += prior["gpu_hours_cap"]
    remaining = GATES[gate][1] - reserved
    if remaining * 3600 / manifest["expected_gpus"] < 600:
        raise RuntimeError(f"{gate} cumulative budget cannot afford another validation")
    gate_budget = campaign / f"{gate}_budget.json"
    current = datetime.now(timezone.utc)
    original_deadline = current + timedelta(seconds=GATES[gate][1] * 3600 / manifest["expected_gpus"])
    if gate_budget.exists():
        original_deadline = min(original_deadline, datetime.fromisoformat(json.loads(gate_budget.read_text())["deadline_utc"]))
    if prior_deadlines:
        original_deadline = min(original_deadline, *prior_deadlines)
    save(gate_budget, {"gpu_hours_limit": GATES[gate][1], "deadline_utc": original_deadline.isoformat(),
                       "remaining_gpu_hours": remaining, "reserved_or_spent_gpu_hours": reserved})
    seconds = min(manifest["allocation_cap_seconds"], int(remaining * 3600 / manifest["expected_gpus"]),
                  int((original_deadline - current).total_seconds()))
    if seconds < 600:
        raise RuntimeError(f"{gate} original wall-clock budget cannot afford another validation")
    manifest.update(gpu_hours_cap=remaining, previous_allocated_gpu_hours=reserved,
                    max_container_seconds=max(120, seconds - 120),
                    run_deadline_utc=min(original_deadline, current + timedelta(seconds=seconds)).isoformat())
    command = manifest["research_command"]
    command[command.index("--hours") + 1] = str(manifest["max_container_seconds"] / 3600)
    command[command.index("--gpu-hours") + 1] = str(remaining)
    save(root / "launch_manifest.json", manifest)
    remote = shlex.join(["bash", str(source / "tools/run_full_research_sco.sh"), str(root)])
    argv = ["acp", "jobs", "create", "--workspace-name", "share-space", "--aec2-name", "computing-cluster-5090-01g",
            "--job-name", display, "--container-image-url", IMAGE, "--training-framework", "pytorch",
            "--worker-nodes", "1", "--worker-spec", f"n12lp.nn.i10a.{manifest['expected_gpus']}",
            "--priority", "NORMAL", "--quota-type", "reserved", "--storage-mount", MOUNT, "--command", remote]
    save(root / "submission_request.json", {"argv": argv, "requested_gpus": manifest["expected_gpus"],
                                            "remaining_gate_gpu_hours": remaining})
    save(root / "submission_intent.json", {"display_name": display, "status": "create_requested"})
    try:
        result = query(argv)
    except RuntimeError as exc:
        save(root / "submission_failure.json", {"error": str(exc), "retry": "reconciliation_required"})
        raise
    save(root / "submission.json", result)
    if not result.get("name"):
        raise RuntimeError("create returned no unique job identity")
    details = query(["acp", "jobs", "describe", "--workspace-name", "share-space", result["name"]])
    save(root / "job_description.json", details)
    return {**result, "state": details.get("state"), "root": str(root)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=BASE / "harness_validation/20260915")
    parser.add_argument("--gate", required=True, choices=GATES)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.attempt_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("invalid attempt identity")
    root = prepare(args.campaign.absolute(), args.gate, args.attempt_id)
    print(json.dumps(submit(root) if args.submit else {"prepared": str(root), "submitted": False}))


if __name__ == "__main__":
    main()
