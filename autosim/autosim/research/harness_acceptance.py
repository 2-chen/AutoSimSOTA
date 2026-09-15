"""Independent evidence checks for native admission and real research completion."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from .common import atomic_json, digest, now, read_json


def native_receipt_verdict(receipt: dict, expected_gpus: int) -> dict:
    failures = []
    if not isinstance(receipt, dict):
        return {"passed": False, "failures": ["malformed native receipt"]}
    def device_list(value):
        if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
            failures.append("malformed device UUID list")
            return []
        return value
    devices = device_list(receipt.get("verified_devices", []))
    if receipt.get("passed") is not True or len(devices) != expected_gpus or len(set(devices)) != expected_gpus:
        failures.append("native receipt does not certify the requested distinct devices")
    requested = device_list(receipt.get("requested_devices", []))
    allocation = receipt.get("allocation", [])
    if not isinstance(allocation, list) or any(not isinstance(row, dict) for row in allocation):
        allocation = []
    allocated = device_list([row.get("uuid") for row in allocation])
    if (len(requested) != expected_gpus or len(set(requested)) != expected_gpus
            or set(requested) != set(devices) or not set(requested) <= set(allocated)):
        failures.append("verified UUIDs differ from the requested allocation")
    evidence = receipt.get("concurrency_evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    workers = evidence.get("workers", [])
    if not isinstance(workers, list) or any(not isinstance(row, dict) for row in workers):
        failures.append("malformed native worker evidence")
        workers = []
    if receipt.get("primary_failure") or evidence.get("primary_failure"):
        failures.append("native generation still reports a primary failure")
    if len(workers) != expected_gpus or {str(r.get("uuid")) for r in workers} != set(devices):
        failures.append("native workers do not match the verified UUID set")
    if len({str(r.get("generation")) for r in workers}) != 1:
        failures.append("native workers belong to different generations")
    if len({str(r.get("pid")) for r in workers}) != expected_gpus:
        failures.append("native worker processes are not distinct")
    if len({str(r.get("worker")) for r in workers}) != expected_gpus:
        failures.append("logical worker identities are not distinct")
    for worker in workers:
        if (worker.get("phase") != "completed" or type(worker.get("steps")) is not int
                or worker.get("steps", 0) <= 0):
            failures.append("worker has no completed real rollout steps")
        if (type(worker.get("pid")) is not int or worker.get("pid", 0) <= 0
                or not worker.get("process_start_identity")
                or any(not isinstance(worker.get(name), str) or not worker.get(name)
                       for name in ("worker", "uuid", "generation", "claim", "attempt_id", "host"))):
            failures.append("worker identity evidence is incomplete")
    overlap = None
    try:
        if any(type(r.get(k)) not in (int, float) for r in workers for k in ("work_started", "work_ended")):
            raise ValueError("work interval must contain numeric non-boolean times")
        starts = [float(r["work_started"]) for r in workers]
        ends = [float(r["work_ended"]) for r in workers]
        if not all(math.isfinite(t) for t in [*starts, *ends]) or any(b <= a for a, b in zip(starts, ends)):
            raise ValueError("invalid work interval")
        overlap = min(ends) - max(starts)
        minimum = float(evidence["minimum_overlap_seconds"])
        if not math.isfinite(minimum) or minimum <= 0 or overlap < minimum:
            failures.append("actual rollout intervals do not meet the declared overlap")
        reported = float(evidence["overlap_seconds"])
        if not math.isfinite(reported) or abs(overlap - reported) > .001:
            failures.append("reported overlap differs from the actual worker intervals")
    except (KeyError, TypeError, ValueError):
        failures.append("missing or invalid rollout time evidence")
    if evidence.get("passed") is not True or receipt.get("verified_concurrent_workers") != expected_gpus:
        failures.append("concurrent execution was not verified")
    return {"passed": not failures, "failures": failures, "expected_gpus": expected_gpus,
            "verified_devices": devices, "verified_rollout_concurrency": len(workers),
            "overlap_seconds": overlap, "cache_mode": receipt.get("cache_mode"),
            "startup_attempts": len(receipt.get("startup_attempts", []))}


def inventory_verdict(document: dict, receipt_path: Path, allocation_path: Path | None = None) -> list[str]:
    """Bind device claims to independently captured, local allocation inventories."""
    failures = []
    evidence = document.get("inventory_evidence", {})
    inventory_path = receipt_path.parent / "inventory.json"
    if (not isinstance(evidence, dict) or not inventory_path.is_file()
            or evidence.get("sha256") != digest(inventory_path)
            or Path(evidence.get("artifact", "")).absolute() != inventory_path.absolute()):
        return ["native receipt has no intact independent device inventory"]
    inventory = read_json(inventory_path)
    allocated = {row["uuid"] for row in inventory.get("gpus", [])
                 if row.get("index") in inventory.get("allowed", [])}
    requested = set(document.get("requested_devices", []))
    if not requested or not requested <= allocated:
        failures.append("native devices lie outside the observed container allocation")
    host = inventory.get("host")
    if (not host or evidence.get("host") != host or
            any(row.get("host") != host for row in document.get("concurrency_evidence", {}).get("workers", []))):
        failures.append("worker host differs from independently captured inventory")
    if allocation_path is not None:
        if not allocation_path.is_file():
            failures.append("outer allocation inventory is missing")
        else:
            outer = read_json(allocation_path)
            outer_devices = {row["uuid"] for row in outer.get("gpus", [])
                             if row.get("index") in outer.get("allowed", [])}
            if outer.get("host") != host or outer_devices != allocated or requested != outer_devices:
                failures.append("native inventory differs from the outer frozen allocation")
    return failures


def validate_run(run_root: Path, *, expected_gpus: int, stage: str = "research",
                 source_manifest: Path | None = None, source_root: Path | None = None) -> dict:
    """Keep performance and valid scientific branches separate from harness health."""
    run_root = Path(run_root)
    state = read_json(run_root / "run_state.json")
    failures, phases, native = [], [], []
    listed = run_root / "native_admission_runs.json"
    paths = ([Path(r["artifact"]) for r in read_json(listed)] if listed.exists()
             else [run_root / "device_probe/device_probe.json"])
    if not paths:
        failures.append("native admission list is empty")
    receipt_documents = []
    for path in paths:
        if not path.is_file():
            failures.append("native admission artifact missing")
            continue
        document = read_json(path)
        receipt_documents.append(document)
        verdict = native_receipt_verdict(document, expected_gpus)
        if verdict["passed"]:
            allocation_path = source_manifest.parent / "allocation_inventory.json" if source_manifest else None
            inventory_failures = inventory_verdict(document, path, allocation_path)
            verdict["failures"].extend(inventory_failures)
            verdict["passed"] = not verdict["failures"]
        native.append({**verdict, "artifact": str(path), "sha256": digest(path)})
        failures.extend(verdict["failures"])
    environment_path = run_root / "environment_manifest.json"
    environment = read_json(environment_path) if environment_path.exists() else {}
    if environment.get("status") != "passed":
        failures.append("environment/checkpoint contract not verified")
    source_verified = None
    if source_manifest is not None:
        if source_root is None:
            raise ValueError("source root is required with a source manifest")
        source_verified = all((Path(source_root) / name).is_file() and
                              digest(Path(source_root) / name) == expected
                              for name, expected in read_json(Path(source_manifest)).items())
        if not source_verified:
            failures.append("frozen source changed")
    if stage == "native_admission":
        policy_file = run_root / "harness_policy.json"
        policy = read_json(policy_file).get("validation", {}) if policy_file.exists() else {}
        repetitions = policy.get("probe_repetitions")
        if type(repetitions) is not int or len(receipt_documents) != repetitions:
            failures.append("native probe repetitions differ from the frozen validation policy")
        else:
            modes = ["cold"] * repetitions
            if repetitions > 1:
                modes[-1] = "warm"
            if [row.get("cache_mode") for row in receipt_documents] != modes:
                failures.append("cold/warm probe coverage differs from the frozen policy")
        if policy.get("inject_failure_worker") is not None:
            first = receipt_documents[0] if receipt_documents else {}
            injection = first.get("validation_injection", {})
            attempts = first.get("startup_attempts", [])
            if (injection.get("applied") is not True or len(attempts) < 2
                    or attempts[0].get("passed") is not False or attempts[-1].get("passed") is not True):
                failures.append("injected worker failure and successful bounded recovery were not demonstrated")
        if state.get("stage") != "native_admission_complete" or state.get("native_admission_passed") is not True:
            failures.append("actual research CLI did not complete native admission")
        if state.get("rounds") or state.get("final_confirmation_opened"):
            failures.append("admission-only validation unexpectedly entered research")
    elif stage == "research":
        if state.get("status") not in {"completed", "completed_without_target_improvement"}:
            failures.append("research did not reach a normal business terminal state")
        for field in ("execution_complete", "result_valid", "export_runtime_verified", "api_closed_loop_completed"):
            if state.get(field) is not True:
                failures.append(f"research contract requires {field}")
        rows = sorted((run_root / "rounds").glob("round_*/round_result.json"))
        valid = 0
        for path in rows:
            row = read_json(path)
            phases.append({"round": row.get("round"), "status": row.get("status"), "artifact": str(path)})
            if row.get("status") != "completed":
                continue
            valid += 1
            exposure = read_json(Path(row["training_exposure_audit"]))
            if not any(p.get("source_kind") == "research_requested_collection" and
                       int(p.get("yielded_samples", 0)) > 0 for p in exposure.get("parts", [])):
                failures.append("API-requested new data did not reach a real training batch")
            checkpoint = Path(row["checkpoint"]) / "model.safetensors"
            if not checkpoint.is_file() or digest(checkpoint) != row.get("checkpoint_sha256"):
                failures.append("trained checkpoint identity changed")
            evaluation = read_json(Path(row["development_evaluation"]) / "evaluation_metrics.json")
            if evaluation.get("execution_mode") != "real_simulation" or not evaluation.get("episodes"):
                failures.append("candidate did not complete native development evaluation")
        if not valid:
            failures.append("no valid trained candidate completed the real API loop")
        export = read_json(run_root / "export_validation.json") if (run_root / "export_validation.json").exists() else {}
        if (export.get("export_runtime_verified") is not True or
                export.get("native_episode_execution_mode") != "real_simulation" or
                int((export.get("native_episode_summary") or {}).get("episode_count", 0)) < 1):
            failures.append("exported repository has not completed its own native episode")
        if not state.get("final_confirmation_opened") and not (run_root / "final_confirmation_not_opened.json").exists():
            failures.append("missing evidence for the legitimate unopened final branch")
    else:
        raise ValueError("unknown harness acceptance stage")
    report = {"schema_version": 1, "created_at": now(), "passed": not failures,
              "stage": stage, "run_root": str(run_root), "failures": failures,
              "environment_contract_passed": environment.get("status") == "passed",
              "native_admission_passed": bool(native) and all(row["passed"] for row in native),
              "verified_rollout_concurrency": min((row["verified_rollout_concurrency"] for row in native), default=0),
              "failure_recovery_passed": any(row.get("validation_injection", {}).get("applied") is True
                  and len(row.get("startup_attempts", [])) > 1 and row.get("passed") is True
                  for row in receipt_documents),
              "source_manifest_verified": source_verified, "native_admission": native,
              "phases": phases, "execution_complete": bool(state.get("execution_complete")),
              "result_valid": bool(state.get("result_valid")),
              "export_runtime_verified": bool(state.get("export_runtime_verified")),
              "performance_improved": bool(state.get("performance_target_achieved")),
              "final_branch_executed": bool(state.get("final_confirmation_opened")),
              "scope": "execution evidence; performance improvement is a separate claim"}
    if source_manifest is not None:
        admission_path = source_manifest.parent / "gate_admission.json"
        admission = read_json(admission_path) if admission_path.exists() else {}
        g0_ref = admission.get("g0", {})
        g0_path = Path(g0_ref.get("path", ""))
        g0 = read_json(g0_path) if g0_path.is_file() and digest(g0_path) == g0_ref.get("sha256") else {}
        report.update(repair_patch_validated=g0.get("api_coding", {}).get("passed") is True,
                      continuation_idempotency_passed=g0.get("continuation_idempotency_passed") is True,
                      prerequisite_evidence=admission,
                      continuation_journal=str(source_manifest.parent / "continuation/continuation.json"))
    atomic_json(run_root / "harness_acceptance.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--expected-gpus", type=int, required=True, choices=(1, 2, 4, 8))
    parser.add_argument("--stage", choices=("research", "native_admission"), default="research")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    result = validate_run(args.run_root, expected_gpus=args.expected_gpus, stage=args.stage,
                          source_manifest=args.source_manifest, source_root=args.source_root)
    print({"passed": result["passed"], "failures": result["failures"]})
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
