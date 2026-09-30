"""Controlled CPU mechanism ablations, NOT robot-policy or official baselines.

All variants use the same frozen Executor, subprocess supervisor, validators,
fixture, and at-most-two-launch allowance. Only durable receipt reuse after a
controller crash and bounded infrastructure retry are toggled. Fault incidence
is deliberately balanced, not an estimate of deployment failure frequency.
"""
import argparse
import hashlib
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

from autosim.experiment_system.executor import Executor, Job
from autosim.research.common import atomic_json, digest, immutable_json, now, read_json


MODULE = "autosim.experiment_validation.fault_suite"
SCENARIOS = ("clean", "transient_exit_137", "controller_crash_after_receipt",
             "invalid_artifact", "evaluation_exit_137", "zero_score_fixture", "unknown_receipt")
VARIANTS = {
    "fixed_no_recovery": (False, False),
    "fixed_with_retry": (False, True),
    "durable_no_retry": (True, False),
    "durable_with_retry": (True, True),
}


def fixture(case: Path, attempt: Path, scenario: str):
    events = case / "executions"
    events.mkdir(exist_ok=True)
    ordinal = len(list(events.glob("*.json"))) + 1
    event = events / f"{uuid.uuid4().hex}.json"
    started, cpu = time.monotonic(), time.process_time()
    record = {"ordinal": ordinal, "pid": os.getpid(), "started_at": now(), "scenario": scenario}
    atomic_json(event, record)
    # Actual bounded CPU work, identical for every variant and every retry.
    payload = b"bounded-cpu-fixture" * 65536
    checksum = hashlib.sha256()
    for _ in range(64):
        checksum.update(payload)
    record.update(checksum=checksum.hexdigest(), work_units=64,
                  work_wall_seconds=time.monotonic() - started,
                  work_cpu_seconds=time.process_time() - cpu, work_completed=True)
    atomic_json(event, record)
    if scenario == "evaluation_exit_137" or (scenario == "transient_exit_137" and ordinal == 1):
        # Known infrastructure exit-code injection, NOT a claim of real GPU OOM.
        os._exit(137)
    if scenario == "invalid_artifact":
        (attempt / "result.json").write_text("{invalid", encoding="utf-8")
    else:
        result = {"status": "completed", "fixture_checksum": checksum.hexdigest(),
                  "synthetic_cpu_fixture": True}
        if scenario == "zero_score_fixture":
            result["success_rate"] = 0  # Tests control semantics; not a robot result.
        atomic_json(attempt / "result.json", result)


def controller(root: Path, inject_marker: Path | None):
    class ControlledExecutor(Executor):
        def _commit(self, job, directory, attempt, receipt):
            if inject_marker:
                # The child has really completed and persisted its receipt.
                # Crash point is identical in all four mechanism variants.
                atomic_json(inject_marker, {"pid": os.getpid(), "receipt": str(attempt / "receipt.json")})
                while True:
                    time.sleep(.1)
            return super()._commit(job, directory, attempt, receipt)
    executor = ControlledExecutor(root)
    result = executor.run(Job(**read_json(root / "fixture/job.json")))
    atomic_json(root / "controller_result.json", result)


def launch(root: Path, marker: Path | None = None):
    command = [sys.executable, "-m", MODULE, "controller", "--root", str(root)]
    if marker:
        command += ["--marker", str(marker)]
    with (root / "controller.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            if marker:
                deadline = time.monotonic() + 10
                while not marker.exists():
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError(f"fault injection point not reached: {root}")
                    time.sleep(.005)
                receipt = Path(read_json(marker)["receipt"])
                if read_json(receipt).get("returncode") != 0:
                    raise RuntimeError("cannot inject post-completion fault without successful receipt")
                process.kill()  # Only this fixture controller; never production processes.
                process.wait(timeout=5)
                return {"status": "injected_controller_crash"}
            process.wait(timeout=10)
            if process.returncode:
                raise RuntimeError(f"fixture controller failed: {root}")
            return read_json(root / "controller_result.json")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def run_case(case: Path, scenario: str, variant: str, sources: dict):
    durable, retry = VARIANTS[variant]
    case.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    root = case / "invocation_0"
    stage = "evaluate" if scenario in {"evaluation_exit_137", "zero_score_fixture"} else "train"
    job = Job("fixture", stage,
              [sys.executable, "-m", MODULE, "fixture", "--case", str(case),
               "--attempt", "{attempt}", "--scenario", scenario],
              str(Path.cwd()), {"result.json": "result"}, sources,
              timeout_seconds=3, budget_seconds=6, max_attempts=2,
              environment={"CUDA_VISIBLE_DEVICES": ""})
    Executor(root).register(job)
    if scenario == "unknown_receipt":
        (root / "fixture/attempt_001").mkdir()
    crash = scenario == "controller_crash_after_receipt"
    result = launch(root, case / "injected.json" if crash else None)
    recovery_started = time.monotonic()
    if crash:
        if not durable:
            # Fixed script restarts its step; no receipt reuse. Outputs remain
            # independently versioned, never overwritten or silently accepted.
            root = case / "invocation_1"
            Executor(root).register(job)
        result = launch(root)
    elif retry and result["status"] == "retry_from_new_attempt":
        result = launch(root)
    recovery_seconds = time.monotonic() - recovery_started if crash or (
        retry and scenario == "transient_exit_137") else None
    committed = result["status"] == "committed"
    if committed:
        # Same final validation in all variants, including fixed controls.
        Executor(root).committed("fixture")
    events = [read_json(p) for p in sorted((case / "executions").glob("*.json"))]
    receipts = [read_json(p) for p in case.glob("invocation_*/fixture/attempt_*/receipt.json")]
    commits = list(case.glob("invocation_*/fixture/commit.json"))
    safe_stop = scenario in {"invalid_artifact", "evaluation_exit_137", "unknown_receipt"}
    expected = {
        "invalid_artifact": "requires_artifact_audit",
        "evaluation_exit_137": "quarantine_evaluation",
        "unknown_receipt": "requires_audit_unknown_execution",
    }.get(scenario, "committed")
    lost_receipt_seconds = 0.0
    if crash and not durable:
        lost_receipt_seconds = read_json(case / "invocation_0/fixture/attempt_001/receipt.json")["elapsed_seconds"]
    row = {"scenario": scenario, "variant": variant, "status": result["status"],
           "committed": committed, "expected_status": expected,
           "expected_outcome_reached": result["status"] == expected,
           "requires_review": result["status"] in {
               "requires_artifact_audit", "quarantine_evaluation", "requires_audit_unknown_execution",
               "retry_from_new_attempt"},
           "actual_human_interventions": None,
           "fixture_launches": len(events), "commit_count": len(commits),
           "successful_work_reexecutions": max(0, len(events) - 1) if crash else 0,
           "successful_receipt_work_discarded_seconds": lost_receipt_seconds,
           "wrong_accept": safe_stop and committed,
           "evaluation_reexecutions": max(0, len(events) - 1) if stage == "evaluate" else 0,
           "total_wall_seconds": time.monotonic() - start, "recovery_wall_seconds": recovery_seconds,
           "worker_wall_seconds": sum(r["elapsed_seconds"] for r in receipts),
           "fixture_cpu_seconds": sum(e.get("work_cpu_seconds", 0) for e in events),
           "raw_directory": str(case), "synthetic_fault_microbenchmark": True,
           "policy_performance_claim": False}
    if len(events) > 2 or len(commits) > 1:
        raise AssertionError("launch or commit budget violated")
    atomic_json(case / "case_result.json", row)
    return row


def summarize(rows):
    result = []
    for variant in VARIANTS:
        for scenario in SCENARIOS:
            group = [r for r in rows if r["variant"] == variant and r["scenario"] == scenario]
            if not group:
                continue
            result.append({"variant": variant, "scenario": scenario, "repeats": len(group),
                           **{k: sum(r[k] for r in group) for k in (
                               "committed", "expected_outcome_reached", "fixture_launches",
                               "successful_work_reexecutions", "wrong_accept", "evaluation_reexecutions",
                               "successful_receipt_work_discarded_seconds", "fixture_cpu_seconds")},
                           "mean_wall_seconds": sum(r["total_wall_seconds"] for r in group) / len(group)})
    return result


def run(output: Path, repeats: int, order_seed: int):
    if not 1 <= repeats <= 20:
        raise ValueError("repeats must be in [1, 20]")
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), package / "experiment_system/executor.py",
             package / "experiment_system/worker.py", package / "research/common.py"]
    sources = {str(p.resolve()): digest(p) for p in paths}
    schedule = [(repeat, scenario, variant) for repeat in range(repeats)
                for scenario in SCENARIOS for variant in VARIANTS]
    random.Random(order_seed).shuffle(schedule)
    manifest = {"created_at": now(), "sources": sources, "repeats": repeats,
                "order_seed": order_seed, "schedule": schedule,
                "max_actual_fixture_launches_per_case": 2, "fixture_timeout_seconds": 3,
                "shared_validation": True, "gpu_usage": False,
                "limits": ["synthetic CPU faults, not representative deployment incidence",
                           "fixed controls are in-house mechanism ablations, not published systems",
                           "does not test robot learning benefit, real OOM or semantic parity",
                           "actual human time not measured; required review is not human time"]}
    immutable_json(output / "manifest.json", manifest)
    rows = []
    for repeat, scenario, variant in schedule:
        case = output / f"repeat_{repeat:02d}" / scenario / variant
        rows.append(run_case(case, scenario, variant, sources))
        atomic_json(output / "progress.json", {"completed_cases": len(rows), "total_cases": len(schedule), "updated_at": now()})
    report = {"status": "completed", "finished_at": now(), "case_count": len(rows),
              "manifest_sha256": digest(output / "manifest.json"), "summary": summarize(rows),
              "cases": rows, "policy_performance_claim": False, "formal_system_comparison_complete": False}
    atomic_json(output / "results.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    suite = sub.add_parser("run")
    suite.add_argument("--output", type=Path, required=True)
    suite.add_argument("--repeats", type=int, default=5)
    suite.add_argument("--order-seed", type=int, default=91006600)
    worker = sub.add_parser("fixture")
    worker.add_argument("--case", type=Path, required=True)
    worker.add_argument("--attempt", type=Path, required=True)
    worker.add_argument("--scenario", choices=SCENARIOS, required=True)
    ctrl = sub.add_parser("controller")
    ctrl.add_argument("--root", type=Path, required=True)
    ctrl.add_argument("--marker", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.output, args.repeats, args.order_seed)
    elif args.command == "fixture":
        fixture(args.case, args.attempt, args.scenario)
    else:
        controller(args.root, args.marker)
