"""Deterministic scheduler fault checks, using a private artifact namespace."""
from pathlib import Path
from .accounting import BudgetLedger
from .common import atomic_json, read_json
from .resource_contracts import ResourceRequest
from .scheduler import Job, Scheduler, ScheduleRefused


def validate_faults(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    devices = [{"index": i, "uuid": f"fixture-{i}", "memory_total_mib": 32000,
                "memory_used_mib": 0, "capabilities": {"cuda": "verified"}} for i in range(2)]
    host = {"cpu_cores": 4, "ram_mib": 8192, "shm_mib": 1024, "scratch_mib": 1024, "io_slots": 1}
    jobs = [Job("a", "probe", input_digest="bank0", code_digest="version1"),
            Job("b", "probe", depends_on=("a",), input_digest="bank1", code_digest="version1")]
    ledger = BudgetLedger(root / "budget.json", wall_limit_seconds=60, gpu_hours_limit=.1, devices=devices)
    def scheduler(rows=jobs):
        return Scheduler(jobs=rows, devices=devices, output=root / "scheduler", ledger=ledger,
                         use_leases=False, host_capacity=host)
    first = scheduler()
    first.start(first.plan_step()["assignments"][0]); first.complete("a")
    restored = scheduler(); restored.restore()
    assert restored.state("a") == "completed" and restored.state("b") == "ready"
    before = len(ledger.charges); restored.restore(); assert len(ledger.charges) == before
    assignment = restored.plan_step()["assignments"][0]
    restored.start(assignment)
    claim_path = restored.job_output("b") / "claim.json"
    original = read_json(claim_path); atomic_json(claim_path, {**original, "token": "stale"})
    rejected = False
    try: restored.complete("b")
    except ScheduleRefused: rejected = True
    assert rejected
    atomic_json(claim_path, original); restored.complete("b", status="failed", detail={"injection":"oom"})
    resumed = scheduler(); resumed.restore(); assert resumed.state("b") == "failed"
    tiny = BudgetLedger(root / "tiny.json", wall_limit_seconds=60, gpu_hours_limit=.0001, devices=devices)
    assert not tiny.may_start(devices=2, estimate_seconds=10)["allowed"]
    impossible = Scheduler(jobs=[Job("oom", "probe", resources=ResourceRequest(per_gpu_memory_mib=64000))],
        devices=devices, output=root / "oom", host_capacity=host, use_leases=False)
    assert not impossible.plan_step()["assignments"]
    result = {"passed": True, "checks": ["completed_work_not_replayed", "dependency_resume",
        "idempotent_cost_settlement", "stale_token_rejected", "failure_not_silently_retried",
        "budget_exhaustion_refused", "memory_admission_refused"],
        "scope": "injected scheduler faults; real CUDA checkpoint probe is a separate artifact"}
    atomic_json(root / "result.json", result)
    return result
