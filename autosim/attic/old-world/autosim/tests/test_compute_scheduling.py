import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from autosim.research.accounting import BudgetLedger
from autosim.research.resource_contracts import ResourceRequest
from autosim.research.scheduler import Job, Scheduler, ScheduleRefused


DEVICES = [{"uuid": f"GPU-{i}", "index": i, "class_name": "small" if i == 0 else "large",
            "memory_total_mib": 8000 if i == 0 else 32000, "memory_used_mib": 500,
            "capabilities": {"train": "verified", "render": "verified" if i == 0 else "unknown"}}
           for i in range(2)]


def scheduler(tmp_path, jobs, **kwargs):
    return Scheduler(jobs=jobs, devices=kwargs.pop("devices", DEVICES), output=tmp_path / "jobs",
                     use_leases=False, host_capacity={"cpu_cores": 4, "ram_mib": 4096,
                     "shm_mib": 128, "scratch_mib": 10000, "io_slots": 2}, **kwargs)


def test_zero_gpu_work_runs_without_any_device(tmp_path):
    s = scheduler(tmp_path, [Job("audit", "audit", device_count=0,
                                resources=ResourceRequest(cpu_cores=1))], devices=[])
    a = s.plan_step()["assignments"][0]
    assert a.devices == ()
    s.start(a)
    s.complete("audit")
    assert s.state("audit") == "completed"


def test_cpu_audit_does_not_consume_the_single_gpu_job_limit(tmp_path):
    s = scheduler(tmp_path, [Job("audit", "audit", device_count=0, resources=ResourceRequest(cpu_cores=1)),
                            Job("train", "training", resources=ResourceRequest(cpu_cores=1))], max_parallel_jobs=1)
    assert {a.job.name for a in s.plan_step()["assignments"]} == {"audit", "train"}


def test_heterogeneous_capability_and_memory_placement(tmp_path):
    s = scheduler(tmp_path, [Job("train", "training", resources=ResourceRequest(
        per_gpu_memory_mib=16000, capabilities=("train",)))])
    assert s.plan_step()["assignments"][0].devices[0]["index"] == 1
    s = scheduler(tmp_path, [Job("render", "evaluation", resources=ResourceRequest(
        per_gpu_memory_mib=16000, capabilities=("render",)))])
    assert not s.plan_step()["assignments"]
    assert s.plan_step()["waiting"][0].reason == "capability_or_memory"


def test_cpu_resources_are_reserved_in_the_plan_and_at_admission(tmp_path):
    jobs = [Job(n, "training", resources=ResourceRequest(cpu_cores=3)) for n in ("a", "b")]
    s = scheduler(tmp_path, jobs)
    step = s.plan_step()
    assert len(step["assignments"]) == 1
    assert step["waiting"][0].reason == "host_resources"
    a = step["assignments"][0]
    s.start(a)
    with pytest.raises(ScheduleRefused):
        s.start(a)
    s.complete("a")
    assert s.plan_step()["assignments"][0].job.name == "b"


def test_parallel_plan_cannot_spend_the_same_budget_twice(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", wall_limit_seconds=1000,
                          gpu_hours_limit=.1, devices=DEVICES)
    s = scheduler(tmp_path, [Job(n, "training", estimate_seconds=250) for n in ("a", "b")], ledger=ledger)
    assert len(s.plan_step()["assignments"]) == 1
    s.start(s.plan_step()["assignments"][0])
    assert not s.plan_step()["assignments"]
    s.complete("a")
    assert not ledger.reservations


def test_concurrent_budget_reservations_are_atomic(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.json", wall_limit_seconds=1000,
                          gpu_hours_limit=.1, devices=DEVICES)
    def reserve(i):
        try:
            ledger.reserve(str(i), devices=1, estimate_seconds=300)
            return True
        except RuntimeError:
            return False
    with ThreadPoolExecutor(8) as pool:
        assert sum(pool.map(reserve, range(8))) == 1


def test_cycles_and_transitive_failed_dependencies(tmp_path):
    with pytest.raises(ValueError, match="cyclic"):
        scheduler(tmp_path, [Job("a", "audit", depends_on=("b",)), Job("b", "audit", depends_on=("a",))])
    s = scheduler(tmp_path, [Job("a", "audit"), Job("b", "audit", depends_on=("a",)),
                             Job("c", "audit", depends_on=("b",))])
    s.start(s.plan_step()["assignments"][0])
    s.complete("a", status="failed")
    assert s.state("c") == "blocked"


def test_changed_inputs_refuse_cached_results(tmp_path):
    s = scheduler(tmp_path, [Job("a", "audit", input_digest="first")])
    s.start(s.plan_step()["assignments"][0]); s.complete("a")
    s = scheduler(tmp_path, [Job("a", "audit", input_digest="different")])
    with pytest.raises(ScheduleRefused, match="inputs changed"):
        s.restore()
