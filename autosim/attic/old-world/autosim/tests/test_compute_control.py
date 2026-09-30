import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pytest

from autosim.research.common import object_digest, read_json
from autosim.research.compute_agent import ComputeAgent
from autosim.research.compute_proposals import validate_proposal
from autosim.research.compute_planner import ComputePlanner, publish_execution_plan
from autosim.research.patch_validation import checked_function
from autosim.research.profiling import ProfileStore
from autosim.research.scheduler import Job
from autosim.research.fault_validation import validate_faults


class Client:
    model = "fixture"
    base_url = "https://fixture.invalid"
    calls = 0
    fail = False
    def chat_with_metadata(self, *args, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("transport unavailable")
        return '{"ok":true}', {"usage":{"total_tokens":11}}


def test_request_cache_prevents_duplicate_api_and_charges_once(tmp_path):
    client = Client(); agent = ComputeAgent(tmp_path, client)
    a = agent.request("test", {"a":1}, system="bounded")
    b = agent.request("test", {"a":1}, system="bounded")
    assert a == b and client.calls == 1
    assert read_json(tmp_path / "budget.json")["charged_tokens"] == 11


def test_unknown_remote_outcome_is_reserved_and_not_reissued(tmp_path):
    client = Client(); client.fail = True
    agent = ComputeAgent(tmp_path, client)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            agent.request("test", {}, system="bounded")
    assert client.calls == 1
    assert read_json(tmp_path / "budget.json")["reserved_tokens"] > 0


@pytest.mark.parametrize("repair_works", [True, False])
def test_invalid_api_schema_gets_one_charged_repair_then_stops(tmp_path, repair_works):
    snapshot = {"host":{"cpu_cores":4}}
    valid = {"schema_version":1,"based_on_snapshot":object_digest(snapshot),
             "evidence_refs":["host.cpu_cores"],"actions":[{
                 "type":"set_execution_parameter","parameter":"omp_threads","value":2}]}
    class PlannerClient(Client):
        def chat_with_metadata(self,*args,**kwargs):
            self.calls += 1
            schema = 1 if repair_works and self.calls == 2 else "1"
            return json.dumps({**valid,"schema_version":schema}), {"usage":{"total_tokens":11}}
    client = PlannerClient(); agent = ComputeAgent(tmp_path,client)
    if repair_works:
        assert agent.plan(snapshot,{"omp_threads":4})["schema_version"] == 1
        assert agent.plan(snapshot,{"omp_threads":4})["schema_version"] == 1
    else:
        with pytest.raises(ValueError): agent.plan(snapshot,{"omp_threads":4})
    assert client.calls == 2
    assert read_json(tmp_path / "budget.json")["charged_tokens"] == 22
    assert len(list((tmp_path / "rejections").glob("*.json"))) == (1 if repair_works else 2)


def test_action_contract_rejects_scientific_changes_and_stale_input():
    base = {"schema_version":1,"based_on_snapshot":"s","evidence_refs":["host.cpu_cores"],
            "actions":[{"type":"set_execution_parameter","parameter":"global_batch","value":32}]}
    with pytest.raises(ValueError): validate_proposal(base, snapshot_digest="s", limits={})
    base["actions"][0]["parameter"] = "omp_threads"
    with pytest.raises(ValueError): validate_proposal(base, snapshot_digest="new", limits={})
    with pytest.raises(ValueError): validate_proposal(base, snapshot_digest="s", limits={"omp_threads":4})


@pytest.mark.parametrize("schema", [True, 1.0, "1"])
def test_schema_version_requires_an_integer(schema):
    proposal = {"schema_version":schema,"based_on_snapshot":"s",
        "evidence_refs":["host.cpu_cores"],"actions":[{
            "type":"set_execution_parameter","parameter":"omp_threads","value":2}]}
    with pytest.raises(ValueError,match="schema"):
        validate_proposal(proposal,snapshot_digest="s",limits={"omp_threads":4})


@pytest.mark.parametrize("code", [
    'import os\ndef _padding_audit(lengths, chunk_size=50): return 1',
    'def _padding_audit(lengths, chunk_size=50): return open("/tmp/unsafe", "w")',
    'def _padding_audit(lengths, chunk_size=50): return lengths.__class__',
    'def _padding_audit(lengths, chunk_size=sum(range(1000000000))): return 1',
    'def _padding_audit(lengths, chunk_size=50): return _padding_audit(lengths)',
])
def test_numeric_patch_cannot_execute_os_code_or_expensive_defaults(code):
    with pytest.raises(ValueError): checked_function(code)


def test_concurrent_profile_samples_are_preserved_and_environment_scoped(tmp_path):
    store = ProfileStore(tmp_path)
    context = dict(workload="audit",code_digest="v1",environment_id="node1",device_class="cpu")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: store.record(context, seconds=n+1), range(12)))
    measured = store.estimate(context)
    assert measured["count"] == 12 and measured["confidence"] == "measured"
    assert store.estimate({**context,"environment_id":"node2"})["confidence"] == "unknown"


def test_resource_migration_does_not_change_job_input_identity():
    assert Job("x","probe",device_count=1,input_digest="bank").digest == Job(
        "x","probe",device_count=4,input_digest="bank",estimate_by_class={"h100":2}).digest
    assert Job("x","probe",input_digest="bank").digest != Job("x","probe",input_digest="other").digest


def test_execution_versions_share_scientific_protocol(tmp_path):
    a = publish_execution_plan(tmp_path,{"uuid":"old"},protocol_digest="science")
    b = publish_execution_plan(tmp_path,{"uuid":"new"},protocol_digest="science")
    assert a != b
    assert read_json(a)["protocol_digest"] == read_json(b)["protocol_digest"] == "science"


def test_fault_receipts_are_not_success_retries(tmp_path):
    assert validate_faults(tmp_path)["passed"]


def test_planner_never_exceeds_explicit_parallel_limit(tmp_path):
    plan = {"usable":[{"uuid":str(i)} for i in range(8)],"max_parallel_jobs":2,"parallel_cap":2,
            "host_capacity":{"cpu_cores":4,"ram_mib":8192,"io_slots":2}}
    planner = ComputePlanner(tmp_path)
    a = planner.propose(plan)
    assert a["parameters"]["max_parallel_jobs"] == 2
    assert planner.propose(plan) == a


def test_audit_optimizer_small_workload_never_calls_api(tmp_path):
    from autosim.research.compute_codegen import AuditOptimizer
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/episodes.jsonl").write_text('{"length": 10}\n')
    client = Client()
    optimizer = AuditOptimizer(tmp_path / "optimizer", ComputeAgent(tmp_path / "api", client))
    assert optimizer.prepare(dataset, remaining_seconds=1000) is None
    assert client.calls == 0
    decisions = list((tmp_path / "optimizer/decisions").glob("*.json"))
    assert read_json(decisions[0])["reason"] == "imminent_audit_too_small_to_repay_codegen"


def test_audit_optimizer_budget_fallback_does_not_read_data_or_api(tmp_path):
    from autosim.research.compute_codegen import AuditOptimizer
    client = Client()
    optimizer = AuditOptimizer(tmp_path / "optimizer", ComputeAgent(tmp_path / "api", client))
    assert optimizer.prepare(tmp_path / "absent", remaining_seconds=100) is None
    assert client.calls == 0
    decision = read_json(next((tmp_path / "optimizer/decisions").glob("*.json")))
    assert decision["reason"] == "insufficient_wall_budget"


def test_activated_patch_timeout_restores_reference_for_the_same_inputs(tmp_path, monkeypatch):
    from autosim import robosyn_data
    from autosim.research import compute_codegen
    from autosim.research.common import atomic_json, digest
    from autosim.research.data_execution import audit_kernel_context
    reference = robosyn_data._padding_audit
    source = compute_codegen.extract_function(Path(robosyn_data.__file__), "_padding_audit")
    candidate = tmp_path / "candidate.py"; candidate.write_text(source)
    base = tmp_path / "reference.py"; base.write_text(source)
    registry = tmp_path / "active.json"
    atomic_json(registry, {"patch_id":"fixture", "candidate":str(candidate),
        "candidate_sha256":digest(candidate),"source_path":str(base),"base_sha256":digest(base),
        "verdict":{"passed":True}})
    def timeout(*args, **kwargs):
        raise TimeoutError("injected bounded-worker timeout")
    monkeypatch.setattr(compute_codegen, "measure", timeout)
    with audit_kernel_context(registry) as activation:
        assert robosyn_data._padding_audit(iter([1,3]), 2) == reference([1,3], 2)
        assert activation["status"] == "rolled_back_for_call"
    assert robosyn_data._padding_audit is reference
    assert len(list((tmp_path / "rollbacks").glob("*.json"))) == 1


def test_local_scratch_capacity_is_not_the_shared_output_filesystem(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace
    from autosim.research import host_resources
    from autosim.research.resource_contracts import host_inventory
    monkeypatch.setattr(host_resources, "capacity", lambda: {
        "cpu":{"effective_cpus":4},"memory":{"available_mib":256}})
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(
        free=(10 if str(path)=="/tmp" else 1000)*2**20))
    measured = host_inventory(tmp_path)
    assert measured["scratch_mib"] == 10
    assert measured["output_free_mib"] == 1000


@pytest.mark.parametrize("augmentation,explicit_workers,expected", [
    ("none", None, 1), ("photometric_mild", None, 4), ("none", 3, 3)])
def test_worker_tuning_preserves_stochastic_transform_rng_contract(
        tmp_path, monkeypatch, augmentation, explicit_workers, expected):
    from autosim.research.runtime import Runtime
    from autosim.research import data_version
    from tests.test_runtime_sharding import SPEC
    trainer = tmp_path / "policy/act/scripts/train.py"
    trainer.parent.mkdir(parents=True); trainer.write_text("# fixture\n")
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text('{"codebase_version":"v2.1"}')
    monkeypatch.setattr(data_version, "training_versions", lambda *args: [])
    runtime = Runtime(tmp_path,tmp_path / "runtime",repo_path=tmp_path,
        execution_resources={"cpu_cores":2},cost_model={"execution_parameters":{"loader_workers":1}})
    seen = []
    def run(command, output, timeout, **kwargs):
        seen.append(command)
        checkpoint = Path(command[command.index("--output-dir")+1]) / "checkpoints/000002/pretrained_model"
        checkpoint.mkdir(parents=True); (checkpoint / "model.safetensors").write_bytes(b"fixture")
    monkeypatch.setattr(runtime,"run",run)
    params = {"image_augmentation_profile":augmentation}
    if explicit_workers is not None:
        params["num_workers"] = explicit_workers
    runtime.train(SPEC,dataset,tmp_path / "training",steps=2,params=params)
    assert seen[0][seen[0].index("--num-workers")+1] == str(expected)


@pytest.mark.parametrize("io_limit", [1,2])
def test_planned_io_limit_controls_real_cpu_job_concurrency(tmp_path, io_limit):
    import threading
    import time
    from autosim.research.runtime import Runtime
    from autosim.research.job_runner import run_phase, PhaseJob
    from autosim.research.resource_contracts import ResourceRequest
    lock = threading.Lock()
    active = peak = 0
    def audit(bound):
        nonlocal active, peak
        with lock:
            active += 1; peak = max(peak, active)
        time.sleep(.04)
        with lock:
            active -= 1
        return {"passed":True}
    plan = {"usable":[],"unified_scheduler":True,"max_parallel_jobs":1,
        "execution_parameters":{"io_slots":io_limit},
        "host_capacity":{"cpu_cores":4,"ram_mib":4096,"io_slots":4}}
    results = run_phase(phase="cpu_io", runtime=Runtime(tmp_path,tmp_path / "runtime"),
        plan=plan, output=tmp_path / "schedule",
        jobs=[PhaseJob(str(i),"audit",audit,device_count=0,reserve_seconds=0,
            resources=ResourceRequest(cpu_cores=1,io_slots=1)) for i in range(4)])
    assert len(results)==4 and peak==io_limit
    assert plan["host_capacity"]["io_slots"]==4


def test_timing_profiles_pool_shapes_while_replay_identity_stays_distinct():
    from autosim.research.runtime import Runtime
    base = {"spec":"task","steps":100,"params":{"batch_size":4},"seed":1,"weights":"first"}
    other = {**base,"seed":2,"weights":"second"}
    assert object_digest(base)!=object_digest(other)
    assert Runtime._profile_digest("training",base)==Runtime._profile_digest("training",other)
    assert Runtime._profile_digest("training",base)!=Runtime._profile_digest("training",{**base,"steps":200})
    assert Runtime._profile_digest("evaluation",{"spec":"task"},block_size=8)!=Runtime._profile_digest(
        "evaluation",{"spec":"task"},block_size=16)


def test_unpatched_audit_cannot_observe_another_threads_scoped_kernel(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from autosim import robosyn_data
    from autosim.research import data_execution
    reference = robosyn_data._padding_audit
    started = threading.Event()
    def activate(_):
        robosyn_data._padding_audit = lambda *_: "scoped override"
        return {"status":"fixture"}
    monkeypatch.setattr(data_execution,"activate_audit_kernel",activate)
    def unpatched():
        started.set()
        with data_execution.audit_kernel_context(None):
            return robosyn_data._padding_audit
    with ThreadPoolExecutor(max_workers=1) as executor:
        with data_execution.audit_kernel_context(tmp_path / "registry"):
            future = executor.submit(unpatched)
            assert started.wait(2)
            assert not future.done()
        assert future.result(timeout=2) is reference
