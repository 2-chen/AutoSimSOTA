#!/usr/bin/env python3
"""One already-allocated SCO node: component validation, no AutoResearch loop."""
from __future__ import annotations
import argparse
import dataclasses
import os
import signal
import sys
import threading
import time
from pathlib import Path

from autosim.research.accounting import BudgetLedger, UtilizationSampler
from autosim.research.common import atomic_json, digest, object_digest, read_json
from autosim.research.data_execution import small_dataset
from autosim.research.devices import discover, select, identity_is_safe
from autosim.research.job_runner import PhaseJob, run_phase
from autosim.research.registry import load_task
from autosim.research.resource_contracts import ResourceRequest, host_inventory
from autosim.research.runtime import Runtime, startup_census


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gpus", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seconds", type=int, default=3540)
    parser.add_argument("--training-steps", type=int, default=100)
    parser.add_argument("--native-episodes", type=int, default=16)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--api-evidence", type=Path)
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("--reuse-components", type=Path)
    args = parser.parse_args()
    root = args.output.absolute(); root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    summary = {"kind": "compute_component_validation", "requested_gpus": args.expected_gpus,
               "started": started, "tests": {}, "full_autoresearch": False, "status": "running"}
    atomic_json(root / "summary.json", summary)
    inventory = discover(disk_path=root)
    atomic_json(root / "inventory.json", inventory)
    devices = [d for d in inventory["gpus"] if d["index"] in inventory["allowed"]]
    summary["actual_gpus"] = len(devices)
    if len(devices) != args.expected_gpus or any("5090" not in d.get("model", "") for d in devices):
        summary.update(status="failed", error="allocation does not match requested 5090 count")
        atomic_json(root / "summary.json", summary)
        return 2
    for device in devices:
        # Probe work is admitted without claiming unmeasured simulation/train capabilities.
        device["selection"] = select(device, "identity").as_dict()
    safe, reason = identity_is_safe(inventory)
    if not safe:
        raise RuntimeError(f"identity addressing refused: {reason}")
    host = host_inventory(root)
    plan = {"schema_version": 2, "unified_scheduler": True, "mode": "identity", "usable": devices,
            "max_parallel_jobs": len(devices), "host_capacity": host, "legacy_equivalence": False}
    control_snapshot = {"usable":[dict(d) for d in devices],"host_capacity":host,
                        "max_parallel_jobs":len(devices),"parallel_cap":len(devices)}
    atomic_json(root / "compute_control_request.json", {"plan":control_snapshot,
                "request_digest":object_digest(control_snapshot),"output":str(root)})
    ledger = BudgetLedger(root / "budget.json", wall_limit_seconds=args.seconds,
                          gpu_hours_limit=args.expected_gpus, devices=devices)
    ledger.begin_allocation(root.name, len(devices), started_epoch=started)
    runtime = Runtime(args.workspace, root, deadline=time.monotonic() + max(0,args.seconds-(time.time()-started)),
                      repo_path=args.workspace / "RoboSynChallenge",
                      eval_repo_path=args.workspace / "RoboSynChallenge", plan=plan)
    runtime.compute_ledger = ledger
    if args.continue_from:
        previous = read_json(args.continue_from / "summary.json")
        reconciliation = read_json(args.continue_from / "merge_reconciliation.json")
        previous_bank = read_json(args.continue_from / "fixed_native_result.json")
        from autosim.robosyn_data import evaluation_seed_bank
        expected_bank = evaluation_seed_bank(2000000211, args.native_episodes)
        if (previous["requested_gpus"] != args.expected_gpus or reconciliation["status"] != "passed"
                or [r["episode_seed"] for r in previous_bank["episodes"]] != expected_bank):
            raise ValueError("continuation evidence differs from the frozen validation matrix")
        for name in ("T2_queue", "T1_native_concurrency", "T2_native_fixed_bank"):
            if previous["tests"][name]["status"] != "passed":
                raise ValueError(f"previous component did not pass: {name}")
            summary["tests"][name] = {**previous["tests"][name],"evidence_source":str(args.continue_from),
                "reused_without_execution":True}
        summary["component_continuation"] = {"source":str(args.continue_from),
            "merge_receipt_sha256":digest(args.continue_from / "fixed_native/merge_receipt.json"),
            "remaining_components":"collection, data, short ACT training, native short chain, API execution"}
    collected = {}
    if args.reuse_components:
        previous = read_json(args.reuse_components / "summary.json")
        training = read_json(args.reuse_components / "training_results.json")
        if previous["requested_gpus"] != args.expected_gpus or len(training) != args.expected_gpus:
            raise ValueError("component source has a different resource shape")
        for row in training.values():
            if digest(Path(row["checkpoint"]) / "model.safetensors") != row["weight_sha256"]:
                raise ValueError("previous training checkpoint changed")
            recipe = read_json(Path(row["checkpoint"]).parents[3] / f"recipe_{args.training_steps}.json")
            if (recipe["steps"] != args.training_steps or recipe["seed"] != 3014
                    or recipe["params"]["batch_size"] != 4 or recipe["params"]["num_workers"] != 1):
                raise ValueError("previous training fixture differs from this validation contract")
        collection_path = args.reuse_components / "collection_results.json"
        if collection_path.exists():
            collected = read_json(collection_path)
        names = ["T3_data_audit","T3_short_training"]
        if collected and previous["tests"].get("T4_collection",{}).get("status")=="passed":
            names.append("T4_collection")
        if not args.continue_from:
            blocks = list((args.reuse_components / "schedules/T2_queue").glob("block_*/fixed_cuda.json"))
            if len(blocks)!=16 or any(read_json(p).get("steps")!=1000 for p in blocks):
                raise ValueError("previous queue does not match the fixed work contract")
            names += ["T2_queue","T5_single_gpu_checkpoint","T5_orphan_cleanup"]
        for name in names:
            if previous["tests"][name]["status"] != "passed":
                raise ValueError(f"cannot reuse unsuccessful component {name}")
            summary["tests"][name] = {**previous["tests"][name],"reused_without_execution":True,
                                      "evidence_source":str(args.reuse_components)}
        summary["component_reuse"] = {"source":str(args.reuse_components),
            "summary_sha256":digest(args.reuse_components / "summary.json"),
            "collection_results_sha256":digest(collection_path) if collection_path.exists() else None,
            "training_results_sha256":digest(args.reuse_components / "training_results.json")}
    spec = load_task(runtime.repo, "click_bell")
    checkpoint = runtime.repo / "checkpoints/ACT_sim_click_bell"
    per_worker_cpu = max(1, min(4, int(host["cpu_cores"] // len(devices))))
    resources = ResourceRequest(cpu_cores=per_worker_cpu, ram_mib=4096, per_gpu_memory_mib=2048)
    def phase(name, jobs):
        print(f"[{name}] starting {len(jobs)} work items", flush=True)
        summary["stage"] = name; atomic_json(root / "summary.json", summary)
        try:
            result = run_phase(phase=name, jobs=jobs, runtime=runtime, plan=plan,
                               output=root / "schedules", ledger=ledger, run_id=root.name)
            summary["tests"][name] = {"status": "passed", "items": len(result)}
            return result
        except Exception as exc:
            summary["tests"][name] = {"status": "failed", "error": str(exc)[:700]}
            raise
        finally:
            atomic_json(root / "summary.json", summary)

    def cuda(bound):
        output = bound.output / "cuda.json"
        env_selection = bound.training_selection()
        bound.run([str(bound.python), "-m", "autosim.research.validation_worker",
                   "--uuid", bound.selection.uuid, "--output", str(output), "--seconds", "3"],
                  bound.output / "process", 120, selection=env_selection)
        return read_json(output)

    def fixed_cuda(bound):
        output = bound.output / "fixed_cuda.json"
        bound.run([str(bound.python), "-m", "autosim.research.validation_worker", "--uuid", bound.selection.uuid,
                   "--output", str(output), "--steps", "1000"], bound.output / "process", 120,
                   selection=bound.training_selection())
        return read_json(output)

    sampler = UtilizationSampler(interval_seconds=2)
    try:
        with sampler:
            cuda_rows = phase("T0_cuda", [PhaseJob(f"cuda_{i:02d}", "probe", cuda,
                                  estimate_seconds=30, reserve_seconds=0, resources=resources)
                                          for i in range(len(devices))])
            for device in devices:
                if device["uuid"] not in {row["uuid"] for row in cuda_rows.values()}:
                    raise RuntimeError("CUDA probe did not cover all requested devices")
                device["capabilities"] = {"cuda": "verified"}
            if not args.continue_from and "T2_queue" not in summary["tests"]:
                phase("T2_queue", [PhaseJob(f"block_{i:02d}", "probe", fixed_cuda,
                                            estimate_seconds=15, reserve_seconds=0, resources=resources,
                                            input_digest=object_digest({"block": i, "seed": 14}))
                                    for i in range(16)])
            from autosim.research.fault_validation import validate_faults
            summary["tests"]["T5_scheduler_faults"] = validate_faults(root / "faults")
            def contract_probe(bound):
                world = len(bound.assigned_devices)
                attempts = []
                variants = [("default", {})]
                if world > 1:
                    variants.append(("socket_diagnostic", {"NCCL_P2P_DISABLE":"1",
                        "NCCL_IB_DISABLE":"1", "NCCL_NET":"Socket"}))
                for label, overrides in variants:
                    output = bound.output / label
                    command = [str(bound.python), "-m", "torch.distributed.run", "--standalone",
                               f"--nproc-per-node={world}", "-m", "autosim.research.training_contract_probe",
                               "--output", str(output)]
                    selection = dataclasses.replace(bound.selection,
                        cuda_visible=",".join(d["uuid"] for d in bound.assigned_devices), torch_index=0,
                        extra_env={**bound.selection.extra_env,"NCCL_DEBUG":"INFO",**overrides})
                    try:
                        bound.run(command, output / "process", 180, selection=selection)
                        rows = [read_json(output / f"rank_{rank}.json") for rank in range(world)]
                        if not all(r["passed"] for r in rows):
                            raise RuntimeError("training contract probe failed")
                        return {"verified":True,"ranks":rows,"transport_variant":label,"prior_attempts":attempts,
                                "scope":"transport/linear-model contract, not ACT DDP certification"}
                    except RuntimeError as exc:
                        attempts.append({"variant":label,"error":str(exc),"evidence":str(output)})
                return {"verified":False,"attempts":attempts,"automatic_ACT_DDP":False}
            if args.reuse_components and not args.continue_from:
                summary["tests"]["T3_T5_training_contract"] = {
                    "status":"not_run","optional_capability":True,"automatic_ACT_DDP":False,
                    "reason":"remaining-component attempt; previous transport failure retained separately",
                    "evidence_source":str(args.reuse_components)}
            else:
                contract = phase("T3_T5_training_contract", [PhaseJob("contract", "probe", contract_probe,
                    device_count=len(devices), estimate_seconds=180, reserve_seconds=0,
                    resources=dataclasses.replace(resources, cpu_cores=len(devices),ram_mib=2048*len(devices),homogeneous=True))])["contract"]
                if not contract["verified"]:
                    summary["tests"]["T3_T5_training_contract"].update(status="not_verified",optional_capability=True,
                        fallback="independent single-GPU ACT jobs",evidence=contract)
                    fallback = phase("T5_single_gpu_checkpoint", [PhaseJob("checkpoint", "probe",contract_probe,
                        estimate_seconds=90,reserve_seconds=0,resources=resources)])["checkpoint"]
                    if not fallback["verified"]:
                        raise RuntimeError("single-GPU checkpoint contract also failed")
                if len(devices) in (2,4):
                    def orphan(bound):
                        output = bound.output / "orphan"
                        bound.run([str(bound.python),"-m","autosim.research.orphan_probe","--output",str(output)],
                                  bound.output / "process",120,selection=bound.training_selection())
                        return read_json(output / "result.json")
                    phase("T5_orphan_cleanup", [PhaseJob("orphan", "probe", orphan,
                        estimate_seconds=60,reserve_seconds=0,resources=resources)])
            if not args.skip_training and not args.reuse_components:
                dataset = runtime.repo / "lerobot_dataset/RoboSynChallenge/cobotmagic_Sim_click_bell"
                fixture = small_dataset(dataset, root / "fixture", 2)
                audits = phase("T3_data_audit", [PhaseJob("data_audit", "audit",
                    lambda bound: bound.prepare_data(spec, fixture, root / "data_audit"),
                    estimate_seconds=60, reserve_seconds=0, device_count=0,
                    resources=ResourceRequest(cpu_cores=2, ram_mib=4096, io_slots=1))])
                def train(bound):
                    trained = bound.train(spec, fixture, bound.output / "training", steps=args.training_steps,
                                          params={"batch_size": 4, "num_workers": 1}, seed=3014,
                                          pretrained=checkpoint)
                    bound.run([str(bound.train_python),"-m","autosim.research.act_checkpoint_probe",
                        "--checkpoint",str(trained),"--output",str(bound.output / "checkpoint_reload.json"),
                        "--expected-updates",str(args.training_steps)],bound.output / "reload_process",120,
                        selection=bound.training_selection())
                    return {"checkpoint": str(trained), "weight_sha256": digest(trained / "model.safetensors"),
                            "device_uuid": bound.selection.uuid,"reload":read_json(bound.output / "checkpoint_reload.json")}
                trained = phase("T3_short_training", [PhaseJob(f"train_{i:02d}", "training", train,
                    estimate_seconds=300, reserve_seconds=0,
                    resources=dataclasses.replace(resources, per_gpu_memory_mib=12000))
                                                       for i in range(len(devices))])
                atomic_json(root / "training_results.json", trained)
                for d in devices:
                    d.setdefault("capabilities", {})["train"] = "verified"
            elif args.skip_training:
                summary["tests"]["T3_short_training"] = {"status": "not_run"}
            if not args.skip_native:
                if not args.continue_from:
                    barrier = threading.Barrier(len(devices), timeout=120)
                    def native(bound):
                        barrier.wait()
                        bound.native_barrier = {"directory":str(root / "native_ready_barrier"),"count":len(devices)}
                        data = bound.evaluate(spec, checkpoint, bound.output / "native", episodes=1,
                                              master_seed=2000000201, purpose="smoke",
                                              startup_attempts=1, smoke_timeout=900)
                        return {"device_uuid": bound.selection.uuid, "episodes": data["episodes"],
                                "census": startup_census(bound.output / "native"),
                                "process": read_json(bound.output / "native/process/process.json")}
                    results = phase("T1_native_concurrency", [PhaseJob(f"native_{i:02d}", "evaluation", native,
                                 estimate_seconds=750, reserve_seconds=0, resources=resources)
                                                              for i in range(len(devices))])
                    if len({row["device_uuid"] for row in results.values()}) != len(devices):
                        raise RuntimeError("native workers did not cover the device allocation")
                    intervals = []
                    for row in results.values():
                        if len(row["episodes"]) != 1:
                            raise RuntimeError("native episode result missing")
                        process = row["process"]
                        from datetime import datetime
                        ready = [c for c in row["census"] if c["phase"] == "evaluation_reset_started"]
                        if not ready:
                            raise RuntimeError("native worker did not reach a real episode reset")
                        aligned = ready[0].get("default_device", {})
                        target = next(d for d in devices if d["uuid"] == row["device_uuid"])
                        if aligned.get("torch_index") != target["index"]:
                            raise RuntimeError("native default CUDA device differs from leased device")
                        own = (ready[0].get("memory") or {}).get("own", {})
                        if own.get(row["device_uuid"], 0) < 1024:
                            raise RuntimeError("native worker's own process has no substantial target-device allocation")
                        begin = datetime.fromisoformat(ready[0]["time"]).timestamp()
                        end = datetime.fromisoformat(process["finished_at"]).timestamp()
                        intervals.append((begin, end))
                    overlap = min(b for _, b in intervals) - max(a for a, _ in intervals)
                    if overlap <= 0:
                        raise RuntimeError("no shared native-worker execution interval")
                    summary["tests"]["T1_native_concurrency"].update(
                        verified_worker_count=len(results), episode_execution_overlap_seconds=overlap,
                        note="post-ready barrier; real episode progress, target-device own PID memory and CUDA alignment checked")
                    for d in devices:
                        d["capabilities"].update(physics="verified", render="verified", policy_inference="verified")
                    # Fixed bank workload: one scheduler leaf per block, bank identities independent of placement.
                    jobs = []
                    from autosim.robosyn_data import evaluation_seed_bank
                    bank = evaluation_seed_bank(2000000211, args.native_episodes)
                    atomic_json(root / "fixed_native_bank.json", bank)
                    count = min(len(devices), len(bank))
                    # Existing native shard worker receives the full master bank and exact offsets.
                    from autosim.research.evaluation_merge import build_shard_plan, merge_evaluation_shards, shard_dir
                    selections = [select(d, "identity") for d in devices[:count]]
                    bank_plan = build_shard_plan(episodes=len(bank), bank=bank,
                                                selections=[s.as_dict() for s in selections])
                    eval_root = root / "fixed_native"
                    atomic_json(eval_root / "shard_plan.json", bank_plan)
                    for i, block in enumerate(bank_plan["blocks"]):
                        def shard(bound, i=i, block=block):
                            return bound._run_shard(index=i, spec=spec, checkpoint=checkpoint,
                                selection=bound.selection, shard_root=shard_dir(eval_root, i), block=block,
                                count=count, episodes=len(bank), master_seed=2000000211,
                                purpose="development", policy="act", startup_attempts=1, smoke_timeout=900)
                        jobs.append(PhaseJob(f"eval_block_{i:02d}", "evaluation", shard,
                                              estimate_seconds=750, reserve_seconds=0, resources=resources))
                    phase("T2_native_fixed_bank", jobs)
                    merged = merge_evaluation_shards(output=eval_root, purpose="development",
                        master_seed=2000000211, episodes=len(bank), bank=bank,
                        merge_command=["compute-validation", "fixed-native-bank"])
                    atomic_json(root / "fixed_native_result.json", merged)
                try:
                    if not summary["tests"].get("T4_collection",{}).get("reused_without_execution"):
                        collection_masters = [310912001 if i == 0 else 2200000+i for i in range(len(devices))]
                        atomic_json(root / "capacity_collection_contract.json", {
                            "master_seeds":collection_masters,"attempt_budget_per_worker":1,
                            "purpose":"capacity and artifact integration, not fixed-workload speed or success-rate measurement",
                            "first_worker":"historical original-distribution regression fixture, episode seed 223407107",
                            "previous_matrix_attempts_remain_unchanged":True})
                        collected = phase("T4_collection", [PhaseJob(f"collect_{i:02d}", "collection",
                        lambda bound, i=i: bound.collect_bounded(spec, bound.output / "collection",
                            attempt_budget=1, target_episodes=1, master_seed=collection_masters[i],
                            profile="full_random", timeout=900),
                        estimate_seconds=750, reserve_seconds=0, resources=resources) for i in range(len(devices))])
                except RuntimeError:
                    # Retain the failed component; independent ACT and API checks still
                    # provide useful evidence within this already allocated budget.
                    pass
                atomic_json(root / "collection_results.json", collected)
            else:
                summary["tests"]["T1_native_concurrency"] = {"status": "not_run"}
            if not args.skip_training:
                if not args.skip_native and len(devices) in (1, 2, 4):
                    useful = next((row for _, row in sorted(collected.items())
                                   if row.get("accepted_episodes", 0)>0), None)
                    if useful is None:
                        # A separate, predeclared integration fixture. Original failed attempts
                        # remain in the capacity bank and its denominator is never changed.
                        atomic_json(root / "short_chain/fixture_contract.json", {
                            "master_seed":310912001,"attempt_budget":1,"target_episodes":1,
                            "source":"historical original-distribution collector regression fixture",
                            "known_successful_episode_seed":223407107,"performance_or_success_rate_claim":False})
                        chain_collection = phase("T4_chain_collection", [PhaseJob("fixture", "collection",
                            lambda bound: bound.collect_bounded(spec, root / "short_chain/collection",
                                attempt_budget=1,target_episodes=1,master_seed=310912001,
                                profile="full_random",timeout=900),
                            estimate_seconds=600,reserve_seconds=0,resources=resources)])
                        useful = next((row for row in chain_collection.values() if row.get("accepted_episodes",0)>0), None)
                    if useful is None:
                        summary["tests"]["T4_short_chain"] = {"status":"not_run","reason":"bounded fixture also yielded zero usable episodes"}
                    else:
                        chain_data = Path(useful["dataset_root"])
                        phase("T4_prepare", [PhaseJob("prepare", "audit",
                            lambda bound: bound.prepare_data(spec, chain_data, root / "short_chain/audit"),
                            device_count=0, resources=ResourceRequest(cpu_cores=2, ram_mib=4096, io_slots=1),
                            estimate_seconds=60, reserve_seconds=0)])
                        weights = phase("T4_train", [PhaseJob("train", "training",
                            lambda bound: bound.train(spec, chain_data, root / "short_chain/train", steps=100,
                                params={"batch_size":4,"num_workers":1},seed=3314,pretrained=checkpoint),
                            estimate_seconds=180,reserve_seconds=0,resources=resources)])["train"]
                        evaluated = phase("T4_evaluate", [PhaseJob("evaluate", "evaluation",
                            lambda bound: bound.evaluate(spec, weights, root / "short_chain/evaluation",episodes=1,
                                master_seed=2000000241,purpose="smoke",startup_attempts=1,smoke_timeout=900),
                            estimate_seconds=600,reserve_seconds=0,resources=resources)])
                        summary["tests"]["T4_short_chain"]={"status":"passed","checkpoint":str(weights),
                            "source_data":str(chain_data),"evaluation_episodes":1}
        if args.api_evidence:
            evidence = read_json(args.api_evidence / "codegen_result.json")
            decision = read_json(args.api_evidence / "planner/planner_state.json")
            summary["tests"]["T6_api"] = {"status":"passed" if decision["mode"] == "api" and evidence["status"] == "validated" else "failed",
                "planning_mode":decision["mode"],"codegen":evidence,"live_api_location":"connected control host",
                "activation":"not promoted globally: whole-run payback has not been established"}
        control_response = root / "compute_control_response.json"
        if control_response.exists():
            response = read_json(control_response)
            if response["request_digest"] != object_digest(control_snapshot):
                raise ValueError("compute response does not match this allocation snapshot")
            decision = response["decision"]
            if decision["mode"] != "api":
                raise ValueError("compute response did not pass API planning validation")
            params = decision["parameters"]
            limit = int(params["max_parallel_jobs"]); cores = int(params["omp_threads"])
            if not (1 <= limit <= len(devices) and 1 <= cores and cores*limit <= host["cpu_cores"]):
                raise ValueError("API plan exceeds actual allocation")
            plan["max_parallel_jobs"] = limit
            activated = phase("T6_api_execution", [PhaseJob(f"api_cuda_{i}","probe",fixed_cuda,
                estimate_seconds=30,reserve_seconds=0,
                resources=ResourceRequest(cpu_cores=cores,ram_mib=2048,per_gpu_memory_mib=2048))
                for i in range(16)])
            if any(row.get("omp_threads") != cores for row in activated.values()):
                raise RuntimeError("API thread parameter was not applied to the actual GPU worker")
            summary["tests"]["T6_api_execution"].update(applied={"max_parallel_jobs":limit,"omp_threads":cores},
                request_digest=response["request_digest"],scope="bounded GPU execution parameters; scientific settings unchanged")
        required = all(row.get("status", "passed" if row.get("passed") else "failed") == "passed"
                       for row in summary["tests"].values() if not row.get("optional_capability"))
        summary["status"] = "passed" if required and not (args.skip_native or args.skip_training) else "partial"
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:1000])
    finally:
        ledger.end_allocation()
        summary.update(finished=time.time(), elapsed_seconds=time.time()-started,
                       allocated_gpu_hours=ledger.allocated_gpu_hours)
        atomic_json(root / "telemetry.json", sampler.summary())
        atomic_json(root / "capabilities.json", devices)
        atomic_json(root / "summary.json", summary)
        print(f"validation {summary['status']}: {root / 'summary.json'}", flush=True)
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
