"""Bounded single-GPU integration queue, gated on CPU acceptance and frozen code.

Does not claim formal comparison completion. Unsupported adapters stop at their
declared capability boundary. One unavailable backend cannot hide other results.
"""
import argparse
import os
import time
from pathlib import Path

from autosim.research.common import assert_frozen, atomic_json, digest, immutable_json, now, read_json
from .executor import Executor, Job, lease


def system_sources(workspace, output):
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [workspace / "autosim/tests/test_experiment_system.py",
              workspace / "AutoSimSOTA/RoboSynChallenge/policy/dp/scripts/train.py"]
    paths += list((output / "runtime_env").glob("pyvenv.cfg"))
    paths += list((output / "runtime_env").glob("lib/python3.10/site-packages/*.pth"))
    return {str(p.absolute()): digest(p) for p in paths}


def native_jobs(workspace, output, executor, sources, environment):
    from .plugins import ROBOTWIN_TASKS, RoboTwinPlugin
    native = output / "native/RoboTwin"
    textures = output / "texture_state.json"
    if not textures.exists() or read_json(textures).get("status") != "completed":
        return []
    interpreter = output / "runtime_env/bin/python"
    env = dict(environment, PYTHONPATH=os.pathsep.join([str(workspace / "autosim"),
               str(workspace / "AutoSimSOTA/RoboSynChallenge"), str(native)]))
    env["TORCH_EXTENSIONS_DIR"] = str(output / "torch_extensions")
    env["XDG_CACHE_HOME"] = str(output / "runtime_cache")
    base = [str(interpreter), "-m", "autosim.experiment_system.robotwin_worker"]
    shared = ["--repo", str(native), "--attempt", "{attempt}"]
    jobs = []
    for index, task in enumerate(ROBOTWIN_TASKS):
        contract = RoboTwinPlugin(native).contract(task)
        inputs = {**sources, **contract.sources, str(textures): digest(textures)}
        inputs.update({str(p): digest(p) for p in (output / "runtime_env").glob("lib/python3.10/site-packages/curobo/curobolib/*.so")})
        def job(name, stage, command, dependency=(), gpu="0", timeout=1800):
            return Job(name, stage, command, str(native), {"result.json": "result"}, dict(inputs),
                       dependencies=dependency, environment=env, timeout_seconds=timeout,
                       budget_seconds=timeout, gpu=gpu)
        prefix = f"robotwin_{task}"
        common = [*shared, "--task", task]
        jobs.append(job(prefix + "_probe", "probe", [*base, "probe", *common, "--seed", str(92006000 + index * 10000)], timeout=600))
        if not get_result(executor, prefix + "_probe"):
            continue
        jobs.append(job(prefix + "_collect", "collect", [*base, "collect", *common, "--episodes", "3",
                        "--seed", str(92006100 + index * 10000)], (prefix + "_probe",)))
        collected = executor.committed(prefix + "_collect")
        if not collected:
            continue
        collection_result = executor.root / (prefix + "_collect") / collected["attempt"] / "result.json"
        jobs.append(job(prefix + "_convert", "admit", [*base, "convert", *common,
                         "--collection-result", str(collection_result)], (prefix + "_collect",), gpu=None))
        converted = get_result(executor, prefix + "_convert")
        if not converted:
            continue
        dataset = converted["dataset"]
        backend = [str(workspace / "AutoSimSOTA/.venv/bin/python"), "-m", "autosim.experiment_system.backend_worker"]
        trainer_common = ["--workspace", str(workspace), "--benchmark", "robotwin", "--native-repo", str(native),
                          "--task", task, "--attempt", "{attempt}", "--dataset", dataset]
        jobs.append(job(prefix + "_admit", "admit", [*backend, "admit", *trainer_common], (prefix + "_convert",), gpu=None))
        admitted = get_result(executor, prefix + "_admit")
        if not admitted:
            continue
        inputs.update(admitted["verified_files"])
        for policy in ("act", "dp"):
            training = prefix + f"_{policy}_train"
            jobs.append(job(training, "train", [*backend, "train", *trainer_common, "--policy", policy,
                            "--steps", "200", "--seed", "91006100"], (prefix + "_admit",), timeout=2400))
            trained = get_result(executor, training)
            if not trained:
                continue
            jobs.append(job(prefix + f"_{policy}_evaluate", "evaluate", [*base, "evaluate", *common,
                            "--checkpoint", trained["checkpoint"], "--policy", policy, "--episodes", "3",
                            "--seed", str(94006000 + index * 10000)], (training,)))
    return jobs


def get_result(executor, name):
    record = executor.committed(name)
    return read_json(executor.root / name / record["attempt"] / "result.json") if record else None


def build_jobs(workspace, output, executor, sources):
    from autosim.research.runtime import Runtime
    runtime = Runtime(workspace, output)
    environment = {k: v for k, v in runtime.environment().items()
                   if k in {"PYTHONPATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "EMBODICHAIN_DATA_ROOT",
                            "XDG_CACHE_HOME", "MPLCONFIGDIR", "OMP_NUM_THREADS", "PYTHONUNBUFFERED", "PYTHONFAULTHANDLER"}}
    environment["HF_DATASETS_CACHE"] = str(output / "dataset_cache")
    base = [str(runtime.python), "-m", "autosim.experiment_system.backend_worker"]
    shared = ["--workspace", str(workspace), "--attempt", "{attempt}"]
    native = output / "native/RoboTwin"
    render_env = dict(environment, PYTHONPATH=str(workspace / "autosim"))
    jobs = [Job("robotwin_render", "probe", [str(output / "runtime_env/bin/python"),
                "-m", "autosim.experiment_system.backend_worker", "robotwin-render", *shared],
                str(native), {"result.json": "result"}, sources, environment=render_env,
                timeout_seconds=180, max_attempts=1, budget_seconds=180, gpu="0")]
    jobs.extend(native_jobs(workspace, output, executor, sources, environment))
    jobs.append(Job("robosyn_click_collect", "collect", [*base, "collect", *shared, "--episodes", "3"],
                    str(runtime.repo), {"result.json": "result"}, sources, environment=environment,
                    timeout_seconds=1800, budget_seconds=1800, gpu="0"))
    collected = get_result(executor, "robosyn_click_collect")
    if not collected:
        return jobs
    dataset = collected["dataset"]
    jobs.append(Job("robosyn_click_admit", "admit", [*base, "admit", *shared, "--dataset", dataset],
                    str(runtime.repo), {"result.json": "result"}, sources,
                    dependencies=("robosyn_click_collect",), environment=environment,
                    timeout_seconds=600, budget_seconds=600))
    admitted = get_result(executor, "robosyn_click_admit")
    if not admitted:
        return jobs
    train_sources = {**sources, **admitted["verified_files"]}
    for policy in ("act", "dp"):
        train = f"robosyn_click_{policy}_train"
        jobs.append(Job(train, "train", [*base, "train", *shared, "--dataset", dataset,
                                         "--policy", policy, "--steps", "200", "--seed", "91006100"],
                        str(runtime.repo), {"result.json": "result"}, train_sources,
                        dependencies=("robosyn_click_admit",), environment=environment,
                        timeout_seconds=2400, budget_seconds=2400, gpu="0"))
        trained = get_result(executor, train)
        if trained:
            jobs.append(Job(f"robosyn_click_{policy}_evaluate", "evaluate",
                            [*base, "evaluate", *shared, "--checkpoint", trained["checkpoint"],
                             "--policy", policy, "--episodes", "3", "--seed", "91006200"],
                            str(runtime.eval_repo), {"result.json": "result"}, {**train_sources, **trained["verified_files"]},
                            dependencies=(train,), environment=environment,
                            timeout_seconds=1800, budget_seconds=1800, gpu="0"))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-hours", type=float, default=48)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.absolute()
    output = (args.output or workspace / "autosim/output/experiment_system_20260906").absolute()
    if not 0 < args.max_hours <= 72:
        raise ValueError("queue wall-clock budget must be in (0, 72] hours")
    if not read_json(output / "cpu_acceptance.json")["passed"]:
        raise ValueError("CPU acceptance gate failed")
    sources = system_sources(workspace, output)
    immutable_json(output / "frozen_system.json", sources)
    previous = read_json(output / "protected_previous_core.json")
    sources = {**sources, **previous}
    executor = Executor(output / "jobs")
    state = {"started_at": now(), "status": "running", "formal_experiments_complete": False}
    start = time.monotonic()
    with lease(output / "queue.lock"):
        while time.monotonic() - start < args.max_hours * 3600:
            assert_frozen(sources)
            assert_frozen(previous)
            statuses = {}
            for job in build_jobs(workspace, output, executor, sources):
                statuses[job.name] = executor.run(job)
                state.update(jobs=statuses, updated_at=now())
                atomic_json(output / "queue_state.json", state)
            from .plugins import ROBOTWIN_TASKS
            wanted = {"robotwin_render", "robosyn_click_act_evaluate", "robosyn_click_dp_evaluate"}
            wanted.update(f"robotwin_{task}_{policy}_evaluate" for task in ROBOTWIN_TASKS for policy in ("act", "dp"))
            if all(statuses.get(name, {}).get("status") == "committed" for name in wanted):
                state.update(status="integration_gate_reached", remaining=["semantic_native_parity", "system_comparisons",
                    "independent_operator_measurement", "formal_policy_experiments"])
                break
            terminal = {"requires_adapter_or_data_fix", "requires_artifact_audit", "requires_audit_unknown_execution",
                        "quarantine_evaluation", "budget_exhausted"}
            if statuses and all(s["status"] == "committed" or s["status"] in terminal for s in statuses.values()):
                # build_jobs may expose a new downstream stage next iteration.
                next_names = {j.name for j in build_jobs(workspace, output, executor, sources)}
                texture_state = output / "texture_state.json"
                textures_ready = texture_state.exists() and read_json(texture_state).get("status") == "completed"
                if next_names == set(statuses) and textures_ready:
                    state.update(status="requires_capability_review")
                    break
            if args.once:
                state.update(status="one_pass_finished")
                break
            time.sleep(20)
        else:
            state.update(status="queue_wait_budget_exhausted")
        state["updated_at"] = now()
        atomic_json(output / "queue_state.json", state)


if __name__ == "__main__":
    main()
