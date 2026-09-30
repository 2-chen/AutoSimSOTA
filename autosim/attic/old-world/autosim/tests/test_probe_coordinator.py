"""Real child processes exercise IPC, failure propagation and process-group reaping."""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from autosim.research.common import run_command, read_json
from autosim.research.compute_probe import retryable_generation
from autosim.research.probe_coordinator import ProbeCoordinator, group_alive


WORKER = r'''
import json, os, resource, signal, subprocess, sys, time
from pathlib import Path
from autosim.research.native_lifecycle import NativeLifecycle, PeerCancelled
from autosim.research.common import atomic_json
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
root, mode, delay = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
life = NativeLifecycle(root)
try:
    life.start()
    atomic_json(root / 'startup.json', {'phase':'environment_constructing'})
    life.constructing()
    time.sleep(delay)
    if mode == 'crash':
        if life.config.get('inject_before_ready'):
            atomic_json(root / 'injected_failure.json', {**life.identity, 'kind':'SIGSEGV',
                        'phase':'constructing','reset_started':False})
        os.kill(os.getpid(), signal.SIGSEGV)
    if mode == 'hang':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(60)
    life.ready()
    life.before_reset()
    if mode == 'post_reset_crash': os.kill(os.getpid(), signal.SIGSEGV)
    finish = time.monotonic() + .45
    value = 0
    while time.monotonic() < finish:
        life.step_started()
        value += sum(i*i for i in range(10000))
        life.step_completed()
    life.work_finished()
    if mode == 'grandchild':
        child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])
        atomic_json(root / 'grandchild.json', {'pid':child.pid})
    atomic_json(root / 'business.json', {'value':value, 'steps':life.steps})
    life.completed()
except BaseException as exc:
    atomic_json(root / 'failure.json', {'type':type(exc).__name__,'monotonic':time.monotonic()})
    life.failed(exc)
    raise
'''


def launch(coordinator, root, name, mode="ok", delay=0, overrides=None):
    config = {**coordinator.worker_configuration(name), **(overrides or {})}
    env = {**os.environ, "AUTOSIM_NATIVE_COORDINATOR": json.dumps(config)}
    directory = root / name
    try:
        process = run_command([sys.executable, "-c", WORKER, str(directory), mode, str(delay)],
                              cwd=root, env=env, output=directory / "process", timeout=12,
                              metadata={"job": name})
        status = "completed"
    except RuntimeError:
        status = "failed"
    return {"status": status, "artifact_directory": str(directory)}


def wave(root, modes, *, timeout=5, construction=2, overrides=None):
    names = [f"native_{i}" for i in range(len(modes))]
    root.mkdir(parents=True, exist_ok=True)
    with ProbeCoordinator(root, names, timeout_seconds=timeout, construction_concurrency=construction,
                          minimum_overlap_seconds=.1, termination_grace_seconds=.3) as coordinator:
        for i, name in enumerate(names):
            coordinator.bind(name, f"GPU-{i}")
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            futures = {name: pool.submit(launch, coordinator, root, name, mode[0], mode[1],
                                         (overrides or {}).get(name)) for name, mode in zip(names, modes)}
            results = {name: task.result() for name, task in futures.items()}
        receipt = coordinator.overlap_receipt()
    assert read_json(root / "cleanup.json")["confirmed_reaped"]
    return results, receipt


def test_four_actual_workers_share_a_rollout_window_after_serial_construction(tmp_path):
    results, receipt = wave(tmp_path / "wave", [("ok", .08)] * 4, construction=1)
    assert all(row["status"] == "completed" for row in results.values())
    assert receipt["passed"] and receipt["verified_rollout_concurrency"] == 4
    assert receipt["construction_concurrency"] == 1 and receipt["overlap_seconds"] >= .1
    assert len({row["uuid"] for row in receipt["workers"]}) == 4
    for row in results.values():
        assert read_json(Path(row["artifact_directory"]) / "business.json")["value"] > 0


def test_primary_sigsegv_cancels_peers_and_new_generation_does_real_work(tmp_path):
    start = time.monotonic()
    results, failed = wave(tmp_path / "first", [("crash", .25), ("ok", .05)])
    assert time.monotonic() - start < 5
    assert not failed["passed"] and failed["primary_failure"]["worker"] == "native_0"
    assert retryable_generation(results, failed["primary_failure"], tmp_path / "first")
    assert not any((Path(row["artifact_directory"]) / "reset_started.json").exists() for row in results.values())
    _, recovered = wave(tmp_path / "second", [("ok", .02)] * 2)
    assert recovered["passed"] and recovered["generation"] != failed["generation"]


def test_slow_live_constructor_is_allowed_without_false_peer_failure(tmp_path):
    _, receipt = wave(tmp_path / "slow", [("ok", .65), ("ok", .01)])
    assert receipt["passed"] and receipt["primary_failure"] is None


def test_native_hang_ignoring_term_is_killed_and_resources_reaped(tmp_path):
    start = time.monotonic()
    _, receipt = wave(tmp_path / "hang", [("hang", 0), ("ok", 0)], timeout=.8)
    assert time.monotonic() - start < 5
    assert not receipt["passed"] and receipt["primary_failure"]["reason"] == "native_startup_timeout"
    assert all(not group_alive(row["pgid"]) for row in receipt["workers"])


def test_reset_started_prevents_retry_after_native_crash(tmp_path):
    results, receipt = wave(tmp_path / "scored", [("post_reset_crash", 0), ("ok", 0)])
    assert not receipt["passed"]
    assert not retryable_generation(results, receipt["primary_failure"], tmp_path / "scored")


@pytest.mark.parametrize("bad", [{"generation":"stale"}, {"claim":"old-claim"},
                                  {"uuid":"GPU-unallocated"}, {"attempt_id":"99"}])
def test_stale_or_wrong_membership_cannot_release_barrier(tmp_path, bad):
    _, receipt = wave(tmp_path / "wrong", [("ok", 0), ("ok", 0)], timeout=.5,
                      overrides={"native_0": bad})
    assert not receipt["passed"]
    assert not (tmp_path / "wrong/native_1/reset_started.json").exists()


def test_ready_files_from_previous_implementation_have_no_authority(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "native_0.json").write_text('{"pid":123}')
    (root / "native_1.json").write_text('{"pid":456}')
    _, receipt = wave(root, [("hang", 0), ("ok", 0)], timeout=.5)
    assert not receipt["passed"]
    assert not (root / "native_1/reset_started.json").exists()


def test_guard_reaps_grandchild_after_successful_group_leader_exit(tmp_path):
    results, receipt = wave(tmp_path / "descendant", [("grandchild", 0)], construction=1)
    assert receipt["passed"]
    pid = read_json(tmp_path / "descendant/native_0/grandchild.json")["pid"]
    from autosim.research.native_lifecycle import process_identity
    assert process_identity(pid) is None


def test_duplicate_binding_is_rejected_before_start(tmp_path):
    with ProbeCoordinator(tmp_path, ["a", "b"], timeout_seconds=1) as coordinator:
        coordinator.bind("a", "GPU-0")
        with pytest.raises(ValueError, match="duplicate"):
            coordinator.bind("b", "GPU-0")


@pytest.mark.parametrize("persistent_failure", [False, True])
def test_admission_dispatcher_automatically_recovers_or_exhausts_shared_attempt_budget(
        tmp_path, monkeypatch, persistent_failure):
    """GPU discovery/work is a CPU fixture; dispatcher, lifecycle and recovery are real."""
    from types import SimpleNamespace
    from autosim.research import compute_probe
    from autosim.research.runtime import Runtime
    from autosim.research.common import atomic_json

    repo = tmp_path / "RoboSynChallenge"
    checkpoint = repo / "checkpoints/ACT_sim_cpu_fixture"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"cpu fixture, no GPU certification")
    devices = [{"uuid": f"GPU-{i}", "index": i, "class_name": "cpu-fixture", "model": "cpu-fixture",
                "memory_total_mib": 32000, "memory_used_mib": 0} for i in range(2)]
    monkeypatch.setattr(compute_probe, "discover", lambda **kw: {
        "host": os.uname().nodename, "gpus": devices, "allowed": [0, 1]})
    monkeypatch.setattr(compute_probe, "host_inventory", lambda *a: {
        "host": os.uname().nodename, "cpu_cores": 4, "ram_mib": 16000,
        "shm_mib": 1000, "scratch_mib": 16000, "io_slots": 2})
    monkeypatch.setattr(compute_probe, "load_task", lambda *a: SimpleNamespace(name="cpu_fixture"))
    monkeypatch.setattr(compute_probe, "judge", lambda result, **kw: {"passed": result["status"] == "completed"})
    published = []
    monkeypatch.setattr(compute_probe, "write_probe_receipt", lambda *a: published.append(a[-1]))

    class CPURuntime(Runtime):
        @property
        def python(self):
            return Path(sys.executable)
        def run(self, command, output, timeout, **kw):
            # Only CUDA's arithmetic smoke is replaced; the actual workers below
            # execute in separate guarded processes through the production IPC.
            assert "autosim.research.validation_worker" in command
            atomic_json(output / "cpu_cuda_fixture.json", {"simulated_cuda_only": True})
            return {"status": "completed"}

    monkeypatch.setattr(compute_probe, "Runtime", CPURuntime)

    def cpu_arm(*, name, runtime, output, selection, **kwargs):
        output.mkdir(parents=True, exist_ok=True)
        fault = runtime.native_barrier.get("inject_before_ready") or (persistent_failure and name == "native_0")
        mode = "crash" if fault else "ok"
        try:
            run_command([sys.executable, "-c", WORKER, str(output), mode, ".1"], cwd=repo,
                        env={**os.environ, "AUTOSIM_NATIVE_COORDINATOR": json.dumps(runtime.native_barrier)},
                        output=output / "process", timeout=10, metadata={"job": name})
            status = "completed"
        except RuntimeError:
            status = "failed"
        return {"name": name, "status": status, "selection": selection.as_dict(),
                "artifact_directory": str(output), "startup_census": [],
                "execution_mode": "synthetic_cpu_fixture"}

    monkeypatch.setattr(compute_probe, "run_arm", cpu_arm)
    config = tmp_path / "harness.json"
    atomic_json(config, {"validation": {"stage": "native_admission", "inject_failure_worker": 0,
                                       "inject_failure_generation": 1}})
    output = tmp_path / "admission"
    monkeypatch.setattr(sys, "argv", ["compute_probe", "--workspace", str(tmp_path), "--output", str(output),
        "--task", "cpu_fixture", "--harness-config", str(config), "--incident-timeout", "120",
        "--smoke-timeout", "30", "--minimum-overlap-seconds", ".1"])
    assert compute_probe.main() == (2 if persistent_failure else 0)
    receipt = read_json(output / "device_probe.json")
    assert len(receipt["startup_attempts"]) == (3 if persistent_failure else 2)
    assert receipt["startup_attempts"][0]["retryable"]
    assert len({a["generation"] for a in receipt["startup_attempts"]}) == len(receipt["startup_attempts"])
    assert receipt["passed"] is (not persistent_failure)
    from autosim.research.harness_acceptance import native_receipt_verdict
    verdict = native_receipt_verdict(receipt, 2)
    assert verdict["passed"] is (not persistent_failure), verdict["failures"]
    assert all(row["host"] == os.uname().nodename for row in receipt["concurrency_evidence"]["workers"])
    from autosim.research.common import digest
    observed_inventory = read_json(output / "inventory.json")
    assert receipt["inventory_evidence"]["sha256"] == digest(output / "inventory.json")
    assert receipt["requested_devices"] == [row["uuid"] for row in observed_inventory["gpus"]
                                             if row["index"] in observed_inventory["allowed"]]
    assert receipt["validation_injection"]["applied"]
    assert len(receipt["validation_injection"]["evidence"]) == 1
    assert len(published) == (0 if persistent_failure else 1)
    assert len(read_json(output / "budget.json")["jobs"]) == 2 * len(receipt["startup_attempts"])
    if persistent_failure:
        # Re-entering the same output cannot obtain three more free attempts.
        assert compute_probe.main() == 2
        assert len(read_json(output / "device_probe.json")["startup_attempts"]) == 3
    else:
        warm_output = tmp_path / "warm_admission"
        arguments = list(sys.argv)
        arguments[arguments.index("--output") + 1] = str(warm_output)
        arguments += ["--probe-repetition", "1", "--cache-mode", "warm",
                      "--cache-namespace", receipt["native_cache_namespace"]]
        monkeypatch.setattr(sys, "argv", arguments)
        assert compute_probe.main() == 0
        warm = read_json(warm_output / "device_probe.json")
        assert native_receipt_verdict(warm, 2)["passed"]
        assert warm["cache_mode"] == "warm" and warm["native_cache_namespace"] == receipt["native_cache_namespace"]
        assert len(warm["startup_attempts"]) == 1
        assert warm["validation_injection"]["requested_worker"] == 0
        assert not warm["validation_injection"]["applied"]
