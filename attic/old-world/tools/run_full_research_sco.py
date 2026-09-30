#!/usr/bin/env python3
"""One bounded SCO allocation running the actual repository research CLI."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from autosim.llm_client import LLMClient
from autosim.research.common import atomic_json, digest, now, read_json
from autosim.research.devices import discover
from autosim.research.repository_autoresearch import (
    MilestoneConfig, RepositoryAutoResearch, load_deepseek_environment,
)


def main():
    root = Path(sys.argv[1]).absolute()
    manifest = read_json(root / "launch_manifest.json")
    expected_gpus = int(manifest.get("expected_gpus", 4))
    if expected_gpus not in {1, 2, 4, 8}:
        raise ValueError("unsupported single-node allocation size")
    os.environ["SCO_WORKER_SPEC"] = f"n12lp.nn.i10a.{expected_gpus}"
    os.environ["AUTOSIM_LAUNCH_MANIFEST"] = str(root / "launch_manifest.json")
    source = root / "source"
    run_root = Path(manifest["run_root"])
    os.environ["AUTOSIM_RUN_ROOT"] = str(run_root)
    started, monotonic_start = now(), time.monotonic()
    # Reserve 60 seconds for process cleanup. Platform setup/release is reconciled
    # from SCO timestamps; this clock is explicitly a container execution proxy.
    deadline = monotonic_start + manifest["max_container_seconds"] - 60
    if manifest.get("run_deadline_utc"):
        remaining = datetime.fromisoformat(manifest["run_deadline_utc"]).timestamp() - time.time()
        deadline = min(deadline, monotonic_start + remaining - 60)
    previous_gpu_hours = float(manifest.get("previous_allocated_gpu_hours", 0))
    child = None
    stopped = False
    phase = "preparing_environment"
    code = 1

    def heartbeat(**extra):
        elapsed = time.monotonic() - monotonic_start
        state_file = run_root / "run_state.json"
        state = read_json(state_file) if state_file.exists() else {}
        atomic_json(root / "allocation_heartbeat.json", {
            "started_at": started, "updated_at": now(), "host": os.uname().nodename,
            "phase": phase, "research_status": state.get("status"),
            "research_stage": state.get("stage"), "research_updated_at": state.get("updated_at"),
            "container_elapsed_seconds": elapsed, "allocated_gpu_hours_proxy": elapsed * expected_gpus / 3600,
            "previous_allocated_gpu_hours": previous_gpu_hours,
            "cumulative_allocated_gpu_hours_proxy": previous_gpu_hours + elapsed * expected_gpus / 3600,
            "allocated_gpus": expected_gpus,
            "run_deadline_utc": manifest.get("run_deadline_utc"),
            "accounting_basis": "container entrypoint; reconcile SCO setup/release separately",
            "max_container_seconds": manifest["max_container_seconds"], **extra,
        })

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        journal = root / "continuation/continuation.json"
        if journal.exists():
            attempts = read_json(journal).get("attempts", [])
            if attempts and attempts[-1].get("status") == "running":
                from autosim.research.continuation import process_gone
                identity = attempts[-1].get("process")
                if identity and not process_gone(identity):
                    try:
                        os.killpg(identity["pid"], signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        heartbeat()
        if deadline - time.monotonic() < 120:
            raise TimeoutError("remaining original run budget cannot afford startup")
        for relative, expected in read_json(root / "source_manifest.json").items():
            if digest(source / relative) != expected:
                raise RuntimeError(f"frozen source changed: {relative}")
        if manifest.get("harness_config_sha256") and digest(root / "harness_config.json") != manifest["harness_config_sha256"]:
            raise RuntimeError("frozen harness configuration changed")
        dependency = Path(os.environ["TORCH_HOME"]) / "hub/checkpoints/resnet18-f37072fd.pth"
        if digest(dependency) != "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec":
            raise RuntimeError("offline backbone checksum mismatch")
        subprocess.run(["bash", "/data/AutoResearch/AutoSimSOTA/install_optix_runtime.sh"],
                       check=True, timeout=120)
        optix = Path("/tmp/optix-ld-extra")
        if optix.is_file() and optix.read_text().strip():
            os.environ["LD_LIBRARY_PATH"] = optix.read_text().strip() + ":" + os.environ["LD_LIBRARY_PATH"]
        inventory = discover(disk_path=root)
        atomic_json(root / "allocation_inventory.json", inventory)
        allocated = [d for d in inventory["gpus"] if d["index"] in inventory["allowed"]]
        if len(allocated) != expected_gpus or any(manifest.get("expected_model", "5090") not in (d.get("model") or "") for d in allocated):
            raise RuntimeError("allocation differs from the frozen requested GPU count/model")
        phase = "api_preflight"
        heartbeat()
        config = load_deepseek_environment()
        atomic_json(run_root / "llm_configuration.json", config)
        research = RepositoryAutoResearch(MilestoneConfig(
            repo_input=str(source / "RoboSynChallenge"), output_root=root / "runs",
            run_id=manifest["run_id"], task="click_bell"))
        research.api_preflight(LLMClient())
        phase = "research_cli_running"
        heartbeat()
        if manifest.get("schema_version", 1) >= 2:
            from autosim.research.continuation_runner import run_bounded_research
            from autosim.research.harness_acceptance import validate_run
            from autosim.research.harness_config import load_harness_policy
            policy = load_harness_policy(root / "harness_config.json")
            progress = []
            stop_heartbeat = threading.Event()
            def heartbeat_loop():
                while not stop_heartbeat.wait(10):
                    # A mutable sequence gives the supervisor the new commits at exit.
                    progress[:] = list((run_root / "harness/artifact_commits").glob("*.json"))
                    heartbeat()
            thread = threading.Thread(target=heartbeat_loop, daemon=True)
            thread.start()
            try:
                result = run_bounded_research(root, manifest=manifest, policy=policy,
                    deadline_epoch=time.time() + max(0, deadline-time.monotonic()),
                    env=dict(os.environ), progress_receipts=progress,
                    budget_check=lambda reserve: not stopped and time.monotonic() + reserve < deadline)
                code = result["returncode"]
                phase = "cli_exited"
                if code == 0:
                    verdict = validate_run(run_root, expected_gpus=expected_gpus,
                        stage=manifest.get("acceptance_stage", "research"),
                        source_manifest=root / "source_manifest.json", source_root=source)
                    if not verdict["passed"]:
                        code = 2
                        phase = "harness_acceptance_failed"
            finally:
                stop_heartbeat.set()
                thread.join(timeout=1)
            heartbeat(returncode=code)
            return code
        print("Starting actual autosim research CLI; log:", root / "research.log", flush=True)
        with (root / "research.log").open("a", buffering=1) as log:
            child = subprocess.Popen(manifest["research_command"], cwd=source,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None and not stopped and time.monotonic() < deadline:
                heartbeat(cli_pid=child.pid)
                try:
                    child.wait(timeout=min(30, max(.1, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            if child.poll() is None:
                phase = "stopping" if stopped else "budget_exhausted"
                request_stop(signal.SIGTERM, None)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
                code = 124
            else:
                code = child.returncode
                phase = "cli_exited"
    except Exception as exc:
        # Credentials and request/response bodies are never logged here.
        phase = "launcher_failed"
        atomic_json(root / "launcher_failure.json", {"error_type": type(exc).__name__,
                                                     "error": str(exc)[:500], "at": now()})
        print("Launcher failed:", type(exc).__name__, flush=True)
    finally:
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        heartbeat(returncode=code, finished_at=now())
    return code


if __name__ == "__main__":
    raise SystemExit(main())
