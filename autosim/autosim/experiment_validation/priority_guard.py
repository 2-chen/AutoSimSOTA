"""Reserve the next scheduling window without taking GPU or active-job locks.

The existing production scheduler explicitly respects continuation/controller.lock.
This bounded guard uses that public cooperative gate, not process suspension. It
cannot preempt running experiments and does not guarantee fairness against other
schedulers. Restarting a guard does not extend its immutable expiry time.
"""
import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from autosim.experiment_system.executor import lease, verify_sources
from autosim.research.common import atomic_json, immutable_json, now, read_json


def process_identity(pid):
    try:
        # comm may contain spaces or parentheses; starttime is field 22.
        suffix = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return None if suffix[0] == "Z" else suffix[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def release_reason(request, *, current_time=None):
    expiry = datetime.fromisoformat(request["expires_at"])
    if expiry.tzinfo is None:
        raise ValueError("timezone-aware expiry required")
    if (current_time or datetime.now(timezone.utc)) >= expiry:
        return "reservation_budget_exhausted"
    if process_identity(request["queue_pid"]) != request["queue_process_start"]:
        return "queue_process_exited_or_replaced"
    queue = read_json(Path(request["queue_state"]))
    if queue.get("status") != "running":
        return "queue_not_running:" + str(queue.get("status"))
    verify_sources(request["sources"])
    return None


def run(request_path: Path, output: Path, *, poll_seconds=10):
    if not .01 <= poll_seconds <= 30:
        raise ValueError("bounded polling interval required")
    request = read_json(request_path)
    if not request.get("queue_process_start") or not request.get("sources"):
        raise ValueError("pinned process identity and sources required")
    lock = Path(request["scheduler_gate"])
    if lock.name != "controller.lock" or lock.parent.name != "continuation":
        raise ValueError("only the existing continuation scheduling gate is supported")
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / "request.json", request)
    state = {"started_at": now(), "pid": os.getpid(), "gpu_lock_acquired": False,
             "active_job_interrupted": False, "scheduler_gate": str(lock),
             "expires_at": request["expires_at"], "reservation_acquired": False}
    with lease(output / "guard.lock"):
        try:
            while True:
                reason = release_reason(request)
                if reason:
                    break
                try:
                    with lease(lock):
                        state.update(reservation_acquired=True, acquired_at=now())
                        while True:
                            reason = release_reason(request)
                            if reason:
                                break
                            state.update(status="reserving_next_window", heartbeat_at=now())
                            atomic_json(output / "state.json", state)
                            time.sleep(poll_seconds)
                    break
                except BlockingIOError:
                    state.update(status="waiting_existing_gate_owner", heartbeat_at=now())
                    atomic_json(output / "state.json", state)
                    time.sleep(poll_seconds)
        except Exception as exc:
            reason = f"guard_error:{type(exc).__name__}:{exc}"
        finally:
            state.update(status="released", reason=locals().get("reason", "interrupted"), finished_at=now())
            atomic_json(output / "state.json", state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.request, args.output)
