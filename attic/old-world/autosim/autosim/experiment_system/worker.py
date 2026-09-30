"""Subprocess supervisor: retain inherited leases and persist completion receipt."""
import os
import signal
import subprocess
import sys
import time
import ctypes
from pathlib import Path

from autosim.research.common import atomic_json, now, read_json
from .executor import verify_sources


def descendants(pid):
    """Linux-only process ownership; never select unrelated processes by name."""
    found = []
    try:
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except FileNotFoundError:
        return found
    for value in children:
        child = int(value)
        found.extend(descendants(child))
        found.append(child)
    return found


def cleanup_owned_children():
    owned = descendants(os.getpid())
    for pid in owned:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if owned:
        time.sleep(.1)
    for pid in descendants(os.getpid()):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                break
        except ChildProcessError:
            break


def main():
    # Native training wrappers may start independent sessions. Adopt orphaned
    # grandchildren so timeout cleanup cannot release a GPU lease over live work.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError("Linux child-subreaper support required")
    def interrupted(signum, frame):
        raise InterruptedError(f"supervisor received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    directory = Path(sys.argv[1])
    request = read_json(directory / "request.json")
    start = time.monotonic()
    result = {"started_at": now(), "supervisor_pid": os.getpid()}
    process = None
    try:
        verify_sources(request["sources"])
        environment = os.environ.copy()
        environment.update(request["environment"])
        with (directory / "stdout.log").open("w") as log:
            process = subprocess.Popen(request["command"], cwd=request["cwd"], env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(directory / "started.json", {**result, "child_pid": process.pid})
            try:
                code = process.wait(timeout=request["timeout"])
                kind = "infrastructure" if code in {-11, -6, 137} else "execution_or_contract"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code, kind = process.returncode, "timeout"
            result.update(returncode=code, failure_kind=kind if code else None)
    except Exception as exc:
        result.update(returncode=125, failure_kind="execution_or_contract", error=f"{type(exc).__name__}: {exc}")
    finally:
        cleanup_owned_children()
        result.update(finished_at=now(), elapsed_seconds=time.monotonic() - start)
        atomic_json(directory / "receipt.json", result)


if __name__ == "__main__":
    main()
