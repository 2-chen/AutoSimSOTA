"""Own one child process group and terminate it if its dispatcher disappears."""
from __future__ import annotations
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    parent, receipt, command = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    child = None
    interrupted = False
    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        if child is not None:
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if sys.platform == "linux":
        # No preexec_fn in the multithreaded dispatcher; prctl is set in this fresh interpreter.
        ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0)
    if os.getppid() != parent:
        return 125
    child = subprocess.Popen(command, start_new_session=True)
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps({"guard_pid": os.getpid(), "worker_pid": child.pid,
                                     "worker_pgid": child.pid, "parent_pid": parent}))
    os.replace(temporary, receipt)
    journal = os.environ.get("AUTOSIM_JOB_JOURNAL_ROOT")
    if journal:
        root = Path(journal); root.mkdir(parents=True, exist_ok=True)
        target = root / f"{child.pid}.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps({"worker_pid":child.pid,"guard_pid":os.getpid(),
                                  "host":os.uname().nodename,"source":str(receipt)}))
        os.replace(temp,target)
    deadline = None
    while child.poll() is None:
        if interrupted or os.getppid() != parent:
            if deadline is None:
                stop(signal.SIGTERM, None)
                deadline = time.monotonic() + 10
            if time.monotonic() >= deadline:
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
        time.sleep(.05)
    # Clean up remaining members even when the group leader exited normally.
    try: os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    code = child.returncode
    if code < 0:
        if -code not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(-code, signal.SIG_DFL)
        os.kill(os.getpid(), -code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
