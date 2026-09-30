"""Timeouts stop subprocess trees rather than only the immediate launcher."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from autosim.research.common import bounded_run, run_command
from autosim.research.process_executor import run_process


def _gone(pid: int, *, seconds: float = 1.5) -> bool:
    """Whether a process is no longer running, tolerated against the race that says it is.

    Three tests here watch a killed child leave `/proc`, and all three used to do it as
    `if not path.exists() or path.read_text()...`. Between the two calls the process can be
    reaped, and the kernel answers a read of `/proc/<pid>/stat` for a process mid-reap with
    `ESRCH` -- so the read raises `ProcessLookupError`. That is the *good* outcome arriving as
    a crash, which under load it did: it failed once in a full-suite run and never alone.

    A check that can fail on its own timing is not measuring what it claims, so the read is
    guarded and the two ways of being gone -- absent, or present as a zombie -- are one answer.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        status = Path(f"/proc/{pid}/stat")
        if not status.exists():
            return True
        try:
            if status.read_text().split()[2] == "Z":
                return True
        except (OSError, IndexError):
            # ESRCH or a truncated read: the process is being reaped, which is gone.
            return True
        time.sleep(0.05)
    return False


def test_timeout_terminates_spawned_child(tmp_path):
    code = ("import subprocess,sys,time; "
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
            "print(p.pid,flush=True); time.sleep(30)")
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        bounded_run([sys.executable, "-c", code], cwd=tmp_path,
                    env=dict(os.environ), timeout=0.5)
    pid = int(caught.value.output.strip().splitlines()[0])
    assert _gone(pid), "spawned benchmark worker survived the timeout"


def test_zero_exit_launcher_with_live_child_is_not_a_completed_probe(tmp_path):
    code = ("import subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'], "
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
            "print(p.pid,flush=True)")
    done = bounded_run([sys.executable, "-c", code], cwd=tmp_path,
                       env=dict(os.environ), timeout=3)
    assert done.returncode == 125
    pid = int(done.stdout.strip())
    assert _gone(pid), "child of a completed launcher survived the process-group audit"


def test_streamed_stage_detects_and_kills_child_after_parent_exit(tmp_path):
    code = ("import subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'], "
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
            "print(p.pid,flush=True)")
    log = tmp_path / "stage.log"
    with log.open("w", encoding="utf-8") as stream:
        outcome = run_process([sys.executable, "-c", code], cwd=tmp_path,
                              env=dict(os.environ), timeout=3,
                              stdout=stream, stderr=subprocess.STDOUT)
    assert outcome.launched and outcome.returncode == 0
    assert outcome.orphaned_children is True


def test_escaped_child_holding_pipe_cannot_hang_timeout_cleanup(tmp_path):
    marker = tmp_path / "escaped.pid"
    code = ("import pathlib,subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'], "
            "start_new_session=True); "
            "pathlib.Path(sys.argv[1]).write_text(str(p.pid))")
    began = time.monotonic()
    try:
        result = run_process([sys.executable, "-c", code, str(marker)], cwd=tmp_path,
                             env=dict(os.environ), timeout=0.3)
        assert result.timed_out and result.orphaned_children
        assert time.monotonic() - began < 5
    finally:
        if marker.exists():
            os.kill(int(marker.read_text()), 9)


def test_legacy_recorded_command_uses_same_process_cleanup(tmp_path):
    output = tmp_path / "attempt"
    result = run_command([sys.executable, "-c", "print('succ: 0.5')"],
                         cwd=tmp_path, env=dict(os.environ), output=output, timeout=3)
    assert result["status"] == "completed"
    assert result["pid"] > 0
    assert result["success_rate"] == 0.5
    assert not (output / "worker_identity.json").exists()
