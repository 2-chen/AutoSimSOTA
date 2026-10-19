import hashlib
import os
import signal
import shutil
import subprocess
import sys
import time
import threading
import uuid
from pathlib import Path

import pytest

from autosim.research.common import isolated_argv
from autosim.research.process_executor import (capture_process_identity,
                                               inspect_process_identity,
                                               requested_containment,
                                               run_process,
                                               run_process_stream,
                                               terminate_recorded_process)


@pytest.mark.parametrize('runner', [run_process, run_process_stream])
def test_live_process_window_can_extend_without_restarting(tmp_path, runner):
    control = {"deadline_epoch": time.time() + 0.2, "cancelled": False}
    def extend():
        time.sleep(0.08)
        control["deadline_epoch"] = time.time() + 1.0
    thread = threading.Thread(target=extend)
    thread.start()
    result = runner([sys.executable, "-c", "import time; time.sleep(.35); print('done')"],
                         cwd=tmp_path, env=os.environ.copy(), timeout=2,
                         live_control=lambda: dict(control))
    thread.join()
    assert result.returncode == 0 and not result.timed_out
    assert result.stdout.strip() == "done"


@pytest.mark.parametrize('runner', [run_process, run_process_stream])
def test_live_window_extension_cannot_raise_the_hard_timeout(tmp_path, runner):
    started = time.monotonic()
    result = runner([sys.executable, '-c', 'import time; time.sleep(2)'],
        cwd=tmp_path, env=os.environ.copy(), timeout=.15,
        live_control=lambda: {'deadline_epoch': time.time()+100, 'cancelled': False})
    assert result.timed_out and result.returncode != 0
    assert time.monotonic()-started < 1.5


@pytest.mark.parametrize('runner', [run_process, run_process_stream])
def test_live_window_can_still_shorten_before_hard_timeout(tmp_path, runner):
    deadline = time.time()+.05
    result = runner([sys.executable, '-c', 'import time; time.sleep(2)'],
        cwd=tmp_path, env=os.environ.copy(), timeout=2,
        live_control=lambda: {'deadline_epoch': deadline, 'cancelled': False})
    assert result.timed_out and result.returncode != 0


@pytest.mark.parametrize('decouple_callbacks', [False, True])
@pytest.mark.parametrize('cancel', [False, True])
def test_stream_live_control_survives_launcher_exit(tmp_path, decouple_callbacks, cancel):
    # A native shell may exit while a worker still owns its stdout/stderr pipes.
    # The soft window and cancellation govern the job, not just the launcher PID.
    started = time.monotonic()
    wall_started = time.time()
    command = [sys.executable, '-c',
        "import subprocess, sys; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1.5)']); "
        "print('launcher-exited', flush=True)"]
    result = run_process_stream(command, cwd=tmp_path, env=os.environ.copy(),
        timeout=3, decouple_callbacks=decouple_callbacks,
        live_control=lambda: {'deadline_epoch': wall_started + (3 if cancel else .3),
                              'cancelled': cancel and time.time() >= wall_started + .3})
    assert 'launcher-exited' in result.stdout
    assert result.cancelled if cancel else result.timed_out
    assert time.monotonic() - started < 1.4
    assert not result.cleanup_diagnostics.get('remaining_pids')


def test_namespace_cleanup_allows_transient_teardown_without_signalling(monkeypatch):
    from types import SimpleNamespace
    import autosim.research.process_executor as executor
    snapshots = iter([[123], [123], []])
    monkeypatch.setattr(executor, "_live_group_members", lambda pgid: next(snapshots))
    monkeypatch.setattr(executor, "terminate_group",
                        lambda process: pytest.fail("transient teardown must not be signalled"))
    orphaned, audit = executor._audit_group_cleanup(SimpleNamespace(pid=123), "pid_namespace")
    assert not orphaned
    assert audit["observed_pids"] == [123]
    assert audit["persistent_pids"] == audit["remaining_pids"] == []
    assert audit["status"] == "namespace_settled"


@pytest.mark.parametrize("containment", ["pid_namespace", "process_group_only"])
def test_cleanup_preserves_persistent_orphan_failure_after_termination(monkeypatch, containment):
    from types import SimpleNamespace
    import autosim.research.process_executor as executor
    terminated = []
    monkeypatch.setattr(executor, "_live_group_members",
                        lambda pgid: [] if terminated else [456])
    monkeypatch.setattr(executor, "terminate_group", lambda process: terminated.append(process.pid))
    orphaned, audit = executor._audit_group_cleanup(SimpleNamespace(pid=123), containment)
    assert orphaned and terminated == [123]
    assert audit["persistent_pids"] == [456] and audit["remaining_pids"] == []
    assert audit["status"] == "terminated_orphans"


def test_live_process_cancellation_stops_process_group(tmp_path):
    control = {"deadline_epoch": time.time() + 2, "cancelled": False}
    def cancel():
        time.sleep(0.08)
        control["cancelled"] = True
    thread = threading.Thread(target=cancel)
    thread.start()
    result = run_process([sys.executable, "-c", "import time; time.sleep(3)"],
                         cwd=tmp_path, env=os.environ.copy(), timeout=4,
                         live_control=lambda: dict(control))
    thread.join()
    assert result.cancelled and not result.timed_out


def _sleeping_process():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                            start_new_session=True)


def test_recorded_process_group_is_identified_and_terminated(tmp_path):
    process = _sleeping_process()
    identity = capture_process_identity(
        process, run_id="run-1", attempt_id="attempt-1", argv=process.args)
    try:
        assert identity["pgid"] == process.pid
        assert inspect_process_identity(identity)["status"] == "matching_running"
        result = terminate_recorded_process(identity, grace_seconds=0.5)
        assert result["status"] == "terminated"
        assert inspect_process_identity(identity)["status"] == "not_running"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def test_process_identity_waits_out_the_pre_exec_empty_cmdline(monkeypatch):
    """A just-forked child may be visible before /proc/cmdline is populated."""
    import autosim.research.process_executor as executor

    process = _sleeping_process()
    original = executor._proc_identity
    reads = []

    def transient_empty(pid):
        observed = original(pid)
        reads.append(pid)
        if len(reads) == 1 and observed is not None:
            return {**observed,
                    "command_sha256": hashlib.sha256(b"").hexdigest()}
        return observed

    monkeypatch.setattr(executor, "_proc_identity", transient_empty)
    try:
        identity = capture_process_identity(
            process, run_id="run-1", attempt_id="exec-race", argv=process.args)
        assert len(reads) >= 2
        assert identity["command_sha256"] != hashlib.sha256(b"").hexdigest()
        assert inspect_process_identity(identity)["status"] == "matching_running"
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait()


def test_pid_with_different_start_identity_is_never_terminated():
    process = _sleeping_process()
    identity = capture_process_identity(
        process, run_id="run-1", attempt_id="attempt-2", argv=process.args)
    wrong = {**identity, "start_ticks": identity["start_ticks"] + 1}
    try:
        result = terminate_recorded_process(wrong, grace_seconds=0.1)
        assert result["status"] == "identity_mismatch"
        assert process.poll() is None
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait()


def test_incomplete_exited_launcher_is_not_perpetually_live():
    process = _sleeping_process()
    full = capture_process_identity(process, run_id='fixture', attempt_id='fast', argv=process.args)
    partial = {k:v for k,v in full.items() if k not in {'start_ticks','command_sha256'}}
    try:
        assert inspect_process_identity(partial)['status'] == 'unverifiable'
        assert terminate_recorded_process(partial)['status'] == 'unverifiable'
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait()
    assert inspect_process_identity(partial)['status'] == 'not_running'


@pytest.mark.parametrize('ticks,group,members,expected',[
    (200,888,[],'not_running'),
    (100,888,[],'identity_mismatch'),
    (50,888,[],'identity_mismatch'),
    (200,123,[],'identity_mismatch'),
    (200,888,[456],'identity_mismatch'),
])
def test_reused_pid_requires_later_birth_and_absent_original_group(
        monkeypatch,ticks,group,members,expected):
    import autosim.research.process_executor as module
    identity={'pid':123,'pgid':123,'start_ticks':100,
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'command_sha256':'old'}
    monkeypatch.setattr(module,'_proc_identity',lambda pid:{
        'pid':pid,'pgid':group,'start_ticks':ticks,'command_sha256':'new'})
    monkeypatch.setattr(module,'_live_group_members',lambda pgid,**kwargs:members)
    monkeypatch.setattr(module.os,'killpg',lambda *a:pytest.fail('never signal a replacement PID/group'))
    assert inspect_process_identity(identity)['status']==expected
    assert terminate_recorded_process(identity)['status']==expected


def test_reused_pid_group_inspection_error_remains_unverifiable(monkeypatch):
    import autosim.research.process_executor as module
    identity={'pid':123,'pgid':123,'start_ticks':100,
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'command_sha256':'old'}
    monkeypatch.setattr(module,'_proc_identity',lambda pid:{
        'pid':pid,'pgid':888,'start_ticks':200,'command_sha256':'new'})
    def denied(pgid,**kwargs):raise PermissionError('fixture')
    monkeypatch.setattr(module,'_live_group_members',denied)
    assert inspect_process_identity(identity)['status']=='unverifiable'


def test_strict_group_absence_does_not_swallow_permission_errors(tmp_path,monkeypatch):
    import autosim.research.process_executor as module
    proc=tmp_path/'proc';(proc/'123').mkdir(parents=True)
    target=proc/'123/stat';target.write_text('fixture')
    original=Path.read_text
    def read(path,*args,**kwargs):
        if path==target:raise PermissionError('fixture group may be invisible')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(module,'Path',lambda value:proc if value=='/proc' else Path(value))
    monkeypatch.setattr(Path,'read_text',read)
    assert module._live_group_members(999)==[]
    with pytest.raises(PermissionError):module._live_group_members(999,strict=True)


def test_incomplete_launcher_with_surviving_group_stays_unverifiable(monkeypatch):
    import autosim.research.process_executor as module
    monkeypatch.setattr(module, '_live_group_members', lambda group:[123])
    assert inspect_process_identity({'pid':2147483646,'pgid':2147483646,
                                    'boot_id':'fixture'})['status'] == 'unverifiable'


def test_streaming_process_delivers_complete_lines_before_exit(tmp_path):
    observed = []
    started = time.monotonic()
    result = run_process_stream(
        [sys.executable, "-c",
         "import time; print('first', flush=True); time.sleep(.3); print('last', end='')"],
        cwd=tmp_path, env=os.environ.copy(), timeout=5,
        on_stdout_line=lambda line: observed.append((line, time.monotonic() - started)))

    assert result.returncode == 0
    assert result.error is None
    assert [line for line, _ in observed] == ["first", "last"]
    assert observed[0][1] < .25
    assert result.stdout == "first\nlast"


def test_streaming_process_accepts_large_stdin_without_argv_or_file_artifact(tmp_path):
    payload = b"research evidence " * 10000
    result = run_process_stream(
        [sys.executable, "-c", "import sys; print(len(sys.stdin.buffer.read()))"],
        cwd=tmp_path, env=os.environ.copy(), timeout=5, input_bytes=payload)
    assert result.returncode == 0
    assert result.stdout.strip() == str(len(payload))
    assert list(tmp_path.iterdir()) == []


def test_streaming_process_rejects_oversized_output_and_stops_group(tmp_path):
    result = run_process_stream(
        [sys.executable, "-c", "import time; print('x' * 10000, flush=True); time.sleep(30)"],
        cwd=tmp_path, env=os.environ.copy(), timeout=5, max_output_bytes=1024)

    assert result.launched
    assert result.error is not None
    assert "exceeded 1024 bytes" in str(result.error)
    assert result.returncode is not None
    assert len(result.stdout or "") <= 1024


def test_streaming_process_emits_heartbeats_while_child_is_silent(tmp_path):
    heartbeats = []
    result = run_process_stream(
        [sys.executable, "-c", "import time; time.sleep(.35); print('finished')"],
        cwd=tmp_path, env=os.environ.copy(), timeout=5,
        on_heartbeat=lambda: heartbeats.append(time.monotonic()), heartbeat_interval=.08)

    assert result.returncode == 0
    assert len(heartbeats) >= 2


def test_streaming_process_emits_heartbeats_while_child_is_silent(tmp_path):
    heartbeats = []
    result = run_process_stream(
        [sys.executable, "-c", "import time; time.sleep(.35); print('finished')"],
        cwd=tmp_path, env=os.environ.copy(), timeout=5,
        on_heartbeat=lambda: heartbeats.append(time.monotonic()), heartbeat_interval=.08)

    assert result.returncode == 0
    assert len(heartbeats) >= 2


def test_live_descendants_with_missing_leader_are_not_assumed_safe_to_retry():
    parent_code = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "time.sleep(0.5)")
    process = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)
    identity = capture_process_identity(
        process, run_id="run-1", attempt_id="attempt-3", argv=process.args)
    try:
        process.wait(timeout=5)
        status = inspect_process_identity(identity)
        assert status["status"] == "unverifiable"
        assert status["remaining_pids"]
        result = terminate_recorded_process(identity, grace_seconds=0.1)
        assert result["status"] == "unverifiable"
    finally:
        try:
            os.killpg(identity["pgid"], signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_pid_namespace_stops_a_descendant_that_escapes_its_process_group(tmp_path):
    """A benchmark's setsid() child must die when the tracked namespace init is killed."""
    bubblewrap = shutil.which("bwrap")
    if not bubblewrap:
        pytest.skip("bubblewrap is not available")
    repo = tmp_path / "repo"
    output = tmp_path / "run"
    repo.mkdir()
    output.mkdir()
    token = "autosim-escape-test-" + uuid.uuid4().hex
    child_code = f"import os,time; os.setsid(); time.sleep(60)  # {token}"
    launcher_code = ("import subprocess,sys,time; "
                     "subprocess.Popen([sys.executable,'-c',sys.argv[1]]); time.sleep(60)")
    command = isolated_argv([sys.executable, "-c", launcher_code, child_code],
                            output=output, repo=repo)
    assert requested_containment(command) == "pid_namespace"

    def escaped_pids():
        found = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                if token.encode() in (entry / "cmdline").read_bytes():
                    found.append(int(entry.name))
            except OSError:
                continue
        return found

    process = subprocess.Popen(command, cwd=repo, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    identity = capture_process_identity(process, run_id="run-1", attempt_id="escape-1",
                                        argv=command)
    try:
        deadline = time.monotonic() + 5
        descendants = []
        while time.monotonic() < deadline and process.poll() is None:
            descendants = escaped_pids()
            if descendants:
                break
            time.sleep(0.02)
        assert descendants, "fixture never started its setsid() descendant"
        assert terminate_recorded_process(identity, grace_seconds=0.2)["status"] == "terminated"
        process.wait(timeout=5)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and escaped_pids():
            time.sleep(0.02)
        assert escaped_pids() == []
    finally:
        if process.poll() is None:
            terminate_recorded_process(identity, grace_seconds=0.2)
            process.wait(timeout=5)
        # Do not leave a worker behind even if this regression assertion fails.
        for pid in escaped_pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
