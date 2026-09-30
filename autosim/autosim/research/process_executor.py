"""One bounded subprocess mechanism for probes and native streaming stages.

Receipts remain with the caller because a probe and a scored stage have different
postconditions. Launch, timeout, process-group cleanup and output capture are shared.
"""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import tempfile
import hashlib
import json
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable


_PROCESS_START_OBSERVER: ContextVar[
    Callable[[subprocess.Popen[Any]], None] | None
] = ContextVar("autosim_process_start_observer", default=None)


@contextmanager
def observe_process_starts(callback: Callable[[subprocess.Popen[Any]], None]):
    """Observe subprocesses launched by bounded helpers during one controller action."""
    token: Token = _PROCESS_START_OBSERVER.set(callback)
    try:
        yield
    finally:
        _PROCESS_START_OBSERVER.reset(token)


@dataclass(frozen=True)
class ProcessAttempt:
    launched: bool
    returncode: int | None
    stdout: str | None = None
    stderr: str | None = None
    timed_out: bool = False
    cancelled: bool = False
    error: Exception | None = None
    orphaned_children: bool = False
    containment_mode: str = "process_group_only"


def requested_containment(command: list[str] | str) -> str:
    """Describe the boundary encoded by the launcher argv, not an OS guarantee."""
    if isinstance(command, list) and command and Path(command[0]).name == "bwrap":
        if "--unshare-pid" in command and "--die-with-parent" in command:
            return "pid_namespace"
    return "process_group_only"


def _live_group_members(pgid: int) -> list[int]:
    """Best-effort Linux audit after the direct process exits."""
    directory = Path("/proc")
    if not directory.is_dir():
        return []
    found: list[int] = []
    for entry in directory.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2:].split()
            if len(fields) > 2 and fields[0] not in {"Z", "X"} and int(fields[2]) == pgid:
                found.append(int(entry.name))
        except (OSError, ValueError):
            continue
    return found


def _proc_identity(pid: int) -> dict[str, Any] | None:
    """Read Linux process start ticks and process group without trusting a bare PID."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2:].split()
        if len(fields) < 20:
            return None
        if fields[0] in {"Z", "X"}:
            return None
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        # A newly forked child can be visible in /proc before exec has populated its
        # command line. Hashing that transient empty value creates an identity that is
        # guaranteed to disagree as soon as the program starts.
        if not command:
            return None
        return {"pid": pid, "state": fields[0],
                "pgid": int(fields[2]), "start_ticks": int(fields[19]),
                "command_sha256": hashlib.sha256(command).hexdigest()}
    except (OSError, ValueError):
        return None


def capture_process_identity(process: subprocess.Popen[Any], *, run_id: str,
                             attempt_id: str, argv: list[str] | str) -> dict[str, Any]:
    """Capture enough host-local identity to distinguish a live attempt from PID reuse."""
    identity = _proc_identity(process.pid)
    deadline = time.monotonic() + 0.5
    empty_command = hashlib.sha256(b"").hexdigest()
    while (identity is None or identity.get("command_sha256") == empty_command) and \
            process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
        identity = _proc_identity(process.pid)
    if identity is None or identity.get("command_sha256") == empty_command:
        # Incomplete identity is intentionally unverifiable on recovery, never a reason to
        # signal a process based on a PID alone.
        identity = {"pid": process.pid, "pgid": process.pid}
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii").strip()
    except OSError:
        boot_id = ""
    identity.update({"schema_version": 1, "run_id": str(run_id),
                     "attempt_id": str(attempt_id), "boot_id": boot_id,
                     "argv_sha256": hashlib.sha256(json.dumps(
                         argv, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")).hexdigest()})
    return identity


def inspect_process_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Report whether the recorded process leader is still the same process."""
    try:
        pid = int(identity["pid"])
        pgid = int(identity["pgid"])
        start_ticks = int(identity["start_ticks"])
        boot_id = str(identity["boot_id"])
        command_sha256 = str(identity["command_sha256"])
    except (KeyError, TypeError, ValueError):
        return {"status": "unverifiable", "why": "recorded process identity is incomplete"}
    try:
        current_boot = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii").strip()
    except OSError:
        return {"status": "unverifiable", "why": "host boot identity is unavailable"}
    if not boot_id or current_boot != boot_id:
        return {"status": "not_running", "why": "host boot identity changed"}
    current = _proc_identity(pid)
    if current is None:
        remaining = _live_group_members(pgid)
        if remaining:
            return {"status": "unverifiable",
                    "why": "process leader is absent but members remain in its process group",
                    "remaining_pids": remaining}
        return {"status": "not_running", "why": "recorded process is absent"}
    if (current["pgid"] != pgid or current["start_ticks"] != start_ticks or
            current["command_sha256"] != command_sha256):
        return {"status": "identity_mismatch",
                "why": "PID exists but process group, start time, or command differs"}
    return {"status": "matching_running", "pid": pid, "pgid": pgid}


def terminate_recorded_process(identity: dict[str, Any], *,
                               grace_seconds: float = 3) -> dict[str, Any]:
    """Terminate only a process group whose recorded leader still matches its identity."""
    observed = inspect_process_identity(identity)
    if observed.get("status") != "matching_running":
        return observed
    pgid = int(observed["pgid"])
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError) as exc:
        return {"status": "termination_failed",
                "why": f"{type(exc).__name__}: {exc}"}
    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline and _live_group_members(pgid):
        time.sleep(0.05)
    remaining = _live_group_members(pgid)
    if remaining:
        # The verified attempt created a new session, so only this run's descendants can
        # share its process group. Keep the group id reserved by checking members immediately
        # before escalation; an empty group is never signalled.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError) as exc:
            return {"status": "termination_failed",
                    "why": f"{type(exc).__name__}: {exc}",
                    "remaining_pids": remaining}
    deadline = time.monotonic() + max(0.5, grace_seconds)
    while time.monotonic() < deadline and _live_group_members(pgid):
        time.sleep(0.05)
    remaining = _live_group_members(pgid)
    return {"status": "terminated" if not remaining else "still_running",
            "why": "recorded process group stopped" if not remaining else
                   "process group remains after termination",
            "remaining_pids": remaining}


def terminate_group(process: subprocess.Popen[Any], *, grace_seconds: float = 3) -> None:
    """Stop a newly created session, then reap its direct process.

    A child that calls setsid() can escape a process group; cgroup-level containment is a
    separate requirement and this function does not claim to provide it.
    """
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # The launcher can exit promptly while its children keep the process group alive.
    # Reaping the parent alone is not evidence that the group has ended.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def run_process(command: list[str] | str, *, cwd: Path, env: dict[str, str],
                timeout: float, shell: bool = False, input: str | None = None,
                stdout: Any = subprocess.PIPE, stderr: Any = subprocess.PIPE,
                during: ContextManager[Any] | None = None,
                on_start: Callable[[subprocess.Popen[Any]], None] | None = None,
                pass_fds: Iterable[int] = (),
                live_control: Callable[[], dict[str, Any]] | None = None
                ) -> ProcessAttempt:
    """Run one isolated session, returning facts rather than interpreting success."""
    if timeout <= 0:
        raise ValueError("process timeout must be positive")
    containment_mode = requested_containment(command)
    try:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, shell=shell, text=True,
            stdout=stdout, stderr=stderr,
            stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
            start_new_session=True, pass_fds=tuple(pass_fds))
    except OSError as exc:
        return ProcessAttempt(False, None, error=exc,
                              containment_mode=containment_mode)
    try:
        observer = on_start or _PROCESS_START_OBSERVER.get()
        if observer is not None:
            observer(process)
        cancelled = False
        with during if during is not None else nullcontext():
            if live_control is None:
                produced, errors = process.communicate(input=input, timeout=timeout)
            else:
                hard_deadline = time.time() + timeout
                pending_input = input
                while True:
                    control = live_control()
                    cancelled = control.get("cancelled") is True
                    allowed_deadline = min(hard_deadline, float(control["deadline_epoch"]))
                    if cancelled:
                        terminate_group(process)
                        produced, errors = process.communicate(timeout=2)
                        break
                    remaining = allowed_deadline - time.time()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        produced, errors = process.communicate(
                            input=pending_input, timeout=min(remaining, 0.5))
                        break
                    except subprocess.TimeoutExpired:
                        pending_input = None
                        continue
        orphaned = bool(_live_group_members(process.pid))
        if orphaned:
            terminate_group(process)
        return ProcessAttempt(True, process.returncode, produced, errors,
                              cancelled=cancelled,
                              orphaned_children=orphaned,
                              containment_mode=containment_mode)
    except subprocess.TimeoutExpired:
        terminate_group(process)
        try:
            produced, errors = process.communicate(timeout=2)
            return ProcessAttempt(True, None, produced, errors, timed_out=True,
                                  containment_mode=containment_mode)
        except subprocess.TimeoutExpired as exc:
            # A worker that creates a new session can retain our output pipes after its
            # launcher is gone. Never wait forever for EOF from outside our process group.
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            return ProcessAttempt(True, None, _partial_text(exc.output),
                                  _partial_text(exc.stderr), timed_out=True,
                                  orphaned_children=True,
                                  containment_mode=containment_mode)
    except OSError as exc:
        terminate_group(process)
        try:
            produced, errors = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            produced, errors = None, None
        return ProcessAttempt(True, None, produced, errors, error=exc,
                              containment_mode=containment_mode)
    except BaseException:
        terminate_group(process)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        raise


def _partial_text(value: str | bytes | None) -> str | None:
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def run_process_stream(command: list[str] | str, *, cwd: Path, env: dict[str, str],
                       timeout: float,
                       input_bytes: bytes | None = None,
                       on_stdout_line: Callable[[str], None] | None = None,
                       on_stderr_line: Callable[[str], None] | None = None,
                       on_heartbeat: Callable[[], None] | None = None,
                       heartbeat_interval: float = 15,
                       max_output_bytes: int = 16 * 1024 * 1024,
                       max_line_bytes: int = 1024 * 1024,
                       during: ContextManager[Any] | None = None,
                       on_start: Callable[[subprocess.Popen[Any]], None] | None = None,
                       pass_fds: Iterable[int] = ()) -> ProcessAttempt:
    """Run one bounded process while delivering newline-delimited output as it arrives.

    The output cap applies to captured stdout plus stderr; the process is terminated when
    exceeded instead of silently truncating a structured protocol stream. Callback failures
    also terminate the process group and are returned as an error receipt.
    """
    if timeout <= 0:
        raise ValueError("process timeout must be positive")
    if max_output_bytes <= 0 or max_line_bytes <= 0 or heartbeat_interval <= 0:
        raise ValueError("stream output limits must be positive")
    containment_mode = requested_containment(command)
    input_stream = None
    try:
        if input_bytes is not None:
            input_stream = tempfile.TemporaryFile()
            input_stream.write(input_bytes)
            input_stream.seek(0)
        process = subprocess.Popen(
            command, cwd=cwd, env=env, text=False, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=input_stream if input_stream is not None else subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=tuple(pass_fds))
    except OSError as exc:
        return ProcessAttempt(False, None, error=exc,
                              containment_mode=containment_mode)
    finally:
        if input_stream is not None:
            input_stream.close()

    output = {"stdout": bytearray(), "stderr": bytearray()}
    partial = {"stdout": bytearray(), "stderr": bytearray()}
    callbacks = {"stdout": on_stdout_line, "stderr": on_stderr_line}
    selector = selectors.DefaultSelector()
    streams = {"stdout": process.stdout, "stderr": process.stderr}
    deadline = time.monotonic() + timeout
    last_heartbeat = time.monotonic()
    timed_out = False
    failure: Exception | None = None
    overflow = False

    def deliver(name: str, raw: bytes, *, final: bool = False) -> None:
        nonlocal failure
        buffer = partial[name]
        buffer.extend(raw)
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                break
            line = bytes(buffer[:newline])
            del buffer[:newline + 1]
            callback = callbacks[name]
            if callback is not None:
                callback(line.decode("utf-8", errors="replace"))
        if len(buffer) > max_line_bytes:
            raise ValueError(f"{name} line exceeded the {max_line_bytes}-byte limit")
        if final and buffer:
            callback = callbacks[name]
            if callback is not None:
                callback(bytes(buffer).decode("utf-8", errors="replace"))
            buffer.clear()

    try:
        observer = on_start or _PROCESS_START_OBSERVER.get()
        if observer is not None:
            observer(process)
        for name, stream in streams.items():
            if stream is not None:
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
        with during if during is not None else nullcontext():
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                ready = selector.select(min(remaining, 0.25))
                for key, _ in ready:
                    name = str(key.data)
                    stream = key.fileobj
                    try:
                        chunk = os.read(stream.fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(stream)
                        stream.close()
                        deliver(name, b"", final=True)
                        continue
                    remaining_bytes = max_output_bytes - sum(len(value) for value in output.values())
                    if len(chunk) > remaining_bytes:
                        output[name].extend(chunk[:max(0, remaining_bytes)])
                        overflow = True
                        raise ValueError(
                            f"combined process output exceeded {max_output_bytes} bytes")
                    output[name].extend(chunk)
                    deliver(name, chunk)
                now = time.monotonic()
                if on_heartbeat is not None and now - last_heartbeat >= heartbeat_interval:
                    on_heartbeat()
                    last_heartbeat = now
    except Exception as exc:
        failure = exc
    except BaseException:
        terminate_group(process)
        raise
    finally:
        selector.close()

    if timed_out or failure is not None:
        terminate_group(process)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        terminate_group(process, grace_seconds=0.2)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            failure = failure or TimeoutError("process did not exit after group termination")
    for stream in streams.values():
        if stream is not None and not stream.closed:
            stream.close()

    stdout_text = output["stdout"].decode("utf-8", errors="replace")
    stderr_text = output["stderr"].decode("utf-8", errors="replace")
    if overflow and failure is None:
        failure = ValueError("process output limit exceeded")
    orphaned = bool(_live_group_members(process.pid))
    if orphaned:
        terminate_group(process)
    return ProcessAttempt(True, process.returncode, stdout_text, stderr_text,
                          timed_out=timed_out, error=failure,
                          orphaned_children=orphaned,
                          containment_mode=containment_mode)
