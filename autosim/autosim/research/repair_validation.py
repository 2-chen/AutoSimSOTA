"""Trusted differential validation with a kernel-enforced untrusted-code process.

Expected answers remain in the parent.  The candidate receives input cases only.
Landlock limits filesystem access; seccomp denies networking, process inspection,
signals and new processes.  Missing kernel facilities disable L2 rather than
silently running candidate Python with the controller's credentials or access.
"""
from __future__ import annotations

import ast
import ctypes
import ctypes.util
import errno
import hashlib
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationSpec:
    """Host-owned contract. Cases have args/kwargs and expected or raises."""
    name: str
    target: str
    function: str
    cases: tuple[dict, ...]
    timeout_seconds: float = 15.0
    native_required: bool = False
    pure_component: bool = False

    def identity(self) -> str:
        body = {"name": self.name, "target": self.target, "function": self.function,
                "cases": self.cases, "timeout_seconds": self.timeout_seconds,
                "native_required": self.native_required, "pure_component": self.pure_component}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class IsolationUnavailable(RuntimeError):
    pass


def check_pure_component(source: str, function: str = "ready_members_match") -> None:
    """Limit code later imported by the supervisor to a pure JSON predicate.

    A test sandbox alone cannot make arbitrary Python safe to import into the
    live coordinator. This structural gate also runs at activation time.
    """
    tree = ast.parse(source)
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(definitions) != 1 or definitions[0].name != function:
        raise ValueError("repair component must contain exactly the registered function")
    definition = definitions[0]
    if (definition.decorator_list or definition.args.vararg or definition.args.kwarg or
            definition.args.kwonlyargs or definition.args.defaults or
            [arg.arg for arg in definition.args.args] != ["expected", "ready", "generation"]):
        raise ValueError("repair component signature cannot change")
    for node in tree.body:
        if node is definition or isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if (isinstance(node, ast.ImportFrom) and node.module == "__future__" and
                all(name.name == "annotations" and name.asname is None for name in node.names)):
            continue
        raise ValueError("repair component may not execute module-level code")
    calls = {"len", "isinstance", "type", "list", "dict", "tuple", "set", "str", "bool", "all", "any",
             "enumerate", "zip", "sorted", "range"}
    attributes = {"get", "add", "keys", "values", "items"}
    forbidden = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef, ast.Lambda, ast.While,
                 ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Await, ast.Yield, ast.YieldFrom,
                 ast.Delete, ast.Raise)
    nodes = list(ast.walk(definition))
    if len(nodes) > 3000:
        raise ValueError("repair component exceeds structural limit")
    for node in nodes:
        if isinstance(node, forbidden) or isinstance(node, ast.FunctionDef) and node is not definition:
            raise ValueError("repair component contains non-pure control or code execution")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("repair component cannot access interpreter internals")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in calls:
            raise ValueError("repair component cannot shadow trusted builtins")
        if isinstance(node, ast.Attribute) and node.attr not in attributes:
            raise ValueError("repair component attribute is outside the JSON contract")
        if isinstance(node, ast.Call):
            allowed = isinstance(node.func, ast.Name) and node.func.id in calls or (
                isinstance(node.func, ast.Attribute) and node.func.attr in attributes)
            if not allowed:
                raise ValueError("repair component call is outside the JSON contract")


def _landlock(read_paths: list[Path], write_path: Path) -> int:
    if sys.platform != "linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise IsolationUnavailable("unsupported Landlock platform")
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 3:
        raise IsolationUnavailable("Landlock ABI 3 or newer is required")

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathBeneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    mask = (1 << 15) - 1
    attr = Ruleset(mask)
    fd = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if fd < 0:
        raise IsolationUnavailable("Landlock ruleset creation failed")
    try:
        readonly = (1 << 0) | (1 << 2) | (1 << 3)
        for path, access in [(p, readonly) for p in read_paths] + [(write_path, mask)]:
            if not path.exists():
                continue
            handle = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                if path.is_file():
                    access &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
                rule = PathBeneath(access, handle)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) < 0:
                    raise IsolationUnavailable("Landlock path rule failed")
            finally:
                os.close(handle)
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            raise IsolationUnavailable("no_new_privs unavailable")
        if libc.syscall(446, fd, 0) < 0:
            raise IsolationUnavailable("Landlock restriction failed")
    finally:
        os.close(fd)
    return int(abi)


def _seccomp(name: str | None) -> None:
    if not name:
        raise IsolationUnavailable("libseccomp unavailable")
    library = ctypes.CDLL(name, use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise IsolationUnavailable("seccomp initialization failed")
    try:
        denied = ("socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
                  "sendto", "sendmsg", "sendmmsg", "recvmsg", "recvmmsg", "ptrace", "process_vm_readv",
                  "process_vm_writev", "pidfd_getfd", "kcmp", "kill", "tkill", "tgkill", "pidfd_send_signal",
                  "fork", "vfork", "clone", "clone3", "execve", "execveat", "mount", "umount2",
                  "pivot_root", "chroot", "unshare", "setns", "bpf", "perf_event_open", "io_uring_setup",
                  "keyctl", "add_key", "request_key", "open_by_handle_at", "reboot", "kexec_load")
        for syscall in denied:
            number = library.seccomp_syscall_resolve_name(syscall.encode())
            if number >= 0 and library.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0) != 0:
                raise IsolationUnavailable("seccomp rule installation failed")
        if library.seccomp_load(context) != 0:
            raise IsolationUnavailable("seccomp filter loading failed")
    finally:
        library.seccomp_release(context)


def _apply_isolation(candidate_root: Path, scratch: Path, timeout: float, backend: str) -> dict:
    # Resolve/load libraries before filesystem confinement; no candidate code has
    # run yet.  No environment inherited by the child contains API credentials.
    import socket  # noqa: F401
    seccomp_library = ctypes.util.find_library("seccomp") or "libseccomp.so.2"
    ctypes.CDLL(seccomp_library or "libseccomp.so.2")
    read_paths = [Path(__file__).resolve(), candidate_root.resolve(), Path(sys.executable).resolve(),
                  Path("/usr/lib"), Path("/usr/local/lib"), Path("/lib"), Path("/lib64"),
                  Path("/etc/ld.so.cache"), Path("/dev/null"), Path("/dev/urandom")]
    for base in {Path(sys.base_prefix), Path(sys.prefix)}:
        if (base / "lib").exists():
            read_paths.append(base / "lib")
    abi = _landlock(read_paths, scratch) if backend == "landlock_seccomp" else None
    if backend not in {"landlock_seccomp", "bubblewrap_seccomp"}:
        raise IsolationUnavailable("unknown isolation backend")
    if backend == "bubblewrap_seccomp":
        # bwrap constructs the filesystem before exec. This marker alone is not
        # trusted: the parent also probes private-read/outside-write denial.
        if not Path("/autosim-bwrap-root").is_dir():
            raise IsolationUnavailable("bubblewrap mount namespace is absent")
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise IsolationUnavailable("no_new_privs unavailable")
    _seccomp(seccomp_library)
    resource.setrlimit(resource.RLIMIT_CPU, (max(1, int(timeout)), max(2, int(timeout) + 1)))
    resource.setrlimit(resource.RLIMIT_AS, (384 * 1024 * 1024, 384 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024, 2 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.chdir(scratch)
    return {"backend": backend, "landlock_abi": abi, "seccomp": True,
            "network": "denied", "process_creation": "denied"}


def _worker() -> int:
    request = json.loads(sys.stdin.read(2 * 1024 * 1024))
    root, scratch = Path(request["root"]), Path(request["scratch"])
    try:
        isolation = _apply_isolation(root, scratch, request["timeout"], request["backend"])
    except Exception as exc:
        print(json.dumps({"isolation_error": type(exc).__name__, "reason": str(exc)[:200]}))
        return 77
    if request.get("probe"):
        import socket
        denied = {}
        for name, operation in {
            "private_read": lambda: Path(request["private_probe"]).read_bytes(),
            "outside_write": lambda: Path(request["write_probe"]).write_text("must never be written"),
            "network": lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM),
            "process": lambda: os.fork(),
        }.items():
            try:
                operation()
            except PermissionError:
                denied[name] = True
            except OSError as exc:
                denied[name] = exc.errno in {errno.EPERM, errno.EACCES, errno.ENOENT}
            else:
                denied[name] = False
        print(json.dumps({"isolation": isolation, "denied": denied}))
        return 0 if all(denied.values()) else 78
    source = root / request["target"]
    namespace = {"__name__": "repair_candidate", "__file__": str(source)}
    try:
        exec(compile(source.read_text(), str(source), "exec"), namespace)
        function = namespace[request["function"]]
        results = []
        for case in request["cases"]:
            try:
                original_input = json.dumps(case, sort_keys=True)
                result = function(*case.get("args", []), **case.get("kwargs", {}))
                # Keep JSON exact: NaN, custom encoders and reprs are not evidence.
                json.dumps(result, allow_nan=False)
                results.append({"value": result} if json.dumps(case, sort_keys=True) == original_input else {"input_mutation": True})
            except Exception as exc:
                results.append({"raises": type(exc).__name__})
        print(json.dumps({"results": results}, allow_nan=False))
        return 0
    except BaseException as exc:
        print(json.dumps({"candidate_error": type(exc).__name__}))
        return 79


def _bubblewrap_command(candidate_root: Path) -> list[str]:
    executable = os.environ.get("AUTOSIM_REPAIR_BWRAP") or shutil.which("bwrap")
    if not executable:
        raise IsolationUnavailable("bubblewrap executable unavailable")
    python = Path(sys.executable).resolve()
    command = [executable, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
               "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "LANG", "C.UTF-8",
               "--dir", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
               "--dir", "/tmp/scratch", "--dir", "/autosim-bwrap-root"]
    # No home, project root, run directory or credential mount. Candidate files
    # are read-only; all writes disappear with the private /tmp filesystem.
    for path in (Path("/usr/lib"), Path("/usr/local/lib"), Path("/lib"), Path("/lib64")):
        if path.exists():
            command += ["--ro-bind", str(path.resolve()), str(path)]
    if Path("/etc/ld.so.cache").exists():
        command += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    command += ["--ro-bind", str(python), "/python", "--ro-bind", str(Path(__file__).resolve()), "/runner.py",
                "--ro-bind", str(candidate_root), "/candidate", "--chdir", "/tmp/scratch",
                "--", "/python", "-I", "-S", "/runner.py", "--worker"]
    return command


def _subprocess(request: dict, *, timeout: float, backend: str = "landlock_seccomp") -> dict:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="autosim-repair-child-") as temporary:
        temp = Path(temporary)
        scratch = temp / "scratch"
        scratch.mkdir(mode=0o700)
        request = {**request, "scratch": str(scratch), "timeout": timeout, "backend": backend}
        private = temp / "controller_private.env"
        private.write_text("API_KEY=not-accessible-to-candidate")
        outside = temp / "outside.txt"
        request.update(private_probe=str(private), write_probe=str(outside))
        command = [sys.executable, "-I", "-S", str(Path(__file__).resolve()), "--worker"]
        if backend == "bubblewrap_seccomp":
            try:
                command = _bubblewrap_command(Path(request["root"]))
            except IsolationUnavailable as exc:
                return {"status": "unavailable", "reason": str(exc), "returncode": 77}
            request.update(root="/candidate", scratch="/tmp/scratch")
        with (temp / "stdout").open("w+b") as stdout, (temp / "stderr").open("w+b") as stderr:
            process = subprocess.Popen(command,
                stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, start_new_session=True,
                close_fds=True, env={"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"})
            try:
                process.communicate(json.dumps(request, allow_nan=False).encode(), timeout=timeout + 3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                return {"status": "timeout", "returncode": process.returncode,
                        "elapsed_seconds": time.monotonic() - started}
            stdout.seek(0)
            stderr.seek(0)
            data = stdout.read(2 * 1024 * 1024 + 1)
            try:
                payload = json.loads(data) if len(data) <= 2 * 1024 * 1024 else {}
            except (ValueError, UnicodeError):
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            return {"status": "completed" if process.returncode == 0 else "failed",
                    "returncode": process.returncode, "payload": payload,
                    "launcher_stderr": stderr.read(1200).decode("utf-8", errors="replace"),
                    "elapsed_seconds": time.monotonic() - started,
                    "outside_write_detected": outside.exists()}


def isolation_capability() -> dict:
    """Exercise deny rules in a disposable process; ABI query alone is insufficient."""
    attempts = {}
    for backend in ("landlock_seccomp", "bubblewrap_seccomp"):
        with tempfile.TemporaryDirectory(prefix="autosim-repair-probe-") as temporary:
            result = _subprocess({"root": temporary, "probe": True}, timeout=5, backend=backend)
        payload = result.get("payload", {})
        passed = (result.get("returncode") == 0 and not result.get("outside_write_detected") and
                  set(payload.get("denied", {})) == {"private_read", "outside_write", "network", "process"} and
                  all(payload["denied"].values()))
        attempts[backend] = result
        if passed:
            return {"l2_available": True, "backend": backend, "reason": "enforced_and_probed", "probes": attempts}
    return {"l2_available": False, "backend": None, "reason": "kernel_isolation_unavailable", "probes": attempts}


def run_contract(root: Path, spec: ValidationSpec) -> dict:
    """Execute a component and compare answers outside the candidate process."""
    from .incidents import permitted_relative_path
    relative = permitted_relative_path(spec.target)
    root = Path(root).resolve()
    target = root / relative
    if not target.is_file() or target.is_symlink() or not target.resolve().is_relative_to(root):
        raise PermissionError("validation target outside candidate")
    if target.stat().st_size > 262144 or not spec.cases or not 0 < spec.timeout_seconds <= 60:
        raise ValueError("validation specification outside limits")
    source = target.read_bytes()
    ast.parse(source)
    if spec.pure_component:
        check_pure_component(source.decode(), spec.function)
    inputs = [{key: case[key] for key in ("args", "kwargs") if key in case} for case in spec.cases]
    capability = isolation_capability()
    if not capability["l2_available"]:
        raise IsolationUnavailable("L2 unavailable: kernel confinement did not pass")
    # Never mount an arbitrary caller root: it might also contain .env, trusted
    # validators or sealed artifacts. This L2 backend deliberately supports only
    # standalone registered components and stages exactly that one source file.
    with tempfile.TemporaryDirectory(prefix="autosim-repair-component-") as temporary:
        isolated_root = Path(temporary)
        isolated_target = isolated_root / relative
        isolated_target.parent.mkdir(parents=True, exist_ok=True)
        isolated_target.write_bytes(source)
        result = _subprocess({"root": str(isolated_root), "target": relative, "function": spec.function, "cases": inputs},
                             timeout=spec.timeout_seconds, backend=capability["backend"])
    if result.get("returncode") == 77:
        raise IsolationUnavailable("candidate isolation unavailable")
    outputs = result.get("payload", {}).get("results", [])
    valid_output = isinstance(outputs, list) and len(outputs) == len(spec.cases) and result.get("returncode") == 0
    failures = []
    for index, case in enumerate(spec.cases):
        expected = {"raises": case["raises"]} if "raises" in case else {"value": case["expected"]}
        # JSON type identity prevents bool(1) from passing a bool contract.
        observed = outputs[index] if valid_output else None
        if json.dumps(expected, sort_keys=True) != json.dumps(observed, sort_keys=True):
            failures.append(index)
    return {"name": spec.name, "contract_sha256": spec.identity(), "target_sha256": hashlib.sha256(source).hexdigest(),
            "passed": not failures, "cases": len(spec.cases), "failed_cases": failures,
            "process": {k: v for k, v in result.items() if k != "payload"},
            "native_required": spec.native_required}


def validate_candidate(reference_root: Path, candidate_root: Path, specs: tuple[ValidationSpec, ...]) -> dict:
    """Baseline must really fail; candidates must pass every trusted case."""
    capability = isolation_capability()
    if not capability["l2_available"]:
        raise IsolationUnavailable("L2 unavailable: kernel confinement did not pass")
    if not specs:
        raise ValueError("repair requires at least one trusted validation contract")
    baseline = [run_contract(reference_root, spec) for spec in specs]
    candidate = [run_contract(candidate_root, spec) for spec in specs]
    baseline_reproduced = any(not row["passed"] and row["process"].get("returncode") == 0 for row in baseline)
    return {"schema_version": 1, "passed": baseline_reproduced and all(row["passed"] for row in candidate),
            "baseline_reproduced": baseline_reproduced, "baseline": baseline, "candidate": candidate,
            "isolation": capability, "requires_native_acceptance": any(spec.native_required for spec in specs),
            "scope": "component contracts; supervisor must perform stage-boundary and native acceptance"}


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit(2)
    raise SystemExit(_worker())
