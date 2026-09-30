"""Non-provider startup probes; no model requests or benchmark code are executed."""
import shutil
import socket
import subprocess


def check_runtime():
    checks = []
    try:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
        checks.append({"name": "loopback_gateway", "ok": True})
    except OSError as exc:
        checks.append({"name": "loopback_gateway", "ok": False,
                       "error": f"{type(exc).__name__}: {exc}", "errno": exc.errno})
    for name, command, required in (
        ("coding_agent_cli", ["claude", "--version"], True),
        ("process_isolation", ["bwrap", "--ro-bind", "/", "/", "--unshare-pid",
                               "--proc", "/proc", "--dev-bind", "/dev", "/dev",
                               "--", "/usr/bin/true"], True),
        ("gpu_discovery", ["nvidia-smi", "--query-gpu=name,uuid",
                           "--format=csv,noheader"], False),
    ):
        try:
            binary = shutil.which(command[0])
            if not binary:
                raise FileNotFoundError(command[0])
            result = subprocess.run([binary, *command[1:]], capture_output=True,
                                    text=True, timeout=20)
            checks.append({"name": name, "ok": result.returncode == 0,
                           "required": required, "returncode": result.returncode,
                           "detail": (result.stdout + result.stderr)[-1200:]})
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append({"name": name, "ok": False, "required": required,
                           "error": f"{type(exc).__name__}: {exc}"[:1200]})
    return {"status": "ready" if all(row["ok"] for row in checks
                                      if row.get("required", True)) else "infrastructure_blocked",
            "checks": checks,
            "note": "GPU discovery is informational; native CUDA/rendering probes remain necessary."}
