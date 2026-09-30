"""Agent-authored native previews, explicitly separate from scored policy demos."""
from pathlib import Path
import shlex
import time
import uuid
from .common import atomic_json, atomic_text, digest, now, read_json, run_local_environment


def capture(output: Path, repo: Path, *, code: str, purpose: str, source_refs: list[str],
            timeout: float, compute=None):
    from .native_context import load_context
    from . import provision, media_manifest
    if not isinstance(code, str) or not 1 <= len(code.encode()) <= 32768:
        raise ValueError("preview code must be bounded")
    if not isinstance(purpose, str) or not purpose.strip() or len(purpose) > 1000:
        raise ValueError("preview needs a bounded purpose")
    if not isinstance(source_refs, list) or not 1 <= len(source_refs) <= 10:
        raise ValueError("preview needs native source references")
    for relative in source_refs:
        if not isinstance(relative, str) or not 1 <= len(relative) <= 512:
            raise ValueError("preview source reference must be a bounded path")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not (repo / path).is_file() or not (repo / path).resolve().is_relative_to(repo.resolve()):
            raise ValueError("preview source reference escaped checkout")
    native = load_context(output, repo)
    identity = uuid.uuid4().hex
    directory = output / "environment_demos" / identity
    if directory.parent.is_symlink():
        raise ValueError("preview directory is unsafe")
    directory.mkdir(parents=True)
    script = directory / "demo.py"
    atomic_text(script, code)
    started = time.time()
    env = run_local_environment(output)
    env = {k: v for k, v in env.items() if not k.upper().endswith(("_KEY", "_TOKEN", "_SECRET", "_PASSWORD"))}
    # Script can write only its own artifacts/scratch, not budget, evidence, env or code.
    home = directory / "home"; home.mkdir()
    env.update(HOME=native["paths"].get("HOME", str(home)), XDG_CACHE_HOME=str(home / ".cache"),
               AUTOSIM_DEMO_DIR=str(directory), PYTHONPATH=str(repo))
    for key, value in native["paths"].items():
        if key != "HOME": env[key] = value
    row = provision.run(f"{shlex.quote(native['interpreter'])} {shlex.quote(str(script))}",
        env=env, cwd=repo, timeout=timeout, output=output / "environment_demo.log",
        compute=compute, writable_paths=(Path("/tmp"), directory))
    media = media_manifest.capture_attempt(output, directory, since=started)
    decoded = []
    for item in media[:3]:
        if item.get("sha256"):
            preview = media_manifest._preview(output, item["path"], item["sha256"])
            if preview.get("status") == "decoded_frame":
                decoded.append({**item, "preview": preview})
    result = {"id": identity, "status": "recorded_unscored" if row["ok"] and decoded else "failed",
              "purpose": purpose, "source_refs": source_refs, "context_identity": native["identity"],
              "script_sha256": digest(script), "evidence_id": row["evidence_id"],
              "evidence_ref": row["evidence_ref"], "media": decoded, "at": now(),
              "scope": "environment preview; policy/task identity not audited; not a measurement"}
    atomic_json(directory / "receipt.json", result)
    atomic_json(output / "environment_demo.json", result)
    return result
