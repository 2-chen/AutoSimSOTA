"""Read-only environment catalog, isolated conda clones and verified wheel reuse.

Only the controller publishes shared objects. Workloads see the store read-only and
install into their own prefix. Catalog entries are proposals, never readiness verdicts.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .common import atomic_json, atomic_text, digest, now, object_digest, read_json

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "autoresearch_cache" / "environments"
PROFILES = (
    {"id": "python-toolchain", "purpose": "Python, package manager and native build tools"},
    {"id": "torch-cuda", "purpose": "PyTorch/CUDA; must verify a real GPU operation"},
    {"id": "mujoco-headless", "purpose": "MuJoCo and headless rendering; reset/step/frame probe required"},
    {"id": "sapien-headless", "purpose": "SAPIEN rendering/physics; version and asset compatibility required"},
)
CORE = {"torch", "torchvision", "numpy", "mujoco", "sapien", "gymnasium", "cmake", "setuptools"}
MAX_STORE_BYTES = 64 * 1024**3


def _bounded_digest(path: Path, deadline: float) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("environment cache integrity deadline reached")
            data = stream.read(8 * 1024**2)
            if not data:
                break
            value.update(data)
    return value.hexdigest()


def allocated_bytes(root: Path) -> int:
    paths = [*(root / "wheels").glob("*/manifest.json"),
             *(root / "snapshots").glob("*/reservation.json")]
    return sum(int(read_json(path).get("bytes") or 0) for path in paths if not path.is_symlink())


def configure(output: Path, root: Path | None = None, *, enabled: bool = True,
              source_repository: Path | None = None) -> dict[str, Any]:
    output = Path(output).resolve()
    path = output / "environment_pool.json"
    if path.is_file():
        held = read_json(path)
        if root is not None and str(Path(root).resolve()) != held.get("root"):
            raise ValueError("environment pool changed; use a fresh run")
        if held.get("enabled") != enabled:
            raise ValueError("environment pool policy changed; use a fresh run")
        return held
    root = Path(root or DEFAULT_ROOT).resolve()
    if root == Path("/") or root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError("environment store must be separate from run output")
    if source_repository is not None:
        source = Path(source_repository).resolve()
        if root.is_relative_to(source) or source.is_relative_to(root):
            raise ValueError("environment store must not modify or contain the source repository")
    if enabled:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name in ("wheels", "snapshots", "registered"):
            (root / name).mkdir(exist_ok=True)
    result = {"schema_version": 1, "enabled": enabled, "root": str(root),
              "created_at": now(), "policy": "read-only bases; isolated installs; probes required"}
    atomic_json(path, result)
    return result


def store_for(output: Path) -> Path | None:
    path = Path(output) / "environment_pool.json"
    if not path.is_file() or path.is_symlink():
        return None
    held = read_json(path)
    if not held.get("enabled"):
        return None
    root = Path(held["root"])
    if (root.is_symlink() or not root.is_dir() or root == Path("/") or
            root.resolve().is_relative_to(Path(output).resolve()) or
            Path(output).resolve().is_relative_to(root.resolve())):
        raise ValueError("unsafe environment store")
    return root.resolve()


@contextmanager
def locked(root: Path, *, timeout: float = 10):
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root / "pool.lock", flags, 0o600)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("environment store busy; skip optional reuse/publication")
                time.sleep(min(.05, max(0, deadline-time.monotonic())))
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def describe(prefix: Path) -> dict[str, Any]:
    """Read package metadata without executing discovered interpreters or site hooks."""
    prefix = Path(prefix).absolute()
    python = prefix / "bin" / "python"
    if not python.is_file():
        raise ValueError("environment has no Python interpreter")
    version = None
    conda_owned_metadata = set()
    for path in (prefix / "conda-meta").glob("*.json"):
        info = read_json(path)
        if info.get("name") == "python":
            version = ".".join(str(info["version"]).split(".")[:2])
        conda_owned_metadata.update(str(file) for file in info.get("files", [])
                                    if str(file).endswith("/direct_url.json"))
    configuration = prefix / "pyvenv.cfg"
    if version is None and configuration.is_file():
        match = re.search(r"(?m)^version(?:_info)?\s*=\s*(\d+\.\d+)",
                          configuration.read_text())
        if match:
            version = match.group(1)
    sites = ([(prefix / "lib" / f"python{version}" / "site-packages")]
             if version else sorted((prefix / "lib").glob("python*/site-packages")))
    sites = [site for site in sites if site.is_dir()]
    packages: dict[str, str] = {}
    unsafe = []
    for site in sites:
        if not site.resolve().is_relative_to(prefix.resolve()):
            unsafe.append("external site-packages directory")
            continue
        for distribution in importlib.metadata.distributions(path=[str(site)]):
            name = str(distribution.metadata.get("Name") or "")
            if name:
                packages[name.lower().replace("_", "-")] = distribution.version
            direct = distribution.read_text("direct_url.json")
            if direct:
                try:
                    info = json.loads(direct)
                    if info.get("dir_info", {}).get("editable"):
                        unsafe.append("editable package " + name)
                    elif info.get("url", "").startswith("file:") and "dir_info" in info:
                        # Conda-built packages often retain local build provenance. That
                        # is not a live reference to the benchmark checkout.
                        metadata_path = getattr(distribution, "_path", None)
                        direct_path = Path(metadata_path) / "direct_url.json" if metadata_path else None
                        owned = (direct_path is not None and direct_path.is_relative_to(prefix) and
                                 direct_path.relative_to(prefix).as_posix() in conda_owned_metadata)
                        if not owned:
                            unsafe.append("repository-local installed package " + name)
                except (ValueError, TypeError):
                    unsafe.append("invalid direct_url metadata")
        for file in site.glob("*.pth"):
            for line in file.read_text(errors="replace").splitlines():
                line = line.strip()
                if "__editable__" in line:
                    unsafe.append("editable site hook")
                elif line and not line.startswith(("#", "import ", "import\t")):
                    target = (site / line).resolve()
                    if not target.is_relative_to(prefix.resolve()):
                        unsafe.append("external site-packages reference")
    kind = "conda" if (prefix / "conda-meta").is_dir() else "venv"
    if kind != "conda":
        unsafe.append("venv cannot be relocated by copying; needs wheel reconstruction")
    version = version or next((site.parent.name.removeprefix("python") for site in sites), "unknown")
    fingerprint = object_digest({"kind": kind, "python": version, "packages": packages,
                                 "unsafe": sorted(set(unsafe))})
    return {"id": object_digest([str(prefix), fingerprint])[:32], "prefix": str(prefix),
            "kind": kind, "python": version, "packages": packages,
            "fingerprint": fingerprint, "cloneable": not unsafe,
            "limitations": sorted(set(unsafe)), "readiness": "unverified"}


def tree_digest(prefix: Path, *, timeout: float = 600) -> str:
    """Hash cached environment bytes, not just package versions; never follow symlinks."""
    checksum = hashlib.sha256()
    deadline = time.monotonic() + timeout
    count = 0
    for directory, names, files in os.walk(prefix, followlinks=False):
        names.sort()
        for name in sorted([*names, *files]):
            path = Path(directory) / name
            count += 1
            if count > 300_000 or time.monotonic() >= deadline:
                raise ValueError("environment integrity scan exceeds its bound")
            if path.is_symlink():
                if not path.resolve().is_relative_to(prefix.resolve()):
                    raise ValueError("environment contains an external symlink")
                value = "link:" + os.readlink(path)
            elif path.is_file():
                value = _bounded_digest(path, deadline)
            else:
                value = "directory"
            checksum.update(json.dumps([path.relative_to(prefix).as_posix(), value]).encode())
    return checksum.hexdigest()


def discover(root: Path, *, prefixes: list[Path] | None = None,
             timeout: float = 15) -> list[dict[str, Any]]:
    candidates = list(prefixes or [])
    registered = []
    for registry in sorted((root / "registered").glob("*.json"))[:32]:
        if not registry.is_symlink():
            try:
                registered.append(Path(read_json(registry)["prefix"]))
            except (OSError, ValueError, KeyError, TypeError):
                continue
    if prefixes is None:
        # Explicit operator choices must not disappear behind the discovery cap.
        candidates.extend(registered)
        candidates.extend([Path(sys.prefix), Path(sys.base_prefix)])
        conda = shutil.which("conda")
        if conda:
            try:
                result = subprocess.run([conda, "env", "list", "--json"],
                    capture_output=True, text=True, timeout=timeout, check=False)
                if result.returncode == 0:
                    candidates += [Path(value) for value in json.loads(result.stdout).get("envs", [])]
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        # App-owned run prefixes are another source of reusable partial dependencies;
        # their existence is not a claim that the repository or simulator passed.
        runs = DEFAULT_ROOT.parents[1] / "autoresearch_runs"
        if runs.is_dir():
            local = [path.parent.parent for path in runs.glob("*/env/bin/python")]
            candidates += sorted(local, key=lambda path: path.stat().st_mtime, reverse=True)[:16]
    else:
        candidates.extend(registered)
    rows = []
    seen = set()
    for prefix in candidates[:32]:
        if str(prefix) in seen:
            continue
        seen.add(str(prefix))
        try:
            rows.append(describe(prefix))
        except (OSError, ValueError, TypeError):
            continue
    return rows


def dependency_key(manifests: dict[str, str], machine: dict[str, Any]) -> str:
    return object_digest({"dependencies": manifests, "platform": {
        key: machine.get(key) for key in ("os", "release", "gpus", "cuda_toolkit")}})


def failed_selection(output: Path, *, base_id: str, error: Exception,
                     seconds: float, phase: str = "before_execution",
                     failure_kind: str = "cache_validation",
                     prior_attempt: dict[str, Any] | None = None) -> dict[str, Any]:
    """Seal validation failures without rewriting the original command receipt."""
    from .evidence_store import capture_attempt_evidence
    identity = uuid.uuid4().hex
    log = output / "provision_attempts" / f"{identity}.log"
    text = f"clone base {base_id}\n{type(error).__name__}: {error}\n"
    if prior_attempt:
        text += "Original command evidence: " + str(prior_attempt.get("evidence_id")) + "\n"
    atomic_text(log, text)
    ref = f"provision_attempts/{identity}.json"
    receipt = {"command": f"clone_base:{base_id}", "phase": phase,
        "ok": False, "returncode": None, "seconds": seconds,
        "failure_kind": failure_kind, "excerpt": text,
        "prior_evidence_id": (prior_attempt or {}).get("evidence_id"),
        **capture_attempt_evidence(output, attempt_id=identity, log=log, receipt_ref=ref,
            status="failed", returncode=None, termination_reason=failure_kind)}
    atomic_json(output / ref, receipt)
    path = output / "provision_progress.json"
    previous = read_json(path) if path.is_file() else {}
    atomic_json(path, {"status": "building", "updated_at": now(),
        "attempts": [*(previous.get("attempts") or []), receipt]})
    return receipt


def catalog(output: Path, manifests: dict[str, str], machine: dict[str, Any]) -> list[dict[str, Any]]:
    root = store_for(output)
    if root is None:
        return []
    key = dependency_key(manifests, machine)
    rows = [row for row in discover(root)
            if not Path(row["prefix"]).resolve().is_relative_to(Path(output).resolve())]
    for path in sorted((root / "snapshots").glob("*/manifest.json"))[:32]:
        if path.is_symlink() or (path.parent / "quarantined.json").exists():
            continue
        try:
            cached = read_json(path)
            row = describe(path.parent / "env")
            if row["fingerprint"] != cached["fingerprint"]:
                continue
            rows.append({**row, "origin": "verified_snapshot",
                         "tree_sha256": cached["tree_sha256"],
                         "dependency_match": cached["dependency_key"] == key,
                         "readiness": "requires_current_repository_probes"})
        except (OSError, ValueError, KeyError):
            continue
    atomic_json(Path(output) / "environment_catalog.json", {
        "schema_version": 1, "created_at": now(), "profiles": list(PROFILES),
        "candidates": rows, "note": "Local references only; models receive opaque IDs."})
    return rows


def short_catalog(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**{key: row.get(key) for key in ("id", "python", "kind", "cloneable",
             "limitations", "readiness", "origin", "dependency_match")},
             "core_packages": {name: version for name, version in row["packages"].items()
                               if name in CORE}, "package_count": len(row["packages"])}
            for row in sorted(rows, key=lambda r: (not r.get("dependency_match"),
                not r.get("cloneable"), -len(CORE.intersection(r["packages"]))))[:12]]


def clone(source: Path, *, destination: Path, output: Path, repo: Path,
          timeout: float = 1800, expected_tree: str | None = None) -> dict[str, Any]:
    """Conda performs prefix relocation; --copy avoids writable shared hardlinks."""
    import shlex
    from .provision import run, conda_executable
    from .common import run_local_environment
    deadline = time.monotonic() + timeout
    row = describe(source)
    if not row["cloneable"]:
        raise ValueError("unsafe base environment: " + "; ".join(row["limitations"]))
    if expected_tree is not None and tree_digest(source, timeout=timeout) != expected_tree:
        root = store_for(output)
        if root is not None and source.parent.resolve().is_relative_to(root / "snapshots"):
            atomic_json(source.parent / "quarantined.json", {
                "reason": "cached environment bytes changed", "at": now()})
        raise ValueError("cached environment bytes changed; refuse reuse (retained for inspection)")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("refusing to overwrite an existing environment prefix")
    # Conda clone installs recorded package URLs from its package caches. An empty
    # run-local cache alone would make an otherwise valid offline clone fail.
    caches = set()
    for metadata in (source / "conda-meta").glob("*.json"):
        info = read_json(metadata)
        linked = (info.get("link") or {}).get("source")
        if linked and Path(linked).is_dir():
            caches.add(Path(linked).resolve().parent)
    command = shlex.join([conda_executable(), "create", "--yes", "--offline", "--copy",
                         "--clone", str(source), "--prefix", str(destination)])
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("environment clone deadline reached before execution")
    environment = run_local_environment(output)
    if caches:
        # Unlike CONDA_ENVS_PATH, conda parses this sequence with commas,
        # not the platform PATH separator. Colons silently create one bogus
        # cache path and cause offline clones to request already-cached URLs.
        environment["CONDA_PKGS_DIRS"] += "," + ",".join(map(str, sorted(caches)))
    result = run(command, env=environment, cwd=repo,
                 timeout=remaining, output=Path(output) / "build.log",
                 read_only_roots=(source, *sorted(caches)))
    if result["ok"]:
        copied = describe(destination)
        if copied["packages"] != row["packages"] or not copied["cloneable"]:
            return failed_selection(output, base_id=row["id"],
                error=ValueError("cloned package inventory or path isolation differs"),
                seconds=timeout - (deadline - time.monotonic()), phase="after_execution",
                failure_kind="clone_verification", prior_attempt=result)
        for directory, names, files in os.walk(destination, followlinks=False):
            for name in [*names, *files]:
                path = Path(directory) / name
                if path.is_symlink() and not path.resolve().is_relative_to(destination.resolve()):
                    return failed_selection(output, base_id=row["id"],
                        error=ValueError("cloned environment retains an external symlink"),
                        seconds=timeout - (deadline - time.monotonic()), phase="after_execution",
                        failure_kind="clone_verification", prior_attempt=result)
    return result


def _wheel_name(path: Path) -> str | None:
    try:
        with zipfile.ZipFile(path) as archive:
            wheels = [name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")]
            if len(wheels) != 1 or archive.getinfo(wheels[0]).file_size > 65536:
                return None
            folder = wheels[0].split("/")[0].removesuffix(".dist-info")
            tags = re.findall(r"^Tag: ([A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+)$",
                              archive.read(wheels[0]).decode(), re.M)
            if not tags or not re.fullmatch(r"[A-Za-z0-9_.+!]+-[A-Za-z0-9_.+!]+", folder):
                return None
            return f"{folder}-{tags[0]}.whl"
    except (OSError, ValueError, zipfile.BadZipFile, UnicodeError):
        return None


def publish_wheels(output: Path, *, max_bytes: int = 16 * 1024**3,
                   timeout: float = 60) -> dict[str, Any]:
    root = store_for(output)
    if root is None:
        return {"status": "disabled"}
    cache = Path(output) / "home" / ".cache" / "pip"
    added, total = [], 0
    deadline = time.monotonic() + timeout
    index_path = Path(output) / "package_cache_export_index.json"
    index = read_json(index_path) if index_path.is_file() else {}
    with locked(root, timeout=min(10, timeout)):
        capacity = MAX_STORE_BYTES - allocated_bytes(root)
        for path in sorted(cache.rglob("*")):
            if time.monotonic() >= deadline:
                break
            if path.is_symlink() or not path.is_file() or path.suffix not in {".whl", ".body"}:
                continue
            size = path.stat().st_size
            stamp = [size, path.stat().st_mtime_ns]
            relative = path.relative_to(cache).as_posix()
            if index.get(relative) == stamp:
                continue
            if size > max_bytes or total + size > min(max_bytes, capacity):
                continue
            name = _wheel_name(path)
            if name is None:
                continue
            sha = _bounded_digest(path, deadline)
            target = root / "wheels" / sha / name
            if target.exists():
                if _bounded_digest(target, deadline) != sha:
                    raise ValueError("shared wheel integrity check failed")
                index[relative] = stamp
                continue
            if shutil.disk_usage(root).free < size * 1.2:
                continue
            target.parent.mkdir(mode=0o700, exist_ok=True)
            temporary = target.parent / ("pending-" + uuid.uuid4().hex)
            try:
                with path.open("rb") as source, temporary.open("xb") as target_file:
                    while True:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("wheel publication deadline reached")
                        chunk = source.read(8 * 1024**2)
                        if not chunk:
                            break
                        target_file.write(chunk)
                if _bounded_digest(temporary, deadline) != sha:
                    raise ValueError("wheel changed during publication")
            except (OSError, ValueError):
                # This is our unpublished staging object, not source/user data.
                temporary.unlink(missing_ok=True)
                raise
            os.replace(temporary, target)
            target.chmod(0o444)
            atomic_json(target.parent / "manifest.json", {
                "sha256": sha, "filename": name, "bytes": size,
                "created_at": now(), "origin_run": Path(output).name,
                "trust": "same-operator verified provisioning; not an upstream signature"})
            total += size
            added.append(name)
            index[relative] = stamp
    atomic_json(index_path, index)
    result = {"status": "published", "wheels": added, "bytes": total}
    atomic_json(Path(output) / "package_cache_publication.json", result)
    return result


def wheel_links(output: Path) -> list[str]:
    root = store_for(output)
    if root is None:
        return []
    links = []
    for manifest in sorted((root / "wheels").glob("*/manifest.json"))[:256]:
        if manifest.is_symlink():
            continue
        metadata = read_json(manifest)
        path = manifest.parent / str(metadata["filename"])
        if (path.is_file() and not path.is_symlink() and path.parent == manifest.parent and
                digest(path) == metadata["sha256"]):
            links.append(path.as_uri())
    return links


def publish_snapshot(output: Path, *, interpreter: Path,
                     manifests: dict[str, str], machine: dict[str, Any],
                     timeout: float = 1800) -> dict[str, Any]:
    started = time.monotonic()
    root = store_for(output)
    if root is None:
        return {"status": "disabled"}
    prefix = interpreter.parent.parent
    row = describe(prefix)
    if not row["cloneable"]:
        return {"status": "not_cacheable", "limitations": row["limitations"]}
    key = dependency_key(manifests, machine)
    identity = object_digest([key, row["fingerprint"]])
    with locked(root, timeout=min(10, timeout)):
        destination = root / "snapshots" / identity
        if (destination / "manifest.json").is_file():
            return {"status": "already_present", "id": identity}
        if destination.exists():
            return {"status": "incomplete_snapshot", "id": identity}
        estimated = 0
        for directory, names, files in os.walk(prefix, followlinks=False):
            for name in files:
                path = Path(directory) / name
                if not path.is_symlink():
                    estimated += path.stat().st_size
            if time.monotonic() - started >= timeout:
                return {"status": "deadline_reached"}
        if (estimated > MAX_STORE_BYTES - allocated_bytes(root) or
                shutil.disk_usage(root).free < estimated * 1.2 + 2 * 1024**3):
            return {"status": "capacity_skipped", "estimated_bytes": estimated}
        destination.mkdir()
        atomic_json(destination / "reservation.json", {"bytes": estimated,
            "note": "incomplete copies remain counted; no automatic deletion"})
        result = clone(prefix, destination=destination / "env", output=destination,
                       repo=destination, timeout=max(.001, timeout - (time.monotonic()-started)))
        if not result["ok"]:
            return {"status": "clone_failed", "id": identity, "evidence": result}
        copied = describe(destination / "env")
        checksum = tree_digest(destination / "env", timeout=max(.001, timeout -
                               (time.monotonic() - started)))
        atomic_json(destination / "manifest.json", {
            "schema_version": 1, "dependency_key": key,
            "fingerprint": copied["fingerprint"], "created_at": now(),
            "tree_sha256": checksum,
            "origin_run": Path(output).name,
            "note": "verified provisioning snapshot; re-probe every consumer"})
    return {"status": "published", "id": identity}


def main(argv: list[str] | None = None) -> int:
    """Operator catalog/registration; never silently download a simulator distribution."""
    import argparse
    parser = argparse.ArgumentParser(prog="autosim environments")
    parser.add_argument("operation", choices=("list", "register"))
    parser.add_argument("--store", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--label", default="operator-base")
    args = parser.parse_args(argv)
    root = args.store.resolve()
    if args.operation == "list":
        print(json.dumps({"profiles": list(PROFILES), "candidates": short_catalog(discover(root))},
                         ensure_ascii=False, indent=2))
        return 0
    if args.prefix is None:
        parser.error("register requires --prefix; a profile is not a downloaded environment")
    row = describe(args.prefix)
    if not row["cloneable"]:
        parser.error("base is not independently cloneable: " + "; ".join(row["limitations"]))
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    (root / "registered").mkdir(exist_ok=True)
    with locked(root):
        atomic_json(root / "registered" / (row["id"] + ".json"), {
            "prefix": row["prefix"], "label": args.label, "fingerprint": row["fingerprint"],
            "created_at": now(), "note": "live read-only source; each run clones and probes"})
    print(json.dumps({"id": row["id"], "status": "registered", "readiness": "unverified"}))
    return 0
