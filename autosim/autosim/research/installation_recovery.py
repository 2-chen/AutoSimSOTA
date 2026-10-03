"""Bounded local resource discovery and evidence-based environment boundaries.

No benchmark-specific installers: agents choose commands and preserve native probes.
Cache metadata is a recovery lead, not proof of compatibility or simulator readiness.
"""
from pathlib import Path
import json
import time
import zipfile


def wheel_import_hints(path: Path) -> list[str]:
    """Read packaging metadata only; distribution names are not Python import names."""
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > 100000:
                return []
            top = [item for item in infos if item.filename.endswith(".dist-info/top_level.txt")]
            if len(top) == 1 and top[0].file_size <= 8192:
                names = archive.read(top[0]).decode("utf-8", "replace").splitlines()
            else:
                names = [item.filename.split("/")[0].removesuffix(".py") for item in infos
                         if "/" in item.filename or item.filename.endswith(".py")]
            return sorted({name for name in names if name.isidentifier()})[:64]
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError):
        return []

from .common import atomic_json, now, object_digest, read_json, sanitize_model_payload


def inventory(output: Path, *, timeout: float = 5, limit: int = 128, persist: bool = True) -> dict:
    from .environment_pool import _wheel_name, short_catalog
    output = Path(output).resolve()
    cache = output / "home/.cache/pip"
    deadline = time.monotonic() + timeout
    wheels, complete = [], True
    if cache.is_dir() and not cache.is_symlink() and cache.resolve().is_relative_to(output):
        for path in cache.rglob("*"):
            if time.monotonic() >= deadline or len(wheels) >= limit:
                complete = False
                break
            if (path.is_symlink() or not path.is_file() or
                    path.suffix not in {".body", ".whl"} or
                    not path.resolve().is_relative_to(output)):
                continue
            name = _wheel_name(path)
            if name:
                wheels.append({"filename": name, "bytes": path.stat().st_size,
                    "import_name_hints": wheel_import_hints(path),
                    "cache_ref": path.relative_to(output).as_posix(),
                    "mtime_ns": path.stat().st_mtime_ns,
                    "suggested_destination": "work/recovered_wheels/" + name,
                    "verification": "wheel metadata only; verify bytes/tags before installing"})
    catalog_path = output / "environment_catalog.json"
    catalog = read_json(catalog_path) if catalog_path.is_file() and not catalog_path.is_symlink() else {}
    candidates = short_catalog(catalog.get("candidates") or [])
    if cache.is_symlink() or (cache.exists() and not cache.resolve().is_relative_to(output)):
        complete = False
    resources = {"local_wheels": sorted(wheels, key=lambda row: row["filename"]),
                 "environment_candidates": candidates, "scan_complete": complete}
    result = {**resources, "digest": object_digest(resources),
        "guidance": "Paths are relative to the RUN ROOT, not checkout. An index outage or "
            "pip's No matching distribution after a connection failure does not prove a "
            "package absent. Inspect these artifacts and compatible isolated environment "
            "clones. Choose a real repair, not stubs. For offline reuse verify wheel contents "
            "and tags, copy a .body to its wheel filename under the suggested destination, "
            "then install via the run prefix. Use --no-index only when dependencies are "
            "available. Separate vendor-only package acquisition from ordinary dependencies, "
            "using bounded timeouts/retries; do not apply an unreachable extra index to every "
            "dependency. Preserve source constraints and native capability probes."}
    if persist:
        snapshot = output / 'installation_inventory_snapshots' / (result['digest'] + '.json')
        if snapshot.parent.is_symlink() or snapshot.is_symlink():
            raise ValueError('unsafe installation inventory snapshot')
        if not snapshot.exists():
            atomic_json(snapshot, result)
        path = output / "installation_recovery_inventory.json"
        previous = read_json(path) if path.is_file() else {}
        result["updated_at"] = (previous.get("updated_at") or now()) if previous.get("digest") == result["digest"] else now()
        if result != previous:
            atomic_json(path, result)
    return result


def review_boundary(client, *, resources: dict, reason: str, assessment: dict | None,
                    failure: dict) -> dict:
    """A resource-bearing failure needs explicit assessment and a read-only review."""
    assessment = assessment or {}
    required = ("local_artifacts", "environment_reuse", "alternative_sources")
    faults = []
    if assessment.get("inventory_digest") != resources["digest"]:
        faults.append("inventory_digest does not match supplied snapshot " + resources['digest'])
    faults.extend(key + " must be a nonempty evidence-backed string" for key in required
                  if not isinstance(assessment.get(key), str) or not assessment[key].strip())
    if faults:
        raise ValueError("unbuildable requires resource_assessment with current inventory_digest "
                         "and evidence-backed local_artifacts, environment_reuse, alternative_sources; "
                         "a failed online probe does not rule out downloaded wheels; invalid fields: " + "; ".join(faults))
    from .agent_client import role_scope
    with role_scope(client, "objective"):
        content, _ = client.chat_with_metadata(
            "Review an ENVIRONMENT RESOURCE BOUNDARY independently, read-only. Return JSON "
            "{approved:bool, reason:string}. Reject unsupported impossibility claims. "
            "Use read_evidence for the supplied failure ID. Check whether local downloaded "
            "wheels, isolated clones or documented alternative sources remain untested. "
            "Metadata is not readiness, but an online timeout is not package absence. "
            "A matching cached wheel must be tested or ruled out with concrete evidence. "
            "Incomplete inventory cannot establish absence. Never weaken native probes.",
            json.dumps(sanitize_model_payload({"reason": reason, "resource_assessment": assessment,
                        "resources": resources, "failure": failure}), ensure_ascii=False),
            max_tokens=1500, timeout=180, read_only=True)
    from .provision import _object
    review = _object(content)
    if review.get("approved") is not True or not str(review.get("reason") or "").strip():
        raise ValueError("resource boundary not independently approved: " + str(review.get("reason")))
    return {"approved": True, "reason": review["reason"], "inventory_digest": resources["digest"],
            "failure_evidence_id": failure.get("evidence_id")}
