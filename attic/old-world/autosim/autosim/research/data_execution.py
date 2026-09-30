"""Content-addressed data preparation and explicitly validated audit kernels."""
from __future__ import annotations
import json
import os
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path

from .common import atomic_json, digest, immutable_json, object_digest, read_json


def small_dataset(source: Path, destination: Path, episodes: int = 2) -> Path:
    source, destination = Path(source), Path(destination)
    info = read_json(source / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("bounded dataset fixture currently supports LeRobot v2.1")
    rows = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines() if line]
    rows = rows[:episodes]
    if not rows or [r["episode_index"] for r in rows] != list(range(len(rows))):
        raise ValueError("fixture requires a contiguous prefix of episode identities")
    identities = {int(r["episode_index"]) for r in rows}
    manifest = {"source": str(source), "info_digest": digest(source / "meta/info.json"),
                "episodes": sorted(identities)}
    immutable_json(destination / "fixture.json", manifest)
    if (destination / "ready.json").exists():
        return destination
    (destination / "meta").mkdir(parents=True, exist_ok=True)
    for path in (source / "meta").glob("*"):
        if not path.is_file() or path.name in {"info.json", "episodes.jsonl", "episodes_stats.jsonl"}:
            continue
        shutil.copy2(path, destination / "meta" / path.name)
    for folder, suffix in (("data", ".parquet"), ("videos", ".mp4")):
        for path in (source / folder).rglob("episode_*" + suffix):
            if int(path.stem.split("_")[-1]) not in identities:
                continue
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copy2(path, target)
    (destination / "meta/episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = source / "meta/episodes_stats.jsonl"
    if stats.exists():
        selected = [line for line in stats.read_text().splitlines()
                    if line and json.loads(line)["episode_index"] in identities]
        (destination / "meta/episodes_stats.jsonl").write_text("\n".join(selected) + "\n")
    info.update(total_episodes=len(rows), total_frames=sum(r["length"] for r in rows),
                total_videos=len(list((destination / "videos").rglob("*.mp4"))),
                total_chunks=1, splits={"train": f"0:{len(rows)}"})
    atomic_json(destination / "meta/info.json", info)
    atomic_json(destination / "ready.json", {"manifest_digest": object_digest(manifest),
                                             "files": len(list(destination.rglob("*.parquet")))})
    return destination


def activate_audit_kernel(registry: Path) -> dict:
    from autosim import robosyn_data
    from .patch_validation import checked_function
    from .compute_codegen import extract_function
    record = read_json(registry)
    path = Path(record["candidate"])
    if not record["verdict"]["passed"] or digest(path) != record["candidate_sha256"]:
        raise ValueError("audit patch receipt no longer matches the candidate")
    if digest(Path(record["source_path"])) != record["base_sha256"]:
        raise ValueError("audit source changed since patch validation")
    if extract_function(Path(robosyn_data.__file__), "_padding_audit") != extract_function(
            Path(record["source_path"]), "_padding_audit"):
        raise ValueError("validated reference differs from the runtime audit function")
    source = path.read_text()
    checked_function(source)
    reference = robosyn_data._padding_audit
    activation = {"patch_id": record["patch_id"], "candidate_sha256": record["candidate_sha256"],
                  "execution":"bounded_subprocess","status":"active_for_call"}
    def isolated_audit(lengths, chunk_size=50):
        from .compute_codegen import measure
        values = list(lengths)
        try:
            return measure(source, [{"lengths":values,"chunk_size":chunk_size}])["outputs"][0]
        except Exception as exc:
            activation.update(status="rolled_back_for_call", reason=type(exc).__name__)
            atomic_json(registry.parent / "rollbacks" / (object_digest(activation)+".json"), activation)
            return reference(values, chunk_size)
    robosyn_data._padding_audit = isolated_audit
    return activation


_KERNEL_LOCK = threading.RLock()


@contextmanager
def audit_kernel_context(registry: Path | None):
    from autosim import robosyn_data
    # Unpatched audits must also wait for an active scoped override to restore the
    # reference; otherwise another thread would silently execute that override.
    with _KERNEL_LOCK:
        original = robosyn_data._padding_audit
        try:
            yield activate_audit_kernel(registry) if registry else None
        finally:
            robosyn_data._padding_audit = original
