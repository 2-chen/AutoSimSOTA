"""Content fingerprints for the exact training bytes, not just episode counts."""
from pathlib import Path

from .common import atomic_json, digest, immutable_json, object_digest, read_json


def fingerprint_dataset(root: Path, store: Path) -> dict:
    root = root.absolute()
    files = sorted(path for folder in ("meta", "data", "videos")
                   for path in (root / folder).rglob("*") if path.is_file())
    if not (root / "meta/info.json").is_file() or not any(path.suffix == ".parquet" for path in files):
        raise ValueError(f"dataset lacks required metadata/parquet: {root}")
    cache_path = store / (object_digest({"root": str(root), "schema_version": 2}) + ".cache.json")
    relative = [str(file.relative_to(root)) for file in files]
    if cache_path.is_file():
        index = read_json(cache_path)
        path = Path(index["manifest"])
        cached = read_json(path) if path.is_file() else {}
        cached_files = cached.get("files", {})
        if set(cached_files) == set(relative) and all(
            cached_files[name].get("bytes") == file.stat().st_size
            and cached_files[name].get("mtime_ns") == file.stat().st_mtime_ns
            for name, file in zip(relative, files)
        ):
            return {"root": str(root), "manifest": str(path),
                    "content_id": cached["content_id"], "file_count": len(files),
                    "total_bytes": sum(row["bytes"] for row in cached_files.values()),
                    "cache_validation": "all_relative_paths_size_and_mtime_ns_matched"}
    contents = {str(file.relative_to(root)): {
                    "bytes": file.stat().st_size, "mtime_ns": file.stat().st_mtime_ns,
                    "sha256": digest(file)} for file in files}
    identity = {name: {"bytes": row["bytes"], "sha256": row["sha256"]}
                for name, row in contents.items()}
    content_id = object_digest(identity)
    path = store / f"content_{content_id}.json"
    # The manifest is content-addressed by construction: its filename is the hash of the
    # relative paths and bytes below. Recording the live absolute path inside it made the
    # record location-dependent, so relocating a dataset produced the same content_id with
    # a different payload and the immutability check rejected it. Location belongs in the
    # sidecar index keyed by root, which is already written per location.
    result = {"schema_version": 2,
              "content_id": content_id, "files": contents,
              "coverage": "all files under meta/data/videos; HF cache and readme excluded",
              "cache_validation": "relative path, size, and mtime_ns before hash reuse"}
    immutable_json(path, result)
    atomic_json(cache_path, {"schema_version": 1, "root": str(root),
                             "manifest": str(path), "content_id": content_id})
    return {"root": str(root), "manifest": str(path), "content_id": result["content_id"],
            "file_count": len(files), "total_bytes": sum(row["bytes"] for row in contents.values()),
            "cache_validation": "all_relative_paths_size_and_mtime_ns_matched"}


def training_versions(root: Path, mixture: Path | None, store: Path) -> list[dict]:
    roots = [Path(row["root"]) for row in read_json(mixture)["datasets"]] if mixture else [root]
    return [fingerprint_dataset(path, store) for path in roots]
