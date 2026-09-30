"""Record native asset and cuRobo content without importing CUDA or editing assets.

This is a reproducibility inventory, not an upstream authenticity proof or a
security boundary. The currently frozen GPU queue does not consume this new
manifest; binding it into job inputs requires an explicit new experiment version.
"""
import argparse
import time
from pathlib import Path

from autosim.research.common import atomic_json, digest, now, read_json


def inventory(root: Path, progress):
    root = root.resolve(strict=True)
    files, issues = {}, []
    last = time.monotonic()
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            issues.append({"path": relative, "kind": "symlink_requires_explicit_review"})
            continue
        if not path.is_file():
            continue
        before = path.stat()
        sha = digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError(f"asset changed during read: {path}")
        files[relative] = {"bytes": after.st_size, "sha256": sha}
        if time.monotonic() - last > 5:
            progress(root, len(files), sum(r["bytes"] for r in files.values()))
            last = time.monotonic()
    return {"root": str(root), "file_count": len(files),
            "bytes": sum(r["bytes"] for r in files.values()), "files": files, "issues": issues}


def run(system: Path, output: Path):
    system = system.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    native = system / "native/RoboTwin"
    assets = native / "assets"
    roots = {name: assets / name for name in ("objects", "embodiments", "files", "background_texture")}
    roots["curobo_installed"] = system / "runtime_env/lib/python3.10/site-packages/curobo"
    reports = {}
    def progress(root, count, size):
        atomic_json(output / "progress.json", {"status": "running", "root": str(root),
                    "files_hashed_in_root": count, "bytes_hashed_in_root": size, "updated_at": now()})
    for name, root in roots.items():
        report = inventory(root, progress)
        path = output / f"{name}.json"
        atomic_json(path, report)
        reports[name] = {"manifest": str(path), "sha256": digest(path),
                         "file_count": report["file_count"], "bytes": report["bytes"], "issues": report["issues"]}
    import yaml
    references = []
    for name in ("curobo_left.yml", "curobo_right.yml"):
        config = assets / "embodiments/aloha-agilex" / name
        kinematics = yaml.safe_load(config.read_text())["robot_cfg"]["kinematics"]
        for key in ("urdf_path", "collision_spheres"):
            path = Path(kinematics[key])
            references.append({"config": str(config), "key": key, "path": str(path),
                               "exists": path.is_file(), "inside_native_assets": path.resolve().is_relative_to(assets.resolve())})
    texture = read_json(system / "texture_state.json")
    summary = {"status": "completed", "finished_at": now(), "elapsed_seconds": time.monotonic() - started,
               "manifests": reports, "selected_embodiment_references": references,
               "texture_retrieval_evidence": {"path": str(system / "texture_state.json"),
                                             "sha256": digest(system / "texture_state.json"),
                                             "retrieval_status": texture.get("status")},
               "local_inventory_checks_passed": not any(r["issues"] for r in reports.values()) and
                   all(r["exists"] and r["inside_native_assets"] for r in references),
               "upstream_object_and_embodiment_revision_verified": False,
               "complete_runtime_dependency_closure_verified": False,
               "gpu_compatibility_verified": False,
               "enforced_by_current_gpu_queue": False}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "progress.json", {"status": "completed", "updated_at": now()})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.system, args.output)
