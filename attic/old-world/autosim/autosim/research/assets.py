"""Download pinned public assets without overwriting existing research artifacts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .common import atomic_json, digest, now, read_json, redact


def acquire(output: Path, task_names: list[str] | None = None, max_gib=120) -> dict:
    from huggingface_hub import HfApi, snapshot_download, get_token

    entries = read_json(output / "hub_inventory.json")
    tasks = set(task_names) if task_names else {r["task"] for r in entries}
    path = output / "asset_status.json"
    status = read_json(path) if path.exists() else {"created_at": now(), "tasks": {}, "max_download_gib": max_gib}
    # Use locally configured credentials without serializing their value.
    token = get_token() or False
    api = HfApi(token=token)
    status["authentication"] = "configured_credential" if token else "anonymous"
    planned_bytes = sum(v.get("bytes", 0) for t in status["tasks"].values()
                        for v in t.values() if v.get("status") == "completed")
    for entry in entries:
        task = entry["task"]
        if task not in tasks:
            continue
        row = status["tasks"].setdefault(task, {})
        for kind in ("model", "dataset"):
            resource = entry[kind]
            if row.get(kind, {}).get("status") == "completed":
                continue
            try:
                if not resource["available"]:
                    raise RuntimeError("official repository unavailable")
                destination = output / "assets" / task / kind
                info = api.repo_info(resource["repository"], repo_type=kind,
                                     revision=resource["revision"], files_metadata=True, timeout=30)
                size = sum(int(f.size or 0) for f in info.siblings)
                planned_bytes += size
                if planned_bytes > max_gib * 1024**3:
                    raise RuntimeError("asset download byte budget exceeded")
                if shutil.disk_usage(output).free < size * 2 + 30 * 1024**3:
                    raise RuntimeError("insufficient free disk including conversion/headroom")
                row[kind] = {**resource, "status": "downloading", "bytes": size, "path": str(destination)}
                atomic_json(path, status)
                print(f"[{task}] downloading {kind}, {size / 1024**3:.2f} GiB", flush=True)
                snapshot_download(resource["repository"], repo_type=kind, revision=resource["revision"],
                                  local_dir=destination, max_workers=3, token=token)
                marker = destination / ("model.safetensors" if kind == "model" else "meta/info.json")
                if not marker.is_file():
                    raise RuntimeError(f"expected asset layout missing: {marker}")
                row[kind].update(status="completed", finished_at=now(), marker_sha256=digest(marker))
            except Exception as exc:
                row.setdefault(kind, {}).update(status="failed", error=redact(f"{type(exc).__name__}: {exc}"))
                print(f"[{task}] {kind}: {type(exc).__name__}", flush=True)
                if getattr(getattr(exc, "response", None), "status_code", None) == 429:
                    status["blocked_reason"] = "hub_rate_limit; resume cached download after cooldown"
                    atomic_json(path, status)
                    return status
            atomic_json(path, status)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--max-gib", default=120, type=int)
    args = parser.parse_args()
    result = acquire(args.output.absolute(), args.tasks, args.max_gib)
    raise SystemExit(int(any(v.get("status") != "completed" for k, row in result["tasks"].items()
                            if not args.tasks or k in args.tasks for v in row.values())))
