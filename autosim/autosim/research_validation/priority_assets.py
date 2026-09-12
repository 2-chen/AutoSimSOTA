"""Acquire one pinned missing task without rewriting the main downloader state."""
import argparse
import shutil
from pathlib import Path

from autosim.research.common import atomic_json, digest, exclusive, now, read_json


def acquire_task(output, task, *, max_gib=32):
    from huggingface_hub import HfApi, get_token, snapshot_download

    entry = next(row for row in read_json(output / "hub_inventory.json") if row["task"] == task)
    path = output / f"priority_assets_{task}.json"
    with exclusive(output / f"priority_assets_{task}.lock"):
        state = read_json(path) if path.exists() else {"task": task, "created_at": now(), "resources": {}}
        token = get_token() or False
        api = HfApi(token=token)
        try:
            planned = 0
            for kind in ("model", "dataset"):
                resource = entry[kind]
                if not resource.get("available") or not resource.get("revision"):
                    raise ValueError("pinned official resource unavailable")
                destination = output / "assets" / task / kind
                info = api.repo_info(resource["repository"], repo_type=kind,
                                     revision=resource["revision"], files_metadata=True, timeout=30)
                size = sum(int(f.size or 0) for f in info.siblings)
                planned += size
                if planned > max_gib * 1024**3 or shutil.disk_usage(output).free < size * 2 + 30 * 1024**3:
                    raise RuntimeError("asset budget or free disk headroom exceeded")
                state["resources"][kind] = {"repository": resource["repository"], "revision": resource["revision"],
                    "status": "downloading", "bytes": size, "path": str(destination)}
                state.update(status="running", updated_at=now())
                atomic_json(path, state)
                print(f"[{task}] {kind}: {size / 1024**3:.2f} GiB, pinned revision", flush=True)
                snapshot_download(resource["repository"], repo_type=kind, revision=resource["revision"],
                                  local_dir=destination, max_workers=2, token=token)
                marker = destination / ("model.safetensors" if kind == "model" else "meta/info.json")
                state["resources"][kind].update(status="completed", marker_sha256=digest(marker), finished_at=now())
                atomic_json(path, state)
            # Main acquire() writes only asset_status.json. Read the inventory
            # afresh under a separate lock; never publish a partial dataset.
            with exclusive(output / "priority_inventory.lock"):
                inventory = read_json(output / "task_inventory.json")
                item = next(row for row in inventory if row["task"] == task)
                item.update(checkpoint_available=True, dataset_available=True,
                    official_checkpoint=state["resources"]["model"]["path"],
                    official_dataset=state["resources"]["dataset"]["path"])
                atomic_json(output / "task_inventory.json", inventory)
            state.update(status="completed", finished_at=now())
        except Exception as exc:
            # Hub exceptions can contain credential-bearing signed URLs.
            state.update(status="failed", error_type=type(exc).__name__, finished_at=now())
            print(f"[{task}] acquisition failed: {type(exc).__name__}", flush=True)
        atomic_json(path, state)
        return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    raise SystemExit(int(acquire_task(args.output.absolute(), args.task)["status"] != "completed"))
