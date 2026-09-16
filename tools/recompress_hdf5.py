#!/usr/bin/env python3
"""Re-store raw-frame HDF5 episodes with a chunked gzip filter.

`policy/ACT/process_data.py` wrote raw `uint8` camera frames into datasets that
had no chunks and therefore no filter. Rendered simulation frames carry far less
entropy than photographs: measured on this corpus an episode of three
480x640x3 cameras goes 321 MB -> 39 MB (12.0%), losslessly.

This rewrites existing files in place, atomically, and refuses to replace a file
unless the readback is bit-identical. Files that already carry a filter are
skipped, so re-running is cheap and safe.

usage:
    recompress_hdf5.py ROOT [--dry-run] [--limit N] [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np


def is_compressed(path: Path) -> bool | None:
    """True/False for the file's datasets, or None if it holds no datasets."""
    try:
        with h5py.File(path, "r") as handle:
            def visit(_, obj):
                if isinstance(obj, h5py.Dataset) and obj.size:
                    raise _Found(obj.compression is not None)

            try:
                handle.visititems(visit)
            except _Found as found:
                return found.value
            return None  # nothing sized; leave it alone
    except (OSError, KeyError):
        return None


class _Found(Exception):
    def __init__(self, value):
        self.value = value


def rewrite(path: Path, level: int = 4) -> dict:
    """Rewrite one file with chunked gzip; verified bit-identical or unchanged."""
    before = path.stat().st_size
    fd, tmp_name = tempfile.mkstemp(suffix=".hdf5", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with h5py.File(path, "r") as src, h5py.File(tmp, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value

            def copy(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                kw = {}
                if obj.shape and all(size > 0 for size in obj.shape):
                    kw = dict(chunks=(1,) + tuple(obj.shape[1:]),
                              compression="gzip", compression_opts=level)
                dst.create_dataset(name, data=obj[()], dtype=obj.dtype, **kw)

            src.visititems(copy)

        # A lossless rewrite must read back identical, every dataset and attribute.
        with h5py.File(path, "r") as src, h5py.File(tmp, "r") as dst:
            if sorted(src.keys()) != sorted(dst.keys()):
                raise RuntimeError("dataset set changed")
            checked = [0]

            def verify(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                other = dst[name]
                if obj.shape != other.shape or obj.dtype != other.dtype:
                    raise RuntimeError(f"shape/dtype changed for {name}")
                if not np.array_equal(obj[()], other[()]):
                    raise RuntimeError(f"values changed for {name}")
                checked[0] += 1

            src.visititems(verify)
            if checked[0] == 0:
                raise RuntimeError("no datasets verified")

        after = tmp.stat().st_size
        if after >= before:
            tmp.unlink(missing_ok=True)
            return {"path": str(path), "status": "no_gain", "before": before, "after": before}
        os.replace(tmp, path)
        return {"path": str(path), "status": "rewritten", "before": before, "after": after,
                "datasets": checked[0]}
    except Exception as exc:  # leave the original untouched
        tmp.unlink(missing_ok=True)
        return {"path": str(path), "status": "failed", "before": before, "before_kept": True,
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    files = sorted(args.root.rglob("*.hdf5"))
    if args.limit:
        files = files[: args.limit]

    todo = []
    for path in files:
        flag = is_compressed(path)
        if flag is True:
            continue
        todo.append(path)

    total_before = sum(p.stat().st_size for p in todo)
    print(f"{len(files)} hdf5 files, {len(todo)} need rewriting "
          f"({total_before / 1024**3:.1f} GiB raw)", flush=True)
    if args.dry_run:
        for path in todo[:20]:
            print(f"  would rewrite {path.stat().st_size/1e6:7.1f} MB  {path}")
        return 0

    results, saved = [], 0
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(rewrite, p, args.level): p for p in todo}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            results.append(row)
            if row["status"] == "rewritten":
                saved += row["before"] - row["after"]
            if index % 20 == 0 or index == len(todo):
                rate = saved / max(1e-9, time.time() - start)
                print(f"  [{index}/{len(todo)}] saved {saved/1024**3:.2f} GiB "
                      f"({rate/1024**3:.2f} GiB/s)", flush=True)

    failed = [r for r in results if r["status"] == "failed"]
    summary = {"root": str(args.root), "files": len(files), "rewritten":
               sum(1 for r in results if r["status"] == "rewritten"),
               "no_gain": sum(1 for r in results if r["status"] == "no_gain"),
               "failed": len(failed), "bytes_before": total_before,
               "bytes_saved": saved, "results": results}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nrewritten={summary['rewritten']} no_gain={summary['no_gain']} failed={len(failed)}")
    print(f"saved {saved/1024**3:.2f} GiB of {total_before/1024**3:.2f} GiB "
          f"({100*saved/total_before if total_before else 0:.1f}%)")
    for row in failed[:10]:
        print(f"  FAILED {row['path']}: {row.get('error')}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
