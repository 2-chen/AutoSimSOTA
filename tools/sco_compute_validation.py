#!/usr/bin/env python3
"""Prepare immutable validation sources and submit one bounded 5090 SCO allocation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE = Path("/data/AutoResearch/AutoSimSOTA")
PROJECT = BASE / "AutoSimSOTA"
IMAGE = "registry.cn-sh-01g.sensecore.cn/zhicheng_ccr/app-robotwin-wam:cci-20260521171320"
MOUNT = "01995892-d478-76d8-aec7-13fd8284477e/250010008:/data"
TERMINAL = {"SUSPENDED", "SUCCEEDED", "FAILED", "DELETED", "CANCELLED", "STOPPED"}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def query(argv):
    creating = argv[:3] == ["acp", "jobs", "create"]
    result = subprocess.run(["sco", *argv, *([] if creating else ["-o", "json"])],
                            capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-2000:])
    if creating:
        matches = set(re.findall(r"\bpt-[a-z0-9]+\b", result.stdout + result.stderr))
        return {"name": next(iter(matches)) if len(matches) == 1 else None,
                "create_output": result.stdout + result.stderr}
    # SCO prints this sentinel even when JSON output was explicitly requested.
    if argv[:3] == ["acp", "jobs", "list"] and result.stdout.strip() == "No jobs found":
        return []
    return json.loads(result.stdout)


def prepare(root):
    snapshot = root / "source"
    if (root / "source_manifest.json").exists():
        for name, expected in json.loads((root / "source_manifest.json").read_text()).items():
            if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError("frozen source changed")
        return snapshot
    snapshot.mkdir(parents=True, exist_ok=True)
    for folder in ("autosim", "tools"):
        shutil.copytree(PROJECT / folder, snapshot / folder, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.egg-info", ".env"))
    for name, target in ((".venv", PROJECT / ".venv"), ("RoboSynChallenge", BASE / "RoboSynChallenge"),
                         ("embodichain_data", PROJECT / "embodichain_data"), ("EmbodiChain", BASE / "EmbodiChain")):
        if not (snapshot / name).exists():
            (snapshot / name).symlink_to(target, target_is_directory=True)
    hashes = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ("autosim", "tools") for p in (snapshot / folder).rglob("*") if p.is_file()}
    save(root / "source_manifest.json", hashes)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-id", required=True)
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--dry-run", action="store_true", help="Prepare and display the command without submission")
    parser.add_argument("--seconds", type=int, default=3300)
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("--reuse-components", type=Path)
    args = parser.parse_args()
    if not 60 <= args.seconds <= 3540:
        raise ValueError("validation runtime must be in [60, 3540] seconds")
    if not args.validation_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("invalid validation id")
    root = BASE / "compute_validation" / args.validation_id
    source = prepare(root)
    output = root / f"{args.gpus}gpu"
    submission = output / "submission.json"
    if submission.exists():
        previous = json.loads(submission.read_text())
        if args.submit and previous.get("name"):
            details = query(["acp", "jobs", "describe", "--workspace-name", "share-space", previous["name"]])
            save(output / "job_description.json", details)
            previous["state"] = details.get("state")
        print(json.dumps({"already_submitted": True, **previous}))
        return
    needed = [source / ".venv/bin/python", source / "RoboSynChallenge/policy/act/.venv/bin/python",
              source / "RoboSynChallenge/checkpoints/ACT_sim_click_bell/model.safetensors",
              source / "RoboSynChallenge/lerobot_dataset/RoboSynChallenge/cobotmagic_Sim_click_bell/meta/info.json",
              BASE / "optix_runtime/nvoptix.bin", BASE / "container_libs/lib/libnvoptix.so.1"]
    needed.append(BASE / "compute_validation/dependencies/torch/hub/checkpoints/resnet18-f37072fd.pth")
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing pre-cached dependencies: {missing}")
    command = shlex.join(["bash", str(source / "tools/run_compute_validation.sh"),
                          "--expected-gpus", str(args.gpus), "--output", str(output), "--seconds", str(args.seconds),
                          "--api-evidence", str(BASE / "compute_validation/api_validation_20260914"),
                          *(["--continue-from",str(args.continue_from.absolute())] if args.continue_from else []),
                          *(["--reuse-components",str(args.reuse_components.absolute())] if args.reuse_components else [])])
    argv = ["acp", "jobs", "create", "--workspace-name", "share-space",
            "--aec2-name", "computing-cluster-5090-01g", "--job-name",
            f"autosim-cv-{args.gpus}g-{args.validation_id}", "--container-image-url", IMAGE,
            "--training-framework", "pytorch", "--worker-nodes", "1", "--worker-spec",
            f"n12lp.nn.i10a.{args.gpus}", "--priority", "NORMAL", "--quota-type", "reserved",
            "--storage-mount", MOUNT, "--command", command]
    save(output / "submission_request.json", {"argv": ["sco", *argv], "gpus": args.gpus,
        "expected_max_execution_seconds": args.seconds+30, "nominal_gpu_hours": args.gpus*(args.seconds+30)/3600, "full_autoresearch": False})
    if not args.submit:
        print(shlex.join(["sco", *argv])); return
    active, all_rows, page = [], [], 1
    while True:
        rows = query(["acp", "jobs", "list", "--workspace-name", "share-space",
                      "--user-name", "250010008", "--page-size", "500", "--page-token", str(page)])
        active.extend({k: r.get(k) for k in ("name", "display_name", "state", "roles")}
                      for r in rows if r["state"] not in TERMINAL)
        all_rows.extend(rows)
        if len(rows) < 500: break
        page += 1
    save(output / "quota_check.json", {"active_jobs": active, "remaining_member_quota": "unknown"})
    display = f"autosim-cv-{args.gpus}g-{args.validation_id}"
    matches = [r for r in all_rows if r.get("display_name") == display]
    if matches:
        if len(matches) != 1:
            raise RuntimeError("multiple matching jobs; manual reconciliation required")
        save(submission, {"name": matches[0]["name"], "reconciled_by_unique_display_name": True})
        print(json.dumps({"already_submitted": True, "name": matches[0]["name"], "state": matches[0]["state"]}))
        return
    if (output / "submission_intent.json").exists():
        raise RuntimeError("prior submission outcome unresolved; refusing duplicate create")
    spent_or_reserved = 0
    for previous in (BASE / "compute_validation").glob("matrix_*/*gpu/submission.json"):
        directory = previous.parent
        summary_path, request_path = directory / "summary.json", directory / "submission_request.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if summary.get("finished") and "allocated_gpu_hours" in summary:
            used = summary["allocated_gpu_hours"]
            details_path = directory / "job_description.json"
            details = json.loads(details_path.read_text()) if details_path.exists() else {}
            end = details.get("complete_time") or details.get("suspend_time")
            if end and details.get("start_time"):
                elapsed = (datetime.fromisoformat(end.replace("Z", "+00:00")) -
                           datetime.fromisoformat(details["start_time"].replace("Z", "+00:00"))).total_seconds()
                used = max(used, int(directory.name.removesuffix("gpu"))*elapsed/3600)
            spent_or_reserved += used
        elif request_path.exists():
            spent_or_reserved += json.loads(request_path.read_text())["nominal_gpu_hours"]
    reservation = args.gpus*(args.seconds+30)/3600
    if spent_or_reserved + reservation > 15:
        raise RuntimeError(f"matrix GPU-hour cap: {spent_or_reserved:.4f} spent/reserved + {reservation:.4f} > 15")
    save(output / "matrix_admission.json", {"limit_gpu_hours":15,"previous_spent_or_reserved":spent_or_reserved,
        "new_reservation_gpu_hours":reservation,"platform_setup_and_release_reconciled_separately":True})
    save(output / "submission_intent.json", {"display_name":display,"state":"create_requested"})
    try:
        created = query(argv)
    except RuntimeError as exc:
        save(output / "submission_failure.json", {"error": str(exc), "retry": "requires_reconciliation"})
        raise
    save(submission, created)
    name = created.get("name")
    if not name:
        raise RuntimeError("create returned no job name; reconcile before resubmission")
    details = query(["acp", "jobs", "describe", "--workspace-name", "share-space", name])
    save(output / "job_description.json", details)
    print(json.dumps({"job": name, "state": details.get("state"), "gpus": args.gpus,
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
