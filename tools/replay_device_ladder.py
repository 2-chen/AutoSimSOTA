"""Re-judge a recorded device-probe receipt with the current ``judge``.

    python tools/replay_device_ladder.py <receipt.json> [<receipt.json> ...]

A refusal is a claim about *evidence*, and the only way to know whether a repair changed the
verdict is to put the same evidence back through the new code.  This reads
``device_probe.json``, rebuilds every arm's ``judge`` inputs from that receipt's own memory
rows (peak, baseline, holders, idle list) and its ``startup_census`` timeline, and prints old
verdict vs replayed verdict side by side.

What this is: a measurement of the *new code on the old data*, which is why it can be quoted
when arguing the exception is correct.  What it is not: a receipt, a prediction of the next
ladder, or a substitute for running one -- a replayed ``passed=True`` says the recorded shapes
are accepted, and the next ladder still has to measure its own container.  The negative
control arm (``C``) is replayed too: if a repair lets the arm that must abort pass, the replay
shows it here rather than in a run that costs GPU hours.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

from autosim.research import device_probe  # noqa: E402  (needs the path above)


def replay(receipt: dict, *, workspace: str) -> tuple[dict, list[dict]]:
    """Return (replayed verdict, arms re-judged with the current code)."""
    allocation = [device["uuid"] for device in receipt["allocation"]]
    arms = []
    for arm in receipt["arms"]:
        row = dict(arm)
        memory = arm.get("memory")
        if memory:
            peak = {uuid: {k: v for k, v in held.items() if k != "baseline_memory_mib"}
                    for uuid, held in memory.items()}
            baseline = {uuid: {"peak_memory_mib": held.get("baseline_memory_mib")}
                        for uuid, held in memory.items()}
            own = arm["judged"]["own_device_uuid"]
            # The exclusion list the original call used is not stored; its *result* is, and
            # idle = idle_uuids - {own}, so the result plus the arm's own card reproduces it.
            idle_uuids = list(arm["judged"]["idle_devices"]) + [own]
            row["judged"] = device_probe.judge(arm, own_uuid=own, allocation=allocation,
                                               baseline=baseline, peak=peak,
                                               idle_uuids=idle_uuids,
                                               census=arm.get("startup_census") or [],
                                               run_root=workspace)
        arms.append(row)
    return device_probe.probe_verdict(arms=arms, allocation=receipt["allocation"],
                                      requested_mode="identity"), arms


def report(path: Path, *, workspace: str) -> dict:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    verdict, arms = replay(receipt, workspace=workspace)
    was = receipt["verdict"]
    print(f"\n=== {path} ===")
    print(f"{'arm':<14}{'old':<7}{'new':<7}standing")
    for old_arm, new_arm in zip(receipt["arms"], arms):
        judged = new_arm.get("judged") or {}
        residuals = judged.get("foreign_holders") or []
        notes = []
        for row in residuals:
            standing = "accepted" if row.get("accepted") else f"REFUSED: {row.get('reason')}"
            notes.append(f"{row['device_uuid'][-8:]}={row['growth_mib']}MiB {standing}")
        print(f"{new_arm['name']:<14}{str((old_arm.get('judged') or {}).get('passed')):<7}"
              f"{str(judged.get('passed')):<7}{'; '.join(notes) or '-'}")
    for key in ("winner", "selection_mode", "verified_every_allocated_device",
                "concurrent_two_devices_verified", "model_discrepancies", "passed"):
        print(f"{key}: {was.get(key)!r} -> {verdict.get(key)!r}")
    print(f"verified now: {sorted(verdict['verified_devices'])}")
    print("still unexplained: " + (", ".join(
        f"{row['arm']}@{row['device_index']} ({row.get('reason')})"
        for row in verdict["cross_card_residual"] if not row.get("accepted")) or "none"))
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("receipts", nargs="+", type=Path)
    parser.add_argument("--workspace", default=str(WORKSPACE),
                        help="the platform root the probe ran with (defaults to this checkout)")
    args = parser.parse_args()
    failed = 0
    for path in args.receipts:
        try:
            failed += not report(path, workspace=args.workspace)["passed"]
        except (OSError, KeyError, ValueError) as exc:
            print(f"{path}: cannot replay -- {type(exc).__name__}: {exc}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
