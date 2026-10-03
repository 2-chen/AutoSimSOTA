"""Read-only provider token accounting, separate from budget reservations."""
from __future__ import annotations

import json
from pathlib import Path


def _single_report(root: Path) -> dict:
    root = Path(root)
    roles, identities = {}, {}
    skipped = 0
    for path in (root / "agent" / "processes").glob("*.json"):
        if path.is_symlink():
            continue
        try:
            record = json.loads(path.read_text())
            identities[record["attempt_id"]] = record.get("role", "unknown")
        except (OSError, ValueError, KeyError, TypeError):
            continue
    for path in (root / "agent" / "pricing").glob("*.json"):
        if path.is_symlink():
            skipped += 1
            continue
        try:
            record = json.loads(path.read_text())
            item = roles.setdefault(identities.get(record.get("turn_id"), "unknown"),
                dict(turns=0, requests_with_usage=0, requests_without_usage=0,
                     input_hit_tokens=0, input_miss_tokens=0, output_tokens=0))
            item["turns"] += 1
            for receipt in (record.get("gateway") or {}).get("receipts", []):
                usage = receipt.get("usage")
                if not usage:
                    item["requests_without_usage"] += 1
                    continue
                counts = {key: usage.get(key, 0) for key in
                          ("input_hit_tokens", "input_miss_tokens", "output_tokens")}
                if any(not isinstance(v, int) or isinstance(v, bool) or v < 0
                       for v in counts.values()):
                    skipped += 1
                    continue
                item["requests_with_usage"] += 1
                for key, value in counts.items():
                    item[key] += value
        except (OSError, ValueError, TypeError, AttributeError):
            skipped += 1
    totals = {key: sum(row[key] for row in roles.values()) for key in
              ("turns", "requests_with_usage", "requests_without_usage",
               "input_hit_tokens", "input_miss_tokens", "output_tokens")}
    for row in [totals, *roles.values()]:
        inputs = row["input_hit_tokens"] + row["input_miss_tokens"]
        row["cache_hit_ratio"] = row["input_hit_tokens"] / inputs if inputs else None
    return {"schema_version": 1, "basis": "known_gateway_receipts_only",
            "unknown_usage_is_not_zero_cost": True, "totals": totals,
            "roles": roles, "skipped_records": skipped}


def usage_report(root: Path) -> dict:
    root = Path(root)
    report = _single_report(root)
    workers = root / "agent_workers"
    children = [] if workers.is_symlink() else list(workers.glob("*"))
    count = 0
    for child in children:
        if child.is_symlink() or not child.is_dir():
            continue
        extra = _single_report(child)
        count += 1
        report["skipped_records"] += extra["skipped_records"]
        for role, row in extra["roles"].items():
            dest = report["roles"].setdefault(role, {key: 0 for key in row})
            for key, value in row.items():
                if key != "cache_hit_ratio":
                    dest[key] += value
    for key in report["totals"]:
        if key != "cache_hit_ratio":
            report["totals"][key] = sum(row[key] for row in report["roles"].values())
    for row in [report["totals"], *report["roles"].values()]:
        total = row["input_hit_tokens"] + row["input_miss_tokens"]
        row["cache_hit_ratio"] = row["input_hit_tokens"] / total if total else None
    report["worker_roots_included"] = count
    return report


if __name__ == "__main__":
    import sys
    print(json.dumps(usage_report(Path(sys.argv[1])), ensure_ascii=False, indent=2))
