"""Read-only provider token accounting, separate from budget reservations."""
from __future__ import annotations

import json
import math
from pathlib import Path


_SUM_FIELDS = ("turns", "requests_total", "requests_with_usage", "requests_without_usage",
               "input_hit_tokens", "input_miss_tokens", "output_tokens",
               "known_cost_usd", "unknown_reserved_usd", "requests_without_cost")


def _row():
    return {**dict.fromkeys(_SUM_FIELDS, 0), "_input_sizes": [], "_profiles": []}


def _money(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


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
            item = roles.setdefault(identities.get(record.get("turn_id"), "unknown"), _row())
            item["turns"] += 1
            for receipt in (record.get("gateway") or {}).get("receipts", []):
                item["requests_total"] += 1
                cost = receipt.get("cost_usd")
                if receipt.get("status") == "priced" and _money(cost):
                    item["known_cost_usd"] += cost
                else:
                    item["requests_without_cost"] += 1
                if receipt.get("status") == "unknown" and _money(receipt.get("reserved_usd")):
                    item["unknown_reserved_usd"] += receipt["reserved_usd"]
                profile = receipt.get("request_profile")
                if isinstance(profile, dict):
                    item["_profiles"].append(profile)
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
                item["_input_sizes"].append(counts["input_hit_tokens"] + counts["input_miss_tokens"])
                for key, value in counts.items():
                    item[key] += value
        except (OSError, ValueError, TypeError, AttributeError):
            skipped += 1
    return {"schema_version": 2, "basis": "stored_gateway_receipts_only_no_repricing",
            "unknown_usage_is_not_zero_cost": True,
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
            dest = report["roles"].setdefault(role, _row())
            for key in _SUM_FIELDS:
                dest[key] += row[key]
            for key in ("_input_sizes", "_profiles"):
                dest[key].extend(row[key])
    totals = _row()
    for row in report["roles"].values():
        for key in _SUM_FIELDS:
            totals[key] += row[key]
        for key in ("_input_sizes", "_profiles"):
            totals[key].extend(row[key])
    report["totals"] = totals
    all_outputs, all_cost = totals["output_tokens"], totals["known_cost_usd"]
    for row in [totals, *report["roles"].values()]:
        total = row["input_hit_tokens"] + row["input_miss_tokens"]
        row["cache_hit_ratio"] = row["input_hit_tokens"] / total if total else None
        row["output_token_share"] = row["output_tokens"] / all_outputs if all_outputs else None
        row["known_cost_share"] = row["known_cost_usd"] / all_cost if all_cost else None
        sizes = sorted(row.pop("_input_sizes"))
        row["input_tokens_per_request"] = {"mean": sum(sizes) / len(sizes),
            "max": sizes[-1], "p95": sizes[math.ceil(len(sizes) * .95) - 1]} if sizes else None
        profiles = row.pop("_profiles")
        # Legacy requests have no fingerprints: absent telemetry is not stability.
        row["request_profiles"] = {"observed_requests": len(profiles),
            "missing_requests": row["requests_total"] - len(profiles),
            "not_provider_cache_measurement": True,
            "distinct_hashes": {field: len({profile[field] for profile in profiles
                if isinstance(profile.get(field), str)}) for field in
                ("envelope_sha256", "system_sha256", "tools_sha256", "first_message_sha256")}}
    report["worker_roots_included"] = count
    report["not_full_inherited_ledger_or_provider_invoice"] = True
    return report


if __name__ == "__main__":
    import sys
    print(json.dumps(usage_report(Path(sys.argv[1])), ensure_ascii=False, indent=2))
