"""Merge evaluation shards into the one payload the rest of the pipeline expects.

Written before the shard runner on purpose: the integrity gates have to exist before
anything is allowed to execute in parallel, otherwise "sharded" becomes a label on an
unverified number.

What is checked, and refused rather than repaired:

  * every shard named by the frozen ``shard_plan.json`` exists and certified a real result
    under the requested purpose;
  * every shard ran the *same* frozen files, checkpoint and master seed -- ``protocol.json``
    is compared across shards modulo the shard block, so a shard built from different code
    cannot be averaged in;
  * each shard's episode seeds equal *its own* contiguous slice of the master seed bank
    (``robosyn_data.evaluation_seed_bank``, byte-identical to the official draw loop), so
    the union can have neither a gap nor a duplicate -- the very pairing the ledger checks
    downstream;
  * each shard's own ``summary`` is recomputed from its rows and must agree, which proves
    the recomputation formula on real data before it is used for the merged number.

What is produced: a merged ``evaluation_metrics.json`` whose ``episodes`` are the shard rows
in shard order with ``episode_index`` renumbered, whose ``config.episode_count`` is the total,
and a merged ``protocol.json`` with no shard block (so frozen-file and horizon checks keep
working unchanged).  The composite ``process/process.json`` keeps the logical command shape
and embeds the sha256 of each shard's real receipt, so the merged number stays traceable to
the processes that produced it.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import atomic_json, digest, now, object_digest, read_json
from .devices import shard_plan

SHARD_PLAN = "shard_plan.json"
MERGE_RECEIPT = "merge_receipt.json"
# Fields a shard may legitimately vary in; everything else in protocol.json must be equal.
PER_SHARD_FIELDS = ("episodes", "shard")
MEAN_REL_TOLERANCE = 1e-9


class ShardMergeRefused(RuntimeError):
    """The shards cannot be merged into one number; the reason is the message."""


def build_shard_plan(*, episodes: int, bank: Sequence[int],
                     selections: Sequence[Mapping[str, Any]]) -> dict:
    """The frozen division of work: contiguous blocks over the master seed bank.

    Written (``immutable_json``) before any shard starts, so a resume cannot re-divide the
    work differently, and the bank digest lets the merge prove that the slices it verifies
    belong to the bank *this* process derives rather than to a bank that has since changed.
    """
    blocks = shard_plan(episodes, len(selections))
    return {
        "schema_version": 1, "episodes": episodes, "count": len(selections),
        "bank_sha256": object_digest([int(seed) for seed in bank[:episodes]]),
        "blocks": [{"index": index, "offset": offset, "size": size,
                    "device_uuid": selection.get("uuid"), "device_index": selection.get("index"),
                    "selection": selection.get("selection")}
                   for index, ((offset, size), selection) in enumerate(zip(blocks, selections))],
    }


def load_shard_plan(output: Path) -> dict:
    path = Path(output) / SHARD_PLAN
    if not path.is_file():
        raise ShardMergeRefused(
            f"no {SHARD_PLAN} in {output}: a sharded evaluation must freeze its division "
            f"before running, so a resume cannot re-shard differently")
    return read_json(path)


def shard_dir(output: Path, index: int) -> Path:
    return Path(output) / "shards" / f"shard_{index:02d}"


def attempt_of(directory: Path) -> int:
    """1 for the shard root, k for ``startup_attempt_k``; the receipt records which ran."""
    name = Path(directory).name
    prefix = "startup_attempt_"
    return int(name[len(prefix):]) if name.startswith(prefix) and name[len(prefix):].isdigit() else 1


def certified_shard_dir(output: Path, index: int) -> Path:
    """Where shard ``index``'s certified payload lives, or a refusal saying why not.

    A crash before the first reset is retried into ``startup_attempt_{k}`` (the crashed
    attempt's directory is left exactly as it was -- it is evidence), so a payload is
    accepted from the shard root or from *one* retry directory.  Two certified payloads
    would mean the shard ran twice and the merge cannot say which number it reports, so
    that is refused rather than resolved by picking a favourite.
    """
    root = shard_dir(output, index)
    found = [directory for directory in [root, *sorted(root.glob("startup_attempt_*"))]
             if (directory / "evaluation_metrics.json").is_file()
             and (directory / "protocol.json").is_file()]
    if not found:
        raise ShardMergeRefused(
            f"shard {index} at {root} has no certified payload "
            f"(metrics={(root / 'evaluation_metrics.json').is_file()}, "
            f"protocol={(root / 'protocol.json').is_file()}); partial output is not merged")
    if len(found) > 1:
        raise ShardMergeRefused(
            f"shard {index} has {len(found)} certified payloads "
            f"({', '.join(str(directory.relative_to(output)) for directory in found)}); "
            f"the merge cannot tell which result the number came from")
    return found[0]


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def expected_seeds(bank: Sequence[int], plan: Mapping[str, Any]) -> list[int]:
    slices: list[int] = []
    for block in plan["blocks"]:
        offset, size = int(block["offset"]), int(block["size"])
        slices.extend(bank[offset:offset + size])
    return slices


def recompute_summary(rows: Sequence[Mapping[str, Any]], *, timeout_action_steps: int) -> dict:
    """The official evaluator's summary, recomputed from episode rows.

    The one field that is *recomputed rather than reproduced* is the mean inference
    latency: the official number averages the raw inference samples pairwise, while a
    shard boundary only preserves per-episode totals, so the merge uses
    ``sum(totals) / sum(calls)``.  The two agree to floating-point associativity, which is
    why the per-shard gate below compares it with a tolerance and every other field
    exactly.
    """
    calls = [int(row.get("inference_call_count") or 0) for row in rows]
    totals = [float(row.get("total_inference_time_seconds") or 0.0) for row in rows]
    steps = [float(row["action_steps"]) for row in rows]
    successful = sum(1 for row in rows if row.get("success"))
    total_calls, total_time = sum(calls), sum(totals)
    return {
        "episode_count": len(rows),
        "success_count": successful,
        "success_rate": successful / len(rows),
        "average_action_steps": statistics.fmean(steps) if steps else None,
        "average_action_steps_ratio": (statistics.fmean(steps) / timeout_action_steps
                                       if steps and timeout_action_steps else None),
        "inference_call_count": total_calls,
        "average_inference_calls_per_episode": total_calls / len(rows),
        "average_inference_time_seconds": (total_time / total_calls) if total_calls else None,
        "average_inference_time_per_episode_seconds": statistics.fmean(totals) if totals else None,
    }


def _compare_summary(recorded: Mapping[str, Any], recomputed: Mapping[str, Any], *, where: str) -> list[str]:
    problems = []
    for key, value in recomputed.items():
        prior = recorded.get(key)
        if isinstance(value, float) and isinstance(prior, (int, float)):
            if not math.isclose(float(value), float(prior), rel_tol=MEAN_REL_TOLERANCE):
                problems.append(f"{where}.summary.{key}: recorded={prior!r} recomputed={value!r}")
        elif prior != value:
            problems.append(f"{where}.summary.{key}: recorded={prior!r} recomputed={value!r}")
    return problems


def _protocol_differences(reference: Mapping[str, Any], other: Mapping[str, Any]) -> dict:
    keys = set(reference) | set(other)
    return {key: {"first": reference.get(key), "other": other.get(key)}
            for key in sorted(keys - set(PER_SHARD_FIELDS))
            if reference.get(key) != other.get(key)}


def merge_evaluation_shards(*, output: Path, purpose: str, master_seed: int, episodes: int,
                            bank: Sequence[int], merge_command: Sequence[str]) -> dict:
    """Combine the shards under ``output/shards``, or refuse with the first broken invariant."""
    output = Path(output)
    plan = load_shard_plan(output)
    blocks = list(plan["blocks"])
    if sum(int(block["size"]) for block in blocks) != episodes:
        raise ShardMergeRefused(
            f"shard plan covers {sum(int(b['size']) for b in blocks)} episodes, "
            f"the request has {episodes}")
    if len(bank) < episodes:
        raise ShardMergeRefused(f"seed bank holds {len(bank)} seeds, the request has {episodes}")
    freeze = plan.get("bank_sha256")
    if freeze and freeze != object_digest([int(seed) for seed in bank[:episodes]]):
        raise ShardMergeRefused(
            "the seed bank this process derives does not match the bank the shard plan "
            "froze; the shards were not divided over this bank")

    rows: list[dict] = []
    protocols: list[dict] = []
    shard_receipts: list[dict] = []
    metrics_documents: list[dict] = []
    problems: list[str] = []
    seen_seeds: dict[int, int] = {}

    for index, block in enumerate(blocks):
        directory = certified_shard_dir(output, index)
        metrics_path = directory / "evaluation_metrics.json"
        protocol_path = directory / "protocol.json"
        metrics, protocol = read_json(metrics_path), read_json(protocol_path)
        if metrics.get("execution_mode") != "real_simulation" or metrics.get("purpose") != purpose:
            problems.append(f"shard {index} is not a certified real result for purpose {purpose}")
        shard_rows = metrics.get("episodes")
        if not isinstance(shard_rows, list):
            problems.append(f"shard {index} has no episode rows")
            continue
        if len(shard_rows) != int(block["size"]):
            problems.append(f"shard {index} holds {len(shard_rows)} episodes, "
                            f"its block is {block['size']}")
        wanted = bank[int(block["offset"]):int(block["offset"]) + int(block["size"])]
        got = [int(row.get("episode_seed", -1)) for row in shard_rows]
        if got != list(wanted):
            problems.append(f"shard {index} seeds do not equal its own bank slice "
                            f"(offset {block['offset']}, size {block['size']})")
        for seed in got:
            if seed in seen_seeds:
                problems.append(f"episode seed {seed} appears in shards "
                                f"{seen_seeds[seed]} and {index}")
            seen_seeds[seed] = index
        if "shard" not in protocol:
            problems.append(f"shard {index} protocol carries no shard block")
        if (directory / "worker_failure.json").is_file():
            problems.append(f"shard {index} recorded a worker failure")
        process_path = directory / "process/process.json"
        process_record = read_json(process_path) if process_path.is_file() else {}
        protocols.append(protocol)
        metrics_documents.append(metrics)
        rows.extend(dict(row) for row in shard_rows)
        shard_receipts.append({
            "index": index, "directory": str(directory), "attempt": attempt_of(directory),
            "offset": int(block["offset"]), "size": int(block["size"]),
            "episode_seeds": got,
            "metrics_sha256": digest(metrics_path), "protocol_sha256": digest(protocol_path),
            "process_sha256": digest(process_path) if process_record else None,
            "process": {"status": process_record.get("status"),
                        "returncode": process_record.get("returncode"),
                        "started_at": process_record.get("started_at"),
                        "finished_at": process_record.get("finished_at"),
                        "elapsed_seconds": process_record.get("elapsed_seconds"),
                        "device": process_record.get("device_uuid")},
        })

    if problems:
        raise ShardMergeRefused("shard integrity check failed: " + "; ".join(problems))

    first = protocols[0]
    for index, protocol in enumerate(protocols[1:], start=1):
        differences = _protocol_differences(first, protocol)
        if differences:
            raise ShardMergeRefused(
                f"shard {index} did not run under the same frozen protocol as shard 0: {differences}")
    if len(rows) != episodes:
        raise ShardMergeRefused(f"merged {len(rows)} episodes, the request has {episodes}")
    if sorted(seen_seeds) != sorted(bank[:episodes]):
        raise ShardMergeRefused(
            "the union of shard seeds is not the requested seed bank (gap or extra episode)")

    for index, metrics in enumerate(metrics_documents):
        row_problems = _compare_summary(
            metrics.get("summary") or {},
            recompute_summary(metrics["episodes"],
                              timeout_action_steps=int(metrics["config"]["timeout_action_steps"])),
            where=f"shard {index}")
        if row_problems:
            raise ShardMergeRefused(
                "a shard's own summary disagrees with its rows, so the merge formula cannot be "
                "trusted: " + "; ".join(row_problems))

    timeout_action_steps = int(metrics_documents[0]["config"]["timeout_action_steps"])
    merged = dict(metrics_documents[0])
    config = dict(merged["config"])
    config["episode_count"] = episodes
    merged.update(config=config, summary=recompute_summary(rows, timeout_action_steps=timeout_action_steps),
                  episodes=[dict(row, episode_index=position) for position, row in enumerate(rows)],
                  created_at=now(), merged_from_shards=len(blocks))
    for key in ("purpose", "execution_mode", "harness", "policy_observation_contract", "rpc_timing_note"):
        merged[key] = _first_shard_value(metrics_documents, key)
    merged["seed_bank_sha256"] = object_digest([int(seed) for seed in bank[:episodes]])

    merged_protocol = {key: value for key, value in first.items() if key not in PER_SHARD_FIELDS}
    merged_protocol["episodes"] = episodes
    merged_protocol["shard_plan_sha256"] = digest(output / SHARD_PLAN)
    _publish(output, merged, merged_protocol, rows, shard_receipts, merge_command)
    return merged


def _first_shard_value(documents: Sequence[Mapping[str, Any]], key: str) -> Any:
    """Certification fields must be identical across shards; copy or refuse."""
    values = [document.get(key) for document in documents]
    if any(value != values[0] for value in values):
        raise ShardMergeRefused(f"certification field {key!r} differs across shards: {values}")
    return values[0]


def _publish(output: Path, merged: dict, protocol: dict, rows: Sequence[Mapping[str, Any]],
             shard_receipts: Sequence[Mapping[str, Any]],
             merge_command: Sequence[str]) -> None:
    telemetry, initializations = [], []
    for row in shard_receipts:
        directory = Path(row["directory"])
        telemetry.extend(_jsonl(directory / "telemetry.jsonl"))
        initializations.extend(_jsonl(directory / "initializations.jsonl"))
    if initializations:
        with (output / "initializations.jsonl").open("w", encoding="utf-8") as stream:
            for row in initializations:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if telemetry:
        with (output / "telemetry.jsonl").open("w", encoding="utf-8") as stream:
            for row in telemetry:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_json(output / "protocol.json", protocol)
    process = output / "process"
    process.mkdir(parents=True, exist_ok=True)
    stamps = [(row["process"].get("started_at"), row["process"].get("finished_at"))
              for row in shard_receipts]
    started = min((value for value, _ in stamps if value), default=now())
    finished = max((value for _, value in stamps if value), default=now())
    atomic_json(process / "process.json", {
        "command": list(merge_command), "cwd": str(Path.cwd()), "timeout": None,
        "request_hash": object_digest({"shards": [row["index"] for row in shard_receipts]}),
        "status": "completed", "returncode": 0,
        "started_at": started, "finished_at": finished,
        "elapsed_seconds": sum(float(row["process"].get("elapsed_seconds") or 0.0)
                               for row in shard_receipts),
        "pid": os.getpid(), "merged_from_shards": len(shard_receipts),
        "shards": list(shard_receipts),
    })
    atomic_json(output / MERGE_RECEIPT, {
        "schema_version": 1, "merged_at": now(), "shard_count": len(shard_receipts),
        "episode_count": len(rows), "shards": list(shard_receipts),
        "summary_formula_note": ("episode rows are copied verbatim; the mean inference latency is "
                                 "sum(episode totals)/sum(calls), which equals the official "
                                 "pairwise mean to floating-point associativity"),
    })
    atomic_json(output / "evaluation_metrics.json", merged)
