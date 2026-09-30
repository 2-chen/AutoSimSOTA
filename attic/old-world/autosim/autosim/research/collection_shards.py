"""Contiguous attempt blocks for a sharded collection, and the audit of what came back.

A collection draws one reset seed per attempt from ``RandomState(master_seed)``, so a shard
that burns ``offset`` identical draws and then collects behaves exactly like the serial
collector's ``offset``-th attempt onward -- the slice is exact, not an approximation.  That
lets one collection be spread over several devices as **blocks of the same attempt stream**:
shard ``k`` owns ``[offset_k, offset_k + size_k)``, the blocks tile the attempt budget, and
no attempt is made twice.

What this buys and what it does not:

* it buys wall-clock: the attempts are independent draws, so k devices finish a k-block
  budget in roughly the serial time of one block (plus one environment construction each);
* it does **not** reproduce the serial dataset.  Serial collection stops at a *global*
  success target, so which attempts get saved depends on how the earlier attempts went --
  a block shard stops at its own share of the target and at the end of its own block.  The
  saved episodes are therefore a different sample of the same declared distribution, and a
  shard whose block runs out first reports partial yield instead of topping itself up from
  a neighbouring block (that would be the selective re-run this pipeline forbids).  The
  divergence is recorded, per shard and in the merged result, rather than smoothed over.

The audit below is the one thing that must hold exactly: every shard's attempts are its own
block of the bank, the blocks do not overlap, and the union is the budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import atomic_json, object_digest, read_json
from .devices import shard_plan
from .evaluation_merge import ShardMergeRefused

COLLECTION_SHARD_PLAN = "collection_shard_plan.json"
SHARD_VARIANT = "contiguous_attempt_blocks"
DECLARED_DIVERGENCE = (
    "sharded collection covers the same attempt blocks but is not byte-identical to a "
    "serial collection: each shard stops at its own share of the success target and at the "
    "end of its own block, so the saved episodes are a different sample of the same "
    "declared collection distribution")


def collection_attempt_bank(master_seed: int, resets: int) -> list[int]:
    """The first ``resets`` reset seeds of the collector's own stream.

    ``CollectionSeedStream`` is ``RandomState(master_seed)`` plus one ``randint(0, 2**31-1)``
    per reset, which is deliberately the same construction as the evaluation bank: one
    definition covers both, and a shard can be audited against the exact stream it has to
    reproduce.  A block of ``n`` attempts therefore reads its seeds and its trailing reset
    from ``bank[offset : offset + n + 1]``.
    """
    from autosim.robosyn_data import evaluation_seed_bank

    return evaluation_seed_bank(master_seed, resets)


def build_collection_shard_plan(*, attempts: int, bank: Sequence[int],
                                selections: Sequence[Mapping[str, Any]],
                                target_episodes: int, master_seed: int) -> dict:
    """The frozen division: contiguous attempt blocks over the master seed bank.

    Written before any shard starts, so a resume cannot re-divide the attempts, and the bank
    digest proves the slices belong to the bank this process derives.
    """
    blocks = shard_plan(attempts, len(selections))
    per_shard_target, remainder = divmod(int(target_episodes), len(selections))
    return {
        "schema_version": 1, "kind": "collection_shard_plan", "variant": SHARD_VARIANT,
        "attempts": int(attempts), "count": len(selections),
        "target_episodes": int(target_episodes), "master_seed": int(master_seed),
        "bank_sha256": object_digest([int(seed) for seed in bank[:attempts]]),
        "declared_divergence": DECLARED_DIVERGENCE,
        "blocks": [
            {"index": index, "offset": offset, "size": size,
             "target_episodes": per_shard_target + (1 if index < remainder else 0),
             "device_uuid": selection.get("uuid"), "device_index": selection.get("index"),
             "selection": selection.get("selection")}
            for index, ((offset, size), selection) in enumerate(zip(blocks, selections))],
    }


def load_collection_shard_plan(output: Path) -> dict:
    path = Path(output) / COLLECTION_SHARD_PLAN
    if not path.is_file():
        raise ShardMergeRefused(
            f"no {COLLECTION_SHARD_PLAN} in {output}: a sharded collection must freeze its "
            f"division before running, so a resume cannot re-divide the attempts")
    return read_json(path)


def shard_dir(output: Path, index: int) -> Path:
    return Path(output) / "shards" / f"shard_{index:02d}"


def _startup_attempt(directory: Path) -> int:
    """1 for the shard root, k for ``startup_attempt_k``; the receipt records which ran."""
    name = Path(directory).name
    prefix = "startup_attempt_"
    return int(name[len(prefix):]) if name.startswith(prefix) and name[len(prefix):].isdigit() else 1


def certified_shard_root(output: Path, index: int) -> Path:
    """Where shard ``index``'s payload lives, or a refusal saying why not.

    A crash before the first reset is retried into ``startup_attempt_{k}`` (the crashed
    attempt's directory is left exactly as it was -- it is evidence), so a payload is
    accepted from the shard root or from *one* retry directory.  The benchmark writes its
    collection manifest in a ``finally`` block, so a manifest means the attempt stream was
    fully accounted for; two of them would mean the block ran twice and the merge cannot say
    which one it reports, so that is refused rather than resolved by picking a favourite.
    """
    root = shard_dir(output, index)
    found = [directory for directory in [root, *sorted(root.glob("startup_attempt_*"))]
             if (directory / "collection.json").is_file()
             and (directory / "bounded_collection_result.json").is_file()]
    if not found:
        raise ShardMergeRefused(
            f"shard {index} at {root} has no collection payload "
            f"(manifest={(root / 'collection.json').is_file()}, "
            f"result={(root / 'bounded_collection_result.json').is_file()}); "
            f"partial output is not merged")
    if len(found) > 1:
        raise ShardMergeRefused(
            f"shard {index} has {len(found)} collection payloads "
            f"({', '.join(str(directory.relative_to(output)) for directory in found)}); "
            f"the merge cannot tell which attempt block it reports")
    return found[0]


def _attempts_and_resets(manifest: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    attempts = [int(row.get("seed", -1)) for row in manifest.get("attempts", [])]
    resets = [int(row.get("seed", -1)) for row in manifest.get("resets", [])]
    return attempts, resets


def audit_shard_coverage(*, plan: Mapping[str, Any], bank: Sequence[int],
                         manifests: Mapping[int, Mapping[str, Any]],
                         offsets: Mapping[int, Mapping[str, Any]] | None = None) -> dict:
    """Verify the shards against the frozen plan; ``plan["master_seed"]`` is the authority.

    Every shard is its own block of the bank, blocks tile the budget, no attempt repeats.

    The one asymmetry: a shard that fills its block makes a trailing reset, and the
    benchmark reserves ``attempts + 1`` scene seeds for exactly that reason -- so the last
    reset seed of a full block equals the *next* block's first attempt seed.  Resets are not
    attempts, so this is expected rather than a collision, and only attempts are checked for
    overlap.

    ``offsets`` is the per-shard seed-offset receipt (see ``collection_worker``).  The
    attempt slices below already *imply* the offset was applied, but only up to a birthday
    collision between two draws of the same bank; the receipt is the direct evidence, and it
    is required wherever a block starts past the beginning of the stream.
    """
    problems: list[str] = []
    seen: dict[int, int] = {}
    rows = []
    for block in plan["blocks"]:
        index, offset, size = int(block["index"]), int(block["offset"]), int(block["size"])
        manifest = manifests.get(index)
        if manifest is None:
            problems.append(f"shard {index} produced no manifest")
            continue
        receipt = (offsets or {}).get(index)
        if offset > 0:
            if receipt is None:
                problems.append(
                    f"shard {index} declares a block starting at {offset} but left no "
                    f"seed-offset receipt: the block start was never evidenced")
            elif int(receipt.get("streams_seen", 0)) != 1:
                problems.append(
                    f"shard {index} never built its offset RandomState "
                    f"(streams_seen={receipt.get('streams_seen')}): it did not skip into its block")
            elif int(receipt.get("offset", -1)) != offset:
                problems.append(
                    f"shard {index} burned {receipt.get('offset')} draws, its block starts at {offset}")
            elif int(receipt.get("master_seed", -1)) != int(plan.get("master_seed", -1)):
                problems.append(f"shard {index} offset a different master seed")
        attempts, resets = _attempts_and_resets(manifest)
        expected = [int(seed) for seed in bank[offset:offset + size]]
        if attempts != expected[:len(attempts)]:
            problems.append(
                f"shard {index} attempts are not its block of the reserved stream "
                f"(offset {offset}, {len(attempts)} attempts)")
        if len(attempts) > size:
            problems.append(f"shard {index} made {len(attempts)} attempts over its block of {size}")
        if resets != [int(seed) for seed in bank[offset:offset + len(attempts) + 1]]:
            problems.append(f"shard {index} resets are not the reserved stream from its offset")
        if int(manifest.get("master_seed", -1)) != int(plan.get("master_seed", -1)):
            problems.append(f"shard {index} was collected from a different master seed")
        for seed in attempts:
            if seed in seen:
                problems.append(f"attempt seed {seed} appears in shards {seen[seed]} and {index}")
            seen[seed] = index
        rows.append({"index": index, "offset": offset, "size": size,
                     "attempts": len(attempts), "resets": len(resets),
                     "last_attempt_seed": attempts[-1] if attempts else None,
                     "streams_seen": None if receipt is None else receipt.get("streams_seen"),
                     "status": manifest.get("status")})
    planned = sum(int(block["size"]) for block in plan["blocks"])
    if planned != int(plan["attempts"]):
        problems.append(f"blocks cover {planned} attempts, not the declared {plan['attempts']}")
    consumed = sum(row["attempts"] for row in rows)
    return {
        "schema_version": 1, "kind": "collection_shard_coverage",
        "passed": not problems, "problems": problems, "shards": rows,
        "planned_attempts": planned, "consumed_attempts": consumed,
        "unspent_attempts": planned - consumed,
        "contiguous_block_coverage": not problems,
        "note": ("coverage and non-overlap only; a shard that stopped at its block end is "
                 "reported as partial yield, never completed from a neighbour's block"),
    }


def _shard_result(shard_root: Path) -> dict:
    path = shard_root / "bounded_collection_result.json"
    if not path.is_file():
        raise ShardMergeRefused(f"shard produced no bounded_collection_result.json: {shard_root}")
    return read_json(path)


def merge_collection_shards(*, output: Path, plan: Mapping[str, Any], bank: Sequence[int],
                            task: str, profile: str, collection_mode: str,
                            attempts: int, target_episodes: int) -> dict:
    """Verify every shard against the frozen plan and publish one merged result.

    The merged payload keeps the serial result's shape so its consumers (dataset admission,
    the mixture builder, the capability records) read it unchanged, and adds the per-shard
    rows: which device held which block, what it yielded, and whether it reached its share.
    """
    output = Path(output)
    manifests: dict[int, dict] = {}
    offsets: dict[int, dict] = {}
    shards: list[dict] = []
    for block in plan["blocks"]:
        index = int(block["index"])
        root = certified_shard_root(output, index)
        manifest_path = root / "collection.json"
        manifest = read_json(manifest_path)
        manifests[index] = manifest
        offset_path = root / "seed_offset.json"
        if offset_path.is_file():
            offsets[index] = read_json(offset_path)
        result = _shard_result(root)
        scene_path = root / "scene_evidence_audit.json"
        scene = read_json(scene_path) if scene_path.is_file() else {}
        accepted = int(result.get("accepted_episodes", 0))
        shards.append({
            "index": index, "offset": int(block["offset"]), "size": int(block["size"]),
            "target_episodes": int(block["target_episodes"]),
            "device_uuid": block.get("device_uuid"), "device_index": block.get("device_index"),
            "directory": str(root.resolve()), "startup_attempt": _startup_attempt(root),
            "attempts": int(result.get("attempts_consumed", len(manifest.get("attempts", [])))),
            "accepted_episodes": accepted,
            "capability_state": result.get("capability_state"),
            "termination": result.get("termination"),
            "dataset_root": result.get("dataset_root"),
            "manifest": str(manifest_path.resolve()),
            "seed_offset": str(offset_path.resolve()) if offset_path.is_file() else None,
            "scene_evidence_audit": str(scene_path.resolve()) if scene_path.is_file() else None,
            "requested_factor_readback": (scene.get("requested_factor_readback") or {}).get("status"),
            "shard_target_reached": accepted >= int(block["target_episodes"]),
        })
    coverage = audit_shard_coverage(plan=plan, bank=bank, manifests=manifests, offsets=offsets)
    atomic_json(output / "collection_shard_coverage.json", coverage)
    if not coverage["passed"]:
        raise ShardMergeRefused(
            f"collection shard coverage audit failed: {coverage['problems']}")

    accepted = sum(row["accepted_episodes"] for row in shards)
    consumed = sum(row["attempts"] for row in shards)
    readbacks = {row["requested_factor_readback"] for row in shards}
    readback_status = ("verified" if readbacks == {"verified"}
                       else "failed" if "failed" in readbacks
                       else "unknown")
    roots = [row["dataset_root"] for row in shards if row["dataset_root"]]
    if len(roots) != len(shards):
        raise ShardMergeRefused(f"shard without a dataset root: {shards}")
    partial = [row["index"] for row in shards if not row["shard_target_reached"]]
    atomic_json(output / "scene_evidence_audit.json", {
        "schema_version": 1, "kind": "collection_scene_evidence_audit_merged",
        "passed": bool(shards) and all((read_json(Path(row["scene_evidence_audit"])).get("passed")
                                        if row["scene_evidence_audit"] else False)
                                       for row in shards),
        "task": task, "requested_profile": profile, "attempt_count": consumed,
        "joined_attempt_count": consumed, "shards": [row["index"] for row in shards],
        "requested_factor_readback": {"status": readback_status,
                                      "per_shard": [row["requested_factor_readback"]
                                                    for row in shards]},
    })
    merged = {
        "schema_version": 1, "kind": "bounded_collection_result",
        "task": task, "profile": profile, "collection_mode": collection_mode,
        "attempt_budget": int(attempts), "target_episodes": int(target_episodes),
        "attempts_consumed": consumed, "accepted_episodes": accepted,
        "capability_state": ("target_reached" if accepted >= int(target_episodes)
                             else "partial_yield" if accepted else "no_observed_yield"),
        "termination": "target_reached" if accepted >= int(target_episodes)
                       else "attempt_budget_exhausted",
        "dataset_root": roots[0], "dataset_roots": roots,
        "manifest": shards[0]["manifest"],
        "scene_evidence_audit": str((output / "scene_evidence_audit.json").resolve()),
        "zero_yield_does_not_prove_incapability": accepted == 0,
        "shard_count": len(shards), "collection_shard_variant": SHARD_VARIANT,
        "shards": shards,
        "coverage": {"passed": coverage["passed"],
                     "path": str((output / "collection_shard_coverage.json").resolve()),
                     "consumed_attempts": coverage["consumed_attempts"],
                     "planned_attempts": coverage["planned_attempts"],
                     "unspent_attempts": coverage["unspent_attempts"]},
        "shards_below_their_share": partial,
        "not_byte_identical_to_serial": True,
        "declared_divergence": DECLARED_DIVERGENCE,
    }
    atomic_json(output / "bounded_collection_result.json", merged)
    return merged
