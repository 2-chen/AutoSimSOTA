"""A spread collection: what it divides, what it refuses, and what it admits it changed.

The collector is faked, but only at its *process boundary*: the fake reads
``--collection_seed`` / ``--collection_seed_offset`` / ``--collection_max_attempts`` /
``--max_episodes`` exactly like ``RoboSynChallenge/scripts/run_env.py`` and writes the same
manifest and scene traces it writes, deriving its attempt seeds from the same stream
construction.  The block arithmetic, the audit and the merge are therefore exercised against
real seed values rather than against a fixture shaped to fit them.

What this file cannot and does not claim: byte-identity with a serial collection.  A spread
collection stops each block at its own share of the success target, so it samples the same
declared distribution differently.  The audit below is the part that must hold exactly --
every attempt seed is its block's, the blocks tile the budget, and no attempt repeats.
"""

import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy

from autosim.research.collection_shards import (audit_shard_coverage,
                                                build_collection_shard_plan,
                                                certified_shard_root,
                                                collection_attempt_bank,
                                                merge_collection_shards, shard_dir)
from autosim.research.collection_worker import _pop_seed_offset, seed_offset_stream
from autosim.research.common import atomic_json, object_digest
from autosim.research.devices import SimDeviceSelection, shard_plan
from autosim.research.evaluation_merge import ShardMergeRefused
from autosim.research.registry import TaskSpec
from autosim.research.runtime import Runtime

MASTER_SEED = 1017355756
COST_MODEL = {"construction_seconds": 360.0, "marginal_seconds": 40.0}
SPEC = TaskSpec(name="click_bell", env_id="click_bell", setting="random",
                max_episode_steps=361, state_dim=14, action_dim=14,
                cameras=("cam_high",), camera_shapes={"cam_high": [3, 240, 320]},
                control_parts=("left_arm",), recorded_fps=30.0, instruction="press the bell",
                gym_config="gym.xml", action_config="action.json", config_hashes={},
                event_families={}, roles={})


def succeeded(seed: int) -> bool:
    """A deterministic per-seed outcome, the way a real attempt is a function of its seed."""
    return seed % 3 == 0


def scene_row(seed: int, profile: str, *, task: str = SPEC.name) -> dict:
    return {"event": "collection_scene_reset", "seed": seed, "task": task,
            "requested_profile": profile, "entities": {"button": {"pose": [[seed]]}},
            "robot_qpos": [0.0], "privileged_training_diagnostics_only": True}


def write_collection(directory: Path, *, offset: int = 0, size: int = 0, target: int,
                     master_seed: int = MASTER_SEED, profile: str = "full_random",
                     collection_mode: str = "expert", dataset: str | None = None,
                     status: str | None = None, seeds: list[int] | None = None) -> dict:
    """What the benchmark leaves behind for one attempt block, field for field.

    ``seeds`` is the block's own reset stream as the collector generated it -- ``size + 1``
    draws, of which the first ``k`` become attempts and the last is the trailing reset.  It
    is passed in by the fake process, which burns the stream through the real shim, so the
    manifest never derives its seeds from the slice the audit is about to expect.
    """
    if seeds is None:
        bank = collection_attempt_bank(master_seed, offset + size + 1)
        seeds = bank[offset:offset + size + 1]
    size = len(seeds) - 1 if seeds else size
    attempts, saved = [], []
    for index in range(size):
        if len(saved) >= target:
            break
        seed = seeds[index]
        row = {"seed": seed, "saved": succeeded(seed)}
        attempts.append(row)
        if row["saved"]:
            saved.append(seed)
    exhausted = status == "failed" or (status is None and len(saved) < target)
    manifest = {
        "schema_version": 2, "kind": "robosyn_expert_collection",
        "status": "failed" if exhausted else "completed",
        "error": (f"RuntimeError: Collection exceeded {size} expert attempts."
                  if exhausted else None),
        "started_at": "2026-09-13T10:00:00+00:00",
        "finished_at": "2026-09-13T10:30:00+00:00",
        "task": SPEC.name, "profile": profile, "collection_mode": collection_mode,
        "master_seed": master_seed, "target_successful_episodes": target,
        "successful_episode_seeds": saved,
        "failed_attempt_seeds": [row["seed"] for row in attempts if not row["saved"]],
        "expert_attempt_count": len(attempts),
        "expert_attempt_success_rate": (len(saved) / len(attempts)) if attempts else None,
        "attempts": attempts,
        "resets": [{"seed": seed, "reason": "collection_reset"}
                   for seed in seeds[:len(attempts) + 1]],
        "dataset_paths": [str(dataset or (directory / "data"))],
        "source_configs": {"gym_config": "gym.xml", "action_config": "action.json"},
        "lineage": {"training_only": True},
    }
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "collection.json", manifest)
    with (directory / "scene_resets.jsonl").open("w", encoding="utf-8") as stream:
        for reset in manifest["resets"]:
            stream.write(json.dumps(scene_row(reset["seed"], profile)) + "\n")
    return manifest


class FixtureRuntime(Runtime):
    """A Runtime whose collection subprocess is replaced by the benchmark's file contract.

    Everything shared with the per-job copies ``for_job`` builds lives on the class:
    ``dataclasses.replace`` carries fields over, not instance attributes.
    """

    records: list = []
    crashing: dict = {}
    crash_every_shard: bool = False
    crash_attempts: int = 1
    delay: float = 0.0
    in_flight: int = 0
    peak_in_flight: int = 0
    gate = threading.Lock()

    @classmethod
    def block_stream(cls, *, directory: Path, offset: int, size: int, master: int) -> list[int]:
        """This block's reset stream, burned the way ``collection_worker.main`` burns it.

        The shim is installed and removed under the class lock because a real shard is its
        own *process* while this fixture's shards are threads in one test process.  The seeds
        still come from the shifted stream, so a burn that does not land on the block start
        ends up in the manifest and the coverage audit refuses it -- rather than the fixture
        being shaped to match whatever the audit expects.
        """
        if not offset:
            return collection_attempt_bank(master, size + 1)
        with cls.gate:
            saved = numpy.random.RandomState
            try:
                seed_offset_stream(offset=offset, master_seed=master,
                                   receipt=directory / "seed_offset.json")
                state = numpy.random.RandomState(master)
                return [int(state.randint(0, 2**31 - 1)) for _ in range(size + 1)]
            finally:
                numpy.random.RandomState = saved

    def run(self, command, output, timeout, **kwargs):
        with type(self).gate:
            type(self).in_flight += 1
            type(self).peak_in_flight = max(type(self).peak_in_flight, type(self).in_flight)
        try:
            return self._run_contract(command, output)
        finally:
            with type(self).gate:
                type(self).in_flight -= 1

    def _run_contract(self, command, output):
        manifest_path = Path(command[command.index("--collection_manifest") + 1])
        master = int(command[command.index("--collection_seed") + 1])
        offset = (int(command[command.index("--collection_seed_offset") + 1])
                  if "--collection_seed_offset" in command else 0)
        size = int(command[command.index("--collection_max_attempts") + 1])
        target = int(command[command.index("--max_episodes") + 1])
        profile = command[command.index("--collection_profile") + 1]
        directory = manifest_path.parent
        selection = self.selection
        entry = {"uuid": selection.uuid if selection else None,
                 "cuda_visible": selection.cuda_visible if selection else None,
                 "gpu_id": ([command[index + 1] for index, token in enumerate(command)
                             if token == "--gpu_id"] or [None])[0],
                 "offset": offset, "size": size, "target": target,
                 "directory": str(directory), "started": time.monotonic()}
        type(self).records.append(entry)
        # Keyed by shard, not by directory: a retry writes into startup_attempt_k and the
        # crash budget is per block, not per directory.
        shard = (directory.parent.name if directory.name.startswith("startup_attempt_")
                 else directory.name)
        crashed = type(self).crashing.get(shard, 0)
        if type(self).crash_every_shard and crashed < type(self).crash_attempts:
            type(self).crashing[shard] = crashed + 1
            entry["failed"] = True
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory / "process" / "process.json",
                        {"status": "failed", "returncode": -11, "pid": 4242})
            raise RuntimeError("synthetic native abort before the first collection reset")
        write_collection(directory, size=size, target=target, master_seed=master,
                         profile=profile, seeds=self.block_stream(
                             directory=directory, offset=offset, size=size, master=master))
        atomic_json(directory / "process" / "process.json",
                    {"status": "completed", "returncode": 0, "pid": 4242})
        time.sleep(type(self).delay)
        entry["finished"] = time.monotonic()
        return {"status": "completed"}


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.bank = collection_attempt_bank(MASTER_SEED, 41)

    def plan(self, *, attempts=40, target=10, devices=3):
        selections = [{"index": index, "uuid": f"GPU-{index}", "selection": None}
                      for index in range(devices)]
        return build_collection_shard_plan(attempts=attempts, bank=self.bank,
                                           selections=selections, target_episodes=target,
                                           master_seed=MASTER_SEED)

    def test_the_blocks_tile_the_budget_and_the_target_is_divided_without_loss(self):
        plan = self.plan()
        self.assertEqual([(block["offset"], block["size"]) for block in plan["blocks"]],
                         shard_plan(40, 3))
        self.assertEqual([block["target_episodes"] for block in plan["blocks"]], [4, 3, 3])
        self.assertEqual(sum(block["target_episodes"] for block in plan["blocks"]), 10)
        self.assertEqual(sum(block["size"] for block in plan["blocks"]), 40)
        self.assertEqual(plan["bank_sha256"], object_digest(self.bank[:40]))
        self.assertEqual(plan["master_seed"], MASTER_SEED)

    def test_a_shard_never_owes_more_episodes_than_it_has_attempts(self):
        for attempts in (8, 9, 12, 33, 40, 100):
            for target in (1, attempts // 3 or 1, attempts - 1, attempts):
                for devices in (2, 3, 4):
                    plan = self.plan(attempts=attempts, target=target, devices=devices)
                    for block in plan["blocks"]:
                        self.assertLessEqual(block["target_episodes"], block["size"],
                                             f"attempts={attempts} target={target} n={devices}")
                    self.assertEqual(sum(block["target_episodes"] for block in plan["blocks"]),
                                     target)

    def test_the_variant_and_its_declared_divergence_are_part_of_the_frozen_plan(self):
        plan = self.plan()
        self.assertEqual(plan["variant"], "contiguous_attempt_blocks")
        self.assertIn("not byte-identical", plan["declared_divergence"])


class ManifestCase(unittest.TestCase):
    def setUp(self):
        self._temp = TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.bank = collection_attempt_bank(MASTER_SEED, 41)

    def plan(self, *, attempts=40, target=10, devices=3):
        selections = [{"index": index, "uuid": f"GPU-{index}", "selection": None}
                      for index in range(devices)]
        return build_collection_shard_plan(attempts=attempts, bank=self.bank,
                                           selections=selections, target_episodes=target,
                                           master_seed=MASTER_SEED)

    def manifests(self, plan, offsets=None, **overrides):
        """One manifest per block, written the way the benchmark writes them."""
        manifests, receipts = {}, {}
        for block in plan["blocks"]:
            offset, size = int(block["offset"]), int(block["size"])
            target = int(block["target_episodes"])
            kwargs = dict(offset=offset, size=size, target=target, profile="full_random")
            kwargs.update(overrides.pop(str(block["index"]), {}))
            manifests[int(block["index"])] = write_collection(
                self.root / f"shard_{block['index']:02d}", **kwargs)
            receipts[int(block["index"])] = {"streams_seen": 1 if offset else 0,
                                             "offset": offset, "master_seed": MASTER_SEED}
        if offsets is not None:
            receipts.update(offsets)
        return manifests, receipts


class CoverageAuditTests(ManifestCase):
    def test_the_blocks_tile_the_budget_without_overlap_and_the_audit_says_so(self):
        plan = self.plan()
        manifests, receipts = self.manifests(plan)
        audit = audit_shard_coverage(plan=plan, bank=self.bank, manifests=manifests,
                                     offsets=receipts)
        self.assertTrue(audit["passed"], audit["problems"])
        self.assertEqual(audit["planned_attempts"], 40)
        self.assertTrue(audit["contiguous_block_coverage"])
        self.assertEqual([row["streams_seen"] for row in audit["shards"]], [0, 1, 1])
        # A shard stops at its own share of the target, so it spends a prefix of its block:
        # the audit requires every attempt to be its block's, not that the block is spent.
        for row in audit["shards"]:
            self.assertLessEqual(row["attempts"], row["size"])
        self.assertEqual(audit["consumed_attempts"],
                         sum(row["attempts"] for row in audit["shards"]))
        self.assertEqual(audit["unspent_attempts"],
                         40 - audit["consumed_attempts"])
        attempts = [entry["seed"] for manifest in manifests.values()
                    for entry in manifest["attempts"]]
        self.assertEqual(len(attempts), len(set(attempts)))
        self.assertTrue(set(attempts) <= set(self.bank[:40]))

    def test_a_shard_that_started_from_the_top_of_the_stream_is_refused(self):
        plan = self.plan()
        manifests, receipts = self.manifests(plan)
        # The bug this audit exists for: every shard collects from the beginning, so the
        # blocks are not its own and the same attempts are made on every device.
        manifests[1] = write_collection(self.root / "shard_01", offset=0, size=13, target=3)
        receipts[1] = {"streams_seen": 0, "offset": 0, "master_seed": MASTER_SEED}
        audit = audit_shard_coverage(plan=plan, bank=self.bank, manifests=manifests,
                                     offsets=receipts)
        self.assertFalse(audit["passed"])
        self.assertIn("not its block of the reserved stream",
                      " ".join(audit["problems"]))
        self.assertIn("appears in shards", " ".join(audit["problems"]))
        self.assertIn("never built its offset RandomState", " ".join(audit["problems"]))

    def test_a_block_start_without_a_receipt_is_a_problem_not_an_assumption(self):
        plan = self.plan()
        manifests, receipts = self.manifests(plan)
        audit = audit_shard_coverage(plan=plan, bank=self.bank, manifests=manifests,
                                     offsets={0: receipts[0]})
        self.assertFalse(audit["passed"])
        self.assertEqual([row["index"] for row in audit["shards"]
                          if row["streams_seen"] is None], [1, 2])

    def test_a_receipt_that_never_matched_the_stream_is_a_problem(self):
        plan = self.plan()
        manifests, receipts = self.manifests(plan)
        receipts[2] = {"streams_seen": 0, "offset": 27, "master_seed": MASTER_SEED}
        audit = audit_shard_coverage(plan=plan, bank=self.bank, manifests=manifests,
                                     offsets=receipts)
        self.assertFalse(audit["passed"])
        self.assertIn("never built its offset RandomState", " ".join(audit["problems"]))

    def test_an_offset_from_a_different_master_seed_is_refused(self):
        plan = self.plan()
        manifests, receipts = self.manifests(plan)
        receipts[1] = {"streams_seen": 1, "offset": 14, "master_seed": MASTER_SEED + 1}
        audit = audit_shard_coverage(plan=plan, bank=self.bank, manifests=manifests,
                                     offsets=receipts)
        self.assertIn("offset a different master seed", " ".join(audit["problems"]))

    def test_a_manifest_collected_under_another_master_seed_is_refused(self):
        plan = self.plan()
        manifests, receipts = self.manifests(plan)
        manifests[2] = write_collection(self.root / "shard_02", offset=27, size=13, target=3,
                                        master_seed=MASTER_SEED + 7)
        audit = audit_shard_coverage(plan=plan, bank=self.bank, manifests=manifests,
                                     offsets=receipts)
        self.assertFalse(audit["passed"])
        self.assertIn("different master seed", " ".join(audit["problems"]))

    def test_a_missing_manifest_is_named_rather_than_skipped(self):
        plan = self.plan()
        manifests, receipts = self.manifests(plan)
        del manifests[1]
        audit = audit_shard_coverage(plan=plan, bank=self.bank, manifests=manifests,
                                     offsets=receipts)
        self.assertIn("shard 1 produced no manifest", audit["problems"])

    def test_blocks_that_do_not_cover_the_declared_budget_are_refused(self):
        plan = self.plan()
        manifests, receipts = self.manifests(plan)
        plan["blocks"][1]["size"] = 12
        audit = audit_shard_coverage(plan=plan, bank=self.bank, manifests=manifests,
                                     offsets=receipts)
        self.assertFalse(audit["passed"])
        self.assertIn("blocks cover 39 attempts, not the declared 40", audit["problems"])

    def test_a_shard_that_ran_past_its_block_is_refused(self):
        # Shard 0 was handed a 15-attempt block where the frozen plan gave it 14, and it
        # spent all 15 -- the wiring bug this check exists for (every shard getting the
        # whole budget).  Its target has to be unreachable within the block, or the shard
        # would have stopped at its share and never reached the 15th attempt.
        plan = self.plan()
        manifests, receipts = self.manifests(plan)
        manifests[0] = write_collection(self.root / "shard_00", offset=0, size=15, target=15)
        audit = audit_shard_coverage(plan=plan, bank=self.bank, manifests=manifests,
                                     offsets=receipts)
        self.assertFalse(audit["passed"])
        self.assertIn("over its block of 14", " ".join(audit["problems"]))
        self.assertIn("not its block of the reserved stream", " ".join(audit["problems"]))


class CertifiedShardRootTests(ManifestCase):
    def payload(self, output, index, *, attempt=1, directory=None):
        root = shard_dir(output, index) / ("" if attempt == 1 else f"startup_attempt_{attempt}")
        write_collection(root, offset=0, size=4, target=1)
        atomic_json(root / "bounded_collection_result.json", {"accepted_episodes": 1})
        return root

    def test_a_payload_is_accepted_from_the_shard_root_or_from_one_retry(self):
        root = self.payload(self.root, 0)
        self.assertEqual(certified_shard_root(self.root, 0), root)
        retried = self.payload(self.root, 1, attempt=2)
        self.assertEqual(certified_shard_root(self.root, 1), retried)

    def test_a_shard_with_no_payload_is_refused_with_what_is_missing(self):
        (shard_dir(self.root, 0)).mkdir(parents=True)
        with self.assertRaises(ShardMergeRefused) as caught:
            certified_shard_root(self.root, 0)
        self.assertIn("partial output is not merged", str(caught.exception))

    def test_two_payloads_from_one_shard_are_refused_rather_than_guessed(self):
        self.payload(self.root, 0)
        self.payload(self.root, 0, attempt=2)
        with self.assertRaises(ShardMergeRefused) as caught:
            certified_shard_root(self.root, 0)
        self.assertIn("cannot tell which attempt block", str(caught.exception))


class MergeTests(ManifestCase):
    def merge(self, *, target=10, attempts=40, mutate=None, plan=None):
        plan = plan or self.plan(attempts=attempts, target=target)
        output = self.root / "out"
        output.mkdir(parents=True, exist_ok=True)
        for block in plan["blocks"]:
            index = int(block["index"])
            root = shard_dir(output, index)
            write_collection(root, offset=int(block["offset"]), size=int(block["size"]),
                             target=int(block["target_episodes"]))
            result = {"accepted_episodes": int(block["target_episodes"]),
                      "attempts_consumed": int(block["size"]),
                      "capability_state": "target_reached", "termination": "target_reached",
                      "dataset_root": str(root / "data")}
            atomic_json(root / "bounded_collection_result.json", result)
            atomic_json(root / "scene_evidence_audit.json",
                        {"passed": True, "requested_factor_readback": {"status": "verified"}})
            atomic_json(root / "seed_offset.json",
                        {"streams_seen": 1 if block["offset"] else 0,
                         "offset": int(block["offset"]), "master_seed": MASTER_SEED})
        if mutate is not None:
            mutate(output, plan)
        return merge_collection_shards(output=output, plan=plan, bank=self.bank,
                                       task=SPEC.name, profile="full_random",
                                       collection_mode="expert", attempts=attempts,
                                       target_episodes=target), output

    def test_the_merged_result_adds_up_and_keeps_the_serial_shape(self):
        merged, output = self.merge()
        self.assertEqual(merged["kind"], "bounded_collection_result")
        self.assertEqual(merged["accepted_episodes"], 10)
        self.assertEqual(merged["attempts_consumed"], 40)
        self.assertEqual(merged["capability_state"], "target_reached")
        self.assertEqual(merged["dataset_roots"],
                         [str(shard_dir(output, index) / "data") for index in range(3)])
        self.assertEqual(merged["dataset_root"], merged["dataset_roots"][0])
        self.assertEqual(merged["manifest"],
                         str((shard_dir(output, 0) / "collection.json").resolve()))
        self.assertEqual([row["startup_attempt"] for row in merged["shards"]], [1, 1, 1])
        self.assertEqual(merged["shards_below_their_share"], [])
        self.assertTrue(merged["coverage"]["passed"])
        self.assertTrue((output / "collection_shard_coverage.json").is_file())
        self.assertTrue(merged["not_byte_identical_to_serial"])
        self.assertEqual(merged["collection_shard_variant"], "contiguous_attempt_blocks")
        # The serial result's consumers read these two files by path, from the merge output.
        self.assertTrue((output / "bounded_collection_result.json").is_file())
        self.assertTrue((output / "scene_evidence_audit.json").is_file())

    def test_a_shard_below_its_share_is_reported_and_lowers_the_merged_state(self):
        def mutate(output, plan):
            root = shard_dir(output, 1)
            result = json.loads((root / "bounded_collection_result.json").read_text())
            result.update(accepted_episodes=1, capability_state="partial_yield",
                          termination="attempt_budget_exhausted")
            atomic_json(root / "bounded_collection_result.json", result)
        merged, _ = self.merge(mutate=mutate)
        self.assertEqual([row["index"] for row in merged["shards"]
                          if not row["shard_target_reached"]], [1])
        self.assertEqual(merged["shards_below_their_share"], [1])
        self.assertEqual(merged["accepted_episodes"], 8)
        self.assertEqual(merged["capability_state"], "partial_yield")
        self.assertEqual(merged["termination"], "attempt_budget_exhausted")

    def test_a_merged_readback_is_verified_only_if_every_shard_verified(self):
        def mutate(output, plan):
            atomic_json(shard_dir(output, 2) / "scene_evidence_audit.json",
                        {"passed": True, "requested_factor_readback": {"status": "failed"}})
        merged, _ = self.merge(mutate=mutate)
        self.assertEqual(merged["shards"][2]["requested_factor_readback"], "failed")
        audit = json.loads((self.root / "out/scene_evidence_audit.json").read_text())
        self.assertEqual(audit["requested_factor_readback"]["status"], "failed")
        # Two separate claims: the seed-to-scene *join* held in every shard (which is what
        # gates the merge), while the *targeting* readback did not in one of them.  A failed
        # readback makes a targeted profile inadmissible downstream; it does not retract the
        # evidence that the attempts were joined to realized scene state.
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["requested_factor_readback"]["per_shard"],
                         ["verified", "verified", "failed"])

    def test_a_shard_without_a_dataset_root_is_refused(self):
        def mutate(output, plan):
            root = shard_dir(output, 0)
            result = json.loads((root / "bounded_collection_result.json").read_text())
            atomic_json(root / "bounded_collection_result.json",
                        dict(result, dataset_root=None))
            # The audit runs before the roots are collected, so this must not be reached by
            # a shard whose attempts are right: the refusal is about the payload, not the block.
        with self.assertRaises(ShardMergeRefused) as caught:
            self.merge(mutate=mutate)
        self.assertIn("shard without a dataset root", str(caught.exception))

    def test_a_failed_coverage_audit_never_publishes_a_merged_result(self):
        def mutate(output, plan):
            root = shard_dir(output, 1)
            manifest = json.loads((root / "collection.json").read_text())
            manifest["attempts"] = manifest["attempts"][:2]
            atomic_json(root / "collection.json", manifest)
        with self.assertRaises(ShardMergeRefused) as caught:
            self.merge(mutate=mutate)
        self.assertIn("coverage audit failed", str(caught.exception))
        self.assertFalse((self.root / "out/bounded_collection_result.json").exists())


class SeedOffsetTests(unittest.TestCase):
    """The shim that starts the collector's stream mid-bank, and the receipt it leaves."""

    def setUp(self):
        self._temp = TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self._real = numpy.random.RandomState
        self.addCleanup(self._restore)

    def _restore(self):
        numpy.random.RandomState = self._real

    def draw(self, seed, count=1):
        state = numpy.random.RandomState(seed)
        return [int(state.randint(0, 2**31 - 1)) for _ in range(count)]

    def test_the_first_draw_is_the_block_start_and_the_stream_continues_from_there(self):
        bank = collection_attempt_bank(MASTER_SEED, 48)
        for offset in (0, 1, 5, 17, 40):
            numpy.random.RandomState = self._real
            seed_offset_stream(offset=offset, master_seed=MASTER_SEED)
            self.assertEqual(self.draw(MASTER_SEED, 2), bank[offset:offset + 2],
                             f"offset {offset} did not land on the block start")
        self._restore()

    def test_only_the_collection_stream_is_shifted(self):
        numpy.random.RandomState = self._real
        seed_offset_stream(offset=9, master_seed=MASTER_SEED)
        # The correction path's own stream (collection_seed ^ 0x5A17C0DE) must stay put:
        # shifting it would silently re-randomize prefix lengths the block does not own.
        self.assertEqual(self.draw(MASTER_SEED ^ 0x5A17C0DE), self.draw(MASTER_SEED ^ 0x5A17C0DE))
        self.assertEqual(self.draw(MASTER_SEED ^ 0x5A17C0DE),
                         collection_attempt_bank(MASTER_SEED ^ 0x5A17C0DE, 1))
        self._restore()

    def test_a_second_collection_stream_is_refused_rather_than_double_burned(self):
        seed_offset_stream(offset=3, master_seed=MASTER_SEED)
        self.draw(MASTER_SEED)
        with self.assertRaises(RuntimeError) as caught:
            self.draw(MASTER_SEED)
        self.assertIn("second RandomState(master_seed)", str(caught.exception))
        self._restore()

    def test_the_receipt_records_that_the_block_start_was_actually_performed(self):
        receipt = self.root / "seed_offset.json"
        seed_offset_stream(offset=17, master_seed=MASTER_SEED, receipt=receipt)
        installed = json.loads(receipt.read_text())
        self.assertEqual((installed["streams_seen"], installed["burned_draws"]), (0, 0))
        self.draw(MASTER_SEED)
        seen = json.loads(receipt.read_text())
        self.assertEqual((seen["streams_seen"], seen["burned_draws"], seen["offset"]),
                         (1, 17, 17))
        self._restore()

    def test_the_offset_flag_is_removed_before_run_envs_parser_sees_it(self):
        argv = ["run_env.py", "--collection_seed", "7", "--collection_seed_offset", "12",
                "--collection_quiet"]
        self.assertEqual(_pop_seed_offset(argv), 12)
        self.assertEqual(argv, ["run_env.py", "--collection_seed", "7", "--collection_quiet"])
        self.assertEqual(_pop_seed_offset(["run_env.py"]), 0)
        with self.assertRaises(ValueError):
            _pop_seed_offset(["run_env.py", "--collection_seed_offset", "-1"])


class ShardedCollectionTests(unittest.TestCase):
    """The Runtime-level runner: what it divides, what it retries, what it refuses."""

    def setUp(self):
        self._temp = TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        FixtureRuntime.records = []
        FixtureRuntime.crashing = {}
        FixtureRuntime.crash_every_shard = False
        FixtureRuntime.crash_attempts = 1
        FixtureRuntime.delay = 0.0
        FixtureRuntime.in_flight = FixtureRuntime.peak_in_flight = 0
        # The per-shard expert contract digests the benchmark's own collector entry point,
        # so the fixture needs a real file there even though no process is ever spawned.
        self.repo = self.root / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "scripts/run_env.py").write_text("# fixture collector\n", encoding="utf-8")
        self.runtime = FixtureRuntime(self.root, self.root / "run", repo_path=self.repo)
        self.runtime.cost_model = COST_MODEL
        self.runtime.shard_min_episodes = 8

    def plan(self, devices, **overrides):
        plan = {"mode": "pinned_index", "max_parallel_jobs": devices, "usable": [
            {"index": index, "uuid": f"GPU-{index}", "model": "NVIDIA GeForce RTX 5090",
             "selection": {"mode": "pinned_index", "index": index, "uuid": f"GPU-{index}",
                           "vulkan_gpu_id": index, "torch_index": 0, "cuda_visible": str(index),
                           "renderer": "hybrid", "extra_env": {}}}
            for index in range(devices)]}
        plan.update(overrides)
        return plan

    def collect(self, *, devices=3, attempts=40, target=10, destination=None):
        self.runtime.plan = self.plan(devices)
        return self.runtime.collect_bounded(
            SPEC, Path(destination or self.root / "collection"), attempt_budget=attempts,
            target_episodes=target, master_seed=MASTER_SEED, profile="full_random",
            collection_mode="expert", timeout=600)

    def test_the_attempt_budget_is_divided_over_the_devices_and_merged_back_whole(self):
        merged = self.collect(devices=3, attempts=40, target=10)
        output = self.root / "collection"
        plan = json.loads((output / "collection_shard_plan.json").read_text())
        self.assertEqual([(block["offset"], block["size"]) for block in plan["blocks"]],
                         shard_plan(40, 3))
        self.assertEqual(len(FixtureRuntime.records), 3)
        self.assertEqual(sorted(record["offset"] for record in FixtureRuntime.records), [0, 14, 27])
        self.assertEqual(sorted(record["gpu_id"] for record in FixtureRuntime.records),
                         ["0", "1", "2"])                    # one block per device flag
        self.assertEqual(merged["shard_count"], 3)
        bank = collection_attempt_bank(MASTER_SEED, 41)
        self.assertEqual(merged["coverage"]["planned_attempts"], 40)
        self.assertEqual(merged["attempts_consumed"], merged["coverage"]["consumed_attempts"])
        self.assertEqual(merged["dataset_roots"],
                         [str(shard_dir(output, index) / "data") for index in range(3)])
        # Each block spent only as much of itself as its own share needed, and every attempt
        # it did make is its own block's -- that is the invariant the whole variant rests on.
        attempts = []
        for row in merged["shards"]:
            seeds = [int(entry["seed"]) for entry in
                     json.loads(Path(row["manifest"]).read_text())["attempts"]]
            self.assertEqual(seeds, bank[row["offset"]:row["offset"] + len(seeds)])
            self.assertLessEqual(len(seeds), row["size"])
            attempts.extend(seeds)
        self.assertEqual(len(attempts), merged["attempts_consumed"])
        self.assertEqual(len(attempts), len(set(attempts)))

    def test_the_blocks_run_at_the_same_time_on_their_own_devices(self):
        FixtureRuntime.delay = 0.15
        merged = self.collect(devices=3, attempts=40, target=10)
        self.assertGreaterEqual(FixtureRuntime.peak_in_flight, 2)
        spans = [(record["started"], record["finished"]) for record in FixtureRuntime.records]
        overlap = min(end for _, end in spans) - max(start for start, _ in spans)
        self.assertGreater(overlap, 0.0)                    # real concurrency, not just threads
        self.assertEqual(len({record["uuid"] for record in FixtureRuntime.records}),
                         merged["shard_count"])

    def test_a_single_device_plan_runs_one_process_with_no_shard_artifacts(self):
        merged = self.collect(devices=1)
        output = self.root / "collection"
        self.assertEqual(len(FixtureRuntime.records), 1)
        self.assertIsNone(FixtureRuntime.records[0]["uuid"])
        self.assertFalse((output / "collection_shard_plan.json").exists())
        self.assertFalse((output / "shards").exists())
        self.assertNotIn("shard_count", merged)                    # the serial result's shape
        self.assertEqual(merged["kind"], "bounded_collection_result")
        self.assertEqual(merged["capability_state"], "target_reached")
        manifest = json.loads((output / "collection.json").read_text())
        seeds = [int(row["seed"]) for row in manifest["attempts"]]
        self.assertEqual(seeds, collection_attempt_bank(MASTER_SEED, 41)[:len(seeds)])
        self.assertEqual(merged["attempts_consumed"], len(seeds))
        self.assertLess(len(seeds), 40)                            # it stopped at its target

    def test_a_shard_that_died_before_its_first_reset_is_retried_from_its_block_start(self):
        FixtureRuntime.crash_every_shard = True
        FixtureRuntime.crash_attempts = 2                    # attempt 1 and 2 crash, 3 works
        merged = self.collect(devices=3, attempts=40, target=10)
        self.assertEqual([row["startup_attempt"] for row in merged["shards"]], [3, 3, 3])
        bank = collection_attempt_bank(MASTER_SEED, 41)
        for row in merged["shards"]:
            manifest = json.loads(Path(row["manifest"]).read_text())
            seeds = [int(entry["seed"]) for entry in manifest["attempts"]]
            # The retry re-collected the same block from its start: nothing was sampled
            # before the crash, so the block's seeds are the block's seeds, not the next
            # ones in the stream.
            self.assertEqual(seeds, bank[row["offset"]:row["offset"] + len(seeds)])
            self.assertGreater(len(seeds), 0)
            self.assertEqual(Path(row["directory"]).name, "startup_attempt_3")
        self.assertTrue(merged["coverage"]["passed"])

    def test_a_shard_that_sampled_before_dying_is_never_retried(self):
        original = FixtureRuntime._run_contract

        def crash_after_sampling(self, command, output):
            result = original(self, command, output)
            manifest = Path(command[command.index("--collection_manifest") + 1])
            if "shard_01" in str(manifest):
                atomic_json(Path(output) / "process.json",
                            {"status": "failed", "returncode": -11, "pid": 4242})
                raise RuntimeError("synthetic abort after the block was sampled")
            return result
        FixtureRuntime._run_contract = crash_after_sampling
        self.addCleanup(setattr, FixtureRuntime, "_run_contract", original)
        with self.assertRaises(RuntimeError) as caught:
            self.collect(devices=3, attempts=40, target=10)
        self.assertIn("collection shards failed", str(caught.exception))
        self.assertFalse((self.root / "collection/bounded_collection_result.json").exists())

    def test_a_shard_that_cannot_reach_its_share_reports_partial_yield(self):
        # Target == budget with the budget divided: no block can reach its own share (about
        # a third of the attempts succeed), so every shard spends its block and reports the
        # shortfall.  It must NOT top itself up from a neighbour's block -- that would be the
        # selective re-run this pipeline forbids, and the coverage audit would refuse it.
        self.runtime.plan = self.plan(3)
        expected = self.runtime.collection_shard_selections(24)
        self.assertGreater(len(expected), 1)
        merged = self.collect(devices=3, attempts=24, target=24)
        self.assertEqual(len(merged["shards"]), len(expected))
        self.assertEqual(merged["capability_state"], "partial_yield")
        self.assertEqual(merged["termination"], "attempt_budget_exhausted")
        self.assertEqual(merged["shards_below_their_share"], list(range(len(expected))))
        self.assertEqual(merged["attempts_consumed"], 24)     # every block was spent
        self.assertEqual(merged["coverage"]["unspent_attempts"], 0)
        self.assertTrue(merged["coverage"]["passed"])

    def test_policy_correction_stays_a_single_process_by_declared_limits(self):
        self.runtime.plan = self.plan(3)
        self.assertEqual(self.runtime.collection_shard_selections(
            40, collection_mode="policy_correction"), [])
        self.assertTrue(self.runtime.collection_shard_selections(40, collection_mode="expert"))

    def test_a_job_bound_runtime_is_never_sharded_again(self):
        self.runtime.plan = self.plan(3)
        self.runtime.selection = SimDeviceSelection(
            mode="pinned_index", index=1, uuid="GPU-1", vulkan_gpu_id=1, torch_index=0,
            cuda_visible="1", renderer="hybrid", extra_env={})
        self.assertEqual(self.runtime.collection_shard_selections(40), [])
        self.assertEqual(self.runtime.shard_selections(40), [])


if __name__ == "__main__":
    unittest.main()
