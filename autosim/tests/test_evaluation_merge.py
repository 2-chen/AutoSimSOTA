import json
import math
import tempfile
import unittest
from pathlib import Path

from autosim.robosyn_data import evaluation_seed_bank
from autosim.research.common import object_digest
from autosim.research.evaluation_merge import (ShardMergeRefused, build_shard_plan,
                                               load_shard_plan, merge_evaluation_shards,
                                               recompute_summary)

MASTER_SEED = 1017355756
TIMEOUT_STEPS = 361
# One recorded three-episode development evaluation, copied verbatim from
# evaluations/official_act_development in the seed-1021 smoke run: rows *and* the summary the
# official evaluator computed from them.  Using a recorded payload rather than a generated
# one is what makes the recomputation below an independent check of the formula.
REAL_ROWS = [
    {"episode_index": 0, "episode_seed": 1613266352, "success": False, "action_steps": 361,
     "inference_call_count": 8, "average_inference_time_seconds": 0.28154862066730857,
     "total_inference_time_seconds": 2.2523889653384686,
     "max_button_press_depth_m": 0.0009423400042578578,
     "failure_stage": "contact_insufficient_press"},
    {"episode_index": 1, "episode_seed": 1337289384, "success": False, "action_steps": 361,
     "inference_call_count": 8, "average_inference_time_seconds": 0.020571886212565005,
     "total_inference_time_seconds": 0.16457508970052004,
     "max_button_press_depth_m": 0.0009423146839253604,
     "failure_stage": "contact_insufficient_press"},
    {"episode_index": 2, "episode_seed": 1335456380, "success": False, "action_steps": 361,
     "inference_call_count": 8, "average_inference_time_seconds": 0.020357056753709912,
     "total_inference_time_seconds": 0.1628564540296793,
     "max_button_press_depth_m": 0.0009423146839253604,
     "failure_stage": "contact_insufficient_press"},
]
REAL_SUMMARY = {
    "episode_count": 3, "success_count": 0, "success_rate": 0.0, "average_action_steps": 361.0,
    "average_action_steps_ratio": 1.0, "inference_call_count": 24,
    "average_inference_calls_per_episode": 8.0,
    "average_inference_time_seconds": 0.1074925212111945,
    "average_inference_time_per_episode_seconds": 0.859940169689556,
}
CONFIG = {
    "policy": "act", "task": "click_bell", "setting": "random",
    "checkpoint_path": "/nonexistent/ACT_sim_click_bell", "dp_num_inference_steps": None,
    "episode_count": 3, "timeout_action_steps": TIMEOUT_STEPS, "seed": MASTER_SEED,
    "diagnostic_profile": None, "act_n_action_steps_override": None, "ranking_eligible": True,
}
CERTIFICATION = {
    "purpose": "development", "execution_mode": "real_simulation",
    "harness": "official_control_loop_observation_only_rpc_v1",
    "policy_observation_contract": {"state_dim": 14, "action_dim": 14, "cameras": ["cam_high"]},
    "rpc_timing_note": "same-machine bridge overhead included on fresh inference calls",
}
FROZEN = {"scripts/eval_policy.py": "aaa", "autosim/research/evaluation.py": "bbb"}


def official_shard_payload(rows, *, episodes, purpose="development"):
    """A shard payload shaped exactly like the official evaluator's output."""
    return {
        "schema_version": 2, "created_at": "2026-09-13T10:14:44+00:00",
        "config": dict(CONFIG, episode_count=episodes),
        "inference_timing_scope": "raw observation preprocessing and transfer through "
                                  "executable action; env.step excluded",
        "platform": {"operating_system": "Linux", "accelerators": ["NVIDIA GeForce RTX 5090"]},
        "diagnostic": None, "summary": official_summary(rows), "episodes": rows,
        **CERTIFICATION,
    }


def official_summary(rows):
    """The official formula, written out longhand so it cannot share code with the merge."""
    steps = [row["action_steps"] for row in rows]
    calls = [row["inference_call_count"] for row in rows]
    totals = [row["total_inference_time_seconds"] for row in rows]
    successful = sum(1 for row in rows if row["success"])
    return {
        "episode_count": len(rows),
        "success_count": successful,
        "success_rate": successful / len(rows),
        "average_action_steps": sum(steps) / len(steps),
        "average_action_steps_ratio": (sum(steps) / len(steps)) / TIMEOUT_STEPS,
        "inference_call_count": sum(calls),
        "average_inference_calls_per_episode": sum(calls) / len(rows),
        "average_inference_time_seconds": sum(totals) / sum(calls),
        "average_inference_time_per_episode_seconds": sum(totals) / len(totals),
    }


class FormulaTests(unittest.TestCase):
    def test_the_recomputation_reproduces_the_official_numbers(self):
        recomputed = recompute_summary(REAL_ROWS, timeout_action_steps=TIMEOUT_STEPS)
        self.assertEqual(recomputed["episode_count"], REAL_SUMMARY["episode_count"])
        self.assertEqual(recomputed["success_count"], REAL_SUMMARY["success_count"])
        self.assertEqual(recomputed["inference_call_count"], REAL_SUMMARY["inference_call_count"])
        for key, recorded in REAL_SUMMARY.items():
            self.assertTrue(math.isclose(recomputed[key], recorded, rel_tol=1e-12),
                            f"{key}: {recomputed[key]!r} != {recorded!r}")

    def test_the_longhand_official_formula_agrees_with_the_recorded_summary(self):
        self.assertEqual(official_summary(REAL_ROWS), REAL_SUMMARY)


class MergeTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.output = Path(self._temp.name) / "evaluation"
        self.output.mkdir(parents=True)
        self.bank = evaluation_seed_bank(MASTER_SEED, 3)

    def selections(self, count):
        return [{"index": index, "uuid": f"GPU-{index}", "selection": {"mode": "pinned_index"}}
                for index in range(count)]

    def write_plan(self, count, *, bank=None, episodes=3, output=None):
        plan = build_shard_plan(episodes=episodes, bank=bank or self.bank,
                                selections=self.selections(count))
        path = Path(output or self.output) / "shard_plan.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan), encoding="utf-8")
        return plan

    def write_shard(self, index, rows, *, plan, protocol_overrides=None, summary=None,
                    status="completed", output=None):
        directory = Path(output or self.output) / "shards" / f"shard_{index:02d}"
        (directory / "process").mkdir(parents=True, exist_ok=True)
        block = plan["blocks"][index]
        payload = official_shard_payload(rows, episodes=block["size"])
        if summary is not None:
            payload["summary"] = summary
        (directory / "evaluation_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        protocol = {"task": {"name": "click_bell"}, "purpose": "development",
                    "frozen_files": dict(FROZEN), "checkpoint_sha256": "ckpt",
                    "seed": MASTER_SEED, "episodes": block["size"], "policy": "act",
                    "shard": {"index": index, "count": plan["count"],
                              "seed_offset": block["offset"], "episodes": block["size"]}}
        protocol.update(protocol_overrides or {})
        (directory / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
        (directory / "process" / "process.json").write_text(json.dumps(
            {"status": status, "returncode": 0, "started_at": "2026-09-13T10:00:00+00:00",
             "finished_at": "2026-09-13T10:07:00+00:00", "elapsed_seconds": 420.5}),
            encoding="utf-8")
        (directory / "initializations.jsonl").write_text(
            "\n".join(json.dumps({"event": "reset", "seed": row["episode_seed"]}) for row in rows) + "\n",
            encoding="utf-8")
        return directory

    def merge(self, **overrides):
        kwargs = dict(output=self.output, purpose="development", master_seed=MASTER_SEED,
                      episodes=3, bank=self.bank, merge_command=["merge-shards"])
        kwargs.update(overrides)
        return merge_evaluation_shards(**kwargs)

    def prepare(self, count, *, rows=None, output=None, merge=True):
        output = Path(output or self.output)
        plan = self.write_plan(count, output=output)
        rows = rows or REAL_ROWS
        for index, block in enumerate(plan["blocks"]):
            self.write_shard(index, rows[block["offset"]:block["offset"] + block["size"]],
                             plan=plan, output=output)
        return (plan, self.merge(output=output)) if merge else plan

    def test_one_shard_and_three_shards_publish_the_same_number(self):
        serial_dir = Path(self._temp.name) / "serial"
        serial_dir.mkdir()
        serial_plan, serial = self.prepare(1, output=serial_dir)
        sharded_plan, sharded = self.prepare(3)
        self.assertEqual((serial_plan["count"], sharded_plan["count"]), (1, 3))
        self.assertEqual(serial["episodes"], sharded["episodes"])
        self.assertEqual(serial["summary"], sharded["summary"])
        self.assertEqual(sharded["config"]["episode_count"], 3)
        self.assertEqual(sharded["merged_from_shards"], 3)

    def test_the_merged_summary_is_the_official_one(self):
        self.prepare(3)
        merged = json.loads((self.output / "evaluation_metrics.json").read_text())
        for key, recorded in REAL_SUMMARY.items():
            self.assertTrue(math.isclose(merged["summary"][key], recorded, rel_tol=1e-12),
                            f"{key}: {merged['summary'][key]!r} != {recorded!r}")
        self.assertEqual([row["episode_index"] for row in merged["episodes"]], [0, 1, 2])
        self.assertEqual([row["episode_seed"] for row in merged["episodes"]], self.bank)

    def test_the_merged_protocol_looks_like_a_single_process_evaluation(self):
        self.prepare(3)
        protocol = json.loads((self.output / "protocol.json").read_text())
        self.assertNotIn("shard", protocol)
        self.assertEqual(protocol["episodes"], 3)
        self.assertEqual(protocol["frozen_files"], FROZEN)
        self.assertEqual(protocol["seed"], MASTER_SEED)
        self.assertEqual(protocol["shard_plan_sha256"],
                         object_digest_obj(self.output / "shard_plan.json"))
        merged = json.loads((self.output / "evaluation_metrics.json").read_text())
        self.assertEqual(merged["seed_bank_sha256"], object_digest(list(self.bank)))

    def test_the_composite_process_record_keeps_each_shard_receipt(self):
        self.prepare(3)
        record = json.loads((self.output / "process/process.json").read_text())
        self.assertEqual((record["status"], record["returncode"]), ("completed", 0))
        self.assertEqual(record["merged_from_shards"], 3)
        self.assertEqual([row["index"] for row in record["shards"]], [0, 1, 2])
        self.assertTrue(all(len(row["metrics_sha256"]) == 64 for row in record["shards"]))
        self.assertEqual(record["elapsed_seconds"], 420.5 * 3)
        self.assertEqual(record["started_at"], "2026-09-13T10:00:00+00:00")
        receipt = json.loads((self.output / "merge_receipt.json").read_text())
        self.assertEqual(receipt["episode_count"], 3)
        self.assertIn("sum(episode totals)/sum(calls)", receipt["summary_formula_note"])

    def test_republishing_the_same_shards_is_byte_identical_apart_from_the_timestamp(self):
        self.prepare(3)
        first = json.loads((self.output / "evaluation_metrics.json").read_text())
        self.merge()
        second = json.loads((self.output / "evaluation_metrics.json").read_text())
        for document in (first, second):
            document.pop("created_at")
        self.assertEqual(first, second)

    def test_a_missing_shard_is_refused_rather_than_partially_merged(self):
        plan = self.write_plan(3)
        self.write_shard(0, REAL_ROWS[:1], plan=plan)
        with self.assertRaises(ShardMergeRefused) as caught:
            self.merge()
        self.assertIn("shard 1", str(caught.exception))
        self.assertFalse((self.output / "evaluation_metrics.json").exists())

    def test_a_payload_in_a_retry_directory_is_merged_and_marked_as_the_attempt(self):
        """A shard that crashed before its first reset retries into its own directory."""
        plan = self.write_plan(1)
        self.write_shard(0, REAL_ROWS, plan=plan)
        shard = self.output / "shards" / "shard_00"
        retry = shard / "startup_attempt_2"
        retry.mkdir(parents=True)
        for name in ("evaluation_metrics.json", "protocol.json", "initializations.jsonl"):
            (retry / name).write_text((shard / name).read_text(), encoding="utf-8")
        for name in ("evaluation_metrics.json", "protocol.json", "initializations.jsonl"):
            (shard / name).unlink()
        merged = self.merge()
        self.assertEqual(merged["episodes"][0]["episode_seed"], self.bank[0])
        receipt = json.loads((self.output / "merge_receipt.json").read_text())
        self.assertEqual(receipt["shards"][0]["attempt"], 2)
        self.assertTrue(receipt["shards"][0]["directory"].endswith("startup_attempt_2"))

    def test_two_certified_payloads_for_one_shard_are_refused(self):
        """Two results for one slice means the number has no single origin; never pick one."""
        plan = self.write_plan(1)
        self.write_shard(0, REAL_ROWS, plan=plan)
        shard = self.output / "shards" / "shard_00"
        retry = shard / "startup_attempt_2"
        retry.mkdir(parents=True)
        for name in ("evaluation_metrics.json", "protocol.json"):
            (retry / name).write_text((shard / name).read_text(), encoding="utf-8")
        with self.assertRaises(ShardMergeRefused) as caught:
            self.merge()
        self.assertIn("2 certified payloads", str(caught.exception))
        self.assertIn("which result the number came from", str(caught.exception))
        self.assertFalse((self.output / "evaluation_metrics.json").exists())

    def test_a_shard_that_ran_another_slice_is_refused(self):
        plan = self.write_plan(3)
        self.write_shard(0, REAL_ROWS[:1], plan=plan)
        self.write_shard(1, REAL_ROWS[:1], plan=plan)        # duplicate of slice 0
        self.write_shard(2, REAL_ROWS[2:], plan=plan)
        with self.assertRaises(ShardMergeRefused) as caught:
            self.merge()
        message = str(caught.exception)
        self.assertIn("seeds do not equal its own bank slice", message)
        self.assertIn("appears in shards", message)

    def test_a_shard_summary_that_disagrees_with_its_rows_is_refused(self):
        plan = self.write_plan(1)
        self.write_shard(0, REAL_ROWS, plan=plan, summary=dict(REAL_SUMMARY, success_count=3))
        with self.assertRaises(ShardMergeRefused) as caught:
            self.merge()
        self.assertIn("summary disagrees with its rows", str(caught.exception))

    def test_a_shard_built_from_other_code_or_another_checkpoint_is_refused(self):
        for overrides in ({"frozen_files": {"scripts/eval_policy.py": "different"}},
                          {"checkpoint_sha256": "other"}, {"seed": 7}):
            with self.subTest(overrides=overrides):
                self.setUp()
                plan = self.write_plan(2)
                self.write_shard(0, REAL_ROWS[:2], plan=plan)
                self.write_shard(1, REAL_ROWS[2:], plan=plan, protocol_overrides=overrides)
                with self.assertRaises(ShardMergeRefused) as caught:
                    self.merge()
                self.assertIn("same frozen protocol", str(caught.exception))

    def test_a_plan_that_does_not_cover_the_request_is_refused(self):
        self.write_plan(2, episodes=2)
        with self.assertRaises(ShardMergeRefused) as caught:
            self.merge(episodes=3)
        self.assertIn("shard plan covers 2 episodes", str(caught.exception))

    def test_a_bank_that_is_not_the_frozen_one_is_refused(self):
        self.prepare(2)
        with self.assertRaises(ShardMergeRefused) as caught:
            self.merge(bank=[1, 2, 3])
        self.assertIn("does not match the bank the shard plan froze", str(caught.exception))

    def test_without_a_frozen_plan_nothing_is_merged(self):
        (self.output / "shard_plan.json").unlink(missing_ok=True)
        with self.assertRaises(ShardMergeRefused) as caught:
            self.merge()
        self.assertIn("must freeze its division", str(caught.exception))

    def test_the_plan_records_each_shards_device_and_frozen_bank(self):
        plan = load_shard_plan(self.output) if (self.output / "shard_plan.json").exists() else None
        self.assertIsNone(plan)
        written = self.write_plan(3)
        self.assertEqual([block["offset"] for block in written["blocks"]], [0, 1, 2])
        self.assertEqual([block["size"] for block in written["blocks"]], [1, 1, 1])
        self.assertEqual([block["device_uuid"] for block in written["blocks"]],
                         ["GPU-0", "GPU-1", "GPU-2"])
        self.assertEqual(written["bank_sha256"], object_digest(list(self.bank)))


def object_digest_obj(path: Path) -> str:
    from autosim.research.common import digest
    return digest(path)


if __name__ == "__main__":
    unittest.main()
