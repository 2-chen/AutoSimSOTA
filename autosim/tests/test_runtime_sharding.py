"""The Runtime-level shard runner: what it divides, what it refuses, when it stays single.

The evaluator is faked, but only at its *process boundary*: the fake reads ``--seed`` /
``--seed-offset`` / ``--episodes`` exactly like the real one and writes the same files into
the same places, deriving episode seeds from ``evaluation_seed_bank`` with the documented
slice semantics.  So the merge gates are exercised against real seed arithmetic rather than
against a fixture that could have been shaped to fit them.
"""

import json
import math
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from autosim.robosyn_data import evaluation_seed_bank
from autosim.research.common import atomic_json
from autosim.research.devices import SimDeviceSelection, shard_count, shard_plan
from autosim.research.registry import TaskSpec
from autosim.research.runtime import Runtime

MASTER_SEED = 1017355756
TIMEOUT_STEPS = 361
COST_MODEL = {"construction_seconds": 360.0, "marginal_seconds": 40.0}
SPEC = TaskSpec(name="click_bell", env_id="click_bell", setting="random",
                max_episode_steps=TIMEOUT_STEPS, state_dim=14, action_dim=14,
                cameras=("cam_high",), camera_shapes={"cam_high": [3, 240, 320]},
                control_parts=("left_arm",), recorded_fps=30.0, instruction="press the bell",
                gym_config="gym.xml", action_config="action.json", config_hashes={},
                event_families={}, roles={})


def official_summary(rows, *, timeout_action_steps=TIMEOUT_STEPS):
    """The official evaluator's summary, longhand, so it cannot share code with the merge."""
    steps = [row["action_steps"] for row in rows]
    calls = [row["inference_call_count"] for row in rows]
    totals = [row["total_inference_time_seconds"] for row in rows]
    successful = sum(1 for row in rows if row["success"])
    return {
        "episode_count": len(rows), "success_count": successful,
        "success_rate": successful / len(rows),
        "average_action_steps": sum(steps) / len(steps),
        "average_action_steps_ratio": (sum(steps) / len(steps)) / timeout_action_steps,
        "inference_call_count": sum(calls),
        "average_inference_calls_per_episode": sum(calls) / len(rows),
        "average_inference_time_seconds": sum(totals) / sum(calls),
        "average_inference_time_per_episode_seconds": sum(totals) / len(totals),
    }


def rows_for(seeds):
    """A deterministic per-seed result, the way a real episode is a pure function of it."""
    rows = []
    for index, seed in enumerate(seeds):
        calls = 8
        per_call = 0.02 + (seed % 1000) / 100000.0
        rows.append({"episode_index": index, "episode_seed": seed, "success": seed % 3 == 0,
                     "action_steps": 100 + seed % 50, "inference_call_count": calls,
                     "average_inference_time_seconds": per_call,
                     "total_inference_time_seconds": per_call * calls,
                     "max_button_press_depth_m": 0.001, "failure_stage": None})
    return rows


class FixtureRuntime(Runtime):
    """A Runtime whose subprocess is replaced by the evaluator's file contract.

    Everything shared with the per-job copies ``for_job`` builds lives on the class:
    ``dataclasses.replace`` carries fields over, not instance attributes, and a per-job
    copy silently losing the fixture's knobs would be a test that lies.
    """

    records: list = []
    failing_shard: int | None = None
    crash_attempts: int = 1          # how many of that shard's attempts crash before one works
    crashes: dict = {}
    last_purpose: str = "development"
    shard_total: int = 0
    delay: float = 0.0
    in_flight: int = 0
    peak_in_flight: int = 0
    gate = threading.Lock()

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
        output_path = Path(command[command.index("--output") + 1])
        master = int(command[command.index("--seed") + 1])
        episodes = int(command[command.index("--episodes") + 1])
        offset = int(command[command.index("--seed-offset") + 1]) if "--seed-offset" in command else 0
        shard = command[command.index("--shard-index") + 1] if "--shard-index" in command else None
        selection = self.selection
        entry = {"started": time.monotonic(),
                 "uuid": selection.uuid if selection else None,
                 "cuda_visible": selection.cuda_visible if selection else None,
                 "torch_index": selection.torch_index if selection else None,
                 "flags": [command[index + 1] for index, token in enumerate(command)
                           if token in ("--device-gpu-id", "--renderer", "--device-torch-index")],
                 "offset": offset, "episodes": episodes, "shard": shard,
                 "output": str(output_path)}
        type(self).records.append(entry)
        protocol = {"task": {"name": SPEC.name}, "purpose": type(self).last_purpose,
                    "frozen_files": {}, "checkpoint_sha256": "ckpt", "seed": master,
                    "policy": "act", "episodes": episodes}
        if shard is not None:
            protocol["shard"] = {"index": int(shard), "count": type(self).shard_total,
                                 "seed_offset": offset, "episodes": episodes}
        atomic_json(output_path / "protocol.json", protocol)
        crash = (shard is not None and shard == str(type(self).failing_shard)
                 and type(self).crashes.get(shard, 0) < type(self).crash_attempts)
        if crash:
            type(self).crashes[shard] = type(self).crashes.get(shard, 0) + 1
            entry["failed"] = True
            atomic_json(output_path / "startup.json", {"phase": "environment_constructing"})
            atomic_json(output / "process.json", {"status": "failed", "returncode": -11, "pid": 4242})
            raise RuntimeError("synthetic native crash before the first reset")
        seeds = evaluation_seed_bank(master, offset + episodes)[offset:]
        rows = rows_for(seeds)
        atomic_json(output_path / "evaluation_metrics.json", {
            "schema_version": 2, "execution_mode": "real_simulation",
            "purpose": type(self).last_purpose,
            "harness": "official_control_loop_observation_only_rpc_v1",
            "policy_observation_contract": {"state_dim": 14, "action_dim": 14},
            "rpc_timing_note": "same-machine bridge overhead on fresh inference calls",
            "config": {"policy": "act", "task": SPEC.name, "setting": "random",
                       "episode_count": episodes, "timeout_action_steps": TIMEOUT_STEPS,
                       "seed": master},
            "summary": official_summary(rows), "episodes": rows})
        with (output_path / "initializations.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps({"event": "reset", "seed": row["episode_seed"]}) + "\n")
        time.sleep(type(self).delay)
        entry["finished"] = time.monotonic()
        atomic_json(output / "process.json", {"status": "completed", "returncode": 0,
                                              "started_at": "2026-09-13T10:00:00+00:00",
                                              "finished_at": "2026-09-13T10:07:00+00:00",
                                              "elapsed_seconds": 420.5, "device_uuid": entry["uuid"]})
        return {"status": "completed"}


class ShardingTestCase(unittest.TestCase):
    def setUp(self):
        self._temp = TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        FixtureRuntime.records = []
        FixtureRuntime.failing_shard = None
        FixtureRuntime.crash_attempts = 1
        FixtureRuntime.crashes = {}
        FixtureRuntime.last_purpose = "development"
        FixtureRuntime.delay = 0.0
        FixtureRuntime.in_flight = FixtureRuntime.peak_in_flight = 0
        self.runtime = FixtureRuntime(self.root, self.root / "run")
        self.runtime.cost_model = COST_MODEL
        self.runtime.shard_min_episodes = 8
        self.checkpoint = self.root / "model"
        atomic_json(self.checkpoint / "config.json", {"synthetic_fixture": True})
        atomic_json(self.checkpoint / "model.safetensors", {"synthetic_fixture": True})

    def plan(self, devices, **overrides):
        plan = {"mode": "pinned_index", "max_parallel_jobs": devices, "usable": [
            {"index": index, "uuid": f"GPU-{index}", "model": "NVIDIA GeForce RTX 5090",
             "selection": {"mode": "pinned_index", "index": index, "uuid": f"GPU-{index}",
                           "vulkan_gpu_id": index, "torch_index": 0, "cuda_visible": str(index),
                           "renderer": "hybrid", "extra_env": {}}}
            for index in range(devices)]}
        plan.update(overrides)
        return plan

    def evaluate(self, *, episodes=40, purpose="development", devices=3, output=None):
        self.runtime.plan = self.plan(devices)
        FixtureRuntime.last_purpose = purpose
        FixtureRuntime.shard_total = devices
        return self.runtime.evaluate(SPEC, self.checkpoint, self._output(output),
                                     episodes=episodes, master_seed=MASTER_SEED, purpose=purpose)

    def _output(self, output):
        return Path(output) if output else self.root / "eval"

    def assertSummariesClose(self, merged, expected):
        for key, value in expected.items():
            self.assertTrue(math.isclose(merged[key], value, rel_tol=1e-12),
                            f"{key}: {merged[key]!r} != {value!r}")


class ShardedEvaluationTests(ShardingTestCase):
    def test_the_bank_is_divided_over_the_devices_and_merged_back_whole(self):
        merged = self.evaluate(episodes=40, devices=3)
        expected = shard_count(40, 3, max_parallel_jobs=3, cost_model=COST_MODEL,
                               min_episodes_per_shard=8)
        self.assertGreater(expected, 1)
        self.assertEqual(merged["merged_from_shards"], expected)
        self.assertEqual(merged["shard_count"], expected)
        self.assertEqual(len(set(merged["shard_devices"])), expected)
        self.assertEqual([row["episode_seed"] for row in merged["episodes"]],
                         evaluation_seed_bank(MASTER_SEED, 40))
        self.assertEqual([row["episode_index"] for row in merged["episodes"]], list(range(40)))
        self.assertSummariesClose(merged["summary"], official_summary(merged["episodes"]))
        self.assertEqual(merged["startup_attempts_per_shard"], [1] * expected)
        protocol = json.loads((self.root / "eval" / "protocol.json").read_text())
        self.assertNotIn("shard", protocol)             # looks like one process, as before
        self.assertEqual(protocol["episodes"], 40)
        self.assertTrue((self.root / "eval" / "merge_receipt.json").is_file())

    def test_each_shard_runs_on_its_own_device_over_its_own_contiguous_block(self):
        self.evaluate(episodes=40, devices=3)
        plan = json.loads((self.root / "eval" / "shard_plan.json").read_text())
        runs = sorted((entry for entry in FixtureRuntime.records if entry["shard"] is not None),
                      key=lambda entry: int(entry["shard"]))   # they finish out of order
        self.assertEqual(len(runs), plan["count"])
        self.assertEqual([entry["uuid"] for entry in runs],
                         [f"GPU-{index}" for index in range(len(runs))])
        self.assertEqual([entry["cuda_visible"] for entry in runs],
                         [str(index) for index in range(len(runs))])
        self.assertEqual([entry["offset"] for entry in runs],
                         [block["offset"] for block in plan["blocks"]])
        self.assertEqual([entry["episodes"] for entry in runs],
                         [block["size"] for block in plan["blocks"]])
        # The engine index is the physical one; the torch index is the in-process one.
        self.assertEqual([entry["flags"] for entry in runs],
                         [[str(index), "hybrid", "0"] for index in range(len(runs))])

    def test_the_shards_really_run_at_the_same_time(self):
        FixtureRuntime.delay = 0.3
        merged = self.evaluate(episodes=40, devices=3)
        self.assertGreater(merged["merged_from_shards"], 1)
        self.assertGreaterEqual(FixtureRuntime.peak_in_flight, 2,
                                "the shard processes never overlapped")

    def test_a_crash_before_the_first_reset_is_retried_in_its_own_directory(self):
        FixtureRuntime.failing_shard = 1
        merged = self.evaluate(episodes=40, devices=3)
        attempts = merged["startup_attempts_per_shard"]
        self.assertEqual(attempts[1], 2)
        self.assertTrue(all(value == 1 for index, value in enumerate(attempts) if index != 1))
        self.assertEqual([row["episode_seed"] for row in merged["episodes"]],
                         evaluation_seed_bank(MASTER_SEED, 40))
        receipt = json.loads((self.root / "eval" / "merge_receipt.json").read_text())
        self.assertEqual(receipt["shards"][1]["attempt"], 2)
        self.assertTrue(receipt["shards"][1]["directory"].endswith("startup_attempt_2"))
        shard = self.root / "eval" / "shards" / "shard_01"
        self.assertTrue((shard / "startup_attempt_2" / "evaluation_metrics.json").is_file())
        # The crashed attempt stays where it was: it is the evidence for the retry.
        self.assertTrue((shard / "startup.json").is_file())
        self.assertFalse((shard / "evaluation_metrics.json").exists())

    def test_one_failing_shard_publishes_no_number(self):
        FixtureRuntime.failing_shard = 2
        FixtureRuntime.crash_attempts = 5        # crashes every attempt, not just the first
        with self.assertRaises(RuntimeError) as caught:
            self.evaluate(episodes=40, devices=3)
        self.assertIn("shards failed", str(caught.exception))
        self.assertIn("synthetic native crash before the first reset", str(caught.exception))
        # The retry is bounded and the shard's own failure is the one reported.
        self.assertEqual(FixtureRuntime.crashes["2"], 3)
        self.assertFalse((self.root / "eval" / "evaluation_metrics.json").exists())
        for shard in ("shard_00", "shard_01"):           # the healthy shards keep their work
            self.assertTrue((self.root / "eval" / "shards" / shard
                             / "evaluation_metrics.json").is_file())

    def test_the_single_process_path_is_taken_whenever_sharding_would_not_pay(self):
        cases = {"single device": dict(devices=1, episodes=40),
                 "smoke purpose": dict(purpose="smoke", episodes=40),
                 "short run": dict(episodes=4)}
        for label, overrides in cases.items():
            with self.subTest(case=label):
                output = self.root / label.replace(" ", "_")
                merged = self.evaluate(output=output, **overrides)
                self.assertNotIn("merged_from_shards", merged)
                self.assertFalse((output / "shard_plan.json").exists())
                self.assertFalse((output / "shards").exists())
                request = json.loads((output / "evaluation_request.json").read_text())
                self.assertNotIn("shard", request)
                self.assertEqual(request["episodes"], merged["config"]["episode_count"])
                run = [entry for entry in FixtureRuntime.records
                       if entry["output"].startswith(str(output))]
                self.assertEqual(len(run), 1)
                self.assertIsNone(run[0]["shard"])
                self.assertIsNone(run[0]["cuda_visible"])    # the legacy path, unbound
                self.assertEqual(run[0]["flags"], [])        # and with no device flags
                self.assertEqual(merged["startup_attempt_count"], 1)

    def test_a_plan_that_froze_no_selection_is_addressed_from_the_device_row(self):
        plan = self.plan(3)
        for device in plan["usable"]:
            device.pop("selection")
        self.runtime.plan = plan
        merged = self.runtime.evaluate(SPEC, self.checkpoint, self.root / "no_sel", episodes=40,
                                       master_seed=MASTER_SEED)
        self.assertGreater(merged["merged_from_shards"], 1)
        self.assertEqual(sorted(entry["uuid"] for entry in FixtureRuntime.records),
                         ["GPU-0", "GPU-1", "GPU-2"])

    def test_a_resumed_evaluation_cannot_re_divide_the_bank(self):
        self.evaluate(episodes=40, devices=3)
        plan_path = self.root / "eval" / "shard_plan.json"
        tampered = {"schema_version": 1, "episodes": 40, "count": 2,
                    "blocks": [{"index": 0, "offset": 0, "size": 20},
                               {"index": 1, "offset": 20, "size": 20}]}
        plan_path.unlink()
        atomic_json(plan_path, tampered)
        with self.assertRaises(ValueError) as caught:
            self.runtime.evaluate(SPEC, self.checkpoint, self.root / "eval", episodes=40,
                                  master_seed=MASTER_SEED)
        self.assertIn("immutable artifact changed", str(caught.exception))
        self.assertEqual(json.loads(plan_path.read_text()), tampered)   # left as found

    def test_the_shard_count_is_clamped_by_the_plan_and_the_bound_job(self):
        self.runtime.plan = self.plan(2, max_parallel_jobs=1)
        self.assertEqual(self.runtime.shard_selections(400), [])       # one job at a time
        self.runtime.plan = self.plan(2, max_parallel_jobs=2)
        self.assertEqual([selection.uuid for selection in self.runtime.shard_selections(400)],
                         ["GPU-0", "GPU-1"])
        self.assertEqual(self.runtime.shard_selections(400, purpose="smoke"), [])
        self.runtime.selection = SimDeviceSelection(
            **self.runtime.plan["usable"][0]["selection"])
        self.assertEqual(self.runtime.shard_selections(400), [])       # already bound to a card


class ShardPlanIdentityTests(unittest.TestCase):
    def test_the_blocks_tile_the_bank_without_gap_or_overlap(self):
        bank = evaluation_seed_bank(MASTER_SEED, 40)
        blocks = shard_plan(40, 3)
        self.assertEqual(blocks, [(0, 14), (14, 13), (27, 13)])
        walked = [seed for offset, size in blocks for seed in bank[offset:offset + size]]
        self.assertEqual(walked, bank)
        self.assertEqual(len(set(walked)), 40)


if __name__ == "__main__":
    unittest.main()
