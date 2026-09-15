from autosim.research.common import read_json
from autosim.robosyn_data import evaluation_seed_bank
from tests.test_runtime_sharding import ShardingTestCase, FixtureRuntime, SPEC, MASTER_SEED


class AdaptiveRuntimeTests(ShardingTestCase):
    def plan(self, devices, **overrides):
        plan = super().plan(devices, **overrides)
        plan.update(unified_scheduler=True, legacy_equivalence=False, gpu_hours_limit=100,
                    host_capacity={"cpu_cores":8,"ram_mib":32768,"shm_mib":8192,
                                   "scratch_mib":100000,"io_slots":4,"host":"fixture"})
        for d in plan["usable"]:
            d.update(memory_total_mib=32768,memory_used_mib=0,
                     capabilities={k:"verified" for k in ("physics","render","policy_inference")})
        return plan

    def test_five_logical_blocks_use_two_devices_then_resume_on_one_without_replay(self):
        self.runtime.plan = self.plan(2)
        FixtureRuntime.shard_total = 5
        FixtureRuntime.delay = .02
        output = self.root / "adaptive_eval"
        first = self.runtime.evaluate(SPEC,self.checkpoint,output,episodes=40,master_seed=MASTER_SEED)
        self.assertEqual(len(FixtureRuntime.records),5)
        self.assertEqual(FixtureRuntime.peak_in_flight,2)
        frozen = read_json(output / "shard_plan.json")
        self.assertEqual(frozen["count"],5)
        self.assertTrue(all("device_uuid" not in block for block in frozen["blocks"]))
        self.runtime.plan = self.plan(1)
        self.runtime.shard_min_episodes = 16
        second = self.runtime.evaluate(SPEC,self.checkpoint,output,episodes=40,master_seed=MASTER_SEED)
        self.assertEqual(len(FixtureRuntime.records),5)
        self.assertEqual(first["episodes"],second["episodes"])
        self.assertEqual([r["episode_seed"] for r in second["episodes"]],evaluation_seed_bank(MASTER_SEED,40))
