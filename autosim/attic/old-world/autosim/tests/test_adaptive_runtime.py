import tempfile
import unittest
from pathlib import Path

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


class ExecutionCodeDigestTest(unittest.TestCase):
    """The digest is the identity of the code that produced a run's artifacts.

    It is keyed on where code lives as well as what it says, which is why generalizing the
    execution layer invalidates resume for runs started before it. This test exists so that
    consequence is discovered here rather than by a resume failing months later.
    """

    def test_the_digest_covers_the_whole_package_not_just_this_module(self):
        from autosim.research.runtime import Runtime
        package = Path(__file__).resolve().parents[1] / "autosim"
        files = sorted(package.rglob("*.py"))
        runtime = Runtime(package, package / "output")

        self.assertGreater(len(files), 20)
        # A digest over runtime.py alone would not change when a method moves out of it,
        # which is exactly the move this refactor performs.
        covered = {str(p.relative_to(package)) for p in files}
        self.assertIn("research/runtime.py", covered)
        self.assertIn("research/repository_autoresearch.py", covered)
        self.assertTrue(runtime._execution_code_digest())

    def test_the_digest_is_stable_for_an_unchanged_tree(self):
        from autosim.research.runtime import Runtime
        package = Path(__file__).resolve().parents[1] / "autosim"
        first = Runtime(package, package / "output")._execution_code_digest()
        second = Runtime(package, package / "output")._execution_code_digest()
        self.assertEqual(first, second)

    def test_a_renamed_module_changes_the_digest(self):
        """Same bytes, different path, different identity.

        This is why generalizing the execution layer breaks resume: the digest is keyed on
        where code lives. Asserted here rather than discovered by a failed resume.
        """
        from autosim.research.runtime import execution_code_digest
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "pkg"
            (package / "a").mkdir(parents=True)
            (package / "a" / "one.py").write_text("X = 1\n", encoding="utf-8")
            before = execution_code_digest(package)
            (package / "a" / "one.py").unlink()
            (package / "a" / "two.py").write_text("X = 1\n", encoding="utf-8")
            self.assertNotEqual(before, execution_code_digest(package))
            # And moving the same file between directories changes it too.
            (package / "a" / "two.py").unlink()
            (package / "b").mkdir()
            (package / "b" / "two.py").write_text("X = 1\n", encoding="utf-8")
            self.assertNotEqual(before, execution_code_digest(package))

    def test_the_digest_ignores_the_workspace(self):
        """It is the installed package, so two runtimes in different places share one."""
        from autosim.research.runtime import _this_package
        self.assertEqual(_this_package().name, "autosim")
