import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from autosim.experiment_validation.integration_remediation import file_manifest
from autosim.experiment_validation.remediation_worker_v2 import aggregate_metrics
from autosim.experiment_validation.remediation_worker_v3 import run_episode
from autosim.experiment_validation.robotwin_collect_v2 import classify_setup_exception
from autosim.research.common import atomic_json
from autosim.experiment_validation.safe_evaluation import NumericalSafetyEnv, _done_like


class FakeSpace:
    low = np.asarray([-1.0, -2.0], dtype=np.float32)
    high = np.asarray([1.0, 2.0], dtype=np.float32)


class FakeEnv:
    single_action_space = FakeSpace()

    @property
    def unwrapped(self):
        return self

    def step(self, action):
        self.applied = np.asarray(action)
        observation = {"robot": {"qpos": np.asarray([[np.nan, 0]], dtype=np.float32)}}
        return observation, 0.0, False, np.asarray([False]), {"elapsed_steps": np.asarray([1])}


class IntegrationRemediationTest(unittest.TestCase):
    def test_action_space_clip_and_nonfinite_state_is_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = FakeEnv()
            spec = SimpleNamespace(action_dim=2, state_dim=2)
            wrapped = NumericalSafetyEnv(env, spec, Path(temporary), None, "smoke")
            result = wrapped.step(np.asarray([[4.0, -3.0]], dtype=np.float32))
            np.testing.assert_array_equal(env.applied, [[1.0, -2.0]])
            self.assertTrue(result[3].all())
            self.assertEqual(wrapped.clipped_action_count, 1)
            self.assertEqual(wrapped.clipped_element_count, 2)
            self.assertEqual(wrapped.invalid_state_episodes[0]["step"], 1)

    def test_done_like_preserves_array_shape(self):
        value = _done_like(np.zeros((2, 1), dtype=bool))
        self.assertEqual(value.shape, (2, 1))
        self.assertTrue(value.all())

    def test_overlay_manifest_ignores_runtime_bytecode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pkg/__pycache__").mkdir(parents=True)
            (root / "pkg/source.py").write_text("source\n")
            (root / "pkg/__pycache__/source.pyc").write_bytes(b"mutable")
            manifest = file_manifest(root)
            self.assertEqual(list(manifest), [str((root / "pkg/source.py").absolute())])

    def test_isolated_episode_metric_aggregation_recomputes_counts(self):
        def metric(seed, success, calls, mean_time):
            return {
                "created_at": "now", "config": {"timeout_action_steps": 361},
                "inference_timing_scope": "scope", "platform": {"gpu": "test"},
                "summary": {"success_count": int(success), "average_action_steps": 10 if success else 361,
                            "inference_call_count": calls, "average_inference_time_seconds": mean_time,
                            "average_inference_time_per_episode_seconds": calls * mean_time},
                "episodes": [{"episode_index": 0, "episode_seed": seed, "success": success}],
            }
        result = aggregate_metrics([metric(11, True, 2, .1), metric(12, False, 1, .4)], 99)
        self.assertEqual(result["summary"]["episode_count"], 2)
        self.assertEqual(result["summary"]["success_count"], 1)
        self.assertEqual(result["summary"]["inference_call_count"], 3)
        self.assertAlmostEqual(result["summary"]["average_inference_time_seconds"], .2)
        self.assertEqual([row["episode_index"] for row in result["episodes"]], [0, 1])

    def test_pre_reset_native_crash_gets_new_bounded_attempt(self):
        class RuntimeStub:
            calls = 0

            def run(self, command, output, timeout, evaluation):
                self.calls += 1
                attempt = output.parent
                atomic_json(output / "process.json", {"status": "failed" if self.calls == 1 else "completed",
                                                       "returncode": -11 if self.calls == 1 else 0,
                                                       "command": command, "cwd": "test"})
                atomic_json(attempt / "startup.json", {"phase": "environment_constructing"})
                if self.calls == 1:
                    raise RuntimeError("startup crash")

        with tempfile.TemporaryDirectory() as temporary:
            runtime = RuntimeStub()
            path, attempts = run_episode(runtime, ["python", "eval"], Path(temporary), 123)
            self.assertEqual(attempts, 2)
            self.assertEqual(path.name, "startup_attempt_2")

    def test_post_reset_failure_is_never_retried(self):
        class RuntimeStub:
            calls = 0

            def run(self, command, output, timeout, evaluation):
                self.calls += 1
                atomic_json(output / "process.json", {"status": "failed", "returncode": -11,
                                                       "command": command, "cwd": "test"})
                atomic_json(output.parent / "startup.json", {"phase": "episode_started"})
                (output.parent / "initializations.jsonl").write_text("reset\n")
                raise RuntimeError("post-reset crash")

        with tempfile.TemporaryDirectory() as temporary:
            runtime = RuntimeStub()
            with self.assertRaises(RuntimeError):
                run_episode(runtime, ["python", "eval"], Path(temporary), 123)
            self.assertEqual(runtime.calls, 1)

    def test_only_declared_unstable_scene_error_is_skippable(self):
        class Unstable(Exception):
            pass

        self.assertEqual(classify_setup_exception(Unstable("scene"), Unstable),
                         "official_scene_unstable")
        self.assertIsNone(classify_setup_exception(RuntimeError("adapter"), Unstable))

    def test_regular_policy_bridge_beats_later_conflicting_package(self):
        bridge = Path(__file__).parents[1] / "autosim/experiment_validation/policy_namespace"
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "robosyn_policy"
            conflict = Path(temporary) / "native/policy"
            (target / "act").mkdir(parents=True)
            conflict.mkdir(parents=True)
            (target / "act/__init__.py").write_text("origin = 'robosyn'\n")
            (conflict / "__init__.py").write_text("origin = 'native'\n")
            old_path = list(sys.path)
            old_root = os.environ.get("AUTOSIM_POLICY_PACKAGE_ROOT")
            stale = {name: module for name, module in sys.modules.items()
                     if name == "policy" or name.startswith("policy.")}
            try:
                for name in stale:
                    del sys.modules[name]
                os.environ["AUTOSIM_POLICY_PACKAGE_ROOT"] = str(target)
                sys.path[:0] = [str(bridge), str(Path(temporary) / "native")]
                module = importlib.import_module("policy.act")
                self.assertEqual(module.origin, "robosyn")
                self.assertEqual(Path(sys.modules["policy"].__path__[0]), target)
            finally:
                for name in list(sys.modules):
                    if name == "policy" or name.startswith("policy."):
                        del sys.modules[name]
                sys.modules.update(stale)
                sys.path[:] = old_path
                if old_root is None:
                    os.environ.pop("AUTOSIM_POLICY_PACKAGE_ROOT", None)
                else:
                    os.environ["AUTOSIM_POLICY_PACKAGE_ROOT"] = old_root


if __name__ == "__main__":
    unittest.main()
