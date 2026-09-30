"""CPU gate tests only; these fixtures are not simulation results."""
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from autosim.research.production import next_action
from autosim.research.collection_worker import (
    TrainingSceneTraceEnv,
    capture_training_scene_evidence,
    with_stepwise_success_history,
)


class ProductionGates(unittest.TestCase):
    def test_collection_scene_evidence_records_realized_state_without_mutating_obs(self):
        class Entity:
            def __init__(self, articulated=False):
                self.articulated = articulated

            def get_local_pose(self, to_matrix=False):
                self.asserted_matrix = to_matrix
                return np.eye(4)

            def get_qpos(self):
                return np.array([[0.0048]])

        class Sim:
            def get_rigid_object(self, uid):
                self.object_uid = uid
                return Entity()

            def get_articulation(self, uid):
                self.articulation_uid = uid
                return Entity(articulated=True)

        class Environment:
            def __init__(self):
                self.unwrapped = self
                self.sim = Sim()

            def reset(self, *args, **kwargs):
                return {"robot": {"qpos": np.arange(14)}}, {"kept": True}

        class Spec:
            name = "fixture_task"
            roles = {"objects": ["cube"], "articulations": ["button"]}
            cameras = ()

        env = Environment()
        obs = {"robot": {"qpos": np.arange(14)}}
        evidence = capture_training_scene_evidence(env, obs, Spec())
        self.assertEqual(evidence["entities"]["button"]["qpos"], [[0.0048]])
        self.assertEqual(evidence["robot_qpos"], list(range(14)))
        self.assertIn("material_texture_parameters",
                      evidence["unavailable_realized_parameters"])
        self.assertNotIn("__autosim", obs)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene_resets.jsonl"
            proxy = TrainingSceneTraceEnv(env, Spec(), path, "targeted_camera")
            returned_obs, returned_info = proxy.reset(seed=123)
            self.assertEqual(returned_info, {"kept": True})
            self.assertEqual(returned_obs["robot"]["qpos"].tolist(), list(range(14)))
            row = json.loads(path.read_text().strip())
            self.assertEqual((row["seed"], row["requested_profile"]),
                             (123, "targeted_camera"))
            self.assertTrue(row["privileged_training_diagnostics_only"])

    def test_terminal_hold_uses_real_steps_and_respects_task_limit(self):
        class Environment:
            def __init__(self, success_at, limit):
                self.steps, self.queries, self.success_at, self.limit = 0, 0, success_at, limit

            def step(self, action):
                self.steps += 1
                return None, 0, False, False, {}

            def get_wrapper_attr(self, name):
                if name == "_autosim_training_step_limit":
                    return self.limit
                def judge():
                    self.queries += 1
                    return self.steps >= self.success_at
                return judge

        def original(env):
            env.step(1)
            env.step(1)
            return True

        wrapped = with_stepwise_success_history(original, settle_steps=75)
        env = Environment(5, 10)
        self.assertTrue(wrapped(env))
        self.assertEqual((env.steps, env.queries), (5, 5))
        env = Environment(99, 6)
        self.assertTrue(wrapped(env))
        self.assertEqual((env.steps, env.queries), (5, 5))

    def test_expert_history_matches_one_query_per_actual_action(self):
        class Environment:
            def __init__(self):
                self.steps, self.queries = [], 0

            def step(self, action):
                self.steps.append(action)
                return ("observation", 0, False, False, {})

            def get_wrapper_attr(self, name):
                self.asserted_name = name
                def judge():
                    self.queries += 1
                    return False
                return judge

        def original(env, actions):
            for action in actions:
                self.assertEqual(env.step(action)[0], "observation")
            return bool(actions)

        env = Environment()
        wrapped = with_stepwise_success_history(original)
        self.assertFalse(wrapped(env, []))
        self.assertEqual(env.queries, 0)
        self.assertTrue(wrapped(env, [1, 2, 3]))
        self.assertEqual(env.steps, [1, 2, 3])
        self.assertEqual(env.queries, 3)
        self.assertEqual(env.asserted_name, "is_task_success")

    def test_missing_expert_requires_official_policy_integration(self):
        failed = {"status": "failed"}
        self.assertEqual(next_action(failed, {}, {}, {}, False), "waiting_official_assets")
        self.assertEqual(next_action(failed, {}, {}, {}, True), "official-smoke")
        self.assertEqual(next_action(failed, {"status": "passed"}, {}, {}, True), "pilot")

    def test_full_smoke_does_not_bypass_two_round_pilot(self):
        smoke = {"status": "passed"}
        self.assertEqual(next_action(smoke, {}, {}, {}, True), "pilot")
        self.assertEqual(next_action(smoke, {}, {"status": "completed_development", "completed_rounds": 1}, {}, True),
                         "blocked_incomplete_pilot")
        pilot = {"status": "completed_development", "completed_rounds": 2}
        self.assertEqual(next_action(smoke, {}, pilot, {}, False), "waiting_official_assets")
        self.assertEqual(next_action(smoke, {}, pilot, {}, True), "research")

    def test_failures_are_not_retried_and_development_is_not_final(self):
        passed, failed = {"status": "passed"}, {"status": "failed"}
        self.assertEqual(next_action({}, failed, {}, {}, True), "blocked_policy_integration")
        self.assertEqual(next_action(passed, {}, failed, {}, True), "blocked_pilot")
        pilot = {"status": "completed_development", "completed_rounds": 2}
        self.assertEqual(next_action(passed, {}, pilot, failed, True), "blocked_research")
        self.assertEqual(next_action(passed, {}, pilot, pilot, True), "development_complete")


if __name__ == "__main__":
    unittest.main()
