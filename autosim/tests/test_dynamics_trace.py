"""The optional native-unit trace must observe without changing control inputs."""
import json
from types import SimpleNamespace

import numpy as np

from autosim.research.evaluation import TraceEnv


def test_diagnostic_trace_records_raw_units_and_does_not_mutate(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOSIM_DYNAMICS_TRACE", "1")
    raw = np.array([[0.1, 0.02, 0.3]])
    robot = SimpleNamespace(joint_names=["arm", "gripper", "mimic"],
        body_data=SimpleNamespace(qpos_limits=np.array([[[-1., 1.], [0., .04], [0., .04]]]),
                                  _target_qpos=np.zeros((1, 3))),
        get_qpos=lambda: raw, get_qvel=lambda: np.ones((1, 3)))
    env = SimpleNamespace(robot=robot, active_joint_ids=[0, 1])
    env.unwrapped = env
    proxy = TraceEnv(env, None, tmp_path, None, "development")
    proxy.seed, proxy.steps = 123, 1
    obs = {"robot": {"qpos": np.array([[0.1, 0.5]])}}
    action = np.array([[.2, .01]])
    proxy._capture_dynamics(action, obs)
    row = json.loads((tmp_path / "dynamics_trace.jsonl").read_text())
    assert row["qpos"] == [0.1, 0.5] and row["raw_qpos_all"] == [0.1, .02, .3]
    assert row["action"] == [.2, .01]
    assert json.loads((tmp_path / "dynamics_contract.json").read_text())["active_joint_ids"] == [0, 1]
    np.testing.assert_array_equal(raw, [[.1, .02, .3]])
    np.testing.assert_array_equal(action, [[.2, .01]])
