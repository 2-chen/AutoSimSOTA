"""Training-only ActionBank binding for incomplete benchmark expert entrypoints.

The production environment, randomization, dynamics, and success methods are
untouched. This only supplies the already published action graph to the expert.
"""

from __future__ import annotations

import importlib
import os
import runpy
import sys
from pathlib import Path
from types import MethodType


def capture_training_scene_evidence(env, observation, spec) -> dict:
    """Read post-reset training diagnostics without changing the environment.

    These values are evidence about what the randomizer actually produced.  They
    are never inserted into the recorded policy observation or action labels.
    Unsupported fields remain explicit in ``missing`` instead of being inferred
    from the requested collection profile.
    """
    from .policy_rpc import numpy_value

    def serial(value):
        if isinstance(value, tuple):
            return [serial(item) for item in value]
        converted = numpy_value(value)
        return converted.tolist() if hasattr(converted, "tolist") else converted

    base = getattr(env, "unwrapped", env)
    entities, cameras, lights, distractors, missing = {}, {}, {}, [], []
    sim = getattr(base, "sim", None)
    if sim is None:
        missing.append("sim")
    else:
        for kind, method in (("objects", "get_rigid_object"),
                             ("articulations", "get_articulation")):
            for uid in spec.roles.get(kind, []):
                try:
                    obj = getattr(sim, method)(uid)
                    value = {"pose": serial(obj.get_local_pose(to_matrix=True))}
                    if kind == "articulations":
                        value["qpos"] = serial(obj.get_qpos())
                    entities[uid] = value
                except Exception as exc:  # diagnostics must not alter collection
                    missing.append(f"entity:{uid}:{type(exc).__name__}")

    robot_qpos = None
    try:
        robot_qpos = serial(observation["robot"]["qpos"])
    except Exception as exc:
        missing.append(f"robot_qpos:{type(exc).__name__}")

    if sim is not None:
        for uid in getattr(spec, "cameras", ()):
            try:
                camera = sim.get_sensor(uid)
                cameras[uid] = {
                    "intrinsics": serial(camera.get_intrinsics()),
                    "local_pose": serial(camera.get_local_pose(to_matrix=True)),
                }
            except Exception as exc:
                missing.append(f"camera:{uid}:{type(exc).__name__}")
        try:
            for uid in sim.get_light_uid_list():
                light = sim.get_light(uid)
                lights[uid] = {"local_pose": serial(
                    light.get_local_pose(to_matrix=True))}
        except Exception as exc:
            missing.append(f"lights:{type(exc).__name__}")
        try:
            for uid in sim.get_rigid_object_uid_list():
                if not (uid.startswith("distractor_pool_")
                        or uid.startswith("distractor_")):
                    continue
                pose = serial(sim.get_rigid_object(uid).get_local_pose(to_matrix=True))
                # The collector hides inactive pool members far below the scene.
                # Keep the threshold and the observed pose explicit in evidence.
                import numpy as np
                array = np.asarray(pose)
                z = array[..., 2, 3]
                if bool(np.any(z > -5.0)):
                    distractors.append({"uid": uid, "pose": pose,
                                        "active_z_threshold": -5.0})
        except Exception as exc:
            missing.append(f"distractors:{type(exc).__name__}")

    # Current wrappers expose camera geometry and light position, but not the
    # realized RGB/intensity or visual material/texture after interval events.
    unavailable = ["light_color_intensity", "material_texture_parameters"]
    return {
        "entities": entities,
        "robot_qpos": robot_qpos,
        "cameras": cameras,
        "lights": lights,
        "active_distractors": distractors,
        "unavailable_realized_parameters": unavailable,
        "missing": missing,
        "privileged_training_diagnostics_only": True,
    }


class TrainingSceneTraceEnv:
    """Transparent reset proxy that persists realized collection evidence."""

    def __init__(self, env, spec, trace_path: Path, requested_profile: str):
        self._env = env
        self._spec = spec
        self._trace_path = trace_path
        self._requested_profile = requested_profile

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        from .common import event

        result = self._env.reset(*args, **kwargs)
        observation = result[0] if isinstance(result, tuple) else result
        try:
            evidence = capture_training_scene_evidence(
                self._env, observation, self._spec)
            event(self._trace_path, "collection_scene_reset",
                  schema_version=1, task=self._spec.name,
                  seed=kwargs.get("seed"),
                  requested_profile=self._requested_profile, **evidence)
        except Exception as exc:  # evidence failure must remain observable
            event(self._trace_path, "collection_scene_reset",
                  schema_version=1, task=self._spec.name,
                  seed=kwargs.get("seed"),
                  requested_profile=self._requested_profile,
                  entities={}, robot_qpos=None, cameras={}, lights={},
                  active_distractors=[],
                  unavailable_realized_parameters=[
                      "light_color_intensity", "material_texture_parameters"],
                  missing=[f"capture:{type(exc).__name__}"],
                  privileged_training_diagnostics_only=True)
        return result


def with_stepwise_success_history(generate, *, settle_steps=0, trace_path=None):
    """Use the policy adapter's per-action success cadence for expert rollouts.

    The original collector only queries success at reset. Stateful task judges
    consequently never observe intermediate grasps, displacements or stability.
    A bounded optional terminal hold lets physical stability criteria mature.
    Hold steps are real recorded actions, never repeated judge-only queries.
    """
    def execute(env, *args, **kwargs):
        import numpy as np
        from .policy_rpc import numpy_value

        class ExpertEnvironment:
            last_action = None
            success = False
            truncated = False
            steps = 0

            def __getattr__(self, name):
                return getattr(env, name)

            def step(self, action):
                result = env.step(action)
                self.last_action = action
                self.steps += 1
                self.success = bool(np.asarray(numpy_value(env.get_wrapper_attr("is_task_success")())).all())
                self.truncated = bool(np.asarray(numpy_value(result[3])).any())
                return result

        wrapped = ExpertEnvironment()
        valid = generate(wrapped, *args, **kwargs)
        original_steps, additional_steps = wrapped.steps, 0
        if valid and settle_steps and wrapped.last_action is not None:
            limit = int(env.get_wrapper_attr("_autosim_training_step_limit"))
            for _ in range(settle_steps):
                if wrapped.success or wrapped.truncated or wrapped.steps >= limit - 1:
                    break
                wrapped.step(wrapped.last_action)
                additional_steps += 1
        if trace_path is not None:
            from .common import event
            event(trace_path, "training_expert_execution", valid_plan=bool(valid),
                  original_steps=original_steps, terminal_hold_steps=additional_steps,
                  last_observed_success=wrapped.success, truncated=wrapped.truncated,
                  judge_queries_after_actions=wrapped.steps)
        return valid

    return execute


def explicit_joint_flip_contract(action_config: dict) -> dict:
    """Recover missing expert attributes from its own explicit validator.

    Do not invent or relax a threshold. Ambiguous/missing defaults fail closed.
    """
    from .common import object_digest

    contracts, visited = {}, set()

    def walk(value):
        if not isinstance(value, (dict, list)) or id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, dict):
            params = value.get("kwargs", {})
            if value.get("name") == "is_qpos_flip" and isinstance(params.get("qpos_ids"), list):
                if not isinstance(params.get("threshold"), (int, float)) or params.get("mode") not in {"delta", "sign"}:
                    raise ValueError("expert joint-flip contract must be explicit")
                contract = {"agent_qpos_flip_ids": params["qpos_ids"],
                            "agent_qpos_flip_threshold": params["threshold"],
                            "agent_qpos_flip_mode": params["mode"]}
                contracts[object_digest(contract)] = contract
            for child in value.values():
                walk(child)
        else:
            for child in value:
                walk(child)

    walk(action_config)
    if len(contracts) != 1:
        raise ValueError("missing or ambiguous joint-flip contract in official expert graph")
    return next(iter(contracts.values()))


def bind_action_bank(env, env_id: str, action_config: dict):
    from .registry import TASK_IDS, load_task

    task = next((name for name, identifier in TASK_IDS.items() if identifier == env_id), None)
    if task is None or load_task(Path.cwd(), task).expert_adapter != "action_bank":
        raise ValueError("task does not declare the training-only ActionBank adapter")
    if not action_config:
        raise ValueError("missing expert action graph")
    module = importlib.import_module(f"robosynchallenge.tasks.{task}.action_bank")
    bank = getattr(module, env_id + "ActionBank")
    base = env.unwrapped
    base.action_config = action_config
    for name, value in explicit_joint_flip_contract(action_config).items():
        if not hasattr(base, name):
            setattr(base, name, value)

    def create_demo(instance, *args, **kwargs):
        instance._init_action_bank(bank, instance.action_config)
        return instance.create_expert_demo_action_list(**kwargs)

    base.create_demo_action_list = MethodType(create_demo, base)
    return env


def main():
    import gymnasium as gym
    import embodichain.lab.scripts.run_env as expert_script
    from .registry import TASK_IDS, load_task

    settle_steps = int(os.environ.get("AUTOSIM_EXPERT_SETTLE_STEPS", "75"))
    if not 0 <= settle_steps <= 100:
        raise ValueError("expert terminal hold budget must be in [0,100]")
    manifest = Path(sys.argv[sys.argv.index("--collection_manifest") + 1])
    requested_profile = (
        sys.argv[sys.argv.index("--collection_profile") + 1]
        if "--collection_profile" in sys.argv else "full_random"
    )
    expert_script.generate_and_execute_action_list = with_stepwise_success_history(
        expert_script.generate_and_execute_action_list, settle_steps=settle_steps,
        trace_path=manifest.parent / "expert_execution.jsonl")

    original_make = gym.make

    def make(*args, **kwargs):
        env = original_make(*args, **kwargs)
        env_id = kwargs.get("id", args[0] if args else None)
        task = next((name for name, identifier in TASK_IDS.items() if identifier == env_id), None)
        if task is None:
            raise ValueError("undeclared training environment")
        spec = load_task(Path.cwd(), task)
        env.unwrapped._autosim_training_step_limit = spec.max_episode_steps
        if spec.expert_adapter != "official":
            action_config = kwargs.get("action_config") or {
                key: value for key, value in kwargs.items() if key not in {"id", "cfg"}}
            env = bind_action_bank(env, env_id, action_config)
        return TrainingSceneTraceEnv(
            env, spec, manifest.parent / "scene_resets.jsonl", requested_profile)

    gym.make = make
    runpy.run_path(str(Path.cwd() / "scripts/run_env.py"), run_name="__main__")


if __name__ == "__main__":
    main()
