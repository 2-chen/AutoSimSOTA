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


def seed_offset_stream(*, offset: int, master_seed: int, receipt: Path | None = None) -> dict:
    """Start the collector's own seed stream ``offset`` draws in, without editing it.

    The benchmark builds one ``np.random.RandomState(collection_seed)`` and draws one
    ``randint(0, 2**31 - 1)`` per reset, so burning ``offset`` identical draws first makes
    this process's first reset the master bank's ``offset``-th seed.  Only the draw *count*
    changes the stream, so the slice is exact rather than an approximation -- the same
    argument, and the same numpy-level shim, as ``evaluation._seeded_numpy``.

    Matching on the *seed value* is what keeps every other RandomState in the process alone:
    the benchmark's correction path builds ``RandomState(collection_seed ^ 0x5A17C0DE)`` and
    the engine seeds itself, and neither should be shifted by a collection block.  A second
    construction with the plain master seed is refused rather than double-burned, because
    that would mean the file grew a second stream and this shim no longer covers what it
    claims to.

    ``receipt`` is rewritten as the stream is seen: this process is expected to end in the
    benchmark's own native exit (see ``main``), so nothing after the run can report what the
    shim did.  The receipt is the shard's direct evidence that its block start was performed
    rather than merely declared -- the coverage audit reads it.
    """
    import numpy

    from .common import atomic_json

    real = numpy.random.RandomState
    state = {"offset": int(offset), "master_seed": int(master_seed), "streams_seen": 0,
             "burned_draws": 0}

    def publish():
        if receipt is not None:
            atomic_json(Path(receipt), dict(state, kind="collection_seed_offset",
                                            schema_version=1))

    class _OffsetRandomState:
        """A RandomState that skips the draws belonging to earlier blocks."""

        def __new__(cls, seed=None, *args, **kwargs):
            instance = real(seed, *args, **kwargs) if seed is not None else real(*args, **kwargs)
            if isinstance(seed, (int, numpy.integer)) and int(seed) == state["master_seed"]:
                state["streams_seen"] += 1
                if state["streams_seen"] > 1:
                    raise RuntimeError(
                        "the collector constructed a second RandomState(master_seed); the "
                        "seed-offset shim covers exactly one collection stream")
                for _ in range(state["offset"]):
                    instance.randint(0, 2**31 - 1)
                state["burned_draws"] = state["offset"]
                publish()
            return instance

    publish()                       # streams_seen=0: the shim is installed and watching
    numpy.random.RandomState = _OffsetRandomState
    return state


def _pop_seed_offset(argv: list[str]) -> int:
    """Take ``--collection_seed_offset`` out of argv: run_env.py's parser does not know it."""
    if "--collection_seed_offset" not in argv:
        return 0
    position = argv.index("--collection_seed_offset")
    offset = int(argv[position + 1])
    del argv[position:position + 2]
    if offset < 0:
        raise ValueError("collection seed offset must not be negative")
    return offset


def main():
    # The benchmark's scripts/run_env.py flushes its collection manifest *before*
    # env.close() precisely because DexSim's destroy() is expected to end the process
    # from native code (os._exit(0)) right there. Our launcher disables that native exit
    # globally, because the official evaluator writes evaluation_metrics.json *after*
    # env.close() and would otherwise lose it -- but with the native exit disabled this
    # collection run only survives close() to segfault in the interpreter's final GC
    # (exit code -11, which autosim records as a failed collection). Restore the
    # teardown the benchmark expects; collection is the only stage that wants it.
    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "1"

    # Under a device plan this process builds the environment with every card visible (the
    # engine resolves its physical index by NVML UUID), so anything that takes the *default*
    # device -- warp's ``wp.launch`` in a contact sensor, a bare ``cuda`` in torch -- would
    # land on card 0 rather than on the card this collection leased.  The plan's own ordinal
    # is in the environment (see ``Runtime.environment``); a legacy single-GPU collection has
    # no such variable and no such ambiguity.
    from .devices import align_process_defaults, default_device_index

    planned_index = default_device_index()
    if planned_index is not None:
        align_process_defaults(planned_index)

    import gymnasium as gym
    import embodichain.lab.scripts.run_env as expert_script
    from .registry import TASK_IDS, load_task

    settle_steps = int(os.environ.get("AUTOSIM_EXPERT_SETTLE_STEPS", "75"))
    if not 0 <= settle_steps <= 100:
        raise ValueError("expert terminal hold budget must be in [0,100]")
    # A sharded collection starts its share of the attempt stream mid-bank.  The offset has
    # to be consumed here rather than forwarded: run_env.py's parser has no such flag.
    offset = _pop_seed_offset(sys.argv)
    manifest = Path(sys.argv[sys.argv.index("--collection_manifest") + 1])
    from .native_cache import configure_native_cache
    configure_native_cache(manifest.parent)
    if offset:
        if "--collection_seed" not in sys.argv:
            raise ValueError("a collection seed offset needs the master --collection_seed")
        seed_offset_stream(offset=offset,
                           master_seed=int(sys.argv[sys.argv.index("--collection_seed") + 1]),
                           receipt=manifest.parent / "seed_offset.json")
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
