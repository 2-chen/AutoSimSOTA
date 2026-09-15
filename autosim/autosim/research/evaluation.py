"""Trusted wrapper around the unchanged official evaluator; observation-only RPC."""

from __future__ import annotations

import argparse
import importlib.util
import os
from collections import deque
from pathlib import Path

from .common import assert_frozen, atomic_json, digest, event, freeze_files, read_json
from .collection_worker import capture_training_scene_evidence
from .policy_rpc import RemoteAdapter, numpy_value, observation_digest, policy_observation
from .registry import load_task
from .native_lifecycle import NativeLifecycle


def _seeded_numpy(offset: int):
    """Reproduce a slice of the master seed bank without editing the official evaluator.

    The evaluator builds one ``np.random.RandomState(seed)`` before its episode loop and
    draws one ``randint(0, 2**31 - 1)`` per episode, so burning ``offset`` identical draws
    first makes episode ``j`` of this process use the master bank's ``offset + j``-th seed.
    Only the draw *count* changes the stream, so this is exact, not an approximation.
    """

    class _SeedOffsetRandom:
        def __init__(self, real, offset: int):
            self._real, self._offset, self._used = real, offset, False

        def RandomState(self, seed, *args, **kwargs):
            if self._used:
                raise RuntimeError(
                    "official evaluator constructed a second RandomState; the seed-offset "
                    "shim covers exactly one stream")
            if offset == 0:
                return self._real.RandomState(seed, *args, **kwargs)
            state = self._real.RandomState(seed, *args, **kwargs)
            for _ in range(offset):
                state.randint(0, 2**31 - 1)
            self._used = True
            return state

        def __getattr__(self, name):
            return getattr(self._real, name)

    class _SeedOffsetNumpy:
        def __init__(self, real, offset: int):
            self._real = real
            self.random = _SeedOffsetRandom(real.random, offset)

        def __getattr__(self, name):
            return getattr(self._real, name)

    return _SeedOffsetNumpy(__import__("numpy"), offset)


#: The evaluator process's own view of its default device, filled by ``_select_cuda`` and
#: reported in every ``startup.json`` the process writes.
_ALIGNMENT: dict = {}


def _select_cuda(index: int, output: Path) -> None:
    """Align torch -- and warp, under a device plan -- with the leased card.

    ``index`` is the leased card's *CUDA* ordinal, not the engine's physical index (see
    ``devices.py``).  This is the official evaluator's own ``select_cuda_device`` call
    (``scripts/eval_policy.py:604``), which runs before the environment and before the policy
    child are built, so it is also the last point at which a library's *default* device can
    still be pointed at the leased card instead of at card zero.

    Measured on 2026-09-13 (4-GPU ladder, cards 1-3): the alignment takes effect -- every
    arm's ``startup.json`` records ``torch_default``/``warp_default`` on its own card -- and
    it removes about half of the leak the second probe saw (1030 MiB on card 0 before, 525
    MiB after).  The remainder is therefore *not* torch's and not warp's default device, and
    the census written here is what tells the two candidate phases apart.
    """
    from .devices import align_process_defaults, default_device_index

    record = align_process_defaults(int(index), warp=default_device_index() is not None)
    from .native_cache import configure_native_cache
    configure_native_cache(output)
    _ALIGNMENT.clear()
    _ALIGNMENT.update(record)
    startup_phase(output, "devices_aligned", policy_has_acted=False)


def _own_memory_mib() -> dict:
    """How much memory *this* process -- and its children -- hold on each card, right now.

    The 2026-09-13 4-GPU ladder could say "525 MiB appeared on card 0 while the arm ran on
    card 1, held by that arm's own pid" and still not say *when*, which is the difference
    between two opposite fixes: memory present before the engine is constructed belongs to a
    library import, memory that appears while it is constructed belongs to the engine.  Read
    at every phase, this turns one number into a timeline.

    Never raises: a diagnostic that can fail would turn a measurement into a crash.
    """
    try:
        from .accounting import describe_process
        from .devices import parse_compute_apps, run_text

        text = run_text(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                         "--format=csv,noheader,nounits"], timeout=20)
        if text.strip().startswith("__error__"):
            return {"error": text.strip()[:120]}
        mine = os.getpid()
        own: dict = {}
        children: dict = {}
        for app in parse_compute_apps(text):
            pid, uuid = app.get("pid"), app.get("gpu_uuid")
            if uuid is None or pid is None:
                continue
            if pid == mine:
                own[uuid] = max(own.get(uuid, 0), app.get("used_memory_mib") or 0)
            elif describe_process(int(pid)).get("parent") == mine:
                children[uuid] = max(children.get(uuid, 0), app.get("used_memory_mib") or 0)
        return {"own": own, "children": children}
    except Exception as exc:                          # noqa: BLE001 -- diagnostics are free
        return {"error": f"{type(exc).__name__}: {exc}"[:120]}


def startup_phase(output: Path, phase: str, **extra) -> None:
    """Write one phase receipt, always carrying whose process and which default device.

    A phase is what a caller reads to tell "died before the engine came up" from "died in the
    first reset".  The pid and the device alignment are here because the cross-card residual
    is a *process* question -- ``1030 MiB on card 0, owner unknown`` was the one finding two
    probes could not act on.

    The per-card census is appended to ``startup_census.jsonl`` as well, because
    ``startup.json`` keeps only the latest phase and the timeline is what names a phase.
    """
    census = _own_memory_mib()
    atomic_json(output / "startup.json", {"phase": phase, "pid": os.getpid(),
                                          "default_device": dict(_ALIGNMENT),
                                          "own_memory_mib": census, **extra})
    event(output / "startup_census.jsonl", "startup_phase", phase=phase, pid=os.getpid(),
          default_device=dict(_ALIGNMENT), memory=census)



def certification_fields(spec, purpose: str) -> dict:
    """Fields only autosim can certify; the official payload cannot know them."""
    return {
        "purpose": purpose,
        "execution_mode": "real_simulation",
        "harness": "official_control_loop_observation_only_rpc_v1",
        "policy_observation_contract": {k: spec.as_dict()[k] for k in
            ("state_dim", "action_dim", "cameras", "camera_shapes")},
        "rpc_timing_note": "same-machine bridge overhead included on fresh inference calls; "
                           "not directly comparable to historical in-process latency",
    }


class TraceEnv:
    """Only reads state. It never calls success, reset, RNG or step for diagnostics."""

    def __init__(self, env, spec, output: Path, adapter, purpose, lifecycle=None):
        self.env, self.spec, self.output, self.adapter = env, spec, output, adapter
        self.purpose = purpose
        self.seed = None
        self.steps = 0
        self._native_barrier_done = False
        self.lifecycle = lifecycle
        self._recent_dynamics = deque(maxlen=4)
        self._nonfinite_recorded = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        if self.lifecycle:
            self.lifecycle.before_reset()
        startup_phase(self.output, "evaluation_reset_started", seed=kwargs.get("seed"))
        obs, info = self.env.reset(*args, **kwargs)
        self.seed, self.steps = kwargs.get("seed"), 0
        self._recent_dynamics.clear()
        fingerprint = observation_digest(policy_observation(obs, self.spec.as_dict()))
        event(self.output / "initializations.jsonl", "reset", seed=self.seed,
              allowed_observation_sha256=fingerprint)
        atomic_json(self.output / "heartbeat.json", {"phase": "episode_started", "seed": self.seed})
        self._trace(obs, info)
        return obs, info

    def step(self, action):
        if self.lifecycle:
            self.lifecycle.step_started()
        result = self.env.step(action)
        if self.lifecycle:
            self.lifecycle.step_completed()
        self.steps += 1
        self._capture_dynamics(action, result[0])
        if self.steps % 10 == 0:
            self._trace(result[0], result[-1])
        return result

    def _capture_dynamics(self, action, obs):
        """Keep four steps in memory; flush once on invalid state, without changing it.

        No extra simulator calls and no camera payloads. Final banks remain sealed.
        A diagnostic I/O failure must not replace the actual evaluator outcome.
        """
        if self.purpose in {"final", "final_confirmation"} or self._nonfinite_recorded:
            return
        try:
            import numpy as np
            def numbers(value):
                array = numpy_value(value).reshape(-1)
                return [float(v) if np.isfinite(v) else None for v in array[:256]]
            state = numpy_value(obs["robot"]["qpos"])
            self._recent_dynamics.append({"step": self.steps, "action": numbers(action),
                "qpos": numbers(state), "qvel": (numbers(obs["robot"]["qvel"])
                                                  if "qvel" in obs["robot"] else None)})
            if os.environ.get("AUTOSIM_DYNAMICS_TRACE") == "1":
                # Explicit diagnostic replay only; keep native units alongside
                # normalized policy observations to identify actuator/contract bugs.
                robot = self.env.unwrapped.robot
                if not (self.output / "dynamics_contract.json").exists():
                    atomic_json(self.output / "dynamics_contract.json", {
                        "joint_names": list(robot.joint_names),
                        "active_joint_ids": numbers(self.env.unwrapped.active_joint_ids),
                        "qpos_limits": numpy_value(robot.body_data.qpos_limits).tolist()})
                event(self.output / "dynamics_trace.jsonl", "dynamics", seed=self.seed,
                    **self._recent_dynamics[-1], raw_qpos_all=numbers(robot.get_qpos()),
                    raw_qvel_all=numbers(robot.get_qvel()),
                    target_qpos_all=numbers(robot.body_data._target_qpos))
            if not np.isfinite(state).all():
                atomic_json(self.output / "nonfinite_dynamics.json", {
                    "seed": self.seed, "first_observed_step": self.steps,
                    "nonfinite_indices": np.flatnonzero(~np.isfinite(state)).tolist(),
                    "recent_steps": list(self._recent_dynamics),
                    "nonfinite_encoding": "null; original observation is never modified",
                    "causal_status": "first observed invalid state; underlying solver cause not established"})
                self._nonfinite_recorded = True
        except Exception:
            pass

    def _trace(self, obs, info):
        if self.purpose in {"final", "final_confirmation"}:
            return
        scene = capture_training_scene_evidence(self.env, obs, self.spec)
        event(self.output / "telemetry.jsonl", "observation", seed=self.seed, step=self.steps,
              task=self.spec.name, family=self.spec.roles["family"],
              entities=scene["entities"], robot_qpos=scene["robot_qpos"],
              cameras=scene["cameras"], lights=scene["lights"],
              active_distractors=scene["active_distractors"],
              unavailable_realized_parameters=scene["unavailable_realized_parameters"],
              missing=scene["missing"],
              privileged_development_diagnostics_only=True)

    def close(self):
        if self.lifecycle:
            self.lifecycle.work_finished()
        path = self.output / "evaluation_metrics.json"
        if path.exists():
            assert_frozen(read_json(self.output / "protocol.json")["frozen_files"])
            result = read_json(path)
            result.update(certification_fields(self.spec, self.purpose))
            atomic_json(path, result)
        self.adapter.close()
        self.env.close()


def certify_official_metrics(output: Path, spec, purpose: str) -> None:
    """Stamp autosim's contract onto the payload the official evaluator wrote.

    The official evaluator closes the environment from a ``finally`` *before* it
    writes ``evaluation_metrics.json`` (scripts/eval_policy.py), so the update
    TraceEnv.close() performs never sees that file.  Do it here instead, and bridge
    the schema difference: official episode rows key their seed ``seed``, while the
    rest of this pipeline keys it ``episode_seed``.
    """
    path = output / "evaluation_metrics.json"
    result = read_json(path)
    rows = result.get("episodes")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("official evaluator wrote no episode rows")
    for row in rows:
        if "episode_seed" not in row:
            if "seed" not in row:
                raise RuntimeError("official episode row carries no seed")
            row["episode_seed"] = row["seed"]
    result.update(certification_fields(spec, purpose))
    atomic_json(path, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--purpose", choices=["smoke", "development", "selection_validation",
                                               "final_confirmation", "confirmation", "final"], required=True)
    parser.add_argument("--policy", choices=["act", "dp"], default="act")
    parser.add_argument("--device-gpu-id", type=int, default=0,
                        help="physical/NVML index handed to the engine as gpu_id")
    parser.add_argument("--device-torch-index", type=int, default=0,
                        help="index this device has inside this process after CUDA_VISIBLE_DEVICES")
    parser.add_argument("--renderer", default="hybrid")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.renderer == "auto":
        raise ValueError("renderer=auto resolves through a CUDA-space index and silently "
                         "falls back to 'hybrid'; pass an explicit renderer")
    if args.seed_offset and args.shard_count == 1 and args.shard_index == 0:
        raise ValueError("--seed-offset requires the shard it belongs to")
    repo, output = Path.cwd(), args.output.absolute()
    if (output / "evaluation_metrics.json").exists():
        raise FileExistsError("refusing to overwrite an existing evaluation")
    output.mkdir(parents=True, exist_ok=True)
    lifecycle = NativeLifecycle(output)
    lifecycle.start()
    # The census starts here, and it starts *before* anything official is loaded: everything
    # autosim imports at module scope (including whatever ``policy_rpc`` pulls in) is already
    # loaded, the official evaluator's own imports are not.  Memory already on a foreign card
    # at this row is autosim's or its imports'; memory that appears only at the next row is the
    # official evaluator's module-scope import (embodichain/dexsim/torch/warp).
    startup_phase(output, "main_start", policy_has_acted=False)
    spec = load_task(repo, args.task)
    source = repo / "scripts/eval_policy.py"
    module_spec = importlib.util.spec_from_file_location("official_robosyn_evaluator", source)
    official = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(official)
    startup_phase(output, "official_imported", policy_has_acted=False)
    import yaml

    config = yaml.safe_load((repo / f"policy/{args.policy}/deploy_policy.yml").read_text())
    config.update(task_name=args.task, setting="random", policy_name=args.policy,
                  checkpoint_path=str(args.checkpoint.absolute()), max_episodes=args.episodes,
                  seed=args.seed, eval_video_log=False, eval_reset_sync_steps=0,
                  headless=True, renderer=args.renderer,
                  pytorch_device=("cuda" if args.device_torch_index == 0
                                  else f"cuda:{args.device_torch_index}"),
                  gpu_id=args.device_gpu_id)
    if config.get("eval_fixed_episode_seed") is not None and (args.seed_offset or args.shard_count > 1):
        raise ValueError("the task config pins eval_fixed_episode_seed; seed slices would be "
                         "meaningless because the evaluator never draws from the bank")
    if args.seed_offset:
        # The evaluator draws from ``np.random`` at module scope, so the shim replaces the
        # module's numpy and intercepts ``np.random.RandomState`` -- not ``np`` alone.
        official.np = _seeded_numpy(args.seed_offset)
    adapter = RemoteAdapter(spec.as_dict())
    official.parse_args_and_config = lambda: config
    official.load_policy_adapter = lambda unused: adapter
    official.create_eval_run_dir = lambda unused: output
    official.select_cuda_device = lambda unused: _select_cuda(args.device_torch_index, output)
    make_env = official.make_env_from_configs

    def wrapped_factory(*factory_args, **factory_kwargs):
        startup_phase(output, "environment_constructing", policy_has_acted=False)
        lifecycle.constructing()
        if lifecycle.config.get("inject_before_ready"):
            # Explicit native-admission validation only; never present in scored workers.
            if args.purpose != "smoke" or not lifecycle.config.get("validation_injection"):
                raise ValueError("native fault injection is restricted to validation probes")
            import signal
            atomic_json(output / "injected_failure.json", {**lifecycle.identity,
                        "kind": "SIGSEGV", "phase": "constructing", "reset_started": False})
            os.kill(os.getpid(), signal.SIGSEGV)
        env, gym_config = make_env(*factory_args, **factory_kwargs)
        startup_phase(output, "environment_ready", policy_has_acted=False)
        lifecycle.ready()
        if lifecycle.config:
            startup_phase(output, "native_barrier_released", policy_has_acted=False)
        return TraceEnv(env, spec, output, adapter, args.purpose, lifecycle), gym_config

    official.make_env_from_configs = wrapped_factory
    files = [source, repo / f"policy/{args.policy}/deploy_policy.py", Path(__file__),
             Path(__file__).with_name("policy_rpc.py"), Path(spec.gym_config), Path(spec.action_config),
             repo / f"robosynchallenge/tasks/{args.task}/{args.task}.py"]
    compatibility = repo / f"policy/{args.policy}/checkpoint_compat.py"
    if compatibility.is_file():
        files.append(compatibility)
    protocol = {"task": spec.as_dict(), "purpose": args.purpose,
                "frozen_files": freeze_files(files), "checkpoint_sha256": digest(args.checkpoint / "model.safetensors"),
                "seed": args.seed, "episodes": args.episodes, "policy": args.policy}
    if args.shard_count > 1:
        # The slice identity belongs to this process only; the merge compares protocol.json
        # across shards *modulo* this block, so it must stay out of everything else.
        protocol["shard"] = {"index": args.shard_index, "count": args.shard_count,
                             "seed_offset": args.seed_offset, "episodes": args.episodes}
    atomic_json(output / "protocol.json", protocol)
    try:
        official.main()
    except BaseException as exc:
        atomic_json(output / "worker_failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        lifecycle.failed(exc)
        raise
    else:
        certify_official_metrics(output, spec, args.purpose)
        lifecycle.completed()
    finally:
        adapter.close()
        lifecycle.close()


if __name__ == "__main__":
    main()
