"""Trusted wrapper around the unchanged official evaluator; observation-only RPC."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from .common import assert_frozen, atomic_json, digest, event, freeze_files, read_json
from .collection_worker import capture_training_scene_evidence
from .policy_rpc import RemoteAdapter, observation_digest, policy_observation
from .registry import load_task


class TraceEnv:
    """Only reads state. It never calls success, reset, RNG or step for diagnostics."""

    def __init__(self, env, spec, output: Path, adapter, purpose):
        self.env, self.spec, self.output, self.adapter = env, spec, output, adapter
        self.purpose = purpose
        self.seed = None
        self.steps = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        atomic_json(self.output / "startup.json", {"phase": "evaluation_reset_started", "seed": kwargs.get("seed")})
        obs, info = self.env.reset(*args, **kwargs)
        self.seed, self.steps = kwargs.get("seed"), 0
        fingerprint = observation_digest(policy_observation(obs, self.spec.as_dict()))
        event(self.output / "initializations.jsonl", "reset", seed=self.seed,
              allowed_observation_sha256=fingerprint)
        atomic_json(self.output / "heartbeat.json", {"phase": "episode_started", "seed": self.seed})
        self._trace(obs, info)
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        self.steps += 1
        if self.steps % 10 == 0:
            self._trace(result[0], result[-1])
        return result

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
        path = self.output / "evaluation_metrics.json"
        if path.exists():
            assert_frozen(read_json(self.output / "protocol.json")["frozen_files"])
            result = read_json(path)
            result.update(purpose=self.purpose, execution_mode="real_simulation",
                          harness="official_control_loop_observation_only_rpc_v1",
                          policy_observation_contract={k: self.spec.as_dict()[k] for k in
                              ("state_dim", "action_dim", "cameras", "camera_shapes")},
                          rpc_timing_note="same-machine bridge overhead included on fresh inference calls; not directly comparable to historical in-process latency")
            atomic_json(path, result)
        self.adapter.close()
        self.env.close()


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
    args = parser.parse_args()
    repo, output = Path.cwd(), args.output.absolute()
    if (output / "evaluation_metrics.json").exists():
        raise FileExistsError("refusing to overwrite an existing evaluation")
    output.mkdir(parents=True, exist_ok=True)
    spec = load_task(repo, args.task)
    source = repo / "scripts/eval_policy.py"
    module_spec = importlib.util.spec_from_file_location("official_robosyn_evaluator", source)
    official = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(official)
    import yaml

    config = yaml.safe_load((repo / f"policy/{args.policy}/deploy_policy.yml").read_text())
    config.update(task_name=args.task, setting="random", policy_name=args.policy,
                  checkpoint_path=str(args.checkpoint.absolute()), max_episodes=args.episodes,
                  seed=args.seed, eval_video_log=False, eval_reset_sync_steps=0,
                  headless=True, renderer="hybrid", pytorch_device="cuda", gpu_id=0)
    adapter = RemoteAdapter(spec.as_dict())
    official.parse_args_and_config = lambda: config
    official.load_policy_adapter = lambda unused: adapter
    official.create_eval_run_dir = lambda unused: output
    make_env = official.make_env_from_configs

    def wrapped_factory(*factory_args, **factory_kwargs):
        atomic_json(output / "startup.json", {"phase": "environment_constructing", "policy_has_acted": False})
        env, gym_config = make_env(*factory_args, **factory_kwargs)
        atomic_json(output / "startup.json", {"phase": "environment_ready", "policy_has_acted": False})
        return TraceEnv(env, spec, output, adapter, args.purpose), gym_config

    official.make_env_from_configs = wrapped_factory
    files = [source, repo / f"policy/{args.policy}/deploy_policy.py", Path(__file__),
             Path(__file__).with_name("policy_rpc.py"), Path(spec.gym_config), Path(spec.action_config),
             repo / f"robosynchallenge/tasks/{args.task}/{args.task}.py"]
    atomic_json(output / "protocol.json", {"task": spec.as_dict(), "purpose": args.purpose,
                "frozen_files": freeze_files(files), "checkpoint_sha256": digest(args.checkpoint / "model.safetensors"),
                "seed": args.seed, "episodes": args.episodes, "policy": args.policy})
    try:
        official.main()
    except BaseException as exc:
        atomic_json(output / "worker_failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
