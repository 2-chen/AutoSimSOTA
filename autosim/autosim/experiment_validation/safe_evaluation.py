"""Versioned RoboSyn smoke evaluator with auditable numerical-failure handling.

This is deliberately not a ranking evaluator.  It preserves the official task,
seeds, success query and horizon, while constraining actions to the environment's
declared action space and turning a policy-induced non-finite simulator state into
an explicit failed episode instead of an infrastructure crash.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

from autosim.research.common import atomic_json, digest, event, freeze_files
from autosim.research.evaluation import TraceEnv
from autosim.research.policy_rpc import RemoteAdapter, numpy_value, observation_digest, policy_observation
from autosim.research.registry import load_task


def _done_like(value):
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return torch.ones_like(value, dtype=torch.bool)
    except ImportError:
        pass
    if isinstance(value, np.ndarray):
        return np.ones_like(value, dtype=bool)
    return True


class NumericalSafetyEnv(TraceEnv):
    """Records action interventions and fails an episode on invalid state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_count = 0
        self.clipped_action_count = 0
        self.clipped_element_count = 0
        self.invalid_state_episodes: list[dict] = []
        self.maximum_abs_raw_action = 0.0
        space = self.env.unwrapped.single_action_space
        self._low = np.asarray(space.low, dtype=np.float32).reshape(1, -1)
        self._high = np.asarray(space.high, dtype=np.float32).reshape(1, -1)
        if self._low.shape != (1, self.spec.action_dim) or self._high.shape != self._low.shape:
            raise ValueError("environment action-space shape differs from frozen contract")
        if np.isnan(self._low).any() or np.isnan(self._high).any() or np.any(self._low > self._high):
            raise ValueError("invalid environment action-space bounds")

    def step(self, action):
        raw = numpy_value(action).astype(np.float32, copy=False)
        if raw.ndim == 1:
            raw = raw[None]
        if raw.shape != (1, self.spec.action_dim) or not np.isfinite(raw).all():
            raise ValueError(f"invalid policy action before simulation: {raw.shape}")
        clipped = np.clip(raw, self._low, self._high)
        changed = raw != clipped
        self.action_count += 1
        self.clipped_action_count += int(changed.any())
        self.clipped_element_count += int(changed.sum())
        self.maximum_abs_raw_action = max(self.maximum_abs_raw_action, float(np.max(np.abs(raw))))
        if changed.any() or self.action_count == 1 or self.action_count % 10 == 0:
            event(self.output / "action_safety.jsonl", "action", seed=self.seed,
                  step=self.steps, clipped=bool(changed.any()), clipped_elements=int(changed.sum()),
                  raw_min=float(raw.min()), raw_max=float(raw.max()),
                  applied_min=float(clipped.min()), applied_max=float(clipped.max()))
        try:
            import torch
            if isinstance(action, torch.Tensor):
                applied = torch.as_tensor(clipped, dtype=action.dtype, device=action.device)
            else:
                applied = clipped
        except ImportError:
            applied = clipped
        result = self.env.step(applied)
        self.steps += 1
        state = numpy_value(result[0]["robot"]["qpos"])
        invalid = state.shape not in {(self.spec.state_dim,), (1, self.spec.state_dim)} or not np.isfinite(state).all()
        if invalid:
            row = {"seed": self.seed, "step": self.steps, "state_shape": list(state.shape),
                   "finite_elements": int(np.isfinite(state).sum()), "total_elements": int(state.size)}
            self.invalid_state_episodes.append(row)
            event(self.output / "action_safety.jsonl", "non_finite_state_episode_failure", **row)
            result = (result[0], result[1], result[2], _done_like(result[3]), result[4])
        elif self.steps % 10 == 0:
            self._trace(result[0], result[-1])
        return result

    def close(self):
        atomic_json(self.output / "safety_summary.json", {
            "status": "completed", "ranking_eligible": False,
            "policy_score_claim": False, "success_criterion_modified": False,
            "horizon_modified": False, "seed_protocol_modified": False,
            "action_rule": "clip_to_environment_single_action_space_bounds",
            "invalid_state_rule": "terminate_and_count_episode_as_failure",
            "action_count": self.action_count,
            "clipped_action_count": self.clipped_action_count,
            "clipped_element_count": self.clipped_element_count,
            "maximum_abs_raw_action": self.maximum_abs_raw_action,
            "invalid_state_episodes": self.invalid_state_episodes,
        })
        super().close()


class NumericalSafetyAdapter(RemoteAdapter):
    """Check truncation before success so invalid-state failures cannot pass."""

    def eval(self, env, model, obs):
        import torch

        timings = []
        info, truncated = None, False
        for _ in range(model.act_step):
            started = time.perf_counter()
            response = model.predict(obs)
            action = torch.as_tensor(response["action"], device=env.unwrapped.device, dtype=torch.float32)
            if response["fresh_inference"]:
                timings.append(time.perf_counter() - started)
            obs, _, _, truncated, info = env.step(action)
            if bool(numpy_value(truncated).any()):
                break
            if env.get_wrapper_attr("is_task_success")():
                break
        return obs, info, truncated, timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy", choices=["act", "dp"], required=True)
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
    adapter = NumericalSafetyAdapter(spec.as_dict())
    official.parse_args_and_config = lambda: config
    official.load_policy_adapter = lambda unused: adapter
    official.create_eval_run_dir = lambda unused: output
    make_env = official.make_env_from_configs

    def wrapped_factory(*factory_args, **factory_kwargs):
        atomic_json(output / "startup.json", {"phase": "environment_constructing", "policy_has_acted": False})
        env, gym_config = make_env(*factory_args, **factory_kwargs)
        wrapped = NumericalSafetyEnv(env, spec, output, adapter, "smoke")
        atomic_json(output / "startup.json", {"phase": "environment_ready", "policy_has_acted": False})
        return wrapped, gym_config

    official.make_env_from_configs = wrapped_factory
    files = [source, repo / f"policy/{args.policy}/deploy_policy.py", Path(__file__),
             Path(__file__).parents[1] / "research/evaluation.py",
             Path(__file__).parents[1] / "research/policy_rpc.py",
             Path(spec.gym_config), Path(spec.action_config),
             repo / f"robosynchallenge/tasks/{args.task}/{args.task}.py"]
    atomic_json(output / "protocol.json", {
        "schema_version": 2, "task": spec.as_dict(), "purpose": "integration_smoke",
        "ranking_eligible": False, "frozen_files": freeze_files(files),
        "checkpoint_sha256": digest(args.checkpoint / "model.safetensors"),
        "seed": args.seed, "episodes": args.episodes, "policy": args.policy,
        "compatibility_remediation": "numerical_failure_is_failed_episode_v2",
    })
    try:
        official.main()
    except BaseException as exc:
        atomic_json(output / "worker_failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
