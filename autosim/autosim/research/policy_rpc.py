"""Observation-only ACT/DP worker. No environment object crosses the pipe."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import time
from typing import Any

import numpy as np


def numpy_value(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def policy_observation(obs: dict, contract: dict) -> dict[str, np.ndarray]:
    state = numpy_value(obs["robot"]["qpos"]).astype(np.float32)
    if state.ndim == 1:
        state = state[None]
    if state.shape != (1, contract["state_dim"]) or not np.isfinite(state).all():
        raise ValueError(f"invalid state shape/values: {state.shape}")
    batch = {"observation.state": np.ascontiguousarray(state)}
    for camera in contract["cameras"]:
        value = numpy_value(obs["sensor"][camera]["color"])
        if value.ndim == 3:
            value = value[None]
        value = value[..., :3]
        if list(value.shape) != [1] + contract["camera_shapes"][camera]:
            raise ValueError(f"camera {camera} violates contract: {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite camera {camera}")
        batch[f"observation.images.{camera}"] = np.ascontiguousarray(value)
    return batch


def observation_digest(batch: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in sorted(batch):
        h.update(key.encode())
        h.update(str(batch[key].dtype).encode())
        h.update(str(batch[key].shape).encode())
        h.update(batch[key].tobytes())
    return h.hexdigest()


def _worker(connection, config: dict, contract: dict) -> None:
    # Only explicit, non-secret model parameters and observation contract are
    # passed as arguments. No task seed, evaluator, telemetry or environment.
    import importlib
    import torch

    try:
        adapter = importlib.import_module(f"policy.{config['policy_name']}")
        policy = adapter.get_model(config)
        device = next(policy.parameters()).device
        allowed = {"observation.state"} | {f"observation.images.{c}" for c in contract["cameras"]}
        image_keys = set(policy.config.image_features)
        if image_keys != allowed - {"observation.state"}:
            raise ValueError("checkpoint cameras differ from task observation contract")
        connection.send({"ready": True})
        while True:
            message = connection.recv()
            if message["operation"] == "close":
                break
            if message["operation"] == "reset":
                policy.reset()
                connection.send({"reset": True})
                continue
            if message["operation"] != "predict" or set(message["batch"]) != allowed:
                raise ValueError("invalid policy request or forbidden observation keys")
            if config["policy_name"] == "act":
                fresh = policy.config.temporal_ensemble_coeff is not None or len(policy._action_queue) == 0
            else:
                queue = getattr(policy, "_queues", {}).get("action")
                fresh = queue is None or len(queue) == 0
            started = time.perf_counter()
            batch = {}
            for key, value in message["batch"].items():
                tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
                if key.startswith("observation.images."):
                    tensor = tensor.permute(0, 3, 1, 2).contiguous()
                    if tensor.max() > 1.5:
                        tensor = tensor / 255.0
                batch[key] = tensor
            with torch.inference_mode():
                action = policy.select_action(batch).detach().cpu().numpy()
            if action.ndim == 1:
                action = action[None]
            if action.shape != (1, contract["action_dim"]) or not np.isfinite(action).all():
                raise ValueError(f"invalid action: {action.shape}")
            connection.send({"action": action, "fresh_inference": fresh,
                             "worker_seconds": time.perf_counter() - started})
    except BaseException as exc:
        try:
            connection.send({"error": f"{type(exc).__name__}: {exc}"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class RemotePolicy:
    def __init__(self, config: dict, contract: dict):
        allowed_config = {k: config[k] for k in (
            "checkpoint_path", "policy_name", "pytorch_device", "act_step", "dp_step",
            "image_key_map", "dp_num_inference_steps", "act_n_action_steps_override") if k in config}
        self.contract = {k: contract[k] for k in ("state_dim", "action_dim", "cameras", "camera_shapes")}
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, allowed_config, self.contract), daemon=True)
        self.process.start()
        child.close()
        try:
            self._receive()
        except BaseException:
            self.close()
            raise
        self.act_step = int(config.get("act_step" if config["policy_name"] == "act" else "dp_step", 10))

    def _receive(self) -> dict:
        if not self.connection.poll(300):
            raise TimeoutError("policy worker did not respond within 300 seconds")
        result = self.connection.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def reset(self) -> None:
        self.connection.send({"operation": "reset"})
        self._receive()

    def predict(self, obs: dict) -> dict:
        self.connection.send({"operation": "predict", "batch": policy_observation(obs, self.contract)})
        return self._receive()

    def close(self) -> None:
        if self.process.is_alive():
            try:
                self.connection.send({"operation": "close"})
                self.process.join(10)
            except (BrokenPipeError, EOFError, OSError):
                pass
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(5)
        self.connection.close()


class RemoteAdapter:
    """Preserves official adapter success-query cadence, including batch return."""

    def __init__(self, contract: dict):
        self.contract = contract
        self.model = None

    def get_model(self, config):
        self.model = RemotePolicy(config, self.contract)
        return self.model

    def reset_model(self, model):
        model.reset()

    def eval(self, env, model, obs):
        import torch

        timings = []
        for _ in range(model.act_step):
            started = time.perf_counter()
            response = model.predict(obs)
            action = torch.as_tensor(response["action"], device=env.unwrapped.device, dtype=torch.float32)
            if response["fresh_inference"]:
                timings.append(time.perf_counter() - started)
            obs, _, _, truncated, info = env.step(action)
            if env.get_wrapper_attr("is_task_success")():
                break
            if bool(numpy_value(truncated).any()):
                break
        return obs, info, truncated, timings

    def close(self):
        if self.model is not None:
            self.model.close()
