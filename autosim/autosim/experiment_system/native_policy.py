"""RoboTwin native evaluator plugin: observation-only inference over existing RPC."""
import numpy as np


def reset_model(model):
    model.reset()


def eval(TASK_ENV, model, observation):
    # Only this whitelist crosses to the inference process, never the environment.
    batch = {"robot": {"qpos": np.asarray(observation["joint_action"]["vector"], dtype=np.float32)},
             "sensor": {camera: {"color": observation["observation"][camera]["rgb"]}
                        for camera in model.contract["cameras"]}}
    action = model.predict(batch)["action"][0]
    TASK_ENV.take_action(action)
