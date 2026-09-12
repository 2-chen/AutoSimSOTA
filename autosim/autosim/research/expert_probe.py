"""Training-only diagnostic instrumentation, preserving all expert validations.

Invoke with the usual scripts.run_env arguments and a small attempt budget.
Records IK and joint-flip checks; it does not relax either check or the judge.
"""
from __future__ import annotations

import functools
import os
import runpy
import sys
from pathlib import Path

from .common import event


def main():
    import numpy as np
    from embodichain.lab.gym.utils import misc

    destination = Path(os.environ["AUTOSIM_EXPERT_PROBE_OUTPUT"])
    original_ik, original_flip = misc.cached_ik, misc.is_qpos_flip

    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value).tolist()

    @functools.wraps(original_ik)
    def cached_ik(*args, **kwargs):
        result = original_ik(*args, **kwargs)
        event(destination, "ik_check", target_pose=array(args[0] if args else kwargs["target_xpos"]),
              control_part=args[2] if len(args) > 2 else kwargs.get("control_part"),
              reference_qpos=array(args[4] if len(args) > 4 else kwargs.get("qpos_seed")),
              ik_valid=bool(result[0]), solution=array(result[1]))
        return result

    @functools.wraps(original_flip)
    def is_qpos_flip(*args, **kwargs):
        result = original_flip(*args, **kwargs)
        event(destination, "joint_flip_check", returned_valid=bool(result),
              input_qpos=array(args[0] if args else kwargs.get("qpos")),
              reference_qpos=array(kwargs.get("qpos_ref")),
              threshold=kwargs.get("threshold"), return_inverse=kwargs.get("return_inverse"))
        return result

    misc.cached_ik, misc.is_qpos_flip = cached_ik, is_qpos_flip
    if os.environ.get("AUTOSIM_EXPERT_STEPWISE_HISTORY") == "1":
        from .collection_worker import main as collection_main
        collection_main()
    else:
        from .registry import load_task
        config = Path(sys.argv[sys.argv.index("--gym_config") + 1])
        task = config.parent.parent.name
        if load_task(Path.cwd(), task).expert_adapter == "action_bank":
            raise ValueError("ActionBank diagnostics now require explicit stepwise-history mode; old attempts are preserved")
        runpy.run_path(str(Path.cwd() / "scripts/run_env.py"), run_name="__main__")


if __name__ == "__main__":
    main()
