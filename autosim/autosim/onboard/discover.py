"""
Task discovery and baseline evaluation.

Scans a robotics repo to find tasks, identify tunable strategy parameters,
and measure baseline performance.
"""

import os, sys, types, warnings, yaml, json
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")


def _setup_robotwin_mocks(repo_path):
    """Setup mocks needed for RoboTwin imports."""
    _m = types.ModuleType('open3d')
    _m.geometry = type('g', (), {'PointCloud': type('pc', (), {})})()
    _m.utility = type('u', (), {'Vector3dVector': lambda x: x})()
    _m.io = type('i', (), {'write_point_cloud': lambda *a, **k: None})()
    sys.modules['open3d'] = _m
    _m2 = types.ModuleType('test_render')
    _m2.Sapien_TEST = lambda: None
    sys.modules['test_render'] = _m2


# Registry of known repos and their task parameter extraction
REPO_ADAPTERS = {}


def register_adapter(repo_name):
    def decorator(fn):
        REPO_ADAPTERS[repo_name] = fn
        return fn
    return decorator


@register_adapter("RoboTwin")
def discover_robotwin(repo_path: str, task_name: str) -> dict:
    """Discover a RoboTwin task and its tunable parameters."""
    _setup_robotwin_mocks(repo_path)
    os.chdir(repo_path)
    sys.path.insert(0, '.')

    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    # Try to import the task module
    import importlib
    try:
        mod = importlib.import_module(f"envs.{task_name}")
        task_cls = getattr(mod, task_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Cannot find task '{task_name}' in {repo_path}: {e}")

    # Load task config
    config_path = os.path.join(repo_path, "task_config/demo_clean.yml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Extract embodiment info
    with open(os.path.join(CONFIGS_PATH, '_embodiment_config.yml')) as f:
        etypes = yaml.safe_load(f)
    embodiment = cfg.get('embodiment', ['aloha-agilex'])

    # Discover strategy parameters from task source
    task_file = os.path.join(repo_path, "envs", f"{task_name}.py")
    params = _extract_strategy_params(task_file)

    return {
        "name": task_name,
        "embodiment": embodiment,
        "params": params,
        "config": cfg,
        "task_file": task_file,
        "task_class": task_name,
    }


def _extract_strategy_params(task_file: str) -> dict:
    """Extract tunable strategy params from task source code."""
    params = {}
    try:
        with open(task_file) as f:
            source = f.read()
    except FileNotFoundError:
        return params

    # Look for getattr calls with defaults (our injected pattern)
    import re
    for match in re.finditer(
        r"getattr\(self,\s*'(_\w+)',\s*([\d.]+)\)",
        source
    ):
        name = match.group(1).lstrip('_')
        default = float(match.group(2))
        params[name] = {"default": default, "range": _guess_range(name, default)}

    # Also check for hardcoded kwargs in play_once
    for match in re.finditer(
        r'(pre_grasp_dis|grasp_dis|lift_z|pre_dis|dis)\s*=\s*([\d.]+)',
        source
    ):
        name = match.group(1)
        default = float(match.group(2))
        if name not in params:
            params[name] = {"default": default, "range": _guess_range(name, default)}

    return params


def _guess_range(name: str, default: float) -> list:
    """Guess reasonable optimization range for a parameter."""
    ranges = {
        "pre_grasp_dis": [0.04, 0.20],
        "grasp_dis": [0.0, 0.04],
        "lift_z": [0.04, 0.15],
        "pre_dis": [0.02, 0.12],
        "dis": [-0.03, 0.03],
    }
    if name in ranges:
        return ranges[name]
    return [default * 0.5, default * 2.0]


def discover_task(repo_path: str, task_name: str) -> dict:
    """Auto-detect repo type and discover task."""
    repo_name = os.path.basename(repo_path.rstrip('/'))
    if repo_name in REPO_ADAPTERS:
        return REPO_ADAPTERS[repo_name](repo_path, task_name)

    # Try RoboTwin adapter by checking for envs/ directory
    if os.path.isdir(os.path.join(repo_path, "envs")):
        return REPO_ADAPTERS["RoboTwin"](repo_path, task_name)

    raise ValueError(f"Unknown repo type: {repo_path}. Supported: {list(REPO_ADAPTERS.keys())}")


def run_baseline(repo_path: str, task_name: str, task_info: dict,
                 num_seeds: int = 50) -> dict:
    """Run baseline evaluation for a task."""
    global os, sys
    _setup_robotwin_mocks(repo_path)
    os.chdir(repo_path)
    sys.path.insert(0, '.')

    import importlib
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    task_cls = getattr(importlib.import_module(f"envs.{task_name}"), task_name)

    with open(f"task_config/demo_clean.yml") as f:
        cfg = yaml.safe_load(f)
    cfg['render_freq'] = 0; cfg['task_name'] = task_name; cfg['save_data'] = False
    cfg['save_path'] = '/tmp/autosim_baseline'
    cfg['domain_randomization']['random_background'] = False
    cfg['domain_randomization']['cluttered_table'] = False
    cfg['domain_randomization']['random_light'] = False

    with open(os.path.join(CONFIGS_PATH, '_embodiment_config.yml')) as f:
        etypes = yaml.safe_load(f)
    embodiment = cfg.get('embodiment', ['aloha-agilex'])
    cfg['left_robot_file'] = etypes[embodiment[0]]['file_path']
    cfg['right_robot_file'] = etypes[embodiment[0]]['file_path']
    cfg['dual_arm_embodied'] = len(embodiment) == 1

    with open(os.path.join(cfg['left_robot_file'], 'config.yml')) as f:
        ecfg = yaml.safe_load(f)
    cfg['left_embodiment_config'] = ecfg; cfg['right_embodiment_config'] = ecfg
    cfg['need_plan'] = True; cfg['clear_cache_freq'] = 999

    # Use default params
    default_params = {k: v['default'] for k, v in task_info['params'].items()}

    success = 0
    for seed in range(num_seeds):
        try:
            task = task_cls()
            for k, v in default_params.items():
                setattr(task, f"_{k}", v)
            task.setup_demo(now_ep_num=seed, seed=seed, **cfg)
            task.play_once()
            if task.plan_success and task.check_success():
                success += 1
            task.close_env()
        except Exception:
            pass

    return {"rate": success / num_seeds, "success": success, "total": num_seeds}
