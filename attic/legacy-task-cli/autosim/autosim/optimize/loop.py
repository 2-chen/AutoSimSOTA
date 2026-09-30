"""
AutoSOTA-style optimization loop.

Flow per iteration:
  1. Select candidate from idea pool
  2. Evaluate: run task with candidate params, measure success rate
  3. Record: update scores.jsonl, track best
  4. Reflect: update idea pool based on results
"""

import os, sys, types, warnings, yaml, json, time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

OUTPUT_DIR = None


def _setup_robotwin(repo_path):
    _m = types.ModuleType('open3d')
    _m.geometry = type('g', (), {'PointCloud': type('pc', (), {})})()
    _m.utility = type('u', (), {'Vector3dVector': lambda x: x})()
    _m.io = type('i', (), {'write_point_cloud': lambda *a, **k: None})()
    sys.modules['open3d'] = _m
    sys.modules['test_render'] = type(sys)('test_render')
    sys.modules['test_render'].Sapien_TEST = lambda: None
    os.chdir(repo_path)
    if '.' not in sys.path:
        sys.path.insert(0, '.')


class OptimizeLoop:
    """Main optimization loop for robotics task strategy parameters."""

    def __init__(self, config: Dict, ideas: List[Dict]):
        self.config = config
        self.repo_path = config['repo_path']
        self.task_name = config['task_name']
        self.ideas = ideas
        self.baseline_rate = config.get('baseline_rate', 0.0)
        self.scores = []
        self.best_rate = self.baseline_rate
        self.best_params = {k: v['default'] for k, v in config['params'].items()}
        self.baseline_params = dict(self.best_params)
        self._rng = np.random.RandomState(42)

    def evaluate(self, params: Dict, seeds: List[int]) -> float:
        """Run task with given params, return success rate."""
        _setup_robotwin(self.repo_path)

        import importlib
        from envs._GLOBAL_CONFIGS import CONFIGS_PATH

        task_cls = getattr(importlib.import_module(f"envs.{self.task_name}"), self.task_name)

        with open("task_config/demo_clean.yml") as f:
            cfg = yaml.safe_load(f)
        cfg['render_freq'] = 0; cfg['task_name'] = self.task_name; cfg['save_data'] = False
        cfg['save_path'] = '/tmp/autosim_opt'
        for k in ['random_background','cluttered_table','random_light']:
            if k in cfg.get('domain_randomization', {}):
                cfg['domain_randomization'][k] = False

        with open(os.path.join(CONFIGS_PATH, '_embodiment_config.yml')) as f:
            etypes = yaml.safe_load(f)
        emb = cfg.get('embodiment', ['aloha-agilex'])
        cfg['left_robot_file'] = etypes[emb[0]]['file_path']
        cfg['right_robot_file'] = etypes[emb[0]]['file_path']
        cfg['dual_arm_embodied'] = len(emb) == 1
        with open(os.path.join(cfg['left_robot_file'], 'config.yml')) as f:
            ecfg = yaml.safe_load(f)
        cfg['left_embodiment_config'] = ecfg; cfg['right_embodiment_config'] = ecfg
        cfg['need_plan'] = True

        success = 0
        for seed in seeds:
            try:
                task = task_cls()
                for k, v in params.items():
                    if not k.startswith('_'):
                        setattr(task, f"_{k}", v)
                task.setup_demo(now_ep_num=seed, seed=seed, **cfg)
                task.play_once()
                if task.plan_success and task.check_success():
                    success += 1
                task.close_env()
            except Exception:
                pass

        return success / len(seeds) if seeds else 0.0

    def run(self, max_iterations: int = 30, seeds_per_eval: int = 15) -> Dict:
        """Run the full optimization loop."""
        eval_seeds = list(range(seeds_per_eval))

        # Shuffle ideas
        self._rng.shuffle(self.ideas)
        candidates = self.ideas[:max_iterations]

        print(f"\n  Evaluating {len(candidates)} candidates ({seeds_per_eval} seeds each)...")

        for i, cand in enumerate(candidates):
            # Remove metadata keys
            params = {k: v for k, v in cand.items() if not k.startswith('_')}
            rate = self.evaluate(params, eval_seeds)

            record = {
                "iteration": i,
                "params": params,
                "primary_score": rate,
                "tier": cand.get('_tier', 'PARAM'),
                "source": cand.get('_source', 'unknown'),
                "timestamp": datetime.now().isoformat(),
            }
            self.scores.append(record)

            improved = rate > self.best_rate
            if improved:
                self.best_rate = rate
                self.best_params = dict(params)

            marker = " ★ BEST" if improved else ""
            print(f"  [{i+1:3d}/{len(candidates)}] "
                  f"rate={rate*100:5.1f}%"
                  f"  pg={params.get('pre_grasp_dis','?'):.2f}"
                  f"  gd={params.get('grasp_dis','?'):.3f}"
                  f"  lz={params.get('lift_z','?'):.2f}"
                  f"{marker}")

        # Save scores.jsonl
        global OUTPUT_DIR
        if OUTPUT_DIR:
            scores_path = os.path.join(OUTPUT_DIR, "scores.jsonl")
            with open(scores_path, "w") as f:
                for s in self.scores:
                    f.write(json.dumps(s) + "\n")

        return {
            "task_name": self.task_name,
            "baseline_params": self.baseline_params,
            "baseline_rate": self.baseline_rate,
            "best_params": self.best_params,
            "best_rate": self.best_rate,
            "num_evaluations": len(self.scores),
            "scores": self.scores,
        }


def set_output_dir(path: str):
    global OUTPUT_DIR
    OUTPUT_DIR = path
