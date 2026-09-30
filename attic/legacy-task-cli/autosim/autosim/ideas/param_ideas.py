"""
Parameter idea generation — AutoSOTA-style tiered candidates.

AutoSOTA tiers: ALGO (architecture) > CODE (logic) > PARAM (tuning)
For robotics tasks, we focus on PARAM tier initially, with room for
ALGO (strategy change) and CODE (motion planning adjustment) tiers.
"""

import numpy as np
from typing import Dict, List


def generate_ideas(params: Dict, n_random: int = 20, n_grid: int = 5,
                   n_local: int = 10, seed: int = 42) -> List[Dict]:
    """
    Generate parameter candidates in tiers.

    Returns: list of parameter dicts with '_tier' and '_source' metadata.
    """
    rng = np.random.RandomState(seed)
    ideas = []

    # Tier 3 (PARAM): Random exploration
    for _ in range(n_random):
        candidate = {}
        for name, info in params.items():
            lo, hi = _get_range(info)
            candidate[name] = float(rng.uniform(lo, hi))
        candidate['_tier'] = 'PARAM'
        candidate['_source'] = 'random_uniform'
        ideas.append(candidate)

    # Tier 3 (PARAM): Local mutation around defaults
    defaults = {k: v['default'] for k, v in params.items()}
    for _ in range(n_local):
        candidate = {}
        for name, info in params.items():
            lo, hi = _get_range(info)
            delta = rng.normal(0, (hi - lo) * 0.05)
            candidate[name] = float(np.clip(defaults[name] + delta, lo, hi))
        candidate['_tier'] = 'PARAM'
        candidate['_source'] = 'local_mutation'
        ideas.append(candidate)

    # Tier 3 (PARAM): Grid search over key parameters
    key_params = list(params.keys())[:3]  # first 3 params
    if len(key_params) >= 2 and n_grid > 0:
        for i in range(n_grid):
            candidate = dict(defaults)
            for kp in key_params:
                lo, hi = _get_range(params[kp])
                candidate[kp] = float(np.linspace(lo, hi, n_grid)[i])
            candidate['_tier'] = 'PARAM'
            candidate['_source'] = 'grid_search'
            ideas.append(candidate)

    return ideas


def _get_range(info: Dict) -> tuple:
    """Get parameter range from info dict."""
    if 'range' in info and isinstance(info['range'], list):
        return info['range'][0], info['range'][1]
    default = info.get('default', 1.0)
    return default * 0.5, default * 2.0
