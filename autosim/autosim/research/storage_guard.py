"""Filesystem headroom checks, not repository-specific retention rules.

The Agent still plans save frequency and retention. This lower-bound check cannot
promise an arbitrarily long native job will fit, and never deletes old artifacts.
"""
import os
import shutil
from pathlib import Path


def preflight(path, *, stage, required_bytes=None):
    default = 2 * 1024**3 if stage in {'train', 'collect', 'prepare_data'} else 256 * 1024**2
    floor = int(os.environ.get('AUTOSIM_MIN_FREE_STORAGE_BYTES', str(default)))
    if floor < 0:
        raise ValueError('storage safety floor must be nonnegative')
    if required_bytes is not None and (isinstance(required_bytes, bool)
            or not isinstance(required_bytes, int) or required_bytes <= 0):
        raise ValueError('projected storage requirement must be a positive integer')
    needed = max(floor, required_bytes or 0)
    probe = Path(path).resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {'sufficient': usage.free >= needed, 'free_bytes':usage.free,
            'required_bytes':needed, 'safety_floor_bytes':floor,
            'projected_bytes':required_bytes, 'checked_directory':str(probe),
            'scope':'launch headroom only; not a reservation or full-job fit guarantee'}
