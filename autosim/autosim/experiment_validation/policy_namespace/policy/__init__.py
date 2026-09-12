"""Resolve a benchmark policy namespace without copying policy source files.

The target path is explicit and is pinned by the integration manifest. This
regular package prevents another repository's unrelated ``policy`` package
from replacing RoboSynChallenge's intentionally implicit namespace package.
"""

from __future__ import annotations

import os
from pathlib import Path


root = Path(os.environ["AUTOSIM_POLICY_PACKAGE_ROOT"]).resolve()
if not root.is_dir():
    raise ImportError(f"AUTOSIM_POLICY_PACKAGE_ROOT is not a directory: {root}")

__path__ = [str(root)]

