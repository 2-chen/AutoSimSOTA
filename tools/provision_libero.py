"""Provision LIBERO's environment, resuming whatever a previous attempt recorded.

Run from the repository root: `.venv/bin/python tools/provision_libero.py`. Long: conda
solving an old pinned dependency set is slow, and the build retries what failed.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "autosim"))

from autosim.llm_client import LLMClient                              # noqa: E402
from autosim.research import provision                                # noqa: E402
from autosim.research.repository_autoresearch import (                # noqa: E402
    PROJECT_ROOT, load_deepseek_environment)

load_deepseek_environment(project_root=PROJECT_ROOT)
repo = Path("/home/wbc/下载/autoresearch/test/LIBERO")
output = PROJECT_ROOT / "autoresearch_runs/provisioning/libero"

started = time.time()
result = provision.build(repo, client=LLMClient(), prefix=output / "env", output=output)
print("elapsed:", round((time.time() - started) / 60, 1), "min", flush=True)
print("verdict:", json.dumps(result["verdict"], ensure_ascii=False)[:500], flush=True)
print("survived:", result["survived"], "attempted:", result["attempted"], flush=True)
for row in result["record"]:
    print("  KEPT", row.get("kind", "cmd"),
          row["command"][:120].replace(str(output), "<out>"), flush=True)
