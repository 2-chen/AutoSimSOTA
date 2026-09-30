#!/usr/bin/env bash
set -euo pipefail
RESEARCH_SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$RESEARCH_SOURCE/autosim${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export EMBODICHAIN_SIM_EXIT_PROCESS=0
export LD_LIBRARY_PATH="/data/AutoResearch/AutoSimSOTA/container_libs/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/data/AutoResearch/AutoSimSOTA/compute_validation/dependencies/torch
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export AUTOSIM_ENV_FILE=/data/AutoResearch/AutoSimSOTA/AutoSimSOTA/.env
export AUTOSIM_REPAIR_BWRAP=/data/AutoResearch/AutoSimSOTA/harness_validation/dependencies/bwrap
# The Python launcher derives SCO_WORKER_SPEC from the frozen requested count.
export AUTOSIM_IMAGE=registry.cn-sh-01g.sensecore.cn/zhicheng_ccr/app-robotwin-wam:cci-20260521171320
cd "$RESEARCH_SOURCE"
"$RESEARCH_SOURCE/.venv/bin/python" - "$1" "$AUTOSIM_REPAIR_BWRAP" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

manifest = json.loads((Path(sys.argv[1]) / "launch_manifest.json").read_text())
if manifest.get("schema_version", 1) >= 2:
    expected = manifest.get("repair_sandbox", {})
    binary = Path(sys.argv[2])
    if (expected.get("binary") != str(binary) or not binary.is_file() or binary.is_symlink()
            or not os.access(binary, os.X_OK)
            or hashlib.sha256(binary.read_bytes()).hexdigest() != expected.get("sha256")):
        raise SystemExit("frozen offline bubblewrap dependency is missing or changed")
PY
exec "$RESEARCH_SOURCE/.venv/bin/python" tools/run_full_research_sco.py "$@"
