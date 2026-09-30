#!/usr/bin/env bash
set -euo pipefail
REPLAY_PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPLAY_PROJECT/autosim${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export EMBODICHAIN_SIM_EXIT_PROCESS=0
export AUTOSIM_DYNAMICS_TRACE=1
export LD_LIBRARY_PATH="/data/AutoResearch/AutoSimSOTA/container_libs/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME=/data/AutoResearch/AutoSimSOTA/compute_validation/dependencies/torch
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
bash /data/AutoResearch/AutoSimSOTA/install_optix_runtime.sh
if [ -s /tmp/optix-ld-extra ]; then
    REPLAY_OPTIX=$(cat /tmp/optix-ld-extra)
    export LD_LIBRARY_PATH="$REPLAY_OPTIX:$LD_LIBRARY_PATH"
fi
cd "$REPLAY_PROJECT"
exec timeout --signal=TERM --kill-after=20s "${AUTOSIM_REPLAY_TIMEOUT_SECONDS:-2160}s" \
    "$REPLAY_PROJECT/.venv/bin/python" tools/replay_nonfinite_sco.py "$@"
