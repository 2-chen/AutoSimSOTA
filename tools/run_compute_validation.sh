#!/usr/bin/env bash
set -euo pipefail
VALIDATION_PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VALIDATION_BASE=/data/AutoResearch/AutoSimSOTA
export PYTHONPATH="$VALIDATION_PROJECT/autosim${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export EMBODICHAIN_SIM_EXIT_PROCESS=0
export LD_LIBRARY_PATH="$VALIDATION_BASE/container_libs/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME="$VALIDATION_BASE/compute_validation/dependencies/torch"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
bash "$VALIDATION_BASE/install_optix_runtime.sh"
if [ -s /tmp/optix-ld-extra ]; then
    VALIDATION_OPTIX=$(cat /tmp/optix-ld-extra)
    export LD_LIBRARY_PATH="$VALIDATION_OPTIX:$LD_LIBRARY_PATH"
fi
cd "$VALIDATION_PROJECT"
test -f "$VALIDATION_PROJECT/RoboSynChallenge/checkpoints/ACT_sim_click_bell/model.safetensors"
test -f "$VALIDATION_PROJECT/RoboSynChallenge/lerobot_dataset/RoboSynChallenge/cobotmagic_Sim_click_bell/meta/info.json"
# This entrypoint never invokes autosim research or the historical research smoke/full modes.
exec timeout --signal=TERM --kill-after=20s 3600s \
    "$VALIDATION_PROJECT/.venv/bin/python" tools/validate_compute_matrix.py --workspace "$VALIDATION_PROJECT" "$@"
