#!/usr/bin/env bash
# Derive what is not yet derived, keeping what is, until both stages have a command.
#
# The derivation is unreliable in one place -- which name a config key has -- and each
# attempt is expensive. What it produces is not: a verified command is recorded, so an
# attempt that succeeds is never repeated. This runs attempts until the record is complete.
set -u
ROOT=/home/wbc/下载/autoresearch/AutoSimSOTA
REPO=/home/wbc/下载/autoresearch/test/LIBERO
OUT=$ROOT/autoresearch_runs/provisioning/libero
KEPT=$OUT/derived_stages.json

for attempt in $(seq 1 8); do
  HAVE=$(python3 -c "
import json,os
p='$KEPT'
print(len(json.load(open(p))) if os.path.isfile(p) else 0)")
  echo "=== attempt $attempt: $HAVE stage(s) already kept"
  if [ "$HAVE" -ge 2 ]; then break; fi
  cd "$ROOT" || exit 1
  PYTHONUNBUFFERED=1 .venv/bin/python tools/research_derived.py "$REPO" "$OUT" 0 \
    > "$ROOT/autoresearch_runs/derived/attempt_$attempt.log" 2>&1
  echo "    exit $? ; kept now: $(python3 -c "
import json,os
p='$KEPT'
print(list(json.load(open(p))) if os.path.isfile(p) else [])")"
done

echo "=== starting the measurement"
cd "$ROOT" || exit 1
PYTHONUNBUFFERED=1 .venv/bin/python tools/research_derived.py "$REPO" "$OUT" 2 \
  '{"train.n_epochs": 50, "eval.n_eval": 20, "device": "cuda", "train.num_workers": 0, "eval.num_workers": 0}' \
  > "$ROOT/autoresearch_runs/derived/libero_full_run.log" 2>&1
echo "measurement exit $?"
