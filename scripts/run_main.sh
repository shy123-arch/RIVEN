#!/bin/bash
# Paper main-table chain for one seed:
#   VQ (HML3D init) -> head 11,500 steps -> VAL selection -> TEST -> clean eval
#
#   set -a; . configs/paths.env; set +a
#   SEED=20261101 GPU=0 bash scripts/run_main.sh
#
# Prerequisites: condition caches (scripts/build_features.sh), SFT epoch2
# adapter, HERO checkout + RVQ checkpoint, and the clean evaluator bundle.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RIVEN_DATA:?set RIVEN_DATA}"
: "${RIVEN_CACHE:?set RIVEN_CACHE}"
: "${RIVEN_STATE:?set RIVEN_STATE}"

SEED=${SEED:-20261101}
GPU=${GPU:-0}
STEPS=${STEPS:-11500}
SAMPLES=${SAMPLES:-15}
CHECKPOINTS=${CHECKPOINTS:-9000,10500,last}
PY=${RIVEN_PY:-python}

"$PY" "$ROOT/experiments/main_table/run_main_table.py" \
  --release "$RIVEN_DATA" --seeds "$SEED" --gpu "$GPU" \
  --steps "$STEPS" --samples "$SAMPLES" --checkpoints "$CHECKPOINTS"
