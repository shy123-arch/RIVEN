#!/bin/bash
# End-to-end RIVEN main_scratch run for one seed.
#
#   set -a; . configs/paths.env; set +a
#   SEED=20261101 GPU=0 bash scripts/run_main.sh
#
# Prerequisites: condition caches (scripts/build_features.sh), Qwen SFT adapter,
# HERO checkout, E2 evaluator checkpoint.  See docs/REPRODUCE.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RIVEN_DATA:?set RIVEN_DATA}"
: "${RIVEN_CACHE:?set RIVEN_CACHE}"
: "${RIVEN_STATE:?set RIVEN_STATE}"

SEED=${SEED:-20261101}
GPU=${GPU:-0}
STEPS=${STEPS:-5100}
SAMPLES=${SAMPLES:-15}
TAG=${TAG:-v4_s${SEED}}
VQ=${RIVEN_VQ:-$RIVEN_CACHE/vq/best.pt}

PY=${RIVEN_PY:-python}
RUN="$ROOT/experiments/ablations/run.py"

if [ ! -f "$VQ" ]; then
  echo "[1/4] tokenizer"
  "$PY" "$ROOT/src/riven/model/vq_train.py" \
    --data "$RIVEN_DATA" --quantizers 6 --steps 3000 --window-max 208 \
    --seed "$SEED" --out "$(dirname "$VQ")"
else
  echo "[1/4] tokenizer exists: $VQ"
fi

echo "[2/4] prepare"
"$PY" "$RUN" prepare --release "$RIVEN_DATA" --vq "$VQ"

echo "[3/4] train head (seed $SEED)"
"$PY" "$RUN" train --ablation main_scratch --release "$RIVEN_DATA" \
  --gpu "$GPU" --steps "$STEPS" --seed "$SEED" --tag "$TAG" --vq "$VQ"

echo "[4/4] generate + evaluate VAL"
"$PY" "$RUN" generate --ablation main_scratch --release "$RIVEN_DATA" \
  --gpu "$GPU" --samples "$SAMPLES" --split val --tag "$TAG"
"$PY" "$RUN" eval --ablation main_scratch --release "$RIVEN_DATA" \
  --gpu "$GPU" --samples "$SAMPLES" --split val --tag "$TAG"

echo "done: $RIVEN_STATE/ablations/runs/$(basename "$RIVEN_DATA")/$TAG/main_scratch"
