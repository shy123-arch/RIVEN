#!/bin/bash
# Build the Qwen condition caches for one data release.
#
#   set -a; . configs/paths.env; set +a
#   SHARDS=2 GPUS="0 1" bash scripts/build_features.sh
#
# Requires RIVEN_DATA, RIVEN_CACHE, RIVEN_SFT_ADAPTER, RIVEN_VIDEO_CACHE,
# RIVEN_QWEN and RIVEN_PY_QWEN.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RIVEN_DATA:?set RIVEN_DATA to the EgoReReal release}"
: "${RIVEN_CACHE:?set RIVEN_CACHE}"
: "${RIVEN_SFT_ADAPTER:?set RIVEN_SFT_ADAPTER to the SFT LoRA adapter}"
: "${RIVEN_VIDEO_CACHE:?set RIVEN_VIDEO_CACHE}"
: "${RIVEN_QWEN:?set RIVEN_QWEN to the Qwen3-VL-4B snapshot}"

SHARDS=${SHARDS:-1}
SPLITS=${SPLITS:-train_real,val}
QWEN_PY=${RIVEN_PY_QWEN:-python}
read -r -a GPUS <<< "${GPUS:-0}"

PROMPT="$ROOT/src/riven/features/cache_prompt_hidden.py"
MERGE="$ROOT/src/riven/features/merge_prompt_hidden.py"
VISUAL="$ROOT/src/riven/features/cache_visual_hidden.py"

mkdir -p "$RIVEN_CACHE/textfeat" "$RIVEN_CACHE/visualfeat"

for i in $(seq 0 $((SHARDS - 1))); do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 \
    "$QWEN_PY" "$PROMPT" --data "$RIVEN_DATA" --adapter "$RIVEN_SFT_ADAPTER" --base "$RIVEN_QWEN" \
    --video-cache "$RIVEN_VIDEO_CACHE" --out-dir "$RIVEN_CACHE/textfeat" \
    --name "${NAME:-a_gt3p_trainval}" --splits "$SPLITS" --shard "$i/$SHARDS" &
done
wait
env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME "$QWEN_PY" "$MERGE" \
  --out-dir "$RIVEN_CACHE/textfeat" --name "${NAME:-a_gt3p_trainval}"

for i in $(seq 0 $((SHARDS - 1))); do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 \
    "$QWEN_PY" "$VISUAL" --data "$RIVEN_DATA" --adapter "$RIVEN_SFT_ADAPTER" --base "$RIVEN_QWEN" \
    --video-cache "$RIVEN_VIDEO_CACHE" --out-dir "$RIVEN_CACHE/visualfeat" \
    --name "${VNAME:-visual_trainval}" --splits "$SPLITS" --shard "$i/$SHARDS" &
done
wait

echo "features built under $RIVEN_CACHE"
echo "blank/shuffle controls: build alternate video caches, then rerun with"
echo "  NAME=blank_gt3p_trainval VNAME=visual_blank_trainval ..."
