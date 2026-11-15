# Reproduction

This document lists the external assets and the exact order of operations.
Nothing here is downloaded automatically; every path is set in
`configs/paths.env`.

## 1. Environments

The original runs used three Conda environments.  Names are arbitrary; only the
package sets matter.

| Variable | Used by | Contents |
|---|---|---|
| `RIVEN_PY_QWEN` | SFT, feature caches | transformers >= 4.57, peft, torch, ModelScope |
| `RIVEN_PY` | VQ, head training, generation | torch, numpy; no transformers needed |
| `RIVEN_PY_EVAL` | E2 / standard FID | torch, numpy, openpyxl (summaries) |

## 2. External assets

| Asset | Variable | Notes |
|---|---|---|
| EgoReReal release | `RIVEN_DATA` | `split.json`, `data_manifest.json`, `motion/humanml263_full.npz`, `motion/gt22_full.npz` |
| Qwen3-VL-4B-Instruct | `RIVEN_QWEN` | ModelScope/HF snapshot |
| Qwen SFT adapter | `RIVEN_SFT_ADAPTER` | LoRA adapter directory with `adapter_config.json` |
| Video tensor cache | `RIVEN_VIDEO_CACHE` | one `<id>.pt` per clip, written by the V->T data pipeline |
| HERO_release | `RIVEN_HERO` | commit `475774d` + `patches/hero-gen.patch` |
| TMR assets | `RIVEN_TMR_ASSETS` | installed by `python src/riven/eval/setup_tmr.py --archive ...` |
| E2 evaluator | `RIVEN_EVAL_DIR` | contains `E2_best.pt` |
| TMR motion bank | `RIVEN_TMR_MOTION_EMB` | optional; required by the trainer's NCE load even when `--lam-nce 0` |

The data release must be readable by the schema checks in `Release.validate()`
(ids, lengths, feature width, manifest rows).  No script reads sample content
for anything other than building training tensors.

## 3. HERO checkout

```bash
git clone https://github.com/JackYu6/HERO_release third_party/HERO
git -C third_party/HERO checkout 475774d
git -C third_party/HERO apply "$PWD/patches/hero-gen.patch"
export RIVEN_HERO="$PWD/third_party/HERO"
```

`finetune_hero.py` changes the working directory to HERO at import time and
imports its `gen`, `utils` and `models` packages.  Keep any other HERO checkout
out of `sys.path` while running RIVEN.

## 4. Qwen SFT

```bash
set -a; . configs/paths.env; set +a
cd src/riven/sft
VHO_PROMPT_MODE=plan_exec "$RIVEN_PY_QWEN" sft.py \
  --mode student --task answer --epochs 3 --lr 1e-4 --accum 8 \
  --r 16 --seed 20261101 --out sft_v4
```

`VHO_DATA` / `RIVEN_DATA` must point to the release: `common.manifest()` reads
`data_manifest.json`, keeps rows whose `group` is `train`/`val`, and uses the
`target` field as the supervision string.

Artifacts: `state/runs/sft_v4/{epoch1,epoch2,epoch3}`, `log.jsonl`,
`selection.json` (lowest VAL NLL per token).  **The paper mainline does not use
`selection.json` blindly**: on 2026-11-13 the SFT checkpoint was pinned to epoch
1 by downstream VAL FID.  Export that choice and keep it fixed across the three
generator seeds:

```bash
export RIVEN_SFT_ADAPTER="$RIVEN_STATE/runs/sft_v4/epoch1"
export RIVEN_CACHE_SUFFIX=_sft_ep1
```

The earlier v4 table (2026-11-10) used epoch2, selected by VAL text NLL
(`selection.json`).  The suffix selects a sibling cache directory so both can
coexist: `cache/<release>` vs `cache/<release>_sft_ep1`.  Use epoch1 for the
paper's final numbers.

## 5. Condition caches

```bash
SHARDS=2 GPUS="0 1" bash scripts/build_features.sh
```

Outputs:

```
$RIVEN_CACHE/textfeat/a_gt3p_trainval.{feats.npy,meta.json}
$RIVEN_CACHE/visualfeat/visual_trainval.shard*.npz
```

The prompt cache keeps the trailing 16 prompt tokens at layers 9/18/27/36; the
visual cache keeps 16 segment means at the same layers.  Both are fp16.

Optional controls:

- `build_base_textfeat.py` builds `a_gt3p_base` from the base Qwen model.
- Blank / shuffled video caches are produced by swapping the video cache and
  rerunning the script with names `blank_gt3p_trainval` /
  `shuffle_gt3p_trainval` and `visual_blank_trainval` / `visual_shuffle_trainval`.

## 6. Tokenizer

```bash
"$RIVEN_PY" src/riven/model/vq_train.py \
  --data "$RIVEN_DATA" --quantizers 6 --steps 3000 --window-max 208 \
  --seed "$SEED" --out "$RIVEN_CACHE/vq"
```

Only the first quantizer layer is loaded from the HumanML3D RVQ checkpoint
(`--all-codebooks` is off, matching the paper); the rest is trained on the
release.  Re-run per data version.

## 7. Action head

```bash
python experiments/ablations/run.py prepare --release "$RIVEN_DATA" --vq "$RIVEN_CACHE/vq/best.pt"
python experiments/ablations/run.py train \
  --ablation main_scratch --release "$RIVEN_DATA" \
  --gpu 0 --steps 5100 --seed 20261101 --tag v4_s20261101 --vq "$RIVEN_CACHE/vq/best.pt"
```

`main_scratch` is the paper configuration: RIV+AEM, learned null, cfg-mem,
RoleAdapter with `riv_prefix=direct`, full-length crop, head from scratch.
`--init-h1` is deliberately absent.

## 8. Generation and evaluation

```bash
python experiments/ablations/run.py generate --ablation main_scratch \
  --release "$RIVEN_DATA" --gpu 0 --samples 15 --split val --tag v4_s20261101
python experiments/ablations/run.py eval --ablation main_scratch \
  --release "$RIVEN_DATA" --gpu 0 --samples 15 --split val --tag v4_s20261101
```

Outputs under `$RIVEN_STATE/ablations/runs/<release>/<tag>/main_scratch/`:

```
train/last.pt
gen/val_S15.npz          # val_ids, val_joints [S,N,T,22,3], lengths
eval/val_S15.r.json
eval/val_S15.fid.json
```

TEST is run once, after all seeds and ablations are frozen:

```bash
--split test
```

## 9. Three seeds

The paper reports three seeds `{20261101, 20261102, 20261103}`.  The condition
caches are shared across seeds; the tokenizer and head are trained per seed.
Compute means and confidence intervals only after all three finish.  The
private experiment drivers that produced the paper's tables are not included;
the public runner writes one `eval/*.json` per seed, from which the same
mean/CI can be recomputed.

## 10. Ablations

```bash
python experiments/ablations/run.py status
python experiments/ablations/run.py generate --ablation riv_off --samples 15 --tag v4_s20261101
python experiments/ablations/run.py eval --ablation riv_off --samples 15 --tag v4_s20261101
```

Mechanism arms (`riv_off`, `aem_off`, `riv_aem_off`, `g_zero`, `g_mean`,
`m_zero`, `m_shuffle`, `e2_*_both`) reuse the main checkpoint.  Training arms
(`perceiver`, `role_time`, `tcclip`, `no_cfg_mem`, `drop_joint`, `window64`,
`scratch_head`, `b1_prefix`, `b2_memory`, `b3_flat`, `c1_tcclip`, `e2_drop25`,
`e2_aem_only`, `e2_riv_only`, `e3_*`, `w1_*`, `w2_hero_vq`, `w3_*`, `w4_*`,
`w6_cfg10`, `base_qwen`, `pt_only`, `sft_only`) need their own run.

## 11. Checks

```bash
python scripts/smoke.sh          # syntax + import graph, no data or GPU
python experiments/ablations/run.py prepare --release "$RIVEN_DATA"
```

`prepare` validates the release schema, cache presence, VQ and (optionally) the
head initialisation.  It does not inspect sample content.

## 12. Known limits

- The E2 evaluator checkpoint and its official TMR archive are external; the
  numbers are only comparable within this evaluator space.
- Audio is not used.
- The release's pending plan/exec rows must not be treated as supervision; the
  SFT manifest already filters by `group`.
- The video tensor cache depends on the V->T preprocessing (16 frames,
  360x640); changing it changes the condition.
