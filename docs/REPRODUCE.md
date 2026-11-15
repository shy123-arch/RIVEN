# Reproduction

Paper main table: RIVEN `main_scratch` on the unseen-source split, 200-epoch
head with VAL checkpoint selection, evaluated by the clean HML3D-init evaluator
(3 evaluator seeds).  This document lists the external assets and the exact
order of operations.

## 1. Environments

| Variable | Used by | Contents |
|---|---|---|
| `RIVEN_PY_QWEN` | SFT, video tensors, feature caches | transformers >= 4.57, peft, qwen-vl-utils, decord |
| `RIVEN_PY` | VQ, head training, generation | torch, numpy |
| `RIVEN_PY_EVAL` | clean evaluator | torch, numpy |

## 2. External assets

| Asset | Variable | Notes |
|---|---|---|
| EgoReReal release | `RIVEN_DATA` | **`umrm_v2280_unseen_20261112`**, 1854/213/213 |
| Clip videos | `RIVEN_CLIP_ROOT` / `RIVEN_CLIP_MAP` | 2,280 mp4, or id -> relative path |
| Qwen3-VL-4B-Instruct | `RIVEN_QWEN` | public weights |
| 10k ego PT adapter | `RIVEN_SFT_INIT_ADAPTER` | `s10k_pt/epoch1` (127 MB); SFT starts here |
| Qwen SFT adapter | `RIVEN_SFT_ADAPTER` | paper main table: `sft_unseen_full/epoch2` |
| HERO_release | `RIVEN_HERO` | commit `475774d` + `patches/hero-gen.patch` |
| HumanML3D RVQ checkpoint | `RIVEN_HML3D_RVQ` | **one** 81 MB file, `net_best_fid.tar` |
| Clean evaluator | `RIVEN_CLEAN_EVAL_ROOT` | 3 × `evaluator_hml3d_clean_s<seed>/best.tar` (87 MB each) |
| Clean eval corpus | `RIVEN_CLEAN_EVAL_DATA` | `motion/`, `text/`, `meta/`, `{train,val,test}_ids.txt` |
| TMR guoh3dfeats sources | `RIVEN_TMR_ASSETS` | only `official/` + `sources.json` are needed |
| TMR motion bank | `RIVEN_TMR_MOTION_EMB` | optional; NCE only |

The clean evaluator, eval corpus, SFT init adapter and RVQ checkpoint are
distributed as `riven_paper_assets_20261115` (594 MB) with `SHA256SUMS`.

The legacy E2 evaluator (`RIVEN_EVAL_DIR`, `E2_best.pt`) is **not** the paper
evaluator; it is kept only under `src/riven/eval/legacy_e2/`.

## 3. HERO checkout

```bash
git clone https://github.com/JackYu6/HERO_release third_party/HERO
git -C third_party/HERO checkout 475774d
git -C third_party/HERO apply "$PWD/patches/hero-gen.patch"
export RIVEN_HERO="$PWD/third_party/HERO"
```

`finetune_hero.py` changes the working directory to HERO at import time and
imports its `models`, `utils` and `gen` packages.

## 4. Clips and video tensors

```bash
export RIVEN_CLIP_ROOT=/data/egoreal/clips        # or RIVEN_CLIP_MAP=clip_map.json
"$RIVEN_PY_QWEN" src/riven/sft/prep_video_cache.py --workers 8
```

Output: `$RIVEN_CACHE/video/<id>.pt` (16 frames, max_pixels 360x640).  The
script stops with the missing paths if a clip cannot be resolved.

## 5. Qwen SFT (plan -> exec)

```bash
cd src/riven/sft
VHO_PROMPT_MODE=plan_exec VHO_DATA="$RIVEN_DATA" \
"$RIVEN_PY_QWEN" sft.py --mode student --task answer --epochs 3 --lr 1e-4 \
  --accum 8 --r 16 --seed 20261101 \
  --init-adapter "$RIVEN_SFT_INIT_ADAPTER" --out sft_unseen_full
```

Artifacts: `{epoch1,epoch2,epoch3}` plus `selection.json` (lowest VAL NLL per
token).  For the paper main table select **epoch2**:

```bash
export RIVEN_SFT_ADAPTER="$RIVEN_STATE/runs/sft_unseen_full/epoch2"
export RIVEN_CACHE_SUFFIX=          # empty
```

`RIVEN_CACHE_SUFFIX=_sft_ep1` selects the epoch1 cache used by the earlier
v4-split experiments; it is not the paper configuration.

## 6. Condition caches

```bash
SHARDS=2 GPUS="0 1" bash scripts/build_features.sh
```

Outputs: `$RIVEN_CACHE/textfeat/a_gt3p_trainval.{feats.npy,meta.json}` and
`$RIVEN_CACHE/visualfeat/visual_trainval.shard*.npz`.

## 7. Tokenizer

```bash
"$RIVEN_PY" src/riven/model/vq_train.py \
  --data "$RIVEN_DATA" --quantizers 6 --steps 3000 --window-max 208 \
  --seed "$SEED" --out "$RIVEN_CACHE/vq"
```

Only the first quantizer layer is loaded from `RIVEN_HML3D_RVQ`; the rest is
trained on the release.  No ViMo weights.

## 8. Head (200 epochs)

```bash
python experiments/main_table/run_main_table.py --gpu 0 --seeds 20261101
```

This runs, per seed: VQ -> `main_scratch` head for 11,500 steps -> VAL
evaluation at 9000 / 10500 / last -> selection -> TEST generation and
evaluation.  Equivalently, step by step:

```bash
python experiments/ablations/run.py train --ablation main_scratch \
  --release "$RIVEN_DATA" --gpu 0 --steps 11500 --seed 20261101 \
  --tag unseen200_s20261101 --vq "$RIVEN_CACHE/vq/best.pt"
```

## 9. Checkpoint selection and TEST

```bash
python experiments/main_table/select_test.py \
  --tag unseen200_s20261101 --gpu 0 --samples 15
```

Rule (paper): among VAL checkpoints, keep those within 0.5pp of the best R@10,
then take the lowest VAL FID.  The selected checkpoint generates TEST S=15;
`selection.json` records all candidates and the TEST metrics.

The paper's saved run used tag `unseen200full_s<seed>` for `main_scratch`; the
configuration is identical to `unseen200_s<seed>`.

## 10. Evaluation

`src/riven/eval/clean/eval_clean.py` runs the three evaluator seeds and reports
512-D FID and R@1/2/3/10:

```bash
python src/riven/eval/clean/eval_clean.py \
  --spec riven=gen/test_S15.npz --split test \
  --release "$RIVEN_DATA" --data "$RIVEN_CLEAN_EVAL_DATA" \
  --eval-root "$RIVEN_CLEAN_EVAL_ROOT" --out eval.json
```

Verified cell (2026-11-15): `unseen200sel_s20261101` TEST, evaluator seed
20261111 reproduces the recorded numbers exactly — FID 1.4959, R@1 .2036,
R@3 .4281, R@10 .7853.

## 11. Three seeds

Paper: model seeds `{20261101, 20261102, 20261103}` × evaluator seeds
`{20261111, 20261112, 20261113}`, TEST 213, S=15.  Report mean and standard
deviation across seeds after all runs finish.

## 12. Ablations

```bash
python experiments/ablations/run.py status
python experiments/ablations/run.py generate --ablation riv_off --samples 15 --tag unseen200_s20261101
python experiments/ablations/run.py eval --ablation riv_off --samples 15 --tag unseen200_s20261101
```

Mechanism arms reuse the selected checkpoint; training arms need their own run.
All arms use the same clean evaluator.

## 13. Checks

```bash
python scripts/smoke.sh
python experiments/ablations/run.py prepare --release "$RIVEN_DATA"
```

`prepare` validates the release schema, caches, VQ and the clean evaluator
bundle.  It does not inspect sample content.

## 14. Dataset package checklist

The data package must contain:

1. `split.json` — `train_real` / `val` / `test` (and `source_key`, `subsets`).
2. `data_manifest.json` — `rows[id]` with `group`, `exec`, `target`, `source`, `clip`.
3. `motion/humanml263_full.npz` — `ids`, `features`, `lengths`.
4. `motion/gt22_full.npz` — `J_<id>` for every split id.
5. The 2,280 clips, plus relative `clip` fields or `clip_map.json`.

Released separately: the clean evaluator (3 seeds) + eval corpus, the 10k PT
adapter, the HumanML3D RVQ checkpoint, Qwen3-VL-4B and HERO_release.

## 15. Known limits

- The clean evaluator and its corpus are external; FID and R are only
  comparable within this evaluator space.  The legacy E2 space is not.
- Audio is not used.
- The SFT epoch, the VAL checkpoint selection and the 3 evaluator seeds are
  part of the protocol; changing them changes the reported numbers.
- The video tensor cache depends on the 16-frame / 360x640 preprocessing.
