# RIVEN ablation harness

One registry describes every arm as a small override of the paper mainline;
one runner renders the exact upstream CLI for train / generate / eval / smoke.
No ablation re-implements the model.

| Stage | Code |
|---|---|
| Train | `src/riven/model/h1_6q_train.py` |
| Generate | `src/riven/model/h1_6q_gen_condswap.py` |
| VQ | `src/riven/model/vq_train.py` (6q, HumanML3D init) |
| Evaluate | `src/riven/eval/eval_r.py`, `src/riven/eval/eval_fid.py` |

## Base configuration (`main_scratch`)

RIV + AEM, learned null, `cfg_mem`, RoleAdapter with `riv_prefix=direct`,
no time tokens, full-length crop, 6q VQ, head from scratch, 5100 steps.

## Arms

| Group | Names |
|---|---|
| Mechanism (inference-only) | `riv_off`, `aem_off`, `riv_aem_off`, `g_zero`, `g_mean`, `m_zero`, `m_shuffle`, `e2_aem_only_both`, `e2_riv_only_both`, `e11_cfg15`, `e11_cfg20` |
| Condition source | `tcclip`, `c1_tcclip`, `perceiver`, `b1_prefix`, `b2_memory`, `b3_flat`, `base_qwen`, `pt_only`, `sft_only` |
| Adapter design | `role_time`, `e3_no_plan`, `e3_plan8`, `e3_vis8`, `e3_riv18`, `e3_riv27`, `e3_aem18`, `e3_aem36`, `e3_pool_last`, `e3_pool_attn`, `e3_same36`, `e3_swap_all`, `w1_swap_inject`, `w1_cap_flat`, `w3_flat6m`, `w3_roleswap_inv`, `w4_riv27_aem18`, `w4_riv27_aem36` |
| Null / dropout | `no_cfg_mem`, `drop_joint`, `w6_cfg10`, `e2_drop25`, `e2_aem_only`, `e2_riv_only` |
| Controls | `blank_video`, `shuffle_video` |
| Objective / data / init | `window64`, `nce_*`, `ep1_*`, `scratch_head`, `w2_hero_vq` |

The table is descriptive; `registry.py` is authoritative and holds the exact
flag overrides.

## Entry points

```bash
cd experiments/ablations

python run.py prepare --release "$RIVEN_DATA" --vq "$RIVEN_CACHE/vq/best.pt"

python run.py train --ablation main_scratch --release "$RIVEN_DATA" \
  --gpu 0 --steps 5100 --seed 20261101 --tag v4_s20261101

python run.py generate --ablation riv_off --release "$RIVEN_DATA" \
  --gpu 0 --samples 15 --tag v4_s20261101

python run.py eval --ablation riv_off --release "$RIVEN_DATA" \
  --gpu 0 --samples 15 --tag v4_s20261101

python run.py smoke --ablation main --gpu 0 --train-smoke
```

Artifacts:

```
$RIVEN_STATE/ablations/data/<release>/ready.json
$RIVEN_STATE/ablations/cache/<release>/{textfeat,visualfeat,tcclip_*.npz}
$RIVEN_STATE/ablations/runs/<release>/<tag>/<ablation>/{train,gen,eval,logs}
```

## Protocol

- VAL 213, GT target length, 15 samples, no rerank.
- R: fixed E2, per-sample, source-excluded 32-pool, 20 pools/query.
- FID: E2 256-D latent with TRAIN-only PCA64, per generation sample.
- Training arms: 3 seeds.  Mechanism arms: same checkpoint, inference only.
- TEST only after all arms are frozen.
