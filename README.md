# RIVEN

**Reaction Intent and Evidence-conditioned Network for egocentric reaction motion generation.**

Given a first-person video that ends right before the wearer reacts, RIVEN predicts
the next 1-4 seconds of full-body motion.  This repository is the code release of
the paper's mainline (`main_scratch`): condition encoding, tokenizer, action head,
generation, the ablation registry, and the fixed-evaluator protocol.

Data, model weights, Qwen adapters, HERO and the TMR assets are **not** included.
`docs/REPRODUCE.md` lists every external asset and where to put it.

## Pipeline

```
pre-reaction video
      │
      ▼
Qwen3-VL-4B + LoRA (plan -> exec SFT)          condition encoder
      │  prompt hidden [layer 36]        visual hidden [layers 9/18/27/36]
      ▼                                        │
  RIV  (reaction intent, g)  ───────┐          │
  AEM  (evidence memory, m)  ───────┤          │
      │  RoleAdapter, direct prefix │          │
      ▼                             ▼          ▼
6-quantizer RVQ tokenizer  ──►  masked transformer (q1) + residual transformer (q2..q6)
      │
      ▼
generated 22-joint motion (GT length, CFG on the RIV/AEM branches)
      │
      ▼
E2 evaluator: per-sample R@1/3/10 (source-excluded 32-pool) + FID (TRAIN PCA64)
```

RIVEN's two channels are deliberately asymmetric: **RIV** is a coarse intent
commitment consumed as the prefix condition (`c = r`), while **AEM** is
addressed as key/value memory and never collapses into the prefix.  A learned
null plus `cfg_mem` supplies the unconditional branch at generation time.

## Repository layout

```
src/riven/model/       h1_6q_train.py, h1_6q_gen_condswap.py, cond_adapters.py,
                       ordered_adapter.py, intent_patch.py, cfg_patch.py,
                       motion_windows.py, vq_train.py, HERO glue
src/riven/sft/         Qwen plan->exec SFT (UMRM-OPD pipeline, copied)
src/riven/features/    Qwen prompt/visual hidden-state caches, TMR embedding bank
src/riven/eval/        E2 evaluator (eval_r.py, eval_fid.py), official TMR adapter
experiments/ablations/ registry.py (all paper ablations) + run.py / prepare.py
configs/               paths.env.example, main.yaml
scripts/               build_features.sh, run_main.sh, smoke.sh
patches/               hero-gen.patch (10-line local HERO change)
tools/                 check_repository.py (static checks, no GPU/data)
docs/                  METHOD.md, REPRODUCE.md, PROVENANCE.md, SOURCE_SNAPSHOT.json
```

## Quick start

```bash
git clone <this repo> riven && cd riven
pip install -r requirements.txt
python scripts/smoke.sh                 # static checks + import graph, no data/GPU

cp configs/paths.env.example configs/paths.env
$EDITOR configs/paths.env               # data, state, cache, HERO, Qwen, adapters
set -a; . configs/paths.env; set +a

# Paper mainline: SFT epoch1 (selected by VAL FID) + its sibling cache.
export RIVEN_SFT_ADAPTER="$RIVEN_STATE/runs/sft_v4/epoch1"
export RIVEN_CACHE_SUFFIX=_sft_ep1

bash scripts/build_features.sh          # Qwen prompt/visual hidden states
SEED=20261101 GPU=0 bash scripts/run_main.sh
```

Full instructions, data contract and evaluation protocol: **`docs/REPRODUCE.md`**.

## What is fixed

- Data version `umrm_v2280_split_v4_20261111`: 1,854 / 213 / 213.
- All methods share one evaluator (E2), GT target length, and 15 samples per item.
- R is per-sample, source-excluded 32-pool; FID uses the E2 256-D latent with a
  TRAIN-only PCA64 basis.  Numbers from different evaluators are not comparable.
- The action head is trained from scratch; the tokenizer is initialised from
  HumanML3D RVQ.  No EgoReReal/URM checkpoint is read.

## Ablations

`experiments/ablations/registry.py` defines every arm as a small override of the
main configuration: mechanism (RIV/AEM off, g/m zeroed or shuffled), condition
source (TC-CLIP, single Perceiver, layer/pooling swaps), training (learned null,
cfg-mem, window, scratch head), and the fair-conditioning controls.

The private experiment-driver scripts (job queues, per-seed orchestration,
Excel/CI summaries) are intentionally not part of this repository; the public
runner covers train / generate / eval for every registered arm.

```bash
python experiments/ablations/run.py status
python experiments/ablations/run.py generate --ablation riv_off --samples 15
python experiments/ablations/run.py eval     --ablation riv_off --samples 15
```

## License

MIT for this repository's code.  Upstream components keep their own licenses;
see `NOTICE.md`.  Data, weights and third-party assets are not covered.
