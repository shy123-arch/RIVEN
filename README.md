# RIVEN

**Reaction Intent and Evidence-conditioned Network for egocentric reaction motion generation.**

Given a first-person video that ends right before the wearer reacts, RIVEN predicts
the next 1-4 seconds of full-body motion.  This repository is the code release of
the paper's main table (`main_scratch` on the unseen-source split): condition
encoding, tokenizer, action head, generation, checkpoint selection, the 60-arm
ablation registry, and the clean evaluator used for every reported number.

Data, model weights and third-party assets are **not** included.
`docs/REPRODUCE.md` lists every external asset and where to put it.

## Pipeline

```
pre-reaction video
      │
      ▼
Qwen3-VL-4B + LoRA (plan -> exec SFT, epoch2)      condition encoder
      │  prompt hidden [layer 36]        visual hidden [layers 9/18/27/36]
      ▼                                        │
  RIV  (reaction intent, g)  ───────┐          │
  AEM  (evidence memory, m)  ───────┤          │
      │  RoleAdapter, direct prefix │          │
      ▼                             ▼          ▼
6-quantizer RVQ tokenizer  ──►  masked transformer (q1) + residual transformer (q2..q6)
      │                        200 epochs -> VAL selection (R@10 within 0.5pp, min FID)
      ▼
generated 22-joint motion (GT length, CFG on the RIV/AEM branches)
      │
      ▼
clean HML3D-init evaluator (3 seeds): 512-D FID + R@1/2/3/10
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
src/riven/sft/         Qwen plan->exec SFT, video-tensor preparation
src/riven/features/    Qwen prompt/visual hidden-state caches
src/riven/eval/clean/  paper evaluator: 512-D FID + R@1/2/3/10 (3 seeds)
src/riven/eval/legacy_e2/  older E2 evaluator; NOT used by the paper
experiments/ablations/ registry.py (all ablations) + run.py / prepare.py
experiments/main_table/ per-seed 200-epoch chain + VAL checkpoint selection
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
$EDITOR configs/paths.env               # data, state, cache, HERO, Qwen, evaluator
set -a; . configs/paths.env; set +a

# Paper condition encoder: SFT epoch2 (VAL NLL) from the 10k PT adapter.
export RIVEN_SFT_INIT_ADAPTER=.../s10k_pt/epoch1
export RIVEN_SFT_ADAPTER=.../sft_unseen_full/epoch2
export RIVEN_CLIP_ROOT=/data/egoreal/clips     # or RIVEN_CLIP_MAP=clip_map.json

"$RIVEN_PY_QWEN" src/riven/sft/prep_video_cache.py --workers 8
bash scripts/build_features.sh          # Qwen prompt/visual hidden states

SEED=20261101 GPU=0 bash scripts/run_main.sh   # VQ -> 200ep head -> select -> TEST
```

Full instructions, data contract and evaluation protocol: **`docs/REPRODUCE.md`**.

## What is fixed

- Data: `umrm_v2280_unseen_20261112`, 1,854 / 213 / 213, source-disjoint.
- Condition: RIVEN `main_scratch` — RIV+AEM, learned null, cfg-mem,
  RoleAdapter with `riv_prefix=direct`, no time tokens.
- Tokenizer: 6q RVQ, 3000 steps; only the first quantizer layer is initialised
  from HumanML3D (no ViMo weights).
- Head: trained from scratch for 11,500 steps (200 epochs), then the VAL
  checkpoint is selected among 9k / 10.5k / last (R@10 within 0.5pp of the
  best, lowest FID).
- Generation: GT target length, 15 samples, no rerank.
- Evaluation: clean HML3D-init evaluator, 3 evaluator seeds, 512-D FID and
  R@1/2/3/10 (t2m, source-excluded 32-pool).  The legacy E2 evaluator in
  `src/riven/eval/legacy_e2/` produces non-comparable numbers.

## Ablations

`experiments/ablations/registry.py` defines every arm as a small override of the
main configuration: mechanism (RIV/AEM off, g/m zeroed or shuffled), condition
source (TC-CLIP, single Perceiver, layer/pooling swaps), training (learned null,
cfg-mem, window, scratch head), and the fair-conditioning controls.

```bash
python experiments/ablations/run.py status
python experiments/ablations/run.py generate --ablation riv_off --samples 15
python experiments/ablations/run.py eval     --ablation riv_off --samples 15
```

## License

MIT for this repository's code.  Upstream components keep their own licenses;
see `NOTICE.md`.  Data, weights and third-party assets are not covered.
