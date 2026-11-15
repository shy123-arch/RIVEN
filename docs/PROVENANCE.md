# Provenance

Snapshot taken 2026-11-15 from the private URM workspace
(`/mnt/data/oss/Wilson/ck/umrm`), commit-free at the time of copying: the
workspace was not a Git repository, so the table below records the exact source
path and SHA-256 (first 12 hex) of every copied file.

A machine-readable copy of the table (full SHA-256 and the private-v4 record)
is `docs/SOURCE_SNAPSHOT.json`.  `python tools/check_repository.py` verifies
that no absolute server path or secret survives and that every registry flag
matches its script's argparse.

**Edit policy.**  Algorithmic code is copied verbatim.  The only changes are
(a) replacing hard-coded absolute paths with environment variables documented in
`configs/paths.env.example`, and (b) import paths that follow the new layout.
The `Δ lines` column counts added+removed lines relative to the source; files
with `0` are byte-identical.

## Core method

| Repo path | Source path | Source SHA-256 | Δ lines |
|---|---|---|---|
| `src/riven/model/h1_train.py` | `umrm_model/experiments/vimo_free_h1_20261101/h1_train.py` | `0980d4b02ad2` | 19 |
| `src/riven/model/h1_6q_train.py` | `umrm_model/experiments/vimo_free_h1_20261101/h1_6q_train.py` | `48ab92c0c1c8` | 8 |
| `src/riven/model/h1_6q_gen_condswap.py` | `umrm_model/experiments/cond_swap_v1572_20261104/h1_6q_gen_condswap.py` | `6b83568a72e3` | 7 |
| `src/riven/model/cond_adapters.py` | `umrm_model/experiments/vimo_free_h1_20261101/cond_adapters.py` | `c8f808a3e066` | 0 |
| `src/riven/model/ordered_adapter.py` | `umrm_model/experiments/vimo_free_h1_20261101/ordered_adapter.py` | `e09edcdfa383` | 0 |
| `src/riven/model/motion_windows.py` | `umrm_model/experiments/vimo_free_h1_20261101/motion_windows.py` | `17cbf9ee9174` | 0 |
| `src/riven/model/intent_patch.py` | `umrm_model/experiments/vimo_free_h1_20261101/intent_patch.py` | `4038bf31e017` | 0 |
| `src/riven/model/cfg_patch.py` | `umrm_model/experiments/vimo_free_h1_20261101/cfg_patch.py` | `9a1e4aff1d7d` | 0 |
| `src/riven/model/vq_train.py` | `umrm_model/experiments/vimo_free_h1_20261101/vq_train.py` | `814e8d0b1e6e` | 8 |
| `src/riven/model/finetune_hero.py` | `umrm_model/experiments/hero_finetune/finetune_hero.py` | `ca9a6880babd` | 13 |
| `src/riven/model/cliponly_data.py` | `umrm_model/experiments/cliponly_v1/data.py` | `76ccd15b0b7c` | 7 |

## Condition encoder / V->T

| Repo path | Source path | Source SHA-256 | Δ lines |
|---|---|---|---|
| `src/riven/sft/sft.py` | `UMRM-OPD/pipeline/sft.py` | `224b688255ad` | 0 |
| `src/riven/sft/common.py` | `UMRM-OPD/pipeline/common.py` | `357c68b36351` | 23 |
| `src/riven/sft/vt_io.py` | `UMRM-OPD/pipeline/vt_io.py` | `01e8727fe9f7` | 0 |
| `src/riven/features/cache_prompt_hidden.py` | `umrm_model/experiments/end2end_v1572_20261028/scripts/cache_prompt_hidden.py` | `5c0f955962b2` | 15 |
| `src/riven/features/cache_visual_hidden.py` | `umrm_model/experiments/end2end_v1572_20261028/scripts/cache_visual_hidden.py` | `f6b706aeaa85` | 13 |
| `src/riven/features/merge_prompt_hidden.py` | `umrm_model/experiments/end2end_v1572_20261028/scripts/merge_prompt_hidden.py` | `8cdad9be41d8` | 7 |
| `src/riven/features/build_tmr_motion_emb_v4.py` | `umrm_model/experiments/vimo_free_h1_20261101/build_tmr_motion_emb_v4.py` | `10057e2061d9` | 10 |
| `src/riven/features/build_base_textfeat.py` | `umrm_model/experiments/vimo_free_h1_20261101/build_base_textfeat.py` | `e64d1b37e304` | 8 |

## Evaluation

| Repo path | Source path | Source SHA-256 | Δ lines |
|---|---|---|---|
| `src/riven/eval/eval_r.py` | `umrm_model/experiments/egoreal_baselines_20261109/eval_r.py` | `f0545af54334` | 7 |
| `src/riven/eval/eval_fid.py` | `umrm_model/experiments/egoreal_baselines_20261109/eval_fid.py` | `ade956ab1f3f` | 7 |
| `src/riven/eval/evaluator_train.py` | `umrm_model/experiments/hero_fair_v1572_20261102/evaluator_train.py` | `effc6d950d60` | 14 |
| `src/riven/eval/tmr_adapter.py` | `umrm_model/experiments/generation_benchmark_v1/tmr_adapter.py` | `1651d2aaff95` | 0 |
| `src/riven/eval/setup_tmr.py` | `umrm_model/experiments/generation_benchmark_v1/setup_tmr.py` | `1452452d9f46` | 3 |

## Ablation harness

| Repo path | Source path | Source SHA-256 | Δ lines |
|---|---|---|---|
| `experiments/ablations/common.py` | `umrm_model/experiments/riven_ablations_20261109/common.py` | `53eed501639a` | 56 |
| `experiments/ablations/registry.py` | `umrm_model/experiments/riven_ablations_20261109/registry.py` | `02cd5a230d1e` | 0 |
| `experiments/ablations/prepare.py` | `umrm_model/experiments/riven_ablations_20261109/prepare.py` | `4318afb632db` | 19 |
| `experiments/ablations/run.py` | `umrm_model/experiments/riven_ablations_20261109/run.py` | `1b6593b8a5ac` | 9 |
| `experiments/ablations/release.py` | `umrm_model/experiments/egoreal_baselines_20261109/common.py` | `cea31d454888` | 8 |

## v4 orchestration (not published)

The v4 job queues, per-seed orchestrator, CI/standard-FID and TEST summary
scripts are kept in the authors' private workspace
(`riven_private/experiments_v4_20261115/`, copied from
`umrm_model/experiments/final_v4_20261111/`) and are **not** part of this
repository.  They are experiment drivers, not part of the method, and they
reference baseline runs that are not released here.  The public
`experiments/ablations/run.py` covers train / generate / eval for every
registered arm.

## HERO

- Upstream: `JackYu6/HERO_release`, commit `475774d` ("fix num_quantizers missing bug").
- Local modification: `gen.py` only, recorded in `patches/hero-gen.patch`.
  It selects the checkpoint named by `opt.which_epoch` and makes visualisation
  optional so batch generation does not fail on headless nodes.
- Apply with `git -C third_party/HERO apply patches/hero-gen.patch`.

## Path edits

Every environment variable is documented in `configs/paths.env.example`.
The original defaults pointed at the private workspace; the new defaults point
inside this repository (`data/egoreal`, `cache/`, `state/`), so a fresh clone
never silently reads the authors' server paths.  No computation, tensor shape,
loss, sampling step, split or evaluator rule was changed.

Two layout-only renames were also applied:

- `experiments/v4/*` now writes under `$RIVEN_STATE/ablations/` instead of
  `$RIVEN_STATE/egoreal_ablations_20261109/`, matching `experiments/ablations/`.
- Baseline main-table paths keep the name `egoreal_baselines_20261109/`; that
  harness is not part of this repository.

## Validation performed on the snapshot

Run on 2026-11-15, against the real v4 release and HERO checkout:

- `python -m compileall src experiments` — pass.
- `bash scripts/smoke.sh` — pass (registry 60 arms, SFT prompt module).
- `experiments/ablations/run.py prepare --release umrm_v2280_split_v4_20261111`
  — schema check pass, `ready: true`, only optional blank/shuffle controls
  reported missing.
- Import of `h1_train`, `h1_6q_train`, `h1_6q_gen_condswap`, `vq_train`,
  `cond_adapters`, `ordered_adapter`, `intent_patch`, `cfg_patch`,
  `motion_windows` in the HERO environment — pass.
- Import of `sft`, `common`, `vt_io` in the Qwen environment — pass.
- Import of `eval_r`, `eval_fid`, `evaluator_train` in the evaluator
  environment — pass.
- Every registry flag exists in the corresponding script's argparse
  (cross-check: 0 missing).

Not run here: training, generation and metric computation, because that
requires the GPU allocation and the external checkpoints.  The commands and
their prerequisites are in `docs/REPRODUCE.md`.

## Condition-feature mainline

Two v4 caches exist.  The earlier table used SFT epoch2 (selected by VAL text
NLL).  The final paper mainline (2026-11-13) pins SFT epoch1, selected by
downstream VAL FID, and reads the sibling cache
`cache/umrm_v2280_split_v4_20261111_sft_ep1/`.  Reproduce it with:

```bash
export RIVEN_SFT_ADAPTER=.../sft_v4/epoch1
export RIVEN_CACHE_SUFFIX=_sft_ep1
```
