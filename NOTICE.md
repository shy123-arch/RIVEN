# Third-party notices

This repository contains RIVEN-specific code plus glue adapted from several
research codebases.  No third-party source tree is vendored here except where
explicitly noted; install the upstream projects and point `configs/paths.env`
at them.

## HERO_release

- Source: https://github.com/JackYu6/HERO_release (commit `475774d`)
- License: MIT, Copyright (c) 2025 Chengjun Yu
- Use here: VQ/RVQ and masked/residual transformer definitions, motion utilities,
  and the standard FID evaluator path.
- Local change: `patches/hero-gen.patch` (generation checkpoint selection and
  optional rendering).  Apply it after checking out the pinned commit.

## HumanML3D

- Source: https://github.com/EricGuo5513/HumanML3D
- Use here: 263-D motion representation and the HumanML3D RVQ initialisation
  of the tokenizer.  HumanML3D data and derived weights are not redistributed;
  obtain them under their original terms.

## Qwen3-VL

- Model: `Qwen/Qwen3-VL-4B-Instruct` (default base of the condition encoder).
- Use here: frozen vision-language backbone plus a LoRA adapter trained for the
  plan -> exec task.  Model weights are not redistributed.

## Clean evaluator

`src/riven/eval/clean/modules.py` and `train.py` implement the standard
HumanML3D/KIT-style evaluator (movement Conv encoder + BiGRU text/motion
encoders, margin contrastive loss) adapted to EgoReReal TRAIN.  They are
derived from the official HumanML3D evaluator code and keep its original
terms; the adapted weights are released separately for evaluation.

## Official TMR / TEMOS assets

- `src/riven/eval/setup_tmr.py` fetches a pinned official TMR archive and
  verifies its checksums.  The archive and the distilled BERT text encoder keep
  their original licenses; they are distributed unmodified in the separate
  evaluation bundle and loaded from `RIVEN_TMR_ASSETS`.
- The bundle's `E2_best.pt` is fitted by this project and released for
  evaluation use with RIVEN.  It is not derived from the dataset's motion
  targets beyond TRAIN-only fitting.

## V->T pipeline (UMRM-OPD)

- `src/riven/sft/` is copied from the UMRM-OPD pipeline
  (https://github.com/shy123-arch/UMRM-OPD) and keeps its prompt/IO contract.
  See `docs/PROVENANCE.md` for the exact source files.
