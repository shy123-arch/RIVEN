# Legacy E2 evaluator (not the paper evaluator)

These files implement the earlier E2 evaluator: a TMR encoder pair adapted on
EgoReReal TRAIN, with 256-D latent FID (PCA64) and 32-pool retrieval.

**The paper main table does not use this evaluator.**  As of 2026-11-13 the
paper reports the clean HML3D-init evaluator in `../clean/` (512-D FID), and
the two evaluator spaces are not comparable.  Keep this directory only for
reproducing older internal experiments.

Assets: `E2_best.pt` in `RIVEN_EVAL_DIR`, plus the official TMR sources and
statistics in `RIVEN_TMR_ASSETS`.
