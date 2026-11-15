# Method

RIVEN predicts a full-body reaction from a first-person clip that ends just
before the wearer acts.  The condition comes from a frozen Qwen3-VL-4B encoder
with a LoRA adapter trained for `plan -> exec`; the generator is a discrete
motion model in the HERO/MoMask family.  RIVEN's contribution is the structure
of the condition: two channels with different roles, plus a training/inference
scheme that keeps them separable.

## Condition encoder

`src/riven/sft/sft.py` fine-tunes Qwen3-VL-4B with LoRA (r=16 on LLM linear
layers, vision tower frozen), lr 1e-4, 3 epochs, accumulation 8, cosine schedule
with 3% warm-up.  The prompt is the `plan_exec` contract in
`src/riven/sft/common.py`: the model writes a structured plan line and a
third-person motion caption.  The paper's mainline pins the epoch selected by
downstream VAL FID, which is not necessarily the lowest teacher-forced NLL.

Two hidden-state caches are built from the adapter:

- **Prompt cache** (`cache_prompt_hidden.py`): the last 16 prompt-token hidden
  states at layers 9/18/27/36.
- **Visual cache** (`cache_visual_hidden.py`): the video tokens split into 16
  contiguous segments and mean-pooled at the same four layers.

## RIV and AEM

`src/riven/model/cond_adapters.py` defines `RoleAdapter`.  It reads the prompt
and visual caches and produces two things:

- **RIV** (`g`, reaction intent): a coarse, single-vector commitment about what
  the wearer is about to do.  Pooled from prompt hidden states at layer 36.
- **AEM** (`m`, addressable evidence memory): a 16-slot key/value memory built
  from visual tokens (layer 27) and plan queries (4 slots, layer 18).  It is
  addressed by the generator, not compressed into a prefix.

The mainline uses `--riv-prefix direct`: the prefix condition is `c = r`, and
AEM is only KV memory.  Role/time tokens are disabled (`--role-no-time`,
`--role-fields none`), keeping the paper's "no extra timestamp supervision"
setting.

## Tokenizer

`src/riven/model/vq_train.py` trains a residual VQ with 6 quantizers,
512 codes per quantizer, 512-D code vectors, on HumanML3D-263 features.
The first quantizer layer is initialised from the HumanML3D RVQ checkpoint;
the remaining five quantizer layers start from random init (the default,
`--all-codebooks` off) and the whole tokenizer is trained on the release.
3000 steps, batch 64, lr 2e-4, random windows up to 208 frames.  No ViMo
checkpoint or statistic is used.

## Generator

`src/riven/model/h1_6q_train.py` trains both stages jointly:

- `MaskTransformer`: predicts quantizer 1; 6 layers, 384 latent, 1024 FFN,
  6 heads, dropout 0.2; condition dimension 512.
- `ResidualTransformer`: predicts quantizers 2..6 from the accumulated codebook
  embeddings, the prefix `g` and the memory `m`.

Main configuration: full-length crop, batch 32, lr 1e-4, 5100 steps
(88 epochs at 35 iter/epoch), head trained from scratch.

Conditioning is made separable during training:

- `--drop-mode learned_null` replaces `r` and/or `e` with learnable null tokens
  before refinement, so the model sees full / RIV-null / AEM-null / both-null
  branches.
- `--cfg-mem` makes the CFG unconditional branch drop the memory as well, which
  is what the `cfg_mode=intent` sampler expects.
- `--cond-drop 0.1` is the per-branch dropout rate.

`intent_patch.py` implements the null branches and the dropout variants;
`cfg_patch.py` changes the memory handling in HERO's transformer.

## Generation

`src/riven/model/h1_6q_gen_condswap.py` samples:

- 18 denoising steps, top-k 0.95;
- `cfg_mode=intent`, cond scale 1.0, residual scale 1.0 (main);
- `--null-branch full/r0/e0/both` selects which learned null replaces the real
  channel, which is exactly the RIV-off / AEM-off mechanism ablation;
- `--ablate-g zero|mean` and `--ablate-m zero|shuffle` are inference-time
  information controls;
- generation length is the GT target length.

## Evaluation

`src/riven/eval/eval_r.py` computes per-sample R@1/3/10 with 32-candidate pools
that exclude the query's source, 20 pools per query, seed 0, text-to-motion
direction.  `eval_fid.py` computes FID in the E2 256-D latent with a PCA basis
fitted on TRAIN only (64 components by default).  Both load the frozen
`E2_best.pt`; `evaluator_train.py` and `tmr_adapter.py` are the reference
evaluator implementation and the checksum-verified official TMR wrapper.
