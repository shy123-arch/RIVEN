"""Ablation registry for RIVEN.

Base configuration = the current paper mainline (`--riv-prefix direct`,
learned null, cfg-mem, RoleAdapter, full-length, 6q VQ + HML3D head init).
Every ablation is expressed as a small override of that base; the runner
renders the exact upstream CLI, so adding an ablation never means new code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from common import DEFAULT_INIT, DEFAULT_VQ

TRAIN_BASE: dict = {
    "--vq": str(DEFAULT_VQ),
    # v4 clean protocol: all ablations train their head from scratch.
    "--cond": "video",
    "--corpus": "a_gt3p_trainval",
    "--visual-name": "visual_trainval",
    "--adapter-type": "role",
    "--role-no-time": True,
    "--role-fields": "none",
    "--slots": 16,
    "--n-visual": 16,
    "--plan-slots": 4,
    "--g-layer": 36,
    "--v-layer": 27,
    "--p-layer": 18,
    "--cfg-mem": True,
    "--intent-mode": "cross",
    "--drop-mode": "learned_null",
    "--riv-prefix": "direct",
    "--cond-drop": 0.1,
    "--crop-mode": "full",
    "--window": 64,
    "--batch": 32,
    "--lr": 1e-4,
    # Match the unseen-split main_scratch run: NCE off (the TMR embedding
    # cache does not cover every new-split id).
    "--lam-nce": 0.0,
    "--eval-every": 250,
    "--save-every": 500,
}

GEN_BASE: dict = {
    "--adapter-type": "role",
    "--role-no-time": True,
    "--intent-mode": "cross",
    "--riv-prefix": "direct",
    "--cfg-mode": "intent",
    "--cfg-mem": True,
    "--cond-scale": 1.0,
    "--res-scale": 1.0,
    "--topk": 0.95,
    "--timesteps": 18,
    "--lengths": "gt",
    "--split": "val",
    "--corpus": "a_gt3p_trainval",
    "--visual-name": "visual_trainval",
    "--null-branch": "full",
}


@dataclass
class Ablation:
    name: str
    description: str
    kind: str                      # inference | train
    train: dict = field(default_factory=dict)
    gen: dict = field(default_factory=dict)

    def train_flags(self) -> dict:
        out = dict(TRAIN_BASE)
        for key, value in self.train.items():
            if value is None:
                out.pop(key, None)
            else:
                out[key] = value
        return out

    def gen_flags(self) -> dict:
        out = dict(GEN_BASE)
        for key, value in self.gen.items():
            if value is None:
                out.pop(key, None)
            else:
                out[key] = value
        return out


ABLATIONS: dict[str, Ablation] = {
    # ---- mechanism: inference branches on ONE trained checkpoint ----
    "main": Ablation("main", "RIVEN full (RIV + AEM, learned null, cfg-mem)", "inference"),
    "main_scratch": Ablation("main_scratch", "RIVEN full, action head trained from scratch on this split", "inference",
                              train={"--init-h1": None, "--init-head-only": None}),
    "base_qwen": Ablation("base_qwen", "Qwen base condition, no 10k PT and no reaction SFT", "train",
                          train={"--init-h1": None, "--init-head-only": None,
                                 "--corpus": "a_gt3p_base", "--visual-name": "visual_base"},
                          gen={"--corpus": "a_gt3p_base", "--visual-name": "visual_base"}),
    "pt_only": Ablation("pt_only", "Qwen 10k PT condition, no reaction SFT", "train",
                        train={"--init-h1": None, "--init-head-only": None,
                               "--corpus": "a_gt3p_pt", "--visual-name": "visual_pt"},
                        gen={"--corpus": "a_gt3p_pt", "--visual-name": "visual_pt"}),
    "sft_only": Ablation("sft_only", "SFT on base Qwen, no PT initialization", "train",
                         train={"--init-h1": None, "--init-head-only": None,
                                "--corpus": "a_gt3p_sftonly", "--visual-name": "visual_sftonly"},
                         gen={"--corpus": "a_gt3p_sftonly", "--visual-name": "visual_sftonly"}),
    # NCE-on PT/SFT ablation. The bank is TRAIN-only and pointed to by the
    # trainer's fixed /tmp path.
    "nce_main_scratch": Ablation("nce_main_scratch", "PT+SFT with TMR-NCE 0.1", "train",
                                  train={"--init-h1": None, "--init-head-only": None,
                                         "--lam-nce": 0.1}),
    "nce_base_qwen": Ablation("nce_base_qwen", "Base Qwen with TMR-NCE 0.1", "train",
                               train={"--init-h1": None, "--init-head-only": None,
                                      "--corpus": "a_gt3p_base", "--visual-name": "visual_base",
                                      "--lam-nce": 0.1},
                               gen={"--corpus": "a_gt3p_base", "--visual-name": "visual_base"}),
    "nce_pt_only": Ablation("nce_pt_only", "PT-only with TMR-NCE 0.1", "train",
                             train={"--init-h1": None, "--init-head-only": None,
                                    "--corpus": "a_gt3p_pt", "--visual-name": "visual_pt",
                                    "--lam-nce": 0.1},
                             gen={"--corpus": "a_gt3p_pt", "--visual-name": "visual_pt"}),
    "nce_sft_only": Ablation("nce_sft_only", "SFT-only with TMR-NCE 0.1", "train",
                              train={"--init-h1": None, "--init-head-only": None,
                                     "--corpus": "a_gt3p_sftonly", "--visual-name": "visual_sftonly",
                                     "--lam-nce": 0.1},
                              gen={"--corpus": "a_gt3p_sftonly", "--visual-name": "visual_sftonly"}),
    # Fixed SFT epoch1 features, NCE 0.1. These are the v4-style arms.
    "nce_ep1_main_scratch": Ablation("nce_ep1_main_scratch", "PT+SFT ep1 with TMR-NCE 0.1", "train",
                                      train={"--init-h1": None, "--init-head-only": None,
                                             "--corpus": "a_gt3p_trainval_ep1",
                                             "--visual-name": "visual_trainval_ep1",
                                             "--lam-nce": 0.1},
                                      gen={"--corpus": "a_gt3p_trainval_ep1",
                                           "--visual-name": "visual_trainval_ep1"}),
    "nce_ep1_sft_only": Ablation("nce_ep1_sft_only", "SFT-only ep1 with TMR-NCE 0.1", "train",
                                  train={"--init-h1": None, "--init-head-only": None,
                                         "--corpus": "a_gt3p_sftonly_ep1",
                                         "--visual-name": "visual_sftonly_ep1",
                                         "--lam-nce": 0.1},
                                  gen={"--corpus": "a_gt3p_sftonly_ep1",
                                       "--visual-name": "visual_sftonly_ep1"}),
    # Same fixed SFT epoch1 features, but NCE off. This isolates the epoch
    # change from the NCE change in the PT/SFT 2x2.
    "ep1_main_scratch": Ablation("ep1_main_scratch", "PT+SFT ep1, no NCE", "train",
                                  train={"--init-h1": None, "--init-head-only": None,
                                         "--corpus": "a_gt3p_trainval_ep1",
                                         "--visual-name": "visual_trainval_ep1",
                                         "--lam-nce": 0.0},
                                  gen={"--corpus": "a_gt3p_trainval_ep1",
                                       "--visual-name": "visual_trainval_ep1"}),
    "ep1_sft_only": Ablation("ep1_sft_only", "SFT-only ep1, no NCE", "train",
                              train={"--init-h1": None, "--init-head-only": None,
                                     "--corpus": "a_gt3p_sftonly_ep1",
                                     "--visual-name": "visual_sftonly_ep1",
                                     "--lam-nce": 0.0},
                              gen={"--corpus": "a_gt3p_sftonly_ep1",
                                   "--visual-name": "visual_sftonly_ep1"}),
    "riv_off": Ablation("riv_off", "RIV replaced by learned null (r0)", "inference",
                        gen={"--null-branch": "r0"}),
    "aem_off": Ablation("aem_off", "AEM replaced by learned null (e0)", "inference",
                        gen={"--null-branch": "e0"}),
    "riv_aem_off": Ablation("riv_aem_off", "both RIV and AEM nulled", "inference",
                            gen={"--null-branch": "both"}),
    "e11_cfg15": Ablation("e11_cfg15", "inference-only CFG at scale 1.5", "inference",
                          gen={"--cond-scale": "1.5"}),
    "e11_cfg20": Ablation("e11_cfg20", "inference-only CFG at scale 2.0", "inference",
                          gen={"--cond-scale": "2.0"}),
    "g_zero": Ablation("g_zero", "intent vector zeroed at inference", "inference",
                       gen={"--ablate-g": "zero"}),
    "g_mean": Ablation("g_mean", "intent vector replaced by batch mean", "inference",
                       gen={"--ablate-g": "mean"}),
    "m_zero": Ablation("m_zero", "evidence memory zeroed at inference", "inference",
                       gen={"--ablate-m": "zero"}),
    "m_shuffle": Ablation("m_shuffle", "evidence memory shuffled across samples", "inference",
                          gen={"--ablate-m": "shuffle"}),

    # ---- condition side (separate training) ----
    "tcclip": Ablation("tcclip", "TC-CLIP (g,m) instead of Qwen condition", "train",
                       train={"--tc-train": "__TC_TRAIN__", "--tc-val": "__TC_VAL__"},
                       gen={"--cond-source": "tcclip", "--tc-val": "__TC_VAL__"}),
    "perceiver": Ablation("perceiver", "single Perceiver adapter (no RoleAdapter split)", "train",
                          train={"--adapter-type": "perceiver", "--role-no-time": None},
                          gen={"--adapter-type": "perceiver", "--role-no-time": None,
                               "--intent-mode": None, "--riv-prefix": None}),
    "role_time": Ablation("role_time", "RoleAdapter with time tokens (main has none)", "train",
                          train={"--role-no-time": None},
                          gen={"--role-no-time": None}),
    "blank_video": Ablation("blank_video", "zeroed visual+text condition (control)", "train",
                            train={"--corpus": "blank_gt3p_trainval",
                                   "--visual-name": "visual_blank_trainval"},
                            gen={"--corpus": "blank_gt3p_trainval",
                                 "--visual-name": "visual_blank_trainval"}),
    "shuffle_video": Ablation("shuffle_video", "shuffled visual+text condition (control)", "train",
                              train={"--corpus": "shuffle_gt3p_trainval",
                                     "--visual-name": "visual_shuffle_trainval"},
                              gen={"--corpus": "shuffle_gt3p_trainval",
                                   "--visual-name": "visual_shuffle_trainval"}),

    # ---- fair-conditioning controls (P0-1, 2026-11-14) ----
    # Same Qwen SFT features / tokenizer / head-init as `main_scratch`; the
    # adapter and the train-time null branch are the only differences.
    "b1_prefix": Ablation("b1_prefix", "Qwen Perceiver, prefix only (AEM nulled)", "train",
                          train={"--adapter-type": "perceiver", "--use-layers": "36",
                                 "--drop-mode": "learned_null", "--null-branch": "e0",
                                 "--intent-mode": "cross", "--riv-prefix": "direct"},
                          gen={"--adapter-type": "perceiver", "--use-layers": "36",
                               "--null-branch": "e0", "--intent-mode": "cross",
                               "--riv-prefix": "direct"}),
    "b2_memory": Ablation("b2_memory", "Qwen Perceiver, memory only (RIV nulled)", "train",
                          train={"--adapter-type": "perceiver", "--use-layers": "36",
                                 "--drop-mode": "learned_null", "--null-branch": "r0",
                                 "--intent-mode": "cross", "--riv-prefix": "direct"},
                          gen={"--adapter-type": "perceiver", "--use-layers": "36",
                               "--null-branch": "r0", "--intent-mode": "cross",
                               "--riv-prefix": "direct"}),
    "b3_flat": Ablation("b3_flat", "Qwen Perceiver, both channels, no role/layer split", "train",
                        train={"--adapter-type": "perceiver", "--use-layers": "36",
                               "--drop-mode": "learned_null", "--null-branch": "full"},
                        gen={"--adapter-type": "perceiver", "--use-layers": "36",
                             "--null-branch": "full"}),
    "c1_tcclip": Ablation("c1_tcclip", "TC-CLIP (g,m) through the RIVEN head", "train",
                          train={"--tc-train": "__TC_TRAIN__", "--tc-val": "__TC_VAL__"},
                          gen={"--cond-source": "tcclip", "--tc-val": "__TC_VAL__"}),

    # ---- E2: condition-dropout rate / single-channel training (2026-11-14) ----
    "e2_drop25": Ablation("e2_drop25", "FULL architecture with per-branch dropout 0.25", "train",
                          train={"--cond-drop": 0.25}),
    "e2_aem_only": Ablation("e2_aem_only", "RoleAdapter, RIV forced null, AEM only", "train",
                            train={"--null-branch": "r0"}, gen={"--null-branch": "r0"}),
    "e2_riv_only": Ablation("e2_riv_only", "RoleAdapter, AEM forced null, RIV only", "train",
                            train={"--null-branch": "e0"}, gen={"--null-branch": "e0"}),
    # Inference-time both-null branch on the two single-channel models (E2).
    "e2_aem_only_both": Ablation("e2_aem_only_both", "AEM-only model, both branches nulled", "inference",
                                 gen={"--null-branch": "both"}),
    "e2_riv_only_both": Ablation("e2_riv_only_both", "RIV-only model, both branches nulled", "inference",
                                 gen={"--null-branch": "both"}),

    # ---- E3: layer / pooling / token-count ablations (2026-11-14) ----
    "e3_riv18": Ablation("e3_riv18", "RIV pooling layer 36 -> 18", "train",
                         train={"--g-layer": 18}, gen={"--g-layer": 18}),
    "e3_riv27": Ablation("e3_riv27", "RIV pooling layer 36 -> 27", "train",
                         train={"--g-layer": 27}, gen={"--g-layer": 27}),
    "e3_aem18": Ablation("e3_aem18", "AEM visual layer 27 -> 18", "train",
                         train={"--v-layer": 18}, gen={"--v-layer": 18}),
    "e3_aem36": Ablation("e3_aem36", "AEM visual layer 27 -> 36", "train",
                         train={"--v-layer": 36}, gen={"--v-layer": 36}),
    "e3_pool_last": Ablation("e3_pool_last", "RIV mean pooling -> last prompt token", "train",
                             train={"--riv-pool": "last"}, gen={"--riv-pool": "last"}),
    "e3_pool_attn": Ablation("e3_pool_attn", "RIV mean pooling -> attention pooling", "train",
                             train={"--riv-pool": "attn"}, gen={"--riv-pool": "attn"}),
    "e3_aem36all": Ablation("e3_aem36all", "AEM visual+plan layers -> 36", "train",
                            train={"--v-layer": 36, "--p-layer": 36},
                            gen={"--v-layer": 36, "--p-layer": 36}),
    "e3_same36": Ablation("e3_same36", "RIV and AEM all read layer 36", "train",
                          train={"--g-layer": 36, "--v-layer": 36, "--p-layer": 36},
                          gen={"--g-layer": 36, "--v-layer": 36, "--p-layer": 36}),
    "e3_swap_all": Ablation("e3_swap_all", "RIV@18, AEM@36 (full layer swap)", "train",
                            train={"--g-layer": 18, "--v-layer": 36, "--p-layer": 36},
                            gen={"--g-layer": 18, "--v-layer": 36, "--p-layer": 36}),
    "e3_no_plan": Ablation("e3_no_plan", "N_q=0: no instruction-side plan queries", "train",
                           train={"--role-mode": "visual"}, gen={"--role-mode": "visual"}),
    "e3_plan8": Ablation("e3_plan8", "N_q=8 plan queries (main: 4)", "train",
                         train={"--plan-slots": 8}, gen={"--plan-slots": 8}),
    "e3_vis8": Ablation("e3_vis8", "N_v=8 visual tokens (main: 16)", "train",
                        train={"--n-visual": 8}, gen={"--n-visual": 8}),

    # ---- W1 (review 2026-11-15): capacity- and source-matched interface controls ----
    # `e3_same36` (all streams read layer 36) and `e3_swap_all` (RIV@18,
    # AEM@36) already hold the adapter/parameter/token budget fixed.  The two
    # arms below close the remaining holes: swapping which source feeds the
    # prefix vs. the memory, and a single-tower control at matched capacity.
    "w1_swap_inject": Ablation(
        "w1_swap_inject", "Role-swap: prefix from visual, memory from prompt", "train",
        train={"--role-swap": True}, gen={"--role-swap": True}),
    "w1_cap_flat": Ablation(
        "w1_cap_flat", "Single-tower Perceiver-4 (20 tokens, cap-matched to RoleAdapter)", "train",
        train={"--adapter-type": "perceiver4", "--slots": 20, "--use-layers": "36"},
        gen={"--adapter-type": "perceiver4", "--slots": 20, "--use-layers": "36"}),

    # ---- W2 (review 2026-11-15): RIVEN under the HERO/NOUS tokenizer ----
    # Identical to `main_scratch`; the CLI passes the HERO RVQ checkpoint via
    # --vq at train/generate time, so RIVEN and HERO/NOUS share a tokenizer.
    "w2_hero_vq": Ablation(
        "w2_hero_vq", "RIVEN head trained/decoded with the HERO/NOUS RVQ", "train",
        train={"--init-h1": None, "--init-head-only": None},
        gen={}),

    # ---- W3/W4 merged design (review round 2026-11-15) ----
    # P0: equal-capacity flat dual path and inverse prefix/memory assignment.
    "w3_flat6m": Ablation(
        "w3_flat6m", "21 tokens, one shared mixer, token0 prefix (6.05M)", "train",
        train={"--adapter-type": "flat6m"}, gen={"--adapter-type": "flat6m"}),
    "w3_roleswap_inv": Ablation(
        "w3_roleswap_inv", "Inverse role swap: pooled AEM prefix, expanded RIV memory", "train",
        train={"--role-inverse": True}, gen={"--role-inverse": True}),
    # P1: layer-content 2x2 crosses (RIV 27/36 x AEM 18/27/36).  The
    # (36,27),(27,27),(36,18),(36,36) cells already exist as e3_* arms.
    "w4_riv27_aem18": Ablation(
        "w4_riv27_aem18", "RIV@27, AEM visual/plan@18", "train",
        train={"--g-layer": 27, "--v-layer": 18, "--p-layer": 18},
        gen={"--g-layer": 27, "--v-layer": 18, "--p-layer": 18}),
    "w4_riv27_aem36": Ablation(
        "w4_riv27_aem36", "RIV@27, AEM visual/plan@36", "train",
        train={"--g-layer": 27, "--v-layer": 36, "--p-layer": 36},
        gen={"--g-layer": 27, "--v-layer": 36, "--p-layer": 36}),

    # ---- W6 (review 2026-11-15): standard 10% CFG training ----
    # `joint` dropout + cfg-mem = one full-condition keep mask shared by g and
    # memory at p=0.1.  The inference uncond branch (force_mask) then matches
    # the training uncond distribution, unlike the learned-null main model
    # whose uncond branch is a learned token.
    "w6_cfg10": Ablation(
        "w6_cfg10", "Standard 10% full-condition dropout, CFG-ready", "train",
        train={"--drop-mode": "joint", "--cond-drop": 0.1,
               "--init-h1": None, "--init-head-only": None},
        gen={}),

    # ---- training objective / conditioning ----
    "no_cfg_mem": Ablation("no_cfg_mem", "CFG drops g only (HERO original), not m", "train",
                           train={"--cfg-mem": None}, gen={"--cfg-mem": None}),
    "drop_joint": Ablation("drop_joint", "joint g+m dropout instead of learned null", "train",
                           train={"--drop-mode": "joint"}),
    "window64": Ablation("window64", "64-frame random window instead of full-length", "train",
                         train={"--crop-mode": "random", "--window": 64}),

    # ---- initialization / tokenizer ----
    "scratch_head": Ablation("scratch_head", "action head trained from scratch (no HML3D init)", "train",
                             train={"--init-h1": None, "--init-head-only": None}),
    # Tokenizer capacity (6q vs 1q) is NOT wired here: the 1q route uses the
    # separate h1_train.py/h1_gen.py code path, not h1_6q_*. See README.  
}

INFERENCE = [n for n, a in ABLATIONS.items() if a.kind == "inference"]
TRAIN = [n for n, a in ABLATIONS.items() if a.kind == "train"]
