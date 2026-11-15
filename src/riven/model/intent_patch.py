#!/usr/bin/env python3
"""Intent / evidence usage variants for the HERO masked+residual head.

HERO refines the global condition g exactly once by cross-attending to the
memory m, then uses it as a prefix token for the whole generation.  Under the
"intent (g) / evidence memory (m)" reading we expose three usage modes:

  cross    : HERO default.  g' = CrossAttn(g, m), fixed across denoising steps.
  direct   : intent never reads the evidence as K/V.  g is used as-is.
  refresh  : intent is re-written from the *current* motion state at every
             denoising call: g' = CrossAttn(g + MLP(pool(x)), m).
             The MLP is zero-initialised, so training starts at `cross`.

Also supports per-slot dropout of the evidence memory during training so the
head cannot collapse onto a single source slot (our m-shuffle ~ m-zero result).

Usage (trainer and generator):
    import intent_patch; intent_patch.apply(TMC)
    ... build models ...
    intent_patch.configure(trans, mode="refresh", mem_drop=0.0)
    intent_patch.configure(res, mode="refresh", mem_drop=0.0)
"""

import torch
import torch.nn as nn


class RefreshMLP(nn.Module):
    """Small state-summary -> intent residual.  Final layer is zero-init.

    Motion hidden states live in latent_dim (384) while the condition vector g
    lives in clip_dim (512), so the MLP maps 384 -> 512.
    """

    def __init__(self, d_in=384, d_out=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s):
        return self.net(s)


def configure(module, mode="cross", mem_drop=0.0):
    module.intent_mode = mode
    module.mem_drop = float(mem_drop)
    if mode == "refresh" and not hasattr(module, "refresh_mlp"):
        d_in = getattr(module, "latent_dim", 384)
        d_out = getattr(module, "clip_dim", 512)
        module.refresh_mlp = RefreshMLP(d_in, d_out).to(next(module.parameters()).device)
    return module


def _masked_state(x, padding_mask):
    """x: (T, b, d) ; padding_mask: (b, T) with True = pad."""
    w = (~padding_mask).transpose(0, 1).float().unsqueeze(-1)  # (T,b,1)
    return (x * w).sum(0) / w.sum(0).clamp(min=1.0)


def _drop_slots(memory, p):
    b, k, d = memory.shape
    keep = torch.bernoulli(torch.full((b, k, 1), 1.0 - p, device=memory.device, dtype=memory.dtype))
    return memory * keep


def _refine(self, cond, memory, x_pre, padding_mask):
    mode = getattr(self, "intent_mode", "cross")
    if mode == "direct":
        return cond
    prefix = getattr(self, "riv_prefix", "cross")
    if prefix == "direct":
        # RIV-only prefix: the decoder sees r directly; AEM is KV memory only.
        return cond
    c = cond.unsqueeze(1)
    if mode == "refresh":
        s = _masked_state(x_pre, padding_mask)
        c = c + self.refresh_mlp(s).unsqueeze(1)
    c_cross = self.global_local_crossatt_block(c, memory)
    if prefix == "residual":
        # Slight RIV upgrade: keep a direct path from r to the decoder while
        # still letting r query the evidence memory.
        c = c + c_cross
    else:
        c = c_cross
    return c.squeeze(1)


def configure_learned_null(module):
    """Add learnable null query r0 and null memory token e0.

    The nulls are optimized with the adapter/head; at inference they define the
    four branch ablations (r,e), (r0,e), (r,e0), (r0,e0) via ``null_branch``.
    Must be called before building the optimizer (training) and before
    ``load_state_dict`` (generation) so the keys exist.
    """
    d = int(getattr(module, "clip_dim", 512))
    if not hasattr(module, "null_r"):
        ref = next(module.parameters())
        module.null_r = nn.Parameter(torch.zeros(d, device=ref.device, dtype=ref.dtype))
        module.null_e = nn.Parameter(torch.zeros(d, device=ref.device, dtype=ref.dtype))
    module.learned_null = True
    if not hasattr(module, "null_branch"):
        module.null_branch = "full"
    if not hasattr(module, "force_null_branch"):
        module.force_null_branch = None
    return module


def set_null_branch(module, branch):
    module.null_branch = branch
    # A forced branch stays null in training too; "full" leaves both active.
    module.force_null_branch = branch if branch != "full" else None
    return module


def _learned_null_keep(self, cond):
    """Per-sample keep masks for RIV (kg) and AEM (km).

    Channels forced to null by ``set_null_branch`` stay null in training as
    well, so single-channel controls (B1/B2) cannot route information through
    the removed branch.  The remaining active channel keeps the same
    ``cond_drop_prob`` regularisation as the full model.
    """
    b = cond.shape[0]
    p = float(getattr(self, "cond_drop_prob", 0.0))
    forced = getattr(self, "force_null_branch", None)
    one = torch.ones(b, 1, device=cond.device, dtype=cond.dtype)
    zero = torch.zeros(b, 1, device=cond.device, dtype=cond.dtype)

    def keep():
        if self.training and p > 0:
            return 1.0 - torch.bernoulli(
                torch.full((b, 1), p, device=cond.device, dtype=cond.dtype))
        return one

    kg = zero if forced in ("r0", "both") else keep()
    km = zero if forced in ("e0", "both") else keep()
    return kg, km


def _replace_nulls(self, cond, memory, kg, km):
    cond_eff = kg * cond + (1.0 - kg) * self.null_r[None, :]
    memory_eff = km[:, :, None] * memory + (1.0 - km[:, :, None]) * self.null_e[None, None, :]
    return cond_eff, memory_eff


def _mask_forward_learned_null(self, motion_ids, cond, padding_mask, force_mask=False, memory=None,
                               zero_m=False, keep_memory=False):
    """Learned-null forward: replace RIV/AEM *before* _refine.

    ``force_mask`` (CFG uncond) removes both branches.  ``zero_m`` is kept for
    compatibility but the null memory is the clean replacement.
    """
    kg, km = _learned_null_keep(self, cond)
    if force_mask:
        kg = torch.zeros_like(kg); km = torch.zeros_like(km)
    cond, memory = _replace_nulls(self, cond, memory, kg, km)
    if zero_m:
        memory = torch.zeros_like(memory)
    x = self.input_process(self.token_emb(motion_ids))
    cond = _refine(self, cond, memory, x, padding_mask)
    cond = self.cond_emb(cond).unsqueeze(0)
    memory = self.cond_emb(memory).permute((1, 0, 2))
    x = self.position_enc(x)
    xseq = torch.cat([cond, x], dim=0)
    padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)
    output = self.seqTransDecoder(tgt=xseq, memory=memory,
                                  tgt_key_padding_mask=padding_mask)[1:]
    return self.output_process(output)


def _res_forward_learned_null(self, motion_codes, qids, cond, padding_mask, force_mask=False, memory=None,
                              zero_m=False, keep_memory=False):
    kg, km = _learned_null_keep(self, cond)
    if force_mask:
        kg = torch.zeros_like(kg); km = torch.zeros_like(km)
    cond, memory = _replace_nulls(self, cond, memory, kg, km)
    if zero_m:
        memory = torch.zeros_like(memory)
    x = self.input_process(motion_codes)
    cond = _refine(self, cond, memory, x, padding_mask)
    q_onehot = self.encode_quant(qids).float().to(x.device)
    q_emb = self.quant_emb(q_onehot).unsqueeze(0)
    cond = self.cond_emb(cond).unsqueeze(0)
    memory = self.cond_emb(memory).permute((1, 0, 2))
    x = self.position_enc(x)
    xseq = torch.cat([cond, q_emb, x], dim=0)
    padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:2]), padding_mask], dim=1)
    output = self.seqTransDecoder(tgt=xseq, memory=memory,
                                  tgt_key_padding_mask=padding_mask)[2:]
    return self.output_process(output)


def patch_learned_null(TMC):
    """Install learned-null forward paths (see configure_learned_null)."""
    TMC.MaskTransformer.trans_forward = _mask_forward_learned_null
    TMC.ResidualTransformer.trans_forward = _res_forward_learned_null


def _mask_forward(self, motion_ids, cond, padding_mask, force_mask=False, memory=None,
                  zero_m=False, keep_memory=False):
    p = getattr(self, "mem_drop", 0.0)
    if self.training and p > 0:
        memory = _drop_slots(memory, p)
    x = self.input_process(self.token_emb(motion_ids))  # (T,b,d)
    cond = _refine(self, cond, memory, x, padding_mask)
    cond = self.mask_cond(cond, force_mask=force_mask)
    if zero_m:
        memory = torch.zeros_like(memory)
    if getattr(self, "_keep", None) is not None and not keep_memory:  # cfg_patch memory dropout
        memory = memory * self._keep[:, :, None]
    cond = self.cond_emb(cond).unsqueeze(0)
    memory = self.cond_emb(memory).permute((1, 0, 2))
    x = self.position_enc(x)
    xseq = torch.cat([cond, x], dim=0)
    padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)
    output = self.seqTransDecoder(tgt=xseq, memory=memory,
                                  tgt_key_padding_mask=padding_mask)[1:]
    return self.output_process(output)


def _res_forward(self, motion_codes, qids, cond, padding_mask, force_mask=False, memory=None,
                 zero_m=False, keep_memory=False):
    p = getattr(self, "mem_drop", 0.0)
    if self.training and p > 0:
        memory = _drop_slots(memory, p)
    x = self.input_process(motion_codes)  # (T,b,d)
    cond = _refine(self, cond, memory, x, padding_mask)
    cond = self.mask_cond(cond, force_mask=force_mask)
    if zero_m:
        memory = torch.zeros_like(memory)
    if getattr(self, "_keep", None) is not None and not keep_memory:  # cfg_patch memory dropout
        memory = memory * self._keep[:, :, None]
    q_onehot = self.encode_quant(qids).float().to(x.device)
    q_emb = self.quant_emb(q_onehot).unsqueeze(0)
    cond = self.cond_emb(cond).unsqueeze(0)
    memory = self.cond_emb(memory).permute((1, 0, 2))
    x = self.position_enc(x)
    xseq = torch.cat([cond, q_emb, x], dim=0)
    padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:2]), padding_mask], dim=1)
    output = self.seqTransDecoder(tgt=xseq, memory=memory,
                                  tgt_key_padding_mask=padding_mask)[2:]
    return self.output_process(output)


def patch_independent_dropout(TMC):
    """Drop g and m independently in training.

    With cfg_mem's joint dropout, f(0, m) and f(g, 0) are both OOD, so dual-axis
    guidance has no in-distribution unconditional branch.  Independent dropout
    makes g-only / m-only / both-dropped all training-time events.
    """

    def _mask_cond_ind(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            kg = torch.zeros(bs, 1, device=cond.device)
            km = torch.zeros(bs, 1, device=cond.device)
        elif self.training and self.cond_drop_prob > 0:
            p = float(self.cond_drop_prob)
            kg = 1.0 - torch.bernoulli(torch.full((bs, 1), p, device=cond.device))
            km = 1.0 - torch.bernoulli(torch.full((bs, 1), p, device=cond.device))
        else:
            kg = torch.ones(bs, 1, device=cond.device)
            km = torch.ones(bs, 1, device=cond.device)
        self._keep = km
        return cond * kg

    TMC.MaskTransformer.mask_cond = _mask_cond_ind
    TMC.ResidualTransformer.mask_cond = _mask_cond_ind


def patch_g_only_dropout(TMC):
    """HERO's original scheme: drop the intent g, never the evidence m."""

    def _mask_cond_g(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            kg = torch.zeros(bs, 1, device=cond.device)
        elif self.training and self.cond_drop_prob > 0:
            p = float(self.cond_drop_prob)
            kg = 1.0 - torch.bernoulli(torch.full((bs, 1), p, device=cond.device))
        else:
            kg = torch.ones(bs, 1, device=cond.device)
        self._keep = torch.ones(bs, 1, device=cond.device)
        return cond * kg

    TMC.MaskTransformer.mask_cond = _mask_cond_g
    TMC.ResidualTransformer.mask_cond = _mask_cond_g


def _mask_cfg(self, motion_ids, cond_vector, padding_mask, cond_scale=3,
              force_mask=False, memory=None):
    mode = getattr(self, "cfg_mode", "both")
    if force_mask:
        return self.trans_forward(motion_ids, cond_vector, padding_mask,
                                  force_mask=True, memory=memory)
    logits = self.trans_forward(motion_ids, cond_vector, padding_mask, memory=memory)
    if cond_scale == 1:
        return logits
    if mode == "intent":
        aux = self.trans_forward(motion_ids, cond_vector, padding_mask,
                                 force_mask=True, memory=memory, keep_memory=True)
    elif mode == "evidence":
        aux = self.trans_forward(motion_ids, cond_vector, padding_mask,
                                 memory=memory, zero_m=True)
    else:
        aux = self.trans_forward(motion_ids, cond_vector, padding_mask,
                                 force_mask=True, memory=memory)
    return aux + (logits - aux) * cond_scale


def _res_cfg(self, motion_codes, q_id, cond_vector, padding_mask, cond_scale=3,
             force_mask=False, memory=None):
    mode = getattr(self, "cfg_mode", "both")
    bs = motion_codes.shape[0]
    qids = torch.full((bs,), q_id, dtype=torch.long, device=motion_codes.device)
    if force_mask:
        logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask,
                                    force_mask=True, memory=memory)
        return self.output_project(logits, qids - 1)
    logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask, memory=memory)
    logits = self.output_project(logits, qids - 1)
    if cond_scale == 1:
        return logits
    if mode == "intent":
        aux = self.trans_forward(motion_codes, qids, cond_vector, padding_mask,
                                 force_mask=True, memory=memory, keep_memory=True)
    elif mode == "evidence":
        aux = self.trans_forward(motion_codes, qids, cond_vector, padding_mask,
                                 memory=memory, zero_m=True)
    else:
        aux = self.trans_forward(motion_codes, qids, cond_vector, padding_mask,
                                 force_mask=True, memory=memory)
    aux = self.output_project(aux, qids - 1)
    return aux + (logits - aux) * cond_scale


def apply(TMC):
    import os
    if os.environ.get("INTENT_PATCH", "1") == "0":
        return
    if getattr(TMC, "_intent_patched", False):
        return
    TMC.MaskTransformer.trans_forward = _mask_forward
    TMC.ResidualTransformer.trans_forward = _res_forward
    TMC.MaskTransformer.forward_with_cond_scale = _mask_cfg
    TMC.ResidualTransformer.forward_with_cond_scale = _res_cfg
    TMC._intent_patched = True
