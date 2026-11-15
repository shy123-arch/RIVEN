#!/usr/bin/env python3
"""Condition-adapter variants for the condition-side exploration.

All adapters consume prompt-end hidden states H [B, T, nL, 2560] and a token
mask [B, T], and return (g [B,512], m [B,K,512]).

  perceiver   : learned queries + cross-attention (current baseline, in h1_train)
  raw         : linear projection of the layer-L tokens; no cross-attention
  two_stream  : g from a late layer, m from one/two middle layers (concat)
  ordered     : linear + positional encoding + self-attention (order preserved)
  perceiver2  : two cross-attention layers
  idq         : queries derived from the global token (input-dependent)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

LAYERS = (9, 18, 27, 36)


def _pick(H, idx):
    x = H[:, :, idx].float()
    return F.layer_norm(x, x.shape[-1:])


def _mean_g(h, mask):
    return (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)


class RawLinear(nn.Module):
    """No cross-attention: g = mean of projected tokens, m = projected tokens."""
    def __init__(self, use_layers=(36,), d_in=2560, d=512, n_slots=16):
        super().__init__()
        self.sel = [LAYERS.index(l) for l in use_layers]
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.glob = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, H, mask):
        x = _pick(H, self.sel)
        if len(self.sel) > 1:
            x = x.mean(2)
        else:
            x = x[:, :, 0]
        h = self.proj(x)
        return self.glob(_mean_g(h, mask)), h


class TwoStream(nn.Module):
    """g from a late layer; m from one or two middle layers, concatenated."""
    def __init__(self, g_layer=36, m_layers=(18, 27), d_in=2560, d=512,
                 n_slots=16, n_enc_layers=1, n_heads=8, dropout=0.1):
        super().__init__()
        self.gi = LAYERS.index(g_layer)
        self.mi = [LAYERS.index(l) for l in m_layers]
        self.proj_g = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.proj_m = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
            for _ in self.mi])
        total = n_slots * len(self.mi)
        self.pos = nn.Parameter(torch.randn(total, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_enc_layers)
        self.ln_mem = nn.LayerNorm(d)
        self.glob = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, H, mask):
        g = self.glob(_mean_g(self.proj_g(_pick(H, [self.gi])[:, :, 0]), mask))
        ms = [p(_pick(H, [i])[:, :, 0]) for p, i in zip(self.proj_m, self.mi)]
        m = torch.cat(ms, 1) + self.pos[None, :sum(x.shape[1] for x in ms)]
        return g, self.ln_mem(self.enc(m))


class OrderedPrompt(nn.Module):
    """Ordered prompt tokens: linear + positional encoding + self-attention."""
    def __init__(self, use_layers=(36,), d_in=2560, d=512, n_slots=16,
                 n_enc_layers=1, n_heads=8, dropout=0.1):
        super().__init__()
        self.sel = [LAYERS.index(l) for l in use_layers]
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_enc_layers)
        self.ln_mem = nn.LayerNorm(d)
        self.glob = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, H, mask):
        x = _pick(H, self.sel)
        if len(self.sel) > 1:
            x = x.mean(2)
        else:
            x = x[:, :, 0]
        h = self.proj(x) + self.pos[None, :x.shape[1]]
        mem = self.enc(h)
        return self.glob(_mean_g(h, mask)), self.ln_mem(mem)


class Perceiver2(nn.Module):
    """Two cross-attention layers over learned queries."""
    def __init__(self, use_layers=(36,), d_in=2560, d=512, n_slots=16, n_heads=8):
        super().__init__()
        self.sel = [LAYERS.index(l) for l in use_layers]
        self.mix = nn.Parameter(torch.zeros(len(self.sel)))
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.q = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.attn1 = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d); self.ln_mem = nn.LayerNorm(d)
        self.glob = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, H, mask):
        x = _pick(H, self.sel)
        x = (x * torch.softmax(self.mix, 0)[None, None, :, None]).sum(2)
        h = self.proj(x)
        q = self.q[None].expand(h.shape[0], -1, -1)
        a1, _ = self.attn1(q, h, h, key_padding_mask=~mask)
        q = self.ln1(q + a1)
        a2, _ = self.attn2(q, h, h, key_padding_mask=~mask)
        mem = self.ln2(q + a2)
        return self.glob(_mean_g(h, mask)), self.ln_mem(mem)


class Perceiver4(nn.Module):
    """Three cross-attention layers over 20 learned queries, capacity-matched
    to RoleAdapter (6.59M vs 6.06M params; 20 memory tokens).

    Unlike RoleAdapter this tower reads the prompt and visual tokens as a
    single stream and produces both g and m from the same queries, so it is
    the "no role separation" control at matched capacity and token count.
    """
    def __init__(self, use_layers=(36,), d_in=2560, d=512, n_slots=20,
                 n_heads=8, n_layers=3):
        super().__init__()
        self.sel = [LAYERS.index(l) for l in use_layers]
        self.mix_p = nn.Parameter(torch.zeros(len(self.sel)))
        self.mix_v = nn.Parameter(torch.zeros(len(self.sel)))
        self.proj_p = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.proj_v = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.q = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.attns = nn.ModuleList([nn.MultiheadAttention(d, n_heads, batch_first=True)
                                    for _ in range(n_layers)])
        self.lns = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.ln_mem = nn.LayerNorm(d)
        self.glob = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, Hp, mp, Hv, mv):
        xp = _pick(Hp, self.sel)
        xp = (xp * torch.softmax(self.mix_p, 0)[None, None, :, None]).sum(2)
        xv = _pick(Hv, self.sel)
        xv = (xv * torch.softmax(self.mix_v, 0)[None, None, :, None]).sum(2)
        h = torch.cat([self.proj_p(xp), self.proj_v(xv)], 1)
        mask = torch.cat([mp, mv], 1)
        q = self.q[None].expand(h.shape[0], -1, -1)
        for attn, ln in zip(self.attns, self.lns):
            a, _ = attn(q, h, h, key_padding_mask=~mask)
            q = ln(q + a)
        return self.glob(_mean_g(h, mask)), self.ln_mem(q)


class Flat6M(nn.Module):
    """Equal-capacity flat dual-path control (21 tokens, one shared mixer).

    Token set is the same content as RoleAdapter:
      1 global token  = masked mean of prompt tokens at g_layer
      16 visual tokens= visual segment tokens at v_layer
      4 plan tokens   = contiguous prompt-segment means at p_layer
    All tokens go through ONE shared projection and ONE self-attention block
    (no role-specific projections).  Token 0 becomes the prefix, tokens 1..20
    become the memory.  Parameter count is matched to RoleAdapter (6.05M).
    """
    def __init__(self, g_layer=36, v_layer=27, p_layer=18, d_in=2560, d=512,
                 n_slots=16, n_plan=4, n_visual=16, n_heads=8):
        super().__init__()
        self.gi = LAYERS.index(g_layer); self.vi = LAYERS.index(v_layer)
        self.pi = LAYERS.index(p_layer)
        self.n_plan = n_plan; self.n_visual = n_visual
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d),
                                  nn.GELU(), nn.Linear(d, d))
        self.type_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=0.1,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.mix = nn.TransformerEncoder(layer, 1)
        self.post = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d),
                                  nn.GELU(), nn.Linear(2 * d, d))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.register_buffer("vis_idx", torch.linspace(0, n_slots - 1, n_visual).round().long(),
                             persistent=False)

    def _plan_tokens(self, hp, mp):
        # Split the valid prompt positions into n_plan contiguous groups and
        # average each group; no learned query parameters.
        b, t, d = hp.shape
        out = []
        for i in range(self.n_plan):
            lo = i * t // self.n_plan
            hi = max(lo + 1, (i + 1) * t // self.n_plan)
            out.append(hp[:, lo:hi].mean(1))
        return torch.stack(out, 1)

    def forward(self, Hp, mp, Hv, mv):
        hp_all = self.proj(_pick(Hp, [self.gi])[:, :, 0])
        g_tok = _mean_g(hp_all, mp).unsqueeze(1)
        hv = self.proj(_pick(Hv, [self.vi])[:, :, 0])
        if self.n_visual < hv.shape[1]:
            hv = hv[:, self.vis_idx]
        hp_p = self.proj(_pick(Hp, [self.pi])[:, :, 0])
        plan = self._plan_tokens(hp_p, mp)
        x = torch.cat([g_tok, hv, plan], 1)
        x = x + torch.cat([self.type_emb[0:1].expand(1, 1, -1),
                           self.type_emb[1:2].expand(1, hv.shape[1], -1),
                           self.type_emb[2:3].expand(1, plan.shape[1], -1)], 1)
        x = self.mix(x)
        x = self.post(x)
        return self.out(x[:, 0]), self.out(x[:, 1:])


class IDQ(nn.Module):
    """Input-dependent queries derived from the global pooled token."""
    def __init__(self, use_layers=(36,), d_in=2560, d=512, n_slots=16, n_heads=8):
        super().__init__()
        self.sel = [LAYERS.index(l) for l in use_layers]
        self.mix = nn.Parameter(torch.zeros(len(self.sel)))
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.qproj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_slots * d))
        self.n_slots = n_slots
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln_mem = nn.LayerNorm(d)
        self.glob = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, H, mask):
        x = _pick(H, self.sel)
        x = (x * torch.softmax(self.mix, 0)[None, None, :, None]).sum(2)
        h = self.proj(x)
        g = _mean_g(h, mask)
        b = h.shape[0]
        q = self.qproj(g).view(b, self.n_slots, -1)
        mem, _ = self.attn(q, h, h, key_padding_mask=~mask)
        return self.glob(g), self.ln_mem(mem + q)


class RoleAdapter(nn.Module):
    """Role-separated g/m:
      g  = masked mean of the prompt tokens at the semantic layer g_layer
      m  = [ visual segment tokens at v_layer + time embedding ]  (time axis)
           [ n_plan queries over prompt tokens at p_layer + type embedding ] (plan)
    """
    def __init__(self, g_layer=18, v_layer=27, p_layer=18, d_in=2560, d=512,
                 n_slots=16, n_plan=4, n_heads=8, dropout=0.1,
                 use_visual=True, use_plan=True, use_type=True,
                 use_time=True, n_visual=16, use_fields=False, riv_pool="mean",
                 swap=False, inverse=False):
        super().__init__()
        self.gi = LAYERS.index(g_layer); self.vi = LAYERS.index(v_layer); self.pi = LAYERS.index(p_layer)
        self.swap = bool(swap)
        self.inverse = bool(inverse)
        self.n_plan = n_plan
        self.use_visual = use_visual; self.use_plan = use_plan; self.use_type = use_type
        self.use_time = use_time; self.n_visual = n_visual
        self.riv_pool = riv_pool
        self.proj_g = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        if riv_pool == "attn":
            self.riv_q = nn.Parameter(torch.randn(1, d) * 0.02)
            self.riv_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
            self.riv_ln = nn.LayerNorm(d)
        if use_visual:
            self.proj_v = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
            if use_time:
                self.time = nn.Parameter(torch.randn(n_visual, d) * 0.02)
            self.register_buffer("vis_idx", torch.linspace(0, n_slots - 1, n_visual).round().long(), persistent=False)
        if use_plan:
            self.proj_p = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
            self.plan_q = nn.Parameter(torch.randn(n_plan, d) * 0.02)
            self.plan_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
            self.plan_ln = nn.LayerNorm(d)
        if use_type:
            self.type_emb = nn.Parameter(torch.randn(2, d) * 0.02)
        if self.inverse:
            k = (n_visual if use_visual else 0) + (n_plan if use_plan else 0)
            self.riv_pos = nn.Parameter(torch.randn(k, d) * 0.02)
        self.use_fields = use_fields
        if use_fields:
            self.emb_loc = nn.Embedding(4, d)
            self.emb_dir = nn.Embedding(9, d)
            self.emb_turn = nn.Embedding(4, d)
            self.emb_post = nn.Embedding(3, d)
            self.emb_dur = nn.Embedding(4, d)
            self.field_type = nn.Parameter(torch.randn(5, d) * 0.02)
        self.ln_mem = nn.LayerNorm(d)
        self.glob = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def _forward_swap(self, Hp, mp, Hv, mv):
        """Role-swap control: prefix from visual evidence, memory from prompt.

        Holds the adapter architecture, parameter count and token counts
        exactly fixed; only which source feeds the prefix vs. the memory is
        exchanged.
        """
        hv = _pick(Hv, [self.vi])[:, :, 0]
        mv_sel = mv
        if self.n_visual < hv.shape[1]:
            hv = hv[:, self.vis_idx]
            mv_sel = mv[:, self.vis_idx]
        hv = self.proj_v(hv)
        if self.use_time:
            hv = hv + self.time[None, :hv.shape[1]]
        g = self.glob(_mean_g(hv, mv_sel))
        hp = self.proj_p(_pick(Hp, [self.pi])[:, :, 0])
        n = min(self.n_visual, hp.shape[1])
        idx = torch.linspace(0, hp.shape[1] - 1, n).round().long().to(hp.device)
        sel = hp[:, idx]
        if self.use_time:
            sel = sel + self.time[None, :sel.shape[1]]
        parts = [sel + (self.type_emb[0] if self.use_type else 0)]
        if self.use_plan:
            q = self.plan_q[None].expand(hp.shape[0], -1, -1)
            plan, _ = self.plan_attn(q, hp, hp, key_padding_mask=~mp)
            plan = self.plan_ln(plan + q)
            parts.append(plan + (self.type_emb[1] if self.use_type else 0))
        return g, self.ln_mem(torch.cat(parts, 1))

    def _forward_inverse(self, Hp, mp, Hv, mv):
        """Inverse assignment: pooled AEM -> prefix, expanded RIV -> memory.

        Content is built exactly as in the full model (visual tokens at
        v_layer, plan queries at p_layer); only the prefix/memory assignment
        is exchanged.
        """
        parts = []
        b = Hp.shape[0]
        if self.use_visual:
            hv = _pick(Hv, [self.vi])[:, :, 0]
            if self.n_visual < hv.shape[1]:
                hv = hv[:, self.vis_idx]
            hv = self.proj_v(hv)
            if self.use_time:
                hv = hv + self.time[None, :hv.shape[1]]
            parts.append(hv + (self.type_emb[0] if self.use_type else 0))
        if self.use_plan:
            hp_p = self.proj_p(_pick(Hp, [self.pi])[:, :, 0])
            q = self.plan_q[None].expand(b, -1, -1)
            plan, _ = self.plan_attn(q, hp_p, hp_p, key_padding_mask=~mp)
            plan = self.plan_ln(plan + q)
            parts.append(plan + (self.type_emb[1] if self.use_type else 0))
        mem = torch.cat(parts, 1)
        ones = torch.ones(mem.shape[:2], device=mem.device, dtype=mp.dtype)
        g = self.glob(_mean_g(mem, ones))
        r = _mean_g(self.proj_g(_pick(Hp, [self.gi])[:, :, 0]), mp)
        m = r[:, None, :].expand(-1, mem.shape[1], -1) + self.riv_pos[None]
        return g, self.ln_mem(m)

    def forward(self, Hp, mp, Hv, mv, plan_ids=None):
        if self.inverse:
            return self._forward_inverse(Hp, mp, Hv, mv)
        if self.swap:
            return self._forward_swap(Hp, mp, Hv, mv)
        hp_g = self.proj_g(_pick(Hp, [self.gi])[:, :, 0])
        if self.riv_pool == "attn":
            q = self.riv_q.expand(hp_g.shape[0], -1, -1)
            g_att, _ = self.riv_attn(q, hp_g, hp_g, key_padding_mask=~mp)
            g = self.riv_ln(g_att.squeeze(1) + q.squeeze(1))
        elif self.riv_pool == "last":
            # Last non-padding prompt token.
            idx = (mp.sum(1) - 1).clamp(min=0)
            d = hp_g.shape[-1]
            g = hp_g.gather(1, idx[:, None, None].expand(-1, 1, d)).squeeze(1)
        else:
            g = _mean_g(hp_g, mp)
        parts = []
        if self.use_visual:
            hv = _pick(Hv, [self.vi])[:, :, 0]
            if self.n_visual < hv.shape[1]:
                hv = hv[:, self.vis_idx]
            hv = self.proj_v(hv)
            if self.use_time:
                hv = hv + self.time[None, :hv.shape[1]]
            parts.append(hv + (self.type_emb[0] if self.use_type else 0))
        if self.use_plan:
            hp_p = self.proj_p(_pick(Hp, [self.pi])[:, :, 0])
            q = self.plan_q[None].expand(hp_p.shape[0], -1, -1)
            plan, _ = self.plan_attn(q, hp_p, hp_p, key_padding_mask=~mp)
            plan = self.plan_ln(plan + q)
            parts.append(plan + (self.type_emb[1] if self.use_type else 0))
        if self.use_fields and plan_ids is not None:
            f = torch.stack([self.emb_loc(plan_ids[:, 0]), self.emb_dir(plan_ids[:, 1]),
                             self.emb_turn(plan_ids[:, 2]), self.emb_post(plan_ids[:, 3]),
                             self.emb_dur(plan_ids[:, 4])], 1)
            parts.append(f + self.field_type[None])
        m = torch.cat(parts, 1)
        return self.glob(g), self.ln_mem(m)
