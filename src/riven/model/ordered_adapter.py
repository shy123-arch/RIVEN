#!/usr/bin/env python3
"""Condition adapters for the m-mechanism redesign (2026-11-04).

OrderedAdapter:
  g = masked mean of prompt-end VLM tokens (semantics / global)
  m = 16 ordered visual-segment tokens + positional encoding, refined by a
      self-attention encoder (keeps HERO's temporal memory contract)

AttrHead:
  auxiliary heads on mean(m) predicting transferable attributes
  (plan fields + measured reaction kinematics), training-only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

LAYERS = (9, 18, 27, 36)


class OrderedAdapter(nn.Module):
    def __init__(self, use_layers=(36,), d_in=2560, d=512, n_slots=16,
                 n_enc_layers=1, n_heads=8, dropout=0.1, m_source="visual"):
        super().__init__()
        self.sel = [LAYERS.index(l) for l in use_layers]
        self.n_slots = n_slots
        self.m_source = m_source
        total = n_slots if m_source == "visual" else 2 * n_slots
        self.mix_t = nn.Parameter(torch.zeros(len(self.sel)))
        self.mix_v = nn.Parameter(torch.zeros(len(self.sel)))
        self.proj_t = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.proj_v = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(total, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_enc_layers)
        self.ln_mem = nn.LayerNorm(d)
        self.glob = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def _mix(self, H, mix):
        H = H[:, :, self.sel].float()
        H = F.layer_norm(H, H.shape[-1:])
        return (H * torch.softmax(mix, 0)[None, None, :, None]).sum(2)

    def forward(self, Ht, mt, Hv, mv=None):
        ht = self.proj_t(self._mix(Ht, self.mix_t))
        g = (ht * mt[..., None]).sum(1) / mt.sum(1, keepdim=True).clamp(min=1)
        hv = self.proj_v(self._mix(Hv, self.mix_v)) + self.pos[None, :Hv.shape[1]]
        if self.m_source == "visual":
            mem_in = hv
        else:
            n = Hv.shape[1]
            mem_in = torch.cat([hv, ht + self.pos[None, n:n + ht.shape[1]]], 1)
        if mv is None or bool(mv.all()):
            mem = self.enc(mem_in)
        else:
            mem = self.enc(mem_in, src_key_padding_mask=~mv)
        return self.glob(g), self.ln_mem(mem)


class AttrHead(nn.Module):
    """Training-only auxiliary heads on mean(m)."""

    def __init__(self, d=512, n_loc=4, n_dir=9, n_turn=4, n_post=3, n_kin=6):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU())
        self.loc = nn.Linear(d, n_loc)
        self.dir = nn.Linear(d, n_dir)
        self.turn = nn.Linear(d, n_turn)
        self.post = nn.Linear(d, n_post)
        self.kin = nn.Linear(d, n_kin)

    def forward(self, m):
        h = self.trunk(m.mean(1))
        return self.loc(h), self.dir(h), self.turn(h), self.post(h), self.kin(h)
