"""CFG-consistent condition dropout for HERO's Mask/Residual transformers.

HERO's stock trans_forward drops only the global condition vector `g` in the
unconditional branch but keeps `memory` intact, so classifier-free guidance
(with cond_scale > 1) is only applied to g.  head.py patches this so `memory`
is dropped too.  Kept here as an opt-in (`--cfg-mem`) so the behaviour of the
already-trained checkpoints can be reproduced exactly.
"""
import torch


def apply(TMC):
    def _mask_cond(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            m = torch.ones(bs, 1, device=cond.device)
        elif self.training and self.cond_drop_prob > 0:
            m = torch.bernoulli(torch.full((bs, 1), float(self.cond_drop_prob), device=cond.device))
        else:
            m = torch.zeros(bs, 1, device=cond.device)
        self._keep = 1.0 - m
        return cond * self._keep

    def _m_trans_forward(self, motion_ids, cond, padding_mask, force_mask=False, memory=None):
        cond = self.global_local_crossatt_block(cond.unsqueeze(1), memory).squeeze(1)
        cond = self.mask_cond(cond, force_mask=force_mask)
        memory = memory * self._keep[:, :, None]
        x = self.input_process(self.token_emb(motion_ids))
        cond = self.cond_emb(cond).unsqueeze(0); memory = self.cond_emb(memory).permute((1, 0, 2))
        x = self.position_enc(x); xseq = torch.cat([cond, x], dim=0)
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:1]), padding_mask], dim=1)
        output = self.seqTransDecoder(tgt=xseq, memory=memory, tgt_key_padding_mask=padding_mask)[1:]
        return self.output_process(output)

    def _r_trans_forward(self, motion_codes, qids, cond, padding_mask, force_mask=False, memory=None):
        cond = self.global_local_crossatt_block(cond.unsqueeze(1), memory).squeeze(1)
        cond = self.mask_cond(cond, force_mask=force_mask)
        memory = memory * self._keep[:, :, None]
        x = self.input_process(motion_codes)
        q_emb = self.quant_emb(self.encode_quant(qids).float().to(x.device)).unsqueeze(0)
        cond = self.cond_emb(cond).unsqueeze(0); memory = self.cond_emb(memory).permute((1, 0, 2))
        x = self.position_enc(x); xseq = torch.cat([cond, q_emb, x], dim=0)
        padding_mask = torch.cat([torch.zeros_like(padding_mask[:, 0:2]), padding_mask], dim=1)
        output = self.seqTransDecoder(tgt=xseq, memory=memory, tgt_key_padding_mask=padding_mask)[2:]
        return self.output_process(output)

    TMC.MaskTransformer.mask_cond = _mask_cond
    TMC.MaskTransformer.trans_forward = _m_trans_forward
    TMC.ResidualTransformer.mask_cond = _mask_cond
    TMC.ResidualTransformer.trans_forward = _r_trans_forward
