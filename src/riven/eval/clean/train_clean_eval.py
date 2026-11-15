#!/usr/bin/env python3
"""Train an EgoReReal FID evaluator with a from-scratch MovementConvEncoder.

This is the standard HumanML3D/KIT-style evaluator: movement Conv1d encoder +
BiGRU, trained jointly on EgoReReal TRAIN. No method's VQ is used.
"""
import argparse, json, os, pickle, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('RIVEN_ROOT', HERE.parents[4]))
HERO = Path(os.environ.get('RIVEN_HERO', ROOT / 'third_party/HERO'))
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERO))
from modules import TextEncoderBiGRUCo, MotionEncoderBiGRUCo, ContrastiveLoss  # noqa: E402
import train as T  # noqa: E402  (reuse EgoDataset / collate / retrieval)


class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class CleanEval(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.movement = MovementConvEncoder(259, 512, 512).to(device)
        self.text = TextEncoderBiGRUCo(300, 15, 512, 512, device).to(device)
        self.motion = MotionEncoderBiGRUCo(512, 1024, 512, device).to(device)

    def encode_motion(self, mot, ml):
        mv = self.movement(mot[..., :-4])
        return self.motion(mv, torch.clamp(ml // 4, min=1))

    def forward(self, mot, ml, we, pos, cl):
        return self.encode_motion(mot, ml), self.text(we, pos, cl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path(os.environ.get('RIVEN_CLEAN_EVAL_DATA', ROOT / 'state/clean_eval_data')))
    ap.add_argument('--release', type=Path, default=Path(os.environ.get('RIVEN_DATA', ROOT / 'data/egoreal')))
    ap.add_argument('--out', type=Path, default=Path(os.environ.get('RIVEN_CLEAN_EVALUATOR_DIR', ROOT / 'state/clean_evaluator')))
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--margin', type=float, default=10.0)
    ap.add_argument('--eval-every', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--smoke-steps', type=int, default=0)
    ap.add_argument('--init-hml3d', type=Path, default=None)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20261111)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = f'cuda:{args.gpu}'
    args.out.mkdir(parents=True, exist_ok=True)

    mean = np.load(args.data / 'meta' / 'mean.npy')
    std = np.load(args.data / 'meta' / 'std.npy')
    vecs = np.load(args.data / 'meta' / 'our_vab_data.npy')
    words = pickle.load(open(args.data / 'meta' / 'our_vab_words.pkl', 'rb'))
    idx = pickle.load(open(args.data / 'meta' / 'our_vab_idx.pkl', 'rb'))
    word2vec = {w: vecs[idx[w]] for w in words}

    ids = (args.data / 'train_ids.txt').read_text().split()
    man = json.load(open(args.release / 'data_manifest.json'))['rows']
    src = {k: str(man[k].get('source', '')) for k in ids}
    uniq = sorted(set(src.values()))
    rng = np.random.default_rng(0); rng.shuffle(uniq)
    held = set(uniq[:max(1, int(len(uniq) * 0.15))])
    tr_ids = [k for k in ids if src[k] not in held]
    va_ids = [k for k in ids if src[k] in held]
    if args.limit:
        tr_ids = tr_ids[:args.limit]; va_ids = va_ids[:args.limit]
    print(f'train {len(tr_ids)} heldout {len(va_ids)} sources held {len(held)}', flush=True)

    ds_tr = T.EgoDataset(args.data, tr_ids, mean, std, word2vec)
    ds_va = T.EgoDataset(args.data, va_ids, mean, std, word2vec)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, collate_fn=T.collate, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=64, shuffle=False, collate_fn=T.collate)

    model = CleanEval(device)
    if args.init_hml3d:
        h = torch.load(args.init_hml3d, map_location='cpu')
        model.movement.load_state_dict(h['movement_encoder'])
        model.text.load_state_dict(h['text_encoder'])
        model.motion.load_state_dict(h['motion_encoder'])
        print(f'initialized from official HumanML3D evaluator epoch {h.get("epoch")}', flush=True)
    loss_fn = ContrastiveLoss(args.margin)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    best = -1.0; step = 0
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for mot, ml, we, pos, cl in dl_tr:
            mot, ml, we, pos, cl = [t.to(device) for t in (mot, ml, we, pos, cl)]
            order = torch.argsort(ml, descending=True)
            me = model.encode_motion(mot[order], ml[order])
            te = model.text(we, pos, cl)[order]
            pos_lab = torch.zeros(te.shape[0], device=device)
            loss = loss_fn(te, me, pos_lab)
            sh = int(torch.randint(0, te.shape[0] - 1, (1,)).item())
            neg_idx = torch.arange(sh, sh + te.shape[0], device=device) % te.shape[0]
            loss = loss + loss_fn(te, me[neg_idx], torch.ones_like(pos_lab))
            opt.zero_grad(); loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            opt.step(); step += 1; tot += float(loss)
            if args.smoke_steps and step >= args.smoke_steps:
                print(f'smoke step {step} loss {tot/step:.4f}', flush=True)
                return
        if ep % args.eval_every == 0 or ep == args.epochs:
            model.eval(); tes, mes = [], []
            with torch.no_grad():
                for mot, ml, we, pos, cl in dl_va:
                    mot, ml, we, pos, cl = [t.to(device) for t in (mot, ml, we, pos, cl)]
                    order = torch.argsort(ml, descending=True)
                    me = model.encode_motion(mot[order], ml[order])
                    te = model.text(we, pos, cl)[order]
                    mes.append(me); tes.append(te)
            r = T.retrieval(torch.cat(tes), torch.cat(mes))
            print(f'epoch {ep} loss {tot/max(1,len(dl_tr)):.4f} heldout {r}', flush=True)
            if r['R@1'] > best:
                best = r['R@1']
                torch.save({'movement_encoder': model.movement.state_dict(),
                            'text_encoder': model.text.state_dict(),
                            'motion_encoder': model.motion.state_dict(),
                            'epoch': ep, 'heldout': r}, args.out / 'best.tar')
    print('done best', best, flush=True)


if __name__ == '__main__':
    main()
