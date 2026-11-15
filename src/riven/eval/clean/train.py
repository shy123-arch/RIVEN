"""Train the HERO-style (text_mot_match) evaluator on EgoReReal TRAIN -> Track A'.

Same models/loss/hyperparameters as the official HumanML3D evaluator training
(train_tex_mot_match.py: margin contrastive loss, Adam 1e-4, clip 0.5, batch 32,
unit_length 4, max_text_len 20, dims 512/512/1024/512). The movement encoder is
our v4 VQ tokenizer's Encoder (EgoReReal decomp), frozen.
"""
import argparse, json, os, pickle, sys
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('RIVEN_ROOT', HERE.parents[4]))
HERO = Path(os.environ.get('RIVEN_HERO', ROOT / 'third_party/HERO'))
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERO))
from modules import TextEncoderBiGRUCo, MotionEncoderBiGRUCo, ContrastiveLoss  # noqa: E402
from models.vq.encdec import Encoder as VQEncoder  # noqa: E402

POS_OTHER = 14


class EgoDataset(Dataset):
    def __init__(self, data: Path, ids, mean, std, word2vec, max_text_len=20):
        self.items = []
        for k in ids:
            f = np.load(data / 'motion' / f'{k}.npy').astype(np.float32)
            toks = (data / 'text' / f'{k}.txt').read_text().strip().split('#')[1].split()
            toks = toks[:max_text_len]
            if len(toks) == 0:
                continue
            we = np.stack([word2vec.get(w, word2vec['unk']) for w in toks]).astype(np.float32)
            pos = np.zeros((len(toks), 15), np.float32)
            pos[:, POS_OTHER] = 1
            self.items.append((k, (f - mean) / std, we, pos, len(toks)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    batch = sorted(batch, key=lambda x: x[4], reverse=True)  # caption length, official collate
    B = len(batch)
    T = max(b[1].shape[0] for b in batch)
    L = max(b[4] for b in batch)
    mot = np.zeros((B, T, 263), np.float32)
    ml = np.zeros(B, np.int64)
    we = np.zeros((B, L, 300), np.float32)
    pos = np.zeros((B, L, 15), np.float32)
    cl = np.zeros(B, np.int64)
    for i, (k, f, w, p, l) in enumerate(batch):
        mot[i, :f.shape[0]] = f
        ml[i] = f.shape[0]
        we[i, :l] = w
        pos[i, :l] = p
        cl[i] = l
    return (torch.from_numpy(mot), torch.from_numpy(ml), torch.from_numpy(we),
            torch.from_numpy(pos), torch.from_numpy(cl))


class TrackAPrime(torch.nn.Module):
    def __init__(self, vq_ckpt, init_hero=None, device='cuda:0'):
        super().__init__()
        ck = torch.load(vq_ckpt, map_location='cpu')
        vq = ck['vq_model']
        enc = VQEncoder(263, 512, down_t=2, stride_t=2, width=512, depth=3)
        enc.load_state_dict({k[len('encoder.'):]: v for k, v in vq.items()
                             if k.startswith('encoder.')})
        for p in enc.parameters():
            p.requires_grad_(False)
        self.movement = enc.eval().to(device)
        self.text = TextEncoderBiGRUCo(300, 15, 512, 512, device).to(device)
        self.motion = MotionEncoderBiGRUCo(512, 1024, 512, device).to(device)
        if init_hero:
            h = torch.load(init_hero, map_location='cpu')
            missing = []
            for mod, key in ((self.text, 'text_encoder'), (self.motion, 'motion_encoder')):
                try:
                    mod.load_state_dict(h[key])
                except Exception as e:
                    missing.append(str(e)[:80])
            print('hero init issues:', missing, flush=True)

    def encode_motion(self, mot, ml):
        x = mot.permute(0, 2, 1)
        with torch.no_grad():
            mv = self.movement(x).permute(0, 2, 1)
        return self.motion(mv, torch.clamp(ml // 4, min=1))

    def forward(self, mot, ml, we, pos, cl):
        return self.encode_motion(mot, ml), self.text(we, pos, cl)


def retrieval(text_emb, motion_emb):
    sim = text_emb @ motion_emb.T
    ranks = 1 + (sim > sim.diag()[:, None]).sum(1) + ((sim == sim.diag()[:, None]).sum(1) - 1) / 2
    return {f'R@{k}': float((ranks <= k).float().mean()) for k in (1, 3, 10)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path(os.environ.get('RIVEN_CLEAN_EVAL_DATA', ROOT / 'state/clean_eval_data')))
    ap.add_argument('--release', type=Path, default=Path(os.environ.get('RIVEN_DATA', ROOT / 'data/egoreal')))
    ap.add_argument('--vq', type=Path, default=Path(os.environ.get('RIVEN_VQ', ROOT / 'cache/vq/best.pt')))
    ap.add_argument('--init-hero', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=Path(os.environ.get('RIVEN_CLEAN_EVALUATOR_DIR', ROOT / 'state/clean_evaluator')))
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--margin', type=float, default=10.0)
    ap.add_argument('--eval-every', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--smoke-steps', type=int, default=0)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20261111)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    held = set(uniq[:max(1, int(len(uniq) * 0.15))])
    tr_ids = [k for k in ids if src[k] not in held]
    va_ids = [k for k in ids if src[k] in held]
    if args.limit:
        tr_ids = tr_ids[:args.limit]
        va_ids = va_ids[:args.limit]
    print(f'train {len(tr_ids)} heldout {len(va_ids)} sources held {len(held)}', flush=True)

    ds_tr = EgoDataset(args.data, tr_ids, mean, std, word2vec)
    ds_va = EgoDataset(args.data, va_ids, mean, std, word2vec)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, collate_fn=collate, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=64, shuffle=False, collate_fn=collate)

    model = TrackAPrime(args.vq, args.init_hero, device)
    loss_fn = ContrastiveLoss(args.margin)
    opt = optim.Adam(list(model.text.parameters()) + list(model.motion.parameters()), lr=args.lr)

    best = -1.0
    step = 0
    for ep in range(1, args.epochs + 1):
        model.text.train(); model.motion.train()
        tot = 0.0
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
            clip_grad_norm_(list(model.text.parameters()) + list(model.motion.parameters()), 0.5)
            opt.step(); step += 1; tot += float(loss)
            if args.smoke_steps and step >= args.smoke_steps:
                print(f'smoke step {step} loss {tot/step:.4f}', flush=True)
                return
        if ep % args.eval_every == 0 or ep == args.epochs:
            model.text.eval(); model.motion.eval()
            tes, mes = [], []
            with torch.no_grad():
                for mot, ml, we, pos, cl in dl_va:
                    mot, ml, we, pos, cl = [t.to(device) for t in (mot, ml, we, pos, cl)]
                    order = torch.argsort(ml, descending=True)
                    me = model.encode_motion(mot[order], ml[order])
                    te = model.text(we, pos, cl)[order]
                    mes.append(me); tes.append(te)
            r = retrieval(torch.cat(tes), torch.cat(mes))
            print(f'epoch {ep} loss {tot/max(1,len(dl_tr)):.4f} heldout {r}', flush=True)
            if r['R@1'] > best:
                best = r['R@1']
                torch.save({'text_encoder': model.text.state_dict(),
                            'motion_encoder': model.motion.state_dict(),
                            'movement_encoder': model.movement.state_dict(),
                            'epoch': ep, 'heldout': r}, args.out / 'best.tar')
    print('done best', best, flush=True)


if __name__ == '__main__':
    main()
