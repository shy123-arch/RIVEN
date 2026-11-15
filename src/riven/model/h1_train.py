#!/usr/bin/env python3
"""Strict ViMo-free single-stage H1 (B + H1).

- VQ_v: trained on v1572 with HML3D RVQ init (no ViMo), 1 quantizer.
- H1: MaskTransformer (video cond = (g,m)), NO residual transformer.
- Optional HML3D text-motion pretraining using VQ_v codes and the text adapter.
- v1572 video fine-tune on TRAIN only.
- No ViMo checkpoints/statistics anywhere.
"""
import argparse, copy, importlib.util, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# External locations are configured through configs/paths.env (see
# configs/paths.env.example).  Every default is relative to this repository.
ROOT = Path(os.environ.get("RIVEN_ROOT", Path(__file__).resolve().parents[3]))
HERO = Path(os.environ.get("RIVEN_HERO", ROOT / "third_party/HERO"))
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("RIVEN_DATA", os.environ.get("UMRM_DATA", str(ROOT / "data/egoreal"))))
CACHE = Path(os.environ.get("RIVEN_CACHE", os.environ.get("UMRM_CACHE", str(ROOT / "cache"))))
HERO_GLUE = Path(os.environ.get("RIVEN_HERO_GLUE", HERE / "finetune_hero.py"))
sys.path.insert(0, str(HERO))
_cs = importlib.util.spec_from_file_location("FH", HERO_GLUE)
FH = importlib.util.module_from_spec(_cs); _cs.loader.exec_module(FH)
from models.vq.model import RVQVAE
import models.mask_transformer.transformer_memo_cross as TMC
torch = FH.torch
dev = torch.device("cuda:0")
LAYERS = (9, 18, 27, 36)


class Adapter(nn.Module):
    def __init__(self, use_layers=(36,), d_in=2560, d=512, n_slots=16):
        super().__init__()
        self.sel = [LAYERS.index(l) for l in use_layers]
        self.mix = nn.Parameter(torch.zeros(len(self.sel)))
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.q = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
        self.ln_mem = nn.LayerNorm(d); self.glob = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
    def forward(self, H, mask):
        H = H[:, :, self.sel].float(); H = F.layer_norm(H, H.shape[-1:])
        x = (H * torch.softmax(self.mix, 0)[None, None, :, None]).sum(2)
        h = self.proj(x); b = h.shape[0]
        q = self.q[None].expand(b, -1, -1)
        mem, _ = self.attn(q, h, h, key_padding_mask=~mask)
        g = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.glob(g), self.ln_mem(mem + q)


class VisualAdapter(nn.Module):
    """Perceiver over 16 visual-token segment means (Qwen video tokens)."""
    def __init__(self, use_layers=(9, 18, 27, 36), d_in=2560, d=512, n_slots=16):
        super().__init__()
        self.sel = [LAYERS.index(l) for l in use_layers]
        self.mix = nn.Parameter(torch.zeros(len(self.sel)))
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.q = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
        self.ln_mem = nn.LayerNorm(d)
        self.glob = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, H, mask=None):
        # H: [B, 16, nL, 2560]
        H = H[:, :, self.sel].float()
        H = (H * torch.softmax(self.mix, 0)[None, None, :, None]).sum(2)
        h = self.proj(H) + self.pos[None]
        q = self.q[None].expand(h.shape[0], -1, -1)
        mem, _ = self.attn(q, h, h)
        g = h.mean(1)
        return self.glob(g), self.ln_mem(mem + q)


class RawCond(nn.Module):
    """Skip the Perceiver: project the 16 layer-36 tokens directly to memory."""
    def __init__(self, d=512):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(2560), nn.Linear(2560, d), nn.GELU(), nn.Linear(d, d))
        self.glob = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
    def forward(self, H, mask):
        H = H[:, :, 3].float(); H = F.layer_norm(H, H.shape[-1:])
        h = self.proj(H)
        g = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.glob(g), h


def load_tf(name):
    meta = json.loads((CACHE / "textfeat" / f"{name}.meta.json").read_text())
    f = np.load(CACHE / "textfeat" / f"{name}.feats.npy", mmap_mode="r")
    return meta, f


def load_visual(name, cache_dir):
    d = Path(cache_dir)
    parts = sorted(d.glob(f"{name}.shard*.npz"))
    if not parts:
        raise FileNotFoundError(f"no visual shards for {name} in {d}")
    ids, feats = [], []
    for p in parts:
        z = np.load(p)
        ids += [str(x) for x in z["ids"]]
        feats.append(z["feats"])
    feats = np.concatenate(feats, 0)
    return {"ids": ids, "feats": feats, "row": {k: i for i, k in enumerate(ids)}}


def visual_batch(meta, ids, device):
    H = np.stack([meta["feats"][meta["row"][k]] for k in ids]).astype(np.float16)
    return torch.from_numpy(H).to(device), torch.ones((len(ids), H.shape[1]), dtype=torch.bool, device=device)


def tf_batch(meta, f, tids, device):
    L = max(meta["lengths"][t] for t in tids)
    H = np.zeros((len(tids), L, 4, 2560), np.float16); m = np.zeros((len(tids), L), bool)
    for i, t in enumerate(tids):
        n = meta["lengths"][t]; H[i, :n] = f[meta["offsets"][t]:meta["offsets"][t] + n]; m[i, :n] = True
    return torch.from_numpy(H).to(device), torch.from_numpy(m).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vq", default=str(HERE / "runs/vq_h1_512/best.pt"))
    ap.add_argument("--hml3d-steps", type=int, default=0)
    ap.add_argument("--adapter-init", default="text", choices=["text", "random"])
    ap.add_argument("--init-h1", default="", help="continue from an h1 checkpoint (trans+adapter)")
    ap.add_argument("--lam-nce", type=float, default=0.0)
    ap.add_argument("--slots", type=int, default=16)
    ap.add_argument("--cond", default="video", choices=["video", "text"])
    ap.add_argument("--cond-mode", default="perceiver", choices=["perceiver", "raw"])
    ap.add_argument("--lam-text-nce", type=float, default=0.0)
    ap.add_argument("--text-teacher", default=os.environ.get("RIVEN_TEXT_TEACHER", ""))
    ap.add_argument("--shuffle-text", action="store_true")
    ap.add_argument("--lam-mem-nce", type=float, default=0.0)
    ap.add_argument("--p-teacher", type=float, default=0.0)
    ap.add_argument("--train-frac", type=float, default=1.0, help="random subset of TR videos (data-scaling diagnostic)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    ck = torch.load(args.vq, map_location="cpu")
    mean, std = ck["mean"], ck["std"]; nb = ck["nb_code"]; q = ck["num_quantizers"]
    a = SimpleNamespace(num_quantizers=q, shared_codebook=False, quantize_dropout_prob=0.0,
                        quantize_dropout_cutoff_index=0, nb_code=nb, code_dim=512, mu=0.99)
    vq = RVQVAE(a, 263, nb, 512, 512, 2, 2, 512, 3, 3, "relu", None).to(dev)
    vq.load_state_dict(ck["vq_model"]); vq.eval()
    for p in vq.parameters():
        p.requires_grad_(False)

    z = np.load(DATA / "motion/humanml263_full.npz")
    ids = [str(x) for x in z["ids"]]; feats = z["features"]; lens = z["lengths"]; pos = {k: i for i, k in enumerate(ids)}
    split = json.loads((DATA / "split.json").read_text())
    tr_v = [str(x) for x in split["train_real"]]; va_v = [str(x) for x in split["val"]]
    if args.train_frac < 1.0:
        perm = np.random.default_rng(20261027).permutation(len(tr_v))
        n_keep = max(64, int(round(len(tr_v) * args.train_frac)))
        tr_v = [tr_v[i] for i in sorted(perm[:n_keep])]
        print(json.dumps({"train_frac": args.train_frac, "train_videos": len(tr_v)}), flush=True)
    cond_corpus = "a_gt3p_trainval" if args.cond == "video" else "gt3p_trainval"
    vmeta, vf = load_tf(cond_corpus); vpos = {x["id"]: x["tid"] for x in vmeta["items"]}
    print(json.dumps({"cond": args.cond, "corpus": cond_corpus}), flush=True)
    tmeta, tf = load_tf("gt3p_trainval"); tpos = {x["id"]: x["tid"] for x in tmeta["items"]}
    teacher = None
    use_teacher = (args.lam_text_nce > 0) or (args.lam_mem_nce > 0) or (args.p_teacher > 0)
    if use_teacher:
        teacher = Adapter().to(dev)
        teacher.load_state_dict(torch.load(args.text_teacher, map_location="cpu")["adapter"])
        teacher.eval()
        for p_ in teacher.parameters(): p_.requires_grad_(False)
        print(json.dumps({"text_teacher": args.text_teacher, "p_teacher": args.p_teacher}), flush=True)

    # precompute v1572 codes with VQ_v
    codes = {}
    with torch.no_grad():
        for k in tr_v + va_v:
            i = pos[k]; L = int(lens[i]) // 4 * 4
            x = torch.from_numpy(((feats[i, :L] - mean) / std).astype(np.float32))[None].to(dev)
            c, _ = vq.encode(x)
            codes[k] = c[0].cpu().numpy()
    print(json.dumps({"vq": str(args.vq), "codes": len(codes)}), flush=True)

    opt = SimpleNamespace(num_tokens=nb)
    trans = TMC.MaskTransformer(code_dim=512, cond_mode="video", latent_dim=384, ff_size=1024,
                                num_layers=6, num_heads=6, dropout=0.2, clip_dim=512,
                                cond_drop_prob=0.1, opt=opt).to(dev)
    ad_video = RawCond().to(dev) if args.cond_mode == "raw" else Adapter(n_slots=args.slots).to(dev)
    if args.init_h1:
        st0 = torch.load(args.init_h1, map_location="cpu")
        trans.load_state_dict(st0["trans"]); ad_video.load_state_dict(st0["adapter"])
        print(json.dumps({"init_h1": args.init_h1}), flush=True)

    class Readout(nn.Module):
        def __init__(self, d=512, out=256):
            super().__init__()
            self.net = nn.Sequential(nn.LayerNorm(d * 2), nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, out))
        def forward(self, g, m):
            return self.net(torch.cat([g, m.mean(1)], -1))

    readout = Readout().to(dev)
    txt_head = nn.Linear(512, 512).to(dev)
    mem_head = nn.Linear(512, 512).to(dev)
    zc = np.load(Path(os.environ.get("RIVEN_TMR_MOTION_EMB", CACHE / "tmr_motion_emb.npz")))
    zpos = {str(k): i for i, k in enumerate(zc["ids"])}; Zall = torch.tensor(zc["emb"])
    def nce(zh, zb, tau=0.07):
        a = F.normalize(zh, dim=-1); b = F.normalize(zb, dim=-1)
        return F.cross_entropy(a @ b.T / tau, torch.arange(len(b), device=b.device))
    params = list(trans.parameters()) + list(ad_video.parameters()) + list(readout.parameters()) + list(txt_head.parameters()) + list(mem_head.parameters())
    optm = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(optm, lambda s: min(1.0, (s + 1) / 200) * (0.1 if s >= 0.7 * args.steps else 1.0))

    # ---- optional HML3D text-motion pretraining (B) ----
    if args.hml3d_steps > 0:
        hz = np.load(Path(os.environ.get("RIVEN_HML3D_MOTION", CACHE / "hml3d_motion.npz")))
        hnames = [str(x) for x in hz["names"]]; hpos = {k: i for i, k in enumerate(hnames)}
        ho, hfeat_all = hz["offsets"], hz["feats"]
        hmeta, hfeat = load_tf("hml3d_train"); hitems = hmeta["items"]
        hmeta_v, hfeat_v = load_tf("hml3d_val"); hval_items = hmeta_v["items"]
        ad_text = Adapter().to(dev)
        p2 = list(trans.parameters()) + list(ad_text.parameters())
        o2 = torch.optim.AdamW(p2, lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-5)
        s2 = torch.optim.lr_scheduler.LambdaLR(o2, lambda t: min(1.0, (t + 1) / 200) * (0.1 if t >= 0.7 * args.hml3d_steps else 1.0))
        W = args.window

        def hbatch(bs):
            its = [hitems[rng.integers(len(hitems))] for _ in range(bs)]
            x = np.zeros((bs, W, 263), np.float32); Ls = []; tids = []; slid = []
            for b, it in enumerate(its):
                mi = hpos[it["motion"]]; seg = hfeat_all[ho[mi] + it["start"]:ho[mi] + it["end"]]
                n = (it["end"] - it["start"]) // 4 * 4
                w = min(W, n); w = max(4, w // 4 * 4)
                s0 = rng.integers(0, n - w + 1); s0 = s0 // 4 * 4
                x[b, :w] = ((seg[s0:s0 + w] - mean) / std).astype(np.float32)
                Ls.append(w // 4); tids.append(it["tid"]); slid.append(w)
            with torch.no_grad():
                c, _ = vq.encode(torch.from_numpy(x).to(dev))
            ids_t = torch.full((bs, max(Ls)), nb + 1, dtype=torch.long)
            for b, l in enumerate(Ls):
                ids_t[b, :l] = c[b, :l, 0].cpu()
            Hf, mk = tf_batch(hmeta, hfeat, tids, dev)
            g, m = ad_text(Hf, mk)
            return ids_t.to(dev), g, torch.tensor(Ls, device=dev), m

        @torch.no_grad()
        def hval_ce(n=128):
            trans.eval(); ad_text.eval(); tot = 0.0; cnt = 0
            for j in range(0, min(n, len(hval_items)), 64):
                its = hval_items[j:j + 64]
                x = np.zeros((len(its), W, 263), np.float32); Ls = []; tids = []
                for b, it in enumerate(its):
                    mi = hpos[it["motion"]]; seg = hfeat_all[ho[mi] + it["start"]:ho[mi] + it["end"]]
                    n = (it["end"] - it["start"]) // 4 * 4; w = max(4, min(W, n) // 4 * 4)
                    x[b, :w] = ((seg[:w] - mean) / std).astype(np.float32); Ls.append(w // 4); tids.append(it["tid"])
                with torch.no_grad():
                    c, _ = vq.encode(torch.from_numpy(x).to(dev))
                ids_t = torch.full((len(its), max(Ls)), nb + 1, dtype=torch.long)
                for b, l in enumerate(Ls):
                    ids_t[b, :l] = c[b, :l, 0].cpu()
                Hf, mk = tf_batch(hmeta_v, hfeat_v, tids, dev)
                g, m = ad_text(Hf, mk)
                ce, _, _ = trans(ids_t.to(dev), g, torch.tensor(Ls, device=dev), m)
                tot += float(ce) * len(its); cnt += len(its)
            trans.train(); ad_text.train()
            return tot / cnt

        print(json.dumps({"hml3d_items": len(hitems), "hml3d_val": len(hval_items)}), flush=True)
        for st in range(args.hml3d_steps + 1):
            if st > 0:
                ids_t, g, Ls, m = hbatch(args.batch)
                ce, _, acc = trans(ids_t, g, Ls, m)
                o2.zero_grad(); ce.backward(); torch.nn.utils.clip_grad_norm_(p2, 1.0); o2.step(); s2.step()
            if st % args.eval_every == 0:
                r = {"phase": "hml3d", "step": st, "val_ce": round(hval_ce(), 5)}
                if st > 0:
                    r.update({"tr_ce": float(ce), "tr_acc": float(acc), "tr_aux": float(aux), "tr_txt": float(txt_aux), "tr_mem": float(mem_aux)})
                print(json.dumps(r), flush=True)
            if st > 0 and st % args.save_every == 0:
                torch.save({"trans": FH.no_clip(trans.state_dict()), "adapter": ad_text.state_dict(),
                            "vq": str(args.vq), "phase": "hml3d", "step": st}, out / f"hml3d_step_{st}.pt")
        if args.adapter_init == "text":
            ad_video.load_state_dict(copy.deepcopy(ad_text.state_dict()))
        print(json.dumps({"hml3d_pretrain_done": True}), flush=True)

    def sample_v1572(bs):
        ii = rng.choice(len(tr_v), bs, replace=False); Xc, Lc, keys = [], [], []
        for j in ii:
            k = tr_v[j]; n = int(lens[pos[k]]) // 4 * 4
            w = min(args.window, n)
            w = max(4, w // 4 * 4)
            s = rng.integers(0, n - w + 1); s = s // 4 * 4
            Xc.append(codes[k][s // 4:(s + w) // 4, 0]); Lc.append(w // 4); keys.append(k)
        Lmax = max(Lc)
        ids_t = torch.full((bs, Lmax), nb + 1, dtype=torch.long)
        for b, c in enumerate(Xc):
            ids_t[b, :len(c)] = torch.from_numpy(c)
        Hf, mk = tf_batch(vmeta, vf, [vpos[k] for k in keys], dev)
        g, m = ad_video(Hf, mk)
        tHf = tmk = None
        if use_teacher:
            tHf, tmk = tf_batch(tmeta, tf, [tpos[k] for k in keys], dev)
        return ids_t.to(dev), g, torch.tensor(Lc, device=dev), m, keys, tHf, tmk

    @torch.no_grad()
    def val_ce(n=128):
        trans.eval(); ad_video.eval(); tot = 0.0; cnt = 0
        for j in range(0, min(n, len(va_v)), 64):
            ks = va_v[j:j + 64]
            ids_t = torch.full((len(ks), max(len(codes[k]) for k in ks)), nb + 1, dtype=torch.long)
            Ls = []
            for b, k in enumerate(ks):
                c = codes[k][:, 0]; ids_t[b, :len(c)] = torch.from_numpy(c); Ls.append(len(c))
            Hf, mk = tf_batch(vmeta, vf, [vpos[k] for k in ks], dev)
            g, m = ad_video(Hf, mk)
            ce, _, acc = trans(ids_t.to(dev), g, torch.tensor(Ls, device=dev), m)
            tot += float(ce) * len(ks); cnt += len(ks)
        trans.train(); ad_video.train()
        return tot / cnt

    log = open(out / "train.jsonl", "w"); t0 = time.time()
    for step in range(args.steps + 1):
        if step > 0:
            ids_t, g, Ls, m, keys, tHf, tmk = sample_v1572(args.batch)
            gt_t = mt_t = None
            if use_teacher:
                with torch.no_grad():
                    gt_t, mt_t = teacher(tHf, tmk)
                if args.shuffle_text:
                    perm = torch.randperm(len(gt_t), device=dev)
                    gt_t = gt_t[perm]; mt_t = mt_t[perm]
            g_ce, m_ce = g, m
            if args.p_teacher > 0 and rng.random() < args.p_teacher:
                g_ce, m_ce = gt_t, mt_t
            ce, _, acc = trans(ids_t, g_ce, Ls, m_ce)
            aux = torch.zeros((), device=dev)
            if args.lam_nce > 0:
                zh = readout(g, m)
                zt = Zall[[zpos[k] for k in keys]].to(dev)
                aux = nce(zh, zt)
                zc_ = zh - zh.mean(0)
                var = torch.relu(1.0 - torch.sqrt(zc_.var(0) + 1e-4)).mean()
                cov = zc_.T @ zc_ / max(1, len(zc_) - 1); off = cov - torch.diag(torch.diag(cov))
                aux = aux + 0.1 * var + 0.05 * (off ** 2).sum() / 256
            txt_aux = torch.zeros((), device=dev)
            if use_teacher:
                vg = txt_head(g)
                txt_aux = nce(vg, gt_t)
                vc = vg - vg.mean(0)
                var = torch.relu(1.0 - torch.sqrt(vc.var(0) + 1e-4)).mean()
                cov = vc.T @ vc / max(1, len(vc) - 1); off = cov - torch.diag(torch.diag(cov))
                txt_aux = txt_aux + 0.1 * var + 0.05 * (off ** 2).sum() / 512
            mem_aux = torch.zeros((), device=dev)
            if args.lam_mem_nce > 0:
                vm = mem_head(m.mean(1)); tm = mt_t.mean(1)
                mem_aux = nce(vm, tm)
                vc = vm - vm.mean(0)
                var = torch.relu(1.0 - torch.sqrt(vc.var(0) + 1e-4)).mean()
                cov = vc.T @ vc / max(1, len(vc) - 1); off = cov - torch.diag(torch.diag(cov))
                mem_aux = mem_aux + 0.1 * var + 0.05 * (off ** 2).sum() / 512
            loss = ce + args.lam_nce * aux + args.lam_text_nce * txt_aux + args.lam_mem_nce * mem_aux
            optm.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); optm.step(); sched.step()
        if step % args.eval_every == 0:
            r = {"step": step, "val_ce": round(val_ce(), 5), "s": round(time.time() - t0)}
            if step > 0:
                r.update({"tr_ce": float(ce), "tr_acc": float(acc), "tr_aux": float(aux), "tr_txt": float(txt_aux), "tr_mem": float(mem_aux)})
            log.write(json.dumps(r) + "\n"); log.flush(); print(json.dumps(r), flush=True)
        if step > 0 and step % args.save_every == 0:
            torch.save({"trans": FH.no_clip(trans.state_dict()), "adapter": ad_video.state_dict(),
                        "readout": readout.state_dict(), "vq": str(args.vq), "step": step, "args": vars(args)}, out / f"step_{step}.pt")
    torch.save({"trans": FH.no_clip(trans.state_dict()), "adapter": ad_video.state_dict(),
                "readout": readout.state_dict(), "vq": str(args.vq), "step": args.steps, "args": vars(args)}, out / "last.pt")


if __name__ == "__main__":
    main()
