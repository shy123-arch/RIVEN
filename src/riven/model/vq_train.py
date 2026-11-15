#!/usr/bin/env python3
"""ViMo-free VQ for URM v1572 (B + H1 plan, stage 1).

- init encoder/decoder + first codebook from HumanML3D RVQ (NOT ViMo)
- mean/std recomputed from v1572 TRAIN features
- 1 quantizer (single-stage H1; no residual transformer)
- train on v1572 TRAIN only, select by VAL reconstruction MPJPE
"""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get("RIVEN_ROOT", Path(__file__).resolve().parents[3]))
HERO = Path(os.environ.get("RIVEN_HERO", ROOT / "third_party/HERO"))
DATA = Path(os.environ.get("RIVEN_DATA", ROOT / "data/egoreal"))
sys.path.insert(0, str(HERO))
_cs = importlib.util.spec_from_file_location(
    "FH", Path(os.environ.get("RIVEN_HERO_GLUE", Path(__file__).resolve().parent / "finetune_hero.py")))
FH = importlib.util.module_from_spec(_cs); _cs.loader.exec_module(FH)
from models.vq.model import RVQVAE
torch = FH.torch

dev = torch.device("cuda:0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA), help="data release directory (split.json + motion/*.npz)")
    ap.add_argument("--all-codebooks", action="store_true", help="load all 6 HML3D codebooks, not only q1")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--window-max", type=int, default=0, help="0 = fixed --window; >0 = random windows up to this length")
    ap.add_argument("--window-min", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--nb-code", type=int, default=512)
    ap.add_argument("--quantizers", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--no-init", action="store_true",
                    help="skip HumanML3D initialization (random tokenizer control, E10)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    data = Path(args.data)
    z = np.load(data / "motion/humanml263_full.npz")
    ids = [str(x) for x in z["ids"]]; feats = z["features"].astype(np.float32); lens = z["lengths"]
    pos = {k: i for i, k in enumerate(ids)}
    split = json.loads((data / "split.json").read_text())
    tr = [str(x) for x in split["train_real"]]; va = [str(x) for x in split["val"]]
    G = np.load(data / "motion/gt22_full.npz")

    # v1572 TRAIN mean/std (no ViMo)
    cnt = 0; s1 = np.zeros(263, np.float64); s2 = np.zeros(263, np.float64)
    for k in tr:
        n = int(lens[pos[k]]); x = feats[pos[k], :n]
        s1 += x.sum(0); s2 += (x * x).sum(0); cnt += n
    mean = (s1 / cnt).astype(np.float32); std = np.sqrt(s2 / cnt - (s1 / cnt) ** 2).astype(np.float32) + 1e-6
    print(json.dumps({"frames": cnt, "mean_abs": float(np.abs(mean).mean()), "std_mean": float(std.mean())}), flush=True)
    np.save(out / "mean.npy", mean); np.save(out / "std.npy", std)

    norm = ((feats - mean) / std).astype(np.float32)
    targets = {k: G[f"J_{k}"].astype(np.float64) for k in tr + va}

    class NS: pass
    a = NS(); a.num_quantizers = args.quantizers; a.shared_codebook = False
    a.quantize_dropout_prob = 0.0; a.quantize_dropout_cutoff_index = 0
    a.nb_code = args.nb_code; a.code_dim = 512; a.mu = 0.99
    vq = RVQVAE(a, 263, args.nb_code, 512, 512, 2, 2, 512, 3, 3, "relu", None).to(dev)
    # HML3D init (NOT ViMo).  --no-init keeps the random initialisation and is
    # the tokenizer-prior control (E10).
    if args.no_init:
        print(json.dumps({"init": "random", "init_loaded": 0}), flush=True)
    else:
        rvq = Path(os.environ.get(
            "RIVEN_HML3D_RVQ",
            HERO / "checkpoints/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2/model/net_best_fid.tar"))
        ck = torch.load(rvq, map_location="cpu")
        sd = ck["net"]; own = vq.state_dict(); load = {}
        for k, v in sd.items():
            if not args.all_codebooks and k.startswith("quantizer.layers.") and not k.startswith("quantizer.layers.0."):
                continue
            if k in own and own[k].shape == v.shape:
                load[k] = v
        missing = [k for k in own if k not in load]
        vq.load_state_dict(load, strict=False)
        print(json.dumps({"init": "hml3d", "init_loaded": len(load), "missing": len(missing)}), flush=True)

    opt = torch.optim.AdamW(vq.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(args.steps * 0.7), int(args.steps * 0.9)], gamma=0.5)

    def sample(bs):
        keys = [tr[rng.integers(len(tr))] for _ in range(bs)]
        ns = [int(lens[pos[k]]) for k in keys]
        if args.window_max <= 0:
            ws = [min(args.window, n) for n in ns]
        else:
            ws = []
            for n in ns:
                lo = min(args.window_min, n); hi = min(args.window_max, n)
                w = int(rng.integers(lo, hi + 1)); w = max(4, w // 4 * 4); ws.append(min(w, n))
        W = max(ws); xs = np.zeros((bs, W, 263), np.float32); mask = np.zeros((bs, W), bool)
        for b, (k, n, w) in enumerate(zip(keys, ns, ws)):
            s = rng.integers(0, n - w + 1)
            xs[b, :w] = norm[pos[k], s:s + w]; mask[b, :w] = True
        return torch.from_numpy(xs).to(dev), torch.from_numpy(mask).to(dev)

    @torch.no_grad()
    def val_mpjpe(n=0):
        vq.eval(); errs = []
        for k in (va if n == 0 else va[:n]):
            i = pos[k]; L = int(lens[i]) // 4 * 4
            x = torch.from_numpy(norm[i, :L])[None].to(dev)
            codes, _ = vq.encode(x)
            rec = vq.forward_decoder(codes).cpu().numpy()[0]
            j = FH.to_canonical_joints(rec[:L], mean, std)
            errs.append(FH.score(j, targets[k][:L], np.ones(L, bool))["mpjpe_m"])
        vq.train()
        return float(np.mean(errs))

    best = (float("inf"), 0)
    t0 = time.time(); log = open(out / "train.jsonl", "w")
    for step in range(args.steps + 1):
        if step > 0:
            x, mask = sample(args.batch)
            x_hat, commit, ppl = vq(x)
            m3 = mask[..., None].float()
            rec = ((x_hat - x).abs() * m3).sum() / (m3.sum() * x.shape[-1] + 1e-6)
            loc = (((x_hat - x)[..., 4:67]).abs() * m3).sum() / (m3.sum() * 63 + 1e-6)
            loss = rec + 0.5 * loc + 0.02 * commit
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % args.eval_every == 0:
            m = val_mpjpe()
            r = {"step": step, "val_recon_mpjpe": m, "s": round(time.time() - t0)}
            if step > 0:
                r.update({"loss": float(loss), "rec": float(rec), "loc": float(loc),
                          "commit": float(commit), "ppl": float(ppl)})
            log.write(json.dumps(r) + "\n"); log.flush(); print(json.dumps(r), flush=True)
            if m < best[0]:
                best = (m, step)
                torch.save({"vq_model": vq.state_dict(), "step": step, "mpjpe": m,
                            "nb_code": args.nb_code, "num_quantizers": args.quantizers,
                            "mean": mean, "std": std}, out / "best.pt")
    torch.save({"vq_model": vq.state_dict(), "step": args.steps, "nb_code": args.nb_code,
                "num_quantizers": args.quantizers, "mean": mean, "std": std}, out / "last.pt")
    (out / "result.json").write_text(json.dumps({"best_val_recon_mpjpe": best[0], "best_step": best[1],
                                                 "steps": args.steps, "nb_code": args.nb_code,
                                                 "quantizers": args.quantizers}, indent=1))


if __name__ == "__main__":
    main()
