#!/usr/bin/env python3
"""Group 5 evaluation with the paper's clean HML3D-init EgoReReal evaluator.

Mirrors unseen_split_20261112/eval_old_unseen.py (512-D FID, 32-pool source-
excluded R@1/3/10) but accepts arbitrary name=path npz specs. Runs the three
clean evaluator seeds and averages.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("RIVEN_ROOT", HERE.parents[4]))
REL = Path(os.environ.get("RIVEN_DATA", ROOT / "data/egoreal"))
DATA = Path(os.environ.get("RIVEN_CLEAN_EVAL_DATA", ROOT / "state/clean_eval_data"))
EVAL_ROOT = Path(os.environ.get("RIVEN_CLEAN_EVAL_ROOT", ROOT / "state/clean_evaluators"))
DEV = os.environ.get("RIVEN_DEVICE", "cuda:0")

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(Path(os.environ.get("RIVEN_HERO", ROOT / "third_party/HERO"))))
from tmr_adapter import preprocess  # noqa: E402
import train as T  # noqa: E402
from train_clean_eval import CleanEval  # noqa: E402


def frechet(a, b) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    d, u = np.linalg.eigh(ca)
    sq = (u * np.sqrt(np.maximum(d, 0))) @ u.T
    cross = sq @ cb @ sq
    roots = np.sqrt(np.maximum(np.linalg.eigvalsh((cross + cross.T) / 2), 0)).sum()
    return float(max(0.0, np.square(a.mean(0) - b.mean(0)).sum()
                     + np.trace(ca) + np.trace(cb) - 2 * roots))


@torch.no_grad()
def encode_feats(model, feats, mean, std, chunk=256):
    out = []
    for s in range(0, len(feats), chunk):
        batch = [((f.astype(np.float32) - mean) / (std + 1e-12)) for f in feats[s:s + chunk]]
        lens = np.array([len(f) for f in batch], np.int64)
        Tmax = int(lens.max()); x = np.zeros((len(batch), Tmax, 263), np.float32)
        for i, f in enumerate(batch):
            x[i, :len(f)] = f
        order = np.argsort(-lens); inv = np.argsort(order)
        emb = model.encode_motion(torch.from_numpy(x[order]).to(DEV),
                                  torch.from_numpy(lens[order]).to(DEV)).cpu().numpy()
        out.append(emb[inv])
    return np.concatenate(out, 0)


def ranks_for(text_emb, motion_emb, groups):
    S, N, _ = motion_emb.shape
    rng = np.random.default_rng(0)
    acc = {k: np.zeros((S, N), np.float64) for k in (1, 2, 3, 10)}
    for s in range(S):
        for i in range(N):
            neg = [j for j in range(N) if j != i and groups[j] != groups[i]]
            if len(neg) < 31:
                continue
            rr = []
            for _ in range(20):
                cols = np.r_[i, rng.choice(neg, 31, replace=False)]
                v = text_emb[i] @ motion_emb[s, cols].T
                rr.append(1 + (v > v[0]).sum() + ((v == v[0]).sum() - 1) / 2)
            rr = np.asarray(rr)
            for k in (1, 2, 3, 10):
                acc[k][s, i] = float((rr <= k).mean())
    return {f"R@{q}": float(acc[q].mean()) for q in (1, 2, 3, 10)}, acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="append", required=True, help="name=path.npz")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--eval-seeds", default="20261111,20261112,20261113")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    ap.add_argument("--release", type=Path, default=REL)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    specs = [s.split("=", 1) for s in a.spec]
    seeds = [s for s in a.eval_seeds.split(",") if s]

    split = json.loads((a.release / "split.json").read_text())
    man = json.loads((a.release / "data_manifest.json").read_text())["rows"]
    ids = [str(x) for x in split[a.split]]
    source_key = split.get("source_key", {})
    groups = [source_key[k]["source_key"] if source_key else man[k].get("source", "") for k in ids]
    cats = {k: man[k].get("category", "?") for k in ids}

    mean = np.load(a.data / "meta/mean.npy").astype(np.float32)
    std = np.load(a.data / "meta/std.npy").astype(np.float32)
    vecs = np.load(a.data / "meta/our_vab_data.npy")
    words = pickle.load(open(a.data / "meta/our_vab_words.pkl", "rb"))
    idx = pickle.load(open(a.data / "meta/our_vab_idx.pkl", "rb"))
    word2vec = {w: vecs[idx[w]] for w in words}
    ds = T.EgoDataset(a.data, ids, mean, std, word2vec)
    items = {k: (f, w, p, l) for k, f, w, p, l in ds.items}
    ids = [k for k in ids if k in items]
    groups = [source_key[k]["source_key"] if source_key else man[k].get("source", "") for k in ids]
    cats = {k: man[k].get("category", "?") for k in ids}

    result = {"schema": "group5-clean-eval/v1", "split": a.split, "ids": len(ids),
              "eval_seeds": seeds, "protocol": "clean HML3D-init + EgoReReal TRAIN; "
              "512-D FID; R t2m 32-pool source-excluded; S=15; per-repeat",
              "per_eval": {}, "specs": {}}
    pool = ProcessPoolExecutor(max_workers=a.workers)
    for es in seeds:
        evdir = a.eval_root / f"evaluator_hml3d_clean_s{es}"
        if not (evdir / "best.tar").is_file():
            raise SystemExit(f"missing evaluator {evdir}")
        model = CleanEval(DEV)
        ck = torch.load(evdir / "best.tar", map_location="cpu", weights_only=True)
        model.movement.load_state_dict(ck["movement_encoder"])
        model.text.load_state_dict(ck["text_encoder"])
        model.motion.load_state_dict(ck["motion_encoder"])
        model.eval()
        with torch.no_grad():
            text_emb = []
            for k in ids:
                _, w, p, l = items[k]
                text_emb.append(model.text(torch.from_numpy(w)[None].to(DEV),
                                           torch.from_numpy(p)[None].to(DEV),
                                           torch.tensor([l]).to(DEV)).cpu().numpy()[0])
            text_emb = np.stack(text_emb)
        gt_feats = [np.load(a.data / "motion" / f"{k}.npy") for k in ids]
        gt_emb = encode_feats(model, gt_feats, mean, std)
        gt_ranks, _ = ranks_for(text_emb, gt_emb[None], groups)
        result["per_eval"][es] = {"gt": gt_ranks, "methods": {}}
        print(json.dumps({"eval_seed": es, "gt_R@10": gt_ranks["R@10"]}), flush=True)

        for name, path in specs:
            z = np.load(path, allow_pickle=True)
            got = [str(x) for x in z["val_ids"]]
            order = [got.index(k) for k in ids]
            J = z["val_joints"][:, order]
            L = z["lengths"][order].astype(int)
            glen = np.array([len(gt_feats[i]) for i in range(len(ids))])
            n_fid = np.minimum(L, glen)
            embs_r, fids = [], []
            for si in range(J.shape[0]):
                seqs_fid, seqs_r = [], []
                for i in range(len(ids)):
                    x = J[si, i]
                    if x.shape[0] == 22 and x.shape[1] != 22:
                        x = x.transpose(1, 0, 2)
                    seqs_fid.append(x[:int(n_fid[i])])
                    seqs_r.append(x[:int(L[i])])
                feats_fid = list(pool.map(preprocess, seqs_fid, chunksize=8))
                feats_r = list(pool.map(preprocess, seqs_r, chunksize=8))
                E_fid = encode_feats(model, feats_fid, mean, std)
                E_r = encode_feats(model, feats_r, mean, std)
                embs_r.append(E_r)
                fids.append(frechet(gt_emb, E_fid))
            embs_r = np.stack(embs_r)
            rm, acc = ranks_for(text_emb, embs_r, groups)
            result["per_eval"][es]["methods"][name] = {
                "fid": float(np.mean(fids)), "fid_sd": float(np.std(fids, ddof=1)),
                "fid_per_sample": [float(x) for x in fids],
                "R": rm,
                "per_clip": {f"R@{q}": acc[q].mean(0).tolist() for q in (1, 2, 3, 10)},
                "R@10_by_category": {f"cat_{c}": float(acc[10][:, [i for i, k in enumerate(ids) if cats[k] == c]].mean())
                                     for c in sorted(set(cats.values()))},
            }
            print(json.dumps({"eval_seed": es, "method": name,
                              "fid": round(result["per_eval"][es]["methods"][name]["fid"], 3),
                              "R@1": round(rm["R@1"], 4), "R@10": round(rm["R@10"], 4)}), flush=True)
        del model
        torch.cuda.empty_cache()

    # aggregate over evaluator seeds
    names = [n for n, _ in specs]
    for name in names:
        recs = [result["per_eval"][es]["methods"][name] for es in seeds
                if name in result["per_eval"][es]["methods"]]
        agg = {"n_eval_seeds": len(recs),
               "fid": float(np.mean([r["fid"] for r in recs])),
               "fid_sd_eval": float(np.std([r["fid"] for r in recs], ddof=1)) if len(recs) > 1 else 0.0,
               "fid_per_sample": np.mean([r["fid_per_sample"] for r in recs], 0).tolist(),
               "R": {k: float(np.mean([r["R"][k] for r in recs])) for k in ("R@1", "R@2", "R@3", "R@10")}}
        for k in ("R@1", "R@2", "R@3", "R@10"):
            agg.setdefault("per_clip", {})[k] = np.mean([r["per_clip"][k] for r in recs], 0).tolist()
        result["specs"][name] = agg
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({n: {"fid": round(v["fid"], 3), **{k: round(x, 4) for k, x in v["R"].items()}}
                      for n, v in result["specs"].items()}, indent=1))
    pool.shutdown()


if __name__ == "__main__":
    main()
