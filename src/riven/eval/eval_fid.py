#!/usr/bin/env python3
"""E2 motion-space FID on any EgoReReal release (TRAIN-only PCA, GT-length aligned).

Same fixed evaluator and protocol as `split_1104_20261104/fid_1104.py`:
E2 256-d mean latent, TRAIN-only PCA64 as the primary number, per-sample FID
over generation samples, GT length as the alignment length.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("RIVEN_ROOT", HERE.parents[2]))
EVAL_DIR = Path(os.environ.get("RIVEN_EVAL_DIR", str(ROOT / "state/evaluator")))
sys.path.insert(0, str(HERE))
import evaluator_train as E  # noqa: E402


def frechet(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ca = np.cov(a, rowvar=False)
    cb = np.cov(b, rowvar=False)
    d, u = np.linalg.eigh(ca)
    sq = (u * np.sqrt(np.maximum(d, 0))) @ u.T
    cross = sq @ cb @ sq
    roots = np.sqrt(np.maximum(np.linalg.eigvalsh((cross + cross.T) / 2), 0)).sum()
    return float(max(0, np.square(a.mean(0) - b.mean(0)).sum()
                     + np.trace(ca) + np.trace(cb) - 2 * roots))


def summary(xs):
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs)),
            "per_sample": [float(x) for x in xs]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, help="name=generated.npz")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    split_doc = json.loads((args.release / "split.json").read_text())
    train = [str(x) for x in split_doc["train_real"]]
    ids = [str(x) for x in split_doc[args.split]]
    model = E.OfficialTMR(E.ASSETS / "distilbert-base-uncased", "cuda:0")
    ck = torch.load(EVAL_DIR / "E2_best.pt", map_location="cpu", weights_only=True)
    model.motion_encoder.load_state_dict(ck["motion_encoder"])
    model.motion_encoder.eval()
    del ck, model.backbone, model.text_encoder

    def embed(seqs):
        out = []
        for start in range(0, len(seqs), 32):
            fs = [(torch.as_tensor(E.preprocess(x), device=model.device) - model.mean)
                  / (model.std + 1e-12) for x in seqs[start:start + 32]]
            x = torch.nn.utils.rnn.pad_sequence(fs, batch_first=True)
            leng = torch.tensor([len(f) for f in fs], device=model.device)
            mask = torch.arange(x.shape[1], device=model.device)[None] < leng[:, None]
            with torch.no_grad():
                out.append(model.motion_encoder({"x": x, "mask": mask})[:, 0].cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    with np.load(args.release / "motion/gt22_full.npz") as z:
        train_m = embed([z["J_" + k] for k in train])
        gt = [z["J_" + k] for k in ids]
    mean = train_m.mean(0)
    _, _, vt = np.linalg.svd(train_m - mean, full_matrices=False)
    basis = vt[:64].T
    proj = lambda x: (x - mean) @ basis  # noqa: E731

    rng = np.random.default_rng(0)
    n = len(ids)
    floors64, floors256 = [], []
    for _ in range(20):
        perm = rng.permutation(len(train))
        x = train_m[perm[:n]]
        y = train_m[perm[n:2 * n]]
        if len(y) < n:
            continue
        floors256.append(frechet(x, y))
        floors64.append(frechet(proj(x), proj(y)))
    result = {
        "schema": "egoreal-baseline-fid/v1",
        "release": str(args.release),
        "split": args.split,
        "protocol": "E2 256D mean latent; TRAIN-only PCA64; GT-length aligned; per-sample FID",
        "train_gt_floor": {"fid64": summary(floors64), "fid256": summary(floors256)},
        "models": [],
    }
    for spec in args.model:
        name, sep, path = spec.partition("=")
        if not sep:
            raise ValueError("expected name=path")
        with np.load(path) as z:
            keys = [str(k) for k in z["val_ids"]]
            order = [keys.index(k) for k in ids]
            J = z["val_joints"][:, order]
            if J.ndim == 5 and J.shape[2] == 22 and J.shape[3] != 22:
                J = J.transpose(0, 1, 3, 2, 4)  # accept [S,N,22,T,3] generators
            L = z["lengths"][order]
        lengths = np.minimum(L, np.array([len(g) for g in gt]))
        G = embed([g[:int(m)] for g, m in zip(gt, lengths)])
        f64, f256 = [], []
        for s in range(len(J)):
            P = embed([J[s, i, : int(m)] for i, m in enumerate(lengths)])
            f64.append(frechet(proj(G), proj(P)))
            f256.append(frechet(G, P))
        result["models"].append({"name": name, "samples": len(J),
                                 "fid64": summary(f64), "fid256": summary(f256)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"models": [(m["name"], m["fid64"]["mean"], m["fid256"]["mean"])
                                 for m in result["models"]]}, indent=2))


if __name__ == "__main__":
    main()
