#!/usr/bin/env python3
"""Fixed-evaluator reaction retrieval (E2 / UMRM TMR-like, 32-pool, per-sample).

Input npz contract shared by every baseline generator:
    val_ids   [N]            ids in any order
    val_joints[S, N, T, 22, 3]  generated 22-joint motions (metres, canonical)
    lengths   [N]            generated motion length in frames

Protocol is identical to `split_1104_20261104/eval_1104.py`: source-excluded
32-candidate pools, 20 pools per query, seed 0, text-to-motion direction,
per-sample R@1/3/10 averaged over generation samples.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("RIVEN_ROOT", HERE.parents[2]))
EVAL_DIR = Path(os.environ.get("RIVEN_EVAL_DIR", str(ROOT / "state/evaluator")))
sys.path.insert(0, str(HERE))
import evaluator_train as E  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    split_doc = json.loads((args.release / "split.json").read_text())
    rows = json.loads((args.release / "data_manifest.json").read_text())["rows"]
    key = "train_real" if args.split == "train" else args.split
    ids_all = [str(x) for x in split_doc[key]]
    # 749 new mocap rows have a placeholder exec (no real text label).  Text-to-motion
    # retrieval is only meaningful on labeled reactions, so R is computed there.
    ids = [k for k in ids_all
           if rows[k].get("exec_src") != "placeholder_pending_plan_exec"
           and (rows[k].get("exec") or "").strip()]
    source_key = split_doc.get("source_key")
    groups = [source_key[k]["source_key"] for k in ids] if source_key else None
    subsets = split_doc.get("subsets", {})
    seen = set(map(str, subsets.get(f"{args.split}_seen", [])))
    unseen = set(map(str, subsets.get(f"{args.split}_unseen", [])))
    subset = {"all": set(ids)}
    if seen:
        subset["seen"] = seen
    if unseen:
        subset["unseen"] = unseen
    cats = sorted({rows[k].get("category", "?") for k in ids})
    for c in cats:
        subset[f"cat_{c}"] = {k for k in ids if rows[k].get("category") == c}
        if unseen:
            subset[f"unseen_cat_{c}"] = {k for k in ids if rows[k].get("category") == c and k in unseen}

    model = E.OfficialTMR(E.ASSETS / "distilbert-base-uncased", "cuda:0")
    ck = torch.load(EVAL_DIR / "E2_best.pt", map_location="cpu", weights_only=True)
    model.text_encoder.load_state_dict(ck["text_encoder"])
    model.motion_encoder.load_state_dict(ck["motion_encoder"])
    model.text_encoder.eval()
    model.motion_encoder.eval()
    del ck

    texts = [rows[k]["exec"] for k in ids]
    tokens = []
    for start in range(0, len(texts), 32):
        inp = model.tokenizer(texts[start:start + 32], return_tensors="pt",
                              padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            hidden = model.backbone(**inp).last_hidden_state
        tokens += [h[m] for h, m in zip(hidden, inp.attention_mask.bool())]
    del model.backbone

    def enc_text():
        out = []
        with torch.no_grad():
            for start in range(0, len(tokens), 32):
                chunk = tokens[start:start + 32]
                x = torch.nn.utils.rnn.pad_sequence(chunk, batch_first=True).to(model.device)
                leng = torch.tensor([len(c) for c in chunk], device=model.device)
                mask = torch.arange(x.shape[1], device=model.device)[None] < leng[:, None]
                out.append(F.normalize(model.text_encoder({"x": x, "mask": mask})[:, 0], dim=-1).cpu().numpy())
        return np.concatenate(out)

    def enc_motion(seqs):
        out = []
        with torch.no_grad():
            for start in range(0, len(seqs), 32):
                chunk = [((torch.as_tensor(E.preprocess(x), device=model.device) - model.mean)
                          / (model.std + 1e-12)) for x in seqs[start:start + 32]]
                x = torch.nn.utils.rnn.pad_sequence(chunk, batch_first=True)
                leng = torch.tensor([len(c) for c in chunk], device=model.device)
                mask = torch.arange(x.shape[1], device=model.device)[None] < leng[:, None]
                out.append(F.normalize(model.motion_encoder({"x": x, "mask": mask})[:, 0], dim=-1).cpu().numpy())
        return np.concatenate(out)

    with np.load(args.npz) as z:
        got = [str(x) for x in z["val_ids"]]
        if len(got) != len(ids_all) or set(got) != set(ids_all):
            raise ValueError("generated npz membership mismatch")
        order = [got.index(k) for k in ids]
        J = z["val_joints"][:, order]
        if J.ndim == 5 and J.shape[2] == 22 and J.shape[3] != 22:
            J = J.transpose(0, 1, 3, 2, 4)  # accept [S,N,22,T,3] generators
        L = z["lengths"][order].astype(np.int64)
    T = enc_text()
    rng = np.random.default_rng(0)
    per_sample = []
    for s in range(J.shape[0]):
        M = enc_motion([J[s, i, : int(L[i])] for i in range(len(ids))])
        S = T @ M.T
        row = {}
        for name, query_ids in subset.items():
            qrows = []
            for k in query_ids:
                i = ids.index(k)
                neg = ([j for j in range(len(ids)) if j != i] if groups is None
                       else [j for j in range(len(ids)) if groups[j] != groups[i]])
                if len(neg) < 31:
                    continue
                rr = []
                for _ in range(args.repeats):
                    cols = np.r_[i, rng.choice(neg, 31, replace=False)]
                    v = S[i, cols]
                    rr.append(1 + (v > v[0]).sum() + ((v == v[0]).sum() - 1) / 2)
                rr = np.asarray(rr)
                qrows.append({f"R@{q}": float((rr <= q).mean()) for q in (1, 3, 10)}
                             | {"rank": float(rr.mean())})
            row[name] = {m: float(np.mean([x[m] for x in qrows])) for m in ("R@1", "R@3", "R@10")}
            row[name]["mean_rank"] = float(np.mean([x["rank"] for x in qrows]))
            row[name]["queries"] = len(qrows)
        per_sample.append(row)

    result = {
        "schema": "egoreal-baseline-r/v1",
        "release": str(args.release),
        "split": args.split,
        "npz": str(args.npz),
        "samples": int(J.shape[0]),
        "rows_total": len(ids_all),
        "rows_labeled": len(ids),
        "excluded_placeholder": len(ids_all) - len(ids),
        "protocol": ("E2 fixed; " + ("source-excluded" if groups else "random") +
                     " 32-pool; 20 pools/query; seed0; m2t; per-sample"),
        "all": {m: float(np.mean([x["all"][m] for x in per_sample]))
                for m in ("R@1", "R@3", "R@10", "mean_rank")},
        "subsets": {k: len(v) for k, v in subset.items()},
        "per_sample": per_sample,
    }
    for name in ("seen", "unseen"):
        if name in subset:
            result[name] = {m: float(np.mean([x[name][m] for x in per_sample]))
                            for m in ("R@1", "R@3", "R@10", "mean_rank")}
    for name in [k for k in subset if k.startswith("cat_") or k.startswith("unseen_cat_")]:
        result[name] = {m: float(np.mean([x[name][m] for x in per_sample]))
                        for m in ("R@1", "R@3", "R@10", "mean_rank")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("split", "samples", "all")}, indent=2))


if __name__ == "__main__":
    main()
