#!/usr/bin/env python3
"""Merge prompt-hidden shards into a textfeat-compatible corpus.

TextFeats (src/riven/sft/head.py in the upstream V->T pipeline) expects:
  <name>.feats.npy  fp16 [Ntok, nL, 2560]
  <name>.meta.json  {layers, offsets, lengths, items:[{id, tid}], texts}
Each item contributes K consecutive rows; tid indexes offsets/lengths.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("RIVEN_CACHE", Path(__file__).resolve().parents[3] / "cache"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.environ.get("RIVEN_TEXTFEAT", str(CACHE / "textfeat")))
    ap.add_argument("--name", default="a_gt3p_trainval")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--layers", default="9,18,27,36")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    layers = [int(x) for x in a.layers.split(",")]
    parts = sorted(out_dir.glob(f"{a.name}.shard*.npz"), key=lambda p: int(p.stem.split("shard")[-1]))
    assert parts, f"no shards for {a.name} in {out_dir}"
    ids, feats = [], []
    for p in parts:
        z = np.load(p)
        assert len(z["ids"]) == len(z["feats"])
        ids += [str(x) for x in z["ids"]]
        feats.append(z["feats"])
    assert len(ids) == len(set(ids)), "duplicate ids across shards"
    F = np.concatenate(feats, 0)  # [N, K, nL, 2560]
    assert F.shape[1] == a.k and F.shape[2] == len(layers) and F.dtype == np.float16, F.shape
    N, K, L, D = F.shape
    flat = F.reshape(N * K, L, D)
    lens = np.full(N, K, dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    mm = np.lib.format.open_memmap(out_dir / f"{a.name}.feats.npy", mode="w+", dtype=np.float16, shape=flat.shape)
    mm[:] = flat
    mm.flush()
    meta = {
        "layers": layers,
        "k": K,
        "texts": ids,
        "offsets": off.tolist(),
        "lengths": lens.tolist(),
        "items": [{"id": k, "tid": i} for i, k in enumerate(ids)],
        "source": "prompt-end hidden states (last K tokens of video+PROMPT_PLAN_EXEC+assistant header), SFT student, no answer tokens",
        "shards": [p.name for p in parts],
    }
    (out_dir / f"{a.name}.meta.json").write_text(json.dumps(meta, indent=1))
    print("merged", a.name, F.shape, "->", out_dir / f"{a.name}.feats.npy")


if __name__ == "__main__":
    main()
