#!/usr/bin/env python3
"""Cache prompt-end VLM hidden states for E2-A.

Input : pre-reaction video + PROMPT_PLAN_EXEC (no answer tokens)
Output: one shard npz per process, `ids [N]`, `feats [N, K, 4, 2560]` fp16.
        Merge with merge_prompt_hidden.py into a textfeat-compatible corpus
        `<name>.feats.npy` / `<name>.meta.json` that head.py's TextFeats can read.

Run in the qwen env (qwen35_532), one process per GPU:
  CUDA_VISIBLE_DEVICES=6 python cache_prompt_hidden.py --shard 0/4
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("RIVEN_ROOT", Path(__file__).resolve().parents[3]))
PIPE = Path(os.environ.get("RIVEN_SFT_DIR", Path(__file__).resolve().parents[1] / "sft"))
CACHE = Path(os.environ.get("RIVEN_CACHE", ROOT / "cache"))
DEFAULT_DATA = Path(os.environ.get("RIVEN_DATA", ROOT / "data/egoreal"))
DEFAULT_ADAPTER = os.environ.get("RIVEN_SFT_ADAPTER", "")
DEFAULT_BASE = os.environ.get("RIVEN_QWEN", str(ROOT / "models/Qwen3-VL-4B-Instruct"))
DEFAULT_VIDEO_CACHE = os.environ.get("RIVEN_VIDEO_CACHE", str(CACHE / "video"))
DEFAULT_OUT = os.environ.get("RIVEN_TEXTFEAT", str(CACHE / "textfeat"))


def parse_ids(path: Path, splits: str):
    sp = json.loads(path.read_text())
    ids = []
    for s in splits.split(","):
        ids += [str(k) for k in sp[s]]
    assert len(ids) == len(set(ids)), "duplicate ids across requested splits"
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER), help="SFT LoRA adapter; empty string = base model")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--video-cache", default=DEFAULT_VIDEO_CACHE, help="dir containing video/<id>.pt (VHO_CACHE)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--name", default="a_gt3p_trainval")
    ap.add_argument("--splits", default="train_real,val")
    ap.add_argument("--k", type=int, default=16, help="number of trailing prompt tokens to keep")
    ap.add_argument("--layers", default="9,18,27,36")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    si, sn = map(int, a.shard.split("/"))
    layers = tuple(int(x) for x in a.layers.split(","))
    ids = parse_ids(Path(a.data) / "split.json", a.splits)[si::sn]
    if a.limit:
        ids = ids[:a.limit]
    if not ids:
        raise SystemExit("empty shard")

    os.environ["VHO_CACHE"] = a.video_cache
    os.environ.setdefault("VHO_DATA", str(a.data))
    sys.path.insert(0, str(PIPE))
    from common import PROMPT_PLAN_EXEC, PROMPT_SHA, active_prompt  # noqa: E402
    import vt_io  # noqa: E402

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = vt_io.load_model(adapter=(a.adapter or None), merge=True, base=a.base, device=a.device)

    feats, done = [], []
    t0 = time.time()
    for i, k in enumerate(ids):
        PROMPT_TEXT = PROMPT_PLAN_EXEC if os.environ.get("VHO_PROMPT_MODE", "plan_exec") == "plan_exec" else active_prompt()
        b = vt_io.encode(k, PROMPT_TEXT, answer=None)
        with torch.inference_mode():
            out = model(**vt_io.to_dev(b, a.device), output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
        if i == 0:
            print("hidden_states", len(hs), tuple(hs[0].shape), tuple(hs[-1].shape), "layers", layers, flush=True)
        H = torch.stack([hs[l][0, -a.k:, :] for l in layers], dim=1)  # [K, nL, 2560]
        feats.append(H.to(torch.float16).cpu().numpy())
        done.append(k)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(ids)} {round(time.time() - t0)}s", flush=True)

    out = out_dir / f"{a.name}.shard{si}.npz"
    np.savez(out, ids=np.array(done), feats=np.stack(feats))
    (out.with_suffix(".json")).write_text(json.dumps({
        "name": a.name, "shard": a.shard, "n": len(done), "k": a.k, "layers": layers,
        "adapter": a.adapter, "base": a.base, "prompt_sha256": PROMPT_SHA,
        "video_cache": a.video_cache, "elapsed_s": round(time.time() - t0),
    }, indent=1))
    print("saved", out, np.stack(feats).shape, flush=True)


if __name__ == "__main__":
    main()
