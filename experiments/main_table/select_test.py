#!/usr/bin/env python3
"""Paper main-table checkpoint selection and TEST run for one model seed.

Sweeps VAL checkpoints with the clean evaluator, selects by VAL R@10
(FID tie-break), then generates and evaluates TEST from the selected
checkpoint.  This mirrors the 200-epoch protocol used for the paper table.

    python select_test.py --tag unseen200_s20261101 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("RIVEN_ROOT", HERE.parents[2]))
sys.path.insert(0, str(ROOT / "experiments/ablations"))
from common import (ABL_STATE, CLEAN_EVAL_DATA, CLEAN_EVAL_ROOT, EVAL_CLEAN, EVAL_SEEDS,
                    PY, PY_EVAL, clean_env, run_root)

ABLATION = "main_scratch"


def ckpt_path(ckpt_dir: Path, label: str) -> Path:
    return ckpt_dir / ("last.pt" if label == "last" else f"step_{label}.pt")


def clean_eval(npz: Path, out: Path, release: Path, split: str, seeds: str, gpu: int) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY_EVAL, str(EVAL_CLEAN), "--spec", f"{ABLATION}={npz}", "--split", split,
           "--release", str(release), "--data", str(CLEAN_EVAL_DATA),
           "--eval-root", str(CLEAN_EVAL_ROOT), "--eval-seeds", seeds, "--out", str(out)]
    rc = subprocess.call(cmd, env=clean_env(release, gpu))
    if rc:
        raise SystemExit(f"eval failed ({rc}): {split} {npz}")
    return json.loads(out.read_text())["specs"][ABLATION]


def generate(tag: str, split: str, ckpt: Path, release: Path, samples: int, gpu: int) -> Path:
    run_py = ROOT / "experiments/ablations/run.py"
    cmd = [PY, str(run_py), "generate", "--ablation", ABLATION, "--release", str(release),
           "--tag", tag, "--gpu", str(gpu), "--samples", str(samples), "--split", split,
           "--ckpt", str(ckpt)]
    rc = subprocess.call(cmd, env=clean_env(release, gpu))
    if rc:
        raise SystemExit(f"generate failed ({rc}): {split} {ckpt}")
    return run_root(Path(release), ABLATION, tag) / f"gen/{split}_S{samples}.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, default=Path(os.environ.get("RIVEN_DATA", ROOT / "data/egoreal")))
    ap.add_argument("--tag", required=True, help="training tag, e.g. unseen200_s20261101")
    ap.add_argument("--ckpt-dir", type=Path, default=None)
    ap.add_argument("--checkpoints", default="9000,10500,last")
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--eval-seeds", default=EVAL_SEEDS)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    ckpt_dir = a.ckpt_dir or (run_root(a.release, ABLATION, a.tag) / "train")
    labels = [x.strip() for x in a.checkpoints.split(",") if x.strip()]
    eval_dir = run_root(a.release, ABLATION, a.tag) / "eval"
    evals = {}
    for label in labels:
        ck = ckpt_path(ckpt_dir, label)
        if not ck.is_file():
            print(json.dumps({"skip": str(ck), "reason": "missing"}), flush=True)
            continue
        val_tag = f"{a.tag}__ckpt{label}"
        npz = generate(val_tag, "val", ck, a.release, a.samples, a.gpu)
        r = clean_eval(npz, eval_dir / f"val_{label}_S{a.samples}.clean.json",
                       a.release, "val", a.eval_seeds, a.gpu)
        evals[label] = {"ckpt": str(ck), "val_R": r["R"], "val_fid": r["fid"]}
        print(json.dumps({"label": label, **evals[label]}), flush=True)

    if not evals:
        raise SystemExit("no checkpoints evaluated")
    # Paper rule: keep checkpoints within 0.5pp of the best VAL R@10, then
    # take the lowest VAL FID among them.
    best_r = max(v["val_R"]["R@10"] for v in evals.values())
    near = [k for k, v in evals.items() if best_r - v["val_R"]["R@10"] <= 0.005]
    best = min(near, key=lambda k: evals[k]["val_fid"])
    sel_ckpt = Path(evals[best]["ckpt"])
    test_tag = f"{a.tag}__sel"
    npz = generate(test_tag, "test", sel_ckpt, a.release, a.samples, a.gpu)
    test = clean_eval(npz, eval_dir / f"test_S{a.samples}.clean.json",
                      a.release, "test", a.eval_seeds, a.gpu)
    result = {"schema": "riven-selection/v1", "tag": a.tag, "selected": best,
              "rule": "VAL R@10 within 0.5pp of best, then min VAL FID",
              "candidates": near, "evals": evals, "test": {"npz": str(npz), **test}}
    out = run_root(a.release, ABLATION, a.tag) / "selection.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected": best, "test_fid": test["fid"], **test["R"]}, indent=1))


if __name__ == "__main__":
    main()
