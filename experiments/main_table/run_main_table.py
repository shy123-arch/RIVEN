#!/usr/bin/env python3
"""Per-seed paper main-table chain: tokenizer -> 200-epoch head -> select -> TEST.

Prerequisites (not run here):
  * release mounted at RIVEN_DATA
  * Qwen SFT adapter and text/visual caches under RIVEN_CACHE
  * clean evaluator bundle (RIVEN_CLEAN_EVAL_ROOT / RIVEN_CLEAN_EVAL_DATA)

    python run_main_table.py --gpu 0 --seeds 20261101
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
from common import (PY, VQ_TRAIN, clean_env, data_root, run_root)

STEPS = 11500
TAG = "unseen200_s{seed}"


def call(cmd, release, gpu, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        rc = subprocess.call([str(c) for c in cmd], env=clean_env(release, gpu),
                             stdout=f, stderr=subprocess.STDOUT)
    if rc:
        raise SystemExit(f"failed ({rc}): {' '.join(map(str, cmd))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, default=Path(os.environ.get("RIVEN_DATA", ROOT / "data/egoreal")))
    ap.add_argument("--seeds", default="20261101,20261102,20261103")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--checkpoints", default="9000,10500,last")
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()

    for seed in [s.strip() for s in a.seeds.split(",") if s.strip()]:
        tag = TAG.format(seed=seed)
        vq = data_root(a.release) / f"vq6q_s{seed}/best.pt"
        logs = run_root(a.release, "main_scratch", tag) / "logs"
        if not vq.is_file():
            call([PY, VQ_TRAIN, "--data", a.release, "--quantizers", "6", "--steps", "3000",
                  "--window-max", "208", "--seed", seed, "--out", vq.parent],
                 a.release, a.gpu, logs / "vq.log")
        if not a.skip_train:
            call([PY, ROOT / "experiments/ablations/run.py", "train", "--ablation", "main_scratch",
                  "--release", a.release, "--gpu", a.gpu, "--steps", a.steps,
                  "--seed", seed, "--tag", tag, "--vq", vq],
                 a.release, a.gpu, logs / "train.log")
        call([PY, HERE / "select_test.py", "--release", a.release, "--tag", tag,
              "--gpu", a.gpu, "--samples", a.samples, "--checkpoints", a.checkpoints],
             a.release, a.gpu, logs / "select_test.log")
        print(json.dumps({"seed": seed, "tag": tag,
                          "selection": str(run_root(a.release, "main_scratch", tag) / "selection.json")}), flush=True)


if __name__ == "__main__":
    main()
