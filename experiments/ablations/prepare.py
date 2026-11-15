#!/usr/bin/env python3
"""Check (and where possible materialise) everything an ablation run needs.

Usage:
  python prepare.py --release <release dir> [--cache-src <cache dir>] [--build-tc]
                    [--vq <ckpt>] [--init-h1 <ckpt>] [--link-main-ckpt <ckpt>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (CLEAN_EVAL_DATA, CLEAN_EVAL_ROOT, DEFAULT_INIT, DEFAULT_RELEASE, DEFAULT_VQ,
                    EVAL_SEEDS, Release, atomic_json, build_tc_from_dataset,
                    cache_root, data_root, ensure_cache, link, run_root, version_of)
from registry import ABLATIONS, TRAIN_BASE

REQUIRED_VISUAL = ("visual_trainval",)
REQUIRED_TEXT = ("a_gt3p_trainval",)
OPTIONAL_VISUAL = ("visual_blank_trainval", "visual_shuffle_trainval")
OPTIONAL_TEXT = ("blank_gt3p_trainval", "shuffle_gt3p_trainval")


def check(release: Path, cache_src: Path | None, build_tc: bool, vq: Path, init: Path,
          main_ckpt: Path | None) -> dict:
    info = Release(release).validate()
    cache = ensure_cache(release, cache_src)
    if build_tc and not (cache["cache"] / "tcclip_train.npz").is_file():
        build_tc_from_dataset(release)
        cache["tc"] = {s: cache["cache"] / f"tcclip_{s}.npz" for s in ("train", "val")}
    problems = []
    for name in REQUIRED_VISUAL:
        shards = sorted(cache["visualfeat"].glob(f"{name}.shard*.npz"))
        if not shards:
            problems.append(f"missing visual shards: {name}")
    for name in REQUIRED_TEXT:
        if not (cache["textfeat"] / f"{name}.feats.npy").is_file():
            problems.append(f"missing textfeat: {name}")
    warnings = []
    for name in OPTIONAL_VISUAL:
        if not sorted(cache["visualfeat"].glob(f"{name}.shard*.npz")):
            warnings.append(f"control ablation unavailable: visual {name}")
    for name in OPTIONAL_TEXT:
        if not (cache["textfeat"] / f"{name}.feats.npy").is_file():
            warnings.append(f"control ablation unavailable: textfeat {name}")
    if vq and not Path(vq).is_file():
        problems.append(f"missing vq: {vq}")
    if init and not Path(init).is_file():
        warnings.append(f"head init absent (fine for main_scratch): {init}")
    for split in ("train", "val"):
        if not (cache["cache"] / f"tcclip_{split}.npz").is_file():
            warnings.append(f"TC-CLIP control unavailable for split: {split}")
    counts = info["counts"]
    if counts.get("val") != 213 or counts.get("test") != 213 or counts.get("train", 0) < 1000:
        problems.append(f"unexpected split counts: {counts}")
    for seed in [s for s in EVAL_SEEDS.split(",") if s]:
        if not (CLEAN_EVAL_ROOT / f"evaluator_hml3d_clean_s{seed}/best.tar").is_file():
            problems.append(f"missing clean evaluator seed {seed}")
    for rel in ("meta/mean.npy", "meta/std.npy", "meta/our_vab_data.npy",
                "meta/our_vab_words.pkl", "meta/our_vab_idx.pkl",
                "train_ids.txt", "val_ids.txt", "test_ids.txt"):
        if not (CLEAN_EVAL_DATA / rel).is_file():
            problems.append(f"missing clean eval data: {rel}")
    out = run_root(release, "main")
    out.mkdir(parents=True, exist_ok=True)
    if main_ckpt is not None:
        link(main_ckpt, out / "base.pt")
    ready = {
        "version": version_of(release),
        "release": str(release),
        "counts": info["counts"],
        "cache": str(cache["cache"]),
        "clean_evaluators": str(CLEAN_EVAL_ROOT),
        "clean_eval_data": str(CLEAN_EVAL_DATA),
        "eval_seeds": EVAL_SEEDS,
        "vq": str(vq),
        "init_h1": str(init) if init else None,
        "main_ckpt": str(main_ckpt) if main_ckpt else None,
        "ablations": sorted(ABLATIONS),
        "problems": problems,
        "warnings": warnings,
        "ready": not problems,
        "note": "visualfeat/textfeat for a NEW release must be regenerated with Qwen; "
                "VQ must be retrained with vq_train.py on that release.",
    }
    atomic_json(data_root(release) / "ready.json", ready)
    return ready


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    ap.add_argument("--cache-src", type=Path, default=None)
    ap.add_argument("--build-tc", action="store_true", default=True)
    ap.add_argument("--vq", type=Path, default=DEFAULT_VQ)
    ap.add_argument("--init-h1", type=Path, default=None,
                    help="optional HML3D-prepared head init; main_scratch does not use it")
    ap.add_argument("--link-main-ckpt", type=Path, default=None)
    args = ap.parse_args()
    ready = check(args.release, args.cache_src, args.build_tc, args.vq, args.init_h1,
                  args.link_main_ckpt)
    print(json.dumps(ready, indent=2))
    if ready["problems"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
