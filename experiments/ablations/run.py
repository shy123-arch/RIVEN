#!/usr/bin/env python3
"""One entry point for RIVEN ablation experiments.

    python run.py prepare  --release <dir> [--cache-src <cache dir>]
    python run.py status
    python run.py train    --ablation main     --gpu 5 --steps 3000 --seed 20261101
    python run.py train    --ablation perceiver --gpu 5
    python run.py generate --ablation main --gpu 5 --samples 15
    python run.py generate --ablation riv_off --gpu 5 --samples 15
    python run.py eval     --ablation main --split val --samples 15 --gpu 5
    python run.py smoke    --gpu 5 [--train-smoke]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (ABL_STATE, DEFAULT_MAIN_CKPT, DEFAULT_RELEASE, EVAL_FID, EVAL_R, GEN, PY,
                    PY_EVAL, TRAIN, VQ_TRAIN, atomic_json, cache_root, clean_env, data_root,
                    link, run, run_root, version_of)
from registry import ABLATIONS, INFERENCE, TRAIN as TRAIN_ABLATIONS


def render(flags: dict, specials: dict) -> list[str]:
    out = []
    for key, value in flags.items():
        if value is None:
            continue
        if value is True:
            out.append(key)
            continue
        value = specials.get(value, value)
        out += [key, str(value)]
    return out


def specials_for(release: Path, name: str) -> dict:
    cache = cache_root(release)
    return {
        "__TC_TRAIN__": cache / "tcclip_train.npz",
        "__TC_VAL__": cache / "tcclip_all.npz",
        "__VQ1Q__": data_root(release) / "vq1q/last.pt",
    }


def resolve_ckpt(release: Path, ablation: str, explicit: Path | None, tag: str | None = None) -> Path:
    if explicit is not None:
        return explicit
    own = run_root(release, ablation, tag) / "train/last.pt"
    if own.is_file():
        return own
    own_base = run_root(release, ablation, tag) / "base.pt"
    if own_base.is_file():
        return own_base
    main = run_root(release, "main", tag) / "train/last.pt"
    if main.is_file():
        return main
    main_base = run_root(release, "main", tag) / "base.pt"
    if main_base.is_file():
        return main_base
    if version_of(release) == "umrm_v1572_split_1104_20261104" and DEFAULT_MAIN_CKPT.is_file():
        return DEFAULT_MAIN_CKPT
    raise FileNotFoundError(
        f"no checkpoint for {ablation}; train main first or pass --ckpt")


def cmd_train(args, release: Path, ablation: str, tag: str | None = None) -> list:
    spec = ABLATIONS[ablation]
    flags = spec.train_flags()
    flags["--visual-cache"] = str(cache_root(release) / "visualfeat")
    flags["--repr-cache"] = str(cache_root(release) / "textfeat")
    ready_path = data_root(release) / "ready.json"
    if ready_path.is_file():
        ready = json.loads(ready_path.read_text())
        if ready.get("vq"):
            flags["--vq"] = ready["vq"]
        if flags.get("--init-h1") and ready.get("init_h1"):
            flags["--init-h1"] = ready["init_h1"]
    if args.vq is not None:
        flags["--vq"] = str(args.vq)
    out = run_root(release, ablation, tag) / "train"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, TRAIN] + render(flags, specials_for(release, ablation))
    cmd += ["--steps", str(args.steps), "--seed", str(args.seed), "--out", str(out)]
    return cmd


def cmd_generate(args, release: Path, ablation: str, ckpt: Path, tag: str | None = None) -> tuple[list, Path]:
    spec = ABLATIONS[ablation]
    flags = spec.gen_flags()
    flags["--visual-cache"] = str(cache_root(release) / "visualfeat")
    flags["--repr-cache"] = str(cache_root(release) / "textfeat")
    out = run_root(release, ablation, tag) / f"gen/{args.split}_S{args.samples}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, GEN, "--ckpt", str(ckpt)] + render(flags, specials_for(release, ablation))
    cmd += ["--samples", str(args.samples), "--limit", str(args.limit),
            "--split", args.split, "--out", str(out)]
    return cmd, out


def cmd_eval(args, release: Path, ablation: str, tag: str | None = None) -> int:
    npz = run_root(release, ablation, tag) / f"gen/{args.split}_S{args.samples}.npz"
    if not npz.is_file():
        raise FileNotFoundError(npz)
    ev = run_root(release, ablation, tag) / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    r = ev / f"{args.split}_S{args.samples}.r.json"
    f = ev / f"{args.split}_S{args.samples}.fid.json"
    rc = subprocess.call([PY_EVAL, str(EVAL_R), "--release", str(release), "--npz", str(npz),
                          "--split", args.split, "--out", str(r)], env=clean_env(release, args.gpu))
    if rc:
        return rc
    return subprocess.call([PY_EVAL, str(EVAL_FID), "--release", str(release),
                            "--model", f"{ablation}={npz}", "--split", args.split,
                            "--out", str(f)], env=clean_env(release, args.gpu))


def check_npz(path: Path) -> dict:
    import numpy as np
    with np.load(path) as z:
        j = z["val_joints"]
        return {"npz": str(path), "ids": int(len(z["val_ids"])), "joints": list(j.shape),
                "lengths": list(z["lengths"].shape), "finite": bool(np.isfinite(j).all())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "status", "train", "generate", "eval", "smoke"])
    ap.add_argument("--ablation", default="all")
    ap.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000,
                    help="training iterations (35/epoch -> 3000 = 88ep, 7000 = 200ep)")
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--cache-src", type=Path, default=None)
    ap.add_argument("--link-main-ckpt", type=Path, default=None)
    ap.add_argument("--vq", type=Path, default=None)
    ap.add_argument("--init-h1", type=Path, default=None)
    ap.add_argument("--train-smoke", action="store_true")
    ap.add_argument("--tag", default="", help="per-seed output tag")
    args = ap.parse_args()
    release = args.release

    if args.action == "prepare":
        cmd = [sys.executable, str(HERE / "prepare.py"), "--release", str(release)]
        if args.cache_src:
            cmd += ["--cache-src", str(args.cache_src)]
        if args.link_main_ckpt:
            cmd += ["--link-main-ckpt", str(args.link_main_ckpt)]
        if args.vq:
            cmd += ["--vq", str(args.vq)]
        if args.init_h1:
            cmd += ["--init-h1", str(args.init_h1)]
        subprocess.check_call(cmd)
        return

    if args.action == "status":
        for name in ["main", *INFERENCE[1:], *TRAIN_ABLATIONS]:
            root = run_root(release, name, args.tag or None)
            print(json.dumps({
                "ablation": name,
                "kind": ABLATIONS[name].kind,
                "trained": (root / "train/last.pt").is_file() or (root / "base.pt").is_file(),
                "gen": sorted(p.name for p in (root / "gen").glob("*.npz")) if (root / "gen").is_dir() else [],
            }))
        return

    names = ([args.ablation] if args.ablation != "all" else list(ABLATIONS))
    for name in names:
        if name not in ABLATIONS:
            raise SystemExit(f"unknown ablation {name}")
        if args.action == "train":
            guard = run_root(release, name, args.tag or None) / "train/last.pt"
            if guard.is_symlink():
                raise SystemExit(f"refusing to train: {guard} is a symlink to a read-only checkpoint")
            code = run(cmd_train(args, release, name, args.tag or None), release, args.gpu,
                       run_root(release, name, args.tag or None) / "logs/train.log")
            if code:
                raise SystemExit(f"train failed: {name} ({code})")
            print(json.dumps({"trained": name, "ckpt": str(run_root(release, name, args.tag or None) / "train/last.pt")}))
        elif args.action == "generate":
            ckpt = resolve_ckpt(release, name, args.ckpt, args.tag or None)
            cmd, out = cmd_generate(args, release, name, ckpt, args.tag or None)
            code = run(cmd, release, args.gpu, run_root(release, name, args.tag or None) / "logs/generate.log")
            if code:
                raise SystemExit(f"generate failed: {name} ({code})")
            print(json.dumps({"generated": name, **check_npz(out)}))
        elif args.action == "eval":
            code = cmd_eval(args, release, name, args.tag or None)
            if code:
                raise SystemExit(f"eval failed: {name} ({code})")
            print(json.dumps({"evaluated": name}))
        elif args.action == "smoke":
            args.limit = args.limit or 1
            args.samples = 1
            if args.train_smoke and (ABLATIONS[name].kind == "train" or name == "main"):
                args.steps = min(args.steps, 20)
                code = run(cmd_train(args, release, name, args.tag or None), release, args.gpu,
                           run_root(release, name, args.tag or None) / "logs/smoke_train.log")
                if code:
                    raise SystemExit(f"smoke train failed: {name} ({code})")
                print(json.dumps({"smoke_train_ok": name,
                                  "ckpt": str(run_root(release, name, args.tag or None) / "train/last.pt")}))
            elif ABLATIONS[name].kind == "inference":
                pass
            else:
                continue
            if ABLATIONS[name].kind == "inference" and name != "main" and args.train_smoke:
                continue
            ckpt = resolve_ckpt(release, name, args.ckpt, args.tag or None)
            cmd, out = cmd_generate(args, release, name, ckpt, args.tag or None)
            code = run(cmd, release, args.gpu, run_root(release, name, args.tag or None) / "logs/smoke.log")
            if code:
                raise SystemExit(f"smoke generate failed: {name} ({code})")
            print(json.dumps({"smoke_ok": name, **check_npz(out)}))


if __name__ == "__main__":
    main()
