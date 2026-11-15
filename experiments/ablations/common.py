"""Shared plumbing for RIVEN ablation experiments.

The RIVEN mainline trainer/generator already exist and are validated:

  train  src/riven/model/h1_6q_train.py
  gen    src/riven/model/h1_6q_gen_condswap.py
  VQ     $RIVEN_CACHE/vq/best.pt (trained with vq_train.py on the same release)
  init   optional HML3D head init ($RIVEN_INIT_H1); main_scratch trains from scratch

This package only adds:
  * a registry of ablation definitions (exact flag overrides),
  * a data/cache readiness check per release,
  * one runner for train / generate / eval / smoke on any data version.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RIVEN_ROOT", Path(__file__).resolve().parents[2]))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from release import Release, STATE_ROOT  # noqa: E402  (schemas only; no sample content)

SRC = ROOT / "src/riven"
CODE = SRC / "model"
TRAIN = CODE / "h1_6q_train.py"
VQ_TRAIN = CODE / "vq_train.py"
GEN = CODE / "h1_6q_gen_condswap.py"
VISUAL_CACHE = SRC / "features/cache_visual_hidden.py"
EVAL_CLEAN = SRC / "eval/clean/eval_clean.py"
EVAL_LEGACY_R = SRC / "eval/legacy_e2/eval_r.py"
EVAL_LEGACY_FID = SRC / "eval/legacy_e2/eval_fid.py"
CLEAN_EVAL_ROOT = Path(os.environ.get("RIVEN_CLEAN_EVAL_ROOT", STATE_ROOT / "clean_evaluators"))
CLEAN_EVAL_DATA = Path(os.environ.get("RIVEN_CLEAN_EVAL_DATA", STATE_ROOT / "clean_eval_data"))
EVAL_SEEDS = os.environ.get("RIVEN_EVAL_SEEDS", "20261111,20261112,20261113")

PY = os.environ.get("RIVEN_PY", sys.executable)
PY_EVAL = os.environ.get("RIVEN_PY_EVAL", sys.executable)

ABL_STATE = Path(os.environ.get("RIVEN_ABL_STATE", STATE_ROOT / "ablations"))
DEFAULT_RELEASE = Path(os.environ.get("RIVEN_DATA", ROOT / "data/egoreal"))
DEFAULT_VQ = Path(os.environ.get("RIVEN_VQ", ROOT / "cache/vq/best.pt"))
DEFAULT_INIT = Path(os.environ.get("RIVEN_INIT_H1", ROOT / "state/init_head.pt"))
DEFAULT_MAIN_CKPT = Path(os.environ.get("RIVEN_MAIN_CKPT", ROOT / "state/main.pt"))
DEFAULT_CACHE_SRC = Path(os.environ.get("RIVEN_CACHE_SRC", ROOT / "cache"))
DURABLE_TEXTFEAT = Path(os.environ.get("RIVEN_TEXTFEAT", ROOT / "cache/textfeat"))
DATASET_ROOT = Path(os.environ.get("RIVEN_DATASET", STATE_ROOT / "dataset"))


def version_of(release: Path) -> str:
    return Path(release).name


def data_root(release: Path) -> Path:
    return ABL_STATE / "data" / version_of(release)


def cache_root(release: Path) -> Path:
    """Condition-feature root for a release.

    The paper main table uses the base cache built from the SFT epoch selected
    by VAL NLL (epoch2 on the unseen split), so ``RIVEN_CACHE_SUFFIX`` stays
    empty.  ``_sft_ep1`` selects the sibling epoch1 cache used by the earlier
    v4-split experiments and is not the paper configuration.
    """
    suffix = os.environ.get("RIVEN_CACHE_SUFFIX", "")
    return ABL_STATE / "cache" / (version_of(release) + suffix)


def run_root(release: Path, name: str, tag: str | None = None) -> Path:
    base = ABL_STATE / "runs" / version_of(release)
    return (base / tag / name) if tag else (base / name)


def clean_env(release: Path, gpu: int) -> dict:
    env = os.environ.copy()
    for key in ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", OMP_NUM_THREADS="8", PYTHONUNBUFFERED="1",
               RIVEN_DATA=str(release), RIVEN_CACHE=str(cache_root(release)),
               UMRM_DATA=str(release), UMRM_CACHE=str(cache_root(release)),
               CUDA_VISIBLE_DEVICES=str(gpu))
    return env


def run(cmd: list, release: Path, gpu: int, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write("$ " + " ".join(map(str, cmd)) + "\n")
        stream.flush()
        return subprocess.Popen([str(c) for c in cmd], env=clean_env(release, gpu),
                                stdout=stream, stderr=subprocess.STDOUT).wait()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def link(source: Path, target: Path) -> None:
    source, target = Path(source).resolve(), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.is_symlink() and target.resolve() == source:
            return
        raise RuntimeError(f"refusing to replace {target}")
    target.symlink_to(source)


def ensure_cache(release: Path, cache_src: Path | None = None) -> dict:
    """Materialise visualfeat + textfeat on durable storage and build TC caches."""
    cache = cache_root(release)
    cache.mkdir(parents=True, exist_ok=True)
    src = cache_src or DEFAULT_CACHE_SRC
    visual = cache / "visualfeat"
    text = cache / "textfeat"
    if not visual.is_dir() or not any(visual.glob("visual_trainval.shard*.npz")):
        if not (src / "visualfeat").is_dir():
            raise FileNotFoundError(f"visualfeat missing; provide --cache-src ({src})")
        if visual.exists():
            shutil.rmtree(visual)
        shutil.copytree(src / "visualfeat", visual)
    if not text.is_dir() or not any(text.glob("a_gt3p_trainval.feats.npy")):
        if (src / "textfeat").is_dir():
            if text.exists():
                shutil.rmtree(text)
            shutil.copytree(src / "textfeat", text)
        elif DURABLE_TEXTFEAT.is_dir():
            link(DURABLE_TEXTFEAT, text)
        else:
            raise FileNotFoundError("textfeat missing; provide --cache-src")
    tc = {split: cache / f"tcclip_{split}.npz" for split in ("train", "val")}
    return {"cache": cache, "visualfeat": visual, "textfeat": text, "tc": tc}


def build_tc_from_dataset(release: Path) -> None:
    """Split the shared TC-CLIP condition cache into train/val npz for the TC control."""
    dataset = DATASET_ROOT / version_of(release)
    cond = dataset / "cond/tcclip.npz"
    if not cond.is_file():
        raise FileNotFoundError(cond)
    import numpy as np
    split = json.loads((release / "split.json").read_text())
    groups = {"train": [str(x) for x in split["train_real"]],
              "val": [str(x) for x in split["val"]]}
    with np.load(cond) as z:
        ids = [str(x) for x in z["ids"]]
        row = {k: i for i, k in enumerate(ids)}
        g, m = z["g"], z["m"]
    cache = cache_root(release)
    cache.mkdir(parents=True, exist_ok=True)
    for name, keys in groups.items():
        idx = [row[k] for k in keys]
        path = cache / f"tcclip_{name}.npz"
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, ids=np.array(keys), g=g[idx], m=m[idx])
        tmp.replace(path)
