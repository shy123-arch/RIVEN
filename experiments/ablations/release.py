"""Shared plumbing for the EgoReReal main-table baseline harness.

The harness keeps every method on the same, versioned data contract:

    release/<version>/
      split.json                 # train_real / val / test (+ source_key, subsets)
      data_manifest.json         # rows[id] -> {exec, category, source, clip, ...}
      motion/humanml263_full.npz # ids, features [N,208,263], lengths, valid
      motion/gt22_full.npz       # J_<id>  -> [T,22,3] canonical GT

`build_dataset.py` turns that release into `HumanML3D`-style files shared by all
upstream codebases, plus one frozen TC-CLIP condition cache.  Method adapters
never read the raw release; they read the built dataset only.

No function in this module inspects real sample content: only schemas, ids,
lengths and counts are used.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("RIVEN_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_RELEASE = Path(os.environ.get("RIVEN_DATA", ROOT / "data/egoreal"))
STATE_ROOT = Path(os.environ.get("RIVEN_STATE", ROOT / "state"))
HARNESS_STATE = STATE_ROOT / "baselines"

SPLITS = ("train", "val", "test")
FEAT_DIM = 263
MAX_FRAMES = 208
FPS = 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(data)
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    atomic_write(path, text.encode("utf-8"))


def atomic_json(path: Path, obj) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    with tmp.open("wb") as stream:
        np.save(stream, array)
    os.replace(tmp, path)


def model_length(actual: int) -> int:
    """HERO/MoMask quantize motion length to a multiple of 4 (>>= 2 tokens)."""
    return (int(actual) + 3) // 4 * 4


@dataclass(frozen=True)
class Release:
    """Validated, schema-only view of one EgoReReal data release."""

    root: Path

    @property
    def split_json(self) -> dict:
        return json.loads((self.root / "split.json").read_text())

    @property
    def manifest(self) -> dict:
        return json.loads((self.root / "data_manifest.json").read_text())

    def splits(self) -> dict[str, list[str]]:
        split = self.split_json
        return {
            "train": [str(x) for x in split["train_real"]],
            "val": [str(x) for x in split["val"]],
            "test": [str(x) for x in split["test"]],
        }

    def motions(self):
        return np.load(self.root / "motion/humanml263_full.npz")

    def gt22(self):
        return np.load(self.root / "motion/gt22_full.npz")

    def validate(self) -> dict:
        splits = self.splits()
        ids = [i for name in SPLITS for i in splits[name]]
        if len(ids) != len(set(ids)):
            raise ValueError("split membership is not disjoint")
        with self.motions() as z:
            motion_ids = [str(x) for x in z["ids"]]
            features = z["features"]
            lengths = z["lengths"]
        if set(ids) != set(motion_ids):
            raise ValueError("split ids and motion ids differ")
        index = {k: i for i, k in enumerate(motion_ids)}
        width = int(features.shape[1])
        if not 4 <= width <= MAX_FRAMES:
            raise ValueError(f"invalid feature width {width}")
        rows = self.manifest["rows"]
        missing = [k for k in ids if k not in rows]
        if missing:
            raise ValueError(f"manifest missing {len(missing)} ids")
        for k in ids:
            if not 4 <= int(lengths[index[k]]) <= width:
                raise ValueError(f"invalid motion length for one id")
            if features[index[k]].shape != (width, FEAT_DIM):
                raise ValueError("invalid motion feature schema")
        return {
            "release": str(self.root),
            "counts": {name: len(v) for name, v in splits.items()},
            "motion_entries": len(motion_ids),
            "feature_width": int(features.shape[1]),
            "split_sha256": self.split_json.get("split_sha256"),
        }


def load_split(release: Release) -> dict:
    return release.splits()


def train_mean_std(release: Release) -> tuple[np.ndarray, np.ndarray]:
    """Frame-wise mean/std over TRAIN real frames (pre-padding).

    Same convention as HumanML3D: statistics over real frames only; the
    `feat_bias` rescaling of the first 4 / last 4 dims happens inside each
    upstream dataset class, so these files stay unbiased.
    """
    split = release.splits()
    with release.motions() as z:
        ids = [str(x) for x in z["ids"]]
        index = {k: i for i, k in enumerate(ids)}
        features = z["features"]
        lengths = z["lengths"]
        frames = np.concatenate(
            [features[index[k], : int(lengths[index[k]])] for k in split["train"]],
            axis=0,
        )
    if frames.shape[0] < 1000 or frames.shape[1] != FEAT_DIM:
        raise ValueError("invalid TRAIN frame matrix")
    return frames.mean(axis=0).astype(np.float32), frames.std(axis=0).astype(np.float32)


def condition_cache_path(dataset_dir: Path) -> Path:
    return Path(dataset_dir) / "cond/tcclip.npz"


def load_conditions(dataset_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    path = condition_cache_path(dataset_dir)
    if not path.is_file():
        raise FileNotFoundError(f"run build_dataset.py first: {path}")
    with np.load(path) as z:
        ids = [str(x) for x in z["ids"]]
        g, m = z["g"], z["m"]
    if g.shape != (len(ids), 512) or m.shape != (len(ids), 16, 512):
        raise ValueError("invalid TC-CLIP condition schema")
    return {k: (g[i], m[i]) for i, k in enumerate(ids)}


def split_manifest(dataset_dir: Path) -> dict:
    meta = json.loads((Path(dataset_dir) / "meta.json").read_text())
    return meta


def method_state(method: str, version: str) -> Path:
    return HARNESS_STATE / "runs" / version / method


def default_version(release: Release) -> str:
    return release.root.name
