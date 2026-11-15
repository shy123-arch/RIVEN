#!/usr/bin/env python3
"""Static checks for the RIVEN repository.  No data, GPU or torch required.

    python tools/check_repository.py

Exits non-zero on the first category of failure.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABS_PATH = re.compile(r"/mnt/|/tmp/|/home/[^/]+/")
SECRETS = re.compile(r"api[_-]?key|cookie|password|secret|bearer\s", re.I)


def py_files(paths):
    for p in paths:
        yield from sorted(p.rglob("*.py"))


def check_compile():
    r = subprocess.run([sys.executable, "-m", "compileall", "-q", "src", "experiments", "tools"],
                       cwd=ROOT)
    return r.returncode == 0


def check_paths():
    bad = []
    for p in py_files([ROOT / "src", ROOT / "experiments", ROOT / "scripts"]):
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if ABS_PATH.search(line):
                bad.append(f"{p.relative_to(ROOT)}:{i}")
    return bad


def check_secrets():
    bad = []
    for p in py_files([ROOT / "src", ROOT / "experiments", ROOT / "scripts"]):
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if SECRETS.search(line):
                bad.append(f"{p.relative_to(ROOT)}:{i}")
    return bad


def argparse_options(path):
    tree = ast.parse(Path(path).read_text())
    opts = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument":
            if n.args and isinstance(n.args[0], ast.Constant):
                opts.add(n.args[0].value)
    return opts


def check_registry():
    sys.path.insert(0, str(ROOT / "experiments/ablations"))
    import common  # noqa: E402
    import registry  # noqa: E402
    train = argparse_options(common.TRAIN) | {"--steps", "--seed", "--out", "--visual-cache", "--repr-cache"}
    gen = argparse_options(common.GEN) | {"--ckpt", "--samples", "--limit", "--split", "--out",
                                          "--visual-cache", "--repr-cache", "--tc-val"}
    bad = []
    for name, arm in registry.ABLATIONS.items():
        bad += [f"{name}: train {f}" for f in arm.train_flags() if f not in train]
        bad += [f"{name}: gen {f}" for f in arm.gen_flags() if f not in gen]
    return bad, len(registry.ABLATIONS)


def check_required():
    required = [
        "README.md", "LICENSE", "NOTICE.md", "requirements.txt",
        "configs/paths.env.example", "configs/main.yaml",
        "docs/METHOD.md", "docs/REPRODUCE.md", "docs/PROVENANCE.md", "docs/SOURCE_SNAPSHOT.json",
        "patches/hero-gen.patch",
        "src/riven/model/h1_6q_train.py", "src/riven/model/h1_6q_gen_condswap.py",
        "src/riven/sft/prep_video_cache.py",
        "src/riven/eval/eval_r.py", "src/riven/eval/eval_fid.py",
        "experiments/ablations/registry.py", "experiments/ablations/run.py",
    ]
    return [p for p in required if not (ROOT / p).is_file()]


def main() -> int:
    ok = True

    if not check_compile():
        print("FAIL compile")
        ok = False

    for title, bad in (("absolute path", check_paths()), ("secret-like string", check_secrets())):
        if bad:
            ok = False
            print(f"FAIL {title} ({len(bad)}):")
            for b in bad:
                print("  ", b)

    missing = check_required()
    if missing:
        ok = False
        print("FAIL missing files:", missing)

    try:
        json.loads((ROOT / "docs/SOURCE_SNAPSHOT.json").read_text())
    except Exception as exc:  # noqa: BLE001
        ok = False
        print("FAIL SOURCE_SNAPSHOT.json:", exc)

    bad, n = check_registry()
    if bad:
        ok = False
        print(f"FAIL registry flags ({len(bad)}):")
        for b in bad:
            print("  ", b)
    else:
        print(f"ok: registry {n} arms match argparse")

    if ok:
        print("repository checks passed")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
