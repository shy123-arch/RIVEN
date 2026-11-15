#!/bin/bash
# Dependency-free smoke test: syntax + import graph of the parts that do not
# need torch, HERO, Qwen, data or a GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m compileall -q src experiments
python tools/check_repository.py
python - <<'PY'
import sys
sys.path.insert(0, 'experiments/ablations')
import common, registry
assert 'main_scratch' in registry.ABLATIONS
assert registry.ABLATIONS['main_scratch'].kind == 'inference'
print('ok: ablations common + registry (%d entries)' % len(registry.ABLATIONS))

sys.path.insert(0, 'src/riven/sft')
import importlib.util
spec = importlib.util.spec_from_file_location('sft_common', 'src/riven/sft/common.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert m.PROMPT_PLAN_EXEC.startswith('This first-person video')
print('ok: SFT prompt module')
PY

echo "smoke ok"
