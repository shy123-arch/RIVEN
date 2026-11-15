#!/usr/bin/env python3
"""Build a textfeat-format corpus from the frozen-Qwen (no LoRA) p16 cache.

The main generator reads CACHE/textfeat/<name>.{feats.npy,meta.json}.  This
script converts autoresearch/cache/base.shard*.npz (p16: last 16 prompt
positions, 4 layers, 2560-D) into that format so the same training recipe can
be run with base-Qwen features instead of the v1572 SFT LoRA features.

Aggregate metadata only; no sample content is printed.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('RIVEN_ROOT', Path(__file__).resolve().parents[3]))
EXP = Path(__file__).resolve().parent
CACHE = Path(os.environ.get('RIVEN_CACHE', ROOT / 'cache'))
SRC = CACHE / 'textfeat'
NAME = 'base_gt3p_trainval'
SFT_META = CACHE / 'textfeat/a_gt3p_trainval.meta.json'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    meta = json.loads(SFT_META.read_text())
    items = meta['items']
    n = len(items)
    ids = [str(it['id']) for it in items]
    if len(set(ids)) != n:
        raise ValueError('duplicate ids in SFT meta')

    shards = sorted(SRC.glob('base.shard*.npz'))
    if not shards:
        raise FileNotFoundError('base p16 shards missing')
    feats_by_id = {}
    shard_info = []
    for path in shards:
        with np.load(path) as z:
            sid = [str(x) for x in z['ids']]
            p16 = z['p16']
            if p16.ndim != 4 or p16.shape[1:] != (16, 4, 2560):
                raise ValueError(f'p16 schema mismatch: {path} {p16.shape}')
            if len(sid) != p16.shape[0]:
                raise ValueError(f'id/p16 count mismatch: {path}')
            for i, k in enumerate(sid):
                if k in feats_by_id:
                    raise ValueError(f'duplicate id across shards: {k}')
                feats_by_id[k] = p16[i]
            shard_info.append({'path': str(path), 'sha256': sha256(path),
                               'items': len(sid), 'p16_shape': list(p16.shape),
                               'dtype': str(p16.dtype)})
    missing = [k for k in ids if k not in feats_by_id]
    extra = [k for k in feats_by_id if k not in set(ids)]
    if missing or extra:
        raise ValueError(f'id coverage mismatch: missing={len(missing)} extra={len(extra)}')

    lengths = [16] * n
    offsets = list(range(0, n * 16, 16))
    out = np.empty((n * 16, 4, 2560), np.float16)
    for i, k in enumerate(ids):
        out[offsets[i]:offsets[i] + 16] = feats_by_id[k]

    dest = CACHE / 'textfeat'
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / f'{NAME}.feats.npy', out)
    new_meta = {
        'layers': [9, 18, 27, 36],
        'k': 16,
        'offsets': offsets,
        'lengths': lengths,
        'items': items,
        'source': 'frozen Qwen3-VL-4B base (no LoRA), prompt-end last 16 positions',
        'shards': shard_info,
    }
    (dest / f'{NAME}.meta.json').write_text(json.dumps(new_meta, indent=2) + '\n')
    manifest = {
        'name': NAME,
        'items': n,
        'feats_shape': list(out.shape),
        'feats_dtype': str(out.dtype),
        'feats_sha256': sha256(dest / f'{NAME}.feats.npy'),
        'meta_sha256': sha256(dest / f'{NAME}.meta.json'),
        'shards': shard_info,
    }
    (dest / f'{NAME}.build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
