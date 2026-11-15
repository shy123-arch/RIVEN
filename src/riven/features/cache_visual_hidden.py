#!/usr/bin/env python3
"""Cache Qwen visual-token hidden states for the 1104 split.

For each video: find prefix_mm == 2 (visual tokens), split into 16 contiguous
segments, mean-pool each segment at layers 9/18/27/36.
Output: [K=16, nL=4, 2560] per item, compatible with a Perceiver adapter.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(os.environ.get('RIVEN_ROOT', Path(__file__).resolve().parents[3]))
PIPE = Path(os.environ.get('RIVEN_SFT_DIR', Path(__file__).resolve().parents[1] / 'sft'))
CACHE = Path(os.environ.get('RIVEN_CACHE', ROOT / 'cache'))
DEFAULT_DATA = Path(os.environ.get('RIVEN_DATA', ROOT / 'data/egoreal'))
DEFAULT_BASE = os.environ.get('RIVEN_QWEN', str(ROOT / 'models/Qwen3-VL-4B-Instruct'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(DEFAULT_DATA))
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--base', default=DEFAULT_BASE)
    ap.add_argument('--video-cache', default=os.environ.get('RIVEN_VIDEO_CACHE', str(CACHE / 'video')))
    ap.add_argument('--out-dir', default=os.environ.get('RIVEN_VISUALFEAT', str(CACHE / 'visualfeat')))
    ap.add_argument('--name', default='visual_trainval')
    ap.add_argument('--splits', default='train_real,val')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--layers', default='9,18,27,36')
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    si, sn = map(int, a.shard.split('/'))
    layers = tuple(int(x) for x in a.layers.split(','))
    sp = json.loads((Path(a.data) / 'split.json').read_text())
    ids = []
    for s in a.splits.split(','):
        ids += [str(x) for x in sp[s]]
    ids = ids[si::sn]
    os.environ['VHO_CACHE'] = a.video_cache
    os.environ.setdefault('VHO_DATA', a.data)
    sys.path.insert(0, str(PIPE))
    from common import PROMPT_PLAN_EXEC
    import vt_io
    model = vt_io.load_model(adapter=a.adapter, merge=True, base=a.base, device=a.device)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    feats, done = [], []
    t0 = time.time()
    for i, k in enumerate(ids):
        b = vt_io.encode(k, PROMPT_PLAN_EXEC, answer=None)
        with torch.inference_mode():
            out = model(**vt_io.to_dev(b, a.device), output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
        vmask = b['mm_token_type_ids'][0].bool()  # visual token type = 1? caller checks
        # Qwen video token type in this cache is 2; fall back to nonzero.
        idx = torch.where(b['mm_token_type_ids'][0] == 2)[0]
        if len(idx) == 0:
            idx = torch.where(b['mm_token_type_ids'][0] != 0)[0]
        # split visual tokens into K contiguous segments
        chunks = torch.chunk(idx, a.k)
        seg = []
        for c in chunks:
            seg.append(c)
        H = torch.zeros((a.k, len(layers), 2560), dtype=torch.float16)
        for li, l in enumerate(layers):
            h = hs[l][0]
            for j, c in enumerate(seg):
                if len(c):
                    H[j, li] = h[c].float().mean(0).to(torch.float16).cpu()
        feats.append(H.numpy()); done.append(k)
        if (i + 1) % 20 == 0:
            print(f'{i+1}/{len(ids)} {round(time.time()-t0)}s', flush=True)
    np.savez(out_dir / f'{a.name}.shard{si}.npz', ids=np.array(done), feats=np.stack(feats))
    (out_dir / f'{a.name}.shard{si}.json').write_text(json.dumps({
        'name': a.name, 'shard': a.shard, 'n': len(done), 'k': a.k, 'layers': layers,
        'adapter': a.adapter, 'base': a.base, 'video_cache': a.video_cache,
        'elapsed_s': round(time.time()-t0),
    }, indent=1))
    print('saved', out_dir / f'{a.name}.shard{si}.npz', np.stack(feats).shape)


if __name__ == '__main__':
    main()
