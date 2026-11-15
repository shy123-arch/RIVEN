"""Build the EgoReReal evaluation corpus for the clean evaluator.

For every id in TRAIN/VAL/TEST:
  motion/<id>.npy   263-D official guoh3dfeats (T-1, 20 Hz)
  text/<id>.txt     "<caption>#<tokens>#nan#nan"
and
  train_ids.txt / val_ids.txt / test_ids.txt
  meta/{mean,std,our_vab_data,our_vab_words,our_vab_idx}

Word vectors come from GloVe 6B 300-D (RIVEN_GLOVE).  The paper's frozen
corpus is shipped as a release asset, so this script is only needed when
rebuilding it.

    RIVEN_DATA=<unseen release> RIVEN_GLOVE=... \
    RIVEN_CLEAN_EVAL_DATA=<out> python prepare.py
"""
import json
import os
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('RIVEN_ROOT', HERE.parents[4]))
REL = Path(os.environ.get('RIVEN_DATA', ROOT / 'data/egoreal'))
GT = Path(os.environ.get('RIVEN_GT22', REL / 'motion/gt22_full.npz'))
GLOVE = Path(os.environ.get('RIVEN_GLOVE', ROOT / 'data/glove/glove.6B.300d.txt'))
OUT = Path(os.environ.get('RIVEN_CLEAN_EVAL_DATA', ROOT / 'state/clean_eval_data'))

sys.path.insert(0, str(HERE.parent))  # tmr_adapter
from tmr_adapter import preprocess  # noqa: E402

man = json.loads((REL / 'data_manifest.json').read_text())['rows']
split = json.loads((REL / 'split.json').read_text())
splits = {
    'train': [str(x) for x in split['train_real']],
    'val': [str(x) for x in split['val']],
    'test': [str(x) for x in split['test']],
}
all_ids = splits['train'] + splits['val'] + splits['test']
train_set = set(splits['train'])
gt = np.load(GT)
(OUT / 'motion').mkdir(parents=True, exist_ok=True)
(OUT / 'text').mkdir(parents=True, exist_ok=True)
(OUT / 'meta').mkdir(parents=True, exist_ok=True)
for name, ids in splits.items():
    (OUT / f'{name}_ids.txt').write_text('\n'.join(ids) + '\n')

tok_re = re.compile(r"[a-z0-9']+")
train_feats, vocab = [], Counter()
for i, k in enumerate(all_ids, 1):
    feat = preprocess(gt['J_' + k]).astype(np.float32)
    np.save(OUT / 'motion' / f'{k}.npy', feat)
    cap = (man[k].get('exec') or '').strip()
    toks = tok_re.findall(cap.lower())
    (OUT / 'text' / f'{k}.txt').write_text(cap + '#' + ' '.join(toks) + '#nan#nan\n')
    if k in train_set:
        train_feats.append(feat)
        vocab.update(toks)
    if i % 250 == 0:
        print('preprocessed', i, flush=True)

F = np.concatenate(train_feats, 0)  # TRAIN only, as in the paper evaluator
mean = F.mean(0).astype(np.float32)
std = F.std(0).astype(np.float32)
std[std < 1e-6] = 1.0
np.save(OUT / 'meta' / 'mean.npy', mean)
np.save(OUT / 'meta' / 'std.npy', std)

words = ['unk'] + [w for w, c in vocab.most_common() if c >= 3 and w != 'unk']
idx = {w: i for i, w in enumerate(words)}
vecs = np.zeros((len(words), 300), np.float32)
found = 0
with open(GLOVE, encoding='utf-8') as fh:
    for line in fh:
        parts = line.rstrip().split(' ')
        w = parts[0]
        if w in idx:
            vecs[idx[w]] = np.asarray(parts[1:], np.float32)
            found += 1
vecs[idx['unk']] = vecs.mean(0)
np.save(OUT / 'meta' / 'our_vab_data.npy', vecs)
with open(OUT / 'meta' / 'our_vab_words.pkl', 'wb') as fh:
    pickle.dump(words, fh)
with open(OUT / 'meta' / 'our_vab_idx.pkl', 'wb') as fh:
    pickle.dump(idx, fh)
print(f'done: {len(all_ids)} items, vocab {len(words)} (glove hit {found}), '
      f'mean/std over {len(train_feats)} TRAIN items', flush=True)
