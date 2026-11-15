#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
_orig_load=torch.load
def _load(*a,**k):
 k.pop('weights_only',None); return _orig_load(*a,**k)
torch.load=_load
ROOT=Path(os.environ.get('RIVEN_ROOT', Path(__file__).resolve().parents[3]))
CACHE=Path(os.environ.get('RIVEN_CACHE', ROOT/'cache')); STATE=CACHE/'tmr'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'eval'))
import evaluator_train as E
RELEASE=Path(os.environ.get('RIVEN_DATA', ROOT/'data/egoreal'))
OUT=Path(os.environ.get('RIVEN_TMR_OUT', STATE/'tmr_motion_emb.npz'))
TMP=Path(os.environ.get('RIVEN_TMR_MOTION_EMB', CACHE/'tmr_motion_emb.npz'))
split=json.loads((RELEASE/'split.json').read_text()); keys=[str(x) for x in split['train_real']]+[str(x) for x in split['val']]
print('keys',len(keys),len(set(keys)),flush=True)
with np.load(RELEASE/'motion/gt22_full.npz') as z: seqs=[z['J_'+k] for k in keys]
O=E.OfficialTMR(E.ASSETS/'distilbert-base-uncased','cuda:0'); O.motion_encoder.eval(); out=[]
with torch.no_grad():
 for start in range(0,len(seqs),128):
  chunk=[((torch.as_tensor(E.preprocess(x),device=O.device)-O.mean)/(O.std+1e-12)) for x in seqs[start:start+128]]
  x=torch.nn.utils.rnn.pad_sequence(chunk,batch_first=True)
  mask=torch.arange(x.shape[1],device=O.device)[None]<torch.tensor([len(c) for c in chunk],device=O.device)[:,None]
  e=O.motion_encoder({'x':x,'mask':mask})[:,0]; out.append(F.normalize(e,dim=-1).cpu().numpy())
emb=np.concatenate(out).astype(np.float32); STATE.mkdir(parents=True,exist_ok=True); np.savez(OUT,ids=np.array(keys),emb=emb)
if TMP.is_symlink() or TMP.exists(): TMP.unlink()
os.symlink(OUT,TMP)
print(json.dumps({'items':len(keys),'shape':list(emb.shape),'out':str(OUT)}),flush=True)
