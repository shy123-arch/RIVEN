"""UMRM evaluator candidates. TRAIN-only fitting/selection; VAL audit; never TEST.

E1: bilateral contrastive adaptation. E2: official TMR reconstruction, sampled
latent alignment, four KL terms and contrastive weights. Sentence-similarity
negative filtering is explicitly disabled (no verified MPNet asset).
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('RIVEN_ROOT', HERE.parents[2]))
ASSETS=Path(os.environ.get('RIVEN_TMR_ASSETS', str(HERE/'tmr_assets')))
STATE=Path(os.environ.get('RIVEN_EVAL_STATE', str(ROOT/'state/evaluator')))
RELEASE=Path(os.environ.get('RIVEN_DATA', str(ROOT/'data/egoreal')))
TEXT_RELEASE=Path(os.environ.get('RIVEN_TEXT_RELEASE', str(RELEASE)))
EXPECT_TRAIN=int(os.environ.get('RIVEN_EVAL_EXPECT_TRAIN', '1096'))
EXPECT_VAL=int(os.environ.get('RIVEN_EVAL_EXPECT_VAL', '213'))
sys.path.insert(0,str(ASSETS.parent))
from tmr_adapter import OfficialTMR, preprocess, find_model, official_module

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def status(**kw):
    kw.update(pid=os.getpid(),test_enabled=False,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    p=STATE/'status.tmp';p.write_text(json.dumps(kw,indent=2));p.replace(STATE/'status.json')

def group_split(groups,seed=0,frac=.15):
    unique=sorted(set(groups));rng=np.random.default_rng(seed);rng.shuffle(unique)
    held=set(unique[:max(1,int(len(unique)*frac))])
    tr=[i for i,g in enumerate(groups) if g not in held]
    iv=[i for i,g in enumerate(groups) if g in held]
    if not tr or not iv: raise ValueError('source-group split empty')
    return tr,iv

def metric(S):
    # Average tied ranks avoid spuriously perfect retrieval on collapsed embeddings.
    diag=np.diag(S)[:,None]
    ranks=1+(S>diag).sum(1)+((S==diag).sum(1)-1)/2
    return {**{f'R@{k}':float((ranks<=k).mean()) for k in [1,3,10]},
            'mean_rank':float(ranks.mean()),'median_rank':float(np.median(ranks))}

def report(T,M,groups,seed=0):
    S=T@M.T; rng=np.random.default_rng(seed)
    result={'items':len(S),'t2m':metric(S),'m2t':metric(S.T)}
    perm=rng.permutation(len(S))
    result['shuffled_m2t']=metric(S[:,perm].T)
    pooled=[]
    for mat in [S,S.T]:
        rr=[]
        for i in range(len(S)):
            neg=[j for j in range(len(S)) if groups[j]!=groups[i]]
            if len(neg)<31: continue
            for _ in range(20):
                cols=np.r_[i,rng.choice(neg,31,replace=False)]
                v=mat[i,cols];r=1+(v>v[0]).sum()+((v==v[0]).sum()-1)/2
                rr.append(r)
        pooled.append({f'R@{k}':float(np.mean(np.array(rr)<=k)) if rr else None for k in [1,3,10]})
    result['source_excluded_32pool']={'t2m':pooled[0],'m2t':pooled[1],'repeats':20,'seed':seed}
    result['embedding_std']={'text':float(T.std(0).mean()),'motion':float(M.std(0).mean())}
    return result

def kl(q,p):
    mu,lv=q;mp,lp=p
    diff=lv-lp
    return .5*(diff.exp()+(mp-mu).square()/lp.exp()-1-diff).mean()

def objective(t,m,x,mask,decoder=None):
    labels=torch.arange(len(t),device=t.device)
    if decoder is None:
        zt,zm=t[:,0],m[:,0]
    else:
        zt=t[:,0]+torch.randn_like(t[:,0])*(.5*t[:,1]).exp()
        zm=m[:,0]+torch.randn_like(m[:,0])*(.5*m[:,1]).exp()
    logits=F.normalize(zt,dim=-1)@F.normalize(zm,dim=-1).T/.1
    contrastive=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2
    if decoder is None: return contrastive
    rt=decoder({'z':zt,'mask':mask});rm=decoder({'z':zm,'mask':mask})
    rec=F.smooth_l1_loss(rt[mask],x[mask])+F.smooth_l1_loss(rm[mask],x[mask])
    qt=(t[:,0],t[:,1]);qm=(m[:,0],m[:,1]);normal=(torch.zeros_like(zt),torch.zeros_like(zt))
    k=kl(qt,qm)+kl(qm,qt)+kl(qt,normal)+kl(qm,normal)
    return rec+1e-5*F.smooth_l1_loss(zt,zm)+1e-5*k+.1*contrastive

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--seed',type=int,default=0);a=ap.parse_args()
    STATE.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(STATE/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    rng=np.random.default_rng(a.seed)
    status(phase='checking_inputs')
    sp=json.loads((RELEASE/'split.json').read_text())
    train=list(sp['train_real']);val=list(sp['val'])
    if (len(train),len(val))!=(EXPECT_TRAIN,EXPECT_VAL) or set(train)&set(val):
        raise ValueError('formal v1572 split mismatch')
    man=json.loads((TEXT_RELEASE/'data_manifest.json').read_text())
    if man['train_real']!=train or man['val']!=val: raise ValueError('text/full-event split mismatch')
    # Retain only the authorized rows. No TEST feature, text encoding, fit or evaluation.
    rows={k:man['rows'][k] for k in train+val};del man
    texts=[rows[k]['exec'] for k in train+val]
    if any(not isinstance(t,str) or not t.strip() for t in texts): raise ValueError('canonical exec missing')
    groups=[rows[k]['source'] for k in train+val]
    if any(not isinstance(g,str) or not g for g in groups):raise ValueError('source metadata missing')
    tr,iv=group_split(groups[:len(train)],a.seed);va=list(range(len(train),len(train)+len(val)))
    private={'train':[train[i] for i in tr],'internal':[train[i] for i in iv],'val':val}
    (STATE/'private_split.json').write_text(json.dumps(private))
    status(phase='loading_official_TMR',train=len(tr),internal=len(iv),val=len(val))
    O=OfficialTMR(ASSETS/'distilbert-base-uncased','cuda:0')
    O.backbone.requires_grad_(False)
    initial_t=copy.deepcopy(O.text_encoder.state_dict());initial_m=copy.deepcopy(O.motion_encoder.state_dict())
    feats=[];tokens=[]
    with np.load(RELEASE/'motion/gt22_full.npz') as gt:
        for i,key in enumerate(train+val):
            f=preprocess(gt['J_'+key]);f=(torch.tensor(f,device=O.device)-O.mean)/(O.std+1e-12)
            feats.append(f.cpu())
            if i%100==0:status(phase='preparing_motion_features',completed=i,total=len(train)+len(val))
    for start in range(0,len(texts),32):
        inp=O.tokenizer(texts[start:start+32],return_tensors='pt',padding=True,truncation=True,max_length=512).to(O.device)
        with torch.no_grad(): hidden=O.backbone(**inp).last_hidden_state
        for h,m in zip(hidden,inp.attention_mask.bool()):tokens.append(h[m].cpu())
    del texts,rows,O.backbone
    torch.cuda.empty_cache()
    def batch(indices):
        fs=[feats[i] for i in indices];ts=[tokens[i] for i in indices]
        x=torch.nn.utils.rnn.pad_sequence(fs,batch_first=True).to(O.device)
        y=torch.nn.utils.rnn.pad_sequence(ts,batch_first=True).to(O.device)
        mask=torch.arange(x.shape[1],device=O.device)[None]<torch.tensor([len(f) for f in fs],device=O.device)[:,None]
        tm=torch.arange(y.shape[1],device=O.device)[None]<torch.tensor([len(t) for t in ts],device=O.device)[:,None]
        return x,mask,y,tm
    def evaluate(indices):
        O.text_encoder.eval();O.motion_encoder.eval();tt=[];mm=[]
        with torch.no_grad():
            for start in range(0,len(indices),32):
                x,mask,y,tm=batch(indices[start:start+32])
                tt.append(F.normalize(O.text_encoder({'x':y,'mask':tm})[:,0],dim=-1).cpu().numpy())
                mm.append(F.normalize(O.motion_encoder({'x':x,'mask':mask})[:,0],dim=-1).cpu().numpy())
        return report(np.concatenate(tt),np.concatenate(mm),[groups[i] for i in indices],a.seed)
    baseline={'internal':evaluate(iv),'val':evaluate(va)}
    (STATE/'E0.json').write_text(json.dumps(baseline,indent=2))
    config=json.loads((find_model()/'config.json').read_text())
    protocol={'schema':'umrm-domain-evaluator/v1','status':'candidate; not promoted to benchmark',
        'formal_version':'v1572 full-event','fit':'TRAIN source-disjoint internal development split',
        'test_enabled':False,'text':'canonical v1572 exec, single reference, no mirror',
        'selection':'internal mean bilateral R@3; never generator scores or VAL selection',
        'retrieval':'strict matched identity; cosine; average tie rank; no CSLS',
        'pool32':'exclude negatives with query source; negatives can share each other sources',
        'negative_filtering':False,'filter_note':'official MPNet sentence embeddings unavailable; explicit ablation',
        'E2':'official weights recons=1 latent=1e-5 kl=1e-5 contrastive=.1; valid-frame SmoothL1; sampled latents',
        'temperature':.1,'epochs':a.epochs,'lr':1e-5,'batch':32,'seed':a.seed,
        'counts':{'fit':len(tr),'internal':len(iv),'val':len(val)},
        'split_sha256':digest(RELEASE/'split.json'),'text_manifest_sha256':digest(TEXT_RELEASE/'data_manifest.json'),
        'code_sha256':digest(__file__),'official_archive_sha256':digest(ASSETS/'archive_provenance.json')}
    (STATE/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for variant in ['E1','E2']:
        O.text_encoder.load_state_dict(initial_t);O.motion_encoder.load_state_dict(initial_m)
        decoder=None
        if variant=='E2':
            cfg=dict(config['model']['motion_decoder']);cfg.pop('_target_')
            decoder=official_module('umrm_tmr_actor','src/model/actor.py').ACTORStyleDecoder(**cfg).to(O.device)
            decoder.load_state_dict(torch.load(find_model()/'last_weights/motion_decoder.pt',map_location=O.device,weights_only=True))
        params=list(O.text_encoder.parameters())+list(O.motion_encoder.parameters())
        if decoder is not None:params+=list(decoder.parameters())
        optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=1e-6)
        best=-1;best_epoch=0
        for epoch in range(1,a.epochs+1):
            O.text_encoder.train();O.motion_encoder.train()
            if decoder is not None:decoder.train()
            order=rng.permutation(tr);losses=[]
            for start in range(0,len(order),32):
                indices=order[start:start+32]
                if len(indices)<2:continue
                x,mask,y,tm=batch(indices)
                t=O.text_encoder({'x':y,'mask':tm});m=O.motion_encoder({'x':x,'mask':mask})
                loss=objective(t,m,x,mask,decoder)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite evaluator objective')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
                losses.append(float(loss.detach()))
            r=evaluate(iv);score=(r['t2m']['R@3']+r['m2t']['R@3'])/2
            if score>best:
                best=score;best_epoch=epoch
                ck={'text_encoder':O.text_encoder.state_dict(),'motion_encoder':O.motion_encoder.state_dict(),
                    'epoch':epoch,'variant':variant,'protocol':protocol}
                if decoder is not None:ck['motion_decoder']=decoder.state_dict()
                torch.save(ck,STATE/(variant+'_best.pt'))
            status(phase='training',variant=variant,epoch=epoch,epochs=a.epochs,
                   loss=float(np.mean(losses)),best_internal_R3=best,best_epoch=best_epoch)
        ck=torch.load(STATE/(variant+'_best.pt'),map_location=O.device,weights_only=True)
        O.text_encoder.load_state_dict(ck['text_encoder']);O.motion_encoder.load_state_dict(ck['motion_encoder'])
        internal=evaluate(iv);validation=evaluate(va)
        improved=(internal['m2t']['R@10']>=baseline['internal']['m2t']['R@10'] and best>
                  (baseline['internal']['t2m']['R@3']+baseline['internal']['m2t']['R@3'])/2)
        result={'variant':variant,'selected_epoch':best_epoch,'internal':internal,'val':validation,
                'improved_internal':improved,'checkpoint_sha256':digest(STATE/(variant+'_best.pt')),
                'promotion':'candidate only; independent quality validation and freeze review pending',
                'val_train_source_overlap':sum(g in set(groups[:len(train)]) for g in groups[len(train):])}
        (STATE/(variant+'.json')).write_text(json.dumps(result,indent=2))
        del decoder,optimizer,params,ck
        torch.cuda.empty_cache()
    status(phase='candidates_complete',variants=['E0','E1','E2'],benchmark_promoted=False)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        STATE.mkdir(parents=True,exist_ok=True)
        failure={'exception_type':type(exc).__name__,'frames':[{'file':f.filename,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)]}
        if isinstance(exc,ModuleNotFoundError):failure['missing_module']=exc.name
        (STATE/'worker_failure.json').write_text(json.dumps(failure,indent=2))
        status(phase='failed',exception_type=type(exc).__name__)
        raise
