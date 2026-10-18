"""Offline identity-oracle motion probe using only historical YOLO boxes as inputs.

QUT labels select the historical identity and score the next centre; labels are
never supplied as model features. This is NOT online association accuracy.
"""
from pathlib import Path
import sys,json,time
from collections import defaultdict
import numpy as np
from mot_models import Adapter,ROOT
sys.path.insert(0,str(ROOT))
import torch
with torch.inference_mode():
    c=torch.nn.Conv2d(3,4,3).cuda();c(torch.zeros(1,3,16,16,device='cuda'));torch.cuda.synchronize()
del c

def samples():
    result=[]
    for seq in ['M1','M2','M6','M7','S1','S2']:
        records=[json.loads(x) for x in (ROOT/f'artifacts/cadence_fullrate/{seq}/detections.jsonl').read_text().splitlines()]
        first=len(records)-875;byframe={r['frame']:r for r in records};byid=defaultdict(dict)
        gt=json.loads((ROOT/f'data/qut_full/{seq}.gt.json').read_text())
        for g in gt:byid[g['id']][g['frame']]=g
        for tid,labels in byid.items():
            boxes={}
            for f,g in labels.items():
                d=np.asarray(byframe[f]['detections']).reshape(-1,6)
                if not len(d):continue
                centres=(d[:,:2]+d[:,2:4])/2;dist=np.linalg.norm(centres-[g['u'],g['v']],axis=1);j=int(dist.argmin())
                if dist[j]<=20:
                    b=d[j,:4];boxes[f]=np.r_[(b[:2]+b[2:])/2,b[2:]-b[:2]]
            for f in range(max(first+5,min(labels)),max(labels)+1,5):
                if f not in labels or any(k not in boxes for k in range(f-5,f)):continue
                hist=np.stack([boxes[k] for k in range(f-5,f)])
                # Match upstream 8D representation: position plus one-step delta.
                deltas=np.stack([boxes[k]-boxes[k-1] if k-1 in boxes else np.zeros(4) for k in range(f-5,f)])
                cond=np.concatenate([hist,deltas],axis=1)
                result.append(dict(sequence=seq,frame=f,condition=cond,last=hist[-1],target=np.array([labels[f]['u'],labels[f]['v']]),history=hist))
    return result

if __name__=='__main__':
    name=sys.argv[1];data=samples();pred=[];start=time.perf_counter()
    if name in ['diffmot','trackssm']:
        torch.manual_seed(123);m=Adapter(name,1280,720,appearance=False)
        with torch.inference_mode():
            for i in range(0,len(data),64):
                batch=data[i:i+64];d=m.model.generate([x['condition'] for x in batch],1,True,img_w=1280,img_h=720).mean(0)
                last=np.stack([x['last'] for x in batch]);pred.extend((d*np.array([1280,720,1280,720])+last)[:,:2])
    elif name=='hold':pred=[x['last'][:2] for x in data]
    elif name=='linear':
        times=np.arange(5)-2
        pred=[x['history'][:,:2].mean(0)+3*(times[:,None]*x['history'][:,:2]).sum(0)/(times@times) for x in data]
    else:raise ValueError(name)
    groups=defaultdict(list)
    for x,p in zip(data,pred):groups[x['sequence']].append(float(np.linalg.norm(p-x['target'])))
    def stats(values):return dict(samples=len(values),median_px=float(np.median(values)),p90_px=float(np.percentile(values,90)),mean_px=float(np.mean(values)))
    r=dict(method=name,scope='Offline identity-oracle, historical detector boxes only, next approximate GT centre; no association or end-to-end FPS claim',
        data_sequences=['M1','M2','M6','M7','S1','S2'],per_sequence={s:stats(v) for s,v in groups.items()},combined=stats(sum(groups.values(),[])),seconds=time.perf_counter()-start)
    out=ROOT/'reports/mot_motion';out.mkdir(exist_ok=True);(out/f'{name}.json').write_text(json.dumps(r,indent=2));print(json.dumps(r['combined']))
