"""Development-only temporal window comparison, replaying identical neural detections."""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from airavoid.geometry import Config,ThreatEstimator
from airavoid.policy import TemporalPolicy
from airavoid.tracking import Tracker
from scripts.evaluate import evaluate
from airavoid.projective import ProjectiveEstimator
import argparse

p=argparse.ArgumentParser();p.add_argument('--method',choices=['log','projective'],default='log');p.add_argument('--windows',nargs='+',type=float,default=[4.,6.,8.,12.]);args=p.parse_args()
rows=[]
for window in args.windows:
    for seq in ['M1','M2']:
        src=Path(f'demo/{seq}_tiled_initial');out=Path(f'artifacts/{args.method}_window_ablation/w{int(window)}/{seq}');out.mkdir(parents=True,exist_ok=True)
        tracker=Tracker();estimator=(ProjectiveEstimator if args.method=='projective' else ThreatEstimator)(Config(window=window));policy=TemporalPolicy()
        raw=[json.loads(x) for x in (src/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
        detections=[json.loads(x) for x in (src/'detections.jsonl').read_text().splitlines()]
        with (out/'decisions.jsonl').open('w',encoding='utf-8') as f:
            for r,dr in zip(raw,detections):
                t=r['time'];objects=[]
                for tr in tracker.update(dr['detections'],t,np.eye(3)):
                    if tr.hits<3:continue
                    e=estimator.assess(tr.history,t,r['camera']['valid'],t<tr.ambiguous_until)
                    e=policy.apply(tr.id,t,e)
                    objects.append({'id':tr.id,'box':tr.box,'class_id':tr.class_id,'confidence':tr.confidence,'last_seen':tr.last,'assessment':e})
                r['objects']=objects;f.write(json.dumps(r,ensure_ascii=False)+'\n')
        (out/'detections.jsonl').write_bytes((src/'detections.jsonl').read_bytes())
        result=evaluate(out/'decisions.jsonl',f'data/qut/{seq}.gt.json')
        (out/'evaluation.json').write_text(json.dumps(result,indent=2))
        rows.append({'window':window,'sequence':seq,'counts':result['counts'],'events':result['events']})
        print(window,seq,result['counts'],flush=True)
Path(f'reports/{args.method}_window_ablation.json').write_text(json.dumps(rows,indent=2))
