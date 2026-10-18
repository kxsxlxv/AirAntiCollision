"""Offline GT-centroid/identity ablation with measured detector spans; never inference."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import numpy as np
from airavoid.geometry import ThreatEstimator,robust_line
from scripts.evaluate import match

p=argparse.ArgumentParser();p.add_argument('--sequences',nargs='+',default=['M1','M2']);p.add_argument('--root',default='demo');a=p.parse_args()
report=[]
for seq in a.sequences:
    run=Path(a.root)/(seq+'_tiled_initial')
    gt=defaultdict(list)
    for r in json.loads(Path(f'data/qut/{seq}.gt.json').read_text()):gt[r['frame']].append(r)
    history=defaultdict(list);counts=defaultdict(Counter);tau=[];evaluator=ThreatEstimator()
    for r in map(json.loads,(run/'detections.jsonl').read_text().splitlines()):
        labels=gt[r['frame']];pairs=match(r['detections'],labels,20)
        for j,l in enumerate(labels):
            kind=l['kind'];t=r['time'];ident=l['id']
            if j not in pairs:counts[kind]['undetected']+=1;continue
            b=r['detections'][pairs[j]];span=max(b[2]-b[0],b[3]-b[1])
            h=history[ident]
            if h and t-h[-1][0]>.8:h.clear()
            h.append((t,l['u'],l['v'],span));e=evaluator.assess(h,t)
            counts[kind][e['state']]+=1
            if kind=='approach' and e['tau'] is not None:
                # Range-derived approach timescale: NOT physical impact or CPA time.
                ranges=[]
                for frame,items in gt.items():
                    if t-4<=frame/25<=t:
                        for ll in items:
                            if ll['id']==ident and ll['range_offline_only'] is not None:ranges.append((frame/25,ll['range_offline_only']))
                if len(ranges)>10:
                    ranges=sorted(ranges);ra=np.asarray(ranges)
                    value,slope,se,res=robust_line(ra[:,0],ra[:,1],3)
                    if slope<0:
                        ref=value/(-slope)
                        tau.append({'time':t,'image_expansion_tau_s':e['tau'],'offline_range_timescale_s':ref,'absolute_error_s':abs(e['tau']-ref)})
    report.append({'sequence':seq,'oracle_centroid_identity_counts':counts,'tau_comparison_samples':len(tau),'median_absolute_timescale_error_s':float(np.median([x['absolute_error_s'] for x in tau])) if tau else None,'tau_samples':tau,
                   'limitations':'GT centroid/identity only in this offline ablation; detector recall/span errors remain. Range timescale is not ground-truth impact time. Correlated frame samples.'})
Path('reports/oracle_analysis.json').write_text(json.dumps(report,indent=2))
for r in report:print({k:v for k,v in r.items() if k!='tau_samples'})
