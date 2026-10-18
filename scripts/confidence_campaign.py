"""Actual low-threshold neural runs followed by confidence-filter counterfactuals.

Thresholding after class-agnostic score-ordered NMS retains the same high-score
boxes. Camera validity is held at the low-threshold source value for isolation.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
from scripts.reassess_fixed import reassess
from scripts.evaluate import evaluate
from airavoid.provenance import file_sha256

p=argparse.ArgumentParser();p.add_argument('--sequences',nargs='+',default=['M1','M2','S2'])
p.add_argument('--data-root',default='data/qut');p.add_argument('--stride',type=int,default=2)
p.add_argument('--thresholds',nargs='+',type=float,default=[.03,.08,.12,.18,.25,.4,.6]);p.add_argument('--root',default='artifacts/confidence_campaign');a=p.parse_args()
root=Path(a.root);root.mkdir(parents=True,exist_ok=True);low=min(a.thresholds)
for seq in a.sequences:
    out=root/'neural_low'/seq
    cmd=[sys.executable,'-m','airavoid',f'{a.data_root}/{seq}.mp4','--camera','fixed','--stride',str(a.stride),'--tiled','--gpu-preprocess',
         '--conf',str(low),'--tile-edge-margin','2','--image-edge-margin','1','--correlation-seconds','1','--no-render','--output',str(out)]
    with (root/(seq+'_neural.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('Actual neural low-threshold run finished:',seq,flush=True)
results=[]
for threshold in a.thresholds:
    aggregate=defaultdict(Counter);events=[];unmatched=Counter();per_sequence=[]
    for seq in a.sequences:
        source=root/'neural_low'/seq;filtered=root/'filtered'/str(threshold)/seq;filtered.mkdir(parents=True,exist_ok=True)
        original=json.loads((source/'benchmark.json').read_text())
        shutil.copyfile(source/'decisions.jsonl',filtered/'decisions.jsonl')
        with (filtered/'detections.jsonl').open('w') as f:
            for row in map(json.loads,(source/'detections.jsonl').read_text().splitlines()):
                boxes=[b for b in row['detections'] if b[4]>=threshold]
                f.write(json.dumps({**row,'detections':boxes,'observations':boxes})+'\n')
        metadata={k:original[k] for k in ['config','policy_parameters','resolution','source_fps','input_sha256']}
        metadata.update(kind='Confidence-filtered cached neural observations, not another neural run',
                        args={**original['args'],'conf':threshold},source_neural_run=str(source),
                        source_neural_benchmark_sha256=file_sha256(source/'benchmark.json'),fixed_camera_validity_reused=True)
        (filtered/'benchmark.json').write_text(json.dumps(metadata,indent=2))
        out=root/'assessment'/str(threshold)/seq
        reassess(filtered,out,image_edge_margin=1,verify=threshold==low)
        r=evaluate(out/'decisions.jsonl',f'{a.data_root}/{seq}.gt.json');(out/'evaluation.json').write_text(json.dumps(r,indent=2))
        for kind,c in r['counts'].items():aggregate[kind].update(c)
        unmatched.update(r['unmatched_track_frames']);events.extend({'sequence':seq,**e} for e in r['events'])
        per_sequence.append({'sequence':seq,'counts':r['counts']})
    results.append({'threshold':threshold,'counts':aggregate,'events':events,'unmatched_track_frames':unmatched,'per_sequence':per_sequence})
    report={'settings':vars(a),'kind':'Actual low-confidence neural observations and post-NMS threshold counterfactuals',
            'results':results,'limits':['Development/diagnosed sources, no untouched-test claim','Camera validity is fixed from low-threshold inference',
                                       'GT centroids approximate; many low-confidence boxes can inflate nearest-centroid matches','Unmatched warnings are not verified false alarms',
                                       'Only neural_low runs contain actual neural runtime; filtered/reassessed outputs do not']}
    (root/'campaign_summary.json').write_text(json.dumps(report,indent=2))
    if a.root=='artifacts/confidence_campaign':Path('reports/confidence_campaign.json').write_text(json.dumps(report,indent=2))
    print('Threshold',threshold,{k:{kk:n for kk,n in c.items() if kk in ['total','detected','threat','nonthreat','uncertain','untracked']} for k,c in aggregate.items()},flush=True)
