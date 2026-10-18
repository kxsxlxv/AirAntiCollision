"""Actual detector crop-edge ablation. No GT enters inference."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from scripts.evaluate import evaluate

root=Path('artifacts/crop_guard');results=[]
for seq in ['M1','M2','formation_native']:
    video=Path('data/qut')/(seq+'.mp4') if seq.startswith('M') else Path('data/formation_native.mp4')
    for margin in [0,2,8]:
        out=root/seq/str(margin);out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-m','airavoid',str(video),'--camera','fixed' if seq.startswith('M') else 'auto',
             '--stride','2' if seq.startswith('M') else '3','--tiled','--gpu-preprocess','--no-render',
             '--tile-edge-margin',str(margin),'--output',str(out)]
        with (out/'run.log').open('w',encoding='utf-8') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=[json.loads(x) for x in (out/'detections.jsonl').read_text().splitlines()]
        trace=[json.loads(x) for x in (out/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
        row={'sequence':seq,'margin_px':margin,'frames':len(rows),'raw_detections':sum(len(x['detections']) for x in rows),
             'track_states':dict(Counter(o['assessment']['state'] for x in trace for o in x['objects'])),
             'track_ids':len({o['id'] for x in trace for o in x['objects']}),'path':str(out)}
        if seq.startswith('M'):
            report=evaluate(out/'decisions.jsonl',Path('data/qut')/(seq+'.gt.json'))
            (out/'evaluation.json').write_text(json.dumps(report,indent=2))
            row['counts']=report['counts'];row['events']=report['events']
        results.append(row);print(json.dumps(row),flush=True)
report={'kind':'Actual neural crop-edge rejection ablation, development data, no GT online',
        'results':results,'limits':'Formation has no exhaustive per-frame boxes or threat labels. Fewer IDs/detections alone do not prove improved precision. Runtime during exploratory work is not the clean benchmark.'}
Path('reports/crop_guard_trial.json').write_text(json.dumps(report,indent=2))
