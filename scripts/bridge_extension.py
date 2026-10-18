"""Matched full-source extension of optional appearance tracking on studied RGB."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import subprocess
import sys
from scripts.evaluate import evaluate

sources=['M6','S1','S3','S5'];methods={'dense':[],'appearance':['--appearance-bridge'],
    'appearance_independent':['--appearance-bridge','--detector-scale-only']}
root=Path('artifacts/bridge_extension');root.mkdir(parents=True,exist_ok=True);results=[]
for method,flags in methods.items():
    counts=defaultdict(Counter);events=[];runs=[];unmatched=Counter()
    for seq in sources:
        out=root/method/seq;out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-m','airavoid',f'data/qut_full/{seq}.mp4','--camera','fixed','--stride','2',
             '--no-render','--output',str(out),*flags]
        with (out/'run.log').open('w',encoding='utf-8') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        r=evaluate(out/'decisions.jsonl',f'data/qut_full/{seq}.gt.json');(out/'evaluation.json').write_text(json.dumps(r,indent=2))
        b=json.loads((out/'benchmark.json').read_text())
        for kind,c in r['counts'].items():counts[kind].update(c)
        unmatched.update(r['unmatched_track_frames']);events.extend({'sequence':seq,**e} for e in r['events'])
        runs.append({'sequence':seq,'frames':b['frames'],'processing_fps':b['processing_fps'],'counts':r['counts']})
        print('Completed',method,seq,flush=True)
    results.append({'method':method,'counts':counts,'events':events,'runs':runs,'unmatched_track_frames':unmatched})
    Path('reports/bridge_extension.json').write_text(json.dumps({'kind':'Actual NN optional-method matched-cadence extension on already studied full sources',
        'sources':sources,'stride':2,'results':results,'limits':['Development sources, no untouched or detector-training-independent claim',
        'No rendering in comparison; processing FPS is not annotated throughput','M6 other-category warning was already diagnosed before extension',
        'SOT scores and box size are not calibrated observations','No physical collision or CPA labels']},indent=2))
    print('Finished',method,{k:{s:c.get(s,0) for s in ['total','threat','nonthreat','uncertain','untracked']} for k,c in counts.items()},flush=True)
