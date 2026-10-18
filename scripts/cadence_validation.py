"""Actual default-cadence inference on full studied failure sequences.

All frames are processed, unlike the stride5 full-source evaluation. This is a
sampling sensitivity/regression check, not a new untouched test set.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import subprocess
import sys
from scripts.evaluate import evaluate

root=Path('artifacts/cadence_fullrate');root.mkdir(parents=True,exist_ok=True)
sequences=['M1','M2','M4','M6','S2'];counts=defaultdict(Counter);events=[];runs=[]
for seq in sequences:
    out=root/seq
    cmd=[sys.executable,'-m','airavoid',f'data/qut_full/{seq}.mp4','--camera','fixed','--profile','conservative',
         '--stride','1','--no-render','--output',str(out)]
    with (root/(seq+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    r=evaluate(out/'decisions.jsonl',f'data/qut_full/{seq}.gt.json');(out/'evaluation.json').write_text(json.dumps(r,indent=2))
    b=json.loads((out/'benchmark.json').read_text());runs.append({'sequence':seq,'frames':b['frames'],'processing_fps':b['processing_fps'],'counts':r['counts']})
    for kind,c in r['counts'].items():counts[kind].update(c)
    events.extend({'sequence':seq,**e} for e in r['events'])
    report={'kind':'Actual conservative all-frame cadence regression, previously studied complete sources',
            'stride':1,'source_fps':25,'runs':runs,'counts':counts,'events':events,
            'limits':['Frame-level denominators differ from stride5; compare rates/event behavior, not raw totals',
                      'No collision/CPA truth or untouched-test claim','Runtime during exploratory work is not the clean annotated benchmark']}
    Path('reports/cadence_fullrate.json').write_text(json.dumps(report,indent=2))
    print('Completed',seq,r['counts'],flush=True)
