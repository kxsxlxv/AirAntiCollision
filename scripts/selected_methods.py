"""Compare optional crop/appearance methods under the selected conservative settings.

M4 uses its full source to retain the previously diagnosed failure context; other
sources use the documented last35s selection. All are already development data.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import subprocess
import sys
from scripts.evaluate import evaluate

sources={'M1':'data/qut','M2':'data/qut','M4':'data/qut_full','S2':'data/qut'}
methods={'dense':[],'adaptive':['--adaptive-tiles'],'appearance':['--appearance-bridge'],
         'appearance_independent':['--appearance-bridge','--detector-scale-only']}
root=Path('artifacts/selected_methods');root.mkdir(parents=True,exist_ok=True);results=[]
for method,flags in methods.items():
    counts=defaultdict(Counter);events=[];runs=[]
    for seq,data_root in sources.items():
        out=root/method/seq;out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-m','airavoid',f'{data_root}/{seq}.mp4','--profile','conservative','--camera','fixed',
             '--stride','2','--no-render','--output',str(out),*flags]
        with (out/'run.log').open('w',encoding='utf-8') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        r=evaluate(out/'decisions.jsonl',f'{data_root}/{seq}.gt.json');(out/'evaluation.json').write_text(json.dumps(r,indent=2))
        benchmark=json.loads((out/'benchmark.json').read_text())
        for kind,c in r['counts'].items():counts[kind].update(c)
        events.extend({'sequence':seq,**e} for e in r['events'])
        runs.append({'sequence':seq,'data_root':data_root,'frames':benchmark['frames'],'processing_fps':benchmark['processing_fps'],
                     'stages':benchmark['stages'],'input_sha256':benchmark['input_sha256'],'counts':r['counts']})
        print('Finished',method,seq,flush=True)
    row={'method':method,'counts':counts,'events':events,'runs':runs};results.append(row)
    report={'kind':'Actual neural optional-method comparison under selected profile; no GT online','sources':sources,
            'stride':2,'results':results,'limits':['Studied/development sources, not untouched validation','M4 full source and other last35s selections are explicit',
                                               'Detector match counts include only queried frames; appearance methods query every3processedframes',
                                               'No rendering in this comparison; processing FPS is not final annotated-video throughput',
                                               'No true collision/CPA labels; nominal category scores only']}
    Path('reports/selected_methods.json').write_text(json.dumps(report,indent=2))
    print('Completed',method,{k:{kk:v for kk,v in c.items() if kk in ['total','threat','nonthreat','uncertain','untracked']} for k,c in counts.items()},flush=True)
