"""Finish actual stride1 validation on all15complete QUT sources, reusing verified runs."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from scripts.evaluate import evaluate
from airavoid.provenance import file_sha256

root=Path('artifacts/cadence_fullrate');sequences=[f'M{i}' for i in range(1,9)]+[f'S{i}' for i in range(1,8)]
counts=defaultdict(Counter);events=[];runs=[];unmatched=Counter();started=datetime.now(timezone.utc).isoformat()
for seq in sequences:
    out=root/seq;manifest=json.loads(Path(f'data/qut_full/{seq}.manifest.json').read_text());reused=False
    if (out/'benchmark.json').exists():
        b=json.loads((out/'benchmark.json').read_text());a=b['args'];status=json.loads((out/'run_status.json').read_text())
        expected={'camera':'fixed','stride':1,'no_render':True,'conf':.03,'tiled':True,'gpu_preprocess':True,
                  'tile_edge_margin':2.,'image_edge_margin':1.,'correlation_seconds':1.,'allow_recession_clearance':False,
                  'geometry':'log','policy':'confirmed','appearance_bridge':False}
        if any(a.get(k)!=v for k,v in expected.items()) or b['input_sha256']!=manifest['sha256'] or b['frames']!=manifest['frames'] or status['state']!='complete':
            raise ValueError('Existing run differs from selected complete-source configuration: '+seq)
        reused=True
    else:
        command=[sys.executable,'-m','airavoid',f'data/qut_full/{seq}.mp4','--camera','fixed','--stride','1','--no-render','--output',str(out)]
        with (root/(seq+'.log')).open('w',encoding='utf-8') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        b=json.loads((out/'benchmark.json').read_text())
    if file_sha256(f'data/qut_full/{seq}.mp4')!=manifest['sha256']:raise ValueError('Input hash changed: '+seq)
    r=evaluate(out/'decisions.jsonl',f'data/qut_full/{seq}.gt.json');(out/'evaluation.json').write_text(json.dumps(r,indent=2))
    for kind,c in r['counts'].items():counts[kind].update(c)
    unmatched.update(r['unmatched_track_frames']);events.extend({'sequence':seq,**e} for e in r['events'])
    runs.append({'sequence':seq,'frames':b['frames'],'reused_matching_complete_run':reused,'processing_fps':b['processing_fps'],
                 'input_sha256':b['input_sha256'],'benchmark_sha256':file_sha256(out/'benchmark.json'),'counts':r['counts']})
    report={'kind':'Actual all-frame conservative complete-source validation, previously studied data','started_utc':started,
            'updated_utc':datetime.now(timezone.utc).isoformat(),'complete':len(runs)==15,'stride':1,'source_fps':25,
            'processed_frames':sum(x['frames'] for x in runs),'runs':runs,'counts':counts,'events':events,'unmatched_track_frames':unmatched,
            'limits':['Approach/other are nominal categories, not physical collision/CPA truth','No untouched test or detector-training independence',
                      'Initial5 matching complete runs reused; later code additions only affect optional zoom and <=704x416 duplicate work, unused on these1280x720RGB sources',
                      'Exploratory no-render runtime, not annotated or streaming throughput','M3 range-category conflict retained; four missing source IDs repeated explicitly']}
    Path('reports/cadence_all_sources.json').write_text(json.dumps(report,indent=2))
    print('Reused' if reused else 'Completed',seq,{k:{s:c.get(s,0) for s in ['total','threat','nonthreat','uncertain','untracked']} for k,c in r['counts'].items()},flush=True)
