"""Controlled geometry ablations on complete actual fixed-camera observations."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import time
from scripts.reassess_fixed import reassess
from scripts.evaluate import evaluate

sequences=[f'M{i}' for i in range(1,9)]+[f'S{i}' for i in range(1,8)]
variants=[('baseline',{},0),('clip1',{},1),('clip1_hac05',{'correlation_seconds':.5},1),
          ('clip1_hac1',{'correlation_seconds':1.},1),('clip1_span10',{'min_span':10.},1),('clip1_window6',{'window':6.},1)]
all_results=[]
for name,overrides,edge in variants:
    counts=defaultdict(Counter);events=[];unmatched=Counter();runs=[];numeric_error=0.
    for seq in sequences:
        source=Path(f'artifacts/full_gpu_default/{seq}');deadline=time.monotonic()+1200
        while not (source/'benchmark.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Source run unavailable: '+str(source))
            time.sleep(5)
        out=Path(f'artifacts/geometry_campaign/{name}/{seq}')
        result=reassess(source,out,overrides,edge,verify=name=='baseline')
        numeric_error=max(numeric_error,result['source_max_numeric_difference'])
        evaluation=evaluate(out/'decisions.jsonl',f'data/qut_full/{seq}.gt.json')
        (out/'evaluation.json').write_text(json.dumps(evaluation,indent=2));runs.append(seq)
        for kind,c in evaluation['counts'].items():counts[kind].update(c)
        unmatched.update(evaluation['unmatched_track_frames'])
        events.extend({'sequence':seq,**e} for e in evaluation['events'])
        print('Reassessed',name,seq,flush=True)
    row={'variant':name,'overrides':overrides,'image_edge_margin':edge,'sequences':runs,'counts':counts,'events':events,
         'unmatched_track_frames':unmatched,'source_max_numeric_difference':numeric_error}
    all_results.append(row)
    report={'kind':'Geometry-only complete-source development/extension counterfactual; actual observations/camera validity fixed',
            'variants':all_results,'source':'artifacts/full_gpu_default',
            'limits':['No new untouched split; M6 false-warning diagnosis precedes these ablations','M3 range conflicts with nominal approach category',
                      'No collision/CPA ground truth','Reassessment elapsed is not neural or video end-to-end throughput']}
    Path('reports/geometry_campaign.json').write_text(json.dumps(report,indent=2))
    print('Completed variant',name,{k:{s:v for s,v in c.items() if s in ['total','threat','nonthreat','uncertain','untracked']} for k,c in counts.items()},flush=True)
