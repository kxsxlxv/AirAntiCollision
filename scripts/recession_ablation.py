"""Exact counterfactual state replay: contraction alone cannot establish clearance.

Raw measurements and tracks unchanged. Only recession candidate -> uncertainty;
temporal confirmation is recomputed, no GT used until separate evaluation.
"""
import argparse
import json
from pathlib import Path
from airavoid.policy import TemporalPolicy
from scripts.evaluate import evaluate

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('gt');p.add_argument('--output',required=True);a=p.parse_args()
src=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);policy=TemporalPolicy()
with (out/'decisions.jsonl').open('w',encoding='utf-8') as f:
    for r in map(json.loads,(src/'decisions.jsonl').read_text(encoding='utf-8').splitlines()):
        for o in r['objects']:
            e=o['assessment'].copy();e['state']=e.get('candidate_state',e['state']);e['reason']=e.get('candidate_reason',e['reason'])
            for k in ['candidate_state','candidate_reason','confirmation_elapsed_s','confirmation_required_s']:e.pop(k,None)
            if e['reason']=='receding':e.update(state='uncertain',reason='shrink_ambiguous')
            o['assessment']=policy.apply(o['id'],r['time'],e)
        f.write(json.dumps(r,ensure_ascii=False)+'\n')
(out/'detections.jsonl').write_bytes((src/'detections.jsonl').read_bytes())
report=evaluate(out/'decisions.jsonl',a.gt)
(out/'evaluation.json').write_text(json.dumps(report,indent=2))
print(report['counts'])
