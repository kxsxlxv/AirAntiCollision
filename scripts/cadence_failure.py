"""Failure-directed all-frame M6 alternatives; development diagnostics, no default tuning."""
import json
from pathlib import Path
from scripts.reassess_fixed import reassess
from scripts.evaluate import evaluate

variants=[('baseline','log',{}),('projective','projective',{'correlation_seconds':0}),
          ('bounded','bounded',{'correlation_seconds':0}),('window6','log',{'window':6.}),
          ('residual10','log',{'max_log_residual':.1})]
results=[]
for name,method,overrides in variants:
    out=Path('artifacts/cadence_failure/M6')/name
    b=reassess('artifacts/cadence_fullrate/M6',out,overrides,1,verify=name=='baseline',geometry=method)
    r=evaluate(out/'decisions.jsonl','data/qut_full/M6.gt.json');(out/'evaluation.json').write_text(json.dumps(r,indent=2))
    results.append({'variant':name,'geometry':method,'overrides':overrides,'counts':r['counts'],'events':r['events'],
                    'unmatched':r['unmatched_track_frames'],'source_verification_mismatches':b['source_decision_mismatch_frames']})
    Path('reports/cadence_failure.json').write_text(json.dumps({'kind':'M6 all-frame diagnosed-failure geometry counterfactual; fixed actual observations, no NN performance claim',
        'results':results,'limits':['Same source that exposed failure; no independent generalization','Published other category, not physical CPA truth','No change to selected defaults from this isolated case']},indent=2))
    print(name,r['counts'],flush=True)
