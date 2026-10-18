"""Residual-quality sensitivity after the full-rate M6 counterexample; no neural FPS."""
from collections import Counter,defaultdict
import json
from pathlib import Path
from scripts.reassess_fixed import reassess
from scripts.evaluate import evaluate

results=[]
for cutoff in [.08,.1,.12,.16]:
    counts=defaultdict(Counter);events=[]
    for source in sorted(Path('artifacts/confidence_full/neural_low').glob('*/benchmark.json')):
        seq=source.parent.name;out=Path('artifacts/residual_sensitivity')/str(cutoff)/seq
        reassess(source.parent,out,{'max_log_residual':cutoff,'correlation_seconds':1.},1)
        r=evaluate(out/'decisions.jsonl',f'data/qut_full/{seq}.gt.json')
        (out/'evaluation.json').write_text(json.dumps(r,indent=2))
        for kind,c in r['counts'].items():counts[kind].update(c)
        events.extend({'sequence':seq,**e} for e in r['events'])
    row={'max_log_residual':cutoff,'counts':counts,'events':events};results.append(row)
    Path('reports/residual_sensitivity.json').write_text(json.dumps({'kind':'Post-failure quality threshold sensitivity; fixed current low-threshold actual detections',
        'results':results,'limits':['Studied sources and diagnosed M6 failure, not independent validation','Reduced warnings by abstention can reduce useful coverage','No assumption of calibrated residual error or physical safety']},indent=2))
    print(cutoff,counts,flush=True)
