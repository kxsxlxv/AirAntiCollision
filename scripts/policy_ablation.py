import argparse
from collections import Counter
import json
from pathlib import Path
from airavoid.policy import TemporalPolicy
from scripts.evaluate import evaluate

p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('gt');p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
policy=TemporalPolicy()
with (out/'decisions.jsonl').open('w',encoding='utf-8') as f:
    for r in map(json.loads,(Path(a.run)/'decisions.jsonl').read_text(encoding='utf-8').splitlines()):
        for o in r['objects']:o['assessment']=policy.apply(o['id'],r['time'],o['assessment'])
        f.write(json.dumps(r,ensure_ascii=False)+'\n')
(out/'detections.jsonl').write_bytes((Path(a.run)/'detections.jsonl').read_bytes())
r=evaluate(out/'decisions.jsonl',a.gt)
(out/'evaluation.json').write_text(json.dumps(r,indent=2))
print(r['counts'])
