"""Audit exclusion of invalid clipped sizes, on fixed actual full-rate observations."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import time
from unittest.mock import patch
from airavoid.tracking import Tracker
from airavoid.provenance import file_sha256
from scripts import reassess_fixed
from scripts.evaluate import evaluate


class ExcludeClippedTracker(Tracker):
    def _observe(self,tr,*args,**kwargs):
        super()._observe(tr,*args,**kwargs)
        if tr.clipped:tr.history.clear();tr.scale_history.clear()


def main():
    deadline=time.monotonic()+4000
    while True:
        p=Path('reports/cadence_all_sources.json')
        if p.exists() and json.loads(p.read_text()).get('complete'):break
        if time.monotonic()>deadline:raise TimeoutError('All-source observations unavailable')
        time.sleep(3)
    results=[];counts=defaultdict(Counter)
    for seq in [f'M{i}' for i in range(1,9)]+[f'S{i}' for i in range(1,8)]:
        source=Path('artifacts/cadence_fullrate')/seq;out=Path('artifacts/clip_history_audit')/seq
        with patch.object(reassess_fixed,'Tracker',ExcludeClippedTracker):reassess_fixed.reassess(source,out,image_edge_margin=1)
        before=[json.loads(s) for s in (source/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
        after=[json.loads(s) for s in (out/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
        state_differences=[];reason_differences=0;numeric_frames=0
        for a,b in zip(before,after):
            assert [o['id'] for o in a['objects']]==[o['id'] for o in b['objects']]
            changed=False
            for x,y in zip(a['objects'],b['objects']):
                u,v=x['assessment'],y['assessment']
                if u['state']!=v['state']:state_differences.append({'frame':a['frame'],'time':a['time'],'id':x['id'],'old':u['state'],'new':v['state']})
                reason_differences+=u['reason']!=v['reason']
                changed|=any(u.get(k)!=v.get(k) for k in ['samples','span','growth','drift','miss_spans','tau'])
            numeric_frames+=int(changed)
        r=evaluate(out/'decisions.jsonl',f'data/qut_full/{seq}.gt.json');(out/'evaluation.json').write_text(json.dumps(r,indent=2))
        for kind,c in r['counts'].items():counts[kind].update(c)
        results.append({'sequence':seq,'state_differences':state_differences,'reason_different_object_samples':reason_differences,
                        'numeric_different_frames':numeric_frames,'counts':r['counts'],'events':r['events']})
        report={'kind':'Clipped-size exclusion candidate; geometry-only replay of actual full-rate neural observations, no GT online',
                'complete':len(results)==15,'candidate_code_sha256':file_sha256(__file__),'results':results,'counts':counts,
                'limits':['No new neural inference/performance claim','No physical collision ground truth','All available studied sources, not an independent test']}
        Path('reports/clip_history_audit.json').write_text(json.dumps(report,indent=2))
        print(seq,'changed_states',len(state_differences),'changed_numeric_frames',numeric_frames,flush=True)


if __name__=='__main__':main()
