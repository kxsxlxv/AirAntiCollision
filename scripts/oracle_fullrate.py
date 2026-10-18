"""Offline GT-centroid/identity ablation with independent real detector span samples.

Not camera-only inference, not physical collision truth, not a prediction upper
bound. Approximate/interpolated GT positions are deliberately supplied offline to
separate association/centroid limitations from span/model limitations.
"""
from collections import Counter,defaultdict,deque
import json
from pathlib import Path
import time
from airavoid.geometry import Config,ThreatEstimator
from airavoid.policy import TemporalPolicy
from scripts.evaluate import match


def main():
    deadline=time.monotonic()+4000
    while True:
        path=Path('reports/cadence_all_sources.json')
        if path.exists() and json.loads(path.read_text()).get('complete'):break
        if time.monotonic()>deadline:raise TimeoutError('Complete-source cadence validation not finished')
        time.sleep(3)
    global_counts=defaultdict(Counter);results=[]
    for seq in [f'M{i}' for i in range(1,9)]+[f'S{i}' for i in range(1,8)]:
        root=Path('artifacts/cadence_fullrate')/seq;b=json.loads((root/'benchmark.json').read_text());width,height=b['resolution']
        rows={r['frame']:r for r in map(json.loads,(root/'decisions.jsonl').read_text(encoding='utf-8').splitlines())}
        gt=defaultdict(list)
        for label in json.loads(Path(f'data/qut_full/{seq}.gt.json').read_text()):gt[label['frame']].append(label)
        centers=defaultdict(lambda:deque(maxlen=500));spans=defaultdict(lambda:deque(maxlen=500))
        estimator=ThreatEstimator(Config(correlation_seconds=1.));policy=TemporalPolicy();counts=defaultdict(Counter)
        for entry in map(json.loads,(root/'detections.jsonl').read_text().splitlines()):
            t=entry['time'];labels=gt[entry['frame']];boxes=entry['detections'];pairs=match(boxes,labels,20)
            camera_valid=rows[entry['frame']]['camera']['valid']
            for j,label in enumerate(labels):
                ident=label['id'];h=centers[ident];s=spans[ident];clipped=False
                if h and t-h[-1][0]>.8:h.clear();s.clear()
                if j in pairs:
                    box=boxes[pairs[j]];span=max(box[2]-box[0],box[3]-box[1]);clipped=box[0]<=1 or box[1]<=1 or box[2]>=width-1 or box[3]>=height-1
                    if clipped:h.clear();s.clear()
                    if s and t-s[-1][0]>.8:s.clear()
                    if not clipped:s.append((t,label['u'],label['v'],span))
                h.append((t,label['u'],label['v'],s[-1][3] if s else 1.))
                e=estimator.assess(h,t,camera_valid,False,scale_history=s)
                if clipped and camera_valid:e.update(state='uncertain',reason='clipped')
                e=policy.apply(ident,t,e);c=counts[label['kind']];c['total']+=1;c[e['state']]+=1;c['reason_'+e['reason']]+=1
                c['matched_detector_size_now']+=int(j in pairs);c['source_clipped_now']+=int(clipped)
        for kind,c in counts.items():global_counts[kind].update(c)
        results.append({'sequence':seq,'counts':counts})
        report={'kind':'OFFLINE approximate GT centroid + identity, independent actual neural spans; NOT camera-only inference',
            'config':{'window_s':4,'min_history_s':2,'HAC_s':1,'source_edge_margin_px':1,'size_match_tolerance_px':20},
            'complete':len(results)==15,'counts':global_counts,'runs':results,
            'limits':['GT identity/centroid deliberately available in this ablation only; zero untracked by construction',
                      'No object size is fabricated on a detector miss; only independent matched size samples enter its fit',
                      'Annotation coordinates are approximate/interpolated, not true bearings; intervals remain heuristic',
                      'Ranges and category labels are not passed to the estimator','No collision/CPA truth or calibrated upper-bound claim']}
        Path('reports/oracle_fullrate.json').write_text(json.dumps(report,indent=2))
        print(seq,counts,flush=True)


if __name__=='__main__':main()
