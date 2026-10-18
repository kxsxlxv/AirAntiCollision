"""Geometry-only counterfactual using actual observations and cached camera validity.

Fixed-camera sources only: their stabilization transform is exactly identity.
No GT is read and no neural/decode/render performance is claimed.
"""
from collections import Counter
from dataclasses import asdict,fields
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from airavoid.geometry import Config,ThreatEstimator,advisory,apply_camera_scope
from airavoid.tracking import Tracker
from airavoid.policy import TemporalPolicy
from airavoid.detector import NAMES
from airavoid.provenance import file_sha256
from airavoid.projective import ProjectiveEstimator
from airavoid.bounded import BoundedProjectiveEstimator


def reassess(source,output,overrides=None,image_edge_margin=0.,verify=False,geometry='log'):
    source=Path(source);output=Path(output)
    if source.resolve()==output.resolve():raise ValueError('Reassessment output must differ from source run')
    output.mkdir(parents=True,exist_ok=True)
    benchmark=json.loads((source/'benchmark.json').read_text())
    rows=list(map(json.loads,(source/'decisions.jsonl').read_text(encoding='utf-8').splitlines()))
    observations={r['frame']:r for r in map(json.loads,(source/'detections.jsonl').read_text().splitlines())}
    if any(r['camera']['mode']!='fixed' for r in rows):raise ValueError('Only identity-transform fixed-camera runs can be reassessed from this trace')
    valid_fields={f.name for f in fields(Config)}
    params={k:v for k,v in benchmark['config'].items() if k in valid_fields};params.update(overrides or {})
    estimator={'log':ThreatEstimator,'projective':ProjectiveEstimator,'bounded':BoundedProjectiveEstimator}[geometry](Config(**params))
    policy=TemporalPolicy(**benchmark.get('policy_parameters',{}))
    tracker=Tracker(image_edge_margin=image_edge_margin);width,height=benchmark['resolution'];counts=Counter();mismatches=[];max_numeric_error=0.
    start=time.perf_counter()
    with (output/'decisions.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:
            t=row['time'];entry=observations[row['frame']];cam=row['camera']
            if cam.get('evidence_reset'):tracker.reset_evidence();policy.pending.clear()
            tracks=tracker.update(entry.get('observations',entry['detections']),t,np.eye(3),(width,height));objects=[]
            for tr in tracks:
                if tr.hits<3:continue
                e=estimator.assess(tr.history,t,cam['valid'],t<tr.ambiguous_until)
                e['clipped_observation']=bool(tr.clipped)
                if tr.clipped and cam['valid'] and t-tr.last<=.25:e.update(state='uncertain',reason='clipped')
                e=apply_camera_scope(e,cam)
                e=policy.apply(tr.id,t,e);counts[e['state']]+=1
                objects.append({'id':tr.id,'box':tr.box,'class_id':tr.class_id,'class':NAMES[tr.class_id],
                                'source':tr.source,'confidence':tr.confidence,'last_seen':tr.last,'assessment':e})
            if verify:
                actual=[(o['id'],o['assessment']['state'],o['assessment']['reason']) for o in objects]
                original=[(o['id'],o['assessment']['state'],o['assessment']['reason']) for o in row['objects']]
                if actual!=original:mismatches.append({'frame':row['frame'],'actual':actual,'source':original})
                if [o['id'] for o in objects]==[o['id'] for o in row['objects']]:
                    for new,old in zip(objects,row['objects']):
                        for key in ['growth','growth_se','drift','tau','miss_spans','miss_low','miss_high']:
                            v1,v2=new['assessment'].get(key),old['assessment'].get(key)
                            if v1 is None and v2 is None:continue
                            if v1 is None or v2 is None:
                                mismatches.append({'frame':row['frame'],'numeric_field':key,'actual':v1,'source':v2});continue
                            difference=abs(v1-v2);max_numeric_error=max(max_numeric_error,difference)
                            if difference>1e-8:mismatches.append({'frame':row['frame'],'numeric_field':key,'difference':difference})
            f.write(json.dumps({**row,'objects':objects,'advisory':advisory(objects,image_width=width)},ensure_ascii=False,allow_nan=False)+'\n')
    elapsed=time.perf_counter()-start
    shutil.copyfile(source/'detections.jsonl',output/'detections.jsonl')
    report={'kind':'Geometry-only fixed-camera reassessment of cached observations, not neural/decode/render throughput',
            'source_run':str(source),'source_benchmark_sha256':file_sha256(source/'benchmark.json'),
            'geometry':geometry,
            'source_observation_sha256':file_sha256(source/'detections.jsonl'),'input_sha256':benchmark.get('input_sha256'),
            'source_fps':benchmark['source_fps'],'resolution':benchmark['resolution'],'frames':len(rows),'config':asdict(estimator.config),
            'image_edge_margin':image_edge_margin,'policy_parameters':benchmark.get('policy_parameters',{}),
            'states_track_frames':counts,'reassessment_s':elapsed,'source_decision_verification_requested':verify,
            'source_decision_mismatch_frames':len({m['frame'] for m in mismatches}),'source_max_numeric_difference':max_numeric_error,'first_mismatches':mismatches[:10],'no_gt_used':True,
            'source_hashes':{p.name:file_sha256(p) for p in Path('airavoid').glob('*.py')}}
    (output/'benchmark.json').write_text(json.dumps(report,indent=2))
    if verify and mismatches:raise AssertionError('Baseline reconstruction did not reproduce source decisions; see benchmark.json')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',required=True)
    p.add_argument('--correlation-seconds',type=float);p.add_argument('--min-span',type=float);p.add_argument('--window',type=float)
    p.add_argument('--image-edge-margin',type=float,default=0.);p.add_argument('--verify',action='store_true');a=p.parse_args()
    overrides={key:value for key,value in [('correlation_seconds',a.correlation_seconds),('min_span',a.min_span),('window',a.window)] if value is not None}
    print(json.dumps(reassess(a.source,a.output,overrides,a.image_edge_margin,a.verify),indent=2))
