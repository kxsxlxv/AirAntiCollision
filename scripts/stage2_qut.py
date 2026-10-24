"""QUT category-proxy comparison from original full-rate detector caches.

Inference/association first, offline approximate-centre category join second.
No GT IDs or categories are available to the tracker or assessment functions.
"""
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from airavoid.tracking import Tracker
from airavoid.assessment import METHODS
from airavoid.provenance import file_sha256,validate_cache_entry
from stage2_assess_trace import replay,text_open
from evaluate import match


def observations(sequence,stride,output):
    source=ROOT/'artifacts/cadence_fullrate'/sequence
    benchmark=json.loads((source/'benchmark.json').read_text())
    manifest=json.loads((ROOT/f'data/qut_full/{sequence}.manifest.json').read_text())
    if benchmark['args']['conf']!=.03 or benchmark['args']['stride']!=1:
        raise ValueError('Wrong historical detector condition')
    video=ROOT/benchmark['args']['video']
    if file_sha256(video)!=manifest['sha256'] or benchmark.get('input_sha256')!=manifest['sha256']:
        raise ValueError('Historical source hash mismatch')
    identity=np.eye(3);tracker=Tracker(image_edge_margin=1.)
    count=0
    with (source/'detections.jsonl').open() as dfile,(source/'decisions.jsonl').open() as cfile,text_open(output,'wt') as out:
        for dl,cl in zip(dfile,cfile,strict=True):
            d=json.loads(dl);old=json.loads(cl);f=d['frame'];t=d['time']
            if f!=old['frame']:raise ValueError('Historical camera/cache frame mismatch')
            if f%stride:continue
            validate_cache_entry(d,f,t)
            tracks=tracker.update(d['detections'],t,identity,(1280,720))
            measurements=[{'track_id':tr.id,'measured_bbox':tr.box,'confidence':tr.confidence,
                           'class_id':tr.class_id,'time':t,'source':'detector','measured':True,
                           'ambiguous':t<tr.ambiguous_until} for tr in tracks if tr.last==t]
            out.write(json.dumps({'frame':f,'time':t,'source_pts_s':old['source_pts_ms']/1000,
                                  'camera':old['camera'],'stabilization_transform':identity.tolist(),
                                  'tracks':[[tr.id,*tr.box] for tr in tracks if tr.last==t],
                                  'measurements':measurements,'detector_variant':'A',
                                  'normal_inference_gt_access':False})+'\n')
            count+=1
    return {'frames':count,'source_sha256':manifest['sha256'],'detection_cache_sha256':file_sha256(source/'detections.jsonl'),
            'historical_benchmark_sha256':file_sha256(source/'benchmark.json'),
            'observation_trace_sha256':file_sha256(output),'historical_detector_args':benchmark['args'],
            'camera_contract':'Frozen original full-rate fixed-setup camera diagnostics; cadence changes association/assessment only',
            'category_contract':'Previously developed-on approach vs other; no physical collision truth'}


def score(sequence,path,stride):
    # GT is opened only after normal inference has written the completed trace.
    gt=json.loads((ROOT/f'data/qut_full/{sequence}.gt.json').read_text())
    labels=defaultdict(list)
    for g in gt:labels[g['frame']].append(g)
    counts={m:defaultdict(Counter) for m in METHODS};raw={m:defaultdict(Counter) for m in METHODS}
    allwarnings={m:0 for m in METHODS};oldstates={m:{} for m in METHODS}
    events={m:defaultdict(list) for m in METHODS};frames=0;lastt=0
    with text_open(path) as stream:
        for text in stream:
            row=json.loads(text);t=row['time'];lastt=t;frames+=1
            # Reproduce the earlier proxy's <=.25s freshness contract.
            objects=[o for o in row['objects'] if t-o['measurement_time_s']<=.25]
            gs=labels[row['frame']];matched=match(objects,gs,20)
            for method in METHODS:
                for obj in objects:
                    tid=obj['track_id'];warning=obj['results'][method]['decision']=='potential_conflict'
                    prior=oldstates[method].get(tid)
                    if warning and (not prior or not prior[0] or t-prior[1]>.3):allwarnings[method]+=1
                    oldstates[method][tid]=(warning,t)
                for j,g in enumerate(gs):
                    e=objects[matched[j]]['results'][method] if j in matched else None
                    state=e['decision'] if e else 'untracked';kind=g['kind']
                    counts[method][kind][state]+=1
                    raw[method][kind][e['raw_decision'] if e else 'untracked']+=1
                    events[method][g['id']].append((t,state))
    results={}
    for method in METHODS:
        per_event=[]
        for gid,values in events[method].items():
            warnings=[t for t,s in values if s=='potential_conflict']
            per_event.append({'category_id':gid,'category':gid.split('_')[0],
                              'first_warning_s':min(warnings) if warnings else None,
                              'first_warning_after_first_label_s':min(warnings)-values[0][0] if warnings else None,
                              'first_warning_before_last_label_s':values[-1][0]-min(warnings) if warnings else None,
                              'apparent_nonconflict_samples':sum(s=='apparent_nonconflict' for _,s in values),
                              'warning_samples':len(warnings),'reference':'first/last available approximate-centre annotation, not impact'})
        results[method]={'category_counts':counts[method],'raw_category_counts':raw[method],
                         'all_predicted_warning_episodes':allwarnings[method],
                         'all_predicted_warning_episodes_per_camera_hour':allwarnings[method]*3600/(lastt+stride/25),
                         'events':per_event}
    return {'sequence':sequence,'stride':stride,'frames':frames,'source_fps':25,'sampled_fps':25/stride,
            'methods':results,'centroid_matching_tolerance_px':20,
            'scope':'C: category proxy, approximate centres, incomplete distractor labels, development-influenced sources',
            'no_physical_false_safe_claim':True}


def run(args):
    output=ROOT/'artifacts/stage2/qut';output.mkdir(parents=True,exist_ok=True)
    report={'sequences':args.sequences.split(','),'strides':[int(x) for x in args.strides.split(',')],
            'methods':list(METHODS),'runs':{},'definition':'QUT approach-vs-other category proxy; not true collisions/safety',
            'conf':.03,'detector':'unchanged A, original full-rate actual TensorRT observations',
            'default_changes':False,'started':time.time()}
    path=ROOT/'reports/stage2_qut_comparison.json'
    for sequence in report['sequences']:
        for stride in report['strides']:
            label=f'{sequence}/s{stride}';folder=output/sequence/f's{stride}';folder.mkdir(parents=True,exist_ok=True)
            source=folder/'observations.jsonl.gz';assessed=folder/'assessments.jsonl.gz'
            provenance=observations(sequence,stride,source)
            summary=replay(source,assessed,1280,720,METHODS,'fixed','raw')
            result=score(sequence,assessed,stride);result['provenance']=provenance
            result['assessment_trace_sha256']=summary['output_trace_sha256']
            report['runs'][label]=result
            path.write_text(json.dumps(report,indent=2));print('Complete QUT '+label,flush=True)
    report.update(state='complete',finished=time.time());path.write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequences',default='M1,M2,M6,M7,S1,S2');p.add_argument('--strides',default='1,2,5')
    run(p.parse_args())
