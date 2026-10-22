"""Offline NPS identity/geometry join to assessment outputs; no physical threat labels."""
import argparse
from collections import Counter,defaultdict
import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from stage2_score import annotations,match_boxes
from stage2_assess_trace import text_open
from airavoid.assessment import METHODS
from airavoid.provenance import file_sha256


def score(path,clip):
    gt=annotations(clip);byframe=defaultdict(list)
    for r in gt:byframe[r['frame_id']].append(r)
    audit=json.loads((ROOT/'reports/stage2_dataset_audit.json').read_text())
    width,height=audit['clips'][clip]['resolution']
    pts=json.loads((ROOT/f'data/stage2/{clip}_pts.json').read_text())['pts_s']
    pts=np.array(pts)-pts[0]
    stats=defaultdict(Counter);raw=defaultdict(Counter)
    all_predicted=defaultdict(Counter)
    strata=defaultdict(lambda:defaultdict(Counter));reasons=defaultdict(Counter)
    episodes=Counter();previous=defaultdict(dict);events=defaultdict(lambda:defaultdict(list))
    scale_previous={};scale_increments=[];camera=Counter();dts=[];frames=0
    first_t=last_t=None;condition=None
    summary=Path(path).with_suffix('.summary.json')
    methods=json.loads(summary.read_text())['methods'] if summary.exists() else list(METHODS)
    with text_open(path) as src:
        for text in src:
            row=json.loads(text);t=row['time'];objects=row['objects'];camera[row['camera']['status']]+=1
            if first_t is None:first_t=t
            if last_t is not None:dts.append(t-last_t)
            last_t=t;frames+=1;condition=row.get('input_truth_condition')
            current_methods=set(objects[0]['results']) if objects else set()
            if methods and current_methods and set(methods)!=current_methods:raise ValueError('Methods changed within trace')
            g=byframe[row['frame']]
            boxes=np.array([[r['x'],r['y'],r['x']+r['w'],r['y']+r['h']] for r in g]).reshape(-1,4)
            eligible=[o for o in objects if abs(t-o['measurement_time_s'])<1e-6]
            pairs,_=match_boxes(boxes,[o['box'] for o in eligible],.5)
            matched={a:eligible[b] for a,b in pairs}
            for method in methods or []:
                for obj in objects:
                    e=obj['results'][method];tid=obj['track_id'];old=previous[method].get(tid)
                    all_predicted[method][e['decision']]+=1
                    warning=e['decision']=='potential_conflict'
                    if warning and (old is None or not old[0] or t-old[1]>.3):episodes[method]+=1
                    previous[method][tid]=(warning,t)
                for i,label in enumerate(g):
                    span=max(label['w'],label['h']);size='<16' if span<16 else '16-31' if span<32 else '32-63' if span<64 else '>=64'
                    center=np.array([label['x']+label['w']/2,label['y']+label['h']/2])
                    offset=float(np.linalg.norm((center-[width/2,height/2])/[width/2,height/2]))
                    offset_bin='<.3' if offset<.3 else '.3-.699' if offset<.7 else '>=.7'
                    obj=matched.get(i)
                    state=obj['results'][method]['decision'] if obj else 'untracked'
                    stats[method][state]+=1
                    raw[method][obj['results'][method]['raw_decision'] if obj else 'untracked']+=1
                    strata[method]['pixel_span/'+size][state]+=1
                    strata[method]['camera/'+row['camera']['status']][state]+=1
                    strata[method]['normalized_pixel_offset/'+offset_bin][state]+=1
                    confidence=obj.get('confidence') if obj else None
                    conf='missing' if confidence is None else '<.1' if confidence<.1 else '.1-.299' if confidence<.3 else '>=.3'
                    strata[method]['confidence/'+conf][state]+=1
                    if obj:
                        e=obj['results'][method];reasons[method].update(e['reason_codes'])
                        if state=='potential_conflict':events[method][label['object_id']].append(t)
            for i,obj in matched.items():
                label=g[i];gid=label['object_id'];dspan=obj['measured_raw_span'];gspan=max(label['w'],label['h'])
                old=scale_previous.get(gid)
                if old and 0<t-old[0]<.2 and old[3]==obj['track_id']:
                    scale_increments.append(float((np.log(dspan)-np.log(old[1]))-(np.log(gspan)-np.log(old[2]))))
                scale_previous[gid]=(t,dspan,gspan,obj['track_id'])
    if not frames:raise ValueError('Empty assessment trace')
    step=float(np.median(dts)) if dts else float(np.median(np.diff(pts)))
    duration=last_t-first_t+step
    target_bounds={}
    for r in gt:
        gid=r['object_id'];f=r['frame_id']
        if gid not in target_bounds:target_bounds[gid]=[f,f]
        target_bounds[gid]=[min(target_bounds[gid][0],f),max(target_bounds[gid][1],f)]
    metrics={}
    for m in methods or []:
        timing=[]
        for gid,times in events[m].items():
            first,last=target_bounds[gid];lengths=[];start=prior=times[0]
            for now in times[1:]:
                if now-prior>step*1.5:lengths.append(prior-start+step);start=now
                prior=now
            lengths.append(prior-start+step)
            timing.append({'gt_id':gid,'first_warning_s':min(times),
                           'after_first_label_s':min(times)-float(pts[first]),
                           'before_last_label_s':float(pts[last])-min(times),
                           'warning_samples':len(times),'warning_duration_s':len(times)*step,
                           'longest_contiguous_warning_s':max(lengths),
                           'reference':'first/last visible annotation, not impact'})
        total=sum(stats[m].values())
        metrics[m]={'target_frame_opportunities':total,'states':stats[m],
                    'states_denominator':'visible GT target-frame opportunities; untracked means no fresh measured IoU>=.5 match, not necessarily absence of a stale hypothesis',
                    'all_predicted_track_sample_states':all_predicted[m],
                    'decisive_coverage':sum(stats[m][s] for s in ['potential_conflict','apparent_nonconflict'])/max(1,total),
                    'untracked_fraction':stats[m]['untracked']/max(1,total),
                    'raw_candidate_states':raw[m],'reason_codes':reasons[m],'strata':strata[m],
                    'all_predicted_warning_episodes':episodes[m],
                    'all_predicted_warning_episodes_per_camera_hour':episodes[m]*3600/duration,
                    'warning_reference_events':timing}
    return {'sequence':clip,'source_trace_sha256':file_sha256(path),'frames':frames,'camera_duration_s':duration,
            'scorer_source_sha256':file_sha256(Path(__file__)),
            'offline_matcher_source_sha256':file_sha256(ROOT/'scripts/stage2_score.py'),
            'input_truth_condition':condition,'camera_status_frames':camera,
            'bearing_angle_strata':'unavailable: no verified intrinsics; normalized pixel offset reported instead',
            'cadence_sample_dt_s':step,'methods':metrics,
            'paired_bbox_log_scale_increment_error_rms':float(np.sqrt(np.mean(np.square(scale_increments)))) if scale_increments else None,
            'scale_increment_pairs':len(scale_increments),
            'scale_increment_scope':'consecutive matched GT object, same predicted ID, delta <= .2 s; no cadence normalization',
            'truth_tier':'B: recorded 2D geometry and per-clip identity, no physical collision/CPA truth',
            'GT_join':'offline only, fresh measured boxes, cardinality-first Hungarian IoU>=.5',
            'warning_precision_or_collision_recall':None}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('--clip',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=score(a.trace,a.clip);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2));print(a.output)
