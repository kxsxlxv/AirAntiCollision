"""Offline QUT encounter-category proxy metrics; never imported by online inference."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import numpy as np
from airavoid.policy import TemporalPolicy


def match(items,labels,tolerance):
    pairs=[]
    for i,o in enumerate(items):
        if 'box' in o:b=o['box']
        else:b=o[:4]
        x,y=(b[0]+b[2])/2,(b[1]+b[3])/2
        for j,l in enumerate(labels):
            dist=np.hypot(x-l['u'],y-l['v'])
            if dist<=tolerance:pairs.append((dist,i,j))
    ii=set();jj=set();result={}
    for dist,i,j in sorted(pairs):
        if i not in ii and j not in jj:ii.add(i);jj.add(j);result[j]=i
    return result


def evaluate(trace_path,gt_path,tolerance=20):
    trace=[json.loads(x) for x in Path(trace_path).read_text(encoding='utf-8').splitlines()]
    gts=defaultdict(list)
    for row in json.loads(Path(gt_path).read_text()):gts[row['frame']].append(row)
    detectors={r['frame']:r['detections'] for r in map(json.loads,Path(trace_path).with_name('detections.jsonl').read_text().splitlines())}
    counts=defaultdict(Counter);events=defaultdict(list);unmatched=Counter(); ids=defaultdict(set)
    baselines=defaultdict(Counter);baseline_policy=TemporalPolicy();baseline_events=defaultdict(list)
    for r in trace:
        labels=gts[r['frame']];objects=[o for o in r['objects'] if r['time']-o['last_seen']<=.25]
        baseline_states={}
        for o in r['objects']:
            e=o['assessment'];growth=e['growth'];se=e['growth_se']
            valid=e.get('candidate_reason',e['reason']) not in ['aspect','tiny','camera','lost','association','history','scale_history','scale_stale','clipped']
            resolved=e.get('growth_low')>.008 if e.get('growth_low') is not None else growth is not None and se is not None and growth>max(.008,2.5*se)
            candidate='threat' if valid and resolved else 'uncertain'
            if 'candidate_state' in e:
                candidate=baseline_policy.apply(o['id'],r['time'],{'state':candidate,'reason':'growth_only'})['state']
            baseline_states[o['id']]=candidate
        m=match(objects,labels,tolerance);dm=match(detectors.get(r['frame'],[]),labels,tolerance)
        for j,label in enumerate(labels):
            kind=label['kind'];state=objects[m[j]]['assessment']['state'] if j in m else 'untracked'
            counts[kind]['total']+=1;counts[kind][state]+=1
            counts[kind]['detector_queried']+=int(r.get('detector_ran',True))
            counts[kind]['detected']+=int(j in dm)
            events[label['id']].append((r['time'],state))
            if j in m:
                o=objects[m[j]];ids[label['id']].add(o['id']);e=o['assessment']
                counts[kind]['observed_by_'+o.get('source','detector')]+=1
                # Comparable inputs: same tracked boxes, time windows, camera constraints.
                growth_state=baseline_states[o['id']]
                baselines[kind]['growth_only_'+growth_state]+=1
                counts[kind]['reason_'+e['reason']]+=1
            else:
                growth_state='untracked';baselines[kind]['growth_only_untracked']+=1
            baseline_events[label['id']].append((r['time'],growth_state))
        used=set(m.values())
        for i,o in enumerate(objects):
            if i not in used:unmatched[o['assessment']['state']]+=1
    event_results=[]
    dt=np.median(np.diff([r['time'] for r in trace])) if len(trace)>1 else 0
    for ident,values in events.items():
        sustained=0;first=None;maxrun=0;false_safe=False
        for t,state in values:
            sustained=sustained+dt if state=='threat' else 0
            maxrun=max(maxrun,sustained)
            if sustained>=.8 and first is None:first=t-sustained+dt
            false_safe|=state=='nonthreat'
        baseline_run=0.;baseline_max=0.
        for tt,ss in baseline_events[ident]:
            baseline_run=baseline_run+dt if ss=='threat' else 0.
            baseline_max=max(baseline_max,baseline_run)
        event_results.append({'id':ident,'kind':'approach' if ident.startswith('approach') else 'other',
                              'sampled_frames':len(values),'sustained_warning':first is not None,'first_warning_time_s':first,
                              'max_warning_duration_s':maxrun,'any_apparently_non_threat':false_safe,
                              'associated_track_ids':sorted(ids[ident]),'fragments':len(ids[ident]),
                              'growth_only_baseline_sustained_warning':bool(baseline_max>=.8)})
    return {'definition':'QUT encounter-category proxy, not physical collision ground truth',
            'trace':str(trace_path),'gt':str(gt_path),'centroid_tolerance_px':tolerance,
            'warning_persistence_s':.8,'frames':len(trace),'counts':dict(counts),'growth_only_baseline':dict(baselines),
            'events':event_results,'unmatched_track_frames':unmatched,
            'limitations':['centroids approximate/interpolated','no GT boxes or CPA','other objects not necessarily exhaustively labeled','unmatched warnings are not verified false alerts','missed warning and false-safe on approach are distinct']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trace');p.add_argument('gt');p.add_argument('--output',default='reports/evaluation.json');p.add_argument('--tolerance',type=float,default=20)
    a=p.parse_args();r=evaluate(a.trace,a.gt,a.tolerance);Path(a.output).write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r,indent=2))
