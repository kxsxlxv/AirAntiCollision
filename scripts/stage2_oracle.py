"""Offline identity/box upper bounds. GT is permitted ONLY in these labelled traces.

Identity-only keeps every emitted predicted measurement and its exact box/confidence,
relabels matched measurements by offline Hungarian IoU>=.5, resolves their ambiguity,
and leaves unmatched hypotheses under non-overlapping offset predicted IDs. It cannot
restore missed detections or tracker birth delays. Box+ID replaces all measurements
by the verified visible GT boxes. Camera transforms/validity are copied unchanged
from the normal predicted trace; no GT camera estimate is made.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from stage2_score import annotations,match_boxes
from airavoid.provenance import file_sha256


def oracle_rows(rows,gt,condition):
    if condition not in ['oracle_identity_only','oracle_box_and_identity']:raise ValueError(condition)
    byframe=defaultdict(list)
    for r in gt:byframe[r['frame_id']].append(r)
    offset=max((r['object_id'] for r in gt),default=0)+1000000
    for original in rows:
        row=dict(original);labels=byframe[row['frame']];obs=row['measurements'];measures=[]
        if condition=='oracle_identity_only':
            boxes=[[x['x'],x['y'],x['x']+x['w'],x['y']+x['h']] for x in labels]
            pairs,_=match_boxes(boxes,[o['measured_bbox'] for o in obs],.5)
            matched={b:labels[a]['object_id'] for a,b in pairs};idmap={}
            for i,o in enumerate(obs):
                item=dict(o);newid=matched.get(i,offset+int(o['track_id']));idmap[o['track_id']]=newid
                item['track_id']=newid
                if i in matched:item['ambiguous']=False
                measures.append(item)
            row['tracks']=[[float(idmap.get(int(tr[0]),offset+int(tr[0]))),*tr[1:]] for tr in row.get('tracks',[])]
        else:
            for i,r in enumerate(labels):
                measures.append({'track_id':r['object_id'],'measured_bbox':[r['x'],r['y'],r['x']+r['w'],r['y']+r['h']],
                                 'confidence':None,'class_id':0,'time':row['time'],'source':'oracle_box',
                                 'measured':True,'ambiguous':False})
            row['tracks']=[[float(o['track_id']),*o['measured_bbox']] for o in measures]
        row['measurements']=measures;row['input_truth_condition']=condition
        row['oracle_warning']='OFFLINE GT UPPER BOUND; never normal inference'
        row['normal_inference_gt_access']=False;row['offline_oracle_gt_access']=True
        yield row


def build(path,clip,condition,output):
    path=Path(path);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    n=0
    with path.open() as src,output.open('w') as dst:
        for row in oracle_rows(map(json.loads,src),annotations(clip),condition):
            dst.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n');n+=1
    report={'condition':condition,'clip':clip,'frames':n,'source_sha256':file_sha256(path),
            'oracle_implementation_sha256':file_sha256(Path(__file__)),
            'offline_matcher_source_sha256':file_sha256(ROOT/'scripts/stage2_score.py'),
            'output_sha256':file_sha256(output),'camera_source':'unchanged normal predicted trace',
            'identity_only_scope':'same emitted measured boxes/confidences, no recovery of misses/births; offline IoU oracle',
            'offline_gt_access':True,'production_use':False}
    output.with_suffix('.summary.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('--clip',required=True)
    p.add_argument('--condition',choices=['oracle_identity_only','oracle_box_and_identity'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(build(a.trace,a.clip,a.condition,a.output))
