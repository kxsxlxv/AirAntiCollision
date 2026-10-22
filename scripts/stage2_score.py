"""Offline detector and identity metrics. This module alone joins normal outputs to GT."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from stage2_audit import DEFAULT_DATA,REVISION,read_annotations
from mot_metrics import iou,sequence_data,compact,HOTA,Identity,CLEAR
from scipy.optimize import linear_sum_assignment
from airavoid.provenance import file_sha256


def annotations(clip):
    meta=json.loads((DEFAULT_DATA/'NPS'/clip/'meta.json').read_text())
    path=DEFAULT_DATA/'revisions'/REVISION/'annotations/NPS'/f'{clip}.csv'
    audit=json.loads((ROOT/'reports/stage2_dataset_audit.json').read_text())
    if file_sha256(path)!=audit['clips'][clip]['csv_sha256']:
        raise ValueError('GT changed since split/audit was frozen')
    return read_annotations(path,meta)


def gt_array(records):
    return np.asarray([[r['frame_id']+1,r['object_id'],r['x'],r['y'],r['w'],r['h']] for r in records],float).reshape(-1,6)


def match_boxes(gt,pred,threshold=.5):
    overlap=iou(gt,pred)
    if not overlap.size:return [],overlap
    # Prioritize valid match cardinality before overlap quality.
    rows,cols=linear_sum_assignment(np.where(overlap>=threshold,-1.-overlap,0.) )
    return [(int(r),int(c)) for r,c in zip(rows,cols) if overlap[r,c]>=threshold],overlap


def detection_score(rows,gt,threshold,classes=None):
    stats=Counter();by_size=defaultdict(Counter);by_conf=defaultdict(Counter);ious=[];centers=[];scales=[]
    byframe=defaultdict(list)
    for r in gt:byframe[r['frame_id']].append(r)
    for row in rows:
        g=byframe[row['frame']]
        d=[x for x in row['detections'] if classes is None or int(x[5]) in classes]
        gb=np.array([[x['x'],x['y'],x['x']+x['w'],x['y']+x['h']] for x in g]).reshape(-1,4)
        pb=np.array([x[:4] for x in d]).reshape(-1,4)
        matches,overlap=match_boxes(gb,pb,threshold)
        matched={a:b for a,b in matches};matched_pred={b for a,b in matches}
        stats.update(gt=len(g),pred=len(d),tp=len(matches),fp=len(d)-len(matches),fn=len(g)-len(matches))
        for index,x in enumerate(g):
            span=max(x['w'],x['h']);key='<16' if span<16 else '16-31' if span<32 else '32-63' if span<64 else '>=64'
            by_size[key]['gt']+=1;by_size[key]['matched']+=index in matched
            if index in matched:
                j=matched[index];ious.append(overlap[index,j]);centers.append(float(np.linalg.norm((gb[index,:2]+gb[index,2:]-pb[j,:2]-pb[j,2:])/2)))
                scales.append(float(np.log(max(pb[j,2:]-pb[j,:2])/span)))
        for j,b in enumerate(d):
            key='.03-.099' if b[4]<.1 else '.1-.299' if b[4]<.3 else '>=.3'
            by_conf[key]['pred']+=1;by_conf[key]['matched']+=j in matched_pred
    return {**stats,'recall':stats['tp']/max(1,stats['gt']),'precision_wrt_supplied_labels':stats['tp']/max(1,stats['pred']),
            'median_matched_iou':float(np.median(ious)) if ious else None,
            'median_center_error_px':float(np.median(centers)) if centers else None,
            'log_span_error_rms':float(np.sqrt(np.mean(np.square(scales)))) if scales else None,
            'target_pixel_span_strata':by_size,'detector_confidence_strata':by_conf,
            'precision_limit':'Conditional on supplied visible-target annotation completeness; unlabelled targets cannot be audited by this metric'}


def run(args):
    protocol=json.loads((ROOT/'reports/stage2_protocol.json').read_text())
    report={'protocol_sha256':file_sha256(ROOT/'reports/stage2_protocol.json'),'gt_revision':REVISION,
            'split':{k:protocol[k] for k in ('development','evaluation')},
            'detector_training_independence':'user confirmed A and B; no independent training manifest audit',
            'detectors':{},'MOT':{},'physical_collision_accuracy':'not measured; NPS has no physical CPA truth'}
    allgt={c:annotations(c) for c in protocol['development']+protocol['evaluation']}
    for detector in ['A','B']:
        per={}
        for clip,gt in allgt.items():
            folder=ROOT/f'artifacts/stage2/detections/{detector}/{clip}'
            rows=[json.loads(x) for x in (folder/'detections.jsonl').read_text().splitlines()]
            per[clip]={str(th):detection_score(rows,gt,th,{0}) for th in protocol['IoU_thresholds']}
            if detector=='A':per[clip]['class_agnostic_diagnostic_at_.5']=detection_score(rows,gt,.5)
        report['detectors'][detector]=per
        for tracker in ['custom','tracktrack','putr']:
            folder=ROOT/f'artifacts/stage2/tracks/{detector}/{tracker}/s{args.stride}'
            if not (folder/'benchmark.json').exists():continue
            per={};metrics=[HOTA(),Identity(),CLEAR({'PRINT_CONFIG':False})]
            combined={split:{m.get_name():{} for m in metrics} for split in ['development','evaluation']}
            for clip,gt in allgt.items():
                path=folder/f'{clip}.jsonl'
                if not path.exists():continue
                rows=[json.loads(x) for x in path.read_text().splitlines()]
                data=sequence_data(rows,gt_array(gt))
                result={m.get_name():m.eval_sequence(data) for m in metrics}
                per[clip]=compact(result)
                split='development' if clip in protocol['development'] else 'evaluation'
                for name,r in result.items():combined[split][name][clip]=r
            report['MOT'][f'{detector}/{tracker}/s{args.stride}']={
                'per_sequence':per,'combined':{s:compact({m.get_name():m.combine_sequences(mr[m.get_name()]) for m in metrics})
                                             for s,mr in combined.items() if all(mr[m.get_name()] for m in metrics)}}
    path=ROOT/f'reports/stage2_detection_mot_s{args.stride}.json'
    path.write_text(json.dumps(report,indent=2))
    for key,r in report['MOT'].items():
        if 'evaluation' in r['combined']:
            x=r['combined']['evaluation'];print(key,'eval IDF1',x['Identity']['IDF1'],'IDSW',x['CLEAR']['IDSW'],'HOTA',x['HOTA']['HOTA'])
    print('Report:',path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stride',type=int,default=1);run(p.parse_args())
