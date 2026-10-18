"""Raw fixed-YOLO coverage on full UAVSwarm labels, before any tracking."""
from pathlib import Path
import json,sys
import numpy as np
from mot_metrics import iou,ROOT
from scipy.optimize import linear_sum_assignment
results=[]
for seq in ['02','06','28']:
    gt=np.loadtxt(ROOT/f'data/mot_swarm/UAVSwarm-{seq}/gt/gt.txt',delimiter=',',ndmin=2)
    rows=[json.loads(x) for x in (ROOT/f'artifacts/mot/swarm_detections/sw{seq}.jsonl').read_text().splitlines()]
    tp=0;nd=0;spans=[]
    for row in rows:
        g=gt[gt[:,0]==row['frame']+1];boxes=g[:,2:6].copy();spans.extend(np.max(boxes[:,2:],axis=1));boxes[:,2:]+=boxes[:,:2]
        d=np.array(row['detections']).reshape(-1,6);nd+=len(d);s=iou(boxes,d[:,:4])
        if s.size:
            a,b=linear_sum_assignment(-(s>=.5).astype(float)-s*.001);tp+=int((s[a,b]>=.5).sum())
    results.append(dict(sequence='sw'+seq,gt_boxes=len(gt),detections=nd,tp_iou50=tp,
        recall_iou50=tp/len(gt),precision_iou50=tp/max(1,nd),median_gt_span=float(np.median(spans))))
r=dict(scope='Raw post-NMS fixed YOLO26m boxes, all classes pooled; not a tracker upper bound because filters can change localization',runs=results,
    combined=dict(gt_boxes=sum(x['gt_boxes'] for x in results),detections=sum(x['detections'] for x in results),tp_iou50=sum(x['tp_iou50'] for x in results)))
r['combined']['recall_iou50']=r['combined']['tp_iou50']/r['combined']['gt_boxes']
(ROOT/'reports/mot_detector_coverage.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
