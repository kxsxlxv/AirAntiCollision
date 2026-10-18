"""Known 2D warps of real RGB, detector-cache replay. Not real flight/3D simulation."""
import json
from pathlib import Path
import cv2
import numpy as np
from airavoid.tracking import CameraMotion,Tracker
from airavoid.geometry import ThreatEstimator
from airavoid.policy import TemporalPolicy
from scripts.evaluate import evaluate
import argparse

p=argparse.ArgumentParser();p.add_argument('--version',default='v2');args=p.parse_args()
src=Path('demo/M2_tiled_initial')
records=[json.loads(x) for x in (src/'detections.jsonl').read_text().splitlines()]
summary=[]
for condition in ['identity','pan','zoom','roll']:
    for mode in ['fixed','auto']:
        out=Path('artifacts/camera_stress_'+args.version)/f'{condition}_{mode}';out.mkdir(parents=True,exist_ok=True)
        camera=CameraMotion(mode);tracker=Tracker();estimator=ThreatEstimator();policy=TemporalPolicy()
        cap=cv2.VideoCapture('data/qut/M2.mp4');errors=[];valid=0;frames=0;reference=np.eye(3)
        gt_source=json.loads(Path('data/qut/M2.gt.json').read_text());gt_frames={}
        for r in gt_source:gt_frames.setdefault(r['frame'],[]).append(r)
        new_gt=[]
        with (out/'decisions.jsonl').open('w',encoding='utf-8') as f,(out/'detections.jsonl').open('w') as df:
            for rec in records[:250]:
                idx=rec['frame'];t=rec['time'];cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,image=cap.read()
                if not ok:break
                h,w=image.shape[:2]
                zoom=1+.025*t if condition=='zoom' else 1.
                angle=1.3*t if condition=='roll' else 0.
                affine=cv2.getRotationMatrix2D((w/2,h/2),angle,zoom)
                if condition=='pan':affine[:,2]+=np.array([10*t,2*t])
                known=np.eye(3);known[:2]=affine
                warped=cv2.warpAffine(image,affine,(w,h),borderMode=cv2.BORDER_REFLECT_101)
                boxes=[]
                for b in rec['detections']:
                    corners=np.array([[b[0],b[1],1],[b[2],b[1],1],[b[2],b[3],1],[b[0],b[3],1]])
                    xy=corners@affine.T;lo=xy.min(axis=0);hi=xy.max(axis=0)
                    if lo[0]<0 or lo[1]<0 or hi[0]>w or hi[1]>h:continue
                    boxes.append([float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1]),b[4],b[5]])
                for r in gt_frames.get(idx,[]):
                    xy=affine@np.array([r['u'],r['v'],1.])
                    if 0<=xy[0]<w and 0<=xy[1]<h:new_gt.append({**r,'u':float(xy[0]),'v':float(xy[1])})
                transform,cam=camera.update(warped,t,boxes)
                if cam.get('evidence_reset'):
                    tracker.reset_evidence();policy.pending.clear()
                    if cam['valid']:reference=known.copy()
                valid+=int(cam['valid']);frames+=1
                if cam['valid']:
                    grid=np.array([[320,180,1],[640,360,1],[960,540,1]],float)
                    estimated=grid@known.T@transform.T
                    expected=grid@reference.T
                    errors.extend(np.linalg.norm(estimated[:,:2]-expected[:,:2],axis=1).tolist())
                objects=[]
                for tr in tracker.update(boxes,t,transform):
                    if tr.hits<3:continue
                    e=policy.apply(tr.id,t,estimator.assess(tr.history,t,cam['valid'],t<tr.ambiguous_until))
                    objects.append({'id':tr.id,'box':tr.box,'last_seen':tr.last,'assessment':e})
                f.write(json.dumps({'frame':idx,'time':t,'objects':objects,'camera':cam})+'\n')
                df.write(json.dumps({'frame':idx,'time':t,'detections':boxes})+'\n')
        (out/'gt_offline.json').write_text(json.dumps(new_gt))
        evaluation=evaluate(out/'decisions.jsonl',out/'gt_offline.json')
        (out/'evaluation.json').write_text(json.dumps(evaluation,indent=2))
        row={'condition':condition,'mode':mode,'frames':frames,'camera_valid_frames':valid,'median_stabilization_error_px':float(np.median(errors)) if errors else None,'p95_stabilization_error_px':float(np.percentile(errors,95)) if errors else None,'counts':evaluation['counts']}
        summary.append(row);print(row,flush=True)
        Path(f'reports/camera_stress_{args.version}.json').write_text(json.dumps({'kind':'2D augmentation/cache replay, not real airborne validation','details':summary},indent=2))
