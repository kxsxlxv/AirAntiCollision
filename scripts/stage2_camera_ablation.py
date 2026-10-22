"""Independent ORB/background-homography diagnostic on frozen predicted tracks.

Uses only video and detector boxes as masks, never GT. A successful plane fit is
not independent camera-rotation estimation and is not proof of target stabilization.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import cv2
import av
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from airavoid.provenance import file_sha256


class ORBPlane:
    def __init__(self):
        self.orb=cv2.ORB_create(nfeatures=1000);self.prev=None;self.cumulative=np.eye(3)
        self.valid=False;self.epoch=0

    def update(self,image,boxes):
        h,w=image.shape[:2];scale=640/w;nh=round(h*scale)
        gray=cv2.cvtColor(cv2.resize(image,(640,nh)),cv2.COLOR_BGR2GRAY)
        mask=np.full(gray.shape,255,np.uint8)
        for b in boxes:
            x1,y1,x2,y2=np.rint(np.array(b[:4])*scale).astype(int)
            cv2.rectangle(mask,(x1-8,y1-8),(x2+8,y2+8),0,-1)
        kp,des=self.orb.detectAndCompute(gray,mask)
        valid=False;count=0;residual=None;coverage=None;matrix=None
        if self.prev and des is not None and self.prev[1] is not None and min(len(kp),len(self.prev[0]))>=30:
            candidates=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(self.prev[1],des,k=2)
            good=[m for pair in candidates if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance]
            if len(good)>=20:
                first=np.array([self.prev[0][m.queryIdx].pt for m in good],float)
                second=np.array([kp[m.trainIdx].pt for m in good],float)
                H,inliers=cv2.findHomography(first,second,method=cv2.RANSAC,ransacReprojThreshold=2.)
                if H is not None and inliers is not None:
                    use=inliers.ravel().astype(bool);count=int(use.sum())
                    if count:
                        transformed=cv2.perspectiveTransform(first[None].astype(np.float32),H)[0]
                        residual=float(np.median(np.linalg.norm(transformed[use]-second[use],axis=1)))
                        spread=np.ptp(first[use],axis=0);coverage=float(np.prod(spread)/(640*nh))
                        valid=count>=20 and count/len(good)>=.5 and coverage>.08 and residual<1.5
                        if valid:
                            scaling=np.diag([scale,scale,1.]);matrix=np.linalg.inv(scaling)@H@scaling
                            if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-9:valid=False
        reset=valid!=self.valid
        if valid and not self.valid:
            self.cumulative=np.eye(3);self.epoch+=1
        elif valid:
            self.cumulative=matrix@self.cumulative
        transform=np.linalg.inv(self.cumulative) if valid else np.eye(3)
        self.valid=valid;self.prev=(kp,des)
        return transform,{'valid':bool(valid),'status':'ORB_background_plane' if valid else 'ORB_unresolved',
                          'inliers':count,'median_inlier_reprojection_px_at640':residual,
                          'feature_bounding_area_fraction':coverage,'mode':'auto','evidence_reset':reset,
                          'reference_epoch':self.epoch,'metric_motion_available':False,
                          'rotation_translation_disambiguated':False,'validity_scope':'background homography hypothesis only'}


def run(args):
    audit=json.loads((ROOT/'reports/stage2_dataset_audit.json').read_text());report={}
    out=ROOT/'artifacts/stage2/camera_orb';out.mkdir(parents=True,exist_ok=True)
    for clip in args.clips.split(','):
        source=ROOT/f'artifacts/stage2/tracks/A/custom/s1/{clip}.jsonl'
        detfile=ROOT/f'artifacts/stage2/detections/A/{clip}/detections.jsonl'
        detections={r['frame']:r['detections'] for r in map(json.loads,detfile.read_text().splitlines())}
        model=ORBPlane();counts=Counter();residuals=[];start=time.perf_counter()
        with source.open() as trace,av.open(audit['clips'][clip]['video_path']) as video,(out/f'{clip}.jsonl').open('w') as output:
            stream=video.streams.video[0];stream.thread_type='AUTO'
            for index,(text,frame) in enumerate(zip(trace,video.decode(stream),strict=True)):
                row=json.loads(text)
                if row['frame']!=index:raise ValueError('Source/camera frame mismatch')
                boxes=[b for b in detections[index] if int(b[5])==0]
                transform,camera=model.update(frame.to_ndarray(format='bgr24'),boxes)
                counts[camera['status']]+=1
                if camera['median_inlier_reprojection_px_at640'] is not None:residuals.append(camera['median_inlier_reprojection_px_at640'])
                row['camera']=camera;row['stabilization_transform']=transform.tolist()
                row['camera_model']='independent ORB/RANSAC plane';output.write(json.dumps(row)+'\n')
        report[clip]={'frames':sum(counts.values()),'camera_status':counts,'source_track_sha256':file_sha256(source),
                       'detector_mask_sha256':file_sha256(detfile),'output_trace_sha256':file_sha256(out/f'{clip}.jsonl'),
                       'median_background_inlier_residual_px_at640':float(np.median(residuals)) if residuals else None,
                       'elapsed_replay_s':time.perf_counter()-start,
                       'GT_used_by_motion_estimator':False,'target_true_bearing_error':'unmeasured without calibration/attitude truth'}
        print('Complete ORB camera '+clip,flush=True)
    (ROOT/'reports/stage2_camera_ablation.json').write_text(json.dumps({'algorithm':'ORB1000/BF-Hamming ratio .75/RANSAC2px; >=20inliers, >=.5fraction, >.08coverage; 640pxwide',
        'source':'independent implementation using OpenCV primitives; not Purdue author-code reproduction',
        'scope':'plane consistency diagnostic only; no rotation/translation separation','clips':report},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--clips',default='Clip_032,Clip_043');run(p.parse_args())
