"""DINO-sky-conditioned pixel extent refinement of actual cached YOLO seeds.

Experimental measurement only: it is not a new neural detector or verified shape.
No ground truth is read. The cached DINO mask conditions background eligibility.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import cv2
import numpy as np
from airavoid.provenance import file_sha256,cache_identity,validate_cache_entry


def refine(frame,sky,b):
    h,w=frame.shape[:2];cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
    span=max(b[2]-b[0],b[3]-b[1]);reason={'accepted':False,'reason':'tiny'}
    if span<6:return b,reason
    radius=int(min(384,max(48,span*2)))
    left,top=max(0,int(cx)-radius),max(0,int(cy)-radius)
    right,bottom=min(w,int(cx)+radius+1),min(h,int(cy)+radius+1)
    roi=frame[top:bottom,left:right];mask=sky[top:bottom,left:right]
    rim=np.ones(mask.shape,bool)
    if min(mask.shape)>24:rim[12:-12,12:-12]=False
    fraction=float(np.mean(mask[rim]))
    reason.update(sky_rim_fraction=fraction,reason='background_not_sky')
    if fraction<.85:return b,reason
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY);background=cv2.medianBlur(gray,31)
    response=cv2.absdiff(gray,background).astype(float)
    median=float(np.median(response));mad=float(np.median(np.abs(response-median)))
    threshold=max(8.,median+5*1.4826*mad);active=(response>threshold).astype(np.uint8)
    connected=cv2.dilate(active,np.ones((3,3),np.uint8))
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(connected)
    candidates=[]
    for label in range(1,n):
        yy,xx=np.where((labels==label)&(active>0))
        if len(xx)<3:continue
        x1,x2,y1,y2=int(xx.min()),int(xx.max())+1,int(yy.min()),int(yy.max())+1
        if min(x1,y1,roi.shape[1]-x2,roi.shape[0]-y2)<3:continue
        new_span=max(x2-x1,y2-y1)
        if not .5*span<=new_span<=4*span or len(xx)>.2*active.size:continue
        dx=max(x1+left-cx,0,cx-(x2+left));dy=max(y1+top-cy,0,cy-(y2+top))
        distance=float(np.hypot(dx,dy))
        if distance>max(8,span*.4):continue
        candidates.append((distance,-len(xx),[left+x1,top+y1,left+x2,top+y2],len(xx)))
    reason.update(reason='no_local_component',threshold=threshold)
    if not candidates:return b,reason
    _,_,box,area=min(candidates,key=lambda item:(item[0],item[1]))
    reason.update(accepted=True,reason='pixel_component',area_px=area,span_ratio=max(box[2]-box[0],box[3]-box[1])/span,
                  center_shift_px=float(np.hypot((box[0]+box[2])/2-cx,(box[1]+box[3])/2-cy)))
    return box+list(b[4:6])+['pixel'],reason


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',choices=['M1','S2'],default='M1');p.add_argument('--base',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    seq=a.sequence;video=Path(f'data/qut/{seq}.mp4');mask_path=Path(f'artifacts/sky/{seq}_sky_mask.mkv')
    cache={r['frame']:r for r in map(json.loads,Path(a.base).read_text().splitlines())}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video));masks=cv2.VideoCapture(str(mask_path));fps=cap.get(cv2.CAP_PROP_FPS)
    if cap.get(cv2.CAP_PROP_FRAME_COUNT)!=masks.get(cv2.CAP_PROP_FRAME_COUNT) or abs(fps-masks.get(cv2.CAP_PROP_FPS))>.01:
        raise ValueError('Mask source frame count/FPS mismatch')
    digest=file_sha256(video);identity=cache_identity(a.base,video,digest,1280,720,fps)
    index=0;counts=Counter();timings=[];samples=0
    start=time.perf_counter()
    with (out/'detections.jsonl').open('w') as f:
        while True:
            ok,frame=cap.read();mok,mask=masks.read()
            if not ok:break
            if not mok:raise ValueError('Sky mask ended early')
            row=cache.get(index);index+=1
            if row is None:continue
            validate_cache_entry(row,index-1,(index-1)/fps)
            sky=cv2.resize(mask[:,:,0],(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_NEAREST)>127
            before=time.perf_counter();observations=[];metadata=[]
            for b in row['detections']:
                new,detail=refine(frame,sky,b);observations.append(new);metadata.append(detail);counts[detail['reason']]+=1
            timings.append(time.perf_counter()-before)
            if observations:
                ids=cv2.dnn.NMSBoxes([[b[0],b[1],b[2]-b[0],b[3]-b[1]] for b in observations],[b[4] for b in observations],.0,.45)
                observations=[observations[int(i)] for i in np.asarray(ids).ravel()]
            f.write(json.dumps({**row,'observations':observations,'pixel_refinement':metadata})+'\n');samples+=1
    cap.release();masks.release()
    report={'kind':'Experimental pixel-extent refinement with cached YOLO and cached neural DINO sky, not neural throughput',
            'args':vars(a),'input_sha256':digest,'mask_sha256':file_sha256(mask_path),'cache_identity':identity,
            'source_fps':fps,'resolution':[1280,720],'frames':samples,'counts':counts,'total_s':time.perf_counter()-start,
            'mean_refinement_ms':float(np.mean(timings)*1000),'no_gt_used':True,
            'limitations':'Components can be clouds/partial shapes; original detector score is inherited, not refined-box confidence. Constant physical span remains unverified.'}
    (out/'benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
