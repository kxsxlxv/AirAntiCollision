"""DINO-sky-conditioned classical tiny-object proposals + actual YOLO RGB crops.

No GT read. Extra proposal path only; existing model detections are preserved as
inputs before suppression. Cached segmentation and baseline inference are NOT
included in the added-stage timing. Original pixel sizes are retained.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import cv2
import numpy as np
from airavoid.detector import Detector

p=argparse.ArgumentParser();p.add_argument('--sequence',default='M1');p.add_argument('--mask',default='artifacts/sky/M1_sky_mask.mkv');p.add_argument('--base',default='demo/M1_tiled_initial/detections.jsonl');p.add_argument('--output',default='artifacts/sky_proposals/M1');p.add_argument('--topk',type=int,default=12);p.add_argument('--crop-size',type=int,default=96);args=p.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
cap=cv2.VideoCapture(f'data/qut/{args.sequence}.mp4');maskcap=cv2.VideoCapture(args.mask);det=Detector('artifacts/yolo26m_local.engine')
rows=[json.loads(x) for x in Path(args.base).read_text().splitlines()];counts=Counter();elapsed=[];start=time.perf_counter()
with (out/'detections.jsonl').open('w') as f:
    for j,row in enumerate(rows):
        idx=row['frame'];base=row['detections'];boxes=[list(x[:6])+['detector'] for x in base];proposals=[];extra=[]
        if j%5==0:
            tick=time.perf_counter();cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,image=cap.read()
            maskcap.set(cv2.CAP_PROP_POS_FRAMES,idx);mok,mask=maskcap.read()
            if not ok or not mok:raise RuntimeError('Video/mask timeline mismatch')
            h,w=image.shape[:2];sky=cv2.resize(mask[:,:,0],(w,h),interpolation=cv2.INTER_NEAREST)>127
            # Neural sky mask, with a generous border margin; never a classical sky extractor.
            sky=cv2.dilate(sky.astype(np.uint8),np.ones((21,21),np.uint8)).astype(bool)
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
            response=np.maximum(cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,k),cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,k)).astype(float)
            response[~sky]=0
            for b in base:
                x1,y1,x2,y2=b[:4]
                response[max(0,int(y1)-16):min(h,int(y2)+17),max(0,int(x1)-16):min(w,int(x2)+17)]=0
            values=response[sky]
            median=float(np.median(values)) if len(values) else 0
            mad=float(np.median(np.abs(values-median))) if len(values) else 0
            threshold=max(12.,median+6*1.4826*mad)
            for _ in range(args.topk):
                y,x=np.unravel_index(np.argmax(response),response.shape);strength=float(response[y,x])
                if strength<threshold:break
                proposals.append([int(x),int(y),strength])
                r=args.crop_size//2
                response[max(0,y-r):min(h,y+r+1),max(0,x-r):min(w,x+r+1)]=0
                left=max(0,min(w-args.crop_size,x-r));top=max(0,min(h-args.crop_size,y-r))
                b=det._infer(image[top:top+args.crop_size,left:left+args.crop_size])
                for pred in b:
                    pred[[0,2]]+=left;pred[[1,3]]+=top
                    extra.append(pred.tolist()+['zoom'])
            counts['proposal_frames']+=1;counts['proposals']+=len(proposals);counts['zoom_detections_before_suppression']+=len(extra)
            elapsed.append(time.perf_counter()-tick)
            if j==150:
                qa=image.copy()
                for x,y,strength in proposals:cv2.circle(qa,(x,y),8,(0,200,255),1)
                cv2.imwrite(str(out/'proposal_preview.png'),qa)
        combined=boxes+extra
        if combined:
            xywh=[[b[0],b[1],b[2]-b[0],b[3]-b[1]] for b in combined]
            keep=cv2.dnn.NMSBoxes(xywh,[b[4] for b in combined],.18,.45)
            combined=[combined[int(i)] for i in np.asarray(keep).ravel()] if len(keep) else []
        f.write(json.dumps({'frame':idx,'time':row['time'],'detections':combined,'observations':combined,'sky_proposals':proposals,'added_zoom_detections':extra,'detector_ran':True})+'\n')
report={'kind':'extra actual neural RGB crop experiment; baseline and DINO masks precomputed','sequence':args.sequence,'frames':len(rows),'counts':counts,'added_proposal_frame_mean_s':float(np.mean(elapsed)) if elapsed else None,'total_added_experiment_s':time.perf_counter()-start,'settings':vars(args),'no_gt_used':True,'original_pixel_sizes_preserved':True}
(out/'proposal_benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
