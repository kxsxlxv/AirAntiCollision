"""Oracle-selected model seed, then RGB-only LoRAT SOT isolation experiment.

NOT end-to-end detection or main-pipeline performance. GT used once to select the
intended detector seed, subsequently only for evaluation. External code is read-only.
"""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
import cv2
import numpy as np
from airavoid.geometry import ThreatEstimator
from airavoid.policy import TemporalPolicy
from airavoid.render import Renderer
from airavoid.geometry import advisory

reference=Path('C:/Data/University/Specialist/Diploma/FastVision')
sys.path.insert(0,str(reference))
from detection_track.lorat_trt import LoRATTensorRT

root=Path('artifacts/sot_trial');root.mkdir(parents=True,exist_ok=True)
report=[]
for seq in ['M1','M2']:
    byframe=defaultdict(list)
    for l in json.loads(Path(f'data/qut/{seq}.gt.json').read_text()):byframe[l['frame']].append(l)
    detections=[json.loads(x) for x in Path(f'demo/{seq}_tiled_initial/detections.jsonl').read_text().splitlines()]
    for kind in ['approach','other']:
        seed=None
        for rec in detections:
            for label in byframe[rec['frame']]:
                if label['kind']!=kind:continue
                choices=[]
                for b in rec['detections']:
                    u=(b[0]+b[2])/2;v=(b[1]+b[3])/2
                    error=np.hypot(u-label['u'],v-label['v'])
                    if error<10 and min(b[2]-b[0],b[3]-b[1])>=2:
                        choices.append((error,b))
                if choices:
                    err,b=min(choices,key=lambda x:x[0]);seed=(rec['frame'],label['id'],b,err);break
            if seed:break
        if not seed:
            report.append({'sequence':seq,'kind':kind,'status':'no reliable seed'});continue
        initial,ident,box,seed_error=seed
        tracker=LoRATTensorRT('artifacts/lorat_local.engine')
        stream=cv2.cuda_Stream();gpu=cv2.cuda_GpuMat();cap=cv2.VideoCapture(f'data/qut/{seq}.mp4')
        cap.set(cv2.CAP_PROP_POS_FRAMES,initial);ok,frame=cap.read()
        if not ok:raise RuntimeError('Cannot read seed frame')
        gpu.upload(frame,stream);stream.waitForCompletion();tracker.initialize(gpu,box[:4],stream)
        # Actual visual seed crop for QA, explicitly marked as offline oracle seed.
        cx=int((box[0]+box[2])/2);cy=int((box[1]+box[3])/2)
        crop=frame[max(0,cy-35):min(frame.shape[0],cy+35),max(0,cx-45):min(frame.shape[1],cx+45)]
        cv2.imwrite(str(root/f'{seq}_{kind}_seed.png'),cv2.resize(crop,(540,420)))
        history=[];estimator=ThreatEstimator();policy=TemporalPolicy();states=Counter();errors=[];latency=[];records=[]
        score_accepted=0;labels_visible=0
        for idx in range(initial,min(875,initial+500),2):
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read()
            if not ok:break
            tick=time.perf_counter();gpu.upload(frame,stream);stream.waitForCompletion()
            predicted,confidence=tracker.track(gpu,stream)
            latency.append((time.perf_counter()-tick)*1000)
            x1,y1,x2,y2=predicted;u=(x1+x2)/2;v=(y1+y2)/2;t=idx/25
            if confidence>=.5:
                history.append((t,u,v,max(x2-x1,y2-y1)));score_accepted+=1
                e=estimator.assess(history,t)
            else:
                history.clear();e={'state':'uncertain','reason':'sot_confidence'}
            e=policy.apply(1,t,e);states[e['state']]+=1
            gt=next((l for l in byframe[idx] if l['id']==ident),None)
            error=None
            if gt:
                labels_visible+=1;error=float(np.hypot(u-gt['u'],v-gt['v']));errors.append(error)
            records.append({'frame':idx,'time':t,'box':predicted,'sot_score':confidence,'offline_centroid_error_px':error,'assessment':e})
        cap.release()
        (root/f'{seq}_{kind}.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
        row={'sequence':seq,'kind':kind,'seed_frame':initial,'seed_gt_centroid_error_px':seed_error,'evaluated_frames':len(records),'gt_visible_frames':labels_visible,'score_accepted_frames':score_accepted,'centroid_within20px':sum(x<=20 for x in errors),'median_centroid_error_px':float(np.median(errors)) if errors else None,'p95_centroid_error_px':float(np.percentile(errors,95)) if errors else None,'states':states,'mean_upload_and_sot_ms':float(np.mean(latency)),'caveat':'oracle-selected detector seed; no online GT after initialization; SOT score threshold .5 uncalibrated; simultaneous GPU workload possible'}
        report.append(row);print(row,flush=True)
        Path('reports/sot_trial.json').write_text(json.dumps({'external_code_hashes':{name:hashlib.sha256((reference/'detection_track'/name).read_bytes()).hexdigest() for name in ['lorat_trt.py','trt_gpu.py']},'results':report},indent=2))
