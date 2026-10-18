"""Paired native CUDA OpenCV vs CPU preprocessing, same TensorRT weights."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
from airavoid.detector import Detector

summary=[]
for tiled in [False,True]:
    cpu=Detector('artifacts/yolo26m_local.engine',tiled=tiled)
    gpu=Detector('artifacts/yolo26m_local.engine',tiled=tiled,gpu_preprocess=True)
    frames=[]
    for seq in ['M1','M2']:
        cap=cv2.VideoCapture(f'data/qut/{seq}.mp4')
        for idx in [0,100,200,300,400,500,600,700,800]:
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,im=cap.read()
            if ok:frames.append((seq,idx,im))
        cap.release()
    for det in [cpu,gpu]:
        for _ in range(5):det(frames[0][2])
        det.reset_temporal()
    rows=[]
    for j,(seq,idx,frame) in enumerate(frames):
        result={};latency={}
        for name,det in ([('cpu',cpu),('gpu',gpu)] if j%2==0 else [('gpu',gpu),('cpu',cpu)]):
            t=time.perf_counter();result[name]=det(frame);latency[name]=(time.perf_counter()-t)*1000
        used=set();matched=[]
        for aa in result['cpu']:
            ca=np.array([(aa[0]+aa[2])/2,(aa[1]+aa[3])/2]);opts=[]
            for i,bb in enumerate(result['gpu']):
                cb=np.array([(bb[0]+bb[2])/2,(bb[1]+bb[3])/2])
                if i not in used and int(aa[5])==int(bb[5]):opts.append((float(np.linalg.norm(ca-cb)),i))
            if opts and min(opts)[0]<10:
                dist,i=min(opts);used.add(i);bb=result['gpu'][i]
                matched.append({'center_error_px':dist,'max_coordinate_error_px':float(np.max(np.abs(np.array(aa[:4])-bb[:4]))),'confidence_error':abs(aa[4]-bb[4])})
        rows.append({'sequence':seq,'frame':idx,'cpu_detections':len(result['cpu']),'gpu_detections':len(result['gpu']),'matches':matched,'latency_ms':latency})
    matches=[m for r in rows for m in r['matches']]
    out={'tiled':tiled,'frames':len(rows),'cpu_detections':sum(r['cpu_detections'] for r in rows),'gpu_detections':sum(r['gpu_detections'] for r in rows),'matched':len(matches),'median_center_error_px':float(np.median([m['center_error_px'] for m in matches])) if matches else None,'max_confidence_error':max([m['confidence_error'] for m in matches],default=None),'cpu_preprocess_detector_mean_ms':float(np.mean([r['latency_ms']['cpu'] for r in rows])),'gpu_preprocess_detector_mean_ms':float(np.mean([r['latency_ms']['gpu'] for r in rows])),'details':rows}
    summary.append(out);print({k:v for k,v in out.items() if k!='details'},flush=True)
Path('reports/preprocessing_comparison.json').write_text(json.dumps({'kind':'paired detector-stage measurement, not full application FPS','caveat':'CPU/CUDA resize can round differently; tiny detections may change','results':summary},indent=2))
