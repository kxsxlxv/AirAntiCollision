"""Small fixed conversion check: supplied FP32 ONNX CPU vs local FP16-derived TRT."""
import json
from pathlib import Path
import time
import cv2
import numpy as np
from airavoid.detector import Detector

onnx=Detector('../anti_collision_system/best.onnx',backend='cpu')
trt=Detector('artifacts/yolo26m_local.engine')
rows=[]
for seq in ['M1','M2']:
    cap=cv2.VideoCapture(f'data/qut/{seq}.mp4')
    for frame in [0,100,200,300,400,500,600,700,800]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
        if not ok:continue
        t=time.perf_counter();a=onnx(image);cpu_s=time.perf_counter()-t
        t=time.perf_counter();b=trt(image);trt_s=time.perf_counter()-t
        used=set();matches=[]
        for aa in a:
            ca=np.array([(aa[0]+aa[2])/2,(aa[1]+aa[3])/2])
            options=[]
            for i,bb in enumerate(b):
                cb=np.array([(bb[0]+bb[2])/2,(bb[1]+bb[3])/2])
                if i not in used and int(aa[5])==int(bb[5]):options.append((float(np.linalg.norm(ca-cb)),i))
            if options and min(options)[0]<10:
                dist,i=min(options);used.add(i)
                matches.append({'center_error_px':dist,'max_coordinate_error_px':float(np.max(np.abs(np.array(aa[:4])-b[i][:4]))),'confidence_error':abs(aa[4]-b[i][4])})
        rows.append({'sequence':seq,'frame':frame,'onnx_count':len(a),'trt_count':len(b),'matches':matches,'cpu_s':cpu_s,'trt_s':trt_s})
    cap.release()
matches=[m for r in rows for m in r['matches']]
report={'frames':len(rows),'fp32_onnx_detections':sum(r['onnx_count'] for r in rows),'trt_detections':sum(r['trt_count'] for r in rows),'matched':len(matches),
        'median_center_error_px':float(np.median([m['center_error_px'] for m in matches])) if matches else None,
        'max_coordinate_error_px':max([m['max_coordinate_error_px'] for m in matches],default=None),
        'max_confidence_error':max([m['confidence_error'] for m in matches],default=None),
        'precision':'TRT built from supplied FP16 ONNX; engine I/O FP32; ONNX CPU from supplied FP32 model',
        'caveat':'small development conversion spot check, not full accuracy equivalence or fair latency benchmark; other GPU workload may run',
        'frames_detail':rows}
Path('reports/backend_comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='frames_detail'},indent=2))
