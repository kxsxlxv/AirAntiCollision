"""Offline development diagnosis of source-view switches; GT only chooses QA crop."""
from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
from airavoid.detector import Detector

det=Detector('artifacts/yolo26m_local.engine');cap=cv2.VideoCapture('data/qut/M4.mp4')
gt={r['frame']:r for r in json.loads(Path('data/qut/M4.gt.json').read_text()) if r['kind']=='approach'}
rows=[];images=[]
for index in [325,350,375,388,400,425]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
    if not ok:continue
    l=gt[index];cx=l['u'];cy=l['v'];results=[]
    views=[('full',0,0,frame)]
    for y in [0,304]:
        for x in [0,288,576]:views.append((f'tile_{x}_{y}',x,y,frame[y:y+416,x:x+704]))
    qa=frame[max(0,int(cy)-50):min(720,int(cy)+50),max(0,int(cx)-75):min(1280,int(cx)+75)].copy()
    qa=cv2.resize(qa,(600,400),interpolation=cv2.INTER_NEAREST)
    for name,ox,oy,view in views:
        for b in det._infer(view):
            x1,y1,x2,y2=b[:4];u=(x1+x2)/2+ox;v=(y1+y2)/2+oy
            if np.hypot(u-cx,v-cy)>20:continue
            results.append({'view':name,'width':float(x2-x1),'height':float(y2-y1),'confidence':float(b[4]),'center_error_gt':float(np.hypot(u-cx,v-cy))})
    text=f'{index/25:.2f}s '+', '.join(f"{r['view']}:w{r['width']:.1f}/c{r['confidence']:.2f}" for r in results)
    print(text,flush=True)
    cv2.rectangle(qa,(0,0),(600,85),(15,20,30),-1)
    cv2.putText(qa,f'{index/25:.2f}s OFFLINE GT-CENTERED QA CROP',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    for i,r in enumerate(results[:3]):cv2.putText(qa,f"{r['view']} width={r['width']:.1f} conf={r['confidence']:.2f}",(10,45+i*18),cv2.FONT_HERSHEY_SIMPLEX,.45,(100,220,255),1,cv2.LINE_AA)
    rows.append({'time':index/25,'proposals':results});images.append(qa)
cv2.imwrite('reports/M4_span_diagnosis.png',np.vstack([np.hstack(images[:3]),np.hstack(images[3:])]))
Path('reports/M4_span_diagnosis.json').write_text(json.dumps(rows,indent=2))
