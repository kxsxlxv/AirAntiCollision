"""Actual neural per-crop detections for inspecting partial-target duplicates."""
import argparse
import json
from pathlib import Path
import cv2
from airavoid.detector import Detector,native_tiles

p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--frame',type=int,default=210)
p.add_argument('--output',default='reports/tile_edge_audit.json');a=p.parse_args()
cap=cv2.VideoCapture(a.video);cap.set(cv2.CAP_PROP_POS_FRAMES,a.frame);ok,image=cap.read();cap.release()
if not ok:raise ValueError('Frame unavailable')
det=Detector('artifacts/yolo26m_local.engine',gpu_preprocess=True)
det.gpu_preprocessor.upload(image);h,w=image.shape[:2];rows=[]
for x,y in native_tiles(w,h):
    cw,ch=min(704,w-x),min(416,h-y)
    boxes=det._infer_gpu(det.gpu_preprocessor.crop(x,y,cw,ch))
    for box in boxes:
        b=box.tolist();edges=[]
        if x>0:edges.append(b[0])
        if y>0:edges.append(b[1])
        if x+cw<w:edges.append(cw-b[2])
        if y+ch<h:edges.append(ch-b[3])
        b[0]+=x;b[2]+=x;b[1]+=y;b[3]+=y
        rows.append({'tile':[x,y,cw,ch],'box':b,'nearest_internal_edge_px':min(edges) if edges else None})
rows.append({'tile':'full','box':det._infer_gpu(det.gpu_preprocessor.frame).tolist()})
Path(a.output).write_text(json.dumps({'video':a.video,'frame':a.frame,'rows':rows},indent=2))
print(json.dumps(rows,indent=2))
