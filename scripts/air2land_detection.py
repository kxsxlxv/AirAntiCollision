"""Detection-only validation on existing rendered AIR2LAND RGB, no simulator or threat GT."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from airavoid.detector import Detector,NAMES
from airavoid.provenance import file_sha256


def iou(a,b):
    area=lambda r:max(0,r[2]-r[0])*max(0,r[3]-r[1])
    intersection=area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])
    return intersection/max(1e-9,area(a)+area(b)-intersection)


def main():
    manifest=json.loads(Path('data/air2land_detection.json').read_text());results=[]
    for tiled in [False,True]:
        detector=Detector('artifacts/yolo26m_local.engine',.03,tiled=tiled,gpu_preprocess=True,tile_edge_margin=2.)
        first=cv2.imread(manifest['rows'][0]['image'])
        for _ in range(5):detector(first)
        counts=defaultdict(Counter);samples=[];started=time.perf_counter()
        for row in manifest['rows']:
            frame=cv2.imread(row['image'])
            if frame is None:raise RuntimeError('Cannot decode '+row['image'])
            if file_sha256(row['image'])!=row['image_sha256']:raise ValueError('Source image changed after metadata export')
            detections=detector(frame);x,y,w,h=row['box_xywh'];gt=[x,y,x+w,y+h]
            best=max([(iou(b[:4],gt),b) for b in detections],key=lambda pair:pair[0],default=(0.,None))
            c=counts[row['variant']];c['images']+=1;c['predicted_boxes']+=len(detections)
            for threshold in [.1,.25,.5]:c['iou_ge_'+str(threshold)]+=int(best[0]>=threshold)
            c['class0_and_iou_ge_0.5']+=int(any(iou(b[:4],gt)>=.5 and int(b[5])==0 for b in detections))
            samples.append({'variant':row['variant'],'image':row['image'],'gt_box':gt,'best_iou':best[0],
                            'detections':detections,'best_class':NAMES[int(best[1][5])] if best[1] is not None else None})
        elapsed=time.perf_counter()-started
        result={'mode':'native_crops' if tiled else 'full_frame','threshold':.03,'counts':counts,'images':len(samples),
                'decode_hash_and_neural_s':elapsed,'samples':samples};results.append(result)
        Path('reports/air2land_detection.json').write_text(json.dumps({'kind':'Existing synthetic RGB detection diagnostic, not a threat/encounter score',
            'manifest_sha256':file_sha256('data/air2land_detection.json'),'results':results,
            'limits':['No verified source timing/intrinsics/pose units; no threat or metric CPA scores',
                      'Related landing scenario variants; independence not established','Detector training provenance unknown',
                      '317 actual paired images, all selected paired frames; missing-color files explicitly listed',
                      'Unmatched model boxes are not automatically verified false positives','No render FPS or simulator claim']},indent=2))
        print(result['mode'],dict(counts),flush=True)
    # Visual box comparison, cyan source annotation vs orange actual neural boxes.
    sheet=Image.new('RGB',(1600,2000),(15,23,35));font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',18);d=ImageDraw.Draw(sheet)
    for index,variant in enumerate(['000','012','024','035']):
        rows=[s for s in results[-1]['samples'] if s['variant']==variant]
        for col,sample in enumerate([rows[0],min(rows,key=lambda r:r['best_iou'])]):
            frame=cv2.imread(sample['image']);image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(image)
            draw.rectangle(sample['gt_box'],outline='cyan',width=3)
            for box in sample['detections']:draw.rectangle(box[:4],outline='orange',width=2)
            image.thumbnail((800,450));sheet.paste(image,(col*800,index*500+40))
            d.text((col*800+10,index*500+3),f"Вариант {variant} | IoU={sample['best_iou']:.3f} | {'первый' if col==0 else 'минимальный IoU'}",font=font,fill='white')
    sheet.save('reports/air2land_detection_preview.png')


if __name__=='__main__':main()
