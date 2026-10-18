from collections import defaultdict
import json
from pathlib import Path
import cv2
from PIL import Image,ImageDraw,ImageFont
from scripts.evaluate import match
root=Path('artifacts/cadence_fullrate/M6');gt=defaultdict(list)
for g in json.loads(Path('data/qut_full/M6.gt.json').read_text()):gt[g['frame']].append(g)
rows=list(map(json.loads,(root/'decisions.jsonl').read_text(encoding='utf-8').splitlines()));hits=[]
for r in rows:
    objects=[o for o in r['objects'] if r['time']-o['last_seen']<=.25];pairs=match(objects,gt[r['frame']],20)
    for j,i in pairs.items():
        if gt[r['frame']][j]['kind']=='other' and objects[i]['assessment']['state']=='threat':hits.append((r,objects[i],gt[r['frame']][j]))
print('hits',len(hits),'times',[(r['time'],o['id']) for r,o,g in hits])
cap=cv2.VideoCapture('data/qut_full/M6.mp4');sheet=Image.new('RGB',(1800,1000),(15,23,35));draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',16)
for index,k in enumerate([0,len(hits)//3,2*len(hits)//3,len(hits)-1]):
    r,o,g=hits[k];cap.set(cv2.CAP_PROP_POS_FRAMES,r['frame']);ok,frame=cap.read();assert ok
    image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(image);d.rectangle(o['box'],outline='red',width=3);d.ellipse((g['u']-5,g['v']-5,g['u']+5,g['v']+5),outline='cyan',width=2)
    b=o['box'];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2;image=image.crop((cx-140,cy-105,cx+140,cy+105)).resize((840,630));x=(index%2)*900;y=(index//2)*500
    sheet.paste(image.resize((600,450)),(x,y));e=o['assessment'];text=f"t={r['time']:.2f} id={o['id']} score={o['confidence']:.3f}\ng={e['growth']:.3f} se={e['growth_se']:.3f}\ndrift={e['drift']:.2f} miss_hi={e['miss_high']:.2f}\nresid={e['scale_residual']:.3f}\nspan={e['span']:.2f}"
    draw.multiline_text((x+605,y+20),text,font=font,fill='white')
cap.release();sheet.save('reports/M6_fullrate_other_warning.png')
Path('reports/M6_fullrate_other_warning.json').write_text(json.dumps([{'frame':r['frame'],'time':r['time'],'object':o,'offline_label':g} for r,o,g in hits],indent=2))
