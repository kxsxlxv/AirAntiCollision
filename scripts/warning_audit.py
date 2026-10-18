"""Offline unmatched-warning review sheet; unmatched is not synonymous with false."""
from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts.evaluate import match

root=Path('artifacts/full_campaign');episodes=[]
for path in sorted(root.glob('*/decisions.jsonl')):
    seq=path.parent.name;gts=defaultdict(list)
    for g in json.loads(Path(f'data/qut_full/{seq}.gt.json').read_text()):gts[g['frame']].append(g)
    rows=list(map(json.loads,path.read_text(encoding='utf-8').splitlines()));dt=float(np.median(np.diff([r['time'] for r in rows])))
    current={}
    for r in rows:
        objects=[o for o in r['objects'] if r['time']-o['last_seen']<=.25]
        labels=gts[r['frame']];used=set(match(objects,labels,20).values())
        for i,o in enumerate(objects):
            if i in used or o['assessment']['state']!='threat':continue
            b=o['box'];center=np.array([(b[0]+b[2])/2,(b[1]+b[3])/2])
            distances=[float(np.linalg.norm(center-[g['u'],g['v']])) for g in labels]
            nearest=labels[int(np.argmin(distances))] if distances else None
            item={'frame':r['frame'],'time':r['time'],'box':b,'confidence':o['confidence'],'reason':o['assessment']['reason'],
                  'nearest_label_distance_px':min(distances) if distances else None,'nearest_offline_label':nearest}
            key=o['id'];e=current.get(key)
            if e is None or r['time']-e['samples'][-1]['time']>dt*1.5:
                e={'sequence':seq,'track_id':key,'samples':[],'sample_period_s':dt};current[key]=e;episodes.append(e)
            e['samples'].append(item)
for e in episodes:e['duration_s']=len(e['samples'])*e['sample_period_s']
selected=[];per_sequence=defaultdict(int)
for e in sorted(episodes,key=lambda x:-x['duration_s']):
    if per_sequence[e['sequence']]>=3:continue
    selected.append(e);per_sequence[e['sequence']]+=1
    if len(selected)==12:break
font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',15)
sheet=Image.new('RGB',(1440,70+300*max(1,int(np.ceil(len(selected)/3)))),(15,23,35));draw=ImageDraw.Draw(sheet)
draw.text((15,10),'ОФЛАЙН-АУДИТ: предупреждения без совпадения с приближённой GT-точкой (20 пикс)',font=font,fill='white')
draw.text((15,36),'Это не доказанные ложные тревоги: разметка неполная. Красный — модель; голубой — ближайшая GT-точка.',font=font,fill=(180,200,220))
for n,e in enumerate(selected):
    item=e['samples'][len(e['samples'])//2];seq=e['sequence']
    cap=cv2.VideoCapture(f'data/qut_full/{seq}.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,item['frame']);ok,frame=cap.read();cap.release()
    if not ok:raise ValueError('Audit source frame unavailable')
    x,y=(n%3)*480,(n//3)*300+70;b=item['box'];cx,cy=int((b[0]+b[2])/2),int((b[1]+b[3])/2)
    radius=int(max(40,max(b[2]-b[0],b[3]-b[1])*1.25))
    crop=frame[max(0,cy-radius):min(frame.shape[0],cy+radius),max(0,cx-radius):min(frame.shape[1],cx+radius)]
    view=cv2.resize(frame,(320,180));ratio=320/frame.shape[1]
    cv2.rectangle(view,tuple(round(v*ratio) for v in b[:2]),tuple(round(v*ratio) for v in b[2:4]),(80,80,255),1)
    if item['nearest_offline_label']:
        gt=item['nearest_offline_label'];cv2.circle(view,(round(gt['u']*ratio),round(gt['v']*ratio)),3,(255,255,0),1)
    sheet.paste(Image.fromarray(cv2.cvtColor(view,cv2.COLOR_BGR2RGB)),(x,y+35))
    if crop.size:sheet.paste(Image.fromarray(cv2.cvtColor(cv2.resize(crop,(150,150)),cv2.COLOR_BGR2RGB)),(x+325,y+35))
    draw.text((x+5,y+5),f"{seq} #{e['track_id']} • {e['duration_s']:.1f}с • кадр {item['frame']} • det {item['confidence']:.2f}",font=font,fill='white')
    distance=item['nearest_label_distance_px'];detail='нет GT в кадре' if distance is None else f'до ближайшей GT: {distance:.1f} пикс'
    draw.text((x+5,y+225),detail,font=font,fill=(190,205,220))
    draw.text((x+5,y+250),'Справа: увеличенный исходный RGB-фрагмент',font=font,fill=(190,205,220))
sheet.save('reports/unmatched_warning_review.png')
report={'kind':'Qualitative offline review, not verified false-alert ground truth','root':str(root),'centroid_tolerance_px':20,
        'selection':'Longest track-warning episodes, at most3 per sequence, at most12 total','episodes':episodes,
        'selected':[{k:v for k,v in e.items() if k!='samples'} for e in selected],
        'limits':'IDs are track IDs, not guaranteed physical encounter identities. Labels approximate/interpolated/incomplete. Images do not establish CPA.'}
Path('reports/unmatched_warning_audit.json').write_text(json.dumps(report,indent=2))
print('Episodes:',len(episodes),'warning samples:',sum(len(e['samples']) for e in episodes),'selected:',len(selected))
