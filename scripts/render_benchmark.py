"""Paired cached/uncached typography renderer test on actual model decisions."""
from collections import defaultdict,deque
import json
from pathlib import Path
import time
import cv2
import numpy as np
from airavoid.render import Renderer

rows=[json.loads(x) for x in Path('demo/M2_confirmed/decisions.jsonl').read_text(encoding='utf-8').splitlines()]
renderers={'uncached':Renderer(cache_text=False),'cached':Renderer(cache_text=True)}
timings=defaultdict(list);history=defaultdict(lambda:deque(maxlen=100));cap=cv2.VideoCapture('data/qut/M2.mp4')
for i,r in enumerate(rows[:200]):
    cap.set(cv2.CAP_PROP_POS_FRAMES,r['frame']);ok,frame=cap.read()
    if not ok:break
    for o in r['objects']:
        if o['last_seen']==r['time']:
            b=o['box'];history[o['id']].append((r['time'],(b[0]+b[2])/2,(b[1]+b[3])/2))
        o['raw_history']=list(history[o['id']])
    for name in (['cached','uncached'] if i%2 else ['uncached','cached']):
        t=time.perf_counter();image=renderers[name].render(frame,r['objects'],r['time'],r['camera'],r['advisory'])
        if i>=10:timings[name].append((time.perf_counter()-t)*1000)
        if i==194:cv2.imwrite(f'reports/renderer_{name}.png',image)
report={'kind':'paired renderer-stage measurement, same actual RGB and model trace','frames':len(timings['cached']),'results':{k:{'mean_ms':float(np.mean(v)),'p95_ms':float(np.percentile(v,95))} for k,v in timings.items()},'cache_limit_entries':512,'caveat':'not neural/application FPS; history reconstructed from trace for visualization only'}
Path('reports/render_benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
