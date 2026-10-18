"""Render recorded online outputs, never use playback speed as inference FPS."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,av
from airavoid.media import TimestampWriter
METHODS=['tracktrack','botsort','ocsort','diffmot','putr','trackssm']
NAMES={'tracktrack':'TrackTrack + SBS-S50','botsort':'BoT-SORT + SBS-S50','ocsort':'OC-SORT',
       'diffmot':'DiffMOT + SBS-S50','putr':'PuTR','trackssm':'TrackSSM: repaired / reference'}
out=ROOT/'demo/mot_comparison';out.mkdir(parents=True,exist_ok=True)
qa=[]
for seq in ['sw02','sw06','sw28','M2','formation']:
    phase='swarm' if seq.startswith('sw') else 'campaign' if seq=='M2' else 'render_live'
    traces={m:[json.loads(x) for x in (ROOT/f'artifacts/mot/{phase}/{m}/{seq}.jsonl').read_text().splitlines()] for m in METHODS}
    timeline=traces[METHODS[0]]
    if any([r['frame'] for r in rows]!=[r['frame'] for r in timeline] for rows in traces.values()):raise ValueError('Mismatched timelines')
    source=ROOT/f'data/mot_swarm/UAVSwarm-{seq[2:]}/img1' if seq.startswith('sw') else None
    if source:
        files=sorted(source.glob('*.jpg'));fps=30;cap=None
        gt=np.loadtxt(source.parent/'gt/gt.txt',delimiter=',',ndmin=2)
    else:
        cap=cv2.VideoCapture(str(ROOT/('data/qut_full/M2.mp4' if seq=='M2' else 'data/formation_native.mp4')))
        cap.set(cv2.CAP_PROP_POS_FRAMES,timeline[0]['frame']);position=timeline[0]['frame']
        fps=1/np.median(np.diff([r['time'] for r in timeline]))
    writer=TimestampWriter(out/f'{seq}.mp4',fps,(1920,808),encoder='libx264')
    for index,row in enumerate(timeline):
        if source:img=cv2.imread(str(files[row['frame']]))
        else:
            while position<=row['frame']:
                ok,img=cap.read();position+=1
                if not ok:raise ValueError('Unexpected source end')
        h,w=img.shape[:2];canvas=np.full((808,1920,3),18,np.uint8)
        cv2.putText(canvas,f'{seq} | same YOLO26m | recorded ONLINE outputs | playback is not inference speed',
                    (20,30),cv2.FONT_HERSHEY_SIMPLEX,.75,(230,230,230),1,cv2.LINE_AA)
        cv2.putText(canvas,f'source t={row["time"]:.2f}s | IDs are local to each tracker | no future linking',
                    (20,59),cv2.FONT_HERSHEY_SIMPLEX,.6,(180,180,180),1,cv2.LINE_AA)
        for k,m in enumerate(METHODS):
            x=(k%3)*640;y=76+(k//3)*366
            scale=min(640/w,330/h);nw=round(w*scale);nh=round(h*scale)
            ox=x+(640-nw)//2;oy=y+32+(330-nh)//2
            canvas[oy:oy+nh,ox:ox+nw]=cv2.resize(img,(nw,nh))
            cv2.putText(canvas,NAMES[m],(x+10,y+23),cv2.FONT_HERSHEY_SIMPLEX,.62,(0,220,255),1,cv2.LINE_AA)
            if source:
                for g in gt[gt[:,0]==row['frame']+1]:
                    gx,gy,gw,gh=g[2:6]
                    cv2.rectangle(canvas,(round(ox+gx*scale),round(oy+gy*scale)),(round(ox+(gx+gw)*scale),round(oy+(gy+gh)*scale)),(230,230,230),1)
            for tr in traces[m][index]['tracks']:
                tid=int(tr[0]);b=np.asarray(tr[1:5]);b[[0,2]]=b[[0,2]]*scale+ox;b[[1,3]]=b[[1,3]]*scale+oy
                x1,y1,x2,y2=map(round,b);color=(50+tid*37%205,50+tid*71%205,50+tid*113%205)
                cv2.rectangle(canvas,(max(x,x1),max(y+32,y1)),(min(x+639,x2),min(y+363,y2)),color,1)
                cv2.putText(canvas,str(tid),(max(x+1,x1),max(y+47,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
        writer.write(canvas,row['time']-timeline[0]['time'])
        if index==min(len(timeline)//2,200):cv2.imwrite(str(out/f'{seq}.jpg'),canvas)
    duration=timeline[-1]['time']-timeline[0]['time']+1/fps;writer.close(duration)
    if cap:cap.release()
    with av.open(str(out/f'{seq}.mp4')) as container:
        stream=container.streams.video[0];times=[float(f.pts*f.time_base) for f in container.decode(stream)]
        assert len(times)==len(timeline)
        assert all(b>a for a,b in zip(times,times[1:]))
        assert max(abs(t-(r['time']-timeline[0]['time'])) for t,r in zip(times,timeline))<1/90000+1e-7
        qa.append(dict(sequence=seq,frames=len(times),codec=stream.codec_context.name,width=stream.width,height=stream.height,
            expected_duration=duration,container_duration=container.duration/av.time_base,max_pts_error=max(abs(t-(r['time']-timeline[0]['time'])) for t,r in zip(times,timeline))))
    print(seq,len(timeline),flush=True)
(ROOT/'reports/mot_video_qa.json').write_text(json.dumps(dict(complete=True,runs=qa),indent=2))
cards=''.join(f'<article><h2>{s}</h2><video controls preload="metadata" poster="{s}.jpg" src="{s}.mp4"></video></article>' for s in ['M2','formation','sw06','sw28','sw02'])
(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fixed YOLO26m MOT comparison</title><style>body{font:17px system-ui;background:#101827;color:#e2e8f0;margin:30px auto;max-width:1500px;padding:0 20px}video{width:100%}article{margin:35px 0}a{color:#7dd3fc}</style><h1>Online MOT with fixed YOLO26m</h1><p>Six algorithms, identical inputs. Recorded outputs; video playback speed is not inference throughput. TrackSSM uses a repaired interface and reference backend. Full metrics and actual RTX 4060 timings: <a href="../../MOT_RESULTS.md">MOT_RESULTS.md</a>.</p>'+cards,encoding='utf-8')
