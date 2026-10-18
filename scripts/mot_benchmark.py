"""Fixed-YOLO online MOT replay and actual end-to-end benchmarks.

Approximate QUT centre labels permit target-conditional identity continuity only,
not standard MOT HOTA/IDF1: other aircraft and distractors are incompletely labelled.
"""
import argparse,json,time,sys,hashlib,traceback,gc
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime,timezone
from mot_models import Adapter,ROOT,EXT
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
# Bind Torch's cuDNN before custom OpenCV prepends its newer cuDNN to PATH.
# This is process-local; no installed DLL/environment changes.
with torch.inference_mode():
    _conv=torch.nn.Conv2d(3,4,3).cuda()
    _conv(torch.zeros(1,3,16,16,device='cuda'));torch.cuda.synchronize()
del _conv
import cv2
from scipy.optimize import linear_sum_assignment


class ImageSequence:
    def __init__(self,path):
        import configparser
        self.files=sorted((path/'img1').glob('*.jpg'));self.i=0
        c=configparser.ConfigParser();c.read(path/'seqinfo.ini');self.fps=c.getfloat('Sequence','frameRate')
        self.h,self.w=cv2.imread(str(self.files[0])).shape[:2]
    def get(self,k):return {3:self.w,4:self.h,5:self.fps,7:len(self.files)}[k]
    def set(self,k,v):self.i=int(v)
    def read(self):
        if self.i>=len(self.files):return False,None
        image=cv2.imread(str(self.files[self.i]));self.i+=1;return image is not None,image
    def release(self):pass


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_frame(writer,img,tracks,method,t,relative_t):
    annotated=img.copy()
    for tr in tracks:
        tid=int(tr[0]);x1,y1,x2,y2=map(round,tr[1:5]);color=(60+tid*37%195,60+tid*71%195,60+tid*113%195)
        cv2.rectangle(annotated,(x1,y1),(x2,y2),color,1)
        cv2.putText(annotated,str(tid),(max(0,x1),max(30,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
    label=method+(' [repaired/reference]' if method=='trackssm' else '')
    cv2.putText(annotated,f'{label} | fixed YOLO26m | t={t:.2f}',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2,cv2.LINE_AA)
    writer.write(annotated,relative_t)


def identity_metrics(rows,gt,gate=20):
    byframe=defaultdict(list)
    for g in gt:byframe[g['frame']].append(g)
    stats=defaultdict(lambda:dict(total=0,matched=0,switches_consecutive=0,switches_including_reacquisition=0,
                                 fragments=0,ids=Counter(),last_id=None,last_frame=None,last_annotated=None,was_matched=False))
    distances=[];cross_identity=defaultdict(set)
    for row in rows:
        f=row['frame']; gs=byframe[f];tr=row['tracks']
        if not gs:continue
        pts=np.array([[g['u'],g['v']] for g in gs]); boxes=np.array([x[1:5] for x in tr]).reshape(-1,4)
        centres=(boxes[:,:2]+boxes[:,2:])/2
        cost=np.linalg.norm(pts[:,None,:]-centres[None,:,:],axis=2)
        matches={}
        if cost.size:
            a,b=linear_sum_assignment(np.where(cost<=gate,cost,1e6))
            matches={int(i):int(j) for i,j in zip(a,b) if cost[i,j]<=gate}
        for i,g in enumerate(gs):
            s=stats[g['id']];s['total']+=1
            contiguous=s['last_annotated'] is not None and f-s['last_annotated']==1
            if i in matches:
                j=matches[i];tid=int(tr[j][0]);s['matched']+=1;s['ids'][tid]+=1
                distances.append(float(cost[i,j]));cross_identity[tid].add(g['id'])
                if s['last_id'] is not None and s['last_id']!=tid:
                    s['switches_including_reacquisition']+=1
                    if s['last_frame']==f-1:s['switches_consecutive']+=1
                if contiguous and not s['was_matched'] and s['last_id'] is not None:s['fragments']+=1
                s['last_id']=tid;s['last_frame']=f;s['was_matched']=True
            else:s['was_matched']=False
            s['last_annotated']=f
    targets=[]
    for name,s in stats.items():
        targets.append(dict(id=name,**{k:v for k,v in s.items() if not k.startswith('last_') and k!='was_matched'},
            dominant_id_samples=max(s['ids'].values(),default=0),distinct_ids=len(s['ids'])))
    total=sum(x['total'] for x in targets);matched=sum(x['matched'] for x in targets)
    return dict(gate_px=gate,total_label_samples=total,matched=matched,coverage=matched/max(1,total),
        switches_consecutive=sum(x['switches_consecutive'] for x in targets),
        switches_including_reacquisition=sum(x['switches_including_reacquisition'] for x in targets),
        fragments=sum(x['fragments'] for x in targets),
        dominant_identity_coverage=sum(x['dominant_id_samples'] for x in targets)/max(1,total),
        track_ids_matching_multiple_gt=sum(len(v)>1 for v in cross_identity.values()),targets=targets,
        median_center_error=float(np.median(distances)) if distances else None)


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    status=out/'status.json';started=time.perf_counter()
    status.write_text(json.dumps(dict(state='running',args=vars(args))))
    try:
        torch.set_num_threads(4);torch.manual_seed(123);np.random.seed(123)
        if args.live:
            from airavoid.detector import Detector
            detector=Detector(ROOT/'artifacts/yolo26m_local.engine',conf=.03,tiled=True,gpu_preprocess=True,tile_edge_margin=2)
        else:detector=None
        reports=[]
        for seq in args.sequences.split(','):
            if seq.startswith('sw'):
                source=ROOT/f'data/mot_swarm/UAVSwarm-{seq[2:]}'
                video=ROOT/'data/mot_swarm/manifest.json';cache=ROOT/f'artifacts/mot/swarm_detections/{seq}.jsonl';gt=[]
                cap=ImageSequence(source)
            elif seq=='formation':
                video=ROOT/'data/formation_native.mp4';cache=ROOT/'demo/formation_final/detections.jsonl';gt=[]
                cap=cv2.VideoCapture(str(video))
            else:
                video=ROOT/f'data/qut_full/{seq}.mp4';cache=ROOT/f'artifacts/cadence_fullrate/{seq}/detections.jsonl'
                gt=json.loads((ROOT/f'data/qut_full/{seq}.gt.json').read_text())
                cap=cv2.VideoCapture(str(video))
            records=[json.loads(x) for x in cache.read_text().splitlines()]
            w=int(cap.get(3));h=int(cap.get(4));fps=cap.get(5)
            n=int(cap.get(7));start=max(0,n-args.frames) if seq!='formation' else 0
            # Formation cache has stride2; QUT caches have stride1. Missing cache rows
            # are never invented. All methods receive the exact same selected rows.
            selected=[r for r in records if r['frame']>=start]
            if args.max_frames:selected=selected[:args.max_frames]
            effective_fps=1/float(np.median(np.diff([r['time'] for r in selected]))) if len(selected)>1 else fps
            adapter=Adapter(args.method,w,h,effective_fps,args.high,not args.no_appearance)
            cap.set(cv2.CAP_PROP_POS_FRAMES,start);position=start;rows=[];timings=[]
            writer=None
            executor=None;pending=deque()
            if args.render:
                from airavoid.media import TimestampWriter
                writer=TimestampWriter(out/f'{seq}.mp4',effective_fps,(w,h),encoder='h264_nvenc')
                if args.async_export:executor=ThreadPoolExecutor(max_workers=1)
            processing_start=time.perf_counter()
            for r in selected:
                step_start=time.perf_counter()
                while position<=r['frame']:
                    ok,img=cap.read();position+=1
                    if not ok:raise ValueError(f'Decode ended {seq} {position}')
                dets=detector(img) if detector is not None else r['detections']
                with torch.inference_mode():tracks=adapter.update(dets,img,r['time'])
                if args.method in ['tracktrack','botsort','diffmot','trackssm','putr']:torch.cuda.synchronize()
                if writer is not None:
                    frozen=tuple(tuple(float(v) for v in tr) for tr in tracks)
                    if executor:
                        if len(pending)>=2:pending.popleft().result()
                        pending.append(executor.submit(export_frame,writer,img,frozen,args.method,r['time'],r['time']-selected[0]['time']))
                    else:export_frame(writer,img,frozen,args.method,r['time'],r['time']-selected[0]['time'])
                timings.append(time.perf_counter()-step_start)
                row=dict(frame=r['frame'],time=r['time'],tracks=[[float(x) for x in tr] for tr in tracks])
                rows.append(row)
            if executor:
                for future in pending:future.result()
                executor.shutdown(wait=True)
            if writer is not None:writer.close(rows[-1]['time']-rows[0]['time']+1/effective_fps)
            elapsed=time.perf_counter()-processing_start;cap.release()
            (out/f'{seq}.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
            report=dict(sequence=seq,frames=len(rows),source_size=[w,h],source_fps=fps,start_frame=start,
                processed_source_frame_step=float(np.median(np.diff([x['frame'] for x in rows]))) if len(rows)>1 else None,
                update_rate_hz=effective_fps,
                processing_seconds=elapsed,processing_fps=len(rows)/elapsed,
                frame_ms_p50=float(np.percentile(timings,50)*1000),frame_ms_p95=float(np.percentile(timings,95)*1000),
                input_sha256=sha(video),detection_cache_sha256=sha(cache) if detector is None else None,
                metrics=[identity_metrics(rows,gt,g) for g in [5,10,20]] if gt else None)
            reports.append(report)
            print(json.dumps({k:v for k,v in report.items() if k!='metrics'}),flush=True)
            del adapter
            gc.collect();torch.cuda.empty_cache()
        result=dict(state='complete',created_utc=datetime.now(timezone.utc).isoformat(),args=vars(args),
            runtime=dict(torch=torch.__version__,cv2=cv2.__version__,cv2_path=cv2.__file__,gpu=torch.cuda.get_device_name()),
            scope=('Actual decode + fixed YOLO26m + online tracking'+(' + drawing + H264 NVENC export' if args.render else '; no render/export')) if detector else 'Cached fixed YOLO26m detections + actual decode + online tracking; NOT end-to-end detector FPS',
            whole_process_internal_seconds=time.perf_counter()-started,runs=reports)
        (out/'report.json').write_text(json.dumps(result,indent=2));status.write_text(json.dumps(dict(state='complete')))
    except Exception as e:
        status.write_text(json.dumps(dict(state='failed',error=repr(e),traceback=traceback.format_exc()),indent=2));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['custom','tracktrack','botsort','ocsort','diffmot','putr','trackssm'])
    p.add_argument('--sequences',default='M1,M2,M6,M7,S1,S2');p.add_argument('--frames',type=int,default=875)
    p.add_argument('--max-frames',type=int);p.add_argument('--high',type=float,default=.03)
    p.add_argument('--no-appearance',action='store_true');p.add_argument('--live',action='store_true')
    p.add_argument('--render',action='store_true')
    p.add_argument('--async-export',action='store_true',help='Overlap ordered draw/export with inference; one worker, max two pending frames')
    p.add_argument('--output',required=True);run(p.parse_args())
