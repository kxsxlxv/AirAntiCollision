"""Causal MOT on frozen detections; GT annotations are never opened here.

Keep the upstream emitted/smoothed box for MOT scoring. Independently export the
actual matched detector measurement for threat assessment; no predicted size.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mot_models import Adapter
import torch
with torch.inference_mode():
    torch.nn.Conv2d(3,4,3).cuda()(torch.zeros(1,3,16,16,device='cuda'))
    torch.cuda.synchronize()
import cv2
import av
import numpy as np
from airavoid.tracking import CameraMotion
from airavoid.provenance import file_sha256,validate_cache_entry


def matched_measurements(adapter,tracks,detections,t):
    """Use upstream raw matched detection, never a Kalman extent or an IoU guess."""
    raw={}
    ambiguity={}
    if adapter.name=='tracktrack':
        raw={tr.track_id:np.asarray(tr.box,float) for tr in adapter.tracker.tracks
             if tr.end_frame_id==adapter.frame and tr.state==1}
    elif adapter.name=='custom':
        raw={tr.id:np.asarray(tr.box,float) for tr in adapter.tracker.tracks if tr.last==t}
        ambiguity={tr.id:t<tr.ambiguous_until for tr in adapter.tracker.tracks}
    elif adapter.name=='putr':
        # Adapter emits PuTR's ORIG_TLBR (actual current input detections).
        raw={int(tr[0]):np.asarray(tr[1:5],float) for tr in tracks}
    else:raise ValueError('Stage 2 only audits measurement extraction for custom/TrackTrack/PuTR')
    d=np.asarray(detections,float).reshape(-1,6);used=set();result=[]
    for tid in [int(tr[0]) for tr in tracks]:
        if tid not in raw:continue
        distance=np.max(np.abs(d[:,:4]-raw[tid]),axis=1) if len(d) else np.array([])
        matches=np.flatnonzero(distance<1e-3)
        if len(matches)!=1 or int(matches[0]) in used:
            raise ValueError(f'No unique exact input measurement for emitted track {tid}')
        index=int(matches[0]);used.add(index)
        result.append({'track_id':tid,'detection_index':index,'measured_bbox':d[index,:4].tolist(),
                       'confidence':float(d[index,4]),'class_id':int(d[index,5]),
                       'time':t,'source':'detector','measured':True,'ambiguous':ambiguity.get(tid,False)})
    return result


def run(args):
    torch.set_num_threads(4);torch.manual_seed(123);np.random.seed(123)
    protocol=json.loads((ROOT/'reports/stage2_protocol.json').read_text())
    audit=json.loads((ROOT/'reports/stage2_dataset_audit.json').read_text())
    clips=args.clips.split(',') if args.clips else protocol['development']+protocol['evaluation']
    output=Path(args.output or ROOT/f'artifacts/stage2/tracks/{args.detector}/{args.method}/s{args.stride}')
    output.mkdir(parents=True,exist_ok=True)
    summaries={}
    for clip in clips:
        detail=audit['clips'][clip];cache=ROOT/f'artifacts/stage2/detections/{args.detector}/{clip}'
        benchmark=json.loads((cache/'benchmark.json').read_text())
        if file_sha256(cache/'detections.jsonl')!=benchmark['detections_sha256']:
            raise ValueError('Frozen detector cache changed')
        rows={r['frame']:r for r in map(json.loads,(cache/'detections.jsonl').read_text().splitlines())}
        w,h=detail['resolution'];fps=detail['effective_fps_from_pts']
        tick=time.perf_counter();adapter=Adapter(args.method,w,h,fps/args.stride)
        if args.method=='custom':adapter.tracker.image_edge_margin=1.
        init=time.perf_counter()-tick
        camera=CameraMotion('auto');previous=np.eye(3);timings=[];processed=0
        start=time.perf_counter()
        with av.open(detail['video_path']) as container,(output/f'{clip}.jsonl').open('w') as out:
            stream=container.streams.video[0];stream.thread_type='AUTO';origin=None
            for index,frame in enumerate(container.decode(stream)):
                pts=float(frame.pts*frame.time_base)
                if origin is None:origin=pts
                if index%args.stride:continue
                t=pts-origin
                row=rows[index];validate_cache_entry(row,index,t)
                image=frame.to_ndarray(format='bgr24')
                # Matched fixed-wing condition. Five-class A cache is preserved.
                dets=[b for b in row['detections'] if int(b[5])==0]
                transform,cam=camera.update(image,t,dets)
                warp=(np.linalg.inv(transform)@previous)[:2] if cam['valid'] and not cam['evidence_reset'] else np.eye(2,3)
                previous=transform
                tick=time.perf_counter();tracks=adapter.update(dets,image,t,warp=warp)
                timings.append((time.perf_counter()-tick)*1000)
                measurements=matched_measurements(adapter,tracks,dets,t)
                record={'frame':index,'time':t,'source_pts_s':pts,'tracks':[list(map(float,tr)) for tr in tracks],
                        'measurements':measurements,'camera':cam,'stabilization_transform':transform.tolist(),
                        'detection_count':len(dets),'detector_variant':args.detector,
                        'tracker':args.method,'normal_inference_gt_access':False}
                out.write(json.dumps(record,allow_nan=False)+'\n');processed+=1
                if processed%500==0:print(f'{args.detector}/{args.method}/{clip}: {processed} updates',flush=True)
        elapsed=time.perf_counter()-start
        summaries[clip]={'frames':processed,'source_pts_preserved':True,'tracker_initialization_s':init,
                         'elapsed_replay_s':elapsed,'replay_fps_not_detector_fps':processed/elapsed,
                         'association_ms':{k:float(np.percentile(timings,q)) for k,q in [('p50',50),('p95',95),('p99',99)]},
                         'cache_sha256':benchmark['detections_sha256'],
                         'trace_sha256':file_sha256(output/f'{clip}.jsonl'),
                         'measurement_extraction':'exact upstream raw match (<.001 px), no smoothed/predicted scale',
                         'scope':'fixed-wing condition; A retains five-class default outside this experiment'}
        (output/'benchmark.json').write_text(json.dumps({'args':vars(args),'clips':summaries},indent=2))
        print(f'Complete {args.detector}/{args.method}/{clip}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--detector',choices=['A','B'],required=True)
    p.add_argument('--method',choices=['custom','tracktrack','putr'],required=True)
    p.add_argument('--clips');p.add_argument('--stride',type=int,default=1);p.add_argument('--output')
    a=p.parse_args()
    if a.stride<1:p.error('stride must be positive')
    run(a)
