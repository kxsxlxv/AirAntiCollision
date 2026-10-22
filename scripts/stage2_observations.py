"""Freeze genuine detector observations on original NPS video, preserving PTS.

No annotation CSV or GT identity access in this inference module. Source identity
comes from the checked audit/metadata. See stage2_audit.py for offline checks.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
# Bind Torch cuDNN before custom OpenCV adjusts DLL search paths.
with torch.inference_mode():
    torch.nn.Conv2d(3,4,3).cuda()(torch.zeros(1,3,16,16,device='cuda'))
    torch.cuda.synchronize()
import cv2
import av
import numpy as np
from airavoid.detector import Detector,NAMES
from airavoid.detector_p2 import DetectorP2
from airavoid.provenance import file_sha256,validate_cache_entry


def stats(values):
    return {k:float(np.percentile(values,p)) for k,p in [('p50',50),('p95',95),('p99',99)]} | {'mean':float(np.mean(values))}


def make_detector(variant,weights=None):
    if variant=='A':
        p=Path(weights or ROOT/'artifacts/yolo26m_local.engine')
        detector=Detector(p,conf=.03,tiled=True,gpu_preprocess=True,tile_edge_margin=2)
        contract={'variant':'A','input_shape':[1,3,416,704],'taxonomy':NAMES,'conf':.03,
                  'preprocess':'native CUDA OpenCV RGB NCHW /255, centered 114 letterbox',
                  'tiled':True,'tile_shape':[416,704],'cross_box_nms_iou':.45,'tile_edge_margin':2}
    else:
        p=Path(weights or ROOT/'artifacts/stage2/yolo26m_p2_local_fp32.engine')
        detector=DetectorP2(p,conf=.03)
        contract=detector.contract
    contract.update(weights_sha256=file_sha256(p),backend=detector.providers)
    return detector,contract


def run(args):
    torch.set_num_threads(4)
    output=Path(args.output or ROOT/f'artifacts/stage2/detections/{args.detector}')
    output.mkdir(parents=True,exist_ok=True)
    audit=json.loads((ROOT/'reports/stage2_dataset_audit.json').read_text())
    protocol=json.loads((ROOT/'reports/stage2_protocol.json').read_text())
    clips=args.clips.split(',') if args.clips else protocol['development']+protocol['evaluation']
    start=time.perf_counter()
    detector,contract=make_detector(args.detector,args.weights)
    cold=time.perf_counter()-start
    for clip in clips:
        detail=audit['clips'][clip]
        source=Path(detail['video_path'])
        if file_sha256(source)!=detail['video_sha256']:
            raise ValueError('Source video changed since checked audit')
        side=output/clip
        side.mkdir(exist_ok=True)
        if (side/'benchmark.json').exists() and not args.overwrite:
            raise FileExistsError(f'{side} already completed; choose a new output')
        timings=defaultdict(list);calls=0;processed=0
        with av.open(str(source)) as container:
            stream=container.streams.video[0];stream.thread_type='AUTO'
            first=next(container.decode(stream)).to_ndarray(format='bgr24')
        warm=time.perf_counter()
        for _ in range(5):detector(first)
        warm=time.perf_counter()-warm
        detector.reset_temporal()
        origin=None;last=None
        start=time.perf_counter()
        with av.open(str(source)) as container,(side/'detections.jsonl').open('w') as out:
            stream=container.streams.video[0];stream.thread_type='AUTO'
            iterator=iter(container.decode(stream));decode_ms=0.
            i=0
            while True:
                tick=time.perf_counter()
                frame=next(iterator,None)
                if frame is None:break
                image=frame.to_ndarray(format='bgr24')
                decode_ms+=(time.perf_counter()-tick)*1000
                pts=float(frame.pts*frame.time_base)
                if origin is None:origin=pts
                if i%args.stride:
                    i+=1;continue
                if args.max_frames and processed>=args.max_frames:break
                t=pts-origin
                tick=time.perf_counter();boxes=detector(image);infer_ms=(time.perf_counter()-tick)*1000
                calls+=detector.last_network_calls
                row={'schema':'stage2-observation-v1','frame':i,'time':t,'source_pts_s':pts,
                     'detections':boxes,'measurement_source':'detector','measurement_time_s':t,
                     'measured':True,'detector_variant':args.detector}
                validate_cache_entry(row,i,t)
                out.write(json.dumps(row,allow_nan=False)+'\n')
                timings['decode_ms'].append(decode_ms);timings['detector_ms'].append(infer_ms)
                timings['decode_infer_ms'].append(decode_ms+infer_ms)
                decode_ms=0.;processed+=1;i+=1;last=t
                if processed%250==0:print(f'{args.detector} {clip}: {processed}, {processed/(time.perf_counter()-start):.2f} detector FPS',flush=True)
        elapsed=time.perf_counter()-start
        report={'kind':'actual decode + neural detector cache generation; no tracking/assessment/render',
                'args':vars(args),'input_sha256':detail['video_sha256'],'source':str(source),
                'frames':processed,'stride':args.stride,'source_frame_count':detail['frames'],
                'resolution':detail['resolution'],'source_pts_origin_s':origin,'last_processed_time_s':last,
                'source_fps':detail['effective_fps_from_pts'],'sampled_fps':detail['effective_fps_from_pts']/args.stride,
                'contract':contract,'network_calls':calls,'elapsed_s':elapsed,'processing_fps':processed/elapsed,
                'first_model_load_s':cold,'warmup_five_updates_s':warm,
                'stage_ms':{k:stats(v) for k,v in timings.items()},
                'detections_sha256':file_sha256(side/'detections.jsonl'),
                'cv2':cv2.__version__,'cv2_path':cv2.__file__,'gpu':torch.cuda.get_device_name(0),
                'gt_available_to_inference':False,'clock':'original decoded presentation timestamps'}
        (side/'benchmark.json').write_text(json.dumps(report,indent=2))
        print(f'Complete {args.detector} {clip} {processed} frames',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--detector',choices=['A','B'],required=True)
    p.add_argument('--clips');p.add_argument('--weights');p.add_argument('--output')
    p.add_argument('--stride',type=int,default=1);p.add_argument('--max-frames',type=int)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if args.stride<1:p.error('stride must be positive')
    run(args)
