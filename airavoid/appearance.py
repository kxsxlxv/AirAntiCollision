"""Optional appearance bridge using the user's existing CUDA LoRAT adapter.

All measurements are image-derived. No GT is read. SOT scores are uncalibrated and
must never be interpreted as collision probabilities or detector class confidence.
"""
from pathlib import Path
import hashlib
import sys
import time
import cv2
import numpy as np


class AppearanceBridge:
    def __init__(self,engine_path='artifacts/lorat_local.engine',reference_path='C:/Data/University/Specialist/Diploma/FastVision',max_targets=3,score_threshold=.5):
        reference=Path(reference_path)
        if not Path(engine_path).exists():raise FileNotFoundError('Build local LoRAT engine first; see scripts/build_engine.py')
        sys.path.insert(0,str(reference))
        from detection_track.lorat_trt import LoRATTensorRT
        self.pool=[LoRATTensorRT(engine_path) for _ in range(max_targets)]
        self.active={};self.score_threshold=score_threshold
        self.stream=cv2.cuda_Stream();self.gpu=cv2.cuda_GpuMat();self.uploaded_token=None
        self.last_network_calls=0;self.last_rejected=0
        self.metadata={'engine':str(Path(engine_path).resolve()),'score_threshold':score_threshold,'max_targets':max_targets,
                       'engine_sha256':hashlib.sha256(Path(engine_path).read_bytes()).hexdigest(),
                       'external_source_hashes':{name:hashlib.sha256((reference/'detection_track'/name).read_bytes()).hexdigest() for name in ['lorat_trt.py','trt_gpu.py']}}

    def upload(self,frame,token):
        if self.uploaded_token!=token:
            self.gpu.upload(frame,self.stream);self.stream.waitForCompletion();self.uploaded_token=token

    def release_lost(self,tracks,t):
        current={tr.id:tr for tr in tracks}
        for ident in list(self.active):
            if ident not in current or t-current[ident].last>.8:
                tracker,_=self.active.pop(ident);tracker.reset();self.pool.append(tracker)

    def predict(self,frame,tracks,t):
        self.release_lost(tracks,t)
        self.last_network_calls=0;self.last_rejected=0;predictions=[]
        if not self.active:return predictions
        self.upload(frame,t)
        for ident,(tracker,class_id) in list(self.active.items()):
            box,score=tracker.track(self.gpu,self.stream);self.last_network_calls+=1
            if not np.isfinite(score) or not np.isfinite(box).all() or score<self.score_threshold:
                self.last_rejected+=1;continue
            predictions.append([*box,float(score),class_id,'sot'])
        return predictions

    def refresh(self,frame,tracks,t):
        self.release_lost(tracks,t)
        eligible=[tr for tr in tracks if tr.last==t and getattr(tr,'source','detector')=='detector']
        # Correct current state from an independent detector, retain fixed template.
        for tr in eligible:
            if tr.id in self.active:
                tracker,_=self.active[tr.id]
                if t<tr.ambiguous_until:
                    tracker.reset();self.pool.append(tracker);del self.active[tr.id]
                else:
                    tracker.state=np.asarray(tr.box,float)
                    self.active[tr.id]=(tracker,tr.class_id)
        for tr in sorted(eligible,key=lambda tr:-(tr.hits*tr.confidence)):
            if not self.pool:break
            if tr.id in self.active or tr.hits<3 or t<tr.ambiguous_until:continue
            if min(tr.box[2]-tr.box[0],tr.box[3]-tr.box[1])<2:continue
            tracker=self.pool.pop()
            self.upload(frame,t)
            try:tracker.initialize(self.gpu,tr.box,self.stream)
            except ValueError:
                tracker.reset();self.pool.append(tracker);continue
            self.active[tr.id]=(tracker,tr.class_id)


def combine_observations(detections,predictions):
    """Prefer independent detector boxes; retain SOT only where detector has no match."""
    result=list(detections)
    for p in predictions:
        pc=np.array([(p[0]+p[2])/2,(p[1]+p[3])/2]);span=max(p[2]-p[0],p[3]-p[1],4)
        duplicate=False
        for d in detections:
            dc=np.array([(d[0]+d[2])/2,(d[1]+d[3])/2])
            ds=max(d[2]-d[0],d[3]-d[1],4)
            # The weak detector can change bird/airplane labels on the same tiny
            # physical object. Class disagreement must not duplicate its observations.
            if max(span,ds)/min(span,ds)<3 and np.linalg.norm(pc-dc)<max(8,.6*min(span,ds)):
                duplicate=True;break
        if not duplicate:result.append(p)
    return result
