from collections import deque
from dataclasses import dataclass,field
import cv2
import numpy as np


def center(box):return np.array([(box[0]+box[2])/2,(box[1]+box[3])/2])


@dataclass
class Track:
    id:int
    box:list
    confidence:float
    class_id:int
    last:float
    history:deque=field(default_factory=lambda:deque(maxlen=500))
    scale_history:deque=field(default_factory=lambda:deque(maxlen=500))
    raw_history:deque=field(default_factory=lambda:deque(maxlen=100))
    ambiguous_until:float=0.
    hits:int=0
    source:str='detector'
    clipped:bool=False


class Tracker:
    """Conservative motion-gated association; ambiguous close matches invalidate risk."""
    def __init__(self,max_gap=.8,image_edge_margin=0.):
        self.tracks=[];self.next_id=1;self.max_gap=max_gap;self.image_edge_margin=image_edge_margin

    def reset_evidence(self):
        for tr in self.tracks:
            tr.history.clear();tr.scale_history.clear()

    def update(self,detections,t,transform,image_size=None):
        self.tracks=[x for x in self.tracks if t-x.last<=self.max_gap]
        candidates=[]
        for i,tr in enumerate(self.tracks):
            pred=center(tr.box)
            if len(tr.raw_history)>=3:
                prev=tr.raw_history[-3];last=tr.raw_history[-1]
                vel=(np.array(last[1:3])-prev[1:3])/max(last[0]-prev[0],.01)
                pred=pred+vel*(t-tr.last)
            for j,b in enumerate(detections):
                dist=np.linalg.norm(center(b)-pred)
                old=max(tr.box[2]-tr.box[0],tr.box[3]-tr.box[1],4)
                new=max(b[2]-b[0],b[3]-b[1],4)
                ratio=abs(np.log(new/old))
                gate=max(16,old*1.8)+30*(t-tr.last)
                if dist<gate and ratio<1.0:
                    cost=dist/gate+ratio*.3+(.15 if int(b[5])!=tr.class_id else 0)
                    candidates.append((cost,i,j))
        matched_i=set();matched_j=set()
        for cost,i,j in sorted(candidates):
            if i in matched_i or j in matched_j:continue
            tr=self.tracks[i]
            competition=[v for v,ii,jj in candidates if (ii==i and jj!=j) or (jj==j and ii!=i)]
            if competition and min(competition)<cost+.15:
                tr.ambiguous_until=t+2.
                tr.history.clear();tr.scale_history.clear()
            self._observe(tr,detections[j],t,transform,image_size)
            matched_i.add(i);matched_j.add(j)
        for j,b in enumerate(detections):
            if j in matched_j:continue
            tr=Track(self.next_id,b[:4],float(b[4]),int(b[5]),t)
            self.next_id+=1
            self._observe(tr,b,t,transform,image_size);self.tracks.append(tr)
        return self.tracks

    def _observe(self,tr,b,t,transform,image_size=None):
        tr.box=list(map(float,b[:4]));tr.confidence=float(b[4]);tr.class_id=int(b[5]);tr.last=t;tr.hits+=1
        tr.source=str(b[6]) if len(b)>6 else 'detector'
        tr.clipped=False
        if image_size is not None and self.image_edge_margin>0:
            w,h=image_size;m=self.image_edge_margin
            tr.clipped=b[0]<=m or b[1]<=m or b[2]>=w-m or b[3]>=h-m
        if tr.clipped:
            # Apparent span is truncated; do not let it seed later growth evidence.
            tr.history.clear();tr.scale_history.clear()
        pos=center(b);stable=transform@np.r_[pos,1]
        scale=np.sqrt(abs(np.linalg.det(transform[:2,:2])))
        span=max(b[2]-b[0],b[3]-b[1])*scale
        if not tr.clipped:
            tr.history.append((t,float(stable[0]),float(stable[1]),float(span)))
            if tr.source in ['detector','zoom']:tr.scale_history.append((t,float(stable[0]),float(stable[1]),float(span)))
        tr.raw_history.append((t,float(pos[0]),float(pos[1])))


class CameraMotion:
    """Background similarity diagnostic, masked KLT/RANSAC; no metric ego motion."""
    def __init__(self,mode='auto'):
        self.mode=mode;self.prev=None;self.cumulative=np.eye(3);self.previous_boxes=[]
        self.bad_until=-1.
        self.was_valid=False;self.epoch=0

    def update(self,frame,t,boxes):
        small_height=max(32,round(frame.shape[0]*640/frame.shape[1]))
        gray=cv2.cvtColor(cv2.resize(frame,(640,small_height)),cv2.COLOR_BGR2GRAY)
        valid=False;inliers=0;motion=None;status='unobserved';transform=np.eye(3);full=None
        if self.prev is not None:
            mask=np.full(self.prev.shape,255,np.uint8)
            sx=640/frame.shape[1];sy=small_height/frame.shape[0]
            for box in self.previous_boxes:
                x1,y1,x2,y2=box[:4]
                cv2.rectangle(mask,(int(x1*sx)-10,int(y1*sy)-10),(int(x2*sx)+10,int(y2*sy)+10),0,-1)
            pts=cv2.goodFeaturesToTrack(self.prev,250,.015,12,mask=mask)
            if pts is not None and len(pts)>=15:
                nxt,st,err=cv2.calcOpticalFlowPyrLK(self.prev,gray,pts,None,winSize=(21,21),maxLevel=3)
                use=st.ravel().astype(bool)&(err.ravel()<25)
                if np.sum(use)>=12:
                    mat,inside=cv2.estimateAffinePartial2D(pts[use],nxt[use],method=cv2.RANSAC,ransacReprojThreshold=1.2)
                    if mat is not None:
                        inliers=int(inside.sum());frac=inliers/max(1,np.sum(use))
                        motion=float(np.median(np.linalg.norm(nxt[use]-pts[use],axis=2)))
                        scale=float(np.hypot(mat[0,0],mat[0,1]))
                        angle=float(abs(np.arctan2(mat[1,0],mat[0,0])))
                        valid=inliers>=15 and frac>.65 and .97<scale<1.03 and angle<.03 and motion<30
                        status='background_similarity' if valid else 'unreliable'
                        if valid:
                            full=np.eye(3)
                            scaling=np.diag([sx,sy])
                            full[:2,:2]=np.linalg.inv(scaling)@mat[:2,:2]@scaling
                            full[:2,2]=mat[:2,2]/[sx,sy]
                        if motion>8 or angle>.02 or abs(scale-1)>.015:
                            self.bad_until=t+2.
            if self.mode=='fixed':
                valid= t>=self.bad_until
                status='fixed_setup_assumption' if valid else 'motion_violates_fixed_setup'
            else:
                valid=valid and t>=self.bad_until
        elif self.mode=='fixed':valid=True;status='fixed_setup_assumption'
        self.prev=gray;self.previous_boxes=boxes
        if self.mode=='unknown':valid=False;status='unknown_camera'
        reset=False
        if self.mode=='auto':
            # Unobserved camera intervals destroy the old stabilization reference.
            # Start a new gauge and require fresh target evidence after reacquisition.
            if valid and not self.was_valid:
                self.cumulative=np.eye(3);self.epoch+=1;reset=True
            elif valid and full is not None:
                self.cumulative=full@self.cumulative
            elif self.was_valid and not valid:
                reset=True
            if valid:transform=np.linalg.inv(self.cumulative)
        self.was_valid=bool(valid)
        return transform,{'valid':bool(valid),'status':status,'inliers':inliers,'flow_px_at_640':motion,
                          'mode':self.mode,'metric_motion_available':False,'rotation_translation_disambiguated':False,
                          'validity_scope':'background image similarity only' if self.mode=='auto' else 'declared setup / image diagnostic',
                          'evidence_reset':reset,'reference_epoch':self.epoch}
