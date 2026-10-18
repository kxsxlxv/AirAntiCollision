"""Optional neural-sky-conditioned tiny-object recall path, in original pixels."""
import cv2
import numpy as np
import json
from pathlib import Path
from .provenance import file_sha256
from .detector import interior_boxes


class SkyProposals:
    def __init__(self,mask_path,source_fps,interval=5,topk=12,crop_size=96,source_hash=None,source_frames=None,all_region=False):
        self.cap=None
        if mask_path is None:
            if not all_region:raise ValueError('A sky mask or explicit all-pixel proposal region is required')
            self.provenance={'status':'no_sky_model','region':'all_pixels','input_sha256':source_hash}
        else:
            self.cap=cv2.VideoCapture(str(mask_path))
            if not self.cap.isOpened():raise FileNotFoundError(mask_path)
            fps=self.cap.get(cv2.CAP_PROP_FPS)
            try:
                if not np.isfinite(fps) or abs(fps-source_fps)>.01:raise ValueError('Sky mask FPS must match source RGB FPS')
                frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if source_frames and frames!=source_frames:raise ValueError('Sky mask frame count must match source RGB')
                manifest_path=Path(str(mask_path)+'.json')
                self.provenance={'status':'legacy_unverified','reason':'No mask/source identity manifest',
                                 'mask_sha256':file_sha256(mask_path),'frames':frames}
                if manifest_path.exists():
                    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
                    if not source_hash or manifest.get('input_sha256')!=source_hash:raise ValueError('Sky mask source SHA-256 differs from input RGB')
                    if manifest.get('mask_sha256')!=self.provenance['mask_sha256']:raise ValueError('Sky mask SHA-256 differs from manifest')
                    self.provenance.update(status='sha256_verified',manifest=manifest)
            except Exception:
                self.cap.release();raise
        self.all_region=all_region
        self.interval=interval;self.topk=topk;self.crop_size=crop_size
        self.last_calls=0;self.last_proposals=[]
        self.kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

    def augment(self,detector,frame,base,source_frame,processed_frame):
        self.last_calls=0;self.last_proposals=[]
        if processed_frame%self.interval:return base
        h,w=frame.shape[:2]
        if self.cap is None:sky=np.ones((h,w),bool)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,source_frame);ok,mask=self.cap.read()
            if not ok:raise RuntimeError('Sky mask missing a source-aligned frame')
            sky=cv2.resize(mask[:,:,0],(w,h),interpolation=cv2.INTER_NEAREST)>127
            sky=cv2.dilate(sky.astype(np.uint8),np.ones((21,21),np.uint8)).astype(bool)
        if self.all_region:sky[:]=True
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        response=np.maximum(cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,self.kernel),cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,self.kernel)).astype(float)
        response[~sky]=0
        for b in base:
            x1,y1,x2,y2=b[:4]
            response[max(0,int(y1)-16):min(h,int(y2)+17),max(0,int(x1)-16):min(w,int(x2)+17)]=0
        values=response[sky]
        median=float(np.median(values)) if len(values) else 0.
        mad=float(np.median(np.abs(values-median))) if len(values) else 0.
        threshold=max(12.,median+6*1.4826*mad)
        extra=[]
        for _ in range(self.topk):
            y,x=np.unravel_index(np.argmax(response),response.shape);strength=float(response[y,x])
            if strength<threshold:break
            self.last_proposals.append([int(x),int(y),strength]);r=self.crop_size//2
            response[max(0,y-r):min(h,y+r+1),max(0,x-r):min(w,x+r+1)]=0
            left=max(0,min(w-self.crop_size,x-r));top=max(0,min(h-self.crop_size,y-r))
            self.last_calls+=1
            # Keep the CPU crop resize used in the evaluated recall experiment.
            # This is additional model inference, not a classical target classifier.
            crop=frame[top:top+self.crop_size,left:left+self.crop_size]
            predictions=detector._infer(crop)
            predictions=interior_boxes(predictions,left,top,crop.shape[1],crop.shape[0],w,h,
                                       getattr(detector,'tile_edge_margin',0.))
            for b in predictions:
                b[[0,2]]+=left;b[[1,3]]+=top
                extra.append(b.tolist()+['zoom'])
        combined=list(base)+extra
        if combined:
            ids=cv2.dnn.NMSBoxes([[b[0],b[1],b[2]-b[0],b[3]-b[1]] for b in combined],[b[4] for b in combined],detector.conf,.45)
            combined=[combined[int(i)] for i in np.asarray(ids).ravel()] if len(ids) else []
        return combined

    def close(self):
        if self.cap is not None:self.cap.release()
