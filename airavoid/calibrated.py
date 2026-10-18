"""Optional size-normalized CPA under explicitly supplied pinhole calibration.

No metric target dimension is inferred. All distances remain in assumed constant
projected target-span units. Fit intervals are sensitivity diagnostics, not CIs.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .geometry import ThreatEstimator


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    focal_px: float
    cx: float
    cy: float
    rectified: bool
    provenance: str

    def __post_init__(self):
        values=[self.width,self.height,self.focal_px,self.cx,self.cy]
        if not np.all(np.isfinite(values)) or min(self.width,self.height,self.focal_px)<=0:
            raise ValueError('Calibration must contain finite positive dimensions/focal length')
        if self.rectified is not True:
            raise ValueError('Calibrated geometry requires rectified pinhole RGB with square pixels')
        if not isinstance(self.provenance,str) or not self.provenance.strip():
            raise ValueError('Calibration provenance must be stated; never invent intrinsics')

    @classmethod
    def load(cls,path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))

    def validate_resolution(self,width,height):
        if (width,height)!=(self.width,self.height):
            raise ValueError('Calibration resolution differs from source RGB; rescale it explicitly')


class CalibratedCPAEstimator(ThreatEstimator):
    def __init__(self,calibration,config=None):
        super().__init__(config)
        self.calibration=calibration
        self.normals=np.random.default_rng(60841).normal(size=(512,6))

    def assess(self,history,time,camera_valid=True,ambiguous=False):
        base=super().assess(history,time,camera_valid,ambiguous)
        out={**base,'method':'calibrated_span_cpa','state':'uncertain',
             'miss_spans':None,'miss_low':None,'miss_high':None,'tau':None,
             'distance_meaning':'CPA in constant projected target-span units, not metres',
             'interval_meaning':'fitted-covariance sensitivity; calibration/shape bias excluded'}
        c=self.config;k=self.calibration
        a=np.asarray([h for h in history if time-h[0]<=c.window],float)
        if not len(a) or not camera_valid or ambiguous or time-a[-1,0]>.25 or len(a)<c.min_samples or a[-1,0]-a[0,0]<c.min_history:
            return out
        s=np.maximum(a[:,3],1.)
        if np.median(s)<c.min_span:out['reason']='tiny';return out
        # If s=f*W/Z with constant transverse W, these equal [X,Y,Z]/W.
        y=np.c_[(a[:,1]-k.cx)/s,(a[:,2]-k.cy)/s,k.focal_px/s]
        x=np.c_[np.ones(len(a)),a[:,0]-a[-1,0]]
        floors=np.maximum(1/np.median(s),np.maximum(.025,1/np.median(s))*np.median(np.abs(y),axis=0))
        weights=np.ones(len(a))
        for _ in range(4):
            fit=np.linalg.lstsq(x*np.sqrt(weights[:,None]),y*np.sqrt(weights[:,None]),rcond=None)[0]
            res=y-x@fit
            sigma=np.maximum(floors,1.4826*np.median(np.abs(res-np.median(res,axis=0)),axis=0))
            weights=np.minimum(1,1.5/np.maximum(np.max(np.abs(res)/sigma,axis=1),1e-9))
        r,v=fit
        if r[2]<=0 or np.any((x@fit)[:,2]<=0):out['reason']='aspect';return out
        predicted_span=k.focal_px/(x@fit)[:,2]
        rs=float(np.sqrt(np.mean((np.log(s)-np.log(predicted_span))**2)))
        out['scale_residual']=rs
        if rs>c.max_log_residual:out['reason']='aspect';return out
        residual_cov=(res*weights[:,None]).T@res/max(1,len(a)-2)+np.diag(floors**2)
        covariance=np.kron(np.linalg.pinv(x.T@(weights[:,None]*x)),residual_cov)
        eigenvalues,vectors=np.linalg.eigh(covariance)
        draws=fit.ravel()+self.normals@(vectors@np.diag(np.sqrt(np.maximum(eigenvalues,0)))).T
        rr,vv=draws[:,:3],draws[:,3:]
        vv2=np.sum(vv*vv,axis=1)
        times=-np.sum(rr*vv,axis=1)/np.maximum(vv2,1e-12)
        # Closest approach over future time only, including t=0.
        misses=np.linalg.norm(rr+np.maximum(times,0)[:,None]*vv,axis=1)
        tau=float(-r@v/max(v@v,1e-12));miss=float(np.linalg.norm(r+max(tau,0)*v))
        lo,hi=np.quantile(misses,[.01,.99]);tl,th=np.quantile(times,[.01,.99])
        out.update(miss_spans=miss,miss_low=float(lo),miss_high=float(hi),tau=max(0.,tau),
                   cpa_time_low=float(tl),cpa_time_high=float(th),relative_position_spans=r.tolist(),
                   relative_velocity_spans_per_s=v.tolist(),reason='overlap')
        # Require agreement of the sensitivity family on a future encounter.
        if tl>0 and th<c.horizon and hi<c.threat_miss:
            out.update(state='threat',reason='cpa_close')
        elif tl>0 and lo>c.nonthreat_miss:
            out.update(state='nonthreat',reason='cpa_far')
        elif th<=0:out['reason']='cpa_past'
        return out
