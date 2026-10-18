"""Experimental joint inverse-span fit: exact straight-line pinhole projection model.

Sensitivity intervals propagate joint fitted covariance, not calibrated safety bounds.
"""
import numpy as np
from .geometry import Config,ThreatEstimator


class ProjectiveEstimator(ThreatEstimator):
    def __init__(self,config=None):
        super().__init__(config)
        self.normals=np.random.default_rng(9917).normal(size=(256,6))

    def assess(self,history,time,camera_valid=True,ambiguous=False):
        c=self.config
        # Common gates and schema; never inherit a decision from the log-span model.
        base=super().assess(history,time,camera_valid,ambiguous)
        base['method']='joint_inverse_span'
        a=np.asarray([h for h in history if time-h[0]<=c.window],float)
        if not len(a) or not camera_valid or ambiguous or time-a[-1,0]>.25 or len(a)<c.min_samples or a[-1,0]-a[0,0]<c.min_history:
            return base
        out={**base,'state':'uncertain','reason':'growth_unresolved','miss_spans':None,'miss_low':None,'miss_high':None,'tau':None}
        spans=np.maximum(a[:,3],1)
        if np.median(spans)<c.min_span:out['reason']='tiny';return out
        origin=a[-1,1:3]
        q=1/spans
        y=np.c_[q,(a[:,1:3]-origin)*q[:,None]]
        x=np.c_[np.ones(len(a)),a[:,0]-a[-1,0]]
        # Coordinate origin reduces numerical cancellation; it cancels at q=0.
        floors=np.array([max(.025*np.median(q),1/np.median(spans)**2),1/np.median(spans),1/np.median(spans)])
        weights=np.ones(len(a))
        for _ in range(4):
            fit=np.linalg.lstsq(x*weights[:,None]**.5,y*weights[:,None]**.5,rcond=None)[0]
            res=y-x@fit
            sigma=np.maximum(floors,1.4826*np.median(np.abs(res-np.median(res,axis=0)),axis=0))
            rr=np.max(np.abs(res)/sigma,axis=1)
            weights=np.minimum(1,1.5/np.maximum(rr,1e-9))
        aq,ax,ay=fit[0];bq,bx,by=fit[1]
        if aq<=0 or np.any(x@fit[:,0]<=0):out['reason']='aspect';return out
        predicted_spans=1/(x@fit[:,0])
        scale_res=float(np.sqrt(np.mean((np.log(spans)-np.log(predicted_spans))**2)))
        span=1/aq;g=-bq/aq
        velocity=(fit[1,1:]*aq-fit[0,1:]*bq)/aq**2
        drift=float(np.linalg.norm(velocity))
        residual_cov=(res*weights[:,None]).T@res/max(1,len(a)-2)
        # Preserve empirical cross-output covariance, add measurement variance floors.
        residual_cov+=np.diag(floors**2)
        time_cov=np.linalg.pinv(x.T@(weights[:,None]*x))
        covariance=np.kron(time_cov,residual_cov)
        eigenvalues,vectors=np.linalg.eigh(covariance)
        samples=fit.ravel()+self.normals@(vectors@np.diag(np.sqrt(np.maximum(eigenvalues,0)))).T
        growth=-samples[:,3]/np.maximum(samples[:,0],1e-9)
        gl,gh=np.quantile(growth,[.01,.99]);eg=float(np.std(growth))
        sigma_px=max(1.,float(np.sqrt(np.mean(np.sum(res[:,1:]**2,axis=1)))/aq))
        out.update(span=float(span),growth=float(g),growth_se=eg,velocity=velocity.tolist(),drift=drift,
                   position_sigma=sigma_px,scale_residual=scale_res)
        if scale_res>c.max_log_residual:out['reason']='aspect';return out
        if gl>c.growth_floor and bq<0:
            tau=-aq/bq;intercept=fit[0,1:]+fit[1,1:]*tau
            trial_tau=-samples[:,0]/np.minimum(samples[:,3],-1e-9)
            trial_miss=np.linalg.norm(samples[:,1:3]+samples[:,4:6]*trial_tau[:,None],axis=1)
            lo,hi=np.quantile(trial_miss,[.01,.99]);miss=float(np.linalg.norm(intercept))
            out.update(tau=float(tau),miss_spans=miss,miss_low=float(lo),miss_high=float(hi),
                       plane_intercept_spans=intercept.tolist())
            if hi<c.threat_miss and tau<c.horizon:out.update(state='threat',reason='bearing_growth')
            elif lo>c.nonthreat_miss:out.update(state='nonthreat',reason='crossing')
            else:out['reason']='overlap'
        elif gh<-.01:
            if c.allow_recession_clearance:out.update(state='nonthreat',reason='receding')
            else:out.update(state='uncertain',reason='shrink_ambiguous')
        # No-growth crossing candidate omitted until a joint-velocity uncertainty
        # gate is evaluated; this experimental variant is deliberately conservative.
        return out
