"""Observable image-space evidence. No telemetry, metric range or class dimensions."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Config:
    window: float = 4.0
    min_span: float = 6.0
    min_history: float = 2.0
    min_samples: int = 10
    threat_miss: float = 2.0
    nonthreat_miss: float = 5.0
    horizon: float = 60.0
    max_log_residual: float = .16
    growth_floor: float = .008
    allow_recession_clearance: bool = False
    correlation_seconds: float = 0.
    bounded_pixel_error: float = 2.
    bounded_span_fraction: float = .1


def robust_line(t, values, floor,correlation_seconds=0.):
    """Huber IRLS; uncertainty floor prevents interpolated labels becoming exact."""
    t=np.asarray(t,float); y=np.asarray(values,float)
    x=np.c_[np.ones(len(t)),t-t[-1]]
    weights=np.ones(len(t))
    for _ in range(4):
        fit=np.linalg.lstsq(x*weights[:,None]**.5,y*weights**.5,rcond=None)[0]
        residual=y-x@fit
        sigma=max(floor,1.4826*np.median(np.abs(residual-np.median(residual))))
        weights=np.minimum(1,1.5*sigma/np.maximum(np.abs(residual),1e-8))
    bread=np.linalg.pinv(x.T@(weights[:,None]*x))
    cov=bread*sigma**2
    if correlation_seconds>0 and len(t)>3:
        # Bartlett-kernel sandwich over elapsed time, including irregular gaps.
        # Huber weighting + short histories make this a sensitivity correction,
        # not an asymptotically calibrated confidence interval.
        kernel=np.maximum(0.,1-np.abs(t[:,None]-t[None,:])/correlation_seconds)
        scores=x*(weights*residual)[:,None]
        meat=scores.T@kernel@scores
        hac=bread@meat@bread*(len(t)/max(1,len(t)-2))
        cov[1,1]=max(cov[1,1],hac[1,1])
    return float(fit[0]),float(fit[1]),float(np.sqrt(cov[1,1])),float(np.sqrt(np.mean(residual**2)))


class ThreatEstimator:
    def __init__(self, config=None):
        self.config=config or Config()

    def assess(self, history, time, camera_valid=True, ambiguous=False,scale_history=None):
        c=self.config
        recent=[h for h in history if time-h[0]<=c.window]
        out={'state':'uncertain','reason':'history','samples':len(recent),'camera_valid':bool(camera_valid),
             'growth':None,'growth_se':None,'drift':None,'miss_spans':None,'miss_low':None,'miss_high':None,
             'tau':None,'velocity':[0.,0.],'position_sigma':None,'scale_residual':None,'span':None}
        if not recent:return out
        a=np.asarray(recent,float)
        out['span']=float(a[-1,3])
        if time-a[-1,0]>.25:
            out['reason']='lost';return out
        if not camera_valid:
            out['reason']='camera';return out
        if ambiguous:
            out['reason']='association';return out
        if len(a)<c.min_samples or a[-1,0]-a[0,0]<c.min_history:return out
        scales=a if scale_history is None else np.asarray([h for h in scale_history if time-h[0]<=c.window],float)
        out['span_samples']=len(scales)
        if len(scales)<c.min_samples or scales[-1,0]-scales[0,0]<c.min_history:
            out['reason']='scale_history';return out
        if time-scales[-1,0]>.8:
            out['reason']='scale_stale';return out
        x,vx,ex,rx=robust_line(a[:,0],a[:,1],1.0,c.correlation_seconds)
        y,vy,ey,ry=robust_line(a[:,0],a[:,2],1.0,c.correlation_seconds)
        logs,g,eg,rs=robust_line(scales[:,0],np.log(np.maximum(scales[:,3],1)),max(.025,1/max(np.median(scales[:,3]),1)),c.correlation_seconds)
        if scale_history is not None:logs+=g*(time-scales[-1,0])
        span=float(np.exp(logs)); drift=float(np.hypot(vx,vy)); ev=float(np.hypot(ex,ey))
        # Residual + regression errors are heuristic model diagnostics, not calibrated CIs.
        out.update(growth=g,growth_se=eg,drift=drift,velocity=[vx,vy],position_sigma=max(1.,rx,ry),
                   scale_residual=rs,span=span)
        if span<c.min_span:
            out['reason']='tiny';return out
        if rs>c.max_log_residual:
            out['reason']='aspect';return out
        gl,gh=g-2.5*eg,g+2.5*eg
        dl,dh=max(0.,drift-2.5*ev),drift+2.5*ev
        if gl>c.growth_floor:
            miss=drift/(span*g)
            low=dl/(span*gh);high=dh/(span*gl)
            out.update(miss_spans=miss,miss_low=low,miss_high=high,tau=1/g)
            if high<c.threat_miss and 1/g<c.horizon:
                out.update(state='threat',reason='bearing_growth')
            elif low>c.nonthreat_miss:
                out.update(state='nonthreat',reason='crossing')
            else:out['reason']='overlap'
        elif gh<-.01:
            if c.allow_recession_clearance:out.update(state='nonthreat',reason='receding')
            else:out.update(state='uncertain',reason='shrink_ambiguous')
        elif dl/max(span,1)>1.0 and gh<.02:
            out.update(state='nonthreat',reason='crossing_no_growth')
        else:out['reason']='growth_unresolved'
        return out


def apply_camera_scope(estimate,camera,allow_similarity_pass=False):
    """A background similarity does not identify ownship rotation vs translation."""
    e=estimate.copy();mode=camera.get('mode','unknown')
    e['camera_motion_assumption']='declared_fixed_setup' if mode=='fixed' else 'background_similarity_only' if mode=='auto' else 'unknown'
    e['image_rule_candidate_state']=e['state']
    if mode=='auto' and e['state']=='nonthreat' and not allow_similarity_pass:
        e.update(state='uncertain',reason='parallax')
    return e


def advisory(objects,image_width=1280):
    """Compare image-only options but abstain from physical maneuver recommendations."""
    directions={'left':np.array([-1.,0.]),'right':np.array([1.,0.]),'up':np.array([0.,-1.]),'down':np.array([0.,1.])}
    flow=image_width/64.
    scores={};bottlenecks={}
    invalid={'aspect','tiny','camera','lost','association','history','scale_history','scale_stale','clipped','bounded_infeasible','parallax'}
    eligible=[obj for obj in objects if obj['assessment']['drift'] is not None and obj['assessment']['camera_valid']
              and obj['assessment'].get('candidate_reason',obj['assessment']['reason']) not in invalid]
    for name,d in directions.items():
        costs=[]
        for obj in eligible:
            e=obj['assessment']; vx,vy=e['velocity'];span=max(e.get('span') or 1,1)
            # One common pixel flow, scaled with raster width for resize invariance.
            # Parallel lateral translation gives common flow only at equal depths.
            # Positive score = more drift than continuing; not miss distance or safety.
            before=np.linalg.norm([vx,vy])/span
            after=np.linalg.norm(np.array([vx,vy])-d*flow)/span
            costs.append((float(after-before),obj['id']))
        scores[name]=min(costs)[0] if costs else None
        bottlenecks[name]=min(costs)[1] if costs else None
    return {'recommendation':None,'reason':'range_and_ownship_unknown','scores':scores,
            'bottleneck_track_ids':bottlenecks,'unresolved_targets':len(objects)-len(eligible),
            'assumption':'hypothetical common image flow of 1/64 raster width/s; parallel lateral translation at equal camera depths; no metric velocity or body-axis mapping inferred',
            'assumed_common_flow_px_per_s':flow,'evaluated_targets':len(eligible),
            'axes':'camera reference image axes; mapping to ownship body axes is unavailable',
            'meaning':'minimum change of normalized bearing drift across visible tracks; not safety'}
