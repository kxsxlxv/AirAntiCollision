"""Set-membership projective sensitivity under explicitly assumed error bounds.

This encloses plane-intercept geometry, not metric CPA or collision probability.
Unknown aspect/camera bias may violate the assumptions; no real-world error bound
has been calibrated. Bounds never shrink merely because frame rate increases.
"""
import numpy as np
from .geometry import ThreatEstimator


def bounded_line(t,lower,upper):
    t=np.asarray(t,float);lo=np.asarray(lower,float);hi=np.asarray(upper,float)
    dt=t[None,:]-t[:,None];valid=dt>1e-9
    if not np.any(valid):return None
    bl=float(np.max((lo[None,:]-hi[:,None])[valid]/dt[valid]))
    bh=float(np.min((hi[None,:]-lo[:,None])[valid]/dt[valid]))
    if bl>bh+1e-10:return None
    b=(bl+bh)/2;relative=t-t[-1]
    al=float(np.max(lo-b*relative));ah=float(np.min(hi-b*relative))
    if al>ah+1e-10:return None
    return {'slope_low':bl,'slope_high':bh,'slope':b,'current':(al+ah)/2}


def product_interval(a,b):
    products=[x*y for x in a for y in b]
    return min(products),max(products)


class BoundedProjectiveEstimator(ThreatEstimator):
    def assess(self,history,time,camera_valid=True,ambiguous=False):
        base=super().assess(history,time,camera_valid,ambiguous)
        c=self.config;recent=[h for h in history if time-h[0]<=c.window]
        base.update(method='bounded_projective',assumed_centroid_error_px=c.bounded_pixel_error,
                    assumed_span_relative_error=c.bounded_span_fraction)
        if not recent or not camera_valid or ambiguous:return base
        a=np.asarray(recent,float)
        if time-a[-1,0]>.25 or len(a)<c.min_samples or a[-1,0]-a[0,0]<c.min_history:return base
        out={**base,'state':'uncertain','reason':'bounded_infeasible','growth':None,'growth_se':None,
             'growth_low':None,'growth_high':None,'drift':None,'miss_spans':None,'miss_low':None,
             'miss_high':None,'tau':None,'velocity':[0.,0.],'position_sigma':c.bounded_pixel_error}
        spans=np.maximum(a[:,3],1.)
        if np.median(spans)<c.min_span:out['reason']='tiny';return out
        sl=spans*(1-c.bounded_span_fraction);sh=spans*(1+c.bounded_span_fraction)
        ql,qh=1/sh,1/sl
        origin=a[-1,1:3];ranges=[]
        for axis in range(2):
            v=a[:,1+axis]-origin[axis]
            candidates=np.stack([(v-e)/s for e in [-c.bounded_pixel_error,c.bounded_pixel_error] for s in [sl,sh]])
            ranges.append((candidates.min(axis=0),candidates.max(axis=0)))
        q=bounded_line(a[:,0],ql,qh)
        px=bounded_line(a[:,0],*ranges[0]);py=bounded_line(a[:,0],*ranges[1])
        if any(f is None for f in [q,px,py]):return out
        aq=q['current'];bq=q['slope'];span=1/aq;g=-bq/aq
        velocity=np.array([(p['slope']*aq-p['current']*bq)/aq**2 for p in [px,py]])
        out.update(span=float(span),growth=float(g),drift=float(np.linalg.norm(velocity)),velocity=velocity.tolist(),reason='growth_unresolved')
        if q['slope_high']>=0:return out
        age=max(0.,time-a[-1,0])
        tl=ql[-1]/(-q['slope_low']);th=qh[-1]/(-q['slope_high'])
        tl-=age;th-=age
        if tl<=0:return out
        gl,gh=1/th,1/tl
        out.update(growth_low=float(gl),growth_high=float(gh),tau_low=float(tl),tau_high=float(th))
        intervals=[]
        for p,bounds in zip([px,py],ranges):
            pr=product_interval((p['slope_low'],p['slope_high']),(tl+age,th+age))
            intervals.append((float(bounds[0][-1]+pr[0]),float(bounds[1][-1]+pr[1])))
        mins=[0. if lo<=0<=hi else min(abs(lo),abs(hi)) for lo,hi in intervals]
        maxs=[max(abs(lo),abs(hi)) for lo,hi in intervals]
        ml,mh=float(np.linalg.norm(mins)),float(np.linalg.norm(maxs))
        tau=-aq/bq-age;point=np.array([p['current']+p['slope']*(tau+age) for p in [px,py]])
        out.update(miss_spans=float(np.linalg.norm(point)),miss_low=ml,miss_high=mh,tau=float(tau),reason='bounded_ambiguous')
        if gl>c.growth_floor:
            if mh<c.threat_miss and th<c.horizon:out.update(state='threat',reason='bearing_growth')
            elif ml>c.nonthreat_miss:out.update(state='nonthreat',reason='crossing')
        return out
