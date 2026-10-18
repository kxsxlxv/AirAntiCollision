"""Small analytical 2D pinhole kinematics tests, NOT a flight simulator or video evidence."""
import json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from airavoid.geometry import ThreatEstimator


def run():
    rng=np.random.default_rng(20261018);counts=defaultdict(Counter);examples=[]
    for condition in ['ideal','centroid_noise','aspect_change','low_pixels','dropouts','camera_pan_unknown']:
        for i in range(200):
            # Analytic X,Z straight-line projections, unit target span, no rendering.
            # Closest approach is independently calculated in the 2D physical plane.
            z0=rng.uniform(15,80);vz=-1.;x0=rng.uniform(-8,8)
            miss=rng.uniform(-15,15);vx=(miss-x0)/z0
            initial_span=rng.uniform(8,25) if condition!='low_pixels' else rng.uniform(1,3)
            focal=z0*initial_span;ts=np.arange(0,4.001,.1);zs=z0+vz*ts;xs=x0+vx*ts
            us=640+focal*xs/zs;spans=focal/zs
            if condition=='centroid_noise':us+=rng.normal(0,1.5,len(us));spans*=np.exp(rng.normal(0,.05,len(us)))
            if condition=='aspect_change':spans*=1+.3*np.sin(ts*4)
            if condition=='camera_pan_unknown':us+=ts*12
            keep=np.ones(len(ts),bool)
            if condition=='dropouts':keep=rng.random(len(ts))>.35;keep[-1]=True
            h=[(float(t),float(u),360.,float(s)) for t,u,s,k in zip(ts,us,spans,keep) if k]
            e=ThreatEstimator().assess(h,4,condition!='camera_pan_unknown')
            r0=np.array([x0,z0]);v=np.array([vx,vz]);cpa_t=-r0@v/(v@v);cpa=float(np.linalg.norm(r0+cpa_t*v))
            # Broad separated bins exclude marginal events instead of tuning them away.
            truth='close' if cpa<1.5 else 'far' if cpa>5 else 'borderline'
            counts[condition][truth+'_'+e['state']]+=1
            if truth=='close' and e['state']=='nonthreat':examples.append({'condition':condition,'cpa_in_target_spans':cpa,'estimate':e})
    report={'seed':20261018,'trials_per_condition':200,'total':1200,'kind':'analytical 2D controlled geometry, not RGB or flight simulation',
            'truth':'true 2D constant-velocity CPA <1.5 spans close, >5 spans far, otherwise borderline',
            'counts':counts,'false_safe_examples':examples,'limitations':'plane crossing miss proxy differs from true CPA; no physical aircraft dimensions, dynamics or real-video validity inferred'}
    Path('reports/analytic_sweep.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':run()
