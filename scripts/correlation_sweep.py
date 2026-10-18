"""Fixed-seed correlated measurement stress; not flight or video evidence."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from airavoid.geometry import Config,ThreatEstimator,robust_line

rng=np.random.default_rng(2026101802);bandwidths=[0.,.5,1.,2.];records=[]
times=np.arange(0,4.001,.08)
def ar_noise(rho,sigma):
    out=np.empty(len(times));out[0]=rng.normal(0,sigma)
    for j in range(1,len(out)):out[j]=rho*out[j-1]+rng.normal(0,sigma*np.sqrt(1-rho**2))
    return out

for rho in [0.,.8,.95]:
    data=[]
    for _ in range(500):data.append(100+3*times+ar_noise(rho,1.5))
    for bw in bandwidths:
        covered=0;errors=[];estimates=[]
        for y in data:
            a,b,se,res=robust_line(times,y,1.,bw);covered+=abs(b-3)<=2.5*se
            errors.append(abs(b-3));estimates.append(se)
        records.append({'kind':'exact linear slope','rho':rho,'bandwidth_s':bw,'trials':500,'covered_2_5se':int(covered),'coverage':covered/500,'median_abs_slope_error':float(np.median(errors)),'median_reported_se':float(np.median(estimates))})

geometry=[]
for rho in [0.,.8,.95]:
    cases=[]
    for i in range(200):
        z0=rng.uniform(20,60);miss=rng.uniform(0,1.) if i<100 else rng.uniform(7,12)
        x0=rng.uniform(-3,3);vx=(miss-x0)/z0;focal=rng.uniform(25,60)*z0
        z=z0-times;u=640+focal*(x0+vx*times)/z+ar_noise(rho,1.5)
        span=focal/z*np.exp(ar_noise(rho,.08))
        h=[(float(t),float(xx),360.,float(ss)) for t,xx,ss in zip(times,u,span)]
        cases.append(('close' if i<100 else 'far',miss,h))
    for bw in bandwidths:
        counts=Counter();intervals=0;covered=0
        estimator=ThreatEstimator(Config(correlation_seconds=bw))
        for truth,miss,h in cases:
            e=estimator.assess(h,float(times[-1]));counts[truth+'_'+e['state']]+=1
            if e['miss_low'] is not None:
                intervals+=1;covered+=e['miss_low']<=miss<=e['miss_high']
        geometry.append({'rho':rho,'bandwidth_s':bw,'trials':200,'counts':counts,'available_plane_miss_intervals':intervals,'intervals_covering_true_plane_miss':int(covered)})
report={'seed':2026101802,'meaning':'coverage is of known analytical slope/plane intercept under specified synthetic measurement noise, not collision probability calibration','linear_trials':records,'geometry_trials':geometry,'limits':'one short window cannot identify arbitrary systematic aspect or camera bias; HAC is a sensitivity correction only'}
Path('reports/correlation_sweep.json').write_text(json.dumps(report,indent=2))
for r in records:print(r)
for r in geometry:print(r)
