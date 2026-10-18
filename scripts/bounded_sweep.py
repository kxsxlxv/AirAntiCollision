"""Enclosure checks under declared bounded errors, including correlated noise."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from airavoid.bounded import bounded_line,BoundedProjectiveEstimator
from airavoid.geometry import Config

rng=np.random.default_rng(2026101803);times=np.arange(0,4.001,.08);results=[]
def ar(rho):
    x=np.empty(len(times));x[0]=rng.normal()
    for j in range(1,len(x)):x[j]=rho*x[j-1]+rng.normal()*np.sqrt(1-rho*rho)
    return x

for rho in [0.,.8,.95]:
    enclosed=0
    for _ in range(500):
        error=np.clip(ar(rho)*1.5,-2,2);y=100+3*times+error
        fit=bounded_line(times,y-2,y+2)
        enclosed+=fit is not None and fit['slope_low']-1e-8<=3<=fit['slope_high']+1e-8
    results.append({'kind':'exact linear bounded measurement','rho':rho,'trials':500,'true_slope_enclosed':int(enclosed),'assumption':'every absolute measurement error <=2; generated clipped AR noise, not unrestricted Gaussian'})
geometry=[]
for fraction in [.03,.05,.1]:
    for rho in [0.,.8,.95]:
        counts=Counter();available=0;covered=0;infeasible=0
        estimator=BoundedProjectiveEstimator(Config(bounded_span_fraction=fraction))
        for i in range(200):
            z0=rng.uniform(10,30);miss=rng.uniform(0,1.) if i<100 else rng.uniform(7,12)
            x0=rng.uniform(-2,2);vx=(miss-x0)/z0;focal=z0*rng.uniform(30,80)
            z=z0-times;true_u=640+focal*(x0+vx*times)/z;true_span=focal/z
            measured_u=true_u+np.clip(ar(rho)*1.5,-2,2)
            # true_span lies in measured_span*(1±fraction), by construction.
            measured_span=true_span/(1+np.clip(ar(rho)*fraction/2,-fraction,fraction))
            h=[(float(t),float(u),360.,float(s)) for t,u,s in zip(times,measured_u,measured_span)]
            e=estimator.assess(h,4);truth='close' if i<100 else 'far';counts[truth+'_'+e['state']]+=1
            infeasible+=e['reason']=='bounded_infeasible'
            if e['miss_low'] is not None:
                available+=1;covered+=e['miss_low']-1e-7<=miss<=e['miss_high']+1e-7
        geometry.append({'span_fraction':fraction,'rho':rho,'trials':200,'counts':counts,'infeasible':infeasible,'available_plane_intervals':available,'true_plane_miss_enclosed':int(covered)})
report={'seed':2026101803,'kind':'analytical 2D pinhole and bounded-error stress; no flight simulation','linear_results':results,'geometry_results':geometry,'limits':'Assumed bounds have not been calibrated on actual detector errors; model violations and true CPA ambiguity remain. Conditional enclosure is not a real-world accuracy guarantee.'}
Path('reports/bounded_sweep.json').write_text(json.dumps(report,indent=2))
for r in results:print(r)
for r in geometry:print(r)
