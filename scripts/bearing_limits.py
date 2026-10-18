"""Analytical wide-bearing falsification: image-plane intercept is not true CPA."""
from collections import defaultdict,Counter
import json
from pathlib import Path
import numpy as np
from airavoid.geometry import ThreatEstimator
from airavoid.projective import ProjectiveEstimator

rows=[];counts=defaultdict(Counter)
for angle in [0,30,60,75,80,85]:
    for z0 in [15.,25.,40.]:
        for plane_miss in [0.,1.,4.,6.,8.,16.]:
            x0=z0*np.tan(np.deg2rad(angle));vz=-1.;vx=(plane_miss-x0)/z0
            focal=1500.;center=20000.;width=40000
            # Large analytical raster width keeps the wide-bearing objects visible
            # and resolved, so lack of pixels does not hide the geometric limitation.
            history=[]
            for t in np.arange(0,4.001,.1):
                z=z0-t;x=x0+vx*t
                history.append((float(t),float(center+focal*x/z),360.,float(focal/z)))
            r=np.array([x0,z0]);v=np.array([vx,vz]);tt=-r@v/(v@v);cpa=float(np.linalg.norm(r+tt*v))
            for name,estimator in [('log',ThreatEstimator()),('projective',ProjectiveEstimator())]:
                e=estimator.assess(history,4)
                truth='close' if cpa<1.5 else 'far' if cpa>5 else 'borderline'
                counts[f'{name}_{angle}deg'][truth+'_'+e['state']]+=1
                rows.append({'method':name,'initial_bearing_deg':angle,'z0_in_spans':z0,'plane_intercept_spans':plane_miss,'true_2d_cpa_spans':cpa,'true_2d_cpa_time_s':float(tt),'state':e['state'],'estimated_plane_miss':e['miss_spans']})
report={'kind':'analytical 2D pinhole counterexample; no images or flight simulation','counts':counts,'rows':rows,
        'false_safe_examples':[r for r in rows if r['true_2d_cpa_spans']<1.5 and r['state']=='nonthreat'],
        'conclusion':'Depth-plane intercept can overestimate true closest approach at wide bearings / large lateral velocity. Unknown intrinsics prevent conversion to guaranteed CPA. Green remains conditional image-model evidence.'}
Path('reports/bearing_limits.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
