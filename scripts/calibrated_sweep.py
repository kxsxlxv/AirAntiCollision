"""Analytical calibration/shape sensitivity, no rendered or flight simulation."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from airavoid.calibrated import CameraCalibration,CalibratedCPAEstimator


def main():
    rows=[];counts=defaultdict(Counter);rng=np.random.default_rng(2026101804)
    for condition in ['ideal','focal_half','focal_double','span_under_half','aspect_growth','aspect_shrink','correlated_noise']:
        focal=1500.;center=20000.
        assumed_focal=focal*({'focal_half':.5,'focal_double':2}.get(condition,1.))
        k=CameraCalibration(40000,720,assumed_focal,center,360.,True,'Analytical known camera; deliberately biased in named cases')
        estimator=CalibratedCPAEstimator(k)
        for angle in [0,30,60,75,80,85]:
            for z0 in [15.,25.,40.]:
                for plane_miss in [0.,1.,4.,6.,8.,16.]:
                    x0=z0*np.tan(np.deg2rad(angle));vx=(plane_miss-x0)/z0
                    r=np.array([x0,z0]);v=np.array([vx,-1.]);tt=-r@v/(v@v)
                    true_cpa=float(np.linalg.norm(r+max(0.,tt)*v))
                    h=[];noise=0.
                    for t in np.arange(0,4.001,.1):
                        z=z0-t;x=x0+vx*t;u=center+focal*x/z;s=focal/z
                        if condition=='span_under_half':s*=.5
                        if condition=='aspect_growth':s*=np.exp(.08*t)
                        if condition=='aspect_shrink':s*=np.exp(-.08*t)
                        if condition=='correlated_noise':
                            noise=.9*noise+rng.normal(0,.4)
                            u+=noise;s*=np.exp(.03*noise)
                        h.append((float(t),float(u),360.,float(s)))
                    e=estimator.assess(h,4.)
                    truth='close' if true_cpa<1.5 else 'far' if true_cpa>5 else 'borderline'
                    counts[condition][truth+'_'+e['state']]+=1
                    rows.append({'condition':condition,'initial_bearing_deg':angle,'z0_spans':z0,'plane_miss':plane_miss,
                                 'true_cpa_spans':true_cpa,'true_time_from_window_end':float(tt-4),
                                 'estimated_cpa_spans':e['miss_spans'],'state':e['state'],'reason':e['reason'],
                                 'low':e['miss_low'],'high':e['miss_high']})
    ideal=[abs(r['estimated_cpa_spans']-r['true_cpa_spans']) for r in rows if r['condition']=='ideal' and r['estimated_cpa_spans'] is not None]
    report={'seed':2026101804,'kind':'Analytical pinhole identities and deliberate violations; not real-video accuracy or a 3D simulator',
            'trials':len(rows),'counts':counts,'ideal_max_cpa_error_spans':max(ideal),
            'false_clear_cases':[r for r in rows if r['true_cpa_spans']<1.5 and r['state']=='nonthreat'],
            'rows':rows,'limits':'Requires real camera calibration, constant transverse shape, straight relative motion and rectified fixed camera. Covariance excludes systematic calibration/shape bias. No real calibrated encounter test is claimed.'}
    Path('reports/calibrated_sweep.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','false_clear_cases']},indent=2))
    print('false clears:',Counter(r['condition'] for r in report['false_clear_cases']))


if __name__=='__main__':main()
