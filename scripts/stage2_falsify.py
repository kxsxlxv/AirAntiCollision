"""Closed-form projected trajectories and independent physical CPA, no simulator.

The state-based comparator receives explicit oracle model state only in this tier.
The vision comparators never receive relative state. Results are model falsification,
not recorded flight evidence or a prevalence estimate.
"""
from collections import Counter,defaultdict
import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from airavoid.assessment import Assessor,Context,Measurement,METHODS,scope_result
from airavoid.calibrated import CameraCalibration


def projection(r,f,cx,cy,disc=False):
    x,y,z=r
    center=np.array([cx+f*x/z,cy+f*y/z])
    if disc:
        beta=np.arctan2(x,z);half=np.arcsin(.5/np.hypot(x,z))
        left=cx+f*np.tan(beta-half);right=cx+f*np.tan(beta+half)
    else:
        left,right=center[0]-f/(2*z),center[0]+f/(2*z)
    return np.array([left,center[1]-f/(4*z),right,center[1]+f/(4*z)])


def scenario(r,v,variant,seed=1):
    rng=np.random.default_rng(seed);f=1500.;cx=20000.;cy=360.
    if variant=='cadence_25':ts=np.arange(0,4.0001,.04)
    elif variant=='cadence_5':ts=np.arange(0,4.0001,.2)
    elif variant=='unequal_spacing':ts=np.r_[0,np.sort(rng.uniform(.01,3.99,47)),4.]
    else:ts=np.arange(0,4.0001,.1)
    hist=[]
    for t in ts:
        pos=r+v*t;box=projection(pos,f,cx,cy,variant=='disc_ideal')
        center=(box[:2]+box[2:])/2;size=box[2:]-box[:2]
        if variant=='aspect_change':size*=np.exp(-.04*t)
        if variant=='detector_noise':
            center+=rng.normal(0,1.5,2);size=np.maximum(1,size+rng.normal(0,1.,2))
        if variant=='tiny_pixels':size*=.2
        if variant=='camera_rotation':center[0]+=40*t;size*=1+.002*t
        if variant=='camera_parallax':center[0]+=20*t
        if variant=='missing' and t>3.6:continue
        box=np.r_[center-size/2,center+size/2]
        hist.append(Measurement(float(t),tuple(box),1,clipped=variant=='clipping' and t>3.8))
    calibration=CameraCalibration(40000,720,f,cx,cy,True,'Analytical fixture only, not real calibration')
    context=Context(4,40000,720,'fixed',True,calibration,
                    rotation_verified=variant not in ('camera_rotation','camera_parallax'),
                    constant_shape_verified=variant not in ('aspect_change','clipping'))
    current=r+4*v
    # Four real model feature positions and instantaneous derivatives on a rigid,
    # fronto-parallel patch of common depth. These are analytic features, NOT box
    # predictions masquerading as optical flow. The flow comparator's valid input
    # model is separately declared and unavailable on our uncalibrated recordings.
    points=current[None,:]+np.array([[-.5,-.25,0],[.5,-.25,0],[.5,.25,0],[-.5,.25,0]])
    norm=points[:,:2]/points[:,2:]
    flow=(v[:2]*points[:,2:]-points[:,:2]*v[2])/points[:,2:]**2
    if variant not in ('camera_rotation','camera_parallax','missing','clipping'):
        context.feature_flow={'measured':True,'points_normalized':norm.tolist(),
                              'velocity_normalized_s':flow.tolist(),'provenance':'analytical patch feature derivatives'}
    return hist,context


def run():
    out=ROOT/'artifacts/stage2/falsification';out.mkdir(parents=True,exist_ok=True)
    estimators={m:Assessor(m) for m in METHODS}
    counts=defaultdict(Counter);errors=defaultdict(list);failures=[];examples=[]
    ncases=0
    variants=['parallel_ideal','disc_ideal','cadence_25','cadence_5','unequal_spacing',
              'detector_noise','aspect_change','missing','clipping','tiny_pixels','camera_rotation','camera_parallax']
    named=[('head_on',np.array([0.,0.,25.]),np.array([0.,0.,-1.])),
           ('crossing',np.array([0.,0.,25.]),np.array([1.,0.,-1.])),
           ('overtaking',np.array([.1,0.,15.]),np.array([0.,0.,-.3])),
           ('recession',np.array([0.,0.,25.]),np.array([0.,0.,1.])),
           ('lateral_flyby',np.array([10.,0.,25.]),np.array([-1.,0.,0.])),
           ('vertical_crossing',np.array([5.,10.,25.]),np.array([-.2,-.4,-1.]))]
    wide=[]
    for angle,z,intercept in itertools.product([0,30,60,75,80,85],[15.,25.,40.],[0.,1.,4.,6.,8.,16.]):
        x=z*np.tan(np.deg2rad(angle))
        wide.append((f'bearing_{angle}_z{z}_intercept{intercept}',np.array([x,0,z]),np.array([(intercept-x)/z,0,-1.])))
    with (out/'cases.jsonl').open('w') as stream:
        for (name,r,v),variant in itertools.product(named+wide,variants):
            hist,c=scenario(r,v,variant,ncases+42);ncases+=1
            current=r+4*v
            # Independent vector minimization, time relative to the assessment.
            tcpa=float(-current@v/(v@v));future=max(0.,tcpa)
            cpa=float(np.linalg.norm(current+future*v))
            truth='near_future' if 0<tcpa<60 and cpa<1.5 else 'far' if cpa>5 else 'other'
            rows={}
            for method,est in estimators.items():
                if method=='wcv_taumod':
                    c.relative_state={'verified':True,'provenance':'analytical oracle state, independent tier',
                                      'position_m':current.tolist(),'velocity_m_s':v.tolist(),
                                      'thresholds':{'dthr':1.5,'zthr':1.5,'tthr':35.,'tcoa':35.}}
                else:c.relative_state=None
                raw=est.assess(hist,c);scoped=scope_result(raw,c)
                key=method+'/'+variant
                counts[key][truth+'/'+raw['decision']]+=1
                if raw['tau_s'] is not None and method=='bauer_disc' and variant=='disc_ideal':
                    errors[method+'/disc_horizontal_cpa'].append(abs(raw['miss_distance']-cpa))
                    errors[method+'/disc_horizontal_tcpa'].append(abs(raw['tau_s']-tcpa))
                bad=truth=='near_future' and raw['decision']=='apparent_nonconflict'
                if bad:
                    fail={'case':name,'variant':variant,'method':method,'true_cpa_m':cpa,'true_tcpa_s':tcpa,
                          'raw_decision':raw['decision'],'scoped_decision':scoped['decision'],
                          'reason_codes':raw['reason_codes'],
                          'assumption_violations':(['disc_target_shape'] if method=='bauer_disc' and variant!='disc_ideal' else [])+
                          (['point_observer_vs_finite_nearmiss_volume'] if method=='rv_c3bf' else [])+
                          ([variant] if variant in ('camera_rotation','camera_parallax','aspect_change','detector_noise','clipping','missing','tiny_pixels') else []),
                          'metric_state_privilege':method=='wcv_taumod'}
                    failures.append(fail)
                rows[method]=raw
            stream.write(json.dumps({'case':name,'variant':variant,'physical_truth':{'cpa_m':cpa,'tcpa_s':tcpa,'class':truth},
                                     'r0_m':r.tolist(),'v_m_s':v.tolist(),'results':rows},allow_nan=False)+'\n')
    summary={'kind':'analytical closed-form projections; not recorded flight evidence; no simulator',
             'cases':ncases,'methods':list(estimators),'counts':counts,
             'critical_false_safe_count':len(failures),'critical_false_safe_by_method':dict(Counter(x['method'] for x in failures)),
             'critical_false_safe_examples':failures[:80],
             'error_diagnostics':{k:{'n':len(v),'median':float(np.median(v)),'max':float(max(v))} for k,v in errors.items()},
             'full_cases':'artifacts/stage2/falsification/cases.jsonl',
             'interpretation':'Raw candidates retained; failures are not silently removed by scope. Sensor/state/shape assumptions differ. WCV oracle-state and nominal point-observer cone are separate comparators, not equal-input RGB rankings.'}
    (ROOT/'reports/stage2_falsification.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('counts','critical_false_safe_examples')},indent=2))


if __name__=='__main__':run()
