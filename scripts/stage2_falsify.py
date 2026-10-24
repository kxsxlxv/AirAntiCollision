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
from airavoid.provenance import file_sha256


def projection(r,f,cx,cy,disc=False,ellipse=False):
    x,y,z=r
    center=np.array([cx+f*x/z,cy+f*y/z])
    if disc:
        beta=np.arctan2(x,z);half=np.arcsin(.5/np.hypot(x,z))
        left=cx+f*np.tan(beta-half);right=cx+f*np.tan(beta+half)
    else:
        left,right=center[0]-f/(2*z),center[0]+f/(2*z)
    if ellipse:
        # Independent analytic tangency of (Y-y)^2/.125^2+(Z-z)^2/.5^2=1
        # to rays Y=mZ. R=1 and H=.25 are fixture truth, never assessor input.
        a=.125**2;b=.5**2
        slope=(y*z+np.array([-1.,1.])*np.sqrt(a*z*z+b*y*y-a*b))/(z*z-b)
        bottom,top=cy+f*slope
    else:bottom,top=center[1]-f/(4*z),center[1]+f/(4*z)
    return np.array([left,bottom,right,top])


def scenario(r,v,variant,seed=1):
    rng=np.random.default_rng(seed);f=1500.;cx=20000.;cy=20000.
    if variant=='cadence_25':ts=np.arange(0,4.0001,.04)
    elif variant=='cadence_5':ts=np.arange(0,4.0001,.2)
    elif variant=='unequal_spacing':ts=np.r_[0,np.sort(rng.uniform(.01,3.99,47)),4.]
    else:ts=np.arange(0,4.0001,.1)
    hist=[]
    for t in ts:
        pos=r+v*t;box=projection(pos,f,cx,cy,variant in ('disc_ideal','disc_ellipse_3d'),variant=='disc_ellipse_3d')
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
    calibration=CameraCalibration(40000,40000,f,cx,cy,True,'Analytical fixture only, not real calibration')
    camera_mode='moving' if variant in ('camera_rotation','camera_parallax') else 'fixed'
    context=Context(4,40000,40000,camera_mode,True,calibration,
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
    counts=defaultdict(Counter);scoped_counts=defaultdict(Counter);errors=defaultdict(list);failures=[]
    ncases=0
    variants=['parallel_ideal','disc_ideal','disc_ellipse_3d','cadence_25','cadence_5','unequal_spacing',
              'detector_noise','aspect_change','missing','clipping','tiny_pixels','camera_rotation','camera_parallax']
    named=[('head_on',np.array([0.,0.,25.]),np.array([0.,0.,-1.])),
           ('crossing',np.array([0.,0.,25.]),np.array([1.,0.,-1.])),
           ('overtaking',np.array([.1,0.,15.]),np.array([0.,0.,-.3])),
           ('recession',np.array([0.,0.,25.]),np.array([0.,0.,1.])),
           ('lateral_flyby',np.array([10.,0.,25.]),np.array([-1.,0.,0.])),
           ('vertical_crossing',np.array([5.,10.,25.]),np.array([-.2,-.4,-1.]))]
    for initial_y,intercept_y in itertools.product([-10.,0.,10.],[0.,4.,12.]):
        named.append((f'vertical_y{initial_y}_intercept{intercept_y}',np.array([0.,initial_y,25.]),
                      np.array([0.,(intercept_y-initial_y)/25.,-1.])))
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
                scoped_counts[key][truth+'/'+scoped['decision']]+=1
                if raw['tau_s'] is not None and method=='bauer_disc' and variant in ('disc_ideal','disc_ellipse_3d'):
                    errors[method+'/'+variant+'_3d_cpa'].append(abs(raw['miss_distance']-cpa))
                    errors[method+'/'+variant+'_3d_tcpa'].append(abs(raw['tau_s']-tcpa))
                    cr=current[[0,2]];cv=v[[0,2]]
                    ht=float(-cr@cv/(cv@cv));hc=float(np.linalg.norm(cr+max(0.,ht)*cv))
                    errors[method+'/'+variant+'_horizontal_cpa'].append(abs(raw['horizontal_miss_distance']-hc))
                    errors[method+'/'+variant+'_horizontal_tcpa'].append(abs(raw['horizontal_tau_s']-ht))
                bad=truth=='near_future' and raw['decision']=='apparent_nonconflict'
                if bad:
                    fail={'case':name,'variant':variant,'method':method,'true_cpa_m':cpa,'true_tcpa_s':tcpa,
                          'raw_decision':raw['decision'],'scoped_decision':scoped['decision'],
                          'reason_codes':raw['reason_codes'],
                          'assumption_violations':(['disc_target_shape'] if method=='bauer_disc' and variant not in ('disc_ideal','disc_ellipse_3d') else [])+
                          (['point_observer_vs_finite_nearmiss_volume'] if method=='rv_c3bf' else [])+
                          ([variant] if variant in ('camera_rotation','camera_parallax','aspect_change','detector_noise','clipping','missing','tiny_pixels') else []),
                          'metric_state_privilege':method=='wcv_taumod'}
                    failures.append(fail)
                rows[method]={**raw,'scoped_decision':scoped['decision'],'scoped_reason_codes':scoped['reason_codes']}
            stream.write(json.dumps({'case':name,'variant':variant,'physical_truth':{'cpa_m':cpa,'tcpa_s':tcpa,'class':truth},
                                     'r0_m':r.tolist(),'v_m_s':v.tolist(),'results':rows},allow_nan=False)+'\n')
    (out/'false_safe_cases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in failures))
    summary={'kind':'analytical closed-form projections; not recorded flight evidence; no simulator',
             'cases':ncases,'base_trajectories':len(named)+len(wide),'variants':variants,
             'methods':list(estimators),'counts':counts,'scoped_counts':scoped_counts,
             'critical_false_safe_count':len(failures),'critical_false_safe_by_method':dict(Counter(x['method'] for x in failures)),
             'scoped_false_safe_count':sum(x['scoped_decision']=='apparent_nonconflict' for x in failures),
             'scoped_false_safe_by_method':dict(Counter(x['method'] for x in failures if x['scoped_decision']=='apparent_nonconflict')),
             'critical_false_safe_examples':failures[:80],
             'error_diagnostics':{k:{'n':len(v),'median':float(np.median(v)),'max':float(max(v))} for k,v in errors.items()},
             'full_cases':'artifacts/stage2/falsification/cases.jsonl',
             'all_false_safe_cases':'artifacts/stage2/falsification/false_safe_cases.jsonl',
             'fixture_units':'R=1 metre is analytical truth only; vision receives no physical size. CPA error numbers in this fixture coincide with horizontal-diameter units.',
             'implementation_sha256':file_sha256(ROOT/'airavoid/assessment.py'),
             'falsification_script_sha256':file_sha256(Path(__file__)),
             'corrections_since_initial':['Full Bauer virtual-disc 3D reconstruction replaces horizontal-only output; horizontal diagnostics retained',
                                          'Added exact independent vertical ellipse projection and nine vertical-trajectory sweeps',
                                          'Enlarged analytic image canvas vertically to contain targets; no synthetic clipping mislabeled as measured',
                                          'Rotation/parallax variants now carry moving-camera metadata; raw results remain retained',
                                          'Horizontal errors compared with independently minimized horizontal range, rather than 3D range'],
             'interpretation':'Raw candidates retained; failures are not silently removed by scope. Sensor/state/shape assumptions differ. WCV oracle-state and nominal point-observer cone are separate comparators, not equal-input RGB rankings.'}
    (ROOT/'reports/stage2_falsification.json').write_text(json.dumps(summary,indent=2))
    aggregate={method:Counter() for method in estimators}
    for key,values in counts.items():
        method=key.split('/')[0]
        for label,number in values.items():aggregate[method][label.split('/')[1]]+=number
    text=['# Stage 2 analytical falsification, 2026-10-22','',
          f'{ncases:,} closed-form projected cases: {len(named)+len(wide)} constant-velocity trajectories × {len(variants)} observation variants. '
          'These are analytical fixtures, not recorded flight evidence or a 3D flight simulator. '
          'Physical CPA is independently minimized from the supplied Cartesian trajectory. '
          'A near encounter means future TCPA inside 60 s and Euclidean CPA <1.5 m; R=1 m only in fixture truth. '
          'Every vision method sees the same boxes, timestamps and explicitly analytical camera intrinsics, without the physical R or state.','',
          '| Method | Potential conflict | Apparent nonconflict | Uncertain | Unobserved | Raw false-safe | After camera scope |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for method,values in aggregate.items():
        text.append('| '+method+' | '+' | '.join(str(values[s]) for s in ['potential_conflict','apparent_nonconflict','uncertain','unobserved'])+
                    f" | {summary['critical_false_safe_by_method'].get(method,0)} | {summary['scoped_false_safe_by_method'].get(method,0)} |")
    text+=['','The columns are diagnostic counts, not an equal-information accuracy ranking: WCV receives verified oracle metric state; '
           'RV-C3BF here is only a nominal horizontal point-observer cone using analytical calibrated patch flow; AirTrack supplies a cue without an invented classifier. '
           'Missing or clipped measurements remain unobserved/uncertain even in the state comparator tier. '
           'The shared image-method thresholds (2/5 diameter units and 60 s) are research thresholds, not transferred flight decision thresholds.','',
           'Bauer includes the published approximate virtual vertical disc (JIRS Eqs. 9–14), reciprocal-size average and full 3D line minimum. '
           'The independent disc/ellipse variants each contain 64 physical near encounters; Bauer has zero false-safe there. '
           'Its 123-case disc/ellipse absolute CPA errors have median '+f"{np.median(errors['bauer_disc/disc_ellipse_3d_3d_cpa']):.5f}"+
           ' diameter and maximum '+f"{max(errors['bauer_disc/disc_ellipse_3d_3d_cpa']):.5f}"+' diameter; TCPA median '+
           f"{np.median(errors['bauer_disc/disc_ellipse_3d_3d_tcpa']):.5f}"+' s and maximum '+f"{max(errors['bauer_disc/disc_ellipse_3d_3d_tcpa']):.5f}"+' s. '
           'The reported finite residual is retained; the virtual disc is an approximation. Horizontal-only diagnostics remain available alongside full 3D results.','',
           'Bauer still fails under parallel-segment shape mismatch, altered scale/aspect, detector noise and unresolved moving-camera perturbations. '
           'The point-observer cone does not represent a finite near-miss volume or include the paper’s robust observer/certified uncertainty bound. '
           'A zero-failure state-oracle or ideal-shape result does not establish safety on uncalibrated moving-camera RGB. '
           'No assumptions were removed from the raw failure accounting.','',
           'All cases, estimates and raw/scoped decisions: `artifacts/stage2/falsification/cases.jsonl`. '
           'All raw false-safe cases and assumption violations: `artifacts/stage2/falsification/false_safe_cases.jsonl`. '
           'Aggregate machine-readable counts and implementation/script SHA-256 digests: `reports/stage2_falsification.json`. '
           'The earlier 1,368-case horizontal-only run remains in `reports/stage2_falsification_initial.json`; '
           'the new suite adds an exact vertical-ellipse projection and vertical trajectories, enlarges the analytical canvas and fixes moving-camera metadata.','',
           'Reproduce: `.venv/Scripts/python.exe scripts/stage2_falsify.py`.']
    (ROOT/'reports/stage2_falsification.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('counts','scoped_counts','critical_false_safe_examples')},indent=2))


if __name__=='__main__':run()
