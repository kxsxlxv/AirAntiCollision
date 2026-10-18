"""Analytical pinhole counterexamples to interpreting background flow as rotation.

No images, flight dynamics, rendering or 3D simulator. Distances are arbitrary
constant target-span units. Ideal background similarities are supplied explicitly.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from airavoid.geometry import Config,ThreatEstimator,apply_camera_scope
from airavoid.projective import ProjectiveEstimator
from airavoid.bounded import BoundedProjectiveEstimator

rows=[];counts=defaultdict(Counter);times=np.arange(0,4.001,.1)
for z0 in [30.,60.,120.]:
    focal=z0*20
    for collision_time in [12.,20.,40.]:
        for lateral in [.2,1.,3.]:
            for forward_fraction in [0.,.5,1.]:
                forward=forward_fraction*z0/collision_time
                for ratio in [.5,1.,3.,100.]:
                    depth=ratio*z0
                    raw=[];stable=[];transforms=[]
                    for t in times:
                        z=z0*(1-t/collision_time);span=focal/z
                        raw.append((float(t),640.,360.,float(span)))
                        # Static planar background viewed by a translating observer:
                        # B_t = scale*B_0 - f*Ux*t/(Zb-Uz*t), about principal point.
                        scale=depth/(depth-forward*t);shift=-focal*lateral*t/(depth-forward*t)
                        transform=np.array([[scale,0.,640*(1-scale)+shift],[0.,scale,360*(1-scale)],[0.,0.,1.]])
                        transforms.append(transform)
                        # Applying its exact inverse to a true constant-bearing target.
                        stable.append((float(t),float(640+focal*lateral*t/depth),360.,float(span/scale)))
                    max_scale_step=0.;max_corner_motion=0.
                    corners=np.array([[160,120,1],[1120,120,1],[160,600,1],[1120,600,1]],float)
                    for first,second in zip(transforms,transforms[1:]):
                        # Our observations are0.1s apart; actual camera thresholds
                        # apply per processed frame at0.1s in this controlled probe.
                        delta=second@np.linalg.inv(first)
                        max_scale_step=max(max_scale_step,abs(delta[0,0]-1))
                        moved=(delta@corners.T).T[:,:2]-corners[:,:2]
                        max_corner_motion=max(max_corner_motion,float(np.max(np.linalg.norm(moved,axis=1))*.5))
                    accepted=bool(max_scale_step<.015 and max_corner_motion<8)
                    for name,estimator in [('log_hac1',ThreatEstimator(Config(correlation_seconds=1.))),
                                           ('projective',ProjectiveEstimator()),('bounded',BoundedProjectiveEstimator())]:
                        a=estimator.assess(raw,4.);b=estimator.assess(stable,4.)
                        scoped=apply_camera_scope(b,{'mode':'auto','valid':True})
                        key=name+('_within_similarity_gates' if accepted else '_outside_similarity_gates')
                        counts[key]['raw_'+a['state']]+=1;counts[key]['stabilized_'+b['state']]+=1
                        counts[key]['conservative_scope_'+scoped['state']]+=1
                        rows.append({'method':name,'z0_spans':z0,'collision_time_s':collision_time,'ownship_lateral_spans_s':lateral,
                                     'ownship_forward_spans_s':forward,'background_depth_spans':depth,'true_cpa_spans':0.,
                                     'raw_state':a['state'],'stabilized_state':b['state'],'stabilized_reason':b['reason'],
                                     'conservative_scope_state':scoped['state'],
                                     'stabilized_miss_low':b['miss_low'],'stabilized_miss_high':b['miss_high'],
                                     'within_configured_small_similarity_motion_gates':accepted,'max_640_corner_step_px':max_corner_motion,
                                     'max_similarity_scale_step':max_scale_step})
report={'kind':'Analytical 2D pinhole projections with finite-depth planar background; not flight simulation or RGB performance',
        'geometry':'Target hasX=0 andZ=z0*(1-t/T), so true collision atT; observer translates, background is static. Perfect inverse background similarity removes some genuine relative-motion evidence.',
        'counts':counts,'rows':rows,'false_clear_within_similarity_gates':[r for r in rows if r['within_configured_small_similarity_motion_gates'] and r['stabilized_state']=='nonthreat'],
        'limits':'Gate check assumes enough correctly matched planar background features; it does not measure actual KLT detection/inliers. Cases are adversarial analytical examples, not prevalence estimates.'}
Path('reports/parallax_limits.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'counts':counts,'false_clear_cases_within_similarity_gates':len(report['false_clear_within_similarity_gates'])},indent=2))
