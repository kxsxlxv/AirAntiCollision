"""Actual KLT/RANSAC on a controlled 2D texture, with analytical target boxes.

This is neither a neural-detection trial nor real encounter imagery. It tests the
camera/geometry interface without constructing or operating a 3D simulator.
"""
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np
from airavoid.tracking import CameraMotion,Tracker
from airavoid.geometry import ThreatEstimator,Config,apply_camera_scope
from airavoid.policy import TemporalPolicy


def run_probe():
    rng=np.random.default_rng(81821)
    texture=rng.integers(0,256,(720,1280,3),dtype=np.uint8)
    texture=cv2.GaussianBlur(texture,(5,5),0)
    camera=CameraMotion('auto');tracker=Tracker(image_edge_margin=1)
    estimator=ThreatEstimator(Config(correlation_seconds=1.));old=TemporalPolicy();guarded=TemporalPolicy()
    records=[];raw_history=[]
    for t in np.arange(0,5.001,.08):
        t=float(t);span=1000/(50*(1-t/20))
        # A translating observer viewing a static textured plane atZb=50.
        # The intruder hasX=0,Z=50*(1-t/20), a true collision atT=20.
        frame=cv2.warpAffine(texture,np.float32([[1,0,-20*t],[0,1,0]]),(1280,720),borderMode=cv2.BORDER_REFLECT)
        box=[640-span/2,360-span/6,640+span/2,360+span/6,1.,0]
        transform,cam=camera.update(frame,t,[box])
        if cam['evidence_reset']:tracker.reset_evidence();old.pending.clear();guarded.pending.clear()
        tr=tracker.update([box],t,transform,(1280,720))[0]
        e=estimator.assess(tr.history,t,cam['valid'])
        old_e=old.apply(tr.id,t,apply_camera_scope(e,cam,True))
        new_e=guarded.apply(tr.id,t,apply_camera_scope(e,cam,False))
        raw_history.append((t,640.,360.,span))
        records.append({'time':t,'camera_valid':cam['valid'],'background_inliers':cam['inliers'],
                        'flow_at640':cam['flow_px_at_640'],'image_candidate':e['state'],
                        'old_confirmed_state':old_e['state'],'guarded_confirmed_state':new_e['state'],
                        'miss_low':e['miss_low'],'miss_high':e['miss_high']})
    return {'kind':'Actual OpenCV KLT/RANSAC on a synthetic2D planar texture plus analytical target boxes; no detector/real-video claim',
            'seed':81821,'samples':len(records),'true_analytical_cpa_spans':0.,'true_collision_time_s':20.,
            'valid_background_fits':sum(r['camera_valid'] for r in records),
            'old_confirmed_states':dict(Counter(r['old_confirmed_state'] for r in records)),
            'guarded_confirmed_states':dict(Counter(r['guarded_confirmed_state'] for r in records)),
            'raw_unstabilized_final_state':estimator.assess(raw_history,records[-1]['time'])['state'],
            'records':records,'limits':'Adversarial controlled observability test, not physical flight prevalence or a calibrated safety validation. Withholding passes fixes this indication by abstention, not by recovering missing motion.'}


if __name__=='__main__':
    report=run_probe();Path('reports/camera_parallax_probe.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
