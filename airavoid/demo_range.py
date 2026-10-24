"""Opt-in NPS demonstration ranging from declared aircraft size and lens preset.

This is an approximate equidistant-fisheye mapping, not camera calibration.
Range never enters encounter classification, tracking or separation selection.
"""
from collections import defaultdict,deque
import math
import numpy as np


class NPSRangeDemo:
    def __init__(self,span_m=1.5,hfov_deg=120.):
        if not math.isfinite(span_m) or span_m<=0 or not math.isfinite(hfov_deg) or not 1<hfov_deg<179:
            raise ValueError('Positive aircraft span and horizontal FOV between 1 and 179 degrees required')
        self.span_m=float(span_m);self.hfov_deg=float(hfov_deg);self.history=defaultdict(deque)
        self.contract={'dataset':'NPS, explicitly declared by the user flag','camera':'GoPro HERO3 Ultra Wide, user confirmed',
            'aircraft_span_m':self.span_m,'size_source':'User supplied 1.5 m aircraft span; transverse full-span approximation',
            'horizontal_fov_deg_assumed':self.hfov_deg,'projection':'approximate equidistant fisheye, square angular pixel scale',
            'calibrated':False,'range_meaning':'approximate line-of-sight distance under transverse target-size assumption',
            'formula':'D = span_m / (2*tan(angular_extent/2)); median of fresh last 0.4 s',
            'limitations':'Unknown aspect/foreshortening, bbox extent and lens errors; no certified bounds',
            'used_for_collision_or_direction':False}

    def _ray(self,x,y,width,height):
        f=width/math.radians(self.hfov_deg)
        p=np.array([(x-width/2)/f,(y-height/2)/f]);theta=float(np.linalg.norm(p))
        return np.array([0.,0.,1.]) if theta<1e-12 else np.r_[p*math.sin(theta)/theta,math.cos(theta)]

    def update(self,demo,objects,time,image_size):
        width,height=image_size;lookup={o['id']:o for o in objects};seen=set()
        demo['range_preset']=self.contract
        for target in demo['targets']:
            tid=target['id'];seen.add(tid);obj=lookup[tid];guard=obj.get('demo_observation',{})
            target.update(range_approx_m=None,range_display_m=None,range_reason='no_fresh_reliable_extent')
            b=obj['box'];x1,y1,x2,y2=b;w=x2-x1;h=y2-y1
            if not guard.get('measured') or guard.get('clipped') or guard.get('ambiguous') or min(w,h)<=0 or max(w,h)<6:
                self.history.pop(tid,None);continue
            # Use the longer measured bbox axis as a projected wingspan proxy.
            a=(x1,(y1+y2)/2) if w>=h else ((x1+x2)/2,y1)
            z=(x2,(y1+y2)/2) if w>=h else ((x1+x2)/2,y2)
            angle=math.acos(float(np.clip(self._ray(*a,width,height)@self._ray(*z,width,height),-1,1)))
            if not 1e-6<angle<math.pi/2:continue
            distance=self.span_m/(2*math.tan(angle/2))
            history=self.history[tid]
            if history and time-history[-1][0]>.25:history.clear()
            history.append((time,distance))
            while history and time-history[0][0]>.4:history.popleft()
            estimate=float(np.median([r for _,r in history]));step=5 if estimate>=20 else 1
            target.update(range_approx_m=estimate,range_display_m=max(step,round(estimate/step)*step),
                range_reason='assumed_size_and_uncalibrated_ultra_wide_projection',range_support_samples=len(history))
        for tid in set(self.history)-seen:del self.history[tid]
        return demo
