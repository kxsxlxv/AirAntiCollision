"""Shared measured-observation contract and independent research comparators.

Equations are independently implemented, not copied author code. See
research/stage2_selection.md and reports/stage2_method_provenance.json.
No calibration, telemetry, confidence probability or target dimension is invented.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .geometry import ThreatEstimator,Config
from .projective import ProjectiveEstimator
from .bounded import BoundedProjectiveEstimator
from .calibrated import CalibratedCPAEstimator,CameraCalibration


STATES={'threat':'potential_conflict','nonthreat':'apparent_nonconflict',
        'uncertain':'uncertain','unobserved':'unobserved'}
LEGACY={v:k for k,v in STATES.items()}
METHODS=('bauer_disc','lai_lin','airtrack_cues','wcv_taumod','rv_c3bf',
         'log','projective','bounded','calibrated','growth')


@dataclass(frozen=True)
class Measurement:
    time: float
    box: tuple
    track_id: int = 0
    confidence: Optional[float] = None
    measured: bool = True
    source: str = 'detector'
    clipped: bool = False
    ambiguous: bool = False
    # Raw extent remains a measurement even if a separate position is stabilized.
    position: Optional[tuple] = None
    extent_scale: float = 1.

    def __post_init__(self):
        if len(self.box)!=4 or not np.isfinite([self.time,*self.box]).all():
            raise ValueError('Finite XYXY measurements required')
        if self.box[2]<=self.box[0] or self.box[3]<=self.box[1]:
            raise ValueError('Positive measured extents required')
        if not np.isfinite(self.extent_scale) or self.extent_scale<=0:
            raise ValueError('Positive finite extent correction required')

    @property
    def center(self):
        return self.position or ((self.box[0]+self.box[2])/2,(self.box[1]+self.box[3])/2)

    @property
    def span(self):
        return max(self.box[2]-self.box[0],self.box[3]-self.box[1])*self.extent_scale

    @property
    def area(self):
        return (self.box[2]-self.box[0])*(self.box[3]-self.box[1])*self.extent_scale**2


@dataclass
class Context:
    time: float
    width: int
    height: int
    camera_mode: str = 'unknown'
    camera_valid: bool = False
    calibration: Optional[CameraCalibration] = None
    rotation_verified: bool = False
    relative_state: Optional[dict] = None
    # Actual calibrated, rotation-compensated feature flow only; never GT on RGB.
    feature_flow: Optional[dict] = None
    constant_shape_verified: bool = False


def line(t,y):
    t=np.asarray(t,float);y=np.asarray(y,float)
    return np.linalg.lstsq(np.c_[np.ones(len(t)),t-t[-1]],y,rcond=None)[0]


def bauer_disc_coordinates(history,calibration):
    """Bauer JIRS 2018 Eqs. 1, 9--14, in horizontal diameter R units.

    Horizontal disc correction and the published *approximate* virtual vertical
    disc extension do not require the real target height/width ratio. The latter
    is only needed to express the vertical CPA in height H units. As in Eq. 11,
    the measured vertical image midpoint is corrected with the virtual size;
    its view angle is atan(y/f). This retains the paper's symmetric ellipse/disc
    approximation, not an exact reconstruction of arbitrary aircraft geometry.

    Returns [X/R,Y/R,Z/R] image-derived samples and the original horizontal-only
    samples. Direct vector minimization of their LS fit is equivalent to the
    paper's horizontal/vertical special-point interpolation (Eqs. 26--29), and
    also handles zero vertical velocity without division by its line slope.
    """
    k=calibration;f=k.focal_px
    edges=np.array([[h.box[0]-k.cx,h.box[2]-k.cx] for h in history],float)
    y=np.array([(h.box[1]+h.box[3])/2-k.cy for h in history],float)
    angles=np.arctan(edges/f)
    beta=angles.mean(axis=1);gamma_x=np.diff(angles,axis=1)[:,0]/2
    sx=np.diff(edges,axis=1)[:,0]*np.cos(angles).sum(axis=1)
    xbar=edges.mean(axis=1)*(1-sx*sx/(16*f*f))
    alpha=np.arctan(y/f)
    gamma_y=np.arctan(np.hypot(1,np.tan(beta))/np.hypot(1,np.tan(alpha))*np.tan(gamma_x))
    virtual_angles=np.c_[alpha-gamma_y,alpha+gamma_y]
    if np.any(np.abs(virtual_angles)>=np.pi/2):
        raise ValueError('Virtual disc crosses the pinhole image-plane horizon')
    virtual_edges=f*np.tan(virtual_angles)
    sy=np.diff(virtual_edges,axis=1)[:,0]*np.cos(virtual_angles).sum(axis=1)
    if not np.isfinite([sx,sy]).all() or min(sx.min(),sy.min())<=0:
        raise ValueError('Invalid corrected virtual-disc extent')
    ybar=y*(1-sy*sy/(16*f*f))
    horizontal=np.c_[2*xbar/sx,2*f/sx]
    coordinates=np.c_[2*xbar/sx,2*ybar/sy,f*(1/sx+1/sy)]
    return coordinates,horizontal,{'vertical_view_angle_convention':'atan(measured_bbox_midpoint_y/f)',
                                  'vertical_extension':'JIRS2018_Eq9_14_approximate_virtual_disc',
                                  'minimum_reconstructed_depth_diameters':float(coordinates[:,2].min())}


def wcv_taumod(position,velocity,dthr=1222.32,zthr=137.16,tthr=35.,tcoa=0.):
    """NASA instantaneous WCV_TAUMOD + WCV_TCOA equations; SI state only.

    XYZ here uses camera-style axes (Y vertical), mapped explicitly to horizontal
    X/Z. Default thresholds correspond to the pinned DO-365 Phase I table. This
    is the instantaneous predicate, not the full DAIDALUS alerting/bands package.
    """
    r=np.asarray(position,float);v=np.asarray(velocity,float)
    if r.shape!=(3,) or v.shape!=(3,) or not np.isfinite([r,v]).all():
        raise ValueError('Finite 3D relative state required')
    s=r[[0,2]];u=v[[0,2]];dot=float(s@u);speed2=float(u@u)
    tcpa=-dot/speed2 if speed2>1e-15 else 0.
    dcpa=float(np.linalg.norm(s+tcpa*u))
    tau=(dthr*dthr-float(s@s))/dot if dot<0 else None
    horizontal=bool(np.linalg.norm(s)<=dthr or (dcpa<=dthr and tau is not None and 0<=tau<=tthr))
    vertical=bool(abs(r[1])<=zthr or (v[1]!=0 and r[1]*v[1]<=0 and -r[1]/v[1]<=tcoa))
    return horizontal and vertical,{'modified_tau_s':tau,'horizontal_dcpa_m':dcpa,
                                    'horizontal_tcpa_s':tcpa,'horizontal_wcv':horizontal,'vertical_wcv':vertical}


def relative_motion_observer(points,velocity,previous=None,alpha=1e-6):
    """Banday–Keshavan Eq. 2/5, normalized de-rotated image flow; no bound claim."""
    p=np.asarray(points,float);d=np.asarray(velocity,float)
    if p.ndim!=2 or p.shape[1]!=2 or p.shape!=d.shape or not np.isfinite([p,d]).all():
        raise ValueError('Paired normalized feature positions/velocities required')
    phi=np.zeros((len(p),2,3));phi[:,0,0]=1;phi[:,1,1]=1;phi[:,:,2]=-p
    phi=phi.reshape(-1,3);r=d.ravel();g=phi.T@phi
    eigen=np.linalg.eigvalsh(g)
    if not len(p) or eigen[0]<1e-8:
        return None,{'gramian_eigenvalues':eigen.tolist(),'reason':'insufficient_excitation'}
    old=np.zeros(3) if previous is None else np.asarray(previous,float)
    xi=old+np.linalg.solve(g+alpha*np.eye(3),phi.T@(r-phi@old))
    return xi,{'gramian_eigenvalues':eigen.tolist(),'residual_rms':float(np.sqrt(np.mean((r-phi@xi)**2)))}


def wcv_future(position,velocity,horizon=60.,**thresholds):
    """Exact event partition of NASA's constant-velocity well-clear predicate.

    Predicate changes occur only at circle/modified-tau/closing/altitude roots.
    Evaluating the intervals between all roots avoids a discrete lookahead grid.
    This independently reproduces conflict existence, not alerting or guidance.
    """
    r=np.asarray(position,float);v=np.asarray(velocity,float)
    d=thresholds.get('dthr',1222.32);z=thresholds.get('zthr',137.16)
    tt=thresholds.get('tthr',35.);tc=thresholds.get('tcoa',0.)
    s=r[[0,2]];u=v[[0,2]];a=float(u@u);b=float(s@u);ss=float(s@s)
    roots=[0.,float(horizon)]
    for coefficients in [(a,2*b,ss-d*d),(a,2*b+tt*a,ss+tt*b-d*d),(a,b)]:
        coef=np.trim_zeros(np.array(coefficients,float),'f')
        if len(coef)>1:
            roots.extend(float(x.real) for x in np.roots(coef) if abs(x.imag)<1e-9 and 0<x.real<horizon)
    if v[1]!=0:
        roots.extend(x for x in [(-z-r[1])/v[1],(z-r[1])/v[1],-r[1]/v[1],-r[1]/v[1]-tc] if 0<x<horizon)
    roots=sorted(set(roots))
    probes=sorted(roots+[(a+b)/2 for a,b in zip(roots,roots[1:])])
    first=None
    for t in probes:
        if wcv_taumod(r+t*v,v,**thresholds)[0]:
            first=t;break
    return first is not None,{'wcv_within_horizon':first is not None,'lookahead_horizon_s':horizon,
                              'first_positive_probe_s':first,'event_boundaries_s':roots}


class Assessor:
    def __init__(self,method,config=None):
        if method not in METHODS:raise ValueError(method)
        self.method=method;self.config=config or Config(correlation_seconds=1. if method=='log' else 0.)
        self.custom={'log':ThreatEstimator,'projective':ProjectiveEstimator,
                     'bounded':BoundedProjectiveEstimator}.get(method,lambda c:None)(self.config)

    def assess(self,history,context,track_id=None):
        c=context;m=self.method
        measured=[h for h in history if h.measured and h.source in ('detector','zoom','oracle_box') and h.time<=c.time]
        recent=[h for h in measured if c.time-h.time<=self.config.window]
        tid=track_id if track_id is not None else (history[-1].track_id if history else 0)
        age=c.time-measured[-1].time if measured else None
        out={'method_id':m,'track_id':tid,'decision':'uncertain','reason_codes':['history'],
             'evidence_age_s':age,'physical_meaning':'image_plane_proxy',
             'tau_s':None,'miss_distance':None,'miss_units':None,'confidence':None,
             'validity':{'measured_samples':len(recent),'camera_mode':c.camera_mode,
                         'camera_diagnostic_valid':bool(c.camera_valid),
                         'rotation_independently_verified':bool(c.rotation_verified),
                         'constant_shape_verified':bool(c.constant_shape_verified)},
             'calibrated_probability':False}
        if not measured or age>.25:
            out.update(decision='unobserved',reason_codes=['no_fresh_measurement']);return out
        last=measured[-1]
        if last.clipped:
            out['reason_codes']=['clipped_measurement'];return out
        if last.ambiguous:
            out['reason_codes']=['association_ambiguous'];return out
        # Explicit model-specific prerequisites; missing state cannot imply safety.
        if m=='wcv_taumod':
            out['physical_meaning']='constant_velocity_metric_well_clear_conflict_within_horizon'
            state=c.relative_state
            if not state or not state.get('verified') or not state.get('provenance'):
                out['reason_codes']=['missing_verified_metric_relative_state'];return out
            conflict,details=wcv_future(state['position_m'],state['velocity_m_s'],self.config.horizon,**state.get('thresholds',{}))
            out.update(details,decision='potential_conflict' if conflict else 'apparent_nonconflict',
                       reason_codes=['future_wcv_taumod_event_partition'],miss_units='m',
                       state_provenance=state['provenance'])
            return out
        if m=='rv_c3bf':
            out['physical_meaning']='nominal_horizontal_point_observer_collision_cone'
            flow=c.feature_flow
            if not c.calibration or not c.rotation_verified or not flow or not flow.get('measured'):
                out['reason_codes']=['missing_calibrated_derotated_target_feature_flow'];return out
            xi,info=relative_motion_observer(flow['points_normalized'],flow['velocity_normalized_s'])
            out.update(info)
            if xi is None:
                out['reason_codes']=['insufficient_excitation'];return out
            k=c.calibration;u=(last.center[0]-k.cx)/k.focal_px
            los=np.array([u,1.]);v=xi[[0,2]]
            if np.linalg.norm(v)<1e-10:
                out['reason_codes']=['relative_direction_unresolved'];return out
            cos_theta=float(-v@los/(np.linalg.norm(v)*np.linalg.norm(los)))
            left=np.arctan((last.box[0]-k.cx)/k.focal_px)
            right=np.arctan((last.box[2]-k.cx)/k.focal_px)
            phi=abs(right-left)/2
            barrier=float(np.cos(phi)-cos_theta)
            out.update(decision='potential_conflict' if barrier<0 else 'apparent_nonconflict',
                       reason_codes=['nominal_cone_only_no_footprint_or_certified_bound'],
                       normalized_relative_velocity_s=xi.tolist(),barrier=barrier,
                       cone_half_angle_rad=float(phi),robust_safety_guarantee=False)
            return out
        if m=='lai_lin':
            if len(measured)<2:
                return out
            a0=measured[0].area;change=last.area-a0
            expansion=float(change/np.hypot(a0,change))
            center=np.array([c.width/2,c.height/2]);prev=measured[-2]
            inward=center-np.array(prev.center);motion=np.array(last.center)-prev.center
            position=float(max(0.,1-np.linalg.norm(np.array(last.center)-center)/(c.height/2)))
            denominator=np.linalg.norm(inward)*np.linalg.norm(motion)
            bearing=float(max(0.,inward@motion/denominator)) if denominator>1e-12 else None
            low=.5*expansion+.333*position+.167*(bearing or 0.)
            high=low if bearing is not None else low+.167
            decision='potential_conflict' if low>.5 else 'apparent_nonconflict' if high<=.5 else 'uncertain'
            out.update(decision=decision,reason_codes=['published_risk_index_bbox_area_adaptation']+
                       (['undefined_bearing_factor_interval'] if bearing is None else []),
                       risk_index=low if bearing is not None else None,risk_index_interval=[low,high],
                       area_factor=expansion,position_factor=position,bearing_factor=bearing,
                       initial_area_time_s=measured[0].time,published_warning_threshold=.5)
            return out
        if len(recent)<self.config.min_samples or recent[-1].time-recent[0].time<self.config.min_history:
            return out
        t=np.array([h.time for h in recent]);spans=np.array([h.span for h in recent])
        positions=np.array([h.center for h in recent])
        if min(spans)<self.config.min_span:
            out['reason_codes']=['tiny_measurements'];return out
        if m=='airtrack_cues':
            first=recent[0];dt=last.time-first.time
            ratio=np.sqrt(last.area/first.area);denominator=ratio-1
            pair=dt/denominator if abs(denominator)>1e-12 else None
            vel=line(t,positions)[1]
            out.update(reason_codes=['cue_only_no_published_collision_classifier'],
                       physical_meaning='camera_depth_plane_intercept_under_constant_projected_shape',
                       published_pair_time_at_later_sample_s=pair,
                       tau_s=pair,
                       pixel_rate_s=float(np.linalg.norm(vel)),angular_rate_deg_s=None,
                       area_pair_times_s=[first.time,last.time])
            return out
        if m=='bauer_disc':
            out['physical_meaning']='3D_disc_model_CPA_in_intruder_horizontal_diameter_units'
            if not c.calibration:
                out['reason_codes']=['missing_verified_intrinsics'];return out
            k=c.calibration;k.validate_resolution(c.width,c.height)
            try:
                coordinates,horizontal,details=bauer_disc_coordinates(recent,k)
            except ValueError:
                out['reason_codes']=['invalid_virtual_disc_projection'];return out
            r,v=line(t,coordinates)
            rh,vh=line(t,horizontal);vhh=float(vh@vh)
            th=float(-rh@vh/vhh) if vhh>1e-12 else None
            mh=float(np.linalg.norm(rh+max(0.,th)*vh)) if th is not None else None
            vv=float(v@v)
            if vv<1e-12:
                out['reason_codes']=['relative_motion_unresolved'];return out
            tau=float(-r@v/vv);miss=float(np.linalg.norm(r+max(0,tau)*v))
            decision=('potential_conflict' if 0<tau<self.config.horizon and miss<self.config.threat_miss
                      else 'apparent_nonconflict' if tau>0 and miss>self.config.nonthreat_miss else 'uncertain')
            out.update(details,decision=decision,reason_codes=['published_virtual_disc_3d_reconstruction'],
                       tau_s=tau,miss_distance=miss,miss_units='intruder_horizontal_diameters',
                       relative_position_diameters=r.tolist(),relative_velocity_diameters_s=v.tolist(),
                       horizontal_tau_s=th,horizontal_miss_distance=mh,
                       relative_horizontal_position=rh.tolist(),relative_horizontal_velocity=vh.tolist(),
                       model_3d_CPA_available=True,true_3d_CPA_available=False,
                       vertical_CPA_in_height_units_available=False,
                       approximation_assumptions=['disc_like_horizontal_shape','symmetric_vertical_ellipse_projection',
                                                  'depth_much_larger_than_radius','constant_relative_velocity',
                                                  'trajectory_frame_compensation'],
                       published_flight_thresholds_applied=False)
            return out
        if m=='growth':
            growth=float(line(t,np.log(spans))[1]);vel=line(t,positions)[1]
            drift=float(np.linalg.norm(vel));miss=drift/(last.span*growth) if growth>0 else None
            decision='potential_conflict' if growth>.008 and 1/growth<60 and miss<2 else 'uncertain'
            out.update(decision=decision,reason_codes=['trivial_constant_bearing_growth'],
                       tau_s=1/growth if growth>0 else None,growth_s=growth,
                       miss_distance=miss,miss_units='span_proxy')
            return out
        history4=[(h.time,*h.center,h.span) for h in recent]
        if m=='calibrated':
            out['physical_meaning']='constant_transverse_span_model_CPA'
            if not c.calibration:
                out['reason_codes']=['missing_verified_intrinsics'];return out
            estimator=CalibratedCPAEstimator(c.calibration,self.config)
        else:
            estimator=self.custom
            out['physical_meaning']='camera_depth_plane_intercept_proxy'
        e=estimator.assess(history4,c.time,c.camera_valid,last.ambiguous)
        out.update(decision=STATES[e['state']],reason_codes=[e['reason']],
                   tau_s=e.get('tau'),miss_distance=e.get('miss_spans'),miss_units='target_span_proxy',legacy=e)
        return out


def scope_result(result,context):
    """Retain raw candidate and apply existing moving-camera clearance scope."""
    out=dict(result);out['raw_decision']=out['decision']
    if context.camera_mode!='fixed' and not context.rotation_verified and out['decision']=='apparent_nonconflict':
        out['decision']='uncertain';out['reason_codes']=[*out['reason_codes'],'unresolved_camera_rotation_translation']
    return out
