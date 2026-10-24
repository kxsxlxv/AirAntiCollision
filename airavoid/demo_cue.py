"""Causal, short-history presentation cue, separate from research assessment.

This is a declared image-space heuristic, not a published-method reproduction,
physical CPA estimate, collision classifier, or safe maneuver command. A valid
background fit permits an image cue; it never verifies physical camera rotation.
Only fresh detector/zoom observations can add scale or trail evidence.
"""
from collections import defaultdict,deque
import numpy as np


MONITORING='MONITORING'
POSSIBLE='POSSIBLE CONVERGENCE'
INSUFFICIENT='INSUFFICIENT EVIDENCE'
REASON_TEXT={
    'collecting_short_measured_history':'Накапливаем короткую историю измерений',
    'unreliable_or_unknown_camera':'Движение камеры не подтверждено — подсказка скрыта',
    'clipped_measurement':'Цель обрезана границей кадра',
    'ambiguous_association':'Неоднозначная связь наблюдений с целью',
    'no_fresh_detector_measurement':'Нет свежего измерения детектора',
    'measurement_discontinuity':'Скачок измерений — собираем историю заново',
    'stable_measured_motion_no_convergence_cue':'Наблюдаем цель; устойчивое сближение по изображению не выделено',
    'unstable_scale_or_aspect':'Размер или форма изображения цели нестабильны',
    'confirming_measured_convergence':'Проверяем устойчивость роста изображения',
    'persistent_growth_with_low_relative_image_drift':'Устойчивый рост при малом смещении по изображению',
    'target_too_small_for_scale_evidence':'Цель слишком мала для оценки изменения размера',
    'no_confirmed_convergence_target':'Нет подтверждённой подсказки сближения',
    'symmetric_central_target_direction_unobservable':'Цель по центру — сторону нельзя обосновать',
    'conflicting_convergence_targets':'Подсказки разных целей противоречат друг другу',
    'direction_occupied_by_other_measured_target':'В выбранной стороне есть другая наблюдаемая цель',
    'confirming_stable_image_direction':'Проверяем устойчивость направления',
    'away_from_confirmed_image_occupancy_with_hysteresis':'Иллюстративная сторона от наблюдаемой цели; безопасный манёвр не определён',
    'stable_measured_target_image_separation_illustration':'Наблюдаем цель; иллюстрация увеличения разделения по изображению, не предупреждение и не безопасный манёвр',
    'conflicting_image_separation_directions':'Иллюстративные направления для разных целей противоречат друг другу',
}


def _fit(times,values):
    """Median pairwise slope, median intercept, and residual MAD; causal only."""
    t=np.asarray(times,float);y=np.asarray(values,float)
    if len(t)>32:
        take=np.unique(np.linspace(0,len(t)-1,32).astype(int));t=t[take];y=y[take]
    dt=t[None,:]-t[:,None];dy=y[None,:]-y[:,None]
    use=dt>1e-7
    slope=float(np.median(dy[use]/dt[use]))
    intercept=float(np.median(y-slope*t))
    mad=float(np.median(np.abs(y-intercept-slope*t)))
    return slope,mad


class EarlyEncounterCue:
    """Persistent optical growth with low image-relative drift over 0.3--0.6 s.

    Constants are fixed generic presentation operating points. They have not been
    calibrated against held-out collision labels. Priority/direction hysteresis
    reduces display flicker; missing or contradictory evidence removes the arrow.
    A stable MONITORING target can support an illustrative image-separation
    direction independently of warning evidence; this does not change its state.
    """
    def __init__(self,window=.6,min_window=.3,confirmation=.16,min_samples=5,
                 min_growth=.035,min_span=6.,direction_confirmation=.16):
        if not 0<min_window<=window or confirmation<0 or min_samples<3:
            raise ValueError('Positive causal window and at least three samples required')
        self.window=float(window);self.min_window=float(min_window)
        self.confirmation=float(confirmation);self.min_samples=int(min_samples)
        self.min_growth=float(min_growth);self.min_span=float(min_span)
        self.direction_confirmation=float(direction_confirmation)
        self.histories=defaultdict(deque);self.pending={};self.last_time=None
        self.priority_id=None;self.priority_since=None;self.challenger=None
        self.direction_vector=None;self.direction_pending=None

    @property
    def contract(self):
        """Serializable operating rules; descriptive metadata, never calibration."""
        return {'kind':'independent_demo_image_cue_not_research_classifier',
                'window_s':self.window,'min_window_s':self.min_window,
                'confirmation_s':self.confirmation,'min_samples':self.min_samples,
                'min_growth_per_s':self.min_growth,'min_span_px':self.min_span,
                'direction_confirmation_s':self.direction_confirmation,
                'fit':'median all pairwise slopes and median-intercept residual MAD; at most32 uniformly selected measured samples',
                'measurement':'fresh detector/zoom XYXY only, explicit measured guard; max(width,height) scale; raw image centres',
                'quality':{'max_scale_log_mad':'max(.025,1/median_span_px)',
                           'max_aspect_log_p90_minus_p10':.25,
                           'minimum_fitted_log_growth':'max(.015,.5/median_span_px,2*scale_log_mad)',
                           'max_measurement_gap_s':.25,'max_normalized_center_jump':2.5,
                           'max_single_log_span_jump':.45,'reset_update_gap_s':.3},
                'drift':'norm(robust velocity_px_s)/median_span_px <= max(.08,2*growth_per_s) for a convergence candidate',
                'warning':'stable sufficient measured history AND growth >= minimum AND positive growth signal AND low relative drift, sustained confirmation',
                'priority':'stable MONITORING or POSSIBLE CONVERGENCE; span/persistence salience or confirmed-growth salience, 30% challenger advantage sustained .25s',
                'direction':'nonwarning image-separation illustration also allowed for stable MONITORING; away from .5s linear image projection, quantized45deg, independent confirmation; unresolved under symmetry/occupancy/conflicts',
                'direction_quality':{'minimum_normalized_image_offset':.04,
                                     'other_target_relative_salience':.6,'minimum_compatible_direction_dot':.1,
                                     'occupied_center_radius':.65,'occupied_direction_dot':.85},
                'camera':'valid declared setup/background fit required; evidence resets clear history; background fit never verifies physical rotation',
                'calibrated_probability':False,'physical_collision_or_safe_maneuver':False,
                'threshold_selection':'fixed generic presentation rules, no held-out collision-label tuning'}

    def reset(self):
        self.histories.clear();self.pending.clear();self.priority_id=None
        self.priority_since=None;self.challenger=None
        self.direction_vector=None;self.direction_pending=None
        self.last_time=None

    def update(self,objects,time,camera,image_size):
        t=float(time);width,height=map(float,image_size)
        if not np.isfinite([t,width,height]).all() or min(width,height)<=0:
            raise ValueError('Finite time and positive source image size required')
        if self.last_time is not None and t<=self.last_time:
            raise ValueError('Demo cue timestamps must strictly increase')
        if self.last_time is not None and t-self.last_time>.3:self.reset()
        valid=bool(camera.get('valid')) and camera.get('mode')!='unknown'
        if camera.get('evidence_reset') or not valid:self.reset()
        self.last_time=t
        targets=[];seen=set()
        for obj in objects:
            tid=int(obj['id'])
            if tid in seen:raise ValueError('Duplicate target identity')
            seen.add(tid)
            box=np.asarray(obj['box'],float)
            if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[2:]<=box[:2]):
                continue
            assessment=obj.get('assessment') or {};research=assessment.get('research') or {}
            observation=obj.get('demo_observation') or {}
            reasons=research.get('reason_codes',[])
            fresh=(obj.get('source') in ('detector','zoom') and
                   abs(float(obj.get('last_seen',-1))-t)<1e-6 and observation.get('measured',True))
            clipped=(bool(observation.get('clipped')) or bool(assessment.get('clipped_observation')) or box[0]<=1 or box[1]<=1 or
                     box[2]>=width-1 or box[3]>=height-1)
            ambiguous=(bool(observation.get('ambiguous')) or bool(obj.get('ambiguous')) or
                       any('association_ambiguous' in str(reason) for reason in reasons) or
                       assessment.get('reason')=='association')
            reason='collecting_short_measured_history'
            history=self.histories[tid]
            if not valid:reason='unreliable_or_unknown_camera'
            elif clipped:reason='clipped_measurement'
            elif ambiguous:reason='ambiguous_association'
            elif not fresh:reason='no_fresh_detector_measurement'
            if not valid or clipped or ambiguous:
                history.clear();self.pending.pop(tid,None)
            elif fresh:
                center=(box[:2]+box[2:])/2;span=float(max(box[2:]-box[:2]))
                aspect=float(np.log((box[2]-box[0])/(box[3]-box[1])))
                if history:
                    old=history[-1]
                    jump=np.linalg.norm(center-np.array(old[1:3]))/max(span,old[3],1.)
                    if t-old[0]>.25 or jump>2.5 or abs(np.log(span/old[3]))>.45:
                        history.clear();self.pending.pop(tid,None);reason='measurement_discontinuity'
                history.append((t,float(center[0]),float(center[1]),span,aspect))
            while history and t-history[0][0]>self.window+1e-9:history.popleft()
            result={'id':tid,'box':box.tolist(),'state':INSUFFICIENT,'reason':reason,
                    'growth_per_s':None,'drift_normalized_per_s':None,'velocity_px_s':None,
                    'window_s':float(history[-1][0]-history[0][0]) if len(history)>1 else 0.,
                    'samples':len(history),'evidence_age_s':float(t-history[-1][0]) if history else None,
                    'measured_trail':[[float(v[0]),float(v[1]),float(v[2])] for v in history],
                    'camera_scope':'image_relative_only_no_verified_rotation',
                    'confidence':None,'calibrated_probability':False,'priority_score':0.,
                    'detector_confidence':float(obj['confidence']) if obj.get('confidence') is not None and np.isfinite(obj['confidence']) else None}
            duration=result['window_s']
            if fresh and valid and not clipped and not ambiguous and len(history)>=self.min_samples and duration>=self.min_window-1e-9:
                a=np.asarray(history);times=a[:,0]-t;span=float(np.median(a[:,3]))
                if span>=self.min_span:
                    growth,residual=_fit(times,np.log(a[:,3]));vx,_=_fit(times,a[:,1]);vy,_=_fit(times,a[:,2])
                    drift=float(np.hypot(vx,vy)/span)
                    aspect_range=float(np.percentile(a[:,4],90)-np.percentile(a[:,4],10))
                    stable=residual<=max(.025,1/span) and aspect_range<=.25
                    signal=growth*duration>max(.015,.5/span,2*residual)
                    converging=(stable and growth>=self.min_growth and signal and drift<=max(.08,2*growth))
                    result.update(growth_per_s=growth,drift_normalized_per_s=drift,
                                  velocity_px_s=[vx,vy],scale_residual_log_mad=residual,
                                  aspect_log_range=aspect_range,state=MONITORING,
                                  priority_score=float(span/max(width,height)+.2*min(1,duration/self.window)),
                                  reason='stable_measured_motion_no_convergence_cue')
                    if not stable:
                        result.update(state=INSUFFICIENT,reason='unstable_scale_or_aspect')
                    if converging:
                        since=self.pending.setdefault(tid,t)
                        result['confirmation_elapsed_s']=float(t-since)
                        result['reason']='confirming_measured_convergence'
                        if t-since>=self.confirmation-1e-9:
                            result.update(state=POSSIBLE,reason='persistent_growth_with_low_relative_image_drift',
                                          priority_score=float(2+max(0.,growth)/(1+drift)))
                    else:self.pending.pop(tid,None)
                else:result['reason']='target_too_small_for_scale_evidence';self.pending.pop(tid,None)
            elif not fresh:self.pending.pop(tid,None)
            result['state_id']=result['state'].lower().replace(' ','_')
            result['reason_text']=REASON_TEXT.get(result['reason'],result['reason'])
            targets.append(result)
        for tid in list(self.histories):
            if tid not in seen:
                self.pending.pop(tid,None)
                if not self.histories[tid] or t-self.histories[tid][-1][0]>.25:del self.histories[tid]
        candidates=[x for x in targets if x['state']==POSSIBLE]
        eligible=[x for x in targets if x['state'] in (MONITORING,POSSIBLE)]
        self._priority(eligible,t)
        direction=self._direction(eligible,targets,t,width,height) if valid else self._unresolved('unreliable_or_unknown_camera')
        if candidates:state=POSSIBLE
        elif any(x['state']==MONITORING for x in targets):state=MONITORING
        else:state=INSUFFICIENT
        direction['reason_text']=REASON_TEXT.get(direction['reason'],direction['reason'])
        return {'state':state,'state_id':state.lower().replace(' ','_'),'targets':targets,
                'priority_id':self.priority_id,'direction':direction,'time':t,
                'meaning':'short_history_image_presentation_cue_not_physical_collision_assessment',
                'research_assessment_changed':False}

    def _priority(self,candidates,t):
        available={x['id']:x for x in candidates}
        if not available:
            self.priority_id=None;self.priority_since=None;self.challenger=None
            return
        best=sorted(candidates,key=lambda x:(-x['priority_score'],x['id']))[0]
        current=available.get(self.priority_id)
        if current is None or (best['state']==POSSIBLE and current['state']!=POSSIBLE):
            self.priority_id=best['id'];self.priority_since=t;self.challenger=None
        elif best['id']!=current['id'] and best['priority_score']>current['priority_score']*1.3:
            if self.challenger is None or self.challenger[0]!=best['id']:self.challenger=(best['id'],t)
            elif t-self.challenger[1]>=.25:self.priority_id=best['id'];self.priority_since=t;self.challenger=None
        else:self.challenger=None

    def _unresolved(self,reason):
        self.direction_vector=None;self.direction_pending=None
        return {'status':'unresolved','vector':None,'reason':reason,
                'meaning':'illustrative_image_plane_cue_not_safe_maneuver'}

    def _direction(self,candidates,targets,t,width,height):
        primary=next((x for x in candidates if x['id']==self.priority_id),None)
        if primary is None:return self._unresolved('no_confirmed_convergence_target')
        def offset(target):
            b=target['box'];v=target['velocity_px_s'] or [0.,0.]
            return np.array([((b[0]+b[2])/2+.5*v[0]-width/2)/(width/2),
                             ((b[1]+b[3])/2+.5*v[1]-height/2)/(height/2)])
        p=offset(primary);norm=float(np.linalg.norm(p))
        if norm<.04:return self._unresolved('symmetric_central_target_direction_unobservable')
        proposed=-p/norm
        for other in candidates:
            if other['id']==primary['id'] or other['priority_score']<.6*primary['priority_score']:continue
            q=offset(other);qn=float(np.linalg.norm(q))
            if qn>.04 and float(proposed@(-q/qn))<.1:
                reason=('conflicting_convergence_targets' if primary['state']==POSSIBLE and other['state']==POSSIBLE
                        else 'conflicting_image_separation_directions')
                return self._unresolved(reason)
        # Do not point toward another currently measured target close to the centre.
        for other in targets:
            if other['id']==primary['id'] or other['evidence_age_s']!=0:continue
            q=offset(other);qn=float(np.linalg.norm(q))
            if .04<qn<.65 and float(proposed@(q/qn))>.85:return self._unresolved('direction_occupied_by_other_measured_target')
        angle=np.arctan2(proposed[1],proposed[0]);angle=np.round(angle/(np.pi/4))*(np.pi/4)
        candidate=np.array([np.cos(angle),np.sin(angle)])
        if self.direction_vector is not None and self.direction_vector@candidate>=.99:
            self.direction_pending=None
        else:
            if self.direction_pending is None or self.direction_pending[0]@candidate<.99:
                self.direction_pending=(candidate,t)
            if t-self.direction_pending[1]<self.direction_confirmation-1e-9:
                return {'status':'unresolved','vector':None,'reason':'confirming_stable_image_direction',
                        'meaning':'illustrative_image_plane_cue_not_safe_maneuver'}
            self.direction_vector=candidate;self.direction_pending=None
        return {'status':'resolved','vector':[float(v) for v in self.direction_vector],
                'reason':('away_from_confirmed_image_occupancy_with_hysteresis' if primary['state']==POSSIBLE else
                          'stable_measured_target_image_separation_illustration'),
                'priority_id':self.priority_id,'warning_evidence':primary['state']==POSSIBLE,
                'meaning':'illustrative_image_plane_cue_not_safe_maneuver'}


DemoEncounterCue=EarlyEncounterCue
