"""Causal bridge from production tracks to the measured Stage 2 contract.

Published methods receive actual XYXY detector observations. Tracker-predicted
extents and background similarity are never promoted to independent range or
verified camera de-rotation. The renderer schema contains the complete research
result separately from its display-only legacy state.
"""
from collections import defaultdict, deque
import json
from pathlib import Path
import numpy as np
from .assessment import Assessor, Measurement, Context, scope_result


class AssessmentRuntime:
    def __init__(self, method, config=None, calibration=None):
        self.assessor=Assessor(method,config)
        self.config=self.assessor.config
        self.calibration=calibration
        self.histories=defaultdict(lambda:deque(maxlen=500))
        self.anchors={}

    def reset(self):
        self.histories.clear()
        self.anchors.clear()

    def retain(self,tracks):
        active={track.id for track in tracks}
        for track_id in set(self.histories)-active:
            del self.histories[track_id]
            self.anchors.pop(track_id,None)

    def assess(self, track, time, camera, image_size):
        history=self.histories[track.id]
        if track.last==time and (not history or history[-1].time!=time):
            # Reset support after clipping or a new ambiguity; preserve a fresh
            # invalid sample so the reason remains visible instead of vanishing.
            if track.clipped or time<track.ambiguous_until or (history and (history[-1].clipped or history[-1].ambiguous)):
                history.clear()
                self.anchors.pop(track.id,None)
            history.append(Measurement(time,tuple(track.box),track_id=track.id,
                confidence=track.confidence,measured=track.source in ('detector','zoom'),
                source=track.source,clipped=track.clipped,
                ambiguous=time<track.ambiguous_until))
            if history[-1].measured and not history[-1].clipped and not history[-1].ambiguous:
                self.anchors.setdefault(track.id,history[-1])
        width,height=image_size
        context=Context(time,width,height,camera_mode=camera['mode'],
            camera_valid=camera['valid'],calibration=self.calibration,
            rotation_verified=False,constant_shape_verified=False)
        inputs=list(history)
        anchor=self.anchors.get(track.id)
        if self.assessor.method=='lai_lin' and anchor is not None and inputs and anchor.time<inputs[0].time:
            inputs=[anchor,*inputs]
        result=scope_result(self.assessor.assess(inputs,context,track.id),context)
        result=scope_for_runtime(result)
        return display_assessment(result,history,camera['valid'])


def scope_for_runtime(result):
    """Deployment restrictions leave the raw published comparison unchanged."""
    out=dict(result)
    if out.get('raw_decision',out['decision'])=='apparent_nonconflict':
        if out['method_id']=='lai_lin':
            out.update(decision='uncertain',reason_codes=[*out['reason_codes'],
                'low_index_is_not_clearance_evidence'])
        elif out['method_id']=='rv_c3bf':
            out.update(decision='uncertain',reason_codes=[*out['reason_codes'],
                'point_cone_without_finite_aircraft_footprint_is_not_clearance'])
    return out


def display_assessment(result, history=(), camera_valid=False):
    """No invented rates: absent model quantities remain None in the renderer."""
    legacy=dict(result.get('legacy') or {})
    state={'potential_conflict':'threat','apparent_nonconflict':'nonthreat',
           'uncertain':'uncertain','unobserved':'uncertain'}[result['decision']]
    out={'state':state,'reason':'research_result','samples':len(history),
         'camera_valid':bool(camera_valid),'growth':None,'growth_se':None,
         'drift':None,'miss_spans':None,'miss_low':None,'miss_high':None,
         'tau':None,'velocity':None,'position_sigma':None,'scale_residual':None,
         'span':history[-1].span if history else None}
    out.update(legacy)
    out.update(state=state,reason='|'.join(result['reason_codes']),research=result,
        reason_text='; '.join(result['reason_codes']),
        method_id=result['method_id'],physical_meaning=result['physical_meaning'])
    # Research tau/miss values may have different meanings and units; they stay
    # in research, never relabelled as legacy image-span CPA.
    return out


def detector_contract(detector, variant):
    """Validate the actual runtime tensor contract and declare model taxonomy."""
    if detector.backend=='tensorrt':
        shape=tuple(detector.buffers[detector.input_name].shape)
        output=tuple(detector.buffers[detector.output_name].shape)
    else:
        shape=tuple(detector.session.get_inputs()[0].shape)
        output=tuple(detector.session.get_outputs()[0].shape)
    expected=(1,3,416,704) if variant=='A' else (1,3,800,1440)
    if shape!=expected or output!=(1,300,6):
        raise ValueError(f'Detector {variant} runtime contract mismatch: {shape}, {output}')
    from .provenance import file_sha256
    digest=file_sha256(detector.weights)
    manifest=Path(__file__).resolve().parents[1]/'reports/stage2_method_provenance.json'
    source=json.loads(manifest.read_text()) if manifest.exists() else {}
    matching={k:v for k,v in source.get('detectors',{}).items() if v['sha256']==digest and k.startswith(variant+'_')}
    if variant=='B' and not matching:
        raise ValueError('P2 weights are not registered in reports/stage2_method_provenance.json; verify exact supplied-export provenance before use')
    provenance={'weights_sha256':digest,'registered_supplied_export':bool(matching),
        'provenance_manifest_sha256':file_sha256(manifest) if manifest.exists() else None,
        'matching_assets':list(matching),'input_dtype':np.dtype(detector.input_dtype).name}
    if variant=='B':return dict(detector.contract,**provenance)
    from .detector import NAMES
    return {'variant':'A','input_shape':list(shape),'output_shape':list(output),
        'semantic_taxonomy':NAMES,'letterbox':'RGB NCHW /255, linear resize, centred 114 pad, round dimensions',
        'preprocess':'CUDA OpenCV' if detector.gpu_preprocessor else 'CPU OpenCV',
        'tiled':bool(detector.tiled),'adaptive_tiles':bool(detector.adaptive),
        'tile_shape':[416,704],'tile_edge_margin_px':detector.tile_edge_margin,
        'cross_box_nms_iou':.45,
        'conf':detector.conf,'weights_path':str(Path(detector.weights).resolve()),**provenance}


def validate_detection_classes(boxes, variant):
    limit=5 if variant=='A' else 1
    for box in boxes:
        class_id=box[5]
        if not np.isfinite(class_id) or class_id!=int(class_id) or not 0<=class_id<limit:
            raise ValueError(f'Detector {variant} emitted invalid class {class_id}')
