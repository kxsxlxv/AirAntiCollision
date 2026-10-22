import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
from .detector import Detector,NAMES
from .geometry import Config,ThreatEstimator,advisory,apply_camera_scope
from .tracking import Tracker,CameraMotion
from .render import Renderer
from .policy import TemporalPolicy
from .projective import ProjectiveEstimator
from .bounded import BoundedProjectiveEstimator
from .calibrated import CameraCalibration,CalibratedCPAEstimator
from .media import SourceClock,TimestampWriter,probe_video,presentation_end,encoder_options
from .provenance import file_sha256,cache_identity,validate_cache_entry
from .lifecycle import RunSession,stop_encoder
from .export import FrameExporter
from .process_export import ProcessFrameExporter
from .assessment import METHODS,STATES
from .assessment_runtime import AssessmentRuntime,detector_contract,validate_detection_classes


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Offline RGB airborne threat research demo')
    p.add_argument('video')
    local_engine=Path(__file__).resolve().parents[1]/'artifacts/yolo26m_local.engine'
    p.add_argument('--weights',help='Explicit matching weights; defaults depend on --detector-variant')
    p.add_argument('--detector-variant',choices=['A','B'],default='A',help='A: supplied five-class 416x704 detector (default); B: supplied fixed-wing-only P2 800x1440 export')
    p.add_argument('--assessment',choices=METHODS,help='Explicit Stage 2 research comparator; default preserves the existing --geometry backend')
    p.add_argument('--demo-showcase',action='store_true',help='Opt-in early image-space encounter cues and 1920x1080 concept presentation; not flight guidance')
    p.add_argument('--output',default='demo/run')
    p.add_argument('--profile',choices=['conservative','reference'],default='conservative',help='Measured conservative defaults, or the earlier reference settings')
    p.add_argument('--conf',type=float,help='Raw detector threshold (conservative .03, reference .18); not a probability')
    p.add_argument('--stride',type=int,default=1)
    p.add_argument('--max-frames',type=int)
    p.add_argument('--camera',choices=['auto','fixed','unknown'],default='auto')
    p.add_argument('--allow-similarity-pass',action='store_true',default=None,help='Research override: allow pass states after background stabilization despite unresolved parallax; reference profile enables this')
    p.add_argument('--policy',choices=['raw','confirmed'],default='confirmed')
    p.add_argument('--geometry',choices=['log','projective','bounded','calibrated'],default='log')
    p.add_argument('--calibration',help='Explicit rectified square-pixel pinhole calibration JSON; required for calibrated geometry')
    p.add_argument('--bounded-pixel-error',type=float,default=2.,help='Assumed per-observation centroid error for bounded geometry, original pixels')
    p.add_argument('--bounded-span-fraction',type=float,default=.1,help='Assumed relative span error for bounded geometry; not calibrated')
    p.add_argument('--allow-recession-clearance',action='store_true',help='Legacy stable-shape hypothesis: allow apparent contraction to clear a target')
    p.add_argument('--correlation-seconds',type=float,help='Bartlett covariance sensitivity bandwidth, seconds (conservative log 1, reference 0); not calibrated')
    crops=p.add_mutually_exclusive_group()
    crops.add_argument('--tiled',dest='tiled',action='store_true')
    crops.add_argument('--full-frame',dest='tiled',action='store_false',help='Use only full-frame inference; can miss tiny targets')
    p.set_defaults(tiled=None)
    p.add_argument('--tile-edge-margin',type=float,help='Reject boxes touching internal crop edges, original pixels (conservative 2, reference 0)')
    p.add_argument('--image-edge-margin',type=float,help='Abstain/reset span history at physical image boundaries (conservative 1, reference 0)')
    p.add_argument('--adaptive-tiles',action='store_true',help='Experimental periodic discovery + target-driven crop scheduling')
    preprocess=p.add_mutually_exclusive_group()
    preprocess.add_argument('--gpu-preprocess',dest='gpu_preprocess',action='store_true',help='Native CUDA OpenCV input conversion and single source upload for TensorRT crops')
    preprocess.add_argument('--cpu-preprocess',dest='gpu_preprocess',action='store_false',help='CPU input conversion; TensorRT model execution still uses the GPU')
    p.set_defaults(gpu_preprocess=None)
    p.add_argument('--sky-mask',help='Optional model-generated sky mask video, exactly source-aligned; see scripts.run_sky_demo')
    p.add_argument('--sky-all-control',action='store_true',help='Ablation: use the same proposal/YOLO zoom path over all pixels, ignoring sky selection')
    p.add_argument('--zoom-proposals',action='store_true',help='Experimental all-pixel tiny proposals with actual YOLO crop confirmation; no sky model')
    p.add_argument('--video-timing',choices=['pts','cfr'],default='pts',help='Preserve presentation timestamps (default) or legacy nominal-rate export')
    p.add_argument('--encoder',choices=['libx264','h264_nvenc'],default='libx264',help='H.264 output encoder; NVENC requires compatible NVIDIA runtime')
    render_mode=p.add_mutually_exclusive_group()
    render_mode.add_argument('--async-render',action='store_true',help='Overlap ordered render/export in a thread, at most two queued frames')
    render_mode.add_argument('--render-process',action='store_true',help='Experimental ordered render/export in a separate process, at most two pending frames')
    p.add_argument('--appearance-bridge',action='store_true',help='Experimental RGB-only LoRAT fill between detector observations')
    p.add_argument('--detector-scale-only',action='store_true',help='Use independent detector span samples; appearance bridge supplies centroid continuity')
    p.add_argument('--detect-every',type=int,help='Run detector every N processed frames (default 3 with bridge, otherwise 1)')
    p.add_argument('--no-render',action='store_true')
    p.add_argument('--detections',help='Replay a model-generated detection JSONL cache; no GT supported')
    p.add_argument('--title',default='RGB • анализ сближения')
    a=p.parse_args(argv)
    if a.weights is None:
        if a.detector_variant=='B':
            a.weights=str(Path(__file__).resolve().parents[1]/'artifacts/stage2/yolo26m_p2_local_fp32.engine')
        else:a.weights=str(local_engine) if local_engine.exists() else '../anti_collision_system/best.onnx'
    if a.detector_variant=='B':
        if a.tiled or a.adaptive_tiles or a.gpu_preprocess or a.sky_mask or a.zoom_proposals or a.appearance_bridge:
            p.error('P2 variant B supports independent full-frame CPU letterboxing only; tiled/GPU-preprocess/sky/zoom/appearance flags are unsupported')
        a.tiled=False;a.gpu_preprocess=False
    if a.assessment and a.geometry!='log':p.error('--assessment and nondefault --geometry are separate backends; choose one')
    if a.demo_showcase and (a.assessment or a.geometry!='log' or a.render_process):
        p.error('--demo-showcase is a separate presentation mode; use default geometry and serial/--async-render export')
    conservative=a.profile=='conservative'
    for key,value in {'conf':.03 if conservative else .18,'tiled':conservative,
                      'tile_edge_margin':2. if conservative else 0.,'image_edge_margin':1. if conservative else 0.,
                      'correlation_seconds':1. if conservative and a.geometry=='log' and a.assessment in (None,'log') else 0.,
                      'allow_similarity_pass':not conservative}.items():
        if getattr(a,key) is None:setattr(a,key,value)
    if a.gpu_preprocess is None:
        a.gpu_preprocess=bool(conservative and not a.detections and str(a.weights).endswith('.engine') and cv2.cuda.getCudaEnabledDeviceCount()>0)
    if a.adaptive_tiles and not a.tiled:p.error('--adaptive-tiles requires crops; omit --full-frame or add --tiled')
    if a.stride<1:p.error('stride must be positive')
    a.detect_every=(3 if a.appearance_bridge else 1) if a.detect_every is None else a.detect_every
    if a.detect_every<1:p.error('detect-every must be positive')
    if not 0<=a.conf<=1:p.error('conf must be in [0,1]')
    if a.max_frames is not None and a.max_frames<1:p.error('max-frames must be positive')
    if not np.isfinite(a.correlation_seconds) or a.correlation_seconds<0:p.error('correlation-seconds must be finite and nonnegative')
    if not np.isfinite(a.tile_edge_margin) or a.tile_edge_margin<0:p.error('tile-edge-margin must be finite and nonnegative')
    if not np.isfinite(a.image_edge_margin) or a.image_edge_margin<0:p.error('image-edge-margin must be finite and nonnegative')
    if a.geometry!='log' and a.correlation_seconds>0:p.error('Temporal covariance experiment currently supports log geometry only')
    if a.geometry=='calibrated' and (not a.calibration or a.camera!='fixed'):p.error('Calibrated geometry requires --calibration and --camera fixed')
    if a.calibration and a.geometry!='calibrated' and a.assessment not in ('bauer_disc','calibrated','rv_c3bf'):p.error('--calibration requires a calibration-aware geometry/assessment backend')
    if not np.isfinite(a.bounded_pixel_error) or a.bounded_pixel_error<0 or not 0<=a.bounded_span_fraction<1:p.error('Bounded pixel error must be finite and nonnegative; span fraction must be in [0,1)')
    if not a.appearance_bridge and a.detect_every!=1:p.error('Skipping detections requires --appearance-bridge')
    if a.detections and a.appearance_bridge:p.error('Replay the bridge observations cache without --appearance-bridge')
    if a.detections and a.sky_mask:p.error('Sky augmentation requires live detector inference; use the separate cached experiment script')
    if a.sky_all_control and not a.sky_mask:p.error('--sky-all-control requires --sky-mask for the matched ablation')
    if a.zoom_proposals and (a.sky_mask or a.detections):p.error('--zoom-proposals requires live inference and no sky mask')
    if a.no_render and (a.async_render or a.render_process):p.error('Render workers require rendered output')
    if a.render_process and a.video_timing!='pts':p.error('--render-process currently requires timestamp-preserving PTS export')
    if Path(a.video).resolve()==(Path(a.output)/'annotated.mp4').resolve():p.error('Output would overwrite the input video')
    if a.detections and Path(a.detections).resolve().parent==Path(a.output).resolve():p.error('Replay output must differ from the input cache directory')
    return a


def execute(a,session):
    pipeline_start=time.perf_counter()
    source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    out=session.output
    cap=cv2.VideoCapture(a.video)
    session.register(cap.release)
    if not cap.isOpened():raise FileNotFoundError(a.video)
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0:raise ValueError('Unknown source FPS; provide a timestamped video')
    source_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_metadata=probe_video(a.video)
    source_duration,source_duration_method=presentation_end(a.video,source_metadata)
    clock=SourceClock(fps);output_end=source_duration
    sky=None
    ok,first=cap.read();cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    if not ok:raise RuntimeError('Video is empty')
    calibration=CameraCalibration.load(a.calibration) if a.calibration else None
    if calibration:calibration.validate_resolution(first.shape[1],first.shape[0])
    provenance_start=time.perf_counter();input_hash=file_sha256(a.video)
    input_hash_s=time.perf_counter()-provenance_start
    if a.sky_mask or a.zoom_proposals:
        from .proposals import SkyProposals
        sky=SkyProposals(a.sky_mask,fps,source_hash=input_hash,source_frames=source_frames,all_region=a.sky_all_control or a.zoom_proposals)
        session.register(sky.close)
    load_start=time.perf_counter()
    cache=None;replay_identity=None;cache_misses=0;replay_provenance=None
    if a.detections:
        cache={int(r['frame']):r for r in map(json.loads,Path(a.detections).read_text().splitlines())}
        if not cache:raise ValueError('Detection cache is empty')
        replay_identity=cache_identity(a.detections,a.video,input_hash,first.shape[1],first.shape[0],fps)
        if replay_identity['status']!='sha256_verified':print('Replay provenance: '+replay_identity['reason'],flush=True)
        metadata_path=Path(a.detections).with_name('benchmark.json')
        upstream=json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
        replay_provenance={'cache_sha256':file_sha256(a.detections),'upstream_benchmark_sha256':file_sha256(metadata_path) if metadata_path.exists() else None,
                           'upstream_summary':{k:upstream[k] for k in ['kind','args','config','providers','weights_sha256'] if k in upstream}}
        model_contract=upstream.get('detector_contract',upstream.get('contract'))
        if model_contract and model_contract['variant']!=a.detector_variant:raise ValueError('Replay detector variant differs from cache provenance')
        if not model_contract and a.detector_variant=='B':raise ValueError('P2 replay requires an explicit upstream runtime/taxonomy contract')
        if model_contract and 'semantic_taxonomy' not in model_contract:model_contract['semantic_taxonomy']=model_contract.get('taxonomy',NAMES)
        if not model_contract:model_contract={'variant':a.detector_variant,'status':'legacy_cache_without_runtime_contract','semantic_taxonomy':NAMES if a.detector_variant=='A' else ['fixed-wing UAV']}
        detector=None;providers=['Replayed observations (no neural inference)']
    else:
        if a.detector_variant=='B':
            from .detector_p2 import DetectorP2
            detector=DetectorP2(a.weights,a.conf)
        else:detector=Detector(a.weights,a.conf,tiled=a.tiled or a.adaptive_tiles,adaptive=a.adaptive_tiles,gpu_preprocess=a.gpu_preprocess,tile_edge_margin=a.tile_edge_margin)
        providers=detector.providers;model_contract=detector_contract(detector,a.detector_variant)
        model_load_s=time.perf_counter()-load_start
        warmup_update_ms=[]
        for _ in range(5):
            warmup_tick=time.perf_counter();detector(first)
            warmup_update_ms.append((time.perf_counter()-warmup_tick)*1000)
        detector.reset_temporal()
    bridge=None
    if a.appearance_bridge:
        from .appearance import AppearanceBridge
        bridge=AppearanceBridge()
    startup=time.perf_counter()-load_start
    camera=CameraMotion(a.camera);tracker=Tracker(image_edge_margin=a.image_edge_margin)
    config=Config(allow_recession_clearance=a.allow_recession_clearance,correlation_seconds=a.correlation_seconds,bounded_pixel_error=a.bounded_pixel_error,bounded_span_fraction=a.bounded_span_fraction)
    assessment_runtime=AssessmentRuntime(a.assessment,config,calibration) if a.assessment else None
    if assessment_runtime:estimator=assessment_runtime.assessor
    elif a.geometry=='calibrated':
        estimator=CalibratedCPAEstimator(calibration,config)
    else:estimator={'log':ThreatEstimator,'projective':ProjectiveEstimator,'bounded':BoundedProjectiveEstimator}[a.geometry](config)
    demo_cue=None
    renderer_type=Renderer
    if a.demo_showcase:
        from .demo_cue import DemoEncounterCue
        demo_cue=DemoEncounterCue()
        if not a.no_render:
            from .demo_render import DemoRenderer
            renderer_type=DemoRenderer
    renderer=renderer_type(a.title,calibration=calibration) if not a.no_render and not a.render_process else None;policy=TemporalPolicy()
    proc=None;timestamp_writer=None
    if not a.no_render and not a.render_process:
        if a.video_timing=='pts':
            timestamp_writer=TimestampWriter(out/'annotated.mp4',fps/a.stride,(renderer.width,renderer.height),encoder=a.encoder)
            session.register(timestamp_writer.abort)
        else:
            opts=[item for key,value in encoder_options(a.encoder).items() for item in ['-'+key,value]]
            cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{renderer.width}x{renderer.height}','-r',str(fps/a.stride),'-i','pipe:0','-an','-c:v',a.encoder,*opts,'-pix_fmt','yuv420p','-movflags','+faststart',str(out/'annotated.mp4')]
            proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=(out/'ffmpeg.log').open('w'))
            session.register(lambda:stop_encoder(proc))
    if a.render_process:exporter=ProcessFrameExporter(out,fps/a.stride,a.title,calibration,a.encoder)
    else:exporter=FrameExporter(renderer,out,writer=timestamp_writer,process=proc,asynchronous=a.async_render) if not a.no_render else None
    if exporter:session.register(exporter.abort)
    stages=defaultdict(list);counts=Counter();demo_counts=Counter();frame_idx=0;processed=0;decode_accum_ms=0.;start=time.perf_counter()
    sample_wall_tick=start
    with (out/'decisions.jsonl').open('w',encoding='utf-8') as trace, (out/'detections.jsonl').open('w',encoding='utf-8') as dettrace:
        while True:
            read_start=time.perf_counter();ok,frame=cap.read();read_ms=(time.perf_counter()-read_start)*1000
            decode_accum_ms+=read_ms
            if not ok:break
            idx=frame_idx;frame_idx+=1
            source_pts_ms=cap.get(cv2.CAP_PROP_POS_MSEC)
            t=clock.update(idx,source_pts_ms)
            if a.sky_mask and abs(t-idx/fps)>1e-4:raise ValueError('Sky-mask workflow requires CFR source timing; VFR mask resampling is not validated')
            if idx%a.stride:continue
            if a.max_frames and processed>=a.max_frames:
                output_end=t;break
            sample_tick=time.perf_counter();tick=sample_tick;predictions=[]
            if bridge:
                predictions=bridge.predict(frame,tracker.tracks,t)
                stages['appearance_predict_ms'].append((time.perf_counter()-tick)*1000)
                stages['appearance_network_calls'].append(bridge.last_network_calls)
            tick=time.perf_counter();detector_ran=False
            if cache is not None:
                entry=cache.get(idx,{})
                cache_misses+=int(idx not in cache)
                validate_cache_entry(entry,idx,t)
                if entry.get('detector_variant',a.detector_variant)!=a.detector_variant:raise ValueError('Detection row variant differs from selected detector')
                rawboxes=entry.get('detections',[]);boxes=entry.get('observations',rawboxes)
                predictions=entry.get('appearance_predictions',[]);detector_ran=entry.get('detector_ran',True)
            else:
                detector_ran=processed%a.detect_every==0
                rawboxes=detector(frame) if detector_ran else []
                if bridge:
                    from .appearance import combine_observations
                    boxes=combine_observations(rawboxes,predictions)
                else:boxes=rawboxes
            stages['detector_ms'].append((time.perf_counter()-tick)*1000)
            validate_detection_classes(rawboxes,a.detector_variant)
            if detector:stages['network_calls'].append(detector.last_network_calls if detector_ran else 0)
            if sky:
                tick=time.perf_counter();rawboxes=sky.augment(detector,frame,rawboxes,idx,processed)
                stages['sky_proposal_ms'].append((time.perf_counter()-tick)*1000)
                stages['proposal_network_calls'].append(sky.last_calls)
                stages['network_calls'][-1]+=sky.last_calls
                detector_ran=detector_ran or sky.last_calls>0
                if bridge:
                    from .appearance import combine_observations
                    boxes=combine_observations(rawboxes,predictions)
                else:boxes=rawboxes
            dettrace.write(json.dumps({'frame':idx,'time':t,'detections':rawboxes,'observations':boxes,'appearance_predictions':predictions,'detector_ran':detector_ran})+'\n')
            tick=time.perf_counter();transform,cam=camera.update(frame,t,boxes)
            if cam.get('evidence_reset'):
                tracker.reset_evidence()
                # Published research methods use raw image coordinates; changing
                # the background stabilization gauge must not move their anchor.
                if not assessment_runtime:policy.pending.clear()
            stages['camera_ms'].append((time.perf_counter()-tick)*1000)
            tick=time.perf_counter();tracks=tracker.update(boxes,t,transform,image_size=(frame.shape[1],frame.shape[0]))
            stages['tracker_ms'].append((time.perf_counter()-tick)*1000)
            if assessment_runtime:assessment_runtime.retain(tracks)
            if bridge:
                tick=time.perf_counter();bridge.refresh(frame,tracks,t)
                stages['appearance_correct_ms'].append((time.perf_counter()-tick)*1000)
            tick=time.perf_counter();objects=[]
            for tr in tracks:
                if tr.hits<3:
                    if assessment_runtime:assessment_runtime.assess(tr,t,cam,(frame.shape[1],frame.shape[0]))
                    continue
                if assessment_runtime:
                    estimate=assessment_runtime.assess(tr,t,cam,(frame.shape[1],frame.shape[0]))
                elif a.detector_scale_only and a.geometry=='log':
                    estimate=estimator.assess(tr.history,t,cam['valid'],t<tr.ambiguous_until,scale_history=tr.scale_history)
                elif a.detector_scale_only:
                    estimate=estimator.assess(tr.scale_history,t,cam['valid'],t<tr.ambiguous_until)
                else:estimate=estimator.assess(tr.history,t,cam['valid'],t<tr.ambiguous_until)
                estimate['clipped_observation']=bool(tr.clipped)
                if tr.clipped and cam['valid'] and t-tr.last<=.25:estimate.update(state='uncertain',reason='clipped')
                if not assessment_runtime:estimate=apply_camera_scope(estimate,cam,a.allow_similarity_pass)
                if a.policy=='confirmed':estimate=policy.apply(tr.id,t,estimate)
                if assessment_runtime:
                    result=estimate['research']
                    result['candidate_decision']=result['decision']
                    if result['decision']!='unobserved':result['decision']=STATES[estimate['state']]
                    result['confirmation_elapsed_s']=estimate.get('confirmation_elapsed_s')
                    result['confirmation_required_s']=estimate.get('confirmation_required_s')
                    if estimate['reason']!='|'.join(result['reason_codes']):result['reason_codes']=[estimate['reason'],*result['reason_codes']]
                    estimate['reason_text']='; '.join(result['reason_codes'])
                counts[estimate['state']]+=1
                objects.append({'id':tr.id,'box':tr.box,'class_id':tr.class_id,'class':model_contract['semantic_taxonomy'][tr.class_id],'class_display':'БПЛА самолётного типа' if a.detector_variant=='B' else None,'source':tr.source,
                                'confidence':tr.confidence,'last_seen':tr.last,'assessment':estimate,'raw_history':list(tr.raw_history)})
                if demo_cue:
                    objects[-1]['demo_observation']={'ambiguous':bool(t<tr.ambiguous_until),'clipped':bool(tr.clipped),
                        'hits':tr.hits,'measured':bool(tr.last==t and tr.source in ('detector','zoom'))}
            advice=advisory(objects,image_width=frame.shape[1])
            if demo_cue:
                advice['demo']=demo_cue.update(objects,t,cam,(frame.shape[1],frame.shape[0]))
                demo_counts.update(target['state'] for target in advice['demo']['targets'])
            stages['threat_ms'].append((time.perf_counter()-tick)*1000)
            if exporter:exporter.submit(frame,objects,t,cam,advice,preview=processed==0 or processed==min(150,source_frames//2))
            for o in objects:o.pop('raw_history')
            trace.write(json.dumps({'frame':idx,'time':t,'source_pts_ms':source_pts_ms,'objects':objects,'camera':cam,'advisory':advice,'detector_ran':detector_ran},ensure_ascii=False,allow_nan=False)+'\n')
            sample_end=time.perf_counter()
            stages['end_to_end_sample_ms'].append((sample_end-sample_wall_tick)*1000)
            sample_wall_tick=sample_end
            stages['decode_ms'].append(decode_accum_ms);decode_accum_ms=0.;processed+=1
            if processed%100==0:print(f'{processed} frames, {processed/(time.perf_counter()-start):.2f} FPS, {dict(counts)}',flush=True)
    cap.release()
    if sky:sky.close()
    if exporter:
        exporter.finish(output_end)
        stages.update(exporter.stages)
    if proc:
        proc.stdin.close()
        if proc.wait():raise RuntimeError('Video encoding failed, see ffmpeg.log')
    elapsed=time.perf_counter()-start
    summaries={k:{'mean':float(np.mean(v)),'p50':float(np.median(v)),'p95':float(np.percentile(v,95)),'p99':float(np.percentile(v,99))} for k,v in stages.items()}
    import torch
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'args':vars(a),'config':asdict(estimator.config),'policy_parameters':{'warning_seconds':policy.warning_seconds,'nonthreat_seconds':policy.nonthreat_seconds},'source_hashes':source_hashes,'frames':processed,'elapsed_s':elapsed,'processing_fps':processed/elapsed,'source_fps':fps,'output_fps':fps/a.stride,'startup_and_warmup_s':startup,'states_track_frames':dict(counts),'stages':summaries,'providers':providers,'hardware':torch.cuda.get_device_name(0),'cv2':cv2.__version__,'cv2_path':cv2.__file__,'cuda_devices':cv2.cuda.getCudaEnabledDeviceCount(),'resolution':list(first.shape[:2][::-1]),'batch':1,'weights_sha256':hashlib.sha256(Path(a.weights).read_bytes()).hexdigest() if detector else None,'caveat':'offline throughput incl decode, stages, trace and encode flush; not certified streaming latency'}
    if bridge:report['appearance_metadata']=bridge.metadata
    if calibration:report['camera_calibration']=asdict(calibration)
    report.update(input_sha256=input_hash,input_hash_s=input_hash_s,replay_identity=replay_identity,
                  cache_missing_sample_frames=cache_misses,decode_stage_meaning='all decoded frames since previous processed sample, including stride-skipped frames')
    report['replay_provenance']=replay_provenance
    report['detector_contract']=model_contract
    report['assessment_backend']=a.assessment or a.geometry
    if demo_cue:
        report['demo_showcase']={'enabled':True,'architecture':'YOLO detector -> original custom tracker -> causal short-history image-space cue -> illustrative separation -> showcase renderer',
            'presentation_states_track_frames':dict(demo_counts),'configuration':getattr(demo_cue,'contract',None),
            'image_coordinate_contract':'original source pixels and normalized image coordinates; no calibrated FOV, range or physical flight guidance',
            'GT_available_to_normal_inference':False,'output_presentation_resolution':[1920,1080]}
    report['assessment_scope']='Stage 2 explicit research comparator, raw measured detector extents; no verified camera de-rotation' if assessment_runtime else 'Original production geometry baseline'
    report['sample_latency_scope']='Sequential decode since prior sample through assessment, trace and render submission; asynchronous encoder drain excluded from percentiles, included in elapsed_s'
    report['cold_start']={'model_load_and_provenance_s':model_load_s,'warmup_update_ms':warmup_update_ms,
        'first_inference_ms':warmup_update_ms[0],'warmup_updates':5} if detector else None
    report['network_calls_total']=int(sum(stages.get('network_calls',[])))
    report['sampled_source_fps']=fps/a.stride
    report['sky_mask_provenance']=sky.provenance if sky else None
    report['live_detector_threshold']=a.conf if detector else None
    report['timestamp_mode']=clock.mode
    report['source_timestamp_origin_ms']=clock.origin*1000 if clock.origin is not None else None
    report['source_timestamp_origin_s']=clock.origin
    report['source_media_metadata']=source_metadata
    report['source_duration_method']=source_duration_method
    report['output_end_time_s']=output_end
    report['output_timestamp_quantization_s']=1/90000 if not a.no_render and a.video_timing=='pts' else None
    report['output_rate_meaning']='Nominal encoder rate; actual presentation intervals are preserved in PTS mode'
    report['stage_overlap']=bool(a.async_render or a.render_process)
    report['render_backend']='disabled' if a.no_render else 'process' if a.render_process else 'thread' if a.async_render else 'serial'
    report['render_worker_metadata']=getattr(exporter,'metadata',None)
    report['max_render_export_pending_frames']=exporter.max_pending if a.async_render or a.render_process else 0
    report['whole_run_wall_s']=time.perf_counter()-pipeline_start
    report['processed_frames_per_whole_run_s']=processed/report['whole_run_wall_s']
    (out/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

def main():
    a=parse_args()
    with RunSession(a.output,{'args':vars(a)}) as session:execute(a,session)


if __name__=='__main__':main()
