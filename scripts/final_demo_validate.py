"""Validate and extract actual encoded frames from a final live detector showcase.

No detector, tracker or GT loader is run here. Every source/output frame is decoded
on CPU; screenshots are unmodified frames extracted from the encoded deliverable.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import av

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def demo_for(row):
    value=row.get('advisory',{}).get('demo')
    if not isinstance(value,dict):raise ValueError('Missing actual advisory.demo trace record')
    return value


def possible(target):return target.get('state')=='POSSIBLE CONVERGENCE' or target.get('state_id')=='possible_convergence'


def choose_frames(rows):
    """Select from actual measured trace evidence, never a preselected timestamp."""
    stable=[];contrast=[];segments={};completed=[];illustrations=[];warnings=[]
    for index,row in enumerate(rows):
        demo=demo_for(row);targets=demo.get('targets',[]);direction=demo.get('direction',{})
        monitoring=[t for t in targets if t.get('state')=='MONITORING' or t.get('state_id')=='monitoring']
        support=sum(min(int(t.get('samples',0)),100)+min(len(t.get('measured_trail',[])),100) for t in targets)
        stable.append((1000*len(monitoring)+support-abs(index-len(rows)*.3)*.01,index))
        valid_vector=direction.get('status')=='resolved' and isinstance(direction.get('vector'),list) and len(direction['vector'])==2 and all(math.isfinite(float(v)) for v in direction['vector']) and math.hypot(*direction['vector'])>1e-9
        if direction.get('status')=='resolved':
            if not valid_vector or direction.get('meaning')!='illustrative_image_plane_cue_not_safe_maneuver':
                raise ValueError('Resolved arrow lacks its finite illustrative-image-plane contract')
        eligible=[t for t in targets if possible(t) and (demo.get('priority_id') is None or t['id']==demo['priority_id'])] if valid_vector else []
        priority=[t for t in targets if demo.get('priority_id') is None or t['id']==demo['priority_id']]
        for target in priority:
            support=float(target.get('window_s') or 0.)*10+min(int(target.get('samples',0)),100)
            if possible(target):warnings.append((support+float(target.get('growth_per_s') or 0.),index,int(target['id'])))
            if valid_vector:illustrations.append((support,index,int(target['id'])))
        active=set()
        for target in eligible:
            tid=int(target['id']);active.add(tid);prior=segments.get(tid)
            strength=float(target.get('growth_per_s') or 0.)
            if prior is None or index!=prior['last_index']+1:
                if prior:completed.append(prior)
                prior={'id':tid,'start_index':index,'last_index':index,'best_index':index,'strength':strength}
            else:
                prior['last_index']=index
                if strength>prior['strength']:prior.update(best_index=index,strength=strength)
            segments[tid]=prior
        for tid in list(segments):
            if tid not in active:completed.append(segments.pop(tid))
        strongest=max((abs(float(t.get('growth_per_s') or 0.))+abs(float(t.get('drift_normalized_per_s') or 0.)) for t in targets),default=0.)
        contrast.append((100*len(targets)+strongest+index/max(1,len(rows)),index))
    completed.extend(segments.values())
    stable_index=max(stable)[1]
    if completed:
        best=max(completed,key=lambda x:(rows[x['last_index']]['time']-rows[x['start_index']]['time'],x['strength']))
        cue_index=best['best_index'];cue={'observed_possible_convergence_with_resolved_arrow':True,
            'observed_resolved_illustrative_arrow':True,
            'target_id':best['id'],'sustained_trace_interval_s':[rows[best['start_index']]['time'],rows[best['last_index']]['time']],
            'selection':'strongest sample within longest consecutive actual convergence/arrow interval'}
    elif warnings:
        _,cue_index,tid=max(warnings)
        cue={'observed_possible_convergence_with_resolved_arrow':False,'observed_resolved_illustrative_arrow':False,
             'target_id':tid,'selection':'actual convergence cue without a resolved arrow; no direction invented',
             'caption':'Observed convergence cue; direction remains unresolved.'}
    elif illustrations:
        _,cue_index,tid=max(illustrations)
        cue={'observed_possible_convergence_with_resolved_arrow':False,'observed_resolved_illustrative_arrow':True,
             'target_id':tid,'selection':'trace-supported illustrative image separation during monitoring, without a warning',
             'caption':'Illustrative direction during MONITORING; no convergence warning observed.'}
    else:
        alternatives=[x for x in contrast if abs(x[1]-stable_index)>max(1,len(rows)//5)] or contrast
        cue_index=max(alternatives)[1];cue={'observed_possible_convergence_with_resolved_arrow':False,
            'observed_resolved_illustrative_arrow':False,
            'selection':'contrasting tracked frame; no observed convergence with resolved arrow in this deliverable',
            'caption':'No observed warning/arrow; this is a contrasting trace-supported frame, not an invented encounter.'}
    if stable_index==cue_index:
        distinct=[x for x in stable if abs(x[1]-cue_index)>max(1,len(rows)//10)] or [x for x in stable if x[1]!=cue_index]
        if distinct:stable_index=max(distinct)[1]
    return stable_index,cue_index,cue


def validate(args):
    run=Path(args.run).resolve();bench=read_json(run/'benchmark.json');status=read_json(run/'run_status.json')
    bargs=bench['args'];source=Path(args.source or bargs['video']);video=run/'annotated.mp4'
    checks={}
    def require(key,value,explanation):
        checks[key]=bool(value)
        if not value:raise ValueError(key+': '+explanation)
    require('run_completed',status['state']=='complete','run_status is not complete')
    require('showcase_enabled',bool(bargs.get('demo_showcase')),'--demo-showcase must be active')
    variant=args.detector_variant
    require('detector_'+variant,bargs['detector_variant']==variant and bench['detector_contract']['variant']==variant,'actual requested detector variant required')
    providers=bench.get('providers',[])
    require('actual_TensorRT_provider',any('tensorrt' in x.lower() for x in providers) and not any('replay' in x.lower() or 'cache' in x.lower() for x in providers),'actual declared TensorRT provider required')
    require('no_detection_replay',not bargs.get('detections') and bench.get('replay_identity') is None and bench.get('replay_provenance') is None,'replay/cache evidence cannot validate live inference')
    expected_shape=[1,3,800,1440] if variant=='B' else [1,3,416,704]
    require('registered_supplied_P2' if variant=='B' else 'registered_supplied_A',bench['detector_contract'].get('registered_supplied_export') and bench['detector_contract']['input_shape']==expected_shape,'wrong/unregistered supplied model contract')
    expected_taxonomy=['fixed-wing UAV'] if variant=='B' else ['airplane','helicopter','bird','balloon','parachute']
    require('declared_detector_taxonomy',bench['detector_contract']['semantic_taxonomy']==expected_taxonomy,'supplied detector class mapping changed')
    require('encoded_NVENC',not bargs.get('no_render') and bargs.get('encoder')=='h264_nvenc','annotated NVENC run required')
    require('async_export',bargs.get('async_render') and bench.get('render_backend')=='thread','bounded asynchronous renderer required')
    require('source_hash',sha(source)==bench['input_sha256'],'source video differs from benchmark')
    weights=Path(bargs['weights'])
    require('weights_hash',weights.exists() and sha(weights)==bench['weights_sha256']==bench['detector_contract']['weights_sha256'],'actual supplied engine differs')
    rows=[json.loads(line) for line in (run/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    require('trace_frame_count',len(rows)==bench['frames'],'decision trace incomplete')
    calls_valid=bench['network_calls_total']==len(rows) if variant=='B' else bench['network_calls_total']>=len(rows)
    require('live_inference_each_frame',calls_valid and all(row.get('detector_ran') for row in rows),'B requires one actual call each frame; tiled A can require several')
    require('original_cadence',bargs['stride']==1 and bargs.get('detect_every',1)==1,'source cadence changed')
    require('measured_full_pipeline_FPS',math.isfinite(bench['processing_fps']) and bench['processing_fps']>0,'actual positive runtime measurement required; no speed acceptance threshold is imposed')
    current_hashes={}
    for name,expected in bench.get('source_hashes',{}).items():
        p=ROOT/'airavoid'/name
        require('source_code_'+name,p.exists() and sha(p)==expected,'source code changed after execution')
        current_hashes[name]=expected
    source_pts=[]
    with av.open(str(source)) as container:
        ss=container.streams.video[0];source_size=[ss.width,ss.height];source_codec=ss.codec_context.name
        for frame in container.decode(ss):
            if frame.pts is None:raise ValueError('Source frame lacks PTS')
            source_pts.append(float(frame.pts*frame.time_base))
    require('source_PTS_monotonic',all(b>a for a,b in zip(source_pts,source_pts[1:])),'source chronology is invalid')
    require('all_source_frames_processed',args.pilot or len(rows)==len(source_pts),'final run must process the entire chosen clip')
    require('source_frame_indices',all(row['frame']==index for index,row in enumerate(rows)),'source frames missing/reordered')
    origin=source_pts[0]
    source_errors=[abs(row['time']-(source_pts[row['frame']]-origin)) for row in rows]
    require('trace_original_PTS',max(source_errors,default=math.inf)<=1e-6,'trace does not preserve original source PTS')
    source_ms_errors=[abs(row['source_pts_ms']/1000-source_pts[row['frame']]) for row in rows]
    require('trace_acquisition_PTS',max(source_ms_errors,default=math.inf)<=1e-6,'raw acquisition timestamps changed')
    stable_index,cue_index,cue=choose_frames(rows);selected={stable_index:'stable_tracking.png',cue_index:'encounter_cue.png'}
    output_pts=[];screenshots={}
    with av.open(str(video)) as container:
        vs=container.streams.video[0];out_size=[vs.width,vs.height];out_codec=vs.codec_context.name
        require('output_1920x1080_or_larger',vs.width>=1920 and vs.height>=1080,'output canvas is below requested resolution')
        require('output_H264',out_codec=='h264','expected H.264 output')
        for index,frame in enumerate(container.decode(vs)):
            if frame.pts is None:raise ValueError('Encoded frame lacks PTS')
            output_pts.append(float(frame.pts*frame.time_base))
            if index in selected:
                target=run/selected[index];frame.to_image().save(target)
                screenshots[selected[index]]={'frame':index,'time_s':rows[index]['time'],'resolution':[frame.width,frame.height],
                    'path':str(target),'sha256':sha(target),'source':'unmodified full-resolution decoded annotated H.264 frame',
                    'demo_trace':demo_for(rows[index])}
        duration=float(vs.duration*vs.time_base) if vs.duration is not None else None
    require('every_output_frame_decoded',len(output_pts)==len(rows),'encoded frame count differs')
    require('output_PTS_monotonic',all(b>a for a,b in zip(output_pts,output_pts[1:])),'encoded chronology invalid')
    output_errors=[abs(p-row['time']) for p,row in zip(output_pts,rows)]
    require('output_original_PTS',max(output_errors,default=math.inf)<=1/90000+1e-9,'encoder changed timestamps')
    require('output_duration',duration is not None and abs(duration-bench['output_end_time_s'])<=1/90000+1e-9,'encoded end timestamp differs')
    require('final_20_to_45_seconds',args.pilot or 20<=duration<=45,'final export must last 20–45 seconds')
    require('two_distinct_screenshots',len(screenshots)==2,'could not select distinct trace-supported moments')
    states=Counter(target.get('state','unknown') for row in rows for target in demo_for(row).get('targets',[]))
    return {'state':'complete','validated_utc':datetime.now(timezone.utc).isoformat(),'acceptance':'pilot_not_final' if args.pilot else 'final_live_demo',
        'run':str(run),'checks':checks,'source':{'path':str(source),'sha256':bench['input_sha256'],'codec':source_codec,'resolution':source_size,'decoded_frames':len(source_pts)},
        'encoded_video':{'path':str(video),'sha256':sha(video),'codec':out_codec,'resolution':out_size,'duration_s':duration,'all_frames_decoded':True,'frames':len(output_pts)},
        'max_source_trace_pts_error_s':max(source_errors),'max_encoded_pts_error_s':max(output_errors),
        'processing_fps':bench['processing_fps'],'whole_run_fps':bench.get('processed_frames_per_whole_run_s'),
        'runtime':{'hardware':bench.get('hardware'),'providers':providers,'cv2':bench.get('cv2'),
                   'cv2_path':bench.get('cv2_path'),'cuda_devices':bench.get('cuda_devices'),
                   'cold_start':bench.get('cold_start'),'sample_latency_scope':bench.get('sample_latency_scope'),
                   'end_to_end_sample_ms':bench.get('stages',{}).get('end_to_end_sample_ms')},
        'actual_detector_contract':bench['detector_contract'],
        'benchmark_sha256':sha(run/'benchmark.json'),'decisions_sha256':sha(run/'decisions.jsonl'),
        'detections_sha256':sha(run/'detections.jsonl'),'runtime_source_hashes':current_hashes,
        'demo_target_sample_states':states,'encounter_screenshot_evidence':cue,'screenshots':screenshots,
        'claim_scope':'Observed tracking and conditional image-plane cue only; no GT supplied to inference, no physical CPA/avoidance guarantee.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(ROOT/'demo/final_showcase'));p.add_argument('--source')
    p.add_argument('--detector-variant',choices=['A','B'],default='B',help='Default validates the primary fixed-wing P2 run; A validates an explicit original five-class live control')
    p.add_argument('--output',default=str(ROOT/'reports/final_demo_validation.json'));p.add_argument('--pilot',action='store_true')
    args=p.parse_args();output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    try:result=validate(args)
    except Exception as exc:
        output.write_text(json.dumps({'state':'failed','error':str(exc),'run':str(Path(args.run).resolve())},indent=2));raise
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps({k:v for k,v in result.items() if k not in ['screenshots','runtime_source_hashes','checks']},indent=2,ensure_ascii=False))
