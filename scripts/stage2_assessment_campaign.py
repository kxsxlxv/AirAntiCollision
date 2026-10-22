"""Shared CPU-only method matrix on frozen NPS traces, with explicit oracle tiers."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from stage2_assess_trace import replay
from stage2_threat_score import score
from stage2_oracle import build
from airavoid.assessment import METHODS
from airavoid.provenance import file_sha256


def work(job):
    trace,output,clip,width,height,camera=job
    output=Path(output);summary=output.with_suffix('.summary.json')
    source_sha=file_sha256(trace);code_sha=file_sha256(ROOT/'airavoid/assessment.py')
    replay_sha=file_sha256(ROOT/'scripts/stage2_assess_trace.py')
    if not (output.exists() and summary.exists() and
            (old:=json.loads(summary.read_text())).get('source_trace_sha256')==source_sha and
            old.get('assessment_source_sha256')==code_sha and old.get('replay_source_sha256')==replay_sha):
        replay(trace,output,width,height,METHODS,'auto',camera)
    result=score(output,clip)
    dest=output.with_suffix('.score.json');dest.write_text(json.dumps(result,indent=2))
    return str(dest),result


def aggregate(results,splitclips):
    bymethod=defaultdict(lambda:{'states':Counter(),'raw_candidate_states':Counter(),'reason_codes':Counter(),
                                  'all_predicted_track_sample_states':Counter(),
                                  'all_predicted_warning_episodes':0,'target_frame_opportunities':0})
    totalduration=0.;sequences=0
    for r in results:
        if r['sequence'] not in splitclips:continue
        totalduration+=r['camera_duration_s'];sequences+=1
        for m,d in r['methods'].items():
            for key in ['states','raw_candidate_states','reason_codes','all_predicted_track_sample_states']:bymethod[m][key].update(d[key])
            for key in ['all_predicted_warning_episodes','target_frame_opportunities']:bymethod[m][key]+=d[key]
    for m,d in bymethod.items():
        n=max(1,d['target_frame_opportunities'])
        d['decisive_coverage']=(d['states']['potential_conflict']+d['states']['apparent_nonconflict'])/n
        d['untracked_fraction']=d['states']['untracked']/n
        d['all_predicted_warning_episodes_per_camera_hour']=d['all_predicted_warning_episodes']*3600/max(1e-12,totalduration)
    return {'sequences':sequences,'camera_duration_s':totalduration,'methods':bymethod}


def main(args):
    protocol=json.loads((ROOT/'reports/stage2_protocol.json').read_text());audit=json.loads((ROOT/'reports/stage2_dataset_audit.json').read_text())
    clips=protocol['development']+protocol['evaluation'];jobs=[];conditions={};oracles=[]
    deadline=time.monotonic()+args.wait_complete_seconds
    while True:
        incomplete=[]
        for detector in args.detectors.split(','):
            for tracker in args.trackers.split(','):
                benchmark=ROOT/f'artifacts/stage2/tracks/{detector}/{tracker}/s1/benchmark.json'
                if not benchmark.exists() or set(json.loads(benchmark.read_text())['clips'])!=set(clips):incomplete.append(f'{detector}/{tracker}')
        if not incomplete:break
        if time.monotonic()>=deadline:raise ValueError('Incomplete tracker conditions: '+','.join(incomplete))
        print('Waiting for frozen complete tracker traces:',','.join(incomplete),flush=True);time.sleep(5)
    for detector in args.detectors.split(','):
        for tracker in args.trackers.split(','):
            folder=ROOT/f'artifacts/stage2/tracks/{detector}/{tracker}/s1'
            benchmark=json.loads((folder/'benchmark.json').read_text())
            if set(benchmark['clips'])!=set(clips):raise ValueError(f'Incomplete frozen tracker condition {folder}')
            for clip in clips:
                trace=folder/f'{clip}.jsonl'
                if file_sha256(trace)!=benchmark['clips'][clip]['trace_sha256']:raise ValueError('Tracker trace changed')
                for camera in args.camera_models.split(','):
                    condition=f'{detector}/{tracker}/{camera}'
                    output=ROOT/f'artifacts/stage2/assessments/{condition}/{clip}.jsonl.gz'
                    w,h=audit['clips'][clip]['resolution'];jobs.append((trace,output,clip,w,h,camera));conditions[str(output.with_suffix('.score.json'))]=condition
            if args.oracle and tracker=='custom':
                for clip in ['Clip_032','Clip_043']:
                    for truth in ['oracle_identity_only','oracle_box_and_identity']:
                        trace=ROOT/f'artifacts/stage2/oracles/{detector}/{truth}/{clip}.jsonl'
                        oracles.append(build(folder/f'{clip}.jsonl',clip,truth,trace))
                        for camera in args.camera_models.split(','):
                            condition=f'{detector}/{truth}/{camera}'
                            output=ROOT/f'artifacts/stage2/assessments/{condition}/{clip}.jsonl.gz'
                            w,h=audit['clips'][clip]['resolution'];jobs.append((trace,output,clip,w,h,camera));conditions[str(output.with_suffix('.score.json'))]=condition
    status={'state':'running','started_utc':datetime.now(timezone.utc).isoformat(),'planned_replays':len(jobs),'complete':[],
            'methods':list(METHODS),'assessment_source_sha256':file_sha256(ROOT/'airavoid/assessment.py'),
            'oracle_builds':oracles,'runtime_scope':'CPU cached observations only; no detector or neural tracking FPS'}
    statuspath=ROOT/'reports/stage2_assessment_campaign_status.json'
    results=defaultdict(list);start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(work,j):j for j in jobs}
        for future in as_completed(pending):
            path,r=future.result();condition=conditions[path];results[condition].append(r)
            status['complete'].append({'condition':condition,'sequence':r['sequence'],'score_path':path})
            statuspath.write_text(json.dumps(status,indent=2));print('Complete',condition,r['sequence'],len(status['complete']), '/',len(jobs),flush=True)
    allresult={'methods':list(METHODS),'protocol_sha256':file_sha256(ROOT/'reports/stage2_protocol.json'),
               'assessment_source_sha256':status['assessment_source_sha256'],
               'truth_tier':'B, user-verified 2D boxes+identity; no physical CPA/collision truth','conditions':{}}
    for condition,rs in results.items():
        allresult['conditions'][condition]={'per_sequence':{r['sequence']:r for r in rs},
                                          'combined':{s:aggregate(rs,protocol[s]) for s in ['development','evaluation']}}
    (ROOT/'reports/stage2_threat_matrix.json').write_text(json.dumps(allresult,indent=2))
    status.update(state='complete',elapsed_s=time.perf_counter()-start,finished_utc=datetime.now(timezone.utc).isoformat())
    statuspath.write_text(json.dumps(status,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--detectors',default='A,B');p.add_argument('--trackers',default='custom,tracktrack,putr')
    p.add_argument('--camera-models',default='raw,similarity');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--wait-complete-seconds',type=float,default=0)
    p.add_argument('--oracle',action='store_true');main(p.parse_args())
