"""Sequential fresh-process RTX 4060 measurement of the actual integrated CLI.

This script opens original RGB videos, never an observation cache. It records
all commands and per-run results, then decodes every exported H.264 frame and
compares presentation timestamps with independently audited source PTS.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def video_qa(path,trace,source_times):
    import av
    import numpy as np
    expected=[row['time'] for row in trace]
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        codec=stream.codec_context.name
        decoded=[];shape=None
        for frame in container.decode(stream):
            decoded.append(float(frame.pts*frame.time_base))
            shape=[frame.width,frame.height]
            if len(decoded)==min(101,len(expected)):
                frame.to_image().save(path.with_name('reviewed_frame.jpg'),quality=95)
        duration=float(stream.duration*stream.time_base) if stream.duration is not None else None
    if len(decoded)!=len(expected):raise ValueError('Export frame count differs from trace')
    delta=max(abs(a-b) for a,b in zip(expected,decoded))
    source_delta=max(abs(row['time']-source_times[row['frame']]) for row in trace)
    if delta>1/90000+1e-9 or source_delta>1e-4:raise ValueError('Original PTS not preserved')
    if any(b<=a for a,b in zip(decoded,decoded[1:])):raise ValueError('Non-increasing export PTS')
    return {'decoded_frames':len(decoded),'codec':codec,'resolution':shape,
        'duration_s':duration,'max_output_pts_error_s':delta,
        'max_source_pts_error_s':source_delta,'all_frames_decoded':True}


def run(args):
    audit=json.loads((ROOT/'reports/stage2_dataset_audit.json').read_text())
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    report_path=ROOT/'reports/stage2_runtime.json'
    report={'state':'running','started_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Actual original-video decode + live neural detector + custom predicted tracker + assessment; serial render/NVENC when enabled',
        'tracker':'original custom motion-gated tracker, GT identities unavailable to inference',
        'selection':'No threat winner promoted; legacy log remains the default; Lai/Lin and AirTrack are explicit research selections',
        'gpu_concurrency':'No other model benchmark permitted during these sequential runs; parent coordinated the single GPU',
        'jetson_orin_nano_super':'unmeasured: no actual device run',
        'runs':[]}
    configs=[]
    for clip in args.clips.split(','):
        for variant in ['A','B']:
            for render in [False,True]:configs.append((clip,variant,None,render))
    if args.research:
        for method in ['lai_lin','airtrack_cues']:
            for variant in ['A','B']:configs.append(('Clip_032',variant,method,True))
    configs=[(*config,False) for config in configs]
    if args.async_controls:
        configs.extend((clip,'A',None,True,True) for clip in args.clips.split(','))
    for clip,variant,method,render,async_render in configs:
        label=f'{clip}_{variant}_{method or "legacy_log"}_{"nvenc_async" if async_render else "nvenc" if render else "no_render"}'
        target=output/label
        cmd=[sys.executable,'-m','airavoid',audit['clips'][clip]['video_path'],
            '--detector-variant',variant,'--output',str(target),
            '--max-frames',str(args.frames),'--camera','auto','--stride','1']
        if method:cmd+=['--assessment',method]
        if render:cmd+=['--encoder','h264_nvenc']
        else:cmd+=['--no-render']
        if async_render:cmd+=['--async-render']
        if (target/'benchmark.json').exists() and not args.resume:
            raise FileExistsError(f'{target} already completed; choose another output or --resume')
        started=time.perf_counter()
        resumed=(target/'benchmark.json').exists()
        if not resumed:
            log=output/(label+'.log')
            with log.open('w',encoding='utf-8') as sink:
                result=subprocess.run(cmd,cwd=ROOT,stdout=sink,stderr=subprocess.STDOUT)
            if result.returncode:
                report.update(state='failed',failed_command=cmd,log=str(log))
                report_path.write_text(json.dumps(report,indent=2))
                raise RuntimeError(f'{label} failed; inspect {log}')
        wall=time.perf_counter()-started
        benchmark=json.loads((target/'benchmark.json').read_text())
        if benchmark['input_sha256']!=audit['clips'][clip]['video_sha256']:
            raise ValueError('Original source changed since independent dataset audit')
        trace=[json.loads(line) for line in (target/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
        metadata=audit['clips'][clip]
        pts_file=json.loads((ROOT/metadata['timestamps_file']).read_text())
        if isinstance(pts_file,dict):
            # Both historical manifest shapes are explicit, never invented CFR.
            source_times=pts_file.get('times_s',pts_file.get('pts_s',pts_file.get('timestamps_s')))
        else:source_times=pts_file
        if source_times is None:raise ValueError('Unknown independently audited PTS schema')
        qa=video_qa(target/'annotated.mp4',trace,source_times) if render else None
        if qa:
            qa['expected_duration_s']=benchmark['output_end_time_s']
            qa['duration_error_s']=abs(qa['duration_s']-qa['expected_duration_s'])
            if qa['duration_error_s']>1/90000+1e-9:raise ValueError('Export duration differs from source interval')
        from airavoid.provenance import file_sha256
        record={'label':label,'clip':clip,'detector':variant,'method':method or 'legacy_log',
            'encoded':render,'command':cmd,'worker_wall_including_process_start_s':None if resumed else wall,
            'resumed_existing_result':resumed,
            'benchmark_path':str((target/'benchmark.json').relative_to(ROOT)),
            'benchmark_sha256':file_sha256(target/'benchmark.json'),
            'decisions_sha256':file_sha256(target/'decisions.jsonl'),
            'detections_sha256':file_sha256(target/'detections.jsonl'),
            'benchmark':benchmark,'video_qa':qa,
            'annotated_video_local_path':str(target/'annotated.mp4') if render else None}
        report['runs'].append(record)
        report_path.write_text(json.dumps(report,indent=2))
        print(label,f'{benchmark["processing_fps"]:.3f} FPS',flush=True)
    report.update(state='complete',completed_utc=datetime.now(timezone.utc).isoformat())
    example=output/'Clip_032_B_lai_lin_nvenc/reviewed_frame.jpg'
    if example.exists():
        preview=ROOT/'reports/stage2_runtime_example.jpg'
        preview.write_bytes(example.read_bytes())
        report['reviewed_example']={'path':str(preview.relative_to(ROOT)),
            'source':'actual encoded Clip_032_B_lai_lin_nvenc frame100',
            'sha256':file_sha256(preview)}
    report_path.write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips',default='Clip_032,Clip_001')
    parser.add_argument('--frames',type=int,default=150)
    parser.add_argument('--output',default=str(ROOT/'artifacts/stage2/live_runtime'))
    parser.add_argument('--research',action='store_true')
    parser.add_argument('--async-controls',action='store_true',help='Also measure existing bounded A render thread on each source clip')
    parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())
