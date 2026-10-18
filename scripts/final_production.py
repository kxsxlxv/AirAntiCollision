"""Sequential final live-neural videos and matched sky-region control."""
import json
from pathlib import Path
import subprocess
import sys
from scripts.evaluate import evaluate


def run(name,command,output):
    root=Path('reports/final_production');root.mkdir(parents=True,exist_ok=True)
    with (root/(name+'.log')).open('w',encoding='utf-8') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    benchmark=json.loads((Path(output)/'benchmark.json').read_text())
    summary={'name':name,'output':str(output),'frames':benchmark['frames'],'processing_fps':benchmark['processing_fps'],
             'whole_run_fps':benchmark['processed_frames_per_whole_run_s'],'states':benchmark['states_track_frames']}
    path=root/(name+'.json');path.write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    return summary


def base_videos():
    python=sys.executable;results=[]
    for name,source,camera in [('M1','data/qut/M1.mp4','fixed'),('M2','data/qut/M2.mp4','fixed'),
            ('S2','data/qut/S2.mp4','fixed'),('formation','data/formation_native.mp4','auto'),
            ('helicopter','data/helicopter_252.mp4','auto')]:
        output=Path('demo')/(name+'_final')
        cmd=[python,'-m','airavoid',source,'--camera',camera,'--stride','2','--encoder','h264_nvenc',
             '--render-process','--output',str(output)]
        results.append(run(name,cmd,output))
        if name in ['M1','M2','S2']:
            evaluation=evaluate(output/'decisions.jsonl',f'data/qut/{name}.gt.json')
            (output/'evaluation.json').write_text(json.dumps(evaluation,indent=2))
    # Two further clean repeats use the exact current implementation and settings.
    for index in [2,3]:
        output=Path('artifacts/final_benchmark')/f'M2_repeat{index}'
        cmd=[python,'-m','airavoid','data/qut/M2.mp4','--camera','fixed','--stride','2',
             '--encoder','h264_nvenc','--render-process','--output',str(output)]
        results.append(run('M2_repeat'+str(index),cmd,output))
    return results


def supplement(results):
    python=sys.executable
    output=Path('demo/S2_sky_final')
    cmd=[python,'-m','scripts.run_sky_demo','data/qut/S2.mp4','--camera','fixed','--stride','2',
         '--encoder','h264_nvenc','--render-process','--output',str(output)]
    root=Path('reports/final_production')
    with (root/'S2_sky_workflow.log').open('w',encoding='utf-8') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    for name,extra,target in [('S2_sky',[],output/'assessment'),
            ('S2_all_region',['--sky-all-control'],Path('artifacts/sky_control/S2_all_region'))]:
        if extra:
            cmd=[python,'-m','airavoid','data/qut/S2.mp4','--camera','fixed','--stride','2','--no-render',
                 '--sky-mask',str(output/'sky_overlay_mask.mkv'),'--output',str(target),*extra]
            results.append(run(name,cmd,target))
        evaluation=evaluate(target/'decisions.jsonl','data/qut/S2.gt.json')
        (target/'evaluation.json').write_text(json.dumps(evaluation,indent=2))
    Path('reports/final_production.json').write_text(json.dumps(results,indent=2))
    # Lower confidence is an experiment, never silently promoted to the default.
    cmd=[python,'-m','scripts.confidence_campaign','--root','artifacts/confidence_floor',
         '--thresholds','.005','.01','.03','.18','--sequences','M1','M2','S2','--stride','2']
    with (root/'confidence_floor.log').open('w',encoding='utf-8') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    cmd=[python,'-m','scripts.render_excerpt','data/qut_full/M6.mp4','artifacts/cadence_fullrate/M6/decisions.jsonl',
         '--start','90','--end','102','--output','demo/M6_failure',
         '--title','M6 • предупреждение по категории other: офлайн-разбор']
    with (root/'M6_failure.log').open('w',encoding='utf-8') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (root/'residual_sensitivity.log').open('w',encoding='utf-8') as log:
        subprocess.run([python,'-m','scripts.residual_sensitivity'],stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--supplement-only',action='store_true');args=parser.parse_args()
    if args.supplement_only:
        names=['M1','M2','S2','formation','helicopter','M2_repeat2','M2_repeat3']
        results=[json.loads(Path('reports/final_production',name+'.json').read_text()) for name in names]
    else:results=base_videos()
    supplement(results)
