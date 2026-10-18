"""Finish queued experiments without overlapping GPU workloads."""
from pathlib import Path
import json,time,subprocess,sys
root=Path(__file__).resolve().parents[1];status=root/'reports/mot_followup_status.json'
while not status.exists() or not json.loads(status.read_text()).get('complete'):time.sleep(5)
out=root/'artifacts/mot';results=[]
def invoke(name,args):
    start=time.perf_counter()
    with (out/f'{name}.log').open('w',encoding='utf-8') as log:
        r=subprocess.run([sys.executable,*args],cwd=root,stdout=log,stderr=subprocess.STDOUT)
    result=dict(name=name,exit_code=r.returncode,external_seconds=time.perf_counter()-start)
    results.append(result);print(json.dumps(result),flush=True)
    (root/'reports/mot_finalize_status.json').write_text(json.dumps(dict(complete=False,runs=results),indent=2))
for method in ['diffmot','trackssm']:invoke('motion_'+method,['scripts/mot_motion_probe.py',method])
for method in ['tracktrack','botsort','ocsort','diffmot','putr','trackssm']:
    invoke('render_live_'+method,['scripts/mot_benchmark.py','--method',method,'--sequences','M2,formation',
        '--max-frames','300','--live','--render','--output',f'artifacts/mot/render_live/{method}'])
for method in ['diffmot','trackssm']:
    invoke('motion_only_'+method,['scripts/mot_benchmark.py','--method',method,'--sequences','sw02,sw06,sw28',
        '--no-appearance','--output',f'artifacts/mot/motion_only/{method}'])
    invoke('motion_only_metrics_'+method,['scripts/mot_metrics.py',f'artifacts/mot/motion_only/{method}'])
(root/'reports/mot_finalize_status.json').write_text(json.dumps(dict(complete=True,runs=results),indent=2))
