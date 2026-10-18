"""Wait for the sequential QUT campaign, then run Swarm and live benchmarks."""
from pathlib import Path
import sys,json,subprocess,time
root=Path(__file__).resolve().parents[1]
status=root/'reports/mot_campaign_status.json'
while not status.exists() or not json.loads(status.read_text()).get('complete'):
    time.sleep(5)
methods=['custom','ocsort','tracktrack','botsort','diffmot','putr','trackssm']
out=root/'artifacts/mot';results=[]
def invoke(name,command):
    start=time.perf_counter()
    with (out/f'{name}.log').open('w',encoding='utf-8') as log:
        r=subprocess.run([sys.executable,*command],cwd=root,stdout=log,stderr=subprocess.STDOUT)
    result=dict(name=name,exit_code=r.returncode,external_seconds=time.perf_counter()-start)
    results.append(result);print(json.dumps(result),flush=True)
    (root/'reports/mot_followup_status.json').write_text(json.dumps(dict(complete=False,runs=results),indent=2))
    return r.returncode
if invoke('swarm_detector',['scripts/mot_swarm_detections.py']):raise RuntimeError('Swarm detection failed')
for method in methods:
    folder=f'artifacts/mot/swarm/{method}'
    rc=invoke('swarm_'+method,['scripts/mot_benchmark.py','--method',method,'--sequences','sw02,sw06,sw28','--output',folder])
    if not rc:invoke('metrics_'+method,['scripts/mot_metrics.py',folder])
for method in methods:
    invoke('live_'+method,['scripts/mot_benchmark.py','--method',method,'--sequences','M2,formation','--live',
        '--max-frames','300','--output',f'artifacts/mot/live/{method}'])
(root/'reports/mot_followup_status.json').write_text(json.dumps(dict(complete=True,runs=results),indent=2))
