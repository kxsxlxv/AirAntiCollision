"""Final fair async-export baselines, visual artifacts, and regression checks."""
from pathlib import Path
import sys,json,time,subprocess,hashlib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/mot'
p=root/'reports/mot_async_repeats.json'
while not p.exists() or not json.loads(p.read_text()).get('complete'):time.sleep(5)
results=[]
def invoke(name,args,report_log=False):
    start=time.perf_counter();log_path=(root/'reports/mot_tests.log') if report_log else out/f'{name}.log'
    with log_path.open('w',encoding='utf-8') as log:r=subprocess.run([sys.executable,*args],cwd=root,stdout=log,stderr=subprocess.STDOUT)
    result=dict(name=name,exit_code=r.returncode,external_seconds=time.perf_counter()-start)
    results.append(result);print(json.dumps(result),flush=True)
    (root/'reports/mot_last_checks.json').write_text(json.dumps(dict(complete=False,runs=results),indent=2))
    return r.returncode
for method in ['botsort','ocsort','diffmot','trackssm']:
    invoke('async_'+method,['scripts/mot_benchmark.py','--method',method,'--sequences','M2,formation',
        '--max-frames','300','--live','--render','--async-export','--output',f'artifacts/mot/async_live/{method}'])
invoke('summary',['scripts/mot_summary.py'])
invoke('comparison_videos',['scripts/mot_render_comparison.py'])
invoke('regression',['-m','unittest','discover','-s','tests','-v'],report_log=True)
invoke('environment',['scripts/check_environment.py'])
(root/'reports/mot_environment.json').write_bytes((root/'reports/environment_final.json').read_bytes())
# The earlier delivery index records the environment check as a file. Record its
# read-only refresh explicitly instead of leaving a silently stale file hash.
index_path=root/'reports/artifact_index.json';index=json.loads(index_path.read_text())
for item in index['reports']:
    if item['path']=='reports/environment_final.json':
        old=item['sha256'];b=(root/item['path']).read_bytes();new=hashlib.sha256(b).hexdigest()
        index.setdefault('followup_updates',[]).append(dict(path=item['path'],previous_sha256=old,new_sha256=new,
            reason='Read-only environment preservation recheck during fixed-YOLO MOT follow-up; build/path/CUDA unchanged'))
        item.update(bytes=len(b),sha256=new)
index_path.write_text(json.dumps(index,indent=2))
(root/'reports/mot_last_checks.json').write_text(json.dumps(dict(complete=True,all_passed=all(x['exit_code']==0 for x in results),runs=results),indent=2))
