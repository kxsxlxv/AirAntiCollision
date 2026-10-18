"""Fresh-process repeated rendered live runs of the two aircraft front-runners."""
from pathlib import Path
import sys,json,subprocess,time,hashlib
root=Path(__file__).resolve().parents[1]
p=root/'reports/mot_finalize_status.json'
while not p.exists() or not json.loads(p.read_text()).get('complete'):time.sleep(5)
results=[]
for method in ['tracktrack','putr']:
    reference=root/f'artifacts/mot/async_live/{method}_1'
    for repetition in [1,2,3]:
        folder=root/f'artifacts/mot/async_live/{method}_{repetition}';folder.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'scripts/mot_benchmark.py','--method',method,'--sequences','M2,formation',
                 '--max-frames','300','--live','--render','--async-export','--output',str(folder)]
        start=time.perf_counter()
        with (folder.parent/f'{method}_{repetition}.log').open('w',encoding='utf-8') as log:r=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        result=dict(method=method,repetition=repetition,exit_code=r.returncode,external_seconds=time.perf_counter()-start)
        if not r.returncode:
            report=json.loads((folder/'report.json').read_text());result['runs']=report['runs']
            result['trace_exact']={s:hashlib.sha256((folder/f'{s}.jsonl').read_bytes()).hexdigest()==hashlib.sha256((reference/f'{s}.jsonl').read_bytes()).hexdigest() for s in ['M2','formation']}
        results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='runs'}),flush=True)
        (root/'reports/mot_async_repeats.json').write_text(json.dumps(dict(complete=False,runs=results),indent=2))
(root/'reports/mot_async_repeats.json').write_text(json.dumps(dict(complete=True,runs=results),indent=2))
