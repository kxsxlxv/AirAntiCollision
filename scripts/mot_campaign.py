"""Sequential GPU campaign; every candidate uses identical detector caches."""
from pathlib import Path
import sys,subprocess,json,time
root=Path(__file__).resolve().parents[1]
out=root/'artifacts/mot/campaign';out.mkdir(parents=True,exist_ok=True)
results=[]
for method in ['custom','ocsort','tracktrack','botsort','diffmot','putr','trackssm']:
    folder=out/method
    command=[sys.executable,str(root/'scripts/mot_benchmark.py'),'--method',method,'--output',str(folder)]
    start=time.perf_counter()
    with (out/f'{method}.log').open('w',encoding='utf-8') as log:
        process=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    result=dict(method=method,exit_code=process.returncode,external_seconds=time.perf_counter()-start,report=str(folder/'report.json'))
    results.append(result)
    (root/'reports/mot_campaign_status.json').write_text(json.dumps(dict(complete=len(results)==7,runs=results),indent=2))
    print(json.dumps(result),flush=True)
