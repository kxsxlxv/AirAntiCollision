"""One predeclared tracker-threshold sensitivity; unchanged detector boxes.

Existing reference confidence .18 versus .03 new-track confidence. Two QUT clips
(multi-target M2 and difficult S2), plus all three selected UAVSwarm sequences.
This is a sensitivity diagnostic, not a new untouched test or exhaustive tuning.
"""
from pathlib import Path
import subprocess,sys,json,time
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'artifacts/mot/confidence18';out.mkdir(parents=True,exist_ok=True)
runs=[]
for method in ['tracktrack','botsort','ocsort','diffmot','putr','trackssm']:
    folder=out/method;start=time.perf_counter()
    command=[sys.executable,'scripts/mot_benchmark.py','--method',method,'--high','.18',
        '--sequences','M2,S2,sw02,sw06,sw28','--output',str(folder)]
    with (out/f'{method}.log').open('w',encoding='utf-8') as log:r=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    result=dict(method=method,exit_code=r.returncode,external_seconds=time.perf_counter()-start)
    if not r.returncode:
        with (out/f'{method}_metrics.log').open('w',encoding='utf-8') as log:
            m=subprocess.run([sys.executable,'scripts/mot_metrics.py',str(folder)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        result['metrics_exit_code']=m.returncode
    runs.append(result);print(json.dumps(result),flush=True)
    (ROOT/'reports/mot_confidence_status.json').write_text(json.dumps(dict(complete=False,runs=runs),indent=2))
(ROOT/'reports/mot_confidence_status.json').write_text(json.dumps(dict(complete=True,runs=runs),indent=2))
