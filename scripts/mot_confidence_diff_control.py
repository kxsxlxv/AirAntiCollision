"""Match the stochastic DiffMOT confidence diagnostic's clip order and seed."""
from pathlib import Path
import sys,json,time,subprocess
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'reports/mot_confidence_status.json'
while not p.exists() or not json.loads(p.read_text()).get('complete'):time.sleep(5)
folder=ROOT/'artifacts/mot/confidence03/diffmot';folder.parent.mkdir(parents=True,exist_ok=True)
with (folder.parent/'diffmot.log').open('w',encoding='utf-8') as log:
    r=subprocess.run([sys.executable,'scripts/mot_benchmark.py','--method','diffmot','--high','.03',
        '--sequences','M2,S2,sw02,sw06,sw28','--output',str(folder)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
with (folder.parent/'diffmot_metrics.log').open('w',encoding='utf-8') as log:
    m=subprocess.run([sys.executable,'scripts/mot_metrics.py',str(folder)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
(ROOT/'reports/mot_confidence_diff_control.json').write_text(json.dumps(dict(complete=True,runs=[dict(exit_code=r.returncode,metrics_exit_code=m.returncode)]),indent=2))
