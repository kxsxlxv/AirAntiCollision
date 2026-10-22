"""Run selected MOT conditions serially in isolated Python processes."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def run(args):
    directory=ROOT/'artifacts/stage2/campaign_logs';directory.mkdir(parents=True,exist_ok=True)
    status={'started_utc':datetime.now(timezone.utc).isoformat(),'state':'running','runs':[]}
    path=ROOT/f'reports/stage2_{args.name}_status.json'
    conditions=[(d,m,int(s)) for d in args.detectors.split(',') for m in args.methods.split(',') for s in args.strides.split(',')]
    for detector,method,stride in conditions:
        command=[sys.executable,str(ROOT/'scripts/stage2_tracking.py'),'--detector',detector,
                 '--method',method,'--stride',str(stride)]
        if args.clips:command.extend(['--clips',args.clips])
        label=f'{args.name}_{detector}_{method}_s{stride}'
        print('Starting '+label,flush=True);start=time.perf_counter()
        status['active']=label;path.write_text(json.dumps(status,indent=2))
        with (directory/f'{label}.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        status['runs'].append({'condition':label,'command':command,'exit_code':result.returncode,
                               'elapsed_s':time.perf_counter()-start,'log':str(directory/f'{label}.log')})
        if result.returncode:
            status['state']='failed';path.write_text(json.dumps(status,indent=2));raise RuntimeError(label)
        path.write_text(json.dumps(status,indent=2));print('Complete '+label,flush=True)
    status.update(state='complete',active=None,finished_utc=datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(status,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--name',default='tracking_campaign');p.add_argument('--detectors',default='A,B')
    p.add_argument('--methods',default='custom,tracktrack,putr');p.add_argument('--strides',default='1')
    p.add_argument('--clips');run(p.parse_args())
