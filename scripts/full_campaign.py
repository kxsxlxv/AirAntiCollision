"""Wait for traceable full-sequence preparation, then evaluate each sequentially."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path('artifacts/full_campaign');root.mkdir(parents=True,exist_ok=True)
sequences=['M1','M2','M3','M4','M5','M6','M7','M8']
for seq in sequences:
    manifest=Path(f'data/qut_full/{seq}.manifest.json')
    while not manifest.exists():time.sleep(2)
    print('Ready:',seq,flush=True)
    subprocess.run([sys.executable,'-m','scripts.batch_evaluate','--sequences',seq,'--profile','reference','--allow-recession-clearance','--tiled','--data-root','data/qut_full','--stride','5','--root',str(root)],check=True)
    aggregate=defaultdict(Counter);events=[];finished=[]
    for path in sorted(root.glob('*/evaluation.json')):
        result=json.loads(path.read_text());finished.append(path.parent.name)
        for kind,count in result['counts'].items():aggregate[kind].update(count)
        events.extend(result['events'])
    (root/'full_summary.json').write_text(json.dumps({'sequences':finished,'counts':aggregate,'events':events,'stride':5,'selection':'all available multi-aircraft source frames; repeated known missing IDs preserve timing','label_type':'encounter category proxy, not collision truth'},indent=2))
print('Full campaign complete',flush=True)
