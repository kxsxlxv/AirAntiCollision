"""Sequential detector runs for all prepared sequences and aggregate proxy metrics."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import subprocess
import sys
from scripts.evaluate import evaluate
from airavoid.cli import parse_args
from airavoid.provenance import file_sha256

p=argparse.ArgumentParser();p.add_argument('--sequences',nargs='+',default=['M3','M4','M5','M6','M7','M8','S1','S2','S3','S4','S5','S6','S7'])
p.add_argument('--profile',choices=['conservative','reference'],default='conservative')
p.add_argument('--allow-recession-clearance',action='store_true')
p.add_argument('--gpu-preprocess',action='store_true');p.add_argument('--cpu-preprocess',action='store_true');p.add_argument('--tile-edge-margin',type=float)
p.add_argument('--tiled',action='store_true');p.add_argument('--adaptive-tiles',action='store_true');p.add_argument('--appearance-bridge',action='store_true');p.add_argument('--root',default='artifacts/heldout_confirmed');p.add_argument('--policy',choices=['raw','confirmed'],default='confirmed');p.add_argument('--data-root',default='data/qut');p.add_argument('--stride',type=int,default=2);a=p.parse_args()
root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
all_results=[];configurations={}
for seq in a.sequences:
    out=root/seq
    cmd=[sys.executable,'-m','airavoid',f'{a.data_root}/{seq}.mp4','--weights','artifacts/yolo26m_local.engine','--camera','fixed','--stride',str(a.stride),'--output',str(out),'--no-render','--policy',a.policy,'--profile',a.profile]
    if a.allow_recession_clearance:cmd.append('--allow-recession-clearance')
    if a.tiled:cmd.append('--tiled')
    if a.adaptive_tiles:cmd.append('--adaptive-tiles')
    if a.appearance_bridge:cmd.append('--appearance-bridge')
    if a.gpu_preprocess:cmd.append('--gpu-preprocess')
    if a.cpu_preprocess:cmd.append('--cpu-preprocess')
    if a.tile_edge_margin is not None:cmd+=['--tile-edge-margin',str(a.tile_edge_margin)]
    expected=vars(parse_args(cmd[3:]))
    print('Running',seq,flush=True)
    if (out/'benchmark.json').exists():
        existing=json.loads((out/'benchmark.json').read_text());old_args=existing.get('args',{})
        keys=['conf','stride','camera','policy','geometry','correlation_seconds','image_edge_margin','tile_edge_margin','tiled','adaptive_tiles','gpu_preprocess','appearance_bridge','detect_every','allow_recession_clearance']
        mismatch={k:(old_args[k],expected[k]) for k in keys if k in old_args and old_args[k]!=expected[k]}
        if mismatch:raise ValueError('Existing run has different effective parameters; use a fresh --root: '+str(mismatch))
        if existing.get('input_sha256') and existing['input_sha256']!=file_sha256(f'{a.data_root}/{seq}.mp4'):
            raise ValueError('Existing run input digest differs; use a fresh --root')
    if not (out/'benchmark.json').exists():
        with (root/f'{seq}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    result=evaluate(out/'decisions.jsonl',f'{a.data_root}/{seq}.gt.json')
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    configurations[seq]=json.loads((out/'benchmark.json').read_text()).get('args',{})
    all_results.append(result)
    aggregate=defaultdict(Counter);events=[]
    for r in all_results:
        for kind,c in r['counts'].items():aggregate[kind].update(c)
        events.extend(r['events'])
    report={'runs':len(all_results),'counts':aggregate,'events':events,'sequences':a.sequences[:len(all_results)],'label_type':'encounter-category proxy only','configuration':{'policy':a.policy,'tiled':a.tiled,'adaptive_tiles':a.adaptive_tiles,'appearance_bridge':a.appearance_bridge,'gpu_preprocess':a.gpu_preprocess,'tile_edge_margin':a.tile_edge_margin,'confidence':.18,'stride':a.stride,'data_root':a.data_root}}
    report['configuration']={k:configurations[seq].get(k) for k in ['profile','conf','stride','camera','policy','geometry','tiled','gpu_preprocess','tile_edge_margin','image_edge_margin','correlation_seconds','appearance_bridge','detect_every','allow_recession_clearance']}
    report['configurations_by_sequence']=configurations
    (root/'aggregate.json').write_text(json.dumps(report,indent=2))
    print(seq,result['counts'],flush=True)
