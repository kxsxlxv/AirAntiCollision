"""Official TrackEval metric implementations on fully labelled UAVSwarm boxes."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'external/mot/python_deps'))
sys.path.insert(0,str(ROOT/'external/mot/TrackTrack/3. Tracker'))
for k,v in [('float',float),('int',int),('bool',bool)]:
    if k not in np.__dict__:setattr(np,k,v)
from trackeval.metrics import HOTA,Identity,CLEAR

def iou(a,b):
    a=np.asarray(a).reshape(-1,4);b=np.asarray(b).reshape(-1,4)
    wh=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
    inter=wh.prod(-1);aa=np.maximum(0,a[:,2:]-a[:,:2]).prod(-1);bb=np.maximum(0,b[:,2:]-b[:,:2]).prod(-1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)

def sequence_data(rows,gt):
    gs=sorted(set(int(g[1]) for g in gt));ts=sorted(set(int(x[0]) for r in rows for x in r['tracks']))
    gm={x:i for i,x in enumerate(gs)};tm={x:i for i,x in enumerate(ts)}
    d=dict(num_gt_ids=len(gs),num_tracker_ids=len(ts),num_timesteps=len(rows),gt_ids=[],tracker_ids=[],similarity_scores=[])
    for r in rows:
        g=gt[gt[:,0]==r['frame']+1];tr=r['tracks'];gb=g[:,2:6].copy();gb[:,2:]+=gb[:,:2]
        d['gt_ids'].append(np.array([gm[int(x)] for x in g[:,1]],dtype=int))
        d['tracker_ids'].append(np.array([tm[int(x[0])] for x in tr],dtype=int))
        d['similarity_scores'].append(iou(gb,[x[1:5] for x in tr]))
    d['num_gt_dets']=sum(map(len,d['gt_ids']));d['num_tracker_dets']=sum(map(len,d['tracker_ids']))
    return d

def compact(results):
    return {m:{k:float(np.mean(v)) if isinstance(v,np.ndarray) else float(v) for k,v in r.items()} for m,r in results.items()}

def evaluate(folder,sequences=('sw02','sw06','sw28')):
    metrics=[HOTA(),Identity(),CLEAR({'PRINT_CONFIG':False})];per={};allresults={m.get_name():{} for m in metrics}
    for seq in sequences:
        path=Path(folder)/f'{seq}.jsonl'
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        gt=np.loadtxt(ROOT/f'data/mot_swarm/UAVSwarm-{seq[2:]}/gt/gt.txt',delimiter=',',ndmin=2)
        if not np.all(gt[:,6:9]==1):raise ValueError('Nontrivial ignore/class/visibility labels need explicit MOT preprocessing')
        data=sequence_data(rows,gt)
        results={m.get_name():m.eval_sequence(data) for m in metrics}
        per[seq]=compact(results)
        for name,r in results.items():allresults[name][seq]=r
    combined=compact({m.get_name():m.combine_sequences(allresults[m.get_name()]) for m in metrics})
    result=dict(protocol='Official TrackEval HOTA/CLEAR/Identity, continuous box IoU, full labels, online outputs only',per_sequence=per,combined=combined)
    (Path(folder)/'mot_metrics.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':print(json.dumps(evaluate(sys.argv[1]),indent=2))
