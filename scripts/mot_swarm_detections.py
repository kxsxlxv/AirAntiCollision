"""Run the unchanged YOLO26m engine on verified UAVSwarm JPEG inputs."""
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
from airavoid.detector import Detector
out=ROOT/'artifacts/mot/swarm_detections';out.mkdir(parents=True,exist_ok=True)
det=Detector(ROOT/'artifacts/yolo26m_local.engine',conf=.03,tiled=True,gpu_preprocess=True,tile_edge_margin=2)
for seq in ['02','06','28']:
    source=ROOT/f'data/mot_swarm/UAVSwarm-{seq}'
    start=time.perf_counter();rows=[]
    for i,path in enumerate(sorted((source/'img1').glob('*.jpg'))):
        img=cv2.imread(str(path));boxes=det(img)
        rows.append(dict(frame=i,time=i/30,detections=boxes))
    path=out/f'sw{seq}.jsonl';path.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    print(seq,len(rows),len(rows)/(time.perf_counter()-start),flush=True)
(out/'manifest.json').write_text(json.dumps(dict(engine_sha256=hashlib.sha256((ROOT/'artifacts/yolo26m_local.engine').read_bytes()).hexdigest(),
    input_manifest_sha256=hashlib.sha256((ROOT/'data/mot_swarm/manifest.json').read_bytes()).hexdigest(),conf=.03,tiled=True,tile_edge_margin=2,gpu_preprocess=True),indent=2))
