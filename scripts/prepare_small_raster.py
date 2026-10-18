"""Resize an existing real M2 clip for a small-raster inference regression."""
import json
from pathlib import Path
import subprocess
from airavoid.provenance import file_sha256

out=Path('data/qut_scaled');out.mkdir(parents=True,exist_ok=True);video=out/'M2_640.mp4'
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i','data/qut/M2.mp4','-vf','scale=640:360',
    '-fps_mode','passthrough','-an','-c:v','libx264','-preset','fast','-crf','16','-movflags','+faststart',str(video)],check=True)
labels=json.loads(Path('data/qut/M2.gt.json').read_text())
for label in labels:label['u']/=2;label['v']/=2
(out/'M2_640.gt.json').write_text(json.dumps(labels))
(out/'M2_640.manifest.json').write_text(json.dumps({'kind':'Derived real RGB resize, same studied encounter, not another independent event',
    'source':'data/qut/M2.mp4','source_sha256':file_sha256('data/qut/M2.mp4'),'output_sha256':file_sha256(video),
    'scale':.5,'resolution':[640,360],'labels':'Original approximate centroids scaled by0.5; 10px tolerance corresponds to original20px',
    'timing':'Source CFR presentation times retained; no temporal augmentation'},indent=2))
