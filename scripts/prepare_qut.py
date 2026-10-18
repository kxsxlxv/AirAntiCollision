"""Lossless timing from published 25 FPS TIFF sequence; labels exported separately."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import cv2

p=argparse.ArgumentParser()
p.add_argument('--root',default='F:/UAV_DATASETS/jasminmartin(Aircraft Collision Course Dataset)')
p.add_argument('--sequences',nargs='+',default=['M1','M2'])
p.add_argument('--seconds',type=float,default=35)
p.add_argument('--full',action='store_true',help='Encode all available frames, maintaining source-ID timing across gaps')
p.add_argument('--end-frame',type=int)
p.add_argument('--output',default='data/qut')
p.add_argument('--split-label',default='unassigned',help='Explicit experiment provenance; do not infer untouched status from a sequence name')
a=p.parse_args()
root=Path(a.root); out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
for seq in a.sequences:
    files=sorted((root/'image-sequences'/seq).glob('*.tif'))
    if a.end_frame:
        files=[f for f in files if int(f.stem)<=a.end_frame]
    if not a.full:files=files[-int(a.seconds*25):]
    mapping={int(f.stem):f for f in files}
    source_ids=list(range(int(files[0].stem),int(files[-1].stem)+1))
    missing=[i for i in source_ids if i not in mapping]
    h,w=cv2.imread(str(files[0])).shape[:2]
    output=out/f'{seq}.mp4'
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','25','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    gt=[]
    for kind,folder in [('approach','target-aircarft-labels'),('other','multi-aircraft-labels')]:
        for label in sorted((root/'Target_Labels'/folder).glob(f'{seq}*.csv')):
            with label.open() as f:
                for row in csv.DictReader(f):
                    source=int(row['frame'])
                    if int(files[0].stem)<=source<=int(files[-1].stem):
                        gt.append({'frame':source-int(files[0].stem),'source_frame':source,'id':kind+'_'+label.stem,'kind':kind,'u':float(row['u']),'v':float(row['v']),'range_offline_only':float(row['range']) if 'range' in row else None})
    previous=None
    for source_id in source_ids:
        if source_id in mapping:previous=cv2.imread(str(mapping[source_id]))
        if previous is None:raise RuntimeError('No initial image for source timeline')
        proc.stdin.write(previous.tobytes())
    proc.stdin.close()
    if proc.wait():raise RuntimeError('ffmpeg failed')
    (out/f'{seq}.gt.json').write_text(json.dumps(gt),encoding='utf-8')
    manifest={'sequence':seq,'source_root':str(root),'source_first':int(files[0].stem),'source_last':int(files[-1].stem),'frames':len(source_ids),'actual_source_files':len(files),'missing_source_ids_repeated_previous':missing,'selection':'all_available' if a.full else f'last_{a.seconds}s','fps':25,'fps_source':'Martin et al 2022 section III','split':a.split_label,'kind':'real RGB / stationary ground camera','sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'label_semantics':'approaching target vs other aircraft; no actual collision ground truth','gt_not_available_to_inference':True}
    (out/f'{seq}.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)
