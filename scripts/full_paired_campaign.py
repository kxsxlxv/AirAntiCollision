"""Paired complete-source default/edge2 validation after development selection.

All sequences have been inspected previously; this is extended validation, not a
new untouched test split. Published nominal approach categories are not CPA truth.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

sequences=[f'M{i}' for i in range(1,9)]+[f'S{i}' for i in range(1,8)]
for seq in sequences:
    manifest=Path(f'data/qut_full/{seq}.manifest.json');deadline=time.monotonic()+1800
    if not manifest.exists():print('Waiting for prepared manifest:',seq,flush=True)
    while not manifest.exists():
        if time.monotonic()>deadline:raise TimeoutError('Source preparation did not finish: '+seq)
        time.sleep(5)
    for margin in [0,2]:
        root=Path('artifacts/full_gpu_default' if margin==0 else 'artifacts/full_gpu_edge2')
        subprocess.run([sys.executable,'-m','scripts.batch_evaluate','--sequences',seq,'--profile','reference','--tiled','--gpu-preprocess',
                        '--tile-edge-margin',str(margin),'--data-root','data/qut_full','--stride','5','--root',str(root)],check=True)
        counts=defaultdict(Counter);events=[];finished=[];frames=0;unmatched=Counter()
        for path in sorted(root.glob('*/evaluation.json')):
            r=json.loads(path.read_text());finished.append(path.parent.name);frames+=r['frames']
            for kind,c in r['counts'].items():counts[kind].update(c)
            unmatched.update(r['unmatched_track_frames'])
            events.extend({'sequence':path.parent.name,**e} for e in r['events'])
        report={'sequences':finished,'counts':counts,'events':events,'processed_frames':frames,'unmatched_track_frames':unmatched,
                'configuration':{'stride':5,'camera':'fixed','gpu_preprocess':True,'tiled':True,'tile_edge_margin':margin,
                                 'policy':'confirmed','allow_recession_clearance':False},
                'kind':'Paired full-source extension of previously inspected sequences, not an untouched test set',
                'source_caveats':['M3 range conflicts with nominal approach category','Approximate/interpolated centroids; no true collision/CPA labels',
                                  'Source-ID gaps repeated from preceding TIFF and explicitly listed in manifests']}
        (root/'full_summary.json').write_text(json.dumps(report,indent=2))
        print('Finished',seq,'edge',margin,'runs',len(finished),flush=True)
