"""Grid-math coverage at every object count seen in completed aviation traces.

This checks cardinality and geometry; it does not inspect every encoded ROI pixel.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from airavoid.aviation_render import AviationRenderer,roi_grid_layout,fresh_observation


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(run):
    run=Path(run).resolve();path=run/'decisions.jsonl';counts=Counter();fresh=Counter();frames=stale_frames=0
    with path.open(encoding='utf-8') as source:
        for text in source:
            row=json.loads(text);objects=[o for o in row['objects'] if AviationRenderer._valid_box(o.get('box'))]
            if len({o['id'] for o in objects})!=len(objects):raise ValueError('Repeated valid ID in a frame')
            n=len(objects);current=sum(fresh_observation(o,row['time']) for o in objects)
            counts[n]+=1;fresh[current]+=1;stale_frames+=current<n;frames+=1
    cases=[]
    for n in sorted(counts):
        cells=roi_grid_layout(n)
        if len(cells)!=n:raise ValueError('Grid omits object cells')
        for i,(x,y,w,h) in enumerate(cells):
            if min(w,h)<=0 or x<0 or y<0 or x+w>3840 or y+h>1080:raise ValueError('Cell outside3840x1080')
            for q,r,s,t in cells[i+1:]:
                if min(x+w,q+s)>max(x,q) and min(y+h,r+t)>max(y,r):raise ValueError('Grid cells overlap')
        cases.append({'objects':n,'frames':counts[n],'ROI_cells':len(cells),'all_inside_canvas':True,'overlap_count':0,
                      'minimum_cell_size':[min((c[2] for c in cells),default=0),min((c[3] for c in cells),default=0)]})
    return {'run':str(run),'decisions_sha256':sha(path),'frames':frames,'maximum_simultaneous_display_objects':max(counts),
            'maximum_simultaneous_fresh_objects':max(fresh),'frames_with_stale_placeholders':stale_frames,
            'object_count_histogram':dict(sorted(counts.items())),'fresh_count_histogram':dict(sorted(fresh.items())),
            'actual_count_cases':cases,'grid_math_for_every_actual_count_passed':True,
            'ROI_pixels_inspected_on_every_encoded_frame':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='*',default=['demo/aviation_nps','demo/aviation_contrast','demo/aviation_approach'])
    p.add_argument('--output',default='reports/aviation_live_layout_validation.json');a=p.parse_args()
    records=[check(path) for path in a.runs];result={'state':'complete','validated_utc':datetime.now(timezone.utc).isoformat(),
        'renderer_source_sha256':sha(ROOT/'airavoid/aviation_render.py'),'frames':sum(r['frames'] for r in records),
        'scope':'Pure grid math for every actually observed object count; selected encoded PNGs visually inspected separately; not per-frame pixel coverage proof',
        'runs':records}
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
