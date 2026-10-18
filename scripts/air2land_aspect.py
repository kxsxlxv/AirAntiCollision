"""Metadata-only aspect/range audit of existing synthetic RGB; no time/unit assumptions."""
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat

manifest=json.loads(Path('data/air2land_detection.json').read_text());report=[]
for variant in ['000','012','024','035']:
    samples=[]
    for row in manifest['rows']:
        if row['variant']!=variant:continue
        d=loadmat(row['metadata'],simplify_cells=True);pose=d['relative_poses'];box=row['box_xywh']
        samples.append({'source_index':int(d['idx']),'range_native_units':float(np.linalg.norm(pose[:,3])),
                        'span_gt_px':max(box[2],box[3]),'ptu_pitch':float(d['ptu_pitch']),'ptu_yaw':float(d['ptu_yaw']),
                        'rotation':pose[:,:3].tolist()})
    ranges=np.array([s['range_native_units'] for s in samples]);spans=np.array([s['span_gt_px'] for s in samples])
    row={'variant':variant,'images':len(samples),'range_max_min_ratio':float(max(ranges)/min(ranges)),
         'span_max_min_ratio':float(max(spans)/min(spans)),'first_last_range':ranges[[0,-1]].tolist(),
         'first_last_span':spans[[0,-1]].tolist(),'samples':samples};report.append(row)
    print({k:v for k,v in row.items() if k!='samples'})
Path('reports/air2land_aspect.json').write_text(json.dumps({'kind':'Supplied existing synthetic RGB metadata audit; no new rendering/simulation',
    'variants':report,'interpretation':'Range norm is invariant to camera rotation; span changes can come from target orientation, occlusion and box convention.',
    'limits':['Native pose units and source frame timing not verified; no metres, seconds, CPA or threat scores',
              'Source metadata provenance not independently flight validated','GT boxes can describe occluded content; this is not a visible-silhouette error calibration']},indent=2))
