"""Export supplied existing synthetic image boxes using a separate SciPy environment."""
import json
from pathlib import Path
import hashlib
from scipy.io import loadmat

root=Path('F:/UAV_DATASETS/AIR2LAND(Synt)/airport2_crash/airport2_crash')
rows=[];missing=[]
for variant in ['000','012','024','035']:
    for metadata in sorted((root/variant/'left').glob('*-meta.mat')):
        image=metadata.with_name(metadata.name.replace('-meta.mat','-color.jpg'))
        if not image.exists():missing.append(str(image));continue
        data=loadmat(metadata,simplify_cells=True)
        box=data['bounding_boxes'].reshape(-1).tolist()
        if len(box)!=4:raise ValueError('Expected one documented xywh aircraft box')
        rows.append({'variant':variant,'image':str(image),'box_xywh':box,'metadata':str(metadata),
                     'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                     'metadata_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest()})
Path('data/air2land_detection.json').write_text(json.dumps({'kind':'Existing rendered synthetic RGB, left camera, detection-only diagnostic',
    'source':'https://github.com/micros-uav/micros_air2land','source_root':str(root),'license':'CC-BY as stated by authors',
    'selection':'Four predetermined variant IDs000,012,024,035; all existing paired image/metadata frames',
    'missing_color_images':missing,'rows':rows,'limits':['Related landing-scenario variants; independence not established',
    'No verified source FPS/intrinsics/unit conventions; do not invent temporal or metric threat scores','No simulator operated']},indent=2))
print('paired images',len(rows),'missing image files',len(missing))
