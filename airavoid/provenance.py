"""Input/cache identity checks for reproducible offline experiments."""
import hashlib
import json
from pathlib import Path
import numpy as np


def file_sha256(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def cache_identity(cache_path,input_path,input_hash,width,height,fps):
    sidecar=Path(cache_path).with_name('benchmark.json')
    if not sidecar.exists():
        return {'status':'unverified_legacy_cache','reason':'No run benchmark beside cache; only frame/time consistency checked'}
    report=json.loads(sidecar.read_text(encoding='utf-8'))
    expected=report.get('input_sha256')
    if expected:
        if expected!=input_hash:raise ValueError('Detection cache input SHA-256 differs from source video')
        return {'status':'sha256_verified','input_sha256':input_hash}
    resolution=report.get('resolution');rate=report.get('source_fps')
    if resolution and list(resolution)!=[width,height]:raise ValueError('Legacy detection cache source resolution differs')
    if rate and abs(rate-fps)>1e-3:raise ValueError('Legacy detection cache source FPS differs')
    original=report.get('args',{}).get('video')
    if original and Path(original).name!=Path(input_path).name:
        raise ValueError('Legacy detection cache source name differs; regenerate from this video')
    return {'status':'unverified_legacy_cache','reason':'Legacy benchmark has no source digest; matching name/resolution/FPS cannot prove content identity'}


def validate_cache_entry(entry,index,time):
    if 'time' in entry and abs(float(entry['time'])-time)>1e-3:
        raise ValueError(f'Detection cache source timestamp differs at frame {index}')
    for field in ['detections','observations','appearance_predictions']:
        for box in entry.get(field,[]):
            values=np.asarray(box[:6],float)
            if len(values)!=6 or not np.isfinite(values).all():raise ValueError('Cache contains nonfinite/malformed observation')
            x1,y1,x2,y2,score,kind=values
            if x2<=x1 or y2<=y1 or not 0<=score<=1 or kind!=int(kind) or not 0<=kind<5:
                raise ValueError('Cache contains invalid box/confidence/class')
