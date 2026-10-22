"""Read-only audit of supplied NPS identities, frame geometry, source PTS and models."""
import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from airavoid.provenance import file_sha256

DEFAULT_DATA = Path('X:/DATASETS/UAV/TRACKING/PreparedTracking_NPS_verified_OK_v1_jpeg95')
DEFAULT_SOURCE = Path('X:/DATASETS/UAV/TRACKING/RGB/NPS-Drones')
REVISION = 'nps-user-verified-train-v2'
DEV = ['Clip_001', 'Clip_008', 'Clip_032']
EVAL = ['Clip_020', 'Clip_034', 'Clip_043']


def read_annotations(path, meta):
    """Explicit schema; no guessed positional columns or cross-video IDs."""
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'frame_id', 'object_id', 'x', 'y', 'w', 'h', 'state', 'identity_status'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Unknown identity CSV schema')
        rows = list(reader)
    seen = set()
    for row in rows:
        for key in ['frame_id', 'object_id']:
            row[key] = int(row[key])
        for key in ['x', 'y', 'w', 'h']:
            row[key] = float(row[key])
        key = (row['frame_id'], row['object_id'])
        if key in seen:
            raise ValueError(f'Duplicate identity/frame: {key}')
        seen.add(key)
        if not 0 <= row['frame_id'] < meta['frame_count']:
            raise ValueError('Annotation frame outside source')
        if not all(__import__('math').isfinite(row[k]) for k in ['x', 'y', 'w', 'h']) or min(row['w'], row['h']) <= 0:
            raise ValueError('Invalid box')
        if row['state'] != 'visible' or row['identity_status'] != 'verified':
            raise ValueError('Non-visible or unverified row needs an explicit evaluation policy')
    return rows


def source_boxes(path):
    result = defaultdict(list)
    with Path(path).open(newline='') as f:
        for raw in csv.reader(f):
            if not raw:
                continue
            values = [float(x) for x in raw if x.strip()]
            frame, count = map(int, values[:2])
            if len(values) != 2 + 4 * count:
                raise ValueError('Unknown DogFight source annotation schema')
            for i in range(count):
                x1, y1, x2, y2 = values[2+4*i:6+4*i]
                result[frame].append((x1, y1, x2-x1, y2-y1))
    return result


def audit(data, source, output):
    import av
    import cv2
    import numpy as np
    import onnx
    revision = data/'revisions'/REVISION
    revision_manifest = json.loads((revision/'COMPLETE.json').read_text())
    report = {'revision': REVISION, 'data_root': str(data), 'source_root': str(source),
              'revision_manifest_sha256': file_sha256(revision/'COMPLETE.json'),
              'gt_isolation': 'annotation adapter used by offline scoring/oracle only',
              'coverage_limit': 'User-verified labels; complete visibility cannot be proven by schema checks alone',
              'detector_training_independence': {'A': 'user confirmed', 'B': 'user confirmed 2026-10-22'},
              'clips': {}, 'errors': []}
    out_data = ROOT/'data/stage2'
    out_data.mkdir(parents=True, exist_ok=True)
    for key, entry in revision_manifest['annotations'].items():
        clip = key.split('/')[-1]
        folder = data/key
        meta = json.loads((folder/'meta.json').read_text())
        if (meta['frame_index_base'], meta['bbox_format'], meta['coordinate_system']) != (0, 'XYWH', 'zero_based_continuous'):
            raise ValueError('Unexpected coordinate contract')
        path = revision/entry['path']
        digest = file_sha256(path)
        if digest != entry['sha256']:
            raise ValueError('Canonical annotation manifest hash mismatch')
        rows = read_annotations(path, meta)
        images = sorted((folder/'frames').glob('*.jpg'))
        ids = [int(p.stem) for p in images]
        if ids != list(range(meta['frame_count'])):
            raise ValueError(f'{clip}: image sequence missing/duplicated frames')
        box_source = source/'Annotations'/meta['source_annotation']
        original = source_boxes(box_source)
        supplied = defaultdict(list)
        continuity = defaultdict(list)
        border = 0
        for r in rows:
            supplied[r['frame_id']].append(tuple(r[k] for k in ['x','y','w','h']))
            continuity[r['object_id']].append(r['frame_id'])
            border += r['x'] < 0 or r['y'] < 0 or r['x']+r['w'] > meta['width'] or r['y']+r['h'] > meta['height']
        mismatches = [f for f in set(original)|set(supplied) if Counter(original[f]) != Counter(supplied[f])]
        detail = {'frames': meta['frame_count'], 'resolution': [meta['width'], meta['height']],
                  'annotations': len(rows), 'ids': len(continuity), 'csv_sha256': digest,
                  'source_annotation_sha256': file_sha256(box_source),
                  'source_annotation_matches_meta_hash': file_sha256(box_source) == meta['source_annotation_sha256'],
                  'geometry_mismatch_frames_vs_source_annotation': mismatches,
                  'crossing_image_border_rows': int(border),
                  'identity_gaps': sum(sum(b-a > 1 for a,b in zip(sorted(fs),sorted(fs)[1:])) for fs in continuity.values()),
                  'id_scope': 'within clip only', 'base_identity_metadata': 'provisional, superseded by verified revision',
                  'split': 'development' if clip in DEV else 'evaluation' if clip in EVAL else 'unused'}
        if clip in DEV+EVAL:
            video = source/'Videos'/meta['source_video']
            detail['video_path'] = str(video)
            detail['video_sha256'] = file_sha256(video)
            if detail['video_sha256'] != meta['source_video_sha256']:
                raise ValueError('Original source video hash mismatch')
            timestamps = []
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                stream.thread_type = 'AUTO'
                detail['time_base'] = str(stream.time_base)
                for i, frame in enumerate(container.decode(stream)):
                    if frame.pts is None:
                        raise ValueError('Missing original presentation timestamp')
                    timestamps.append(float(frame.pts * frame.time_base))
                    if i in {0, meta['frame_count']//2, meta['frame_count']-1}:
                        rgb = frame.to_ndarray(format='bgr24')
                        jpg = cv2.imread(str(folder/'frames'/f'{i:08d}.jpg'))
                        if rgb.shape != jpg.shape or list(rgb.shape[:2][::-1]) != detail['resolution']:
                            raise ValueError('Source/JPEG resolution mismatch')
                        detail.setdefault('jpeg_source_pixel_mae_checks', {})[str(i)] = float(np.abs(rgb.astype(float)-jpg).mean())
            if len(timestamps) != meta['frame_count'] or any(b <= a for a,b in zip(timestamps,timestamps[1:])):
                raise ValueError('Original PTS/frame mapping fails')
            detail['pts_start_s'] = timestamps[0]
            dt = np.diff(timestamps)
            detail['source_dt_s_range'] = [float(min(dt)), float(max(dt))]
            detail['effective_fps_from_pts'] = (len(timestamps)-1)/(timestamps[-1]-timestamps[0])
            timefile = out_data/f'{clip}_pts.json'
            timefile.write_text(json.dumps({'video_sha256': detail['video_sha256'], 'pts_s': timestamps}))
            detail['timestamps_file'] = timefile.relative_to(ROOT).as_posix()
            detail['timestamps_sha256'] = file_sha256(timefile)
        report['clips'][clip] = detail
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    p2 = onnx.load('X:/MODELS/yolo26m-p2-1440/best.onnx', load_external_data=False)
    protocol = {'schema':'stage2-protocol-v1','frozen_before_inference':True,
                'created_utc':datetime.now(timezone.utc).isoformat(),
                'development':DEV,'evaluation':EVAL,'annotation_revision':REVISION,
                'audit_sha256':file_sha256(output),'detector_A_conf':.03,
                'detector_B_conf':.03,'detector_B_conf_is_not_calibrated_to_A':True,
                'threshold_selection':'A frozen prior baseline; B predeclared .03 with dev-only sensitivity before evaluation',
                'NPS_strides':[1,2,5],'QUT_strides':[1,2,5],
                'IoU_thresholds':[.25,.5,.75],
                'trackers':['custom','tracktrack','putr'],
                'methods':['bauer_disc','lai_lin','airtrack_cues','wcv_taumod','log','projective','bounded','calibrated','growth'],
                'GT_normal_inference_access':False,
                'P2_input':[d.dim_value for d in p2.graph.input[0].type.tensor_type.shape.dim],
                'P2_taxonomy':['fixed-wing UAV (user supplied semantics; exported label item)'],
                'real_physical_collision_truth':False}
    (ROOT/'reports/stage2_protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps({'clips':len(report['clips']), 'selected':DEV+EVAL,
                      'geometry_disagreements':{k:len(v['geometry_mismatch_frames_vs_source_annotation']) for k,v in report['clips'].items() if v['geometry_mismatch_frames_vs_source_annotation']},
                      'output':str(output)},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/stage2_dataset_audit.json')
    args=parser.parse_args()
    audit(args.data,args.source,args.output)
