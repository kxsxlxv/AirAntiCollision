"""Opt-in real TartanAviation inference, with acquisition RT rather than CFR time.

Only the original video and its non-GT acquisition clock enter this runtime.
Annotations and ADS-B trajectories are read by the separate offline audit.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from airavoid.assessment_runtime import detector_contract
from airavoid.aviation_regions import compute_attention_regions
from airavoid.aviation_render import AviationRenderer
from airavoid.demo_cue import DemoEncounterCue
from airavoid.detector import Detector, NAMES
from airavoid.export import FrameExporter
from airavoid.lifecycle import RunSession
from airavoid.media import TimestampWriter
from airavoid.provenance import file_sha256
from airavoid.tartan_reference import read_clock
from airavoid.tracking import Tracker


def execute(args, session):
    started = time.perf_counter()
    source = Path(args.video)
    out = session.output
    clock = read_clock(args.clock)
    probe = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames,avg_frame_rate', '-of', 'json', str(source)],
        capture_output=True, text=True, check=True).stdout)['streams'][0]
    # Official extraction starts at 2.png. Both selected original recordings
    # have exactly one fewer encoded frame than SRT ordinals. Fail, don't guess.
    expected_frames = int(probe['nb_frames'])
    if expected_frames != len(clock) - 1:
        raise ValueError('Unverified SRT/video frame correspondence: expected N video frames and N+1 RT ordinals')
    width, height = probe['width'], probe['height']
    origin = clock[2]
    if args.stride < 1:
        raise ValueError('Positive stride required')
    input_hash, clock_hash = file_sha256(source), file_sha256(args.clock)
    cap = cv2.VideoCapture(str(source))
    session.register(cap.release)
    detector = Detector(args.weights, conf=.03, tiled=True, gpu_preprocess=True, tile_edge_margin=2.)
    contract = detector_contract(detector, 'A')
    tracker = Tracker(image_edge_margin=1.)
    cue = DemoEncounterCue()
    renderer = AviationRenderer('TARTAN • RGB / измеренное время камеры') if not args.no_render else None
    camera = {'valid': True, 'mode': 'fixed', 'status': 'declared_static_ground_camera',
              'metric_motion_available': False, 'rotation_translation_disambiguated': False,
              'validity_scope': 'static ground camera per dataset, not onboard motion estimation'}
    exporter = None
    if renderer:
        writer = TimestampWriter(out / 'annotated.mp4', 24 / args.stride,
                                 (renderer.width, renderer.height), encoder='h264_nvenc')
        session.register(writer.abort)
        exporter = FrameExporter(renderer, out, writer=writer, asynchronous=True)
        session.register(exporter.abort)
    counts = Counter()
    calls = processed = decoded = 0
    end_time = None
    with (out / 'detections.jsonl').open('w') as detections, (out / 'decisions.jsonl').open('w', encoding='utf-8') as trace:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index = decoded
            decoded += 1
            ordinal = index + 2
            epoch = clock[ordinal]
            t = epoch - origin
            if index % args.stride:
                continue
            if args.max_frames and processed >= args.max_frames:
                end_time = t
                break
            boxes = detector(frame)
            calls += detector.last_network_calls
            detections.write(json.dumps({'frame': index, 'ordinal': ordinal, 'time': t, 'utc_s': epoch,
                'source_pts_ms': cap.get(cv2.CAP_PROP_POS_MSEC), 'detections': boxes,
                'observations': boxes, 'detector_ran': True}) + '\n')
            tracks = tracker.update(boxes, t, np.eye(3), image_size=(width, height))
            objects = [{'id': tr.id, 'box': tr.box, 'confidence': tr.confidence,
                        'class_id': tr.class_id, 'class': NAMES[tr.class_id], 'source': tr.source,
                        'last_seen': tr.last, 'assessment': {'state': 'uncertain', 'reason': 'physical_conflict_not_assessed'},
                        'demo_observation': {'measured': tr.last == t, 'ambiguous': t < tr.ambiguous_until,
                                             'clipped': tr.clipped, 'hits': tr.hits}}
                       for tr in tracks if tr.hits >= 3 and t - tr.last <= .25]
            demo = cue.update(objects, t, camera, (width, height))
            demo['attention_regions'] = compute_attention_regions(demo['targets'], width, height)
            counts.update(target['state'] for target in demo['targets'])
            advice = {'demo': demo}
            trace.write(json.dumps({'frame': index, 'ordinal': ordinal, 'time': t, 'utc_s': epoch,
                                    'objects': objects, 'camera': camera, 'advisory': advice},
                                   ensure_ascii=False, allow_nan=False) + '\n')
            if exporter:
                exporter.submit(frame, objects, t, camera, advice, preview=processed == 250)
            processed += 1
            if processed % 50 == 0:
                print(f'{processed} frames / acquisition {t:.2f}s / {processed/(time.perf_counter()-started):.2f} FPS / {dict(counts)}', flush=True)
    if not args.max_frames and decoded != expected_frames:
        raise ValueError('Video decoder count differs from verified clock mapping')
    if end_time is None:
        end_time = clock[decoded + 1] - origin + np.median(np.diff(list(clock.values())[-25:]))
    if exporter:
        exporter.finish(end_time)
    elapsed = time.perf_counter() - started
    report = {'kind': 'live_neural_inference', 'args': vars(args), 'input_sha256': input_hash,
              'clock_sha256': clock_hash, 'weights_sha256': file_sha256(args.weights),
              'detector_contract': contract, 'resolution': [width, height], 'frames': processed,
              'decoded_frames': decoded, 'network_calls_total': calls, 'elapsed_s': elapsed,
              'processing_fps_including_startup_and_encoder': processed / elapsed,
              'cue_states_track_frames': dict(counts), 'acquisition_duration_s': float(end_time),
              'source_clock': {'mapping': 'video frame index i -> SRT RT ordinal i+2; official extraction starts at 2',
                               'video_frames': expected_frames, 'srt_records': len(clock),
                               'timestamps': 'causal RT wall-clock observations, normalized to ordinal 2',
                               'annotation_or_adsb_inputs': False,
                               'limitation': 'frame mapping checked by counts and annotation alignment; exposure/RT delay not independently calibrated'},
              'physical_collision_classification': False, 'camera_calibration': None,
              'metric_ranges_used_by_runtime': False, 'runtime_reference_input': False,
              'cv2': cv2.__version__, 'providers': detector.providers}
    (out / 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--clock', required=True)
    parser.add_argument('--output', default='demo/tartan_jun26')
    parser.add_argument('--weights', default='artifacts/yolo26m_local.engine')
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    output = Path(args.output)
    if (output / 'run_status.json').exists():
        parser.error('Choose a new output directory; existing runs are preserved')
    with RunSession(output, {'args': vars(args)}) as session:
        execute(args, session)


if __name__ == '__main__':
    main()
