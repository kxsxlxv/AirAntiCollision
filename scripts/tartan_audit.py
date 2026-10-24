"""Offline-only trajectory / visual-cue comparison for one real Tartan clip."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from airavoid.provenance import file_sha256
from airavoid.tartan_reference import (read_clock, paired_points, read_labels,
    read_raw_adsb, match_objects, interpolate_reference, fit_horizontal_cpa)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(values):
    a = np.array(values, float)
    return None if not len(a) else {'count': len(a), 'median': float(np.median(a)),
                                  'p95': float(np.percentile(a, 95)), 'max': float(np.max(a))}


def trajectory_diagram(points, output):
    """Engineering plan-view diagram of measured reference points, not a safe zone."""
    image = Image.new('RGB', (1600, 1100), '#09131f')
    draw = ImageDraw.Draw(image)
    font_path = 'C:/Windows/Fonts/segoeui.ttf'
    font = ImageFont.truetype(font_path, 25)
    title_font = ImageFont.truetype(font_path, 38)
    draw.text((65, 35), 'TartanAviation • ADS-B reference ground track', font=title_font, fill='white')
    draw.text((65, 95), 'Stationary ground camera • North / East • no aircraft maneuver recommendation', font=font, fill='#a2b5c9')
    x0, y0, scale = 320, 850, .075
    def project(north, east):
        return x0 + east * scale, y0 - north * scale
    for n in range(-1000, 9001, 1000):
        y = project(n, 0)[1]
        draw.line((150, y, 1460, y), fill='#243747')
        draw.text((70, y - 14), str(n // 1000), font=font, fill='#a2b5c9')
    for e in range(-1000, 12001, 1000):
        x = project(0, e)[0]
        draw.line((x, 160, x, 950), fill='#243747')
        draw.text((x - 12, 960), str(e // 1000), font=font, fill='#a2b5c9')
    draw.text((40, 135), 'N km', font=font, fill='white')
    draw.text((1320, 995), 'E km', font=font, fill='white')
    xy = [project(p['north_m'], p['east_m']) for p in points]
    draw.line(xy, fill='#4dd9f0', width=4)
    for x, y in xy:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='#4dd9f0')
    draw.text((xy[0][0] + 20, xy[0][1] - 15), 'START', font=font, fill='white')
    draw.text((xy[-1][0] + 20, xy[-1][1] - 30), 'END', font=font, fill='white')
    draw.ellipse((x0 - 10, y0 - 10, x0 + 10, y0 + 10), fill='white')
    draw.text((x0 - 120, y0 + 35), 'Ground camera', font=font, fill='white')
    closest = min(points, key=lambda p: p['horizontal_range_m'])
    x, y = project(closest['north_m'], closest['east_m'])
    draw.line((x0, y0, x, y), fill='#ffbc60', width=3)
    draw.text((x + 25, y - 35), f"Nearest sampled horizontal range: {closest['horizontal_range_m']/1000:.2f} km", font=font, fill='#ffbc60')
    draw.text((65, 1040), 'Reference geometry only. Vertical ADS-B values may be barometric; no collision labels supplied.', font=font, fill='#a2b5c9')
    image.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default='artifacts/tartan_jun26')
    parser.add_argument('--run', help='Completed genuine inference run; optional for reference-only audit')
    parser.add_argument('--output', default='reports/tartan_jun26')
    args = parser.parse_args()
    folder, output = Path(args.folder), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    clock_file = next(folder.glob('*.srt'))
    clock = read_clock(clock_file)
    points = paired_points(next(folder.glob('*_sink_adsb.pkl')), clock)
    labels = read_labels(folder)
    origin = clock[2]
    ids = {p['icao'] for p in points}
    raw_file = folder / 'adsb/1.csv'
    raw, rejected = read_raw_adsb(raw_file, ids, min(clock.values()) - 5, max(clock.values()) + 5) if raw_file.exists() else ([], 0)
    label_errors, raw_coordinates = [], []
    for p in points:
        for gt in labels.get(p['ordinal'], []):
            if gt['icao'] == p['icao']:
                label_errors.append(abs(gt['horizontal_label_m'] - p['horizontal_range_m']))
        same = [r for r in raw if r['icao'] == p['icao'] and
                abs(r['latitude_deg'] - p['latitude_deg']) < 1e-7 and
                abs(r['longitude_deg'] - p['longitude_deg']) < 1e-7]
        if same:
            r = min(same, key=lambda r: abs(r['utc_s'] - p['utc_s']))
            raw_coordinates.append({'ordinal': p['ordinal'], 'icao': p['icao'],
                                    'raw_minus_paired_time_s': r['utc_s'] - p['utc_s'],
                                    'raw_age_s': r['age_s']})
    write_csv(output / 'reference_points.csv', points)
    write_csv(output / 'raw_adsb_unique.csv', raw)
    write_csv(output / 'raw_coordinate_crosscheck.csv', raw_coordinates)
    closest = min(points, key=lambda p: p['horizontal_range_m'])
    cpa = []
    for p in points:
        fit = fit_horizontal_cpa([q for q in points if q['icao'] == p['icao']], p['utc_s'])
        if fit:
            cpa.append({'ordinal': p['ordinal'], 'icao': p['icao'], 'utc_s': p['utc_s'], **fit})
    write_csv(output / 'reference_causal_horizontal_cpa.csv', cpa)
    trajectory_diagram(points, output / 'trajectory_reference.png')
    gaps = np.diff([p['utc_s'] for p in points])
    report = {'dataset': 'TartanAviation', 'source': 'https://github.com/castacks/TartanAviation/',
              'sample': clock_file.name.replace('_subtitle.srt', ''), 'reference_points': len(points),
              'icao_ids': sorted(ids), 'annotated_frames': len(labels),
              'acquisition_rt_records': len(clock), 'acquisition_span_s': max(clock.values()) - min(clock.values()),
              'paired_reference_gap_s': summary(gaps), 'nearest_sampled_horizontal_reference': closest,
              'annotation_horizontal_distance_vs_paired_reference_abs_error_m': summary(label_errors),
              'raw_unique_reports_in_clip_window': len(raw), 'raw_rejected_records': rejected,
              'paired_points_with_exact_raw_coordinate_match': len(raw_coordinates),
              'raw_minus_paired_timestamp_s': summary([abs(x['raw_minus_paired_time_s']) for x in raw_coordinates]),
              'raw_gnss_altitude_reports': sum(r['altitude_is_gnss'] for r in raw),
              'limitations': ['Stationary ground camera, not moving ownship.',
                'Paired ADS-B and raw CSV are the same receiver source, not independent ground truths.',
                'No verified intrinsics/extrinsics or synchronized ownship motion.',
                'Frame alignment follows official 24fps extraction numbered from 2; clock/exposure delay not calibrated.',
                'Label distance is horizontal km; vector norm is a slant proxy with uncertain vertical datum.',
                'No physical collision labels, safety accuracy, maneuver validation or calibrated collision probabilities.'],
              'provenance': {p.name: file_sha256(p) for p in folder.iterdir() if p.is_file() and p.suffix in ('.pkl', '.srt', '.zip', '.mp4')},
              'raw_csv_sha256': file_sha256(raw_file) if raw_file.exists() else None}
    if args.run:
        run = Path(args.run)
        status = json.loads((run / 'run_status.json').read_text())
        benchmark = json.loads((run / 'benchmark.json').read_text())
        if status['state'] != 'complete' or benchmark['kind'] != 'live_neural_inference':
            raise ValueError('Expected completed live neural inference')
        if benchmark['input_sha256'] != report['provenance'][next(folder.glob('*.mp4')).name] or benchmark['clock_sha256'] != file_sha256(clock_file):
            raise ValueError('Inference source or acquisition clock differs from audit source')
        rows = [json.loads(line) for line in (run / 'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
        detections = {r['frame']: r for r in map(json.loads, (run / 'detections.jsonl').read_text().splitlines())}
        counts, cue_counts, matches, total_gt, detection_matches, airplane_matches = Counter(), Counter(), [], 0, 0, 0
        alignment = Counter()
        for row in rows:
            if row['ordinal'] != row['frame'] + 2 or abs(row['utc_s'] - clock[row['ordinal']]) > 1e-6:
                raise ValueError('Run did not use the declared acquisition clock mapping')
            annotation = labels.get(row['ordinal'], [])
            total_gt += len(annotation)
            fresh = [obj for obj in row['objects'] if abs(obj['last_seen'] - row['time']) < 1e-6]
            for offset in range(-2, 3):
                alignment[str(offset)] += len(match_objects(fresh, labels.get(row['ordinal'] + offset, [])))
            raw_boxes = [{'box': b[:4], 'class_id': int(b[5])} for b in detections[row['frame']]['detections']]
            dm = match_objects(raw_boxes, annotation)
            detection_matches += len(dm)
            airplane_matches += sum(o['class_id'] == 0 for o, _, _ in dm)
            cues = {x['id']: x for x in row['advisory']['demo']['targets']}
            for obj, gt, overlap in match_objects(fresh, annotation):
                cue = cues[obj['id']]
                counts[cue['state']] += 1
                ref = interpolate_reference(points, gt['icao'], row['utc_s'], max_gap=3.1)
                before = interpolate_reference(points, gt['icao'], row['utc_s'] - 1, max_gap=3.1)
                after = interpolate_reference(points, gt['icao'], row['utc_s'] + 1, max_gap=3.1)
                growth = -(math.log(after['horizontal_range_m']) - math.log(before['horizontal_range_m'])) / 2 if before and after else None
                matches.append({'frame': row['frame'], 'ordinal': row['ordinal'], 'time': row['time'],
                                'utc_s': row['utc_s'], 'track_id': obj['id'], 'icao': gt['icao'],
                                'predicted_class': obj['class'], 'reference_model': gt['model'],
                                'iou': overlap, 'measured_span_px': max(obj['box'][2]-obj['box'][0], obj['box'][3]-obj['box'][1]),
                                'label_horizontal_m': gt['horizontal_label_m'],
                                'reference_horizontal_m': ref['horizontal_range_m'] if ref else None,
                                'cue_state': cue['state'], 'cue_reason': cue['reason'],
                                'visual_log_growth_per_s': cue['growth_per_s'],
                                'reference_negative_log_horizontal_range_rate': growth,
                                'reference_bracket_gap_s': ref['bracket_gap_s'] if ref else None})
            cue_counts.update(x['state'] for x in row['advisory']['demo']['targets'])
        write_csv(output / 'matched_observations.csv', matches)
        sign_rows = [r for r in matches if r['visual_log_growth_per_s'] is not None and r['reference_negative_log_horizontal_range_rate'] is not None and abs(r['reference_negative_log_horizontal_range_rate']) >= .005]
        switches = sum(a['track_id'] != b['track_id'] for a, b in zip(matches, matches[1:]) if a['icao'] == b['icao'] and b['time'] - a['time'] < 1.)
        report['inference'] = {'run': str(run), 'benchmark_sha256': file_sha256(run / 'benchmark.json'),
            'weights_sha256': benchmark['weights_sha256'], 'detector_contract': benchmark['detector_contract'],
            'processed_frames': len(rows), 'gt_objects_in_sampled_frames': total_gt,
            'raw_detector_iou03_matches_all_classes': detection_matches,
            'raw_detector_iou03_matches_airplane_class': airplane_matches,
            'fresh_confirmed_track_matches': len(matches), 'matched_cue_states': dict(counts),
            'all_predicted_cue_states_including_unmatched': dict(cue_counts),
            'short_gap_matched_track_id_changes': switches,
            'alignment_sensitivity_matches_ordinal_offsets': dict(alignment),
            'growth_sign_comparison_count': len(sign_rows),
            'growth_sign_agreement_fraction': sum((r['visual_log_growth_per_s'] > 0) == (r['reference_negative_log_horizontal_range_rate'] > 0) for r in sign_rows) / len(sign_rows) if sign_rows else None,
            'scope': 'selected episode, overlap threshold 0.3, no overall MOT or collision accuracy claim'}
        report['inference']['matched_approaching_frames'] = sum(r['reference_negative_log_horizontal_range_rate'] is not None and r['reference_negative_log_horizontal_range_rate'] > .005 for r in matches)
        report['inference']['matched_receding_frames'] = sum(r['reference_negative_log_horizontal_range_rate'] is not None and r['reference_negative_log_horizontal_range_rate'] < -.005 for r in matches)
        report['inference']['positive_visual_growth_frames'] = sum(r['visual_log_growth_per_s'] is not None and r['visual_log_growth_per_s'] > .035 for r in matches)
    (output / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('provenance',)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
