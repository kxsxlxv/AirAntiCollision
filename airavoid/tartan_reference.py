"""Offline TartanAviation reference audit. Never imported by the inference CLI.

Paired frame-indexed ADS-B points are reference measurements, not collision
labels. Their altitude may be barometric; derived vertical separation is not a
verified geometric height. Label distance is horizontal kilometres, not slant.
"""
import ast
import csv
from datetime import datetime, timezone
import io
import math
from pathlib import Path
import pickle
import re
import shlex
import zipfile

import numpy as np


def _numeric_scalar(dtype, payload):
    if dtype.kind not in 'fiu' or dtype.itemsize > 8 or len(payload) != dtype.itemsize:
        raise ValueError('Only small numeric NumPy scalars are permitted')
    return np.frombuffer(payload, dtype=dtype, count=1)[0].item()


class ReferenceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy' and name == 'dtype':
            return np.dtype
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name == 'scalar':
            return _numeric_scalar
        raise ValueError(f'Unsupported pickle global: {module}.{name}')


def read_reference_pickle(path):
    return ReferenceUnpickler(io.BytesIO(Path(path).read_bytes())).load()


def read_clock(path):
    """SRT ordinal -> actual RT epoch; reject malformed/nonmonotonic clocks.

    The displayed SRT intervals contain leading zero intervals and a startup
    discontinuity. They are deliberately not used as observation timestamps.
    """
    text = Path(path).read_text(encoding='utf-8-sig')
    records = re.findall(r'(?m)^(\d+)\s*\n[^\n]+\nRT:\s*(\d+)\s+(\d+)', text)
    if any(not 0 <= int(ns) < 1000000000 for _, _, ns in records):
        raise ValueError('SRT RT nanoseconds are out of range')
    clock = {int(frame): int(sec) + int(ns) * 1e-9 for frame, sec, ns in records}
    if not clock or len(clock) != len(records):
        raise ValueError('Missing or duplicate SRT RT records')
    keys = sorted(clock)
    if list(clock) != keys or keys != list(range(1, keys[-1] + 1)) or any(clock[b] <= clock[a] for a, b in zip(keys, keys[1:])):
        raise ValueError('SRT clock must be contiguous and strictly increasing')
    return clock


def paired_points(path, clock):
    data = read_reference_pickle(path)
    points = []
    for icao, track in data.items():
        for ordinal, values in track.items():
            if ordinal not in clock or len(values) != 9 or not np.isfinite(values).all():
                raise ValueError('Invalid paired ADS-B record')
            lat, lon, altitude, ref_lat, ref_lon, north, east, up, slant = map(float, values)
            norm = math.sqrt(north * north + east * east + up * up)
            if not math.isclose(norm, slant, abs_tol=1e-5):
                raise ValueError('Paired ADS-B vector/range units disagree')
            points.append({'icao': int(icao), 'ordinal': int(ordinal), 'utc_s': clock[ordinal],
                           'latitude_deg': lat, 'longitude_deg': lon, 'altitude_reference_m': altitude,
                           'camera_latitude_deg': ref_lat, 'camera_longitude_deg': ref_lon,
                           'north_m': north, 'east_m': east, 'vertical_proxy_m': up,
                           'horizontal_range_m': math.hypot(north, east), 'slant_proxy_m': slant})
    return sorted(points, key=lambda row: (row['icao'], row['ordinal']))


def read_labels(folder):
    """Frame-local .label indices are NOT persistent aircraft identities."""
    folder = Path(folder)
    identities = read_reference_pickle(next(folder.glob('*_acft_sink.pkl')))
    result = {}
    with zipfile.ZipFile(next(folder.glob('*_labels.zip'))) as archive:
        for info in archive.infolist():
            if not info.filename.endswith('.label'):
                continue
            ordinal = int(Path(info.filename).stem)
            aircraft = identities.get(ordinal, {})
            rows = []
            for line in archive.read(info).decode().splitlines():
                parts = shlex.split(line)
                if len(parts) != 9:
                    raise ValueError('Expected nine annotation fields')
                box = list(map(float, parts[:4]))
                # Identity is linked using the companion pickle, never index[4].
                matches = [int(icao) for icao, value in aircraft.items()
                           if list(map(float, value[1].split()[:4])) == box]
                if len(matches) != 1:
                    raise ValueError('Annotation does not have an unambiguous ICAO match')
                rows.append({'icao': matches[0], 'box': box, 'horizontal_label_m': float(parts[5]) * 1000,
                             'manufacturer': parts[6], 'model': parts[7], 'type': parts[8]})
            result[ordinal] = rows
    return result


def read_raw_adsb(path, ids, start, end):
    """Deduplicate receiver repeats; preserve Age and time resolution limitations."""
    result = {}
    rejected = 0
    id_strings = {str(i) for i in ids}
    with Path(path).open(encoding='utf-8-sig', newline='') as source:
        for row in csv.DictReader(source):
            if row['ID'] not in id_strings:
                continue
            try:
                date = list(map(int, ast.literal_eval(row['Date'])))
                hour, minute, second = map(float, ast.literal_eval(row['Time']))
                epoch = datetime(*date, int(hour), int(minute), int(second),
                                 tzinfo=timezone.utc).timestamp() + second % 1
                if not start <= epoch <= end:
                    continue
                item = {'icao': int(row['ID']), 'utc_s': epoch, 'latitude_deg': float(row['Lat']),
                        'longitude_deg': float(row['Lon']), 'altitude_ft': float(row['Altitude']),
                        'age_s': float(row['Age']), 'altitude_is_gnss': row['AltisGNSS'].lower() == 'true'}
                if not np.isfinite([item[k] for k in ('utc_s', 'latitude_deg', 'longitude_deg', 'altitude_ft', 'age_s')]).all():
                    raise ValueError('Nonfinite ADS-B')
                key = (item['icao'], epoch, item['latitude_deg'], item['longitude_deg'], item['altitude_ft'])
                if key not in result or item['age_s'] < result[key]['age_s']:
                    result[key] = item
            except (ValueError, TypeError, SyntaxError, KeyError):
                rejected += 1
    return sorted(result.values(), key=lambda row: (row['icao'], row['utc_s'])), rejected


def iou(a, b):
    x = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    intersection = x * y
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.


def match_objects(objects, labels, threshold=.3):
    """Offline one-to-one overlap matching; GT never supplies runtime identity."""
    candidates = sorted([(iou(obj['box'], gt['box']), i, j)
                         for i, obj in enumerate(objects) for j, gt in enumerate(labels)], reverse=True)
    used_i, used_j, result = set(), set(), []
    for overlap, i, j in candidates:
        if overlap < threshold or i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        result.append((objects[i], labels[j], overlap))
    return result


def interpolate_reference(points, icao, epoch, max_gap=3.):
    """Offline bracketed reference, no extrapolation; not an online prediction."""
    track = [p for p in points if p['icao'] == icao]
    times = np.array([p['utc_s'] for p in track])
    if len(times) < 2 or epoch < times[0] or epoch > times[-1]:
        return None
    upper = min(max(1, int(np.searchsorted(times, epoch))), len(times) - 1)
    a, b = track[upper - 1], track[upper]
    gap = b['utc_s'] - a['utc_s']
    if gap <= 0 or gap > max_gap:
        return None
    fraction = (epoch - a['utc_s']) / gap
    result = {k: a[k] + fraction * (b[k] - a[k]) for k in
              ('north_m', 'east_m', 'vertical_proxy_m', 'horizontal_range_m', 'slant_proxy_m')}
    result['bracket_gap_s'] = gap
    return result


def fit_horizontal_cpa(points, epoch, history=8., min_history=3.):
    """Causal reference-only constant-velocity diagnostic relative to fixed camera.

    This does not turn the visual pipeline into a telemetry-based flight system.
    No vertical safe/unsafe label is produced from pressure-derived altitudes.
    """
    recent = [p for p in points if epoch - history <= p['utc_s'] <= epoch]
    if len(recent) < 4 or recent[-1]['utc_s'] - recent[0]['utc_s'] < min_history:
        return None
    times = np.array([p['utc_s'] - epoch for p in recent])
    design = np.column_stack((times, np.ones(len(times))))
    velocity, position = np.linalg.lstsq(design, np.array([[p['north_m'], p['east_m']] for p in recent]), rcond=None)[0]
    speed2 = float(velocity @ velocity)
    if speed2 < 1e-6:
        return None
    tcpa = max(0., -float(position @ velocity) / speed2)
    return {'tcpa_horizontal_s': tcpa, 'dcpa_horizontal_m': float(np.linalg.norm(position + tcpa * velocity)),
            'fit_points': len(recent), 'history_s': float(times[-1] - times[0]),
            'meaning': 'reference_only_constant_velocity_to_stationary_camera_not_flight_guidance'}
