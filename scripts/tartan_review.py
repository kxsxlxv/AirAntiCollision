"""Offline review video: white reference boxes are explicitly NOT inference.

Reads completed neural traces, original RGB, annotations and ADS-B solely for
evaluation. Does not generate detector observations or change any cue decision.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from airavoid.media import TimestampWriter
from airavoid.provenance import file_sha256
from airavoid.tartan_reference import read_clock, paired_points, read_labels, match_objects, interpolate_reference


def render(frame, row, labels, points, origin):
    canvas = Image.new('RGB', (1920, 1080), '#09131f')
    draw = ImageDraw.Draw(canvas)
    fonts = {n: ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', n) for n in (20, 24, 28, 36)}
    def text(x, y, value, size=24, color='#d6e6f5'):
        draw.text((x, y), value, font=fonts[size], fill=color)
    text(30, 18, 'TARTAN • ПРОВЕРКА ПО ЗАПИСАННОЙ ТРАЕКТОРИИ', 36)
    text(30, 70, 'БЕЛЫЙ: ЭТАЛОН, ТОЛЬКО ДЛЯ ОЦЕНКИ    |    ЦВЕТ: ПРЕДСКАЗАНИЯ YOLO / ТРЕКЕРА', 24)
    h, w = frame.shape[:2]
    scale = min(1250 / w, 850 / h)
    fw, fh = round(w * scale), round(h * scale)
    x0, y0 = 30 + (1250 - fw) // 2, 145 + (850 - fh) // 2
    canvas.paste(Image.fromarray(cv2.cvtColor(cv2.resize(frame, (fw, fh)), cv2.COLOR_BGR2RGB)), (x0, y0))
    def coords(box):
        return [x0 + box[0] * scale, y0 + box[1] * scale, x0 + box[2] * scale, y0 + box[3] * scale]
    cues = {c['id']: c for c in row['advisory']['demo']['targets']}
    fresh = [o for o in row['objects'] if abs(o['last_seen'] - row['time']) < 1e-6]
    for obj in fresh:
        color = '#ffbc60' if cues[obj['id']]['state'] == 'POSSIBLE CONVERGENCE' else '#46d2e9'
        box = coords(obj['box'])
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0], max(y0, box[1] - 22)), f"{obj['id']} {obj['class']}", font=fonts[20], fill=color,
                  stroke_width=2, stroke_fill='#09131f')
    for gt in labels:
        b = coords(gt['box'])
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        draw.ellipse((cx - 19, cy - 19, cx + 19, cy + 19), outline='#09131f', width=4)
        draw.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), outline='white', width=2)
        draw.line((cx + 18, cy, cx + 100, cy - 50), fill='white', width=2)
        draw.text((cx + 28 if cx < x0 + fw - 180 else cx - 180, cy - 80), 'ЭТАЛОН A319', font=fonts[24], fill='white',
                  stroke_width=3, stroke_fill='#09131f')
    text(1320, 140, f"ВРЕМЯ КАМЕРЫ   {row['time']:05.2f} c", 28)
    text(1320, 192, 'ADS-B • ТОЛЬКО ОФЛАЙН-ЭТАЛОН', 24, '#ffbc60')
    north = [p['north_m'] for p in points]
    east = [p['east_m'] for p in points]
    def xy(n, e):
        return 1400 + e * .052, 720 - n * .052
    for v in range(0, 9000, 1000):
        draw.line((1350, xy(v, 0)[1], 1840, xy(v, 0)[1]), fill='#253b4c')
        draw.line((xy(0, v)[0], 270, xy(0, v)[0], 760), fill='#253b4c')
    draw.line([xy(n, e) for n, e in zip(north, east)], fill='#62768a', width=3)
    cx, cy = xy(0, 0)
    draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill='white')
    text(1320, 770, 'Камера на земле • оси N / E', 20)
    ref = interpolate_reference(points, points[0]['icao'], row['utc_s'], max_gap=3.1)
    if ref:
        x, y = xy(ref['north_m'], ref['east_m'])
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill='#ffbc60')
        draw.line((cx, cy, x, y), fill='#ffbc60', width=2)
        text(1320, 820, f"Горизонтальная дальность ≈ {ref['horizontal_range_m']/1000:.2f} км", 24, '#ffbc60')
    matches = match_objects(fresh, labels)
    text(1320, 875, f"A319 сопоставлен: {'ДА' if matches else 'НЕТ'}", 24)
    if matches:
        obj = matches[0][0]
        states = {'MONITORING': 'КОНТРОЛЬ', 'INSUFFICIENT EVIDENCE': 'НЕДОСТАТОЧНО ДАННЫХ', 'POSSIBLE CONVERGENCE': 'ВОЗМ. СБЛИЖЕНИЕ'}
        text(1320, 920, states[cues[obj['id']]['state']], 24, '#46d2e9')
        text(1320, 960, f"ID {obj['id']} • класс: {obj['class']}", 24)
    text(30, 1030, 'ОФЛАЙН-ОЦЕНКА. Эталонные рамки и ADS-B не поступали в детектор, трекер или предупреждения.', 24, '#ffbc60')
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default='artifacts/tartan_jun26')
    parser.add_argument('--run', default='demo/tartan_jun26')
    parser.add_argument('--output', default='demo/tartan_trajectory_review')
    args = parser.parse_args()
    folder, run, output = Path(args.folder), Path(args.run), Path(args.output)
    if output.exists():
        parser.error('Choose a new output directory')
    benchmark = json.loads((run / 'benchmark.json').read_text())
    video = next(folder.glob('*.mp4'))
    clock_file = next(folder.glob('*.srt'))
    if benchmark['input_sha256'] != file_sha256(video) or benchmark['clock_sha256'] != file_sha256(clock_file):
        raise ValueError('Review source differs from live inference')
    clock = read_clock(clock_file)
    points = paired_points(next(folder.glob('*_sink_adsb.pkl')), clock)
    labels = read_labels(folder)
    rows = {r['frame']: r for r in map(json.loads, (run / 'decisions.jsonl').read_text(encoding='utf-8').splitlines())}
    output.mkdir(parents=True)
    cap = cv2.VideoCapture(str(video))
    writer = TimestampWriter(output / 'evaluation_reference.mp4', 12, (1920, 1080), encoder='h264_nvenc')
    frame_index = rendered = 0
    captures = {'approach': 490, 'nearest_pass': 806, 'receding': 1200}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            row = rows.get(frame_index)
            if row:
                image = render(frame, row, labels.get(row['ordinal'], []), points, clock[2])
                writer.write(image, row['time'])
                rendered += 1
                for name, index in captures.items():
                    if frame_index == index:
                        cv2.imwrite(str(output / (name + '.png')), image)
            frame_index += 1
        writer.close(benchmark['acquisition_duration_s'])
    finally:
        cap.release()
        writer.abort()
    report = {'kind': 'offline_reference_review_not_inference', 'frames': rendered,
              'upstream_benchmark_sha256': file_sha256(run / 'benchmark.json'),
              'output_sha256': file_sha256(output / 'evaluation_reference.mp4'),
              'GT_is_drawn_only_here': True, 'runtime_decisions_changed': False}
    (output / 'review.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
