"""Render the completed QUT category-proxy comparison without rerunning inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write():
    source = ROOT / 'reports/stage2_qut_comparison.json'
    report = json.loads(source.read_text())
    if report.get('state') != 'complete':
        raise ValueError('QUT comparison is incomplete')
    lines = [
        '# Stage 2 QUT category-proxy results', '',
        'Frozen original Detector A observations; fresh causal association at each '
        'cadence. Six complete available recordings: ' + ', '.join(report['sequences']) + '.', '',
        'These recordings influenced prior development. Approximate target-centre '
        'annotations are joined offline with the historical 20 px / .25 s freshness '
        'contract. Approach is not verified collision; other is not verified safety. '
        'This subset does not reproduce the previous complete 15-recording score.', '',
        'KLT camera diagnostics are frozen from the original full-rate fixed-camera '
        'run. Stride changes association and assessment only. All methods receive '
        'identical fresh measured boxes/IDs/timestamps within each condition. '
        'No category or GT identity enters normal tracking or assessment.', '',
        '| Stride | Method | Approach warning | Approach nonconflict | Approach uncertain | Approach untracked | Other warning | Other nonconflict | Other uncertain | Other untracked |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    keys = ['potential_conflict', 'apparent_nonconflict', 'uncertain', 'untracked']
    for stride in report['strides']:
        for method in report['methods']:
            counts = {'approach': Counter(), 'other': Counter()}
            for run in report['runs'].values():
                if run['stride'] == stride:
                    for kind in counts:
                        counts[kind].update(run['methods'][method]['category_counts'].get(kind, {}))
            values = [str(counts[kind][key]) for kind in counts for key in keys]
            lines.append('| ' + ' | '.join([str(stride), method] + values) + ' |')
    lines += [
        '',
        'For Lai–Lin at full rate, 670 approach-category target samples receive '
        'apparent nonconflict despite 1,544 warnings. This falsifies interpreting a '
        'low image-risk index as clearance on this proxy; it does not measure a '
        'physical collision false-safe rate. The live adapter withholds that '
        'clearance interpretation; the research table retains the raw reproduction.', '',
        'Bauer requires verified calibration; NASA requires metric state; RV requires '
        'calibrated, independently de-rotated target feature flow. Their abstentions '
        'here are missing-prerequisite outcomes. AirTrack has no invented collision '
        'classifier. Zero warnings or clearances is not successful discrimination.', '',
        'Per-sequence counts, raw candidates, warning episodes per camera hour and '
        'warning timing relative to first/last available annotation are retained in '
        '[stage2_qut_comparison.json](stage2_qut_comparison.json). Timing is never '
        'referenced to inferred impact. Source-video, historical-cache and resulting '
        'observation/assessment hashes are included per run.', '',
        'Reproduce the research outputs with `scripts/stage2_qut.py`; regenerate this '
        'table with `scripts/stage2_write_qut_results.py`.', '',
        'Source report SHA256: `' + hashlib.sha256(source.read_bytes()).hexdigest() + '`.',
    ]
    target = ROOT / 'reports/stage2_qut_results.md'
    target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    write()
