"""Compare default A decisions with the pre-Stage-2 committed CLI on one cache.

The historical source is extracted read-only from Git into an excluded directory.
This is a cache/replay regression, never a new neural throughput measurement.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    reference = subprocess.check_output(['git', 'rev-parse', args.ref], cwd=ROOT, text=True).strip()
    folder = ROOT / 'external/stage2/default_regression_source' / reference
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / 'source.zip'
    subprocess.run(['git', 'archive', '--format=zip', '--output=' + str(archive), reference, 'airavoid'], cwd=ROOT, check=True)
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            target = (folder / item.filename).resolve()
            if not target.is_relative_to(folder.resolve()):
                raise ValueError('Unexpected archived path')
        source.extractall(folder)
    current = ROOT / 'artifacts/stage2/live_runtime_final/Clip_032_A_legacy_log_no_render'
    benchmark = json.loads((current / 'benchmark.json').read_text())
    output = ROOT / 'artifacts/stage2/default_regression'
    if (output / 'benchmark.json').exists():
        raise FileExistsError('Archive the previous default regression output before rerunning')
    command = [sys.executable, '-m', 'airavoid', benchmark['args']['video'],
               '--weights', str(ROOT / 'artifacts/yolo26m_local.engine'),
               '--detections', str(current / 'detections.jsonl'),
               '--output', str(output), '--camera', 'auto', '--stride', '1',
               '--max-frames', str(benchmark['frames']), '--no-render']
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'process.log').open('w') as sink:
        subprocess.run(command, cwd=folder, stdout=sink, stderr=subprocess.STDOUT, check=True)
    old = [json.loads(x) for x in (output / 'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    new = [json.loads(x) for x in (current / 'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    differences = []
    for i, (before, after) in enumerate(zip(old, new, strict=True)):
        # The added optional display label is a schema extension, with no A value.
        for obj in after['objects']:
            if obj.pop('class_display', None) is not None:
                raise ValueError('Default A unexpectedly has a taxonomy override')
        if before != after:
            differences.append(i)
    report = {'historical_commit': reference, 'frames': len(old),
              'same_default_decisions_after_optional_schema_extension': not differences,
              'different_frame_indices': differences,
              'ignored_schema_extension': 'objects[].class_display=null in Detector A',
              'current_live_run': str(current.relative_to(ROOT)),
              'historical_replay_run': str(output.relative_to(ROOT)),
              'command': command,
              'scope': 'Same genuine A detector cache, original RGB, causal original tracker, all decision fields; replay correctness, not neural FPS'}
    (ROOT / 'reports/stage2_default_regression.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if differences:
        raise RuntimeError('Original default decisions changed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', default='6f7a076', help='Pre-Stage-2 committed baseline')
    run(parser.parse_args())
