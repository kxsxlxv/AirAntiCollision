"""Reproduce the pinned horizontal-only checkpoint without mutating live code.

The committed 'initial' summary predates the NASA future-predicate repair. This
archive reproduces e4c5613's main falsification summary, not that earlier initial
summary. Both committed originals remain unchanged. No GPU or network is used.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
REVISION='e4c56133d714d7513bbca969306892a99ca69d8b'


def blob(path):
    return subprocess.check_output(['git','show',f'{REVISION}:{path}'],cwd=ROOT)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def run():
    folder=ROOT/'artifacts/stage2/falsification_horizontal_reproduction'
    folder.mkdir(parents=True,exist_ok=True)
    sources={name:blob(name) for name in ['airavoid/assessment.py','scripts/stage2_falsify.py']}
    shared={name:blob('airavoid/'+name) for name in ['geometry.py','projective.py','bounded.py','calibrated.py']}
    for name,data in shared.items():
        current=(ROOT/'airavoid'/name).read_text(encoding='utf-8').replace('\r\n','\n')
        if current!=data.decode('utf-8').replace('\r\n','\n'):
            raise ValueError(f'Historical shared baseline changed: {name}; reproduce in the pinned checkout')
    original=blob('reports/stage2_falsification.json')
    (folder/'original_summary.json').write_bytes(original)
    for name,data in sources.items():(folder/Path(name).name).write_bytes(data)
    module=types.ModuleType('airavoid._assessment_horizontal_checkpoint')
    module.__package__='airavoid';module.__file__=str(folder/'assessment.py')
    sys.modules[module.__name__]=module
    exec(compile(sources['airavoid/assessment.py'].decode('utf-8'),module.__file__,'exec'),module.__dict__)
    script=sources['scripts/stage2_falsify.py'].decode('utf-8')
    script=script.replace('from airavoid.assessment import','from airavoid._assessment_horizontal_checkpoint import')
    script=script.replace('artifacts/stage2/falsification','artifacts/stage2/falsification_horizontal_reproduction')
    script=script.replace('reports/stage2_falsification.json','reports/stage2_falsification_horizontal_reproduction.json')
    namespace={'__name__':'__main__','__file__':str(ROOT/'scripts/stage2_falsify.py')}
    exec(compile(script,str(folder/'stage2_falsify.py'),'exec'),namespace)
    old=json.loads(original)
    reproduced=ROOT/'reports/stage2_falsification_horizontal_reproduction.json'
    new=json.loads(reproduced.read_text())
    changed=[key for key in old if key!='full_cases' and old[key]!=new.get(key)]
    trace=folder/'cases.jsonl'
    report={'pinned_revision':REVISION,
            'source_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in sources.items()},
            'shared_baseline_source_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in shared.items()},
            'shared_baselines_verified_equal_to_pinned_sources':True,
            'original_summary_at_revision':'reports/stage2_falsification.json',
            'original_summary_sha256':hashlib.sha256(original).hexdigest(),
            'preserved_summary_snapshot':(folder/'original_summary.json').relative_to(ROOT).as_posix(),
            'archived_trace':trace.relative_to(ROOT).as_posix(),
            'archived_trace_sha256':sha(trace),'archived_trace_lines':sum(1 for _ in trace.open()),
            'reproduced_summary':reproduced.relative_to(ROOT).as_posix(),
            'reproduced_summary_sha256':sha(reproduced),
            'all_checkpoint_metrics_exactly_reproduced':not changed,'differing_metric_fields':changed,
            'earlier_initial_summary':'reports/stage2_falsification_initial.json',
            'earlier_initial_summary_scope':'Earlier instantaneous NASA WCV comparator had 210 critical false-safe; its source was not separately committed, so this archive does not claim to reproduce that unpinned implementation.',
            'reproduce_command':'.venv/Scripts/python.exe scripts/stage2_archive_initial.py',
            'execution_scope':'Pinned sources evaluated in memory; live assessor and current 3D falsification artifacts untouched; no GPU/network.'}
    (ROOT/'reports/stage2_falsification_initial_archive.json').write_text(json.dumps(report,indent=2))
    if changed:raise ValueError('Checkpoint metrics differ: '+str(changed))
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
