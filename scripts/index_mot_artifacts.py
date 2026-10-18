"""Index the completed research deliverables and measured source/configuration evidence."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
for name in ['mot_campaign_status','mot_followup_status','mot_finalize_status','mot_async_repeats','mot_last_checks','mot_confidence_status','mot_confidence_diff_control']:
    r=json.loads((ROOT/f'reports/{name}.json').read_text());assert r['complete'],name
    assert all(x['exit_code']==0 for x in r['runs']),name
for name in ['mot_export_qa','mot_video_qa']:
    assert json.loads((ROOT/f'reports/{name}.json').read_text())['complete']
tests=(ROOT/'reports/mot_tests.log').read_text();assert 'Ran 80 tests' in tests and '\nOK' in tests and 'skipped=' not in tests
def entry(path):
    b=path.read_bytes();assert b
    return dict(path=str(path.relative_to(ROOT)).replace('\\','/'),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
paths=[ROOT/'MOT_RESULTS.md',ROOT/'MOT_RESEARCH.md',ROOT/'data/mot_swarm/manifest.json',ROOT/'tests/test_mot_metrics.py']
paths.extend(p for p in (ROOT/'reports').glob('mot_*') if p.is_file() and p.name!='mot_artifact_index.json')
paths.extend(p for p in (ROOT/'reports/mot_motion').glob('*.json'))
paths.extend(p for p in (ROOT/'scripts').glob('mot*.py'))
paths.extend(ROOT/'scripts'/p for p in ['prepare_mot_swarm.py','write_mot_results.py','index_mot_artifacts.py'])
paths.extend(p for p in (ROOT/'demo/mot_comparison').glob('*') if p.is_file())
for phase in ['campaign','swarm','live','render_live','async_live','motion_only','swarm_detections','confidence18','confidence03']:
    paths.extend(p for p in (ROOT/f'artifacts/mot/{phase}').rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.mp4'])
r=dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),
       detector_unchanged=True,actual_model_names=['TrackTrack','BoT-SORT','OC-SORT','DiffMOT','PuTR','TrackSSM (repaired/reference)'],
       files=[entry(p) for p in sorted(set(paths))])
(ROOT/'reports/mot_artifact_index.json').write_text(json.dumps(r,indent=2));print('Indexed',len(r['files']),'files')
