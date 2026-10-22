"""Inventory completed MVP media and audit causal measured cue support; no GT."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    validation_files=['final_demo_validation.json','final_demo_contrast_validation.json','final_demo_approach_validation.json']
    summary=[];inventory={};errors=[]
    for filename in validation_files:
        validation=json.loads((ROOT/'reports'/filename).read_text(encoding='utf-8'))
        assert validation['state']=='complete' and all(validation['checks'].values()),filename
        folder=Path(validation['run']);rows=[json.loads(x) for x in (folder/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
        detections=[json.loads(x) for x in (folder/'detections.jsonl').read_text().splitlines()]
        assert len(rows)==len(detections)
        resolved=warnings=0;fresh_samples=0
        for row,observed in zip(rows,detections,strict=True):
            assert row['frame']==observed['frame'] and row['time']==observed['time']
            t=row['time'];cue=row['advisory']['demo'];objects={o['id']:o for o in row['objects']}
            for target in cue['targets']:
                trail=target['measured_trail']
                if any(p[0]>t+1e-7 for p in trail):errors.append([folder.name,row['frame'],'future_support'])
                obj=objects[target['id']];guard=obj['demo_observation']
                if guard['measured']:
                    fresh_samples+=1
                    if not any(max(abs(float(a)-float(b)) for a,b in zip(obj['box'],raw[:4]))<1e-3 for raw in observed['detections']):
                        errors.append([folder.name,row['frame'],'box_not_an_actual_detector_measurement'])
                if target['state']=='POSSIBLE CONVERGENCE':
                    warnings+=1
                    if not (guard['measured'] and not guard['ambiguous'] and not guard['clipped'] and row['camera']['valid']):
                        errors.append([folder.name,row['frame'],'warning_without_valid_measured_support'])
            if cue['direction']['status']=='resolved':
                resolved+=1;target=next(x for x in cue['targets'] if x['id']==cue['priority_id'])
                obj=objects[target['id']];guard=obj['demo_observation']
                if not (guard['measured'] and not guard['ambiguous'] and not guard['clipped'] and row['camera']['valid'] and target['state']!='INSUFFICIENT EVIDENCE'):
                    errors.append([folder.name,row['frame'],'arrow_without_reliable_measured_support'])
        for path in folder.iterdir():
            if path.is_file():inventory[path.relative_to(ROOT).as_posix()]={'sha256':sha(path),'bytes':path.stat().st_size,'local_only_video':path.suffix=='.mp4'}
        summary.append({'run':folder.relative_to(ROOT).as_posix(),'frames':len(rows),'fresh_detector_target_samples':fresh_samples,
            'resolved_arrow_frames':resolved,'possible_convergence_target_samples':warnings,
            'duration_s':validation['encoded_video']['duration_s'],'detector':validation['actual_detector_contract']['variant'],
            'source_sha256':validation['source']['sha256'],'output_video_sha256':validation['encoded_video']['sha256']})
    assert not errors,errors[:10]
    default=json.loads((ROOT/'reports/final_demo_default_regression.json').read_text())
    assert default['same_default_decisions']
    for name,count in [('final_demo_tests.log',21),('final_demo_runtime_regression.log',13)]:
        text=(ROOT/'reports'/name).read_text();assert f'Ran {count} tests' in text and text.rstrip().endswith('OK')
    source_files=[ROOT/'README.md',ROOT/'docs/FINAL_DEMO.md',ROOT/'demo/FINAL_DEMO.html']
    for folder,pattern in [('airavoid','*.py'),('scripts','final_demo*.py'),('tests','test_demo*.py'),('reports','final_demo*')]:
        source_files.extend(p for p in (ROOT/folder).glob(pattern) if p.is_file() and p.name not in ['final_demo_artifacts.json','final_demo_audit.log'])
    for path in source_files:inventory[path.relative_to(ROOT).as_posix()]={'sha256':sha(path),'bytes':path.stat().st_size}
    report={'created_utc':datetime.now(timezone.utc).isoformat(),
        'implementation_commit':subprocess.check_output(['git','rev-parse','5b4cbd7'],cwd=ROOT,text=True).strip(),
        'pull_request':'https://github.com/kxsxlxv/AirAntiCollision/pull/3','runs':summary,
        'total_real_neural_updates':sum(x['frames'] for x in summary),'causality_or_measured_support_errors':errors,
        'default_150_A_frames_exactly_preserved':True,'focused_tests_passed':34,
        'assets':inventory,'scope':'Completed selected concept demonstrations; no physical collision accuracy or safe maneuver claim',
        'media_policy':'MP4 files are completed local exports excluded from Git. Relative paths refer to this workspace; screenshots, traces and reports are committed.'}
    output=ROOT/'reports/final_demo_artifacts.json';output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='assets'},indent=2))


if __name__=='__main__':main()
