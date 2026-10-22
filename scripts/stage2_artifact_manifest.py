"""Check final evidence/source consistency and inventory reproducible local assets.

This performs no model inference, annotation changes or package operations.
Run after all comparison jobs and documentation generators have completed.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def write():
    checks = {}
    runtime = read('reports/stage2_runtime.json')
    checks['runtime_complete_14_conditions'] = runtime['state'] == 'complete' and len(runtime['runs']) == 14
    checks['runtime_actual_update_and_decoded_frame_counts'] = (
        runtime['total_live_updates'] == 2100 and runtime['total_annotated_decoded_frames'] == 1500)
    checks['measured_runtime_matches_current_code'] = all(
        digest(ROOT / 'airavoid' / name) == value
        for run in runtime['runs'] for name, value in run['benchmark']['source_hashes'].items())
    checks['render_and_thread_outputs_identical'] = all(
        item['detections_exactly_equal'] and item['decisions_exactly_equal']
        for key in ['serial_export_invariance', 'threaded_export_invariance'] for item in runtime[key])
    checks['all_encoded_frames_decoded_and_original_pts_preserved'] = all(
        run['video_qa']['all_frames_decoded']
        and run['video_qa']['max_output_pts_error_s'] <= 1 / 90000 + 1e-9
        and run['video_qa']['duration_error_s'] <= 1 / 90000 + 1e-9
        for run in runtime['runs'] if run['encoded'])
    checks['P2_replay_decisions_identical'] = read('reports/stage2_runtime_replay.json')['decisions_exactly_equal']
    checks['pre_Stage2_default_decisions_preserved'] = read('reports/stage2_default_regression.json')['same_default_decisions_after_optional_schema_extension']
    checks['original_environment_preserved'] = read('reports/stage2_environment_final.json')['all_checks_pass']
    suite = (ROOT / 'reports/stage2_tests.log').read_text(encoding='utf-8')
    nvenc = (ROOT / 'reports/stage2_nvenc_tests.log').read_text(encoding='utf-8')
    checks['116_regression_tests_no_failure'] = bool(re.search(r'Ran 116 tests', suite) and 'OK (skipped=1)' in suite)
    checks['optional_actual_NVENC_test_passed_separately'] = 'Ran 6 tests' in nvenc and nvenc.rstrip().endswith('OK')
    campaign = read('reports/stage2_assessment_campaign_status.json')
    checks['all_88_NPS_replays_completed'] = campaign['state'] == 'complete' and len(campaign['complete']) == campaign['planned_replays'] == 88
    matrix = read('reports/stage2_threat_matrix.json')
    assessment_hash = digest(ROOT / 'airavoid/assessment.py')
    replay_hash = digest(ROOT / 'scripts/stage2_assess_trace.py')
    checks['NPS_trace_and_implementation_hashes_match'] = True
    for condition, result in matrix['conditions'].items():
        for clip, score in result['per_sequence'].items():
            trace = ROOT / f'artifacts/stage2/assessments/{condition}/{clip}.jsonl.gz'
            summary = json.loads(trace.with_suffix('.summary.json').read_text())
            checks['NPS_trace_and_implementation_hashes_match'] &= (
                score['source_trace_sha256'] == summary['output_trace_sha256'] == digest(trace)
                and summary['assessment_source_sha256'] == assessment_hash
                and summary['replay_source_sha256'] == replay_hash
                and score['scorer_source_sha256'] == digest(ROOT / 'scripts/stage2_threat_score.py')
                and score['offline_matcher_source_sha256'] == digest(ROOT / 'scripts/stage2_score.py'))
    qut = read('reports/stage2_qut_comparison.json')
    checks['QUT_complete_18_conditions'] = qut['state'] == 'complete' and len(qut['runs']) == 18
    checks['QUT_trace_and_implementation_hashes_match'] = True
    for label, run in qut['runs'].items():
        trace = ROOT / f'artifacts/stage2/qut/{label}/assessments.jsonl.gz'
        summary = json.loads(trace.with_suffix('.summary.json').read_text())
        checks['QUT_trace_and_implementation_hashes_match'] &= (
            run['assessment_trace_sha256'] == summary['output_trace_sha256'] == digest(trace)
            and summary['assessment_source_sha256'] == assessment_hash
            and summary['replay_source_sha256'] == replay_hash)
    provenance = read('reports/stage2_method_provenance.json')
    checks['method_provenance_matches_current_implementation'] = provenance['implementation_sha256'] == assessment_hash
    agreement = read('reports/stage2_p2_agreement.json')['summary']
    checks['P2_ONNX_TensorRT_functional_agreement'] = (
        agreement['development_frames'] == 10 and agreement['matched_boxes'] == 19
        and agreement['all_accepted_counts_equal'] and agreement['all_accepted_classes_equal'])
    control = read('reports/stage2_camera_threat_control.json')
    checks['ORB_control_retains_current_replay_and_scoring'] = all(
        result['replay']['assessment_source_sha256'] == assessment_hash
        and result['replay']['replay_source_sha256'] == replay_hash
        and result['ORB_homography']['scorer_source_sha256'] == digest(ROOT / 'scripts/stage2_threat_score.py')
        for result in control['clips'].values())
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks,
              'all_checks_pass': all(checks.values()),
              'scope': 'Execution/source consistency, not scientific acceptance or flight safety',
              'scientific_acceptance': 'Partial; see docs/STAGE2_RESULTS.md'}
    (ROOT / 'reports/stage2_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_checks_pass']:
        raise RuntimeError('Final evidence checks failed: ' + ', '.join(k for k, v in checks.items() if not v))
    files = set()
    for directory, pattern in [('scripts', 'stage2_*.py'), ('tests', 'test_stage2*.py'),
                               ('reports', 'stage2*'), ('docs', 'stage2*'), ('data/stage2', '*')]:
        files.update(p for p in (ROOT / directory).glob(pattern) if p.is_file())
    files.update((ROOT / 'airavoid').glob('*.py'))
    files.update(ROOT / name for name in ['README.md', 'docs/STAGE2_RESULTS.md', 'research/stage2_selection.md',
        'research/monocular_collision_literature_and_code_survey.md',
        'docs/handoffs/STAGE_2_COLLISION_THREAT_RESEARCH.md'])
    manifest_path = ROOT / 'reports/stage2_artifact_manifest.json'
    files.discard(manifest_path)
    inventory = {p.relative_to(ROOT).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size}
                 for p in sorted(files)}
    local = {}
    for folder in ['detections', 'tracks', 'assessments', 'oracles', 'camera_orb', 'camera_threat_control',
                   'qut', 'falsification', 'live_runtime_final', 'default_regression']:
        parent = ROOT / 'artifacts/stage2' / folder
        for path in sorted(parent.rglob('*')):
            if path.is_file() and (path.suffix in ['.json', '.jsonl', '.gz', '.mp4']):
                local[path.relative_to(ROOT).as_posix()] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_snapshot_parent_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'note': 'File hashes identify this final working snapshot; parent commit is before its delivery commit. This manifest excludes itself.',
        'repository_files': inventory, 'excluded_local_evidence': local,
        'supplied_model_assets': provenance['detectors'],
        'source_video_and_annotation_assets': 'reports/stage2_dataset_audit.json, reports/stage2_protocol.json',
        'upstream_sources_and_licenses': 'reports/stage2_method_provenance.json',
        'excluded_asset_policy': 'Source videos, models, external repositories and bulk traces are local prerequisites, not GitHub-hosted downloads.',
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'all_checks_pass': True, 'repository_files': len(inventory),
                      'excluded_local_evidence_files': len(local)}, indent=2))


if __name__ == '__main__':
    write()
