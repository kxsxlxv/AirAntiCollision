"""Index verified deliverables, exact traces and source-code fingerprints."""
from datetime import datetime,timezone
import json
from pathlib import Path
from airavoid.provenance import file_sha256


def entry(path):
    path=Path(path)
    if not path.is_file() or not path.stat().st_size:raise FileNotFoundError(path)
    return {'path':str(path).replace('\\','/'),'bytes':path.stat().st_size,'sha256':file_sha256(path)}


def main():
    qa=json.loads(Path('reports/video_qa.json').read_text());assert qa['status']=='complete'
    cadence=json.loads(Path('reports/cadence_all_sources.json').read_text());assert cadence['complete'] and len(cadence['runs'])==15
    oracle=json.loads(Path('reports/oracle_fullrate.json').read_text());assert oracle['complete']
    external=json.loads(Path('reports/external_benchmark.json').read_text());assert len(external['results'])==9
    assert all(all(r['exact_trace_equality'].values()) for r in external['results'])
    sky=json.loads(Path('reports/sky_comparison.json').read_text());assert sky['mask_free_vs_all_region_exact_trace_equality']
    env=json.loads(Path('reports/environment_final.json').read_text());assert all(v for k,v in env.items() if k.startswith('same_initial_'))
    tests=Path('reports/tests_latest.log').read_text();assert 'Ran 76 tests' in tests and '\nOK' in tests
    videos=[]
    for result in qa['videos']:
        root=Path(result['directory']);report=json.loads((root/'benchmark.json').read_text());files=[root/'annotated.mp4',root/'preview.png',root/'benchmark.json',root/'run_status.json']
        if (root/'decisions.jsonl').exists():files.extend([root/'detections.jsonl',root/'decisions.jsonl'])
        else:files.append(Path(report['decision_trace']))
        videos.append({'directory':str(root),'frames':result['frames_decoded'],'kind':result['kind'],
                       'files':[entry(path) for path in files]})
    reports=['reports/video_qa.json','reports/environment_final.json','reports/tests_latest.log','reports/external_benchmark.json',
             'reports/cadence_all_sources.json','reports/oracle_fullrate.json','reports/sky_comparison.json',
             'artifacts/confidence_full/campaign_summary.json','reports/bridge_extension.json','reports/selected_methods.json',
             'reports/residual_sensitivity.json','reports/confidence_floor_verification.json','reports/warning_timing.json',
             'reports/air2land_detection.json','reports/air2land_aspect.json','reports/small_raster_optimization.json',
             'reports/usage.jsonl','reports/mission_final.json']
    reports.extend(['reports/clip_history_audit.json','reports/clip_current_equivalence.json','reports/final_clip_refresh.json'])
    source_manifests=sorted(Path('data/qut_full').glob('*.manifest.json'))
    source_manifests.extend(p for p in Path('data').glob('*.json') if 'manifest' in p.name)
    gt_files=sorted(Path('data/qut_full').glob('*.gt.json'))
    docs=['README.md','RESULTS.md','RESEARCH.md','GEOMETRY.md','EXPERIMENTS.md','DELIVERY.md','demo/README.md','demo/index.html']
    index={'created_utc':datetime.now(timezone.utc).isoformat(),'entrypoint':'python -m airavoid INPUT.mp4 --camera auto --output OUTPUT',
           'videos':videos,'reports':[entry(p) for p in reports],'documentation':[entry(p) for p in docs],
           'source_manifests':[entry(p) for p in source_manifests],'offline_GT_only':[entry(p) for p in gt_files],
           'core_source':[entry(p) for p in sorted(Path('airavoid').glob('*.py'))],
           'tests':[entry(p) for p in sorted(Path('tests').glob('test_*.py'))],
           'external_models':'Supplied weights and read-only external adapters are identified in per-run benchmarks; locally built engines are runtime-specific',
           'scope':'Research prototype; input/output consistency is verified, physical collision/CPA accuracy and flight safety are not established'}
    Path('reports/artifact_index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    print('Indexed',len(videos),'verified videos and',len(index['core_source']),'core source files')


if __name__=='__main__':main()
