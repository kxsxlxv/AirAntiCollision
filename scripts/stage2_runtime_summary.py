"""Format completed live measurements without running or extrapolating models."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def write():
    path=ROOT/'reports/stage2_runtime.json'
    report=json.loads(path.read_text())
    if report['state']!='complete':raise ValueError('Live benchmark is incomplete')
    rows=[]
    for run in report['runs']:
        b=run['benchmark'];lat=b['stages']['end_to_end_sample_ms']
        rows.append({'clip':run['clip'],'detector':run['detector'],'assessment':run['method'],
            'export':('NVENC threaded' if b['stage_overlap'] else 'NVENC serial') if run['encoded'] else 'none','frames':b['frames'],
            'source_fps':b['source_fps'],'sampled_source_fps':b['sampled_source_fps'],
            'processing_fps':b['processing_fps'],'p50_ms':lat['p50'],
            'p95_ms':lat['p95'],'p99_ms':lat['p99'],
            'network_calls_per_update':b['network_calls_total']/b['frames'],
            'model_load_and_provenance_s':b['cold_start']['model_load_and_provenance_s'],
            'first_detector_update_ms':b['cold_start']['first_inference_ms']})
    with (ROOT/'reports/stage2_runtime.csv').open('w',newline='',encoding='utf-8') as sink:
        writer=csv.DictWriter(sink,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    invariance=[]
    for clip in sorted({r['clip'] for r in report['runs']}):
        for variant in ['A','B']:
            pair=[r for r in report['runs'] if r['clip']==clip and r['detector']==variant and r['method']=='legacy_log' and not r['benchmark']['stage_overlap']]
            if len(pair)!=2:continue
            invariance.append({'clip':clip,'detector':variant,
                'detections_exactly_equal':pair[0]['detections_sha256']==pair[1]['detections_sha256'],
                'decisions_exactly_equal':pair[0]['decisions_sha256']==pair[1]['decisions_sha256']})
    report['serial_export_invariance']=invariance
    report['scope']='Actual original-video decode + live neural detector + custom predicted tracker + assessment; primary serial NVENC export and separately labelled bounded render-thread controls'
    report['fixed_wing_truth_basis']='User confirmed all targets in the six selected NPS clips are fixed-wing; not independent per-frame class annotations'
    report['threaded_export_invariance']=[]
    for run in report['runs']:
        if not run['benchmark']['stage_overlap']:continue
        control=next(r for r in report['runs'] if r['clip']==run['clip'] and r['detector']==run['detector']
            and r['method']==run['method'] and not r['encoded'])
        report['threaded_export_invariance'].append({'clip':run['clip'],'detector':run['detector'],
            'detections_exactly_equal':run['detections_sha256']==control['detections_sha256'],
            'decisions_exactly_equal':run['decisions_sha256']==control['decisions_sha256']})
    report['total_live_updates']=sum(r['benchmark']['frames'] for r in report['runs'])
    report['total_network_calls']=sum(r['benchmark']['network_calls_total'] for r in report['runs'])
    report['total_annotated_decoded_frames']=sum(r['video_qa']['decoded_frames'] for r in report['runs'] if r['video_qa'])
    report['pipeline_source_hashes_identical_across_runs']=all(
        r['benchmark']['source_hashes']==report['runs'][0]['benchmark']['source_hashes'] for r in report['runs'])
    report['measured_pipeline_matches_current_sources']=all(
        hashlib.sha256((ROOT/'airavoid'/name).read_bytes()).hexdigest()==digest
        for name,digest in report['runs'][0]['benchmark']['source_hashes'].items())
    path.write_text(json.dumps(report,indent=2))
    marker='<!-- MEASURED_RUNTIME -->'
    doc=ROOT/'docs/stage2_runtime.md'
    text=doc.read_text(encoding='utf-8').split(marker)[0].rstrip()
    lines=[marker,'','## Actual RTX 4060 results','',
        'Sequential fresh-process measurements, 22 October 2026; 150 consecutive source frames per condition. '
        'A uses 10 model calls/update at 1280×960 and 17 at 1920×1080; B uses one. '
        'Source cadence is 29 Hz for Clip 032 and 29.97 Hz for Clip 001; stride is one. '
        'Reported FPS is offline processing throughput and includes the final encoder drain. '
        'These short samples do not certify sustained streaming or tail latency.','',
        '| Clip | Detector | Assessment | Export | FPS | p50 ms | p95 ms | p99 ms |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['clip']} | {row['detector']} | {row['assessment']} | {row['export']} | {row['processing_fps']:.2f} | {row['p50_ms']:.1f} | {row['p95_ms']:.1f} | {row['p99_ms']:.1f} |")
    lines+=['',
        'Cold-start fields are recorded separately for each condition in the JSON/CSV. '
        'The first inference is a complete detector update, including all A crops; it is not one network invocation. '
        'Processes are fresh, while the operating-system file cache is uncontrolled. '
        'The environment remains the original custom CUDA-enabled OpenCV 5 / Torch 2.12 / TensorRT 11.4 stack.','',
        'Threaded controls use the existing bounded two-frame render/export worker. '
        'Their FPS includes completion of all queued work and encoder drain, while frame percentiles measure '
        'decode through submission/backpressure, not each frame’s completed encoding latency.','',
        f"The matrix performed {report['total_live_updates']} genuine neural-detector updates. "
        f"All {report['total_annotated_decoded_frames']} annotated frames were fully decoded and checked against original PTS, "
        'including frame count, H.264 codec and presentation duration. '
        'Serial encoding invariance is recorded per matched A/B condition in the JSON.','',
        'The source of the table is `reports/stage2_runtime.json`; '
        '`scripts/stage2_runtime_summary.py` regenerates it and the compact CSV. '
        'No Jetson timing is inferred from these desktop measurements.']
    doc.write_text(text+'\n\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'runs':len(rows),'live_updates':report['total_live_updates'],
        'decoded_annotated_frames':report['total_annotated_decoded_frames'],
        'serial_export_invariance':invariance,
        'pipeline_source_hashes_identical_across_runs':report['pipeline_source_hashes_identical_across_runs']},indent=2))


if __name__=='__main__':write()
