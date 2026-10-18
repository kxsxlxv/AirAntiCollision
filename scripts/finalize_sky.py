"""Complete the current three-way S2 proposal comparison after the queued zoom video."""
import json
from pathlib import Path
import subprocess
import sys
import time
from scripts.evaluate import evaluate
from airavoid.provenance import file_sha256


def main():
    deadline=time.monotonic()+1800;zoom=Path('demo/S2_zoom_final')
    while not (zoom/'benchmark.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Queued zoom video did not finish')
        if (zoom/'run_status.json').exists() and json.loads((zoom/'run_status.json').read_text())['state']=='failed':raise RuntimeError('Queued zoom run failed')
        time.sleep(2)
    sky=Path('demo/S2_sky_current');python=sys.executable
    with Path('reports/sky_current_workflow.log').open('w',encoding='utf-8') as log:
        subprocess.run([python,'-m','scripts.run_sky_demo','data/qut/S2.mp4','--camera','fixed','--stride','2',
            '--encoder','h264_nvenc','--render-process','--output',str(sky)],stdout=log,stderr=subprocess.STDOUT,check=True)
    control=Path('artifacts/sky_control_current/S2_all_region')
    with Path('reports/sky_current_control.log').open('w',encoding='utf-8') as log:
        subprocess.run([python,'-m','airavoid','data/qut/S2.mp4','--camera','fixed','--stride','2','--no-render',
            '--sky-mask',str(sky/'sky_overlay_mask.mkv'),'--sky-all-control','--output',str(control)],stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=[]
    for method,root in [('dense',Path('demo/S2_final')),('zoom',zoom),('sky',sky/'assessment'),('all_region_control',control)]:
        r=evaluate(root/'decisions.jsonl','data/qut/S2.gt.json');(root/'evaluation.json').write_text(json.dumps(r,indent=2))
        b=json.loads((root/'benchmark.json').read_text())
        rows.append({'method':method,'directory':str(root),'counts':r['counts'],'events':r['events'],
                     'detection_sha256':file_sha256(root/'detections.jsonl'),'decision_sha256':file_sha256(root/'decisions.jsonl'),
                     'processing_fps':b['processing_fps'],'pipeline_whole_run_fps':b['processed_frames_per_whole_run_s'],
                     'network_calls_mean':b['stages']['network_calls']['mean'],'proposal_calls_mean':b['stages'].get('proposal_network_calls',{}).get('mean')})
    zoom_row=next(r for r in rows if r['method']=='zoom');control_row=next(r for r in rows if r['method']=='all_region_control')
    equal=all(zoom_row[k]==control_row[k] for k in ['detection_sha256','decision_sha256'])
    report={'kind':'Current crop-boundary-guarded actual neural S2 comparison, same source and stride2',
            'rows':rows,'mask_free_vs_all_region_exact_trace_equality':equal,
            'sky_workflow':json.loads((sky/'workflow_benchmark.json').read_text()),
            'limits':['Single studied clip; no independent scene-wide sky or zoom benefit claim','No collision/CPA truth',
                      'Control has no rendering; its FPS is not comparable to rendered paths']}
    Path('reports/sky_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not equal:raise AssertionError('Mask-free all-region path differs from matched control')


if __name__=='__main__':main()
