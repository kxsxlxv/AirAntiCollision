"""Refresh canonical outputs after the narrowly validated clipped-history repair."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from airavoid.provenance import file_sha256


def main():
    workspace=Path.cwd().resolve();archive=workspace/'artifacts/pre_clip_history'
    audit=json.loads(Path('reports/clip_history_audit.json').read_text());assert audit['complete']
    roots=[Path(v['directory']) for v in json.loads(Path('reports/video_qa.json').read_text())['videos']]
    for source in roots:
        destination=archive/source.relative_to('demo')
        if not destination.resolve().is_relative_to(archive.resolve()):raise ValueError('Archive must stay in owned directory')
        shutil.copytree(source,destination)
    earlier=Path('reports/pre_clip_history');earlier.mkdir(parents=True,exist_ok=True)
    for name in ['video_qa.json','sky_comparison.json','external_benchmark.json']:
        shutil.copyfile(Path('reports')/name,earlier/name)
    external=json.loads(Path('reports/external_benchmark.json').read_text())
    for row in external['results']:
        original=row['reference'];row['original_reference_before_archive']=original
        row['reference']=str(archive/Path(original).relative_to('demo'))
        assert all(file_sha256(Path(row['output'])/name)==file_sha256(Path(row['reference'])/name) for name in ['detections.jsonl','decisions.jsonl'])
    external['implementation_phase']='Before final exclusion of invalid clipped sizes; exact references archived, measurements unchanged'
    Path('reports/external_benchmark.json').write_text(json.dumps(external,indent=2))
    cases=[('M1','data/qut/M1.mp4','fixed',[]),('M2','data/qut/M2.mp4','fixed',[]),('S2','data/qut/S2.mp4','fixed',[]),
           ('formation','data/formation_native.mp4','auto',[]),('helicopter','data/helicopter_252.mp4','auto',[]),
           ('S2_zoom','data/qut/S2.mp4','fixed',['--zoom-proposals','--title','RGB • дополнительные YOLO-наблюдения без sky-модели'])]
    results=[]
    for name,source,camera,extra in cases:
        output=Path('demo')/(name+'_final');command=[sys.executable,'-m','airavoid',source,'--camera',camera,'--stride','2',
            '--encoder','h264_nvenc','--render-process','--output',str(output),*extra]
        started=time.perf_counter();timestamp=datetime.now(timezone.utc).isoformat()
        with Path('reports',name+'_clip_refresh.log').open('w',encoding='utf-8') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        elapsed=time.perf_counter()-started;b=json.loads((output/'benchmark.json').read_text())
        results.append({'case':name,'timestamp_utc':timestamp,'external_wall_s':elapsed,'external_fps':b['frames']/elapsed,
                        'processing_fps':b['processing_fps'],'frames':b['frames'],'states_track_frames':b['states_track_frames'],'output':str(output)})
        Path('reports/final_clip_refresh.json').write_text(json.dumps(results,indent=2));print(name,'refreshed',flush=True)
    with Path('reports/final_sky_clip_refresh.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-m','scripts.finalize_sky'],stdout=log,stderr=subprocess.STDOUT,check=True)
    command=[sys.executable,'-m','scripts.render_excerpt','data/qut_full/M6.mp4','artifacts/clip_history_audit/M6/decisions.jsonl',
             '--start','90','--end','102','--output','demo/M6_failure','--title','M6 • предупреждение по категории other: офлайн-разбор']
    with Path('reports/M6_clip_refresh.log').open('w',encoding='utf-8') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    # Verify current fixed-camera logic against fresh actual inference on a showcase.
    with Path('reports/clip_current_reconstruction.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-m','scripts.reassess_fixed',str(archive/'M2_final'),'--output','artifacts/current_reconstruction/M2',
                        '--image-edge-margin','1'],stdout=log,stderr=subprocess.STDOUT,check=True)
    replay=[json.loads(s) for s in Path('artifacts/current_reconstruction/M2/decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    live=[json.loads(s) for s in Path('demo/M2_final/decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    mismatch=[a['frame'] for a,b in zip(replay,live) if a!=b]
    report={'frames':len(live),'exact_parsed_trace_mismatch_frames':mismatch,'source':'Actual previous neural observations, current risk code',
            'reference':'Fresh actual current neural M2 inference','detector_observation_hash_equal':file_sha256(archive/'M2_final/detections.jsonl')==file_sha256('demo/M2_final/detections.jsonl')}
    Path('reports/clip_current_equivalence.json').write_text(json.dumps(report,indent=2))
    if mismatch or len(replay)!=len(live) or not report['detector_observation_hash_equal']:raise AssertionError('Current clipping replay did not reproduce live inference')


if __name__=='__main__':main()
