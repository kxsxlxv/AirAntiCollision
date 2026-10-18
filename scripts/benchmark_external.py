"""Fresh-process rendered timing, including interpreter/import startup, no GT online."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from airavoid.provenance import file_sha256

cases=[('M2','data/qut/M2.mp4','fixed',[],'demo/M2_final'),
       ('S2_zoom','data/qut/S2.mp4','fixed',['--zoom-proposals'],'demo/S2_zoom_final'),
       ('formation','data/formation_native.mp4','auto',[],'demo/formation_final')]
results=[];root=Path('artifacts/external_benchmark');root.mkdir(parents=True,exist_ok=True)
for repeat in range(1,4):
    for name,video,camera,extra,reference in cases:
        output=root/f'{name}_repeat{repeat}';output.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-m','airavoid',video,'--camera',camera,'--stride','2','--encoder','h264_nvenc',
                 '--render-process','--output',str(output),*extra]
        timestamp=datetime.now(timezone.utc).isoformat();started=time.perf_counter()
        with (output/'run.log').open('w',encoding='utf-8') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        wall=time.perf_counter()-started;benchmark=json.loads((output/'benchmark.json').read_text())
        equality={f:file_sha256(output/f)==file_sha256(Path(reference)/f) for f in ['detections.jsonl','decisions.jsonl']}
        row={'case':name,'repeat':repeat,'start_utc':timestamp,'output':str(output),'command':command,
             'frames':benchmark['frames'],'external_process_wall_s':wall,'external_process_fps':benchmark['frames']/wall,
             'pipeline_processing_fps':benchmark['processing_fps'],'pipeline_whole_run_fps':benchmark['processed_frames_per_whole_run_s'],
             'reference':reference,'exact_trace_equality':equality,'stages':benchmark['stages']}
        results.append(row)
        Path('reports/external_benchmark.json').write_text(json.dumps({'kind':'Actual neural rendered fresh-process comparison; parent timer includes Python process/import startup',
            'results':results,'environment':'RTX4060, sequential fresh Python processes, warm machine; no simultaneous project GPU jobs',
            'limits':['Offline throughput, not streaming latency','OS/GPU/filesystem caches uncontrolled; not cold-boot timing',
                      'External wall timer excludes this launcher imports, includes child process imports, model startup and output completion',
                      'S2_zoom title differs from gallery; numeric traces compared, not encoded title pixels']},indent=2))
        print(name,repeat,round(row['external_process_fps'],3),'FPS',equality,flush=True)
        if not all(equality.values()):raise AssertionError('Current implementation differs from recorded reference traces')
