"""Compare actual serial/overlapped model decisions, PTS and decoded video pixels."""
import json
import argparse
from pathlib import Path
import subprocess
from airavoid.provenance import file_sha256

p=argparse.ArgumentParser();p.add_argument('--second',choices=['async','process'],default='async');p.add_argument('--output',default='reports/async_equivalence.json');args=p.parse_args()
paths=[Path('artifacts/async_trial')/name for name in ['serial',args.second]]
def frame_data(folder):
    video=folder/'annotated.mp4'
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries',
             'frame=best_effort_timestamp_time:stream=codec_name,pix_fmt,duration,nb_frames','-of','json',str(video)],text=True))
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(video),'-fps_mode','passthrough','-f','framemd5','-'],text=True)
    hashes=[line.split(',')[-1].strip() for line in raw.splitlines() if line and not line.startswith('#')]
    return meta,[float(x['best_effort_timestamp_time']) for x in meta['frames']],hashes

a,b=map(frame_data,paths)
report={'kind':'Both runs execute actual TensorRT neural inference; rendering schedule is the controlled change',
        'runs':[str(x) for x in paths],'decision_trace_identical':file_sha256(paths[0]/'decisions.jsonl')==file_sha256(paths[1]/'decisions.jsonl'),
        'neural_detections_identical':file_sha256(paths[0]/'detections.jsonl')==file_sha256(paths[1]/'detections.jsonl'),
        'frame_counts':[len(a[1]),len(b[1])],'max_pts_difference_s':max((abs(x-y) for x,y in zip(a[1],b[1])),default=0),
        'decoded_frame_pixel_mismatches':sum(x!=y for x,y in zip(a[2],b[2])),
        'streams':[a[0]['streams'][0],b[0]['streams'][0]]}
Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
assert report['decision_trace_identical'] and report['neural_detections_identical']
assert len(a[1])==len(b[1])==len(a[2])==len(b[2]) and report['max_pts_difference_s']<1e-8
assert report['decoded_frame_pixel_mismatches']==0
