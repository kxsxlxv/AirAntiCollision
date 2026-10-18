"""Real RGB temporal-resampling stress, checks source/trace/output PTS independently."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

source=Path('data/vfr_M2_stress.mp4')
select="select='if(lt(t,4),not(mod(n,2)),not(mod(n,5)))'"
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i','data/qut/M2.mp4','-t','8','-vf',select,'-fps_mode','vfr','-an','-c:v','libx264','-preset','fast','-crf','17',str(source)],check=True)
out=Path('artifacts/vfr_trial')
subprocess.run([sys.executable,'-m','airavoid',str(source),'--gpu-preprocess','--camera','fixed','--output',str(out),'--title','RGB • проверка переменных временных интервалов'],check=True,stdout=(Path('reports/vfr_cli.log')).open('w'),stderr=subprocess.STDOUT)
def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time:stream=duration,codec_name:format=duration','-of','json',str(path)],text=True))
a=probe(source);b=probe(out/'annotated.mp4')
packets=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_packets','-show_entries','packet=pts_time,duration_time','-of','json',str(source)],text=True))['packets']
source_end=max(float(p['pts_time'])+float(p.get('duration_time',0)) for p in packets if 'pts_time' in p)
source_pts=np.array([float(f['best_effort_timestamp_time']) for f in a['frames']]);source_pts-=source_pts[0]
output_pts=np.array([float(f['best_effort_timestamp_time']) for f in b['frames']])
trace_pts=np.array([r['time'] for r in map(json.loads,(out/'decisions.jsonl').read_text().splitlines())])
report={'kind':'temporal stress derived from actual RGB, not a new physical encounter','source_frames':len(source_pts),'output_frames':len(output_pts),'source_intervals_s':np.unique(np.round(np.diff(source_pts),6)).tolist(),'max_trace_pts_error_s':float(np.max(np.abs(source_pts-trace_pts))),'max_output_pts_error_s':float(np.max(np.abs(source_pts-output_pts))),'source_packet_presentation_end_s':source_end,'source_container_duration_metadata_s':float(a['format']['duration']),'source_stream_duration_metadata_s':float(a['streams'][0]['duration']),'output_duration_s':float(b['format']['duration']),'codec':b['streams'][0]['codec_name'],'timestamp_mode':json.loads((out/'benchmark.json').read_text())['timestamp_mode'],'metadata_conflict_note':'Source duration metadata ends before its last displayed frame; output retains every presentation timestamp and the final packet duration.'}
Path('reports/vfr_trial.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
assert report['max_trace_pts_error_s']<1e-5
assert report['max_output_pts_error_s']<1/90000+1e-6
assert abs(report['source_packet_presentation_end_s']-report['output_duration_s'])<1/90000+1e-6
