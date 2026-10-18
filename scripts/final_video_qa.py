"""Decode final videos, validate PTS/count/codec and compare sampled rendered frames."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import av
import cv2
import numpy as np
from airavoid.render import Renderer
from airavoid.tracking import Tracker
from airavoid.provenance import file_sha256


ROOTS=['M1_final','M2_final','S2_final','S2_zoom_final','formation_final','helicopter_final','M6_failure','S2_sky_current/assessment']


def check_video(root):
    root=Path('demo')/root;benchmark=json.loads((root/'benchmark.json').read_text());excerpt='decision_trace' in benchmark
    source=benchmark['input'] if excerpt else benchmark['args']['video']
    trace=Path(benchmark['decision_trace']) if excerpt else root/'decisions.jsonl'
    rows=[json.loads(s) for s in trace.read_text(encoding='utf-8').splitlines()]
    selected=rows
    if excerpt:
        start,end=benchmark['source_time_interval_s'];selected=[r for r in rows if start<=r['time']<end]
    origin=selected[0]['time'];expected=np.array([r['time']-origin for r in selected]);samples={0,len(selected)//2,len(selected)-1}
    decoded={};times=[];digest=hashlib.sha256()
    with av.open(str(root/'annotated.mp4')) as container:
        for index,frame in enumerate(container.decode(video=0)):
            times.append(float(frame.pts*frame.time_base));array=frame.to_ndarray(format='bgr24')
            digest.update(array.tobytes())
            if index in samples:decoded[index]=array
    if len(times)!=len(selected) or len(times)!=benchmark['frames']:raise AssertionError('Frame count mismatch '+str(root))
    pts_error=float(np.max(np.abs(np.array(times)-expected)))
    if pts_error>1.1/90000:raise AssertionError('Source PTS mismatch '+str(root))
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=codec_name,pix_fmt,width,height,duration,nb_frames','-of','json',str(root/'annotated.mp4')],text=True))['streams'][0]
    assert probe['codec_name']=='h264' and probe['pix_fmt']=='yuv420p' and (probe['width'],probe['height'])==(1800,1140)
    target_end=benchmark['source_time_interval_s'][1]-origin if excerpt else benchmark['output_end_time_s']-origin
    if abs(float(probe['duration'])-target_end)>1.1/90000:raise AssertionError('Duration mismatch '+str(root))
    if file_sha256(source)!=benchmark.get('input_sha256'):raise AssertionError('Source identity changed '+str(root))
    observations={r['frame']:r for r in map(json.loads,trace.with_name('detections.jsonl').read_text().splitlines())}
    tracker=Tracker(image_edge_margin=1);cap=cv2.VideoCapture(str(source));renderer=Renderer(benchmark['title'] if excerpt else benchmark['args']['title'])
    selected_map={r['frame']:i for i,r in enumerate(selected)};image_checks=[]
    for row in rows:
        if row['frame']>selected[-1]['frame']:break
        entry=observations[row['frame']]
        if row['camera'].get('evidence_reset'):tracker.reset_evidence()
        tracks=tracker.update(entry.get('observations',entry['detections']),row['time'],np.eye(3))
        index=selected_map.get(row['frame'])
        if index not in samples:continue
        histories={tr.id:list(tr.raw_history) for tr in tracks}
        for obj in row['objects']:obj['raw_history']=histories[obj['id']]
        cap.set(cv2.CAP_PROP_POS_FRAMES,row['frame']);ok,frame=cap.read();assert ok
        actual=renderer.render(frame,row['objects'],row['time'],row['camera'],row['advisory'])
        error=np.square(actual.astype(np.float32)-decoded[index].astype(np.float32));rmse=float(np.sqrt(np.mean(error)))
        if rmse>12:raise AssertionError('Rendered/sample mismatch '+str(root)+' RMSE='+str(rmse))
        image_checks.append({'output_frame':index,'source_frame':row['frame'],'source_time_s':row['time'],'decoded_vs_uncompressed_expected_rmse':rmse})
    cap.release()
    return {'directory':str(root),'kind':benchmark.get('kind','Live neural assessment'),'video_sha256':file_sha256(root/'annotated.mp4'),
            'source_sha256':file_sha256(source),'trace_sha256':file_sha256(trace),'frames_decoded':len(times),
            'max_pts_error_s':pts_error,'media':probe,'decoded_pixels_sha256':digest.hexdigest(),'rendered_sample_checks':image_checks,
            'interpretation':'Lossy H264 is compared with the corresponding source/recorded-state rendering, not with collision ground truth'}


if __name__=='__main__':
    results=[]
    for root in ROOTS:
        results.append(check_video(root));print(root,'verified',results[-1]['frames_decoded'],flush=True)
        Path('reports/video_qa.json').write_text(json.dumps({'status':'in_progress','videos':results},indent=2))
    Path('reports/video_qa.json').write_text(json.dumps({'status':'complete','videos':results,'limitations':[
        'Three sampled frames per video compare source-to-overlay alignment; all frames decode and all PTS are checked',
        'Rendered-frame agreement verifies export consistency, not scientific validity or physical threat truth']},indent=2))
