"""Paired isolated encoder probe, not application or neural inference throughput."""
import json
from pathlib import Path
import time
import subprocess
import cv2
import numpy as np
from airavoid.media import TimestampWriter

root=Path('artifacts/codec_probe');root.mkdir(parents=True,exist_ok=True)
source='artifacts/async_trial/serial/annotated.mp4';cap=cv2.VideoCapture(source);frames=[]
for i in range(20):
    ok,f=cap.read()
    if not ok:raise ValueError('Probe source ended early')
    frames.append(f)
cap.release();results=[];total=240
for rgb in [False,True]:
    path=root/('rgb_input.mp4' if rgb else 'yuv_input.mp4');tick=time.perf_counter()
    w=TimestampWriter(path,12.5,(1800,1140),encoder='h264_nvenc',nvenc_rgb_input=rgb)
    timings=[]
    for i in range(total):
        start=time.perf_counter();w.write(frames[i%len(frames)],i*.08);timings.append((time.perf_counter()-start)*1000)
    w.close(total*.08);elapsed=time.perf_counter()-tick
    meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,pix_fmt,duration,nb_frames,color_space,color_range','-of','json',str(path)],text=True))['streams'][0]
    results.append({'rgb_input':rgb,'frames':total,'elapsed_s':elapsed,'isolated_encoder_frames_per_s':total/elapsed,
                    'write_ms_mean':float(np.mean(timings)),'write_ms_p95':float(np.percentile(timings,95)),'stream':meta})
caps=[cv2.VideoCapture(str(root/name)) for name in ['yuv_input.mp4','rgb_input.mp4']];squared=0.;count=0;maximum=0
for _ in range(total):
    a,fa=caps[0].read();b,fb=caps[1].read()
    if not a or not b:raise ValueError('Encoded probe frame count differs')
    diff=fa.astype(np.float32)-fb.astype(np.float32);squared+=float(np.sum(diff*diff,dtype=np.float64));count+=diff.size;maximum=max(maximum,int(np.max(np.abs(diff))))
for cap in caps:cap.release()
mse=squared/count
report={'kind':'Isolated encoder experiment on20 repeated previously annotated real frames; not neural/application FPS',
        'source':source,'results':results,'between_modes_rgb_rmse':float(np.sqrt(mse)),
        'between_modes_rgb_psnr_db':float(10*np.log10(255**2/mse)) if mse else None,'max_channel_difference':maximum,
        'limits':'Both use QP20 NVENC; RGB/YUV conversion can differ. Repeated frames are a codec microbenchmark, not a new encounter.'}
Path('reports/codec_probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
