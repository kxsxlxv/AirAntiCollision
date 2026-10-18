"""Timed offline RGB→DINO sky→YOLO recall→risk→annotated-video workflow."""
import argparse
import os
# cv2's Windows loader prepends its bundled DLL directory to PATH. Preserve the
# incoming environment before that import so the isolated SkySegDINO Torch runtime
# resolves its own cuDNN libraries rather than the custom OpenCV build's versions.
SKY_ENV=os.environ.copy()
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import cv2
from airavoid.lifecycle import RunSession
from airavoid.provenance import file_sha256


def require_cfr(video):
    """Reject unsupported VFR before launching the expensive segmentation pass."""
    result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_frames',
        '-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)],capture_output=True,text=True,check=True)
    times=[float(frame['best_effort_timestamp_time']) for frame in json.loads(result.stdout)['frames']]
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    if len(times)<2 or not math.isfinite(fps) or fps<=0:raise ValueError('Sky workflow needs a nonempty timestamped CFR source')
    if any(abs(t-times[0]-index/fps)>1e-4 for index,t in enumerate(times)):
        raise ValueError('Sky workflow requires CFR: DINO mask uses frame-index alignment; VFR is unsupported')
    return fps,len(times)


def main():
    p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--output',default='demo/sky_run')
    p.add_argument('--camera',choices=['auto','fixed','unknown'],default='auto');p.add_argument('--stride',type=int,default=2)
    p.add_argument('--sky-project',default='../SkySegDINO');p.add_argument('--encoder',choices=['libx264','h264_nvenc'],default='libx264')
    p.add_argument('--render-process',action='store_true');a=p.parse_args()
    project=Path(__file__).resolve().parents[1];video=Path(a.video).resolve();out=Path(a.output).resolve()
    if out==video.parent or video in [out/'sky_overlay.mp4',out/'assessment/annotated.mp4']:
        p.error('Choose a dedicated output directory that cannot overwrite source RGB')
    started=time.perf_counter();timestamp=datetime.now(timezone.utc).isoformat();sky=Path(a.sky_project).resolve()
    with RunSession(out,{'args':vars(a)},benchmark_name='workflow_benchmark.json'):
        fps,frames=require_cfr(video);input_hash=file_sha256(video)
        checkpoint=sky/'segdino_v2/checkpoints/sky_cityscapes_aeroscapes_ade20k_mapillary_scratch/best_model.pth'
        checkpoint_hash=file_sha256(checkpoint)
        cmd=[str(sky/'.venv/Scripts/python.exe'),str(sky/'segdino_v2/infer_video.py'),str(video),
             '--output',str(out/'sky_overlay.mp4'),'--batch-size','4','--keep-mask','--output-width','1280']
        t=time.perf_counter()
        with (out/'sky.log').open('w',encoding='utf-8') as log:subprocess.run(cmd,cwd=sky,env=SKY_ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
        sky_elapsed=time.perf_counter()-t;mask=out/'sky_overlay_mask.mkv'
        manifest={'input_sha256':input_hash,'mask_sha256':file_sha256(mask),'frames':frames,'fps':fps,
                  'checkpoint_sha256':checkpoint_hash,'segmentation_command':cmd,'alignment':'CFR frame index, source PTS checked before segmentation'}
        Path(str(mask)+'.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        target=out/'assessment';cmd=[sys.executable,'-m','airavoid',str(video),'--output',str(target),'--camera',a.camera,
             '--stride',str(a.stride),'--sky-mask',str(mask),'--encoder',a.encoder,'--title','RGB • дополнительные наблюдения малых целей']
        if a.render_process:cmd.append('--render-process')
        t=time.perf_counter()
        with (out/'assessment.log').open('w',encoding='utf-8') as log:subprocess.run(cmd,cwd=project,stdout=log,stderr=subprocess.STDOUT,check=True)
        main_elapsed=time.perf_counter()-t;benchmark=json.loads((target/'benchmark.json').read_text())
        elapsed=time.perf_counter()-started
        report={'timestamp_utc':timestamp,'kind':'complete offline two-pass workflow, no GT','video':str(video),'input_sha256':input_hash,
                'output':str(target/'annotated.mp4'),'frames':benchmark['frames'],'total_wall_s':elapsed,
                'processed_frames_per_total_wall_s':benchmark['frames']/elapsed,'segmentation_load_infer_and_export_s':sky_elapsed,
                'assessment_load_infer_render_and_export_s':main_elapsed,'assessment_processing_fps':benchmark['processing_fps'],
                'sky_checkpoint_sha256':checkpoint_hash,'caveat':'includes CFR preflight, hashing, both process startups and model/render/export work; batch4 offline, no streaming-latency claim'}
        (out/'workflow_benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
