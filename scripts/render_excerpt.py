"""Render a diagnostic excerpt from actual recorded model decisions, without inference."""
import argparse
from collections import defaultdict,deque,Counter
import json
from pathlib import Path
import cv2
from airavoid.media import TimestampWriter
from airavoid.render import Renderer
from airavoid.provenance import file_sha256
from airavoid.lifecycle import RunSession


def main():
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('trace');p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,required=True);p.add_argument('--end',type=float,required=True)
    p.add_argument('--title',default='RGB • фрагмент записанных решений');a=p.parse_args()
    if a.end<=a.start:p.error('end must exceed start')
    out=Path(a.output)
    with RunSession(out) as session:
        cap=cv2.VideoCapture(a.source);session.register(cap.release);fps=cap.get(cv2.CAP_PROP_FPS)
        history=defaultdict(lambda:deque(maxlen=100));renderer=Renderer(a.title);writer=TimestampWriter(out/'annotated.mp4',fps,(renderer.width,renderer.height),encoder='h264_nvenc')
        session.register(writer.abort);count=0;states=Counter();first_time=None;previous_idx=None
        for row in map(json.loads,Path(a.trace).read_text(encoding='utf-8').splitlines()):
            t=row['time']
            for obj in row['objects']:
                if abs(obj['last_seen']-t)<1e-6:
                    b=obj['box'];history[obj['id']].append((t,(b[0]+b[2])/2,(b[1]+b[3])/2))
                obj['raw_history']=list(history[obj['id']])
            if t<a.start:continue
            if t>=a.end:break
            if previous_idx is None or row['frame']!=previous_idx+1:cap.set(cv2.CAP_PROP_POS_FRAMES,row['frame'])
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Source frame absent')
            previous_idx=row['frame'];image=renderer.render(frame,row['objects'],t,row['camera'],row['advisory'])
            if first_time is None:first_time=t
            writer.write(image,t-first_time);count+=1;states.update(o['assessment']['state'] for o in row['objects'])
            if count==1 or abs(t-96.6)<.01:cv2.imwrite(str(out/'preview.png'),image)
        if not count:raise ValueError('Empty selected interval')
        writer.close(a.end-first_time)
        report={'kind':'Recorded-decision video excerpt; no neural inference or throughput measurement',
                'input':a.source,'input_sha256':file_sha256(a.source),'decision_trace':a.trace,'decision_trace_sha256':file_sha256(a.trace),
                'source_time_interval_s':[first_time,a.end],'frames':count,'states_track_frames':states,
                'history':'Reconstructed from observed box centers for drawing only; recorded assessments and advisory unchanged',
                'timestamp_meaning':'Video starts at0, on-video clock retains original source time','title':a.title}
        (out/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
