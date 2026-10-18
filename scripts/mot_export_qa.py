"""Decode every measured MOT export and check recorded rows, PTS, codec and size."""
from pathlib import Path
import json,sys
import numpy as np
import av
ROOT=Path(__file__).resolve().parents[1]
directories=sorted([* (ROOT/'artifacts/mot/render_live').glob('*'),* (ROOT/'artifacts/mot/async_live').glob('*')])
results=[];equivalence=[]
for directory in directories:
    if not directory.is_dir() or not (directory/'report.json').exists():continue
    report=json.loads((directory/'report.json').read_text())
    for run in report['runs']:
        seq=run['sequence'];rows=[json.loads(x) for x in (directory/f'{seq}.jsonl').read_text().splitlines()]
        expected=np.array([r['time']-rows[0]['time'] for r in rows])
        with av.open(str(directory/f'{seq}.mp4')) as c:
            stream=c.streams.video[0];codec=stream.codec_context.name;size=[stream.width,stream.height]
            times=[];formats=set()
            for frame in c.decode(stream):times.append(float(frame.pts*frame.time_base));formats.add(frame.format.name)
            assert codec=='h264' and size==run['source_size']
            assert len(times)==len(rows) and np.all(np.diff(times)>0)
            error=float(np.max(np.abs(expected-times)));assert error<1/90000+1e-7
            duration=c.duration/av.time_base
            nominal_step=float(np.median(np.diff(expected)))
            assert abs(duration-(expected[-1]+nominal_step))<.001
            results.append(dict(directory=str(directory.relative_to(ROOT)),sequence=seq,frames=len(times),codec=codec,
                                size=size,formats=sorted(formats),duration=duration,max_pts_error=error))
        if directory.parent.name=='async_live':
            method=report['args']['method'];original=ROOT/f'artifacts/mot/render_live/{method}/{seq}.jsonl'
            reference=[json.loads(x) for x in original.read_text().splitlines()]
            assert len(rows)==len(reference)
            same_ids=all([int(t[0]) for t in a['tracks']]==[int(t[0]) for t in b['tracks']] for a,b in zip(rows,reference))
            exact=rows==reference
            errors=[np.max(np.abs(np.asarray(a['tracks'])-np.asarray(b['tracks']))) for a,b in zip(rows,reference) if a['tracks'] and len(a['tracks'])==len(b['tracks'])]
            equivalence.append(dict(directory=str(directory.relative_to(ROOT)),sequence=seq,serial_vs_async_trace_exact=exact,
                                    same_identity_outputs=same_ids,max_numeric_difference=float(max(errors,default=0))))
r=dict(complete=True,videos=len(results),decoded_frames=sum(x['frames'] for x in results),runs=results,serial_async_equivalence=equivalence)
(ROOT/'reports/mot_export_qa.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(videos=r['videos'],frames=r['decoded_frames'],all_serial_async_exact=all(x['serial_vs_async_trace_exact'] for x in equivalence))))
