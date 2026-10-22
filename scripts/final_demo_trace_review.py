"""Review actual screenshot/video evidence and direction continuity, without GT."""
import argparse
from collections import Counter
import json
from pathlib import Path
import math
import av
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]


def review(folder,report_path):
    folder=Path(folder).resolve()
    rows=[json.loads(x) for x in (folder/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    benchmark=json.loads((folder/'benchmark.json').read_text())
    width,height=benchmark['resolution'];states=Counter();episodes=[];active=None;bad=[];priorities=[]
    for row in rows:
        cue=row['advisory']['demo'];direction=cue['direction'];states[cue['state']]+=1
        priorities.append(cue['priority_id'])
        if direction['status']=='resolved':
            priority=next(x for x in cue['targets'] if x['id']==cue['priority_id'])
            b=priority['box'];velocity=priority['velocity_px_s'];v=direction['vector']
            px=((b[0]+b[2])/2+.5*velocity[0]-width/2)/(width/2)
            py=((b[1]+b[3])/2+.5*velocity[1]-height/2)/(height/2)
            if px*v[0]+py*v[1]>=0:bad.append(row['frame'])
            key=(priority['id'],tuple(round(x,5) for x in v))
            if active is None or active['key']!=key:
                if active:episodes.append(active)
                active={'key':key,'start_s':row['time'],'end_s':row['time'],'frames':1}
            else:active.update(end_s=row['time'],frames=active['frames']+1)
        elif active:
            episodes.append(active);active=None
    if active:episodes.append(active)
    indices=sorted(set(round((len(rows)-1)*fraction) for fraction in [0,.08,.2,.35,.5,.65,.8,.99]))
    contact=Image.new('RGB',(1440,4*425),(9,16,25));draw=ImageDraw.Draw(contact)
    output=ROOT/'artifacts/final_demo_qa';output.mkdir(parents=True,exist_ok=True)
    with av.open(str(folder/'annotated.mp4')) as video:
        for index,frame in enumerate(video.decode(video.streams.video[0])):
            if index not in indices:continue
            slot=indices.index(index);x=(slot%2)*720;y=(slot//2)*425
            contact.paste(frame.to_image().resize((720,405)),(x,y+20))
            cue=rows[index]['advisory']['demo']
            draw.text((x+5,y+3),f"Frame {index} / {rows[index]['time']:.2f}s / {cue['state']} / arrow {cue['direction']['status']}",fill=(234,242,249))
    contact_path=output/(folder.name+'_contact.jpg');contact.save(contact_path,quality=92)
    result={'source_run':str(folder),'frames':len(rows),'frame_states':states,'resolved_direction_episodes':episodes,
        'resolved_arrow_frames':sum(x['frames'] for x in episodes),
        'priority_changes':sum(a!=b for a,b in zip(priorities,priorities[1:])),
        'direction_geometry_violations':bad,'contact_sheet':str(contact_path),
        'criterion':'For each resolved arrow, moving a normalized image reference along the arrow increases squared distance from the linearly projected target to first order: -2*p dot v > 0.',
        'scope':'Image-coordinate illustration only; not physical aircraft displacement or flight guidance'}
    Path(report_path).write_text(json.dumps(result,indent=2)+'\n')
    if bad:raise ValueError('An arrow does not increase the declared image separation')
    print(json.dumps({k:v for k,v in result.items() if k!='resolved_direction_episodes'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',default=str(ROOT/'demo/final_showcase'))
    parser.add_argument('--output',default=str(ROOT/'reports/final_demo_trace_review.json'))
    args=parser.parse_args();review(args.run,args.output)
