"""Select a real example using fixed generic demo rules, without GT or new FPS claims."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from airavoid.demo_cue import DemoEncounterCue


def main():
    report={'scope':'Cached actual B/original-tracker observations for selected-demo preflight only; final video requires fresh live neural inference', 'clips':{}}
    for clip,width,height in [('032',1280,960),('043',1280,960),('008',1920,1080)]:
        cue=DemoEncounterCue();hits=Counter();states=Counter();arrow=0;first=None
        path=ROOT/f'artifacts/stage2/tracks/B/custom/s1/Clip_{clip}.jsonl'
        for text in path.read_text().splitlines():
            row=json.loads(text);t=row['time'];objects=[]
            for m in row['measurements']:
                tid=m['track_id'];hits[tid]+=1
                if hits[tid]<3:continue
                b=m['measured_bbox']
                objects.append({'id':tid,'box':b,'confidence':m['confidence'],'source':m['source'],'last_seen':t,
                    'assessment':{},'demo_observation':{'measured':True,'ambiguous':m['ambiguous'],
                    'clipped':b[0]<=1 or b[1]<=1 or b[2]>=width-1 or b[3]>=height-1,'hits':hits[tid]}})
            result=cue.update(objects,t,row['camera'],(width,height))
            states.update(x['state'] for x in result['targets'])
            if result['direction']['status']=='resolved':
                arrow+=1
                if first is None:first=t
        report['clips'][clip]={'target_sample_states':states,'resolved_direction_frames':arrow,'first_resolved_direction_s':first}
    output=ROOT/'reports/final_demo_preflight.json';output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
