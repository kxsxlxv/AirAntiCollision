"""Causal demo-cue preflight on frozen real detector outputs; never load GT.

This only identifies an optional recorded fixed-camera illustration. Neither
encounter categories nor event timestamps enter inference. Final exports must
run fresh neural inference and distinguish this from moving-camera NPS examples.
"""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from airavoid.demo_cue import DemoEncounterCue,POSSIBLE
from airavoid.provenance import file_sha256


def run():
    report={'scope':'Cached genuine full-rate Detector A online outputs, real stationary-camera QUT RGB; no GT or event labels read; not neural FPS or physical collision validation',
            'cue_source_sha256':file_sha256(ROOT/'airavoid/demo_cue.py'),'clips':{}}
    for sequence in ['M1','M2','M6','M7','S1','S2']:
        folder=ROOT/'artifacts/cadence_fullrate'/sequence
        path=folder/'decisions.jsonl';benchmark=json.loads((folder/'benchmark.json').read_text())
        cue=DemoEncounterCue();states=Counter();warnings=0;arrows=0;coupled=0;episodes=[];active=None;last_t=0.;frames=0
        with path.open(encoding='utf-8') as stream:
            for text in stream:
                row=json.loads(text);t=row['time'];last_t=t;frames+=1
                result=cue.update(row['objects'],t,row['camera'],(1280,720))
                states.update(x['state'] for x in result['targets'])
                selected=[x for x in result['targets'] if x['state']==POSSIBLE]
                has_warning=bool(selected)
                arrow=result['direction']['status']=='resolved'
                has_both=has_warning and arrow and result['direction'].get('warning_evidence') is True
                warnings+=has_warning;arrows+=arrow;coupled+=has_both
                if has_warning:
                    if active is None:active={'start_s':t,'start_frame':row['frame'],'end_s':t,'end_frame':row['frame'],
                                             'warning_frames':0,'warning_arrow_frames':0,'ids':set(),'max_measured_span_px':0.,'peak_growth_per_s':0.}
                    active['end_s']=t;active['end_frame']=row['frame'];active['warning_frames']+=1
                    active['warning_arrow_frames']+=has_both
                    for target in selected:
                        active['ids'].add(target['id']);b=target['box']
                        active['max_measured_span_px']=max(active['max_measured_span_px'],b[2]-b[0],b[3]-b[1])
                        active['peak_growth_per_s']=max(active['peak_growth_per_s'],target['growth_per_s'])
                elif active is not None:
                    active['ids']=sorted(active['ids']);episodes.append(active);active=None
        if active is not None:active['ids']=sorted(active['ids']);episodes.append(active)
        episodes.sort(key=lambda x:(-x['warning_arrow_frames'],-x['warning_frames']))
        suggestions=[]
        for episode in episodes[:5]:
            end=min(last_t+1/25,episode['end_s']+5.)
            start=max(0.,end-30.)
            suggestions.append({'start_s':start,'end_s':end,'duration_s':end-start,
                                'cue_interval_s':[episode['start_s'],episode['end_s']],
                                'cue_frames':[episode['start_frame'],episode['end_frame']],
                                'reason':'Optional presentation window around naturally produced cue, never an inference input'})
        report['clips'][sequence]={'source_video':benchmark['args']['video'],'decisions_sha256':file_sha256(path),
                                  'frames':frames,'duration_s':last_t+1/25,'target_sample_states':states,
                                  'warning_frames':warnings,'resolved_direction_frames':arrows,
                                  'confirmed_warning_with_resolved_direction_frames':coupled,
                                  'warning_episodes':episodes,'suggested_windows':suggestions}
        print(sequence,'warning frames',warnings,'warning + arrow',coupled,'episodes',len(episodes),flush=True)
    output=ROOT/'reports/final_demo_qut_preflight.json'
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(output)


if __name__=='__main__':run()
