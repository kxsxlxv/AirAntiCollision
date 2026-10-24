"""Actual live aviation export QA, extending the unchanged earlier video validator."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from final_demo_validate import validate
from airavoid.aviation_regions import compute_attention_regions


def run(args):
    result=validate(args)
    folder=Path(args.run).resolve();benchmark=json.loads((folder/'benchmark.json').read_text(encoding='utf-8'))
    if benchmark['args'].get('demo_layout')!='aviation':raise ValueError('This run did not select the aviation layout')
    if result['encoded_video']['resolution']!=[3840,1080]:raise ValueError('The actual encoded aviation canvas must be3840x1080')
    width,height=result['source']['resolution'];frames=groups=members=0
    with (folder/'decisions.jsonl').open(encoding='utf-8') as src:
        for text in src:
            row=json.loads(text);demo=row.get('advisory',{}).get('demo',{})
            observed=demo.get('attention_regions')
            if not isinstance(observed,list):raise ValueError('Missing actual recorded attention-region payload')
            expected=compute_attention_regions(demo.get('targets',[]),width,height)
            if observed!=expected:raise ValueError('Recorded regions differ from the deterministic presentation helper')
            object_ids={obj['id'] for obj in row['objects']}
            for group in observed:
                if not set(group['member_ids']).issubset(object_ids):raise ValueError('Region refers to an absent object')
                if group.get('complement')!='unknown_or_unmeasured_not_safe':raise ValueError('Unobserved complement was not preserved')
                groups+=1;members+=len(group['member_ids'])
            frames+=1
    result['aviation_ui_contract']={'layout':'aviation','actual_canvas':[3840,1080],'recorded_frames_checked':frames,
        'group_samples':groups,'member_samples':members,'all_region_payloads_match_presentation_helper':True,
        'all_region_members_refer_to_recorded_objects':True,'unobserved_complement_is_not_safe':True,
        'physical_collision_or_free_space_accuracy':'not established by presentation regions'}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--source')
    p.add_argument('--detector-variant',choices=['A','B'],default='B');p.add_argument('--pilot',action='store_true')
    p.add_argument('--output',required=True);a=p.parse_args();output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    try:result=run(a)
    except Exception as exc:
        output.write_text(json.dumps({'state':'failed','error':str(exc),'run':str(Path(a.run).resolve())},indent=2));raise
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({'state':result['state'],'canvas':result['encoded_video']['resolution'],
                      'frames':result['encoded_video']['frames'],'aviation_ui_contract':result['aviation_ui_contract']},indent=2))
