"""Validate completed fresh NPS range exports against their no-range controls."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from final_demo_validate import validate


def run(args):
    args.detector_variant='B';args.pilot=False;args.source=None
    result=validate(args)
    folder=Path(args.run);base=Path(args.baseline)
    before=[json.loads(x) for x in (base/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    after=[json.loads(x) for x in (folder/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    mismatches=[];ranges=[]
    for old,new in zip(before,after,strict=True):
        stripped=copy.deepcopy(new);demo=stripped['advisory']['demo'];demo.pop('range_preset')
        for target in demo['targets']:
            value=target.get('range_approx_m')
            if value is not None:ranges.append(value)
            for key in list(target):
                if key.startswith('range_'):del target[key]
        if old!=stripped:mismatches.append(new['frame'])
    if mismatches:raise ValueError('Range changed pre-existing inference/cues: '+str(mismatches[:10]))
    if not ranges:raise ValueError('No real measured range estimates in completed video')
    result['range_control']={'baseline':str(base),'all_inference_and_cue_fields_unchanged':True,
        'range_target_samples':len(ranges),'minimum_estimate_m':min(ranges),'maximum_estimate_m':max(ranges),
        'physical_range_accuracy':'unmeasured; user-size and approximate lens model only'}
    Path(args.output).write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result['range_control'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
