"""Offline centroid-match sensitivity; never used by online camera inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
from scripts.evaluate import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/full_campaign')
    p.add_argument('--data-root',default='data/qut_full');p.add_argument('--sequences',nargs='+',default=[f'M{i}' for i in range(1,9)])
    p.add_argument('--tolerances',nargs='+',type=float,default=[5,10,20,30,50]);p.add_argument('--output',default='reports/match_sensitivity.json')
    a=p.parse_args();results=[]
    for tolerance in a.tolerances:
        counts={'approach':Counter(),'other':Counter()};events=[];unmatched=Counter()
        for seq in a.sequences:
            r=evaluate(Path(a.root)/seq/'decisions.jsonl',Path(a.data_root)/(seq+'.gt.json'),tolerance)
            for kind,c in r['counts'].items():counts[kind].update(c)
            unmatched.update(r['unmatched_track_frames'])
            events.extend({'sequence':seq,**e} for e in r['events'])
        row={'tolerance_px':tolerance,'counts':counts,'events':events,'unmatched_track_frames':unmatched,
             'approach_events_warned':sum(e['sustained_warning'] for e in events if e['kind']=='approach'),
             'other_events_warned':sum(e['sustained_warning'] for e in events if e['kind']=='other')}
        results.append(row)
        print(tolerance,counts,'events',row['approach_events_warned'],row['other_events_warned'],flush=True)
    report={'settings':vars(a),'kind':'Approximate interpolated-centroid match sensitivity, not CPA accuracy',
            'source_policy':'Existing archived traces retain their original settings, including legacy recession behavior where recorded',
            'results':results,'limits':'GT centroid interpolation and incomplete other-aircraft labels can shift counts; larger tolerance can match wrong objects. No tolerance was selected to optimize the reported score.'}
    Path(a.output).write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
