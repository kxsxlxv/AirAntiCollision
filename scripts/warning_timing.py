"""Warning episodes with explicit sampled support, annotation gaps and offline context."""
from collections import defaultdict,Counter
import json
from pathlib import Path
import numpy as np
from scripts.evaluate import match


def episodes(values,dt):
    result=[];current=None
    for sample in values:
        if sample['state']!='threat':current=None;continue
        if current is None or sample['time']-current['last_time_s']>dt*1.5:
            current={'onset_s':sample['time'],'last_time_s':sample['time'],'sample_count':0,
                     'offline_range_at_onset':sample['range'],'samples':[]};result.append(current)
        current['last_time_s']=sample['time'];current['sample_count']+=1;current['samples'].append(sample)
    for episode in result:
        episode['sample_hold_duration_s']=episode['last_time_s']-episode['onset_s']+dt
        qualified=next((s for s in episode['samples'] if s['time']-episode['onset_s']+dt>=.8-1e-6),None)
        episode['qualification_support_end_s']=qualified['time']+dt if qualified else None
        episode.pop('samples')
    return result


def analyze(root):
    events=[]
    for trace in sorted(Path(root).glob('*/decisions.jsonl')):
        seq=trace.parent.name;gt=defaultdict(list)
        for g in json.loads(Path(f'data/qut_full/{seq}.gt.json').read_text()):gt[g['frame']].append(g)
        rows=[json.loads(s) for s in trace.read_text(encoding='utf-8').splitlines()]
        dt=float(np.median(np.diff([r['time'] for r in rows])));values=defaultdict(list)
        for row in rows:
            labels=gt[row['frame']];objects=[o for o in row['objects'] if row['time']-o['last_seen']<=.25]
            pairs=match(objects,labels,20)
            for j,label in enumerate(labels):
                values[label['id']].append({'time':row['time'],'state':objects[pairs[j]]['assessment']['state'] if j in pairs else 'untracked',
                                           'range':label.get('range_offline_only')})
        for ident,samples in values.items():
            runs=episodes(samples,dt);qualified=next((e for e in runs if e['qualification_support_end_s'] is not None),None)
            counts=Counter(s['state'] for s in samples)
            events.append({'sequence':seq,'id':ident,'category':'approach' if ident.startswith('approach') else 'other',
                           'sample_period_s':dt,'labeled_samples':len(samples),'states':counts,'first_labeled_s':samples[0]['time'],
                           'last_labeled_s':samples[-1]['time'],'warning_episodes':runs,'first_qualified_episode':qualified,
                           'onset_to_last_label_s':samples[-1]['time']-qualified['onset_s'] if qualified else None,
                           'qualification_to_last_label_s':samples[-1]['time']-qualified['qualification_support_end_s'] if qualified else None})
    sensitivity={}
    for threshold in [.24,.5,.8,1.2,2.,4.]:
        counts=Counter()
        for event in events:
            counts[event['category']+'_events']+=1
            counts[event['category']+'_warned']+=int(any(e['sample_hold_duration_s']>=threshold-1e-6 for e in event['warning_episodes']))
        sensitivity[str(threshold)]=counts
    return {'root':str(root),'events':events,'persistence_sensitivity_s':sensitivity}


if __name__=='__main__':
    report={'meaning':'Offline category proxy. Episode onset is known retrospectively after support accumulates; last annotation is not impact or CPA time.',
            'sample_support':'Each processed decision is held until its next nominal sample; annotation gaps greater than1.5period reset episodes.',
            'range':'Source-provided offline annotation field, never read by camera-only inference; M3 range semantics conflict with its category.',
            'campaigns':[analyze('artifacts/confidence_full/assessment/0.03'),analyze('artifacts/cadence_fullrate')]}
    Path('reports/warning_timing.json').write_text(json.dumps(report,indent=2))
    for campaign in report['campaigns']:
        print(campaign['root'],campaign['persistence_sensitivity_s'])
        for event in campaign['events']:
            if event['category']=='approach':
                e=event['first_qualified_episode']
                print(event['sequence'],'onset',e['onset_s'] if e else None,'to_label_end',event['onset_to_last_label_s'],
                      'range',e['offline_range_at_onset'] if e else None,'warning_fraction',event['states'].get('threat',0)/event['labeled_samples'])
