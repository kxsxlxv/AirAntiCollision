"""Event warning persistence sensitivity and offline annotation context."""
from collections import defaultdict,Counter
import json
from pathlib import Path
import numpy as np
from scripts.evaluate import match

output=[]
for campaign,data_root in [('artifacts/full_campaign','data/qut_full'),('artifacts/full_bridge_campaign','data/qut_full')]:
    result={'campaign':campaign,'events':[],'persistence_sensitivity':{}}
    for seq in ['M1','M2','M3','M4','M5','M6','M7','M8']:
        root=Path(campaign)/seq
        if not (root/'decisions.jsonl').exists():continue
        gts=defaultdict(list)
        for g in json.loads(Path(f'{data_root}/{seq}.gt.json').read_text()):gts[g['frame']].append(g)
        values=defaultdict(list)
        rows=[json.loads(x) for x in (root/'decisions.jsonl').read_text().splitlines()]
        dt=float(np.median(np.diff([r['time'] for r in rows])))
        for r in rows:
            labels=gts[r['frame']];objects=[o for o in r['objects'] if r['time']-o['last_seen']<=.25]
            pairs=match(objects,labels,20)
            for j,g in enumerate(labels):
                state=objects[pairs[j]]['assessment']['state'] if j in pairs else 'untracked'
                values[g['id']].append((r['time'],state,g.get('range_offline_only')))
        for ident,vs in values.items():
            run=0.;maxrun=0.;first_any=None;first_sustained=None;first_range=None;warnings=0
            for t,state,range_gt in vs:
                if state=='threat':
                    warnings+=1;run+=dt;maxrun=max(maxrun,run)
                    if first_any is None:first_any=t
                    if run>=.8-1e-6 and first_sustained is None:
                        first_sustained=t-run+dt
                        first_range=next((rr for tt,ss,rr in vs if abs(tt-first_sustained)<dt/2),None)
                else:run=0.
            result['events'].append({'id':ident,'kind':'approach' if ident.startswith('approach') else 'other','labeled_opportunities':len(vs),'warning_frames':warnings,'max_warning_pulse_s':maxrun,'first_any_warning_s':first_any,'first_sustained_warning_s':first_sustained,'offline_GT_range_at_sustained_warning':first_range,'seconds_to_last_available_GT_frame':vs[-1][0]-first_sustained if first_sustained is not None else None})
    for threshold in [0.,.24,.5,.8,1.2,2.]:
        counts=Counter()
        for e in result['events']:
            counts[e['kind']+'_events']+=1
            counts[e['kind']+'_warned']+=int(e['warning_frames']>0 and e['max_warning_pulse_s']>=threshold-1e-6)
        result['persistence_sensitivity'][str(threshold)]=counts
    output.append(result)
Path('reports/event_sensitivity.json').write_text(json.dumps({'meaning':'QUT encounter category proxy; offline GT range is not an online range estimate and last available frame is not impact time','campaigns':output},indent=2))
for r in output:print(r['campaign'],r['persistence_sensitivity'])
