"""Aggregate measured MOT results without mixing published and local scores."""
from pathlib import Path
import json,sys,csv
ROOT=Path(__file__).resolve().parents[1]
methods=['custom','tracktrack','botsort','ocsort','diffmot','putr','trackssm'];rows=[]
for method in methods:
    q=json.loads((ROOT/f'artifacts/mot/campaign/{method}/report.json').read_text())
    s=json.loads((ROOT/f'artifacts/mot/swarm/{method}/mot_metrics.json').read_text())
    qm=[x['metrics'][2] for x in q['runs']];total=sum(x['total_label_samples'] for x in qm)
    row=dict(method=method,qut_label_samples=total,qut_coverage=sum(x['matched'] for x in qm)/total,
        qut_switches_consecutive=sum(x['switches_consecutive'] for x in qm),
        qut_switches_including_reacquisition=sum(x['switches_including_reacquisition'] for x in qm),
        qut_dominant_identity_coverage=sum(sum(t['dominant_id_samples'] for t in x['targets']) for x in qm)/total,
        swarm_HOTA=s['combined']['HOTA']['HOTA']*100,swarm_AssA=s['combined']['HOTA']['AssA']*100,
        swarm_IDF1=s['combined']['Identity']['IDF1']*100,swarm_IDSW=s['combined']['CLEAR']['IDSW'],
        swarm_recall=s['combined']['CLEAR']['CLR_Re'],swarm_precision=s['combined']['CLEAR']['CLR_Pr'])
    for phase in ['live','render_live']:
        path=ROOT/f'artifacts/mot/{phase}/{method}/report.json'
        if path.exists():
            for run in json.loads(path.read_text())['runs']:
                row[f'{phase}_{run["sequence"]}_fps']=run['processing_fps']
    rows.append(row)
result=dict(protocol='Six QUT last-35s clips, approximate centres; three complete predeclared UAVSwarm sequences, official MOT metrics',
    qut_frames_per_method=5250,swarm_frames_per_method=363,swarm_boxes=4616,results=rows,
    limits=['QUT identity proxies are not standard IDF1/HOTA','UAVSwarm subset is small and fixed detector misses many targets',
            'No untouched tuning or detector-training independence claim','TrackSSM is repaired/reference backend',
            'FPS counts processed updates, includes decode+inference+tracking and drawing/export when render_live; startup excluded',
            'QUT updates at25Hz stride1; formation source29.97Hz sampled at14.985Hz stride2; no offline linking'])
repeat_path=ROOT/'reports/mot_async_repeats.json'
if repeat_path.exists():
    repeat=json.loads(repeat_path.read_text())
    for row in rows:
        selected=[x for x in repeat['runs'] if x['method']==row['method'] and x['exit_code']==0]
        if selected:
            for seq in ['M2','formation']:
                values=[next(r['processing_fps'] for r in x['runs'] if r['sequence']==seq) for x in selected]
                row[f'async_{seq}_median_fps']=sorted(values)[len(values)//2]
                row[f'async_{seq}_min_fps']=min(values);row[f'async_{seq}_max_fps']=max(values)
            row['async_runs']=len(selected);row['async_repeat_trace_exact']=all(all(x['trace_exact'].values()) for x in selected)
        else:
            p=ROOT/f'artifacts/mot/async_live/{row["method"]}/report.json'
            if p.exists():
                for run in json.loads(p.read_text())['runs']:row[f'async_{run["sequence"]}_fps']=run['processing_fps']
(ROOT/'reports/mot_comparison.json').write_text(json.dumps(result,indent=2))
fields=list(dict.fromkeys(k for r in rows for k in r))
with (ROOT/'reports/mot_comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
print(json.dumps(rows,indent=2))

sys.path.insert(0,str(ROOT/'external/mot/python_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
names=[r['method'] for r in rows]
for ax,key,title in [(axes[0],'qut_dominant_identity_coverage','QUT: dominant identity coverage'),
                     (axes[1],'swarm_IDF1','UAVSwarm: standard IDF1'),
                     (axes[2],'qut_switches_including_reacquisition','QUT: ID changes incl. reacquisition')]:
    vals=[r[key]*(100 if key=='qut_dominant_identity_coverage' else 1) for r in rows]
    bars=ax.barh(names,vals,color=['#64748b','#2563eb','#059669','#8b5cf6','#d97706','#0891b2','#dc2626'])
    ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    for b,v in zip(bars,vals):ax.text(v,b.get_y()+b.get_height()/2,f' {v:.1f}',va='center',fontsize=9)
    ax.set_xlim(0,max(vals)*1.25)
    if key=='qut_switches_including_reacquisition':
        ax.set_xscale('log');ax.set_xlim(1,max(vals)*1.6);ax.set_title(title+' (log scale)')
fig.suptitle('Fixed YOLO26m, online outputs | TrackSSM repaired/reference | Different datasets give different winners')
fig.savefig(ROOT/'reports/mot_comparison.png',dpi=150);plt.close(fig)
