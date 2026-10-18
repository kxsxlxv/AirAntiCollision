"""Actual approach-state timelines aligned to annotation end, never impact time."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scripts.evaluate import match

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
sequences=[f'M{i}' for i in range(1,9)]+[f'S{i}' for i in range(1,8)]
colors={'threat':'#ed596a','nonthreat':'#33b58d','uncertain':'#e6b64d','untracked':'#657485'}
fig,ax=plt.subplots(figsize=(16,10));fig.subplots_adjust(left=.1,right=.97,bottom=.18,top=.81)
for y,seq in enumerate(sequences):
    root=Path('artifacts/confidence_full/assessment/0.03')/seq;gt=defaultdict(list)
    for g in json.loads(Path(f'data/qut_full/{seq}.gt.json').read_text()):
        if g['kind']=='approach':gt[g['frame']].append(g)
    samples=[]
    for row in map(json.loads,(root/'decisions.jsonl').read_text(encoding='utf-8').splitlines()):
        labels=gt[row['frame']]
        if not labels:continue
        objects=[o for o in row['objects'] if row['time']-o['last_seen']<=.25];pairs=match(objects,labels,20)
        state=objects[pairs[0]]['assessment']['state'] if 0 in pairs else 'untracked'
        samples.append((row['time'],state))
    end=samples[-1][0]
    for state,color in colors.items():
        bars=[(t-end,.2) for t,s in samples if s==state]
        if bars:ax.broken_barh(bars,(y-.32,.64),facecolors=color,edgecolors='none')
    percent=100*sum(s=='threat' for _,s in samples)/len(samples)
    ax.text(1.5,y,f'{percent:.1f}%',va='center',fontsize=11)
ax.set_yticks(range(len(sequences)),[s+(' *' if s=='M3' else '') for s in sequences]);ax.invert_yaxis()
ax.set_xlim(-100,9);ax.set_xticks([-100,-80,-60,-40,-20,0]);ax.axvline(0,color='#26374a',lw=1)
ax.set_xlabel('Секунды до последнего размеченного кадра — это НЕ время до столкновения',labelpad=12)
ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Когда появляется предупреждение: все 15 номинальных подходов',x=.06,y=.96,ha='left',fontsize=21,weight='bold')
fig.text(.06,.905,'Реальный RGB, QUT, неподвижная камера; консервативный профиль; шаг 5 = 0.2 с. Справа — доля предупреждений.',fontsize=12)
fig.text(.06,.865,'Каждая полоса — фактические решения по приблизительной GT-точке. Белая область: нет подходящей разметки.',fontsize=12)
fig.legend([Patch(color=colors[s]) for s in ['threat','uncertain','untracked']],['предупреждение','неопределённость','нет свежего трека'],ncol=3,loc='lower left',bbox_to_anchor=(.06,.055),frameon=False)
fig.text(.06,.025,'* M3: ряд дальности противоречит категории approach. Все источники изучались при разработке; истинного CPA нет.',fontsize=11,color='#754b16')
fig.savefig('reports/warning_timeline.png',dpi=150,facecolor='white')
print('reports/warning_timeline.png')
