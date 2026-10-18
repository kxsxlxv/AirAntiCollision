"""Standalone Russian evidence figure, generated from recorded experiment JSON."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

reference=json.loads(Path('artifacts/full_gpu_default/full_summary.json').read_text())
campaign=json.loads(Path('artifacts/confidence_full/campaign_summary.json').read_text())
selected=next(x for x in campaign['results'] if x['threshold']==.03)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(16,7.8));fig.subplots_adjust(top=.77,bottom=.32,left=.12,right=.98,wspace=.42)
states=['threat','nonthreat','uncertain','untracked'];colors=['#ed596a','#33b58d','#e6b64d','#657485']
labels=['предупреждение','признаки прохода','неопределённость','нет свежего трека']
for ax,kind,title in zip(axes,['approach','other'],['Номинально приближающийся самолёт','Другие размеченные самолёты']):
    rows=[reference['counts'][kind],selected['counts'][kind]]
    for y,row in enumerate(rows):
        left=0.;total=row['total']
        for state,color in zip(states,colors):
            count=row.get(state,0);percent=100*count/total
            ax.barh(y,percent,left=left,color=color,height=.53)
            if percent>3:ax.text(left+percent/2,y,f'{count}\n{percent:.1f}%',ha='center',va='center',fontsize=10,color='white' if state=='untracked' else '#17202a')
            left+=percent
    ax.set_yticks([0,1],['Базовый GPU\nпорог 0.18','Консервативный\nпорог 0.03'])
    ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Доля размеченных возможностей наблюдения, %')
    ax.set_title(title+f"\nn = {rows[0]['total']}",fontsize=13,pad=17)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    if kind=='other':ax.text(.02,-.35,'Предупреждения: базовый 7 кадров / 1.4 с;\nконсервативный 0 по известным «other».',transform=ax.transAxes,fontsize=10)
fig.suptitle('Реальный RGB: 15 полных последовательностей QUT',x=.04,y=.96,ha='left',fontsize=22,weight='bold')
fig.text(.04,.895,'37.02 мин; 11 113 обработанных кадров; неподвижная наземная камера; TensorRT / RTX 4060.',fontsize=12)
fig.text(.04,.855,'Категории встречи — не истинное столкновение или CPA. Все источники уже изучались при разработке.',fontsize=12,color='#754b16')
handles=[plt.Rectangle((0,0),1,1,color=c) for c in colors]
fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.04,.205),ncol=4,frameon=False,fontsize=11)
fig.text(.04,.17,'Консервативный: окно 4 с, HAC 1 с, граница RGB 1 пикс, внутренний край фрагмента 2 пикс; порог YOLO 0.03.',fontsize=11)
fig.text(.04,.125,'15/15 номинальных подходов имеют предупреждение ≥0.8 с; 0/13 «other» — после настройки на изученных данных.',fontsize=11)
fig.text(.04,.08,'M3: дальность противоречит категории approach. GT-точки приблизительны; совпадение в пределах 20 пикс.',fontsize=11,color='#754b16')
fig.text(.04,.035,'Около 23% подходов и 24% «other» остаются без трека. Неопределённость и пропуски не означают безопасность.',fontsize=11)
fig.savefig('reports/results_full_extension.png',dpi=150,facecolor='white')
print('reports/results_full_extension.png')
