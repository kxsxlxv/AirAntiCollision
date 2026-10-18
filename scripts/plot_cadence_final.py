"""Final cadence comparison from recorded neural runs and corrected geometry audit."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

old=next(r for r in json.loads(Path('artifacts/confidence_full/campaign_summary.json').read_text())['results'] if r['threshold']==.03)
new=json.loads(Path('reports/clip_history_audit.json').read_text());assert new['complete']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
fig,axes=plt.subplots(1,2,figsize=(16,8));fig.subplots_adjust(left=.14,right=.98,top=.76,bottom=.31,wspace=.45)
states=['threat','nonthreat','uncertain','untracked'];colors=['#ed596a','#33b58d','#e6b64d','#657485']
for ax,kind,title in zip(axes,['approach','other'],['Номинальный подход','Другие размеченные самолёты']):
    for y,report in enumerate([old,new]):
        c=report['counts'][kind];left=0
        for state,color in zip(states,colors):
            n=c.get(state,0);percent=100*n/c['total'];ax.barh(y,percent,left=left,color=color,height=.53)
            if percent>3:ax.text(left+percent/2,y,f'{n}\n{percent:.1f}%',ha='center',va='center',fontsize=11,color='white' if state=='untracked' else '#17202a')
            left+=percent
    ax.set_yticks([0,1],['Шаг 5\nранний прогон','Каждый кадр\nисправленная история']);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Доля размеченных возможностей, %');ax.set_title(title,fontsize=15,pad=20)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Проверена частота по умолчанию: все 15 полных QUT',x=.04,y=.96,ha='left',fontsize=22,weight='bold')
fig.text(.04,.895,'55 535 кадров / 37.02 мин настоящего RGB; исходная частота 25 кадров/с; неподвижная камера.',fontsize=12)
fig.text(.04,.845,'Шаг 5: 4 031 approach / 3 273 other; каждый кадр: 20 155 approach / 16 361 other.',fontsize=12)
fig.legend([Patch(color=c) for c in colors],['предупреждение','признаки прохода','неопределённость','нет свежего трека'],ncol=4,loc='lower left',bbox_to_anchor=(.04,.21),frameon=False)
fig.text(.04,.17,'Каждый кадр: 40 кадров с предупреждением по other — M6: 26 кадров / 1.04 с; M7: 14 кадров / 0.56 с.',fontsize=12)
fig.text(.04,.12,'Все 15 номинальных подходов предупреждены; проходов по approach нет. Пропуски и неопределённость остаются.',fontsize=12)
fig.text(.04,.075,'Категории не задают истинное столкновение/CPA; M3 имеет противоречивую дальность. Независимого теста нет.',fontsize=11,color='#754b16')
fig.text(.04,.035,'Нейросетевые наблюдения фактические. Исправленная геометрия проверена повторным расчётом; изменение: −1 предупреждение S1.',fontsize=11)
fig.savefig('reports/results_cadence_full.png',dpi=150,facecolor='white')
