"""Static publication-style proxy evidence overview (existing sibling plotting env)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('artifacts/full_campaign')
results=[json.loads((root/seq/'evaluation.json').read_text()) for seq in ['M1','M2','M3','M4','M5','M6','M7','M8']]
fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
colors={'threat':'#db4f59','nonthreat':'#239b83','uncertain':'#e5b44a','untracked':'#a5adba'}
state_names={'threat':'Возможная угроза','nonthreat':'Проход по модели','uncertain':'Неопределённость','untracked':'Нет трека'}
for ax,kind,title in [(axes[0,0],'approach','Цель на сближении: покадровые оценки'),(axes[0,1],'other','Другие самолёты: покадровые оценки')]:
    bottom=np.zeros(8)
    for state in ['threat','nonthreat','uncertain','untracked']:
        y=np.array([r['counts'].get(kind,{}).get(state,0) for r in results])
        ax.bar(np.arange(8),y,bottom=bottom,color=colors[state],label=state_names[state])
        bottom+=y
    ax.set(xticks=np.arange(8),xticklabels=['M1','M2','M3','M4','M5','M6','M7','M8'],title=title,ylabel='Размеченные наблюдения «цель × кадр»')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,fontsize=9)
events=[e for r in results for e in r['events']]
pos=[e for e in events if e['kind']=='approach'];neg=[e for e in events if e['kind']=='other']
selected=[sum(e['sustained_warning'] for e in pos),sum(e['sustained_warning'] for e in neg)]
baseline=[sum(e['growth_only_baseline_sustained_warning'] for e in pos),sum(e['growth_only_baseline_sustained_warning'] for e in neg)]
ax=axes[1,0];x=np.arange(2);ax.bar(x-.18,selected,.36,label='Дрейф + рост + неопределённость',color='#426da9');ax.bar(x+.18,baseline,.36,label='Только рост, те же фильтры и подтверждение',color='#c27449')
ax.set(xticks=x,xticklabels=[f'Цели на сближении (n={len(pos)})',f'Другие самолёты (n={len(neg)})'],ylabel='События с предупреждением ≥0.8 с',title='Сравнение по типу встречного сценария')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),frameon=False,fontsize=8);ax.spines[['top','right']].set_visible(False)
for i,(a,b) in enumerate(zip(selected,baseline)):
    ax.text(i-.18,a+.15,str(a),ha='center');ax.text(i+.18,b+.15,str(b),ha='center')
ax=axes[1,1];ax.axis('off')
total_frames=sum(json.loads((root/seq/'benchmark.json').read_text())['frames'] for seq in ['M1','M2','M3','M4','M5','M6','M7','M8'])
text=(f'Реальное RGB: 8 полных локальных записей QUT\n{total_frames:,} обработанных кадров, шаг 5 (5 Гц)\nНеподвижная наземная камера; исходник 25 кадр/с\n\n'
      'Метки: цель на сближении / другие самолёты\nФизические столкновения и истинное CPA неизвестны\nДопуск сопоставления центров: 20 пикселей\nОтказы от решения и пропуски включены в подсчёт\n\n'
      '2 кадра предупреждения по другому самолёту\n0 кадров ложного «прохода» по цели на сближении\nМного неопределённости и пропущенных наблюдений\n\n'
      'Промах через плоскость камеры условен:\nширокоугольные аналитические случаи выявляют\nложные «проходы». Не для управления полётом.')
ax.text(.02,.98,text,va='top',fontsize=10.5,linespacing=1.45)
fig.suptitle('Оценка встреч по RGB: проверка по меткам сценариев, а не по фактам столкновения',fontsize=15)
fig.savefig('reports/results_overview.png',dpi=160)
fig.savefig('reports/results_overview.pdf')
print('Saved reports/results_overview.png and .pdf')
