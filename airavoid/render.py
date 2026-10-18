"""Russian primary annotation UI. Every explanation maps to estimator reason codes."""
import cv2
import numpy as np
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont

COLORS={'threat':(255,94,103),'nonthreat':(67,214,170),'uncertain':(246,195,83)}
STATES={'threat':'ВОЗМОЖНАЯ УГРОЗА','nonthreat':'ПРИЗНАКИ ПРОХОДА МИМО','uncertain':'НЕДОСТАТОЧНО ДАННЫХ'}
OVERLAY_STATES={'threat':'ВОЗМ. УГРОЗА','nonthreat':'ПРОХОД МИМО','uncertain':'НЕОПРЕДЕЛЕНО'}
REASONS={
 'history':'Накопление временной истории', 'lost':'Нет свежего наблюдения',
 'camera':'Движение камеры не определено', 'association':'Неоднозначное сопоставление целей',
 'tiny':'Слишком мало пикселей для оценки роста', 'aspect':'Размер нестабилен: ракурс / детектор',
 'clipped':'Граница кадра: размер цели обрезан',
 'parallax':'Перенос камеры не отделён от поворота',
 'bearing_growth':'Цель растёт, направление почти постоянно',
 'crossing':'Дрейф велик относительно роста размера', 'receding':'Видимый размер устойчиво уменьшается',
 'crossing_no_growth':'Заметный дрейф без убедительного роста',
 'overlap':'Неопределённость допускает разные исходы',
 'growth_unresolved':'Рост размера пока не подтверждён',
 'shrink_ambiguous':'Уменьшение: удаление или смена ракурса',
 'bounded_infeasible':'Нет траектории в заданных границах ошибок',
 'bounded_ambiguous':'Допустимые траектории дают разные исходы',
 'cpa_close':'Калиброванная модель: малое сближение',
 'cpa_far':'Калиброванная модель: большой промах',
 'cpa_past':'Ближайшая точка модели уже пройдена',
 'confirm_warning':'Признаки угрозы: подтверждаем во времени',
 'scale_history':'Недостаточно независимых измерений размера',
 'scale_stale':'Нет свежего независимого размера цели',
 'confirm_non_threat':'Признаки прохода: подтверждаем во времени'}
CLASSES=['самолёт','вертолёт','птица','аэростат','парашют']
CAMERA_STATUS={'background_similarity':'оценка движения по фону','unreliable':'ненадёжная оценка',
               'unobserved':'нет опорных наблюдений','unknown_camera':'движение не определено',
               'fixed_setup_assumption':'допущение: неподвижна','motion_violates_fixed_setup':'нарушено допущение неподвижности'}


class Renderer:
    width,height=1800,1140
    def __init__(self,title='RGB • анализ сближения',cache_text=True,calibration=None):
        self.title=title
        self.calibration=calibration
        self.cache_text=cache_text;self.text_cache=OrderedDict()
        self.fonts={s:ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',s) for s in [14,16,18,20,22,26,30]}
    def render(self,frame,objects,t,camera,advice):
        canvas=Image.new('RGB',(self.width,self.height),(12,19,30))
        d=ImageDraw.Draw(canvas)
        def text(x,y,s,size=18,color=(220,230,241)):
            s=str(s)
            if not self.cache_text:
                d.text((x,y),s,font=self.fonts[size],fill=color);return
            key=(s,size,color)
            item=self.text_cache.get(key)
            if item is None:
                bb=self.fonts[size].getbbox(s)
                glyph=Image.new('RGBA',(max(1,bb[2]-bb[0]),max(1,bb[3]-bb[1])),(0,0,0,0))
                ImageDraw.Draw(glyph).text((-bb[0],-bb[1]),s,font=self.fonts[size],fill=(*color,255))
                item=(glyph,bb[0],bb[1]);self.text_cache[key]=item
                if len(self.text_cache)>512:self.text_cache.popitem(last=False)
            else:self.text_cache.move_to_end(key)
            glyph,left,top=item
            canvas.paste(glyph,(int(x+left),int(y+top)),glyph)
        text(24,12,self.title,26)
        text(990,20,f'{t:06.2f} с  •  RGB / без дальности и телеметрии',18)
        ratio=min(1280/frame.shape[1],720/frame.shape[0])
        vw,vh=round(frame.shape[1]*ratio),round(frame.shape[0]*ratio)
        offset_x,offset_y=(1280-vw)//2,(720-vh)//2
        image=np.zeros((720,1280,3),np.uint8)
        image[offset_y:offset_y+vh,offset_x:offset_x+vw]=cv2.resize(frame,(vw,vh))
        canvas.paste(Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)),(0,64))
        sx=vw/frame.shape[1];sy=vh/frame.shape[0]
        def in_view(point):return 0<=point[0]<1280 and 64<=point[1]<784
        def view_line(points,color,width=1):
            if len(points)<2:return
            if all(in_view(p) for p in points):d.line(points,fill=color,width=width);return
            for a,b in zip(points,points[1:]):
                visible,start,end=cv2.clipLine((0,64,1280,720),tuple(round(v) for v in a),tuple(round(v) for v in b))
                if visible:d.line([start,end],fill=color,width=width)
        ordered=sorted(objects,key=lambda x:(t-x['last_seen']>.25,{'threat':0,'uncertain':1,'nonthreat':2}[x['assessment']['state']],-x['confidence']))
        occupied=[];labels_to_draw=[]
        def intersection(a,b):
            return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        target_regions=[(o['box'][0]*sx+offset_x,o['box'][1]*sy+64+offset_y,o['box'][2]*sx+offset_x,o['box'][3]*sy+64+offset_y) for o in objects if t-o['last_seen']<=.25]
        for o in ordered:
            # A stale box is an old measurement, not an aligned current observation.
            if t-o['last_seen']>.25:continue
            e=o['assessment'];color=COLORS[e['state']]
            x1,y1,x2,y2=o['box'];x=(x1+x2)/2*sx+offset_x;y=(y1+y2)/2*sy+64+offset_y
            pad=5
            region=(max(0,x1*sx+offset_x-pad),max(64,y1*sy+64+offset_y-pad),min(1279,x2*sx+offset_x+pad),min(783,y2*sy+64+offset_y+pad))
            if region[0]>=region[2] or region[1]>=region[3]:continue
            d.rectangle(region,outline=color,width=2)
            if in_view((x,y)):d.ellipse((max(0,x-3),max(64,y-3),min(1279,x+3),min(783,y+3)),fill=color)
            label=f"#{o['id']}  {OVERLAY_STATES[e['state']]}"
            lw=d.textlength(label,font=self.fonts[16])+8;lh=24
            candidates=[(x-lw/2,y1*sy+64+offset_y-30),(x-lw/2,y2*sy+64+offset_y+8),
                        (x2*sx+offset_x+12,y-12),(x1*sx+offset_x-lw-12,y-12)]
            candidates += [(x-lw/2,y+delta) for delta in [-65,40,-95,70]]
            rectangles=[]
            for xx,yy in candidates:
                xx=max(4,min(xx,1276-lw));yy=max(67,min(yy,755))
                rect=(xx,yy,xx+lw,yy+lh)
                penalty=sum(intersection(rect,r) for r in occupied)*5+sum(intersection(rect,r) for r in target_regions)
                rectangles.append((penalty,rect))
            _,bb=min(rectangles,key=lambda item:item[0]);occupied.append(bb);tx,ty=bb[:2]
            labels_to_draw.append((bb,tx,ty,label,color))
            history=o.get('raw_history',[])
            points=[(h[1]*sx+offset_x,h[2]*sy+64+offset_y) for h in history if t-h[0]<=2.]
            view_line(points,color,width=2)
            # Dotted local image extrapolation, only on a valid observed basis.
            if e['drift'] is not None and camera['mode']=='fixed':
                vx,vy=e['velocity'];ex=x+2*vx*sx;ey=y+2*vy*sy
                for alpha in np.linspace(0,1,12):
                    px=x+(ex-x)*alpha;py=y+(ey-y)*alpha
                    if 0<px<1280 and 64<py<784:d.ellipse((px-1,py-1,px+1,py+1),fill=color)
                if 0<ex<1280 and 64<ey<784:
                    sigma=min(50,max(4,2.5*(e['position_sigma'] or 1)*np.sqrt(sx*sy)))
                    if 0<=ex-sigma and ex+sigma<1280 and 64<=ey-sigma and ey+sigma<784:
                        d.ellipse((ex-sigma,ey-sigma,ex+sigma,ey+sigma),outline=color,width=1)
                    else:
                        angles=np.linspace(0,2*np.pi,65)
                        view_line([(ex+sigma*np.cos(a),ey+sigma*np.sin(a)) for a in angles],color)
        for bb,tx,ty,label,color in labels_to_draw:
            d.rectangle(bb,fill=(12,19,30));text(tx,ty,label,16,color)
        # Fixed panel bounds prevent target overlays and diagnostics from colliding.
        d.rectangle((1280,64,1799,784),fill=(18,29,43))
        text(1302,78,'ПОЧЕМУ ТАКОЕ РЕШЕНИЕ?',22)
        for i,o in enumerate(ordered[:3]):
            y=123+i*203;e=o['assessment'];color=COLORS[e['state']]
            score_label={'sot':'SOT','zoom':'фрагм.','pixel':'пикс.'}.get(o.get('source'),'det')
            text(1302,y,f"#{o['id']}  {CLASSES[o['class_id']]} • {score_label} {o['confidence']:.2f}",20,color)
            text(1302,y+30,STATES[e['state']],18,color)
            why=REASONS[e['reason']]
            # Manually break by measured text width rather than clipping.
            lines=['']
            for word in why.split():
                candidate=(lines[-1]+' '+word).strip()
                if d.textlength(candidate,font=self.fonts[18])>465:lines.append(word)
                else:lines[-1]=candidate
            for j,line in enumerate(lines):text(1302,y+60+j*24,line,18)
            if e['growth'] is not None:
                text(1302,y+112,f"Оценка: рост {100*e['growth']:+.1f}%/с; дрейф {e['drift']:.1f} пикс/с",16)
                if e['miss_spans'] is not None:
                    label='CPA модели' if e.get('method')=='calibrated_span_cpa' else 'Прокси промаха'
                    text(1302,y+138,f"{label}: {e['miss_spans']:.1f} размахов; τ ≈ {e['tau']:.1f} с",16)
                    if e['miss_low'] is not None and e['miss_high'] is not None:
                        text(1302,y+161,f"Пределы модели: [{e['miss_low']:.2g}; {e['miss_high']:.2g}] размахов",14)
                else:text(1302,y+138,f"Размах: {e['span']:.1f} пикс • τ не определено",16)
            else:
                span_label='нет оценки' if e['span'] is None else f"{e['span']:.1f} пикс"
                text(1302,y+112,f"История: {e['samples']} наблюдений • размер: {span_label}",16)
            d.line((1302,y+182,1775,y+182),fill=(49,63,80),width=1)
        if not objects:text(1302,135,'Цели не обнаружены. Это не признак безопасности.',16)
        text(1302,738,'Камера: '+CAMERA_STATUS.get(camera['status'],camera['status']),16)
        text(1302,762,f"Опора: {'условно пригодна' if camera['valid'] else 'не подтверждена'} • точек фона: {camera['inliers']}",14)
        # Crops of actual video pixels; inset clearly indicated, no synthetic targets.
        for i,o in enumerate(ordered[:3]):
            if t-o['last_seen']>.25:
                text(20+i*145,797,f"#{o['id']} • НЕТ RGB",14,COLORS['uncertain'])
                d.rectangle((18+i*145,825,138+i*145,925),outline=(49,63,80))
                text(24+i*145,848,'Нет свежего',14)
                text(24+i*145,872,'наблюдения',14)
                continue
            cx,cy=(np.array(o['box'][:2])+o['box'][2:4])/2
            span=max(o['box'][2]-o['box'][0],o['box'][3]-o['box'][1],12)
            r=int(max(24,span*2));cx=int(cx);cy=int(cy)
            crop=frame[max(0,cy-r):min(frame.shape[0],cy+r),max(0,cx-r):min(frame.shape[1],cx+r)]
            if crop.size:
                canvas.paste(Image.fromarray(cv2.cvtColor(cv2.resize(crop,(120,100)),cv2.COLOR_BGR2RGB)),(18+i*145,825))
                text(20+i*145,797,f"#{o['id']} • УВЕЛИЧЕНО",14,COLORS[o['assessment']['state']])
            else:text(20+i*145,797,f"#{o['id']} • ВНЕ КАДРА",14,COLORS['uncertain'])
        text(500,812,'Увеличенные фрагменты исходного RGB',18)
        text(500,844,'Жёлтый цвет = неопределённость, а не отсутствие угрозы.',16)
        if ordered and ordered[0]['assessment'].get('method')=='bounded_projective':
            e=ordered[0]['assessment']
            text(500,876,f"Допущения ошибок: ±{e['assumed_centroid_error_px']:.1f} пикс; ±{100*e['assumed_span_relative_error']:.0f}% размера. Ракурс и скорость постоянны.",16)
        else:text(500,876,'Прогноз на изображении условен: постоянная скорость и стабильный ракурс.',16)
        text(500,906,'Класс и балл модели могут ошибаться; балл наблюдения не равен вероятности столкновения.',14)
        d.line((0,940,1800,940),fill=(49,63,80),width=2)
        text(22,952,f"ВАРИАНТЫ ПО ОСЯМ КАМЕРЫ • иллюстрация • не оценено целей: {advice['unresolved_targets']}",20)
        names={'left':'ВЛЕВО','right':'ВПРАВО','up':'ВВЕРХ','down':'ВНИЗ'}
        for i,(k,v) in enumerate(advice['scores'].items()):
            ident=advice.get('bottleneck_track_ids',{}).get(k)
            text(22+i*245,988,names[k]+': '+('—' if v is None else f'{v:+.2f}  (#{ident})'),18,(246,195,83) if v is not None and v<0 else (140,185,230))
        if 'assumed_common_flow_px_per_s' in advice:
            flow=advice['assumed_common_flow_px_per_s']
            text(22,1019,f'Допущена одинаковая глубина целей; пробный поток {flow:g} пикс/с. Минимум Δ(дрейф/размах), 1/с. Не мера безопасности.',16)
        else:text(22,1019,'Архивный расчёт: условный поток 1 размах/с для каждой цели. Не мера безопасности.',16)
        text(22,1052,'НАПРАВЛЕНИЕ НЕ РЕКОМЕНДУЕТСЯ: неизвестны дальности, движение и возможности своего аппарата.',18,(246,195,83))
        if self.calibration:
            k=self.calibration
            hfov=np.rad2deg(np.arctan((k.width-k.cx)/k.focal_px)-np.arctan(-k.cx/k.focal_px))
            text(22,1092,f'Линия — история; точки — прогноз 2 с. HFOV ≈ {hfov:.1f}° из калибровки; τ — время CPA модели, не время контакта.',16)
        else:text(22,1092,'Линия — история; точки — экстраполяция на 2 с; круг — ошибка модели. FOV неизвестно. τ — прокси роста, не физическое TTC.',16)
        text(22,1118,'Пределы — условная чувствительность; круг — невязка подгонки (4–50 пикс рисунка). Это не вероятность или гарантия.',14)
        text(1302,952,'Исследовательский офлайн-прототип',18)
        text(1302,984,'Не предназначен для управления полётом',16)
        for i,(state,label) in enumerate([('threat','возможная угроза'),('nonthreat','признаки прохода'),('uncertain','неопределённость')]):
            text(1302,1020+i*26,'● '+label,16,COLORS[state])
        return cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)
