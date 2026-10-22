"""Opt-in 16:9 showcase of measured image cues; no flight-guidance semantics."""
from collections import OrderedDict
from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BACKGROUND=(9,16,25)
PANEL=(16,27,39)
WHITE=(234,242,249)
MUTED=(145,164,182)
CYAN=(90,210,220)
AMBER=(255,184,84)
STATE_LABELS={
    'MONITORING':('НАБЛЮДЕНИЕ',CYAN),
    'POSSIBLE CONVERGENCE':('ВОЗМОЖНОЕ СБЛИЖЕНИЕ',AMBER),
    'INSUFFICIENT EVIDENCE':('НЕДОСТАТОЧНО ДАННЫХ',MUTED)}


def normalized_state(value):
    state=str(value or '').upper().replace('_',' ')
    return state if state in STATE_LABELS else 'INSUFFICIENT EVIDENCE'


def video_placement(width,height,area=(24,100,1472,884)):
    """Uniform fit: return actual source bounds and its one common pixel scale."""
    if width<=0 or height<=0:raise ValueError('Positive source dimensions required')
    x,y,w,h=area;scale=min(w/width,h/height)
    vw,vh=round(width*scale),round(height*scale)
    return (x+(w-vw)//2,y+(h-vh)//2,vw,vh),scale


def direction_label(vector):
    v=np.asarray(vector,float)
    if v.shape!=(2,) or not np.isfinite(v).all() or np.linalg.norm(v)<1e-8:return None
    x,y=v/np.linalg.norm(v)
    horizontal='ВПРАВО' if x>.35 else 'ВЛЕВО' if x<-.35 else ''
    vertical='ВНИЗ' if y>.35 else 'ВВЕРХ' if y<-.35 else ''
    return ' · '.join(word for word in [horizontal,vertical] if word)


class DemoRenderer:
    width,height=1920,1080
    size=(1920,1080)
    video_area=(24,100,1472,884)

    def __init__(self,title='ВОЗДУШНАЯ ОБСТАНОВКА',cache_text=True,calibration=None):
        self.title=title;self.calibration=calibration;self.cache_text=cache_text
        self.text_cache=OrderedDict();self.fonts={};self.focus_id=None
        regular='C:/Windows/Fonts/segoeui.ttf';bold='C:/Windows/Fonts/seguisb.ttf'
        if not Path(bold).exists():bold=regular
        for size in [13,14,15,16,17,18,19,20,21,22,24,26,28,30,32,36,42]:
            self.fonts[(size,False)]=ImageFont.truetype(regular,size)
            self.fonts[(size,True)]=ImageFont.truetype(bold,size)

    def _text(self,canvas,x,y,value,size=18,color=WHITE,bold=False):
        value=str(value);font=self.fonts[(size,bold)]
        if not self.cache_text:
            ImageDraw.Draw(canvas).text((x,y),value,font=font,fill=color);return
        key=(value,size,color,bold)
        item=self.text_cache.get(key)
        if item is None:
            bb=font.getbbox(value)
            glyph=Image.new('RGBA',(max(1,bb[2]-bb[0]),max(1,bb[3]-bb[1])),(0,0,0,0))
            ImageDraw.Draw(glyph).text((-bb[0],-bb[1]),value,font=font,fill=(*color,255))
            item=(glyph,bb[0],bb[1]);self.text_cache[key]=item
            if len(self.text_cache)>768:self.text_cache.popitem(last=False)
        else:self.text_cache.move_to_end(key)
        glyph,left,top=item;canvas.paste(glyph,(int(x+left),int(y+top)),glyph)

    def _wrapped(self,canvas,x,y,value,max_width,size=18,color=WHITE,bold=False,max_lines=3,line_height=None):
        font=self.fonts[(size,bold)];draw=ImageDraw.Draw(canvas);lines=['']
        for word in str(value).replace('\n',' \n ').split(' '):
            if word=='\n':lines.append('');continue
            trial=(lines[-1]+' '+word).strip()
            if draw.textlength(trial,font=font)<=max_width:lines[-1]=trial
            else:
                if lines[-1]:lines.append('')
                chunk=''
                for letter in word:
                    if chunk and draw.textlength(chunk+letter,font=font)>max_width:
                        lines[-1]=chunk;lines.append('');chunk=''
                    chunk+=letter
                lines[-1]=chunk
        if len(lines)>max_lines:lines=lines[:max_lines];lines[-1]+='…'
        for index,line in enumerate(lines):
            while line and draw.textlength(line,font=font)>max_width:line=line[:-2]+'…'
            self._text(canvas,x,y+index*(line_height or size+7),line,size,color,bold)
        return len(lines)*(line_height or size+7)

    @staticmethod
    def _number(value,digits=1,signed=False,suffix=''):
        if value is None:return '—'
        try:value=float(value)
        except (TypeError,ValueError):return '—'
        if not math.isfinite(value):return '—'
        spec=f'+.{digits}f' if signed else f'.{digits}f'
        return format(value,spec).replace('.',',')+suffix

    @staticmethod
    def _fresh(target,objects,time):
        obj=objects.get(target.get('id'))
        if obj is not None and 'last_seen' in obj and time-float(obj['last_seen'])>.25:return False
        trail=target.get('measured_trail') or []
        if trail and time-float(trail[-1][0])>.3:return False
        return True

    def render(self,frame,objects,t,camera,advice):
        if not isinstance(frame,np.ndarray) or frame.ndim!=3 or frame.shape[2]!=3:
            raise ValueError('BGR source image required')
        canvas=Image.new('RGB',self.size,BACKGROUND);draw=ImageDraw.Draw(canvas)
        demo=advice.get('demo') or {};lookup={o['id']:o for o in objects}
        targets=[x for x in demo.get('targets',[]) if self._fresh(x,lookup,t)]
        priority_id=demo.get('priority_id')
        priority=next((x for x in targets if x.get('id')==priority_id),None)
        cue_priority=priority is not None
        if priority is None and targets:
            # A display focus can show a genuine measured aircraft even when
            # geometry cannot rank a threat. It cannot establish a direction.
            priority=next((x for x in targets if x.get('id')==self.focus_id),None)
            if priority is None:
                priority=max(targets,key=lambda x:max(x['box'][2]-x['box'][0],x['box'][3]-x['box'][1]))
            priority_id=priority['id']
        self.focus_id=priority_id if priority is not None else None
        state=normalized_state(demo.get('state') or (priority or {}).get('state'))
        state_text,state_color=STATE_LABELS[state]
        self._text(canvas,25,19,self.title,30,WHITE,True)
        self._text(canvas,26,60,'RGB  /  ИЗМЕРЕНИЯ НА ИЗОБРАЖЕНИИ',14,MUTED)
        self._text(canvas,1550,24,'ДЕМОНСТРАЦИЯ',14,CYAN,True)
        minutes=int(max(0,t)//60);seconds=max(0,t)%60
        self._text(canvas,1766,18,f'{minutes:02d}:{seconds:05.2f}',26,WHITE,True)
        draw.line((24,85,1896,85),fill=(40,58,73),width=1)

        (vx,vy,vw,vh),scale=video_placement(frame.shape[1],frame.shape[0],self.video_area)
        draw.rounded_rectangle((24,100,1496,984),radius=14,fill=(3,8,13),outline=(38,55,70),width=1)
        resized=cv2.resize(frame,(vw,vh),interpolation=cv2.INTER_LINEAR)
        canvas.paste(Image.fromarray(cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)),(vx,vy))
        actual=(vx,vy,vw,vh)
        def point(x,y):return (vx+x*scale,vy+y*scale)
        def line(a,b,color,width=2):
            if not np.isfinite([*a,*b]).all():return
            a=tuple(int(np.clip(round(x),-1000000,1000000)) for x in a)
            b=tuple(int(np.clip(round(x),-1000000,1000000)) for x in b)
            visible,p0,p1=cv2.clipLine(actual,a,b)
            if visible:draw.line((p0,p1),fill=color,width=width)
        labels=[];centers=[]
        for target in sorted(targets,key=lambda x:x.get('id')==priority_id):
            box=target.get('box')
            if box is None or len(box)!=4 or not np.isfinite(box).all():continue
            x1,y1,x2,y2=map(float,box)
            if x2<=x1 or y2<=y1:continue
            center=point((x1+x2)/2,(y1+y2)/2);centers.append(center)
            color=STATE_LABELS[normalized_state(target.get('state'))][1]
            chosen=target.get('id')==priority_id
            trail=[]
            for sample in target.get('measured_trail') or []:
                if len(sample)>=3 and 0<=t-float(sample[0])<=2.5 and np.isfinite(sample[:3]).all():
                    trail.append(point(float(sample[1]),float(sample[2])))
            for a,b in zip(trail,trail[1:]):line(a,b,color,3 if chosen else 2)
            velocity=target.get('velocity_px_s')
            if velocity is not None and len(velocity)==2 and np.isfinite(velocity).all():
                end=np.array(center)+np.asarray(velocity,float)*scale*.5
                delta=end-center;length=float(np.linalg.norm(delta))
                if length>3:
                    steps=min(200,max(1,math.ceil(length/13)))
                    for index in range(steps):
                        a=index/steps;b=min(1,(index+.52)/steps)
                        line(np.array(center)+a*delta,np.array(center)+b*delta,color,2)
            a=point(x1,y1);b=point(x2,y2);pad=9 if chosen else 5
            bounds=(max(vx,a[0]-pad),max(vy,a[1]-pad),min(vx+vw-1,b[0]+pad),min(vy+vh-1,b[1]+pad))
            if bounds[0]>=bounds[2] or bounds[1]>=bounds[3]:continue
            corner=min(13,max(6,(bounds[2]-bounds[0])/3));stroke=3 if chosen else 2
            for xx,dx in [(bounds[0],1),(bounds[2],-1)]:
                for yy,dy in [(bounds[1],1),(bounds[3],-1)]:
                    draw.line(((xx,yy+dy*corner),(xx,yy),(xx+dx*corner,yy)),fill=color,width=stroke)
            label=f"ЦЕЛЬ #{target['id']}" if chosen else f"#{target['id']}"
            if target.get('range_display_m') is not None:label+=f" · ≈{target['range_display_m']:.0f} м"
            font=self.fonts[(17,True)];tw=draw.textlength(label,font=font)+22
            xx=max(vx+3,min(bounds[0],vx+vw-tw-3));yy=max(vy+3,bounds[1]-34)
            if any(abs(xx-a0)<tw and abs(yy-b0)<30 for a0,b0 in labels):yy=min(vy+vh-30,bounds[3]+8)
            labels.append((xx,yy));draw.rounded_rectangle((xx,yy,xx+tw,yy+29),radius=6,fill=(10,22,32),outline=color,width=1)
            self._text(canvas,xx+11,yy+2,label,17,color,True)

        # A single illustrative direction in camera/image axes. No arrow exists
        # while cue evidence leaves its direction unresolved.
        direction=demo.get('direction') or {};vector=direction.get('vector')
        resolved=cue_priority and direction.get('status')=='resolved' and vector is not None and direction_label(vector)
        if resolved:
            vector=np.asarray(vector,float).copy();vector/=np.linalg.norm(vector)
            left_center=24+(vx-24)/2 if vx-24>=130 else 116
            right_center=vx+vw+(1496-vx-vw)/2 if 1496-vx-vw>=130 else 1402
            candidates=[(left_center,844),(right_center,844),(left_center,220),(right_center,220)]
            anchor=next((p for p in candidates if all(np.linalg.norm(np.array(p)-c)>160 for c in centers)),candidates[0])
            cx,cy=anchor
            draw.ellipse((cx-62,cy-62,cx+62,cy+62),fill=(14,30,42),outline=(48,87,104),width=2)
            start=np.array(anchor)-vector*38;end=np.array(anchor)+vector*45;normal=np.array([-vector[1],vector[0]])
            draw.line((tuple(start),tuple(end-vector*19)),fill=CYAN,width=15)
            draw.polygon([tuple(end),tuple(end-vector*29+normal*24),tuple(end-vector*29-normal*24)],fill=CYAN)
            self._text(canvas,cx-50,cy+76,'ОСИ КАДРА',14,CYAN,True)

        # Compact evidence sidebar: all numeric values come from the cue payload.
        draw.rounded_rectangle((1520,100,1896,984),radius=15,fill=PANEL,outline=(37,56,73),width=1)
        self._text(canvas,1542,122,'ОБСТАНОВКА В КАДРЕ',14,MUTED,True)
        self._wrapped(canvas,1541,155,state_text,330,28,state_color,True,2,34)
        draw.line((1542,240,1874,240),fill=(40,59,75),width=1)
        self._text(canvas,1542,256,'ФОКУС НАБЛЮДЕНИЯ',13,MUTED,True)
        self._text(canvas,1542,282,f"ЦЕЛЬ #{priority_id}" if priority is not None else 'НЕ ВЫДЕЛЕН',28,WHITE,True)
        self._text(canvas,1752,294,f'{len(targets)} в кадре',15,MUTED)
        priority_object=lookup.get(priority_id,{})
        class_names=['самолёт','вертолёт','птица','аэростат','парашют']
        class_id=priority_object.get('class_id')
        classification=priority_object.get('class_display')
        if not classification and isinstance(class_id,(int,np.integer)) and 0<=class_id<len(class_names):classification=class_names[class_id]
        if priority is not None:self._wrapped(canvas,1542,319,classification or 'Класс не подтверждён',332,13,MUTED,max_lines=1)
        crop_bounds=(1542,338,1874,492)
        draw.rounded_rectangle(crop_bounds,radius=8,fill=(7,16,24))
        if priority is not None and priority.get('box') is not None:
            box=priority['box'];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
            radius=max(30,max(box[2]-box[0],box[3]-box[1])*2.3)
            left=max(0,int(cx-radius));right=min(frame.shape[1],int(cx+radius))
            top=max(0,int(cy-radius*.465));bottom=min(frame.shape[0],int(cy+radius*.465))
            crop=frame[top:bottom,left:right]
            if crop.size:
                factor=min(332/crop.shape[1],154/crop.shape[0]);cw=round(crop.shape[1]*factor);ch=round(crop.shape[0]*factor)
                thumb=cv2.resize(crop,(cw,ch),interpolation=cv2.INTER_LINEAR)
                canvas.paste(Image.fromarray(cv2.cvtColor(thumb,cv2.COLOR_BGR2RGB)),(1542+(332-cw)//2,338+(154-ch)//2))
        else:self._text(canvas,1570,396,'Нет свежего наблюдения',18,MUTED)
        self._text(canvas,1542,501,'УВЕЛИЧЕННЫЙ ФРАГМЕНТ ИСХОДНОГО RGB',13,MUTED)
        values=priority or {}
        growth=values.get('growth_per_s')
        if growth is not None:
            try:growth=float(growth)*100
            except (TypeError,ValueError):growth=None
        rows=[('Видимый размер',self._number(growth,1,True,' %/с')),
              ('Дрейф / размер',self._number(values.get('drift_normalized_per_s'),2,False,' /с')),
              ('Окно измерений',self._number(values.get('window_s'),1,False,' с'))]
        if demo.get('range_preset'):
            range_m=values.get('range_display_m')
            rows[-1]=('Дальность, оценка','—' if range_m is None else f'≈ {range_m:.0f} м')
        for index,(label,value) in enumerate(rows):
            y=546+index*42;self._text(canvas,1542,y,label,16,MUTED)
            font=self.fonts[(20,True)];self._text(canvas,1874-draw.textlength(value,font=font),y-3,value,20,WHITE,True)
        samples=values.get('samples')
        self._text(canvas,1542,678,f'{samples} независимых измерений' if samples is not None else 'Измерений пока недостаточно',15,MUTED)
        reason=values.get('reason_text') or values.get('reason') or ('Накопление независимых наблюдений.' if targets else 'Цели не наблюдаются. Это не признак безопасности.')
        self._wrapped(canvas,1542,708,reason,332,18,WHITE,False,3,24)
        draw.line((1542,798,1874,798),fill=(40,59,75),width=1)
        self._text(canvas,1542,818,'РАЗДЕЛЕНИЕ В КАДРЕ',13,MUTED,True)
        self._wrapped(canvas,1542,847,direction_label(vector) if resolved else 'НАПРАВЛЕНИЕ НЕ ОПРЕДЕЛЕНО',332,21,CYAN if resolved else MUTED,True,2,27)
        if direction.get('reason_text'):self._wrapped(canvas,1542,927,direction['reason_text'],332,14,MUTED,False,2,19)

        draw.line((26,1004,73,1004),fill=CYAN,width=3)
        self._text(canvas,86,992,'измеренная траектория',15,MUTED)
        for x in [379,394,409]:draw.line((x,1004,x+7,1004),fill=WHITE,width=2)
        self._text(canvas,432,992,'локальная проекция на 0,5 с',15,MUTED)
        preset=demo.get('range_preset')
        range_caption=(f"NPS · размах {preset['aircraft_span_m']:g} м · FOV ≈{preset['horizontal_fov_deg_assumed']:g}°" if preset else 'RGB · ДАЛЬНОСТЬ НЕИЗВЕСТНА')
        self._text(canvas,1380,992,range_caption,13,MUTED)
        draw.rectangle((0,1022,1920,1080),fill=(13,29,39))
        self._text(canvas,26,1035,'Иллюстрация разделения в кадре — не руководство полётом',21,WHITE,True)
        self._text(canvas,1504,1039,'НАБЛЮДЕНИЕ  /  RGB',14,CYAN,True)
        return cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)

    def draw(self,frame,objects,t,camera,advice):
        return self.render(frame,objects,t,camera,advice)


Renderer=DemoRenderer
