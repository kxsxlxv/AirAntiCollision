"""Wide image-coordinate aviation display, opt-in and independent of inference.

The centre panel represents the source raster, not a world/range map. Attention
envelopes and illustrative arrows never identify verified free or safe airspace.
Every ROI is sampled from the current RGB frame only after a fresh observation.
"""
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw
from .demo_render import DemoRenderer,normalized_state,CYAN,AMBER,WHITE,MUTED


BG=(5,12,20)
PANEL=(10,22,33)
EDGE=(39,59,75)
UNKNOWN=(74,91,105)
LABELS={'MONITORING':'КОНТРОЛЬ','POSSIBLE CONVERGENCE':'ВОЗМ. СБЛИЖЕНИЕ',
        'INSUFFICIENT EVIDENCE':'НЕТ ДАННЫХ'}


def fit_rect(width,height,area):
    if width<=0 or height<=0:raise ValueError('Positive image dimensions required')
    x,y,w,h=area;ratio=min(w/width,h/height)
    vw,vh=max(1,round(width*ratio)),max(1,round(height*ratio))
    return (x+(w-vw)//2,y+(h-vh)//2,vw,vh),ratio


def roi_grid_layout(count,area=(2726,124,1094,888),gap=12):
    """Pack every object; maximize the genuine crop area rather than dropping IDs."""
    if count<0:raise ValueError('Nonnegative object count required')
    if not count:return []
    x,y,w,h=area;best=None
    for columns in range(1,count+1):
        rows=math.ceil(count/columns)
        tw=(w-gap*(columns-1))/columns;th=(h-gap*(rows-1))/rows
        if tw<=12 or th<=12:continue
        header=84 if tw>=400 and th>=250 else 72 if tw>=240 and th>=180 else 50
        footer=32 if th>=250 else 28 if th>=180 else 22
        available=max(1,th-header-footer-12)
        scale=min(max(1,tw-16)/1.6,available)
        # Empty cells waste area, while an excessively narrow tile harms labels.
        score=scale*scale*1.6*(count/(rows*columns))**.15
        aspect=tw/th
        if aspect<1.2:score*=(aspect/1.2)**2
        elif aspect>2.6:score*=(2.6/aspect)**2
        if tw<155:score*=tw/155
        candidate=(score,-rows,columns,rows)
        if best is None or candidate>best:best=candidate
    if best is None:raise ValueError('Object count cannot fit the available viewport')
    _,_,columns,rows=best
    rectangles=[]
    for index in range(count):
        col=index%columns;row=index//columns
        left=round(x+col*(w+gap)/columns);right=round(x+(col+1)*(w+gap)/columns-gap)
        top=round(y+row*(h+gap)/rows);bottom=round(y+(row+1)*(h+gap)/rows-gap)
        rectangles.append((left,top,right-left,bottom-top))
    return rectangles


def range_text(target):
    value=target.get('range_display_m')
    if value is None:value=target.get('range_approx_m')
    try:value=float(value)
    except (TypeError,ValueError):return 'D —'
    if not math.isfinite(value) or value<=0:return 'D —'
    text=f'{value:.0f}' if value>=10 or value.is_integer() else f'{value:.1f}'
    return '≈ '+text.replace('.',',')+' м'


def source_class(obj):
    label=str(obj.get('class_display') or obj.get('class') or '')
    lowered=label.lower()
    if 'fixed-wing' in lowered or 'самолётного' in lowered or 'самолетного' in lowered:return 'БПЛА самолётный'
    names=['самолёт','вертолёт','птица','аэростат','парашют']
    mapping={'airplane':'самолёт','helicopter':'вертолёт','bird':'птица','balloon':'аэростат','parachute':'парашют'}
    if lowered in mapping:return mapping[lowered]
    if label:return label
    index=obj.get('class_id')
    return names[int(index)] if isinstance(index,(int,np.integer)) and 0<=index<len(names) else 'класс —'


def fresh_observation(obj,time):
    try:age=float(time)-float(obj['last_seen'])
    except (KeyError,TypeError,ValueError):return False
    if not math.isfinite(age) or not -1e-6<=age<=1e-3:return False
    if obj.get('source','detector') not in ('detector','zoom'):return False
    return obj.get('demo_observation',{}).get('measured',True) is not False


def crop_bounds(box,width,height,aspect=1.6):
    x1,y1,x2,y2=map(float,box);cx=(x1+x2)/2;cy=(y1+y2)/2
    if not math.isfinite(aspect) or aspect<=0:raise ValueError('Positive crop aspect required')
    cw=max(48.,(x2-x1)*4,(y2-y1)*4*aspect);ch=cw/aspect
    left=max(0,int(math.floor(cx-cw/2)));right=min(width,int(math.ceil(cx+cw/2)))
    top=max(0,int(math.floor(cy-ch/2)));bottom=min(height,int(math.ceil(cy+ch/2)))
    return left,top,right,bottom


class AviationRenderer(DemoRenderer):
    width,height=3840,1080
    size=(3840,1080)
    video_area=(20,88,1800,936)
    field_area=(1860,142,800,790)
    roi_area=(2726,124,1094,888)

    def __init__(self,title='ВОЗДУШНЫЕ ОБЪЕКТЫ',cache_text=True,calibration=None):
        if title in ('RGB • анализ сближения','RGB · анализ сближения'):title='ОБЗОР'
        super().__init__(title,cache_text,calibration)
        self.last_layout={}

    @staticmethod
    def _color(state):
        state=normalized_state(state)
        return CYAN if state=='MONITORING' else AMBER if state=='POSSIBLE CONVERGENCE' else MUTED

    @staticmethod
    def _valid_box(box):
        try:b=np.asarray(box,float)
        except (TypeError,ValueError):return False
        return b.shape==(4,) and np.isfinite(b).all() and b[2]>b[0] and b[3]>b[1]

    def _short(self,canvas,x,y,text,width,size=18,color=WHITE,bold=False):
        draw=ImageDraw.Draw(canvas);font=self.fonts[(size,bold)];value=str(text)
        while value and draw.textlength(value,font=font)>width:value=value[:-2]+'…'
        self._text(canvas,x,y,value,size,color,bold)

    def _line(self,draw,clip,a,b,color,width=2,dashed=False):
        points=np.asarray([a,b],float)
        if not np.isfinite(points).all():return
        p,q=[tuple(int(np.clip(round(v),-10000000,10000000)) for v in xy) for xy in points]
        visible,p,q=cv2.clipLine(clip,p,q)
        if not visible:return
        if not dashed:draw.line((p,q),fill=color,width=width);return
        p=np.asarray(p,float);q=np.asarray(q,float);delta=q-p
        n=max(1,int(np.linalg.norm(delta)/11))
        for i in range(n):
            a=p+delta*(i/n);b=p+delta*((i+.48)/n)
            draw.line((tuple(a),tuple(b)),fill=color,width=width)

    def _groups(self,demo,width,height,fresh_ids):
        groups=demo.get('attention_regions')
        if groups is None:
            from .aviation_regions import compute_attention_regions
            groups=compute_attention_regions(demo.get('targets',[]),width,height)
        eligible=set()
        for target in demo.get('targets',[]):
            age=target.get('evidence_age_s')
            if normalized_state(target.get('state')) not in ('MONITORING','POSSIBLE CONVERGENCE'):continue
            if age is None or not np.isfinite(age) or abs(age)>1e-6:continue
            eligible.add(target['id'])
        eligible&=fresh_ids
        valid=[]
        for group in groups:
            hull=np.asarray(group.get('hull_px',[]),float)
            if hull.ndim!=2 or hull.shape[1:]!=(2,) or len(hull)<3 or not np.isfinite(hull).all():continue
            if not group.get('member_ids') or not set(group['member_ids']).issubset(eligible):continue
            valid.append(group)
        return valid

    def _draw_regions(self,canvas,groups,rect,image_size,unknown=False):
        x,y,w,h=rect;sw,sh=image_size
        polygons=[]
        for group in groups:
            polygon=[(float(np.clip(px/sw,0,1)*w),float(np.clip(py/sh,0,1)*h)) for px,py in group['hull_px']]
            polygons.append((polygon,AMBER if group.get('severity')=='convergence' else CYAN))
        overlay=Image.new('RGBA',(w,h),(0,0,0,0));draw=ImageDraw.Draw(overlay)
        if unknown:
            mask=Image.new('L',(w,h),255);md=ImageDraw.Draw(mask)
            for polygon,_ in polygons:md.polygon(polygon,fill=0)
            hatch=Image.new('RGBA',(w,h),(0,0,0,0));hd=ImageDraw.Draw(hatch)
            for offset in range(-h,w,28):hd.line((offset,0,offset+h,h),fill=(*UNKNOWN,35),width=1)
            overlay.paste(hatch,(0,0),mask)
        for polygon,color in polygons:
            draw.polygon(polygon,fill=(*color,26 if unknown else 13),outline=(*color,170 if unknown else 110),width=2)
        base=canvas.crop((x,y,x+w,y+h)).convert('RGBA')
        canvas.paste(Image.alpha_composite(base,overlay).convert('RGB'),(x,y))

    def _symbol(self,draw,center,color,chosen=False,stale=False,convergence=False,radius=None):
        x,y=center;r=radius or (12 if chosen else 8)
        if stale:
            draw.line((x-r,y-r,x+r,y+r),fill=UNKNOWN,width=2)
            draw.line((x-r,y+r,x+r,y-r),fill=UNKNOWN,width=2)
        else:
            if convergence:
                draw.line(((x,y-r-2),(x+r+2,y+r),(x-r-2,y+r),(x,y-r-2)),fill=color,width=2)
                draw.line((x,y-r/2,x,y+r/4),fill=color,width=2)
                draw.ellipse((x-1,y+r/2,x+1,y+r/2+2),fill=color)
            else:draw.line(((x,y-r),(x+r,y),(x,y+r),(x-r,y),(x,y-r)),fill=color,width=3 if chosen else 2)
            if chosen:draw.ellipse((x-r-8,y-r-8,x+r+8,y+r+8),outline=WHITE,width=1)

    def render(self,frame,objects,t,camera,advice):
        if not isinstance(frame,np.ndarray) or frame.ndim!=3 or frame.shape[2]!=3:
            raise ValueError('BGR RGB-source image required')
        source_h,source_w=frame.shape[:2];demo=advice.get('demo') or {}
        cues={target['id']:target for target in demo.get('targets',[])}
        items=[]
        for obj in objects:
            if not self._valid_box(obj.get('box')):continue
            items.append({'object':obj,'cue':cues.get(obj['id'],{}),'id':obj['id'],
                'fresh':fresh_observation(obj,t),'class':source_class(obj)})
        items.sort(key=lambda x:x['id']);fresh=[i for i in items if i['fresh']]
        fresh_ids={i['id'] for i in fresh}
        requested=demo.get('priority_id');primary=next((i for i in fresh if i['id']==requested),None)
        cue_priority=primary is not None
        if primary is None:primary=next((i for i in fresh if i['id']==self.focus_id),None)
        if primary is None and fresh:
            primary=max(fresh,key=lambda i:max(i['object']['box'][2]-i['object']['box'][0],i['object']['box'][3]-i['object']['box'][1]))
        self.focus_id=primary['id'] if primary else None
        groups=self._groups(demo,source_w,source_h,fresh_ids)
        canvas=Image.new('RGB',self.size,BG);draw=ImageDraw.Draw(canvas)
        state=normalized_state(demo.get('state')) if fresh else 'INSUFFICIENT EVIDENCE';color=self._color(state)
        self._short(canvas,22,14,self.title,1420,30,WHITE,True)
        self._text(canvas,22,54,f'RGB  {source_w}×{source_h}',14,MUTED)
        self._text(canvas,1440,20,LABELS[state],22,color,True)
        self._text(canvas,1860,20,'ПОЛЕ КАДРА',24,WHITE,True)
        self._text(canvas,1861,55,'НОРМИРОВАННЫЕ КООРДИНАТЫ',13,MUTED)
        self._text(canvas,2726,20,'ОБЪЕКТЫ',24,WHITE,True)
        self._text(canvas,2880,24,f'{len(items):02d}',21,CYAN,True)
        self._text(canvas,3635,18,f'{int(max(0,t)//60):02d}:{max(0,t)%60:05.2f}',28,WHITE,True)
        draw.line((20,78,3820,78),fill=EDGE,width=1)

        video,ratio=fit_rect(source_w,source_h,self.video_area)
        vx,vy,vw,vh=video
        draw.rectangle((20,88,1820,1024),fill=(2,7,12),outline=EDGE,width=1)
        rgb=cv2.cvtColor(cv2.resize(frame,(vw,vh),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
        canvas.paste(Image.fromarray(rgb),(vx,vy))
        self._draw_regions(canvas,groups,video,(source_w,source_h))
        draw=ImageDraw.Draw(canvas)
        labels=[]
        for item in fresh:
            obj=item['object'];cue=item['cue'];box=obj['box'];chosen=item['id']==self.focus_id
            c=self._color(cue.get('state'));x1,y1,x2,y2=box
            a=(vx+x1*ratio,vy+y1*ratio);b=(vx+x2*ratio,vy+y2*ratio)
            center=((a[0]+b[0])/2,(a[1]+b[1])/2)
            trail=cue.get('measured_trail') or []
            visible=[(vx+p[1]*ratio,vy+p[2]*ratio) for p in trail
                if len(p)>=3 and np.isfinite(p[:3]).all() and 0<=t-p[0]<=2.5]
            for p,q in zip(visible,visible[1:]):self._line(draw,video,p,q,c,3 if chosen else 2)
            velocity=cue.get('velocity_px_s')
            if velocity is not None and np.asarray(velocity).shape==(2,) and np.isfinite(velocity).all():
                end=np.asarray(center)+np.asarray(velocity,float)*ratio*.5
                self._line(draw,video,center,end,c,2,True)
            pad=9 if chosen else 6
            bounds=(max(vx,a[0]-pad),max(vy,a[1]-pad),min(vx+vw-1,b[0]+pad),min(vy+vh-1,b[1]+pad))
            if bounds[0]>=bounds[2] or bounds[1]>=bounds[3]:continue
            corner=14 if chosen else 8
            for x,dx in [(bounds[0],1),(bounds[2],-1)]:
                for y,dy in [(bounds[1],1),(bounds[3],-1)]:
                    draw.line(((x,y+dy*corner),(x,y),(x+dx*corner,y)),fill=c,width=3 if chosen else 2)
            label=f"ОБЪЕКТ {item['id']:02d}" if chosen else f"{item['id']:02d}"
            fw=self.fonts[(17,True)];width=draw.textlength(label,font=fw)+20
            x=max(vx+3,min(bounds[0],vx+vw-width-3));y=max(vy+3,bounds[1]-31)
            if any(abs(x-p)<width and abs(y-q)<30 for p,q in labels):y=min(vy+vh-31,bounds[3]+6)
            labels.append((x,y));draw.rounded_rectangle((x,y,x+width,y+28),radius=5,fill=PANEL,outline=c,width=1)
            self._text(canvas,x+10,y+2,label,17,c,True)

        # Rectangular image field: no attitude, heading, ownship or world-map data.
        draw.rounded_rectangle((1840,88,2688,1024),radius=10,fill=PANEL,outline=EDGE,width=1)
        field,_=fit_rect(source_w,source_h,self.field_area);fx,fy,fw,fh=field
        self._draw_regions(canvas,groups,field,(source_w,source_h),True);draw=ImageDraw.Draw(canvas)
        for frac in [0,.25,.5,.75,1]:
            gx=fx+frac*fw;gy=fy+frac*fh
            draw.line((gx,fy,gx,fy+fh),fill=EDGE,width=1)
            draw.line((fx,gy,fx+fw,gy),fill=EDGE,width=1)
        draw.rectangle((fx,fy,fx+fw,fy+fh),outline=(88,114,132),width=2)
        ox=fx+fw/2;oy=fy+fh/2
        draw.line((ox-18,oy,ox-6,oy),fill=WHITE,width=2);draw.line((ox+6,oy,ox+18,oy),fill=WHITE,width=2)
        draw.line((ox,oy-18,ox,oy-6),fill=WHITE,width=2);draw.line((ox,oy+6,ox,oy+18),fill=WHITE,width=2)
        for frac,label in [(0,'−1'),(.5,'0'),(1,'+1')]:
            self._text(canvas,fx+frac*fw-7,fy+fh+8,label,14,MUTED)
        self._text(canvas,fx+fw+6,fy-8,'−1',13,MUTED)
        self._text(canvas,fx+fw+6,fy+fh-12,'+1',13,MUTED)
        plot_centers=[];occupied_labels=[];field_label_ids=[];group_badges=[]
        clustered_ids={tid for group in groups if len(group.get('member_ids',[]))>1 for tid in group['member_ids']}
        for item in items:
            box=item['object']['box'];cue=item['cue'];c=self._color(cue.get('state')) if item['fresh'] else UNKNOWN
            center=(fx+(box[0]+box[2])/2/source_w*fw,fy+(box[1]+box[3])/2/source_h*fh)
            center=(float(np.clip(center[0],fx,fx+fw)),float(np.clip(center[1],fy,fy+fh)))
            if item['fresh']:
                trail=[(fx+p[1]/source_w*fw,fy+p[2]/source_h*fh) for p in cue.get('measured_trail',[])
                       if len(p)>=3 and np.isfinite(p[:3]).all() and 0<=t-p[0]<=2.5]
                for a,b in zip(trail,trail[1:]):self._line(draw,field,a,b,c,2)
                velocity=cue.get('velocity_px_s')
                if velocity is not None and np.asarray(velocity).shape==(2,) and np.isfinite(velocity).all():
                    end=np.array(center)+np.asarray(velocity)*np.array([fw/source_w,fh/source_h])*.5
                    self._line(draw,field,center,end,c,2,True)
            self._symbol(draw,center,c,item['id']==self.focus_id,not item['fresh'],
                         normalized_state(cue.get('state'))=='POSSIBLE CONVERGENCE')
            if item['id'] not in clustered_ids or item['id']==self.focus_id:
                label=f"{item['id']:02d}";label_width=draw.textlength(label,font=self.fonts[(16,True)])
                lx=min(fx+fw-label_width-4,center[0]+17);ly=max(fy,min(fy+fh-23,center[1]-12))
                self._text(canvas,lx,ly,label,16,c,True)
                occupied_labels.append((lx-3,ly-1,lx+label_width+3,ly+23));field_label_ids.append(item['id'])
            plot_centers.append(center)

        def intersection(a,b):
            return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        symbol_rects=[(x-15,y-15,x+15,y+15) for x,y in plot_centers]
        for group in groups:
            members=group.get('member_ids',[])
            if len(members)<2:continue
            hull=np.asarray(group['hull_px'],float)
            points=hull*np.array([fw/source_w,fh/source_h])+[fx,fy]
            left,top=points.min(axis=0);right,bottom=points.max(axis=0);cx,cy=points.mean(axis=0)
            badge=f'×{len(members)}';bw=draw.textlength(badge,font=self.fonts[(18,True)])+18;bh=28
            def bounded(x,y):
                x=max(fx+4,min(x,fx+fw-bw-4));y=max(fy+4,min(y,fy+fh-bh-4))
                return (x,y,x+bw,y+bh)
            candidates=[bounded(right-bw/2,top-bh-7),bounded(right+7,cy-bh/2),
                bounded(left-bw-7,cy-bh/2),bounded(cx-bw/2,bottom+7)]
            def overlap(rect):return sum(intersection(rect,r) for r in occupied_labels)+sum(intersection(rect,r) for r in symbol_rects)
            if min(map(overlap,candidates))>0:
                candidates.extend(bounded(x,y) for y in range(fy+4,fy+fh-bh,int(bh+8))
                    for x in range(fx+4,fx+fw-int(bw),int(bw+8)))
            rect=min(candidates,key=lambda r:(overlap(r),((r[0]+r[2])/2-cx)**2+((r[1]+r[3])/2-cy)**2))
            c=AMBER if group.get('severity')=='convergence' else CYAN
            draw.rounded_rectangle(rect,radius=5,fill=PANEL,outline=c,width=1)
            self._text(canvas,rect[0]+9,rect[1]+1,badge,18,c,True)
            occupied_labels.append(rect);group_badges.append({'member_ids':list(members),'count':len(members),'rect':list(rect)})

        self._text(canvas,1862,954,'ОСИ ИЗОБРАЖЕНИЯ',14,MUTED,True)
        if primary:
            self._short(canvas,1862,984,f"ОБЪЕКТ {primary['id']:02d}  ·  {source_class(primary['object'])}",560,20,WHITE,True)
        else:self._text(canvas,1862,984,'ОБЪЕКТ —',20,MUTED)
        direction=demo.get('direction') or {};vector=direction.get('vector')
        vector=np.asarray(vector,float) if vector is not None else np.zeros(2)
        resolved=cue_priority and direction.get('status')=='resolved' and vector.shape==(2,) and np.isfinite(vector).all() and np.linalg.norm(vector)>1e-8
        if resolved:
            v=vector/np.linalg.norm(vector)
            left_space=vx-self.video_area[0]
            if left_space>=185:anchor=np.array([20+left_space/2,840.])
            else:anchor=np.array([2560.,900.])
            ax,ay=anchor
            draw.ellipse((ax-75,ay-75,ax+75,ay+75),fill=(10,30,43),outline=(57,108,128),width=2)
            start=anchor-v*43;end=anchor+v*50;normal=np.array([-v[1],v[0]])
            draw.line((tuple(start),tuple(end-v*20)),fill=CYAN,width=17)
            draw.polygon([tuple(end),tuple(end-v*33+normal*26),tuple(end-v*33-normal*26)],fill=CYAN)
            self._text(canvas,ax-75,ay+88,'ИЛЛЮСТРАЦИЯ',14,CYAN,True)

        # All tracked objects receive a tile. Stale positions never select a new
        # crop from the current frame; their card stays visibly unobserved.
        cells=roi_grid_layout(len(items),self.roi_area)
        draw.rounded_rectangle((2708,88,3830,1024),radius=10,fill=PANEL,outline=EDGE,width=1)
        for item,cell in zip(items,cells):
            x,y,w,h=cell;chosen=item['id']==self.focus_id
            c=self._color(item['cue'].get('state')) if item['fresh'] else UNKNOWN
            draw.rounded_rectangle((x,y,x+w,y+h),radius=7,fill=(8,17,26),outline=WHITE if chosen else c,width=3 if chosen else 1)
            large=w>=400 and h>=250;normal=w>=240 and h>=180
            title_size=24 if large else 20 if normal else 16
            class_size=18 if large else 15 if normal else 13
            header=84 if large else 72 if normal else 50
            footer=32 if h>=250 else 28 if h>=180 else 22
            self._short(canvas,x+10,y+6,f"ОБЪЕКТ {item['id']:02d}",w-43,title_size,c,True)
            self._symbol(draw,(x+w-20,y+title_size/2+9),c,False,not item['fresh'],
                         normalized_state(item['cue'].get('state'))=='POSSIBLE CONVERGENCE',6)
            self._wrapped(canvas,x+10,y+title_size+12,item['class'],w-20,class_size,WHITE if item['fresh'] else MUTED,
                          max_lines=2 if large or normal else 1,line_height=class_size+2)
            image_area=(x+8,y+header,max(1,w-16),max(1,h-header-footer-6))
            ix,iy,iw,ih=image_area
            draw.rectangle((ix,iy,ix+iw,iy+ih),fill=(4,10,16))
            obj=item['object'];box=obj['box']
            if item['fresh']:
                a,b,z,q=crop_bounds(box,source_w,source_h,iw/ih);crop=frame[b:q,a:z]
                if crop.size:
                    fitted,_=fit_rect(crop.shape[1],crop.shape[0],image_area);px,py,pw,ph=fitted
                    thumb=cv2.cvtColor(cv2.resize(crop,(pw,ph),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
                    canvas.paste(Image.fromarray(thumb),(px,py))
            else:
                self._line(draw,image_area,(ix,iy),(ix+iw,iy+ih),UNKNOWN,1)
                self._line(draw,image_area,(ix+iw,iy),(ix,iy+ih),UNKNOWN,1)
                self._short(canvas,ix+6,iy+max(0,(ih-18)//2),'НЕТ СВЕЖЕГО RGB' if w>=240 else 'НЕТ RGB',iw-12,14,MUTED)
            extent=f'{max(0,round(box[2]-box[0]))}×{max(0,round(box[3]-box[1]))} пкс'
            distance=range_text(item['cue']) if item['fresh'] else 'D —'
            badge_font=self.fonts[(13,False)];badge_width=min(iw-4,draw.textlength(extent,font=badge_font)+10)
            draw.rectangle((ix+iw-badge_width-3,iy+3,ix+iw-3,iy+23),fill=PANEL)
            self._short(canvas,ix+iw-badge_width+1,iy+3,extent,badge_width-7,13,MUTED)
            growth=item['cue'].get('growth_per_s') if item['fresh'] else None
            valid_growth=isinstance(growth,(int,float,np.floating)) and np.isfinite(growth)
            growth_text=(('↑ ' if growth>=0 else '↓ ')+f'{growth*100:+.0f}%/с') if valid_growth else '—'
            metric_size=16 if large else 13;font=self.fonts[(metric_size,False)]
            distance_width=draw.textlength(distance,font=font)
            self._text(canvas,x+w-10-distance_width,y+h-footer+3,distance,metric_size,MUTED)
            self._short(canvas,x+10,y+h-footer+3,growth_text,w-30-distance_width,metric_size,c)
        if not items:self._text(canvas,2750,160,'ОБЪЕКТЫ НЕ НАБЛЮДАЮТСЯ',22,MUTED,True)

        # Three graphic keys, with no verified-safe / green complement.
        draw.line((22,1056,70,1056),fill=CYAN,width=3);self._text(canvas,82,1044,'измерено',16,MUTED)
        for xx in [255,271,287]:draw.line((xx,1056,xx+7,1056),fill=WHITE,width=2)
        self._text(canvas,310,1044,'проекция 0,5 с',16,MUTED)
        draw.rectangle((520,1047,552,1068),outline=UNKNOWN)
        for xx in [522,532,542]:draw.line((xx,1067,xx+9,1048),fill=UNKNOWN,width=1)
        self._text(canvas,568,1044,'не оценено',16,MUTED)
        self._text(canvas,1250,1045,'Иллюстрация в осях кадра · не руководство полётом',17,WHITE)
        if demo.get('range_preset'):self._text(canvas,3050,1046,'≈ по заданным размаху / FOV',15,MUTED)
        self.last_layout={'source_rect':video,'field_rect':field,'roi_rects':cells,
            'object_ids':[i['id'] for i in items],'fresh_ids':sorted(fresh_ids),'focus_id':self.focus_id,
            'direction_drawn':bool(resolved),'region_count':len(groups),'state':state,
            'field_label_ids':field_label_ids,'group_badges':group_badges}
        return cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)

    def draw(self,frame,objects,t,camera,advice):return self.render(frame,objects,t,camera,advice)


Renderer=AviationRenderer
