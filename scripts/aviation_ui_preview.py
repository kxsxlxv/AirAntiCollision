"""CPU-only aviation UI previews from original RGB plus recorded online traces.

Real previews preserve observations and cue/range metadata; they do not run a
model or consume GT. Synthetic density fixtures are separately and visibly marked
layout tests and must never be presented as recorded-flight evidence.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def jread(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def resolve(path):
    p=Path(path)
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def renderer_class():
    return getattr(importlib.import_module('airavoid.aviation_render'),'AviationRenderer')


def regions_for(demo,width,height):
    compute=getattr(importlib.import_module('airavoid.aviation_regions'),'compute_attention_regions')
    if 'attention_regions' not in demo:
        demo['attention_regions']=compute(demo.get('targets',[]),width,height)
        return True
    return False


def payload_for(run,index):
    run=resolve(run);benchmark=jread(run/'benchmark.json')
    status=jread(run/'run_status.json')
    if status.get('state')!='complete':raise ValueError('Only completed recorded runs can supply real previews')
    source=resolve(benchmark['args']['video'])
    if benchmark.get('input_sha256')!=sha(source):raise ValueError('Original RGB hash differs from recorded run')
    record=None
    with (run/'decisions.jsonl').open(encoding='utf-8') as f:
        for text in f:
            row=json.loads(text)
            if row['frame']==index:record=row;break
    if record is None:raise ValueError(f'No recorded frame {index} in {run}')
    with av.open(str(source)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        for i,frame in enumerate(container.decode(stream)):
            if i==index:
                rgb=frame.to_ndarray(format='bgr24')
                pts=float(frame.pts*frame.time_base);break
        else:raise ValueError('Original RGB source frame absent')
    expected=record.get('source_pts_ms',1000*(record['time']+benchmark.get('source_timestamp_origin_s',0)))/1000
    if abs(pts-expected)>1e-6:raise ValueError('Original RGB does not match recorded acquisition PTS')
    metadata={'kind':'recorded_rgb_and_recorded_online_trace','source_rgb':str(source),'source_rgb_sha256':benchmark['input_sha256'],
              'recorded_run':str(run),'decisions_sha256':sha(run/'decisions.jsonl'),'frame':index,'time_s':record['time'],
              'source_pts_s':pts,'source_resolution':[rgb.shape[1],rgb.shape[0]],
              'recorded_payload_sha256':hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
              'normal_observations_and_states_recomputed':False,'GT_access':False,
              'actual_source_object_ids':[obj['id'] for obj in record['objects']]}
    return rgb,deepcopy(record),metadata


def fixture_for(count):
    if count not in [8,16,32]:raise ValueError('Fixture density must be 8,16 or32')
    width,height=1920,1080;t=10.
    canvas=Image.new('RGB',(width,height),(20,30,42));draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf',30)
    for x in range(0,width,120):draw.line((x,0,x,height),fill=(37,49,62),width=1)
    for y in range(0,height,120):draw.line((0,y,width,y),fill=(37,49,62),width=1)
    draw.text((36,28),f'SYNTHETIC LAYOUT FIXTURE / {count} OBJECTS / NOT RECORDED FLIGHT',font=font,fill=(255,210,95))
    names=[('airplane','самолёт'),('helicopter','вертолёт'),('bird','птица'),('balloon','аэростат'),('parachute','парашют')]
    cluster_count=math.ceil(count/4);columns=math.ceil(math.sqrt(cluster_count*16/9));rows=math.ceil(cluster_count/columns)
    offsets=[(-24,-18),(14,-12),(-8,16),(28,21)];objects=[];targets=[]
    for i in range(count):
        group=i//4;dx,dy=offsets[i%4]
        cx=200+(group%columns)*(width-400)/max(1,columns-1)+dx
        cy=260+(group//columns)*(height-460)/max(1,rows-1)+dy
        bw=20+(i%4)*7;bh=12+(i%3)*4;box=[cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2]
        state='POSSIBLE CONVERGENCE' if i%5==0 else 'INSUFFICIENT EVIDENCE' if i%4==0 else 'MONITORING'
        cls,display=names[i%len(names)];tid=100+i;velocity=[18.*math.sin(i),-12.*math.cos(i)]
        if i%5==0:cls,display='fixed-wing UAV','БПЛА самолётного типа'
        draw.line((cx-bw/2,cy,cx+bw/2,cy),fill=(215,226,236),width=4)
        draw.line((cx,cy-bh/2,cx,cy+bh/2),fill=(215,226,236),width=3)
        draw.text((cx+15,cy-34),f'F{i+1:02d}',font=font,fill=(180,197,213))
        objects.append({'id':tid,'box':box,'class_id':i%len(names),'class':cls,'class_display':display,
                        'source':'detector','confidence':.75,'last_seen':t,
                        'demo_observation':{'measured':True,'ambiguous':False,'clipped':False,'hits':20},
                        'assessment':{'state':'uncertain','reason':'synthetic_layout_fixture_only'}})
        trail=[[t-.5+j*.1,cx-(.5-j*.1)*velocity[0],cy-(.5-j*.1)*velocity[1]] for j in range(6)]
        targets.append({'id':tid,'box':box,'state':state,'state_id':state.lower().replace(' ','_'),
                        'reason':'synthetic_layout_fixture_only','reason_text':'SYNTHETIC LAYOUT FIXTURE',
                        'samples':6,'window_s':.5,'evidence_age_s':0.,'measured_trail':trail,
                        'velocity_px_s':velocity,'growth_per_s':.05 if state=='POSSIBLE CONVERGENCE' else 0.,
                        'drift_normalized_per_s':.1,'priority_score':1. if state=='POSSIBLE CONVERGENCE' else .2,
                        'detector_confidence':.75,'range_approx_m':None,'range_display_m':None})
    demo={'state':'POSSIBLE CONVERGENCE','targets':targets,'priority_id':100,'time':t,
          'meaning':'SYNTHETIC_LAYOUT_FIXTURE_NOT_RECORDED_EVIDENCE','fixture_only':True,
          'direction':{'status':'unresolved','vector':None,'reason':'synthetic_layout_fixture_no_flight_direction',
                       'meaning':'illustrative_image_plane_cue_not_safe_maneuver'}}
    record={'frame':0,'time':t,'objects':objects,'camera':{'valid':True,'mode':'fixed','status':'fixed_setup_assumption'},
            'advisory':{'recommendation':None,'demo':demo}}
    return np.asarray(canvas)[:,:,::-1].copy(),record,{'kind':'synthetic_layout_fixture_only','fixture_object_count':count,
        'no_recorded_flight_or_model_output':True,'states_and_geometry_are_explicit_test_inputs':True,
        'source_resolution':[width,height],'GT_access':False,'expected_roi_ids':[o['id'] for o in objects]}


def render_one(frame,record,metadata,output):
    output=resolve(output);output.parent.mkdir(parents=True,exist_ok=True)
    demo=record['advisory'].get('demo')
    if not isinstance(demo,dict):raise ValueError('Recorded run lacks the demo presentation trace')
    metadata['attention_regions_derived_for_presentation']=regions_for(demo,frame.shape[1],frame.shape[0])
    metadata['attention_regions']=deepcopy(demo['attention_regions'])
    renderer=renderer_class()();tick=time.perf_counter()
    rendered=renderer.render(frame,record['objects'],record['time'],record['camera'],record['advisory'])
    metadata['single_frame_CPU_render_s']=time.perf_counter()-tick
    if not isinstance(rendered,np.ndarray) or rendered.shape!=(1080,3840,3):
        raise ValueError(f'Aviation canvas must be3840x1080BGR, got{getattr(rendered,"shape",None)}')
    picture=Image.fromarray(rendered[:,:,::-1])
    if metadata['kind']=='synthetic_layout_fixture_only':
        draw=ImageDraw.Draw(picture);font=ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf',36)
        draw.rectangle((0,0,3840,58),fill=(100,57,8))
        draw.text((30,6),f'SYNTHETIC LAYOUT FIXTURE — {metadata["fixture_object_count"]} OBJECTS — NOT RECORDED FLIGHT / NO INFERENCE',font=font,fill=(255,239,188))
    picture.save(output)
    metadata.update(output=str(output),output_sha256=sha(output),output_resolution=[3840,1080],
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    renderer_source_sha256=sha(ROOT/'airavoid/aviation_render.py'),
                    grouping_source_sha256=sha(ROOT/'airavoid/aviation_regions.py'),
                    runtime_scope='CPU one-frame rendering of recorded data or labelled layout fixture; not neural-inference FPS')
    for attribute in ['last_layout','layout_metadata']:
        value=getattr(renderer,attribute,None)
        if value is not None:metadata['renderer_'+attribute]=value
    output.with_suffix('.preview.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    print(str(output),flush=True);return metadata


def main(args):
    results=[];outdir=resolve(args.output_dir)
    if args.suite:
        for run,index,name in [('demo/nps_range_showcase',307,'nps_primary'),
                               ('demo/nps_range_contrast',522,'nps_contrast'),
                               ('demo/final_showcase_approach',817,'qut_approach')]:
            results.append(render_one(*payload_for(run,index),outdir/f'{name}.png'))
        for count in [8,16,32]:results.append(render_one(*fixture_for(count),outdir/f'stress_{count:02d}.png'))
    elif args.stress:
        results.append(render_one(*fixture_for(args.stress),args.output or outdir/f'stress_{args.stress:02d}.png'))
    else:
        results.append(render_one(*payload_for(args.run,args.frame),args.output or outdir/f'frame_{args.frame}.png'))
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'preview_manifest.json').write_text(json.dumps({'previews':results,'real_vs_fixture_kept_explicit':True},indent=2,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='demo/nps_range_showcase')
    p.add_argument('--frame',type=int,default=307);p.add_argument('--output')
    p.add_argument('--output-dir',default='artifacts/aviation_ui/previews')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--suite',action='store_true');mode.add_argument('--stress',type=int,choices=[8,16,32])
    main(p.parse_args())
