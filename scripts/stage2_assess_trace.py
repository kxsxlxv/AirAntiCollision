"""Shared frozen-trace assessment, with no GT access or physical-state guesses."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from airavoid.assessment import Assessor,Measurement,Context,METHODS,scope_result,LEGACY,STATES
from airavoid.policy import TemporalPolicy
from airavoid.provenance import file_sha256


def replay(path,output,width,height,methods=METHODS,camera_mode='auto',camera_model='raw'):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    estimators={m:Assessor(m) for m in methods};policies={m:TemporalPolicy() for m in methods}
    histories=defaultdict(list);last={};epoch=None;counts=defaultdict(Counter);timings=defaultdict(list)
    size=0;start=time.perf_counter()
    source_sha=file_sha256(path)
    with Path(path).open() as src,output.open('w') as dest:
        for text in src:
            row=json.loads(text);t=row['time'];cam=row['camera']
            if camera_model!='raw' and cam.get('reference_epoch')!=epoch:
                histories.clear();last.clear();epoch=cam.get('reference_epoch')
                policies={m:TemporalPolicy() for m in methods}
            transform=np.array(row['stabilization_transform'])
            context=Context(t,width,height,camera_mode,bool(cam['valid']))
            objects=[]
            for obs in row['measurements']:
                tid=obs['track_id'];b=obs['measured_bbox'];x1,y1,x2,y2=b
                clipped=x1<=1 or y1<=1 or x2>=width-1 or y2>=height-1
                ambiguous=obs.get('ambiguous',False)
                if clipped or ambiguous:
                    histories[tid].clear()
                if histories[tid] and t-histories[tid][-1].time>.8:
                    histories[tid].clear()
                position=None;scale=1.
                if camera_model=='similarity':
                    center=transform@np.array([(x1+x2)/2,(y1+y2)/2,1.])
                    position=tuple(center[:2]);scale=float(np.sqrt(abs(np.linalg.det(transform[:2,:2]))))
                measurement=Measurement(t,tuple(b),tid,obs['confidence'],source=obs.get('source','detector'),
                                        clipped=clipped,ambiguous=ambiguous,position=position,extent_scale=scale)
                # A clipped sample is returned as invalid but never seeds the next fit.
                last[tid]=measurement
                if not clipped and not ambiguous:histories[tid].append(measurement)
            # Include recently lost known tracks explicitly as unobserved/uncertain.
            for tid in [x for x,h in last.items() if t-h.time<=.8]:
                current=last[tid];history=histories[tid]
                inputs=history if history and history[-1] is current else [*history,current]
                results={}
                for method,estimator in estimators.items():
                    tick=time.perf_counter();raw=estimator.assess(inputs,context,tid)
                    result=scope_result(raw,context)
                    e={'state':LEGACY[result['decision']],'reason':'|'.join(result['reason_codes'])}
                    if e['state']=='unobserved':
                        policies[method].pending.pop(tid,None)
                    else:
                        e=policies[method].apply(tid,t,e)
                        result['decision']=STATES[e['state']]
                        result['confirmation_elapsed_s']=e.get('confirmation_elapsed_s')
                        result['confirmation_required_s']=e.get('confirmation_required_s')
                        if e['reason']!=('|'.join(result['reason_codes'])):
                            result['reason_codes']=[e['reason'],*result['reason_codes']]
                    timings[method].append((time.perf_counter()-tick)*1000)
                    counts[method][result['decision']]+=1;results[method]=result
                objects.append({'track_id':tid,'box':list(current.box),'measurement_time_s':current.time,
                                'measurement_source':current.source,'confidence':current.confidence,
                                'measured_raw_span':max(current.box[2]-current.box[0],current.box[3]-current.box[1]),
                                'extent_correction':current.extent_scale,'results':results})
            dest.write(json.dumps({'schema':'stage2-assessment-v1','frame':row['frame'],'time':t,
                                   'source_pts_s':row.get('source_pts_s'),'camera':cam,
                                   'camera_condition':camera_model,'objects':objects,
                                   'input_truth_condition':row.get('input_truth_condition','predicted_detections_and_IDs')},allow_nan=False)+'\n')
            size+=1
            if size%1000==0:print(f'{Path(path).stem}: {size} assessment frames',flush=True)
    report={'source_trace_sha256':source_sha,'source_trace':str(path),'output_trace_sha256':file_sha256(output),
            'methods':list(methods),'frames':size,'camera_model':camera_model,
            'counts_predicted_track_samples':counts,'elapsed_replay_s':time.perf_counter()-start,
            'runtime_scope':'cached observations only; not neural-inference FPS',
            'policy':{'warning_seconds':.24,'nonconflict_seconds':1.2,'lost_or_uncertain_resets':True},
            'method_assessment_ms':{m:{k:float(np.percentile(a,q)) for k,q in [('p50',50),('p95',95),('p99',99)]} for m,a in timings.items()},
            'raw_published_candidates_retained':True,'confidence_calibration':None}
    output.with_suffix('.summary.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width',type=int,required=True);p.add_argument('--height',type=int,required=True)
    p.add_argument('--camera',choices=['fixed','auto','unknown'],default='auto')
    p.add_argument('--camera-model',choices=['raw','similarity'],default='raw')
    p.add_argument('--methods',default=','.join(METHODS))
    a=p.parse_args();print(json.dumps(replay(a.trace,a.output,a.width,a.height,a.methods.split(','),a.camera,a.camera_model),indent=2))
