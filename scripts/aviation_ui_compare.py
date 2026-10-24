"""Strict UI-only regression on completed LIVE exports, without model execution.

Only advisory.demo.attention_regions may be removed from a copied new decision.
Every other decision field, detector-cache byte, source PTS/frame and source/model
digest must match. Benchmark timing, layout arguments and titles are not compared.
Incomplete runs never produce a successful equivalence result.
"""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_PAIRS=[('demo/nps_range_showcase','demo/aviation_nps'),
               ('demo/nps_range_contrast','demo/aviation_contrast'),
               ('demo/final_showcase_approach','demo/aviation_approach')]


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def first_difference(old,new,path='$'):
    if type(old) is not type(new):return path+' [type]'
    if isinstance(old,dict):
        for key in sorted(set(old)|set(new)):
            if key not in old or key not in new:return path+'.'+key+' [missing/added]'
            found=first_difference(old[key],new[key],path+'.'+key)
            if found:return found
    elif isinstance(old,list):
        if len(old)!=len(new):return path+' [length]'
        for i,(a,b) in enumerate(zip(old,new)):
            found=first_difference(a,b,f'{path}[{i}]')
            if found:return found
    elif old!=new:return path
    return None


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def valid_digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdefABCDEF' for c in value)


def source_clock(row):
    keys=['frame','time']+[key for key in ['source_pts_ms','source_pts_s'] if key in row]
    if len(keys)==2 or any(row.get(key) is None for key in keys):return None
    return canonical({key:row[key] for key in keys})


def compare_pair(baseline,run):
    baseline=Path(baseline).resolve();run=Path(run).resolve()
    result={'baseline':str(baseline),'run':str(run),'status':'incomplete',
            'equivalent_except_attention_regions':False,
            'allowed_new_field':'$.advisory.demo.attention_regions',
            'ignored_benchmark_fields':'performance timings, arguments/title/layout; no decision fields ignored'}
    if baseline==run:
        result.update(status='failed',error='Baseline and new LIVE run must be different directories')
        return result
    required=['run_status.json','benchmark.json','decisions.jsonl','detections.jsonl']
    missing=[str(folder/name) for folder in [baseline,run] for name in required if not (folder/name).is_file()]
    if missing:result['missing_files']=missing;return result
    statuses={name:read_json(folder/'run_status.json').get('state') for name,folder in [('baseline',baseline),('run',run)]}
    result['run_states']=statuses
    if any(state!='complete' for state in statuses.values()):return result
    old_benchmark=read_json(baseline/'benchmark.json');new_benchmark=read_json(run/'benchmark.json')
    checks={}
    for key in ['input_sha256','weights_sha256','frames']:
        a=old_benchmark.get(key);b=new_benchmark.get(key)
        valid=(isinstance(a,int) and not isinstance(a,bool) and a>0 and
               isinstance(b,int) and not isinstance(b,bool) and b>0) if key=='frames' else valid_digest(a) and valid_digest(b)
        checks[key]={'baseline':a,'run':b,'valid_values':valid,'equal_and_present':valid and a==b}
    result['benchmark_checks']=checks
    old_detection_sha=sha(baseline/'detections.jsonl');new_detection_sha=sha(run/'detections.jsonl')
    result['detections']={'baseline_sha256':old_detection_sha,'run_sha256':new_detection_sha,
                          'byte_identical':old_detection_sha==new_detection_sha}
    result['benchmark_sha256']={'baseline':sha(baseline/'benchmark.json'),'run':sha(run/'benchmark.json')}
    result['decision_file_sha256']={'baseline':sha(baseline/'decisions.jsonl'),'run':sha(run/'decisions.jsonl')}
    old_digest=hashlib.sha256();new_digest=hashlib.sha256();old_count=0;new_count=0
    mismatch=[];clock_mismatch=[];details=[];added_regions=0;nonempty_regions=0
    with (baseline/'decisions.jsonl').open(encoding='utf-8') as a,(run/'decisions.jsonl').open(encoding='utf-8') as b:
        for index,(old_text,new_text) in enumerate(zip_longest(a,b),start=1):
            old=json.loads(old_text) if old_text is not None else None
            new=json.loads(new_text) if new_text is not None else None
            old_count+=old is not None;new_count+=new is not None
            copied=copy.deepcopy(new)
            if copied is not None:
                demo=(copied.get('advisory') or {}).get('demo')
                if isinstance(demo,dict) and 'attention_regions' in demo:
                    nonempty_regions+=bool(demo['attention_regions']);added_regions+=1
                    del demo['attention_regions']
            old_canonical=canonical(old) if old is not None else None
            new_canonical=canonical(copied) if copied is not None else None
            if old_canonical is not None:old_digest.update((old_canonical+'\n').encode('utf-8'))
            if new_canonical is not None:new_digest.update((new_canonical+'\n').encode('utf-8'))
            frame={'baseline_frame':old.get('frame') if old is not None else None,
                   'run_frame':new.get('frame') if new is not None else None,'line':index}
            if old_canonical!=new_canonical:
                mismatch.append(frame)
                if len(details)<20:details.append({**frame,'first_differing_path':first_difference(old,copied)})
            if old is None or new is None or source_clock(old) is None or source_clock(old)!=source_clock(new):
                clock_mismatch.append(frame)
    result['decisions']={'baseline_frames':old_count,'run_frames':new_count,
                         'mismatch_count':len(mismatch),'mismatch_frame_ids':mismatch,
                         'first_mismatch_details':details,
                         'baseline_canonical_sha256':old_digest.hexdigest(),
                         'run_after_only_allowed_removal_canonical_sha256':new_digest.hexdigest(),
                         'new_attention_region_field_frames':added_regions,
                         'new_nonempty_attention_region_frames':nonempty_regions,
                         'all_other_fields_exact':not mismatch and old_count==new_count and old_count>0}
    result['source_clock']={'all_frame_time_source_PTS_exact':not clock_mismatch and old_count==new_count and old_count>0,
                            'mismatch_count':len(clock_mismatch),'mismatch_frame_ids':clock_mismatch}
    result['frame_counts_match_completed_benchmarks']=(old_count==old_benchmark.get('frames') and
                                                       new_count==new_benchmark.get('frames') and old_count>0)
    passed=(all(check['equal_and_present'] for check in checks.values()) and
            result['detections']['byte_identical'] and result['decisions']['all_other_fields_exact'] and
            result['source_clock']['all_frame_time_source_PTS_exact'] and result['frame_counts_match_completed_benchmarks'])
    result['status']='passed' if passed else 'failed'
    result['equivalent_except_attention_regions']=bool(passed)
    return result


def self_test():
    """Small positive control plus real negative controls for forbidden changes."""
    with tempfile.TemporaryDirectory(prefix='aviation-ui-compare-') as temp:
        old=Path(temp)/'old';new=Path(temp)/'new';old.mkdir();new.mkdir()
        row={'frame':0,'time':0.,'source_pts_ms':0.,'objects':[{'id':7,'box':[1.,2.,3.,4.]}],
             'advisory':{'demo':{'priority_id':7,'direction':{'status':'resolved','vector':[1.,0.]},
                                  'targets':[{'id':7,'range_m_approx':42.}]}}}
        def write(folder,value,complete=True,detection='same detector bytes\n'):
            (folder/'run_status.json').write_text(json.dumps({'state':'complete' if complete else 'running'}))
            (folder/'benchmark.json').write_text(json.dumps({'frames':1,'input_sha256':hashlib.sha256(b'source').hexdigest(),
                                                            'weights_sha256':hashlib.sha256(b'weights').hexdigest()}))
            (folder/'decisions.jsonl').write_text(json.dumps(value)+'\n')
            (folder/'detections.jsonl').write_text(detection)
        write(old,row);allowed=copy.deepcopy(row);allowed['advisory']['demo']['attention_regions']=[];write(new,allowed)
        assert compare_pair(old,new)['status']=='passed'
        changed=copy.deepcopy(allowed);changed['advisory']['demo']['targets'][0]['range_m_approx']=43.;write(new,changed)
        assert compare_pair(old,new)['status']=='failed'
        assert compare_pair(old,new)['decisions']['mismatch_count']==1
        write(new,allowed,detection='different detector bytes\n');assert compare_pair(old,new)['status']=='failed'
        changed=copy.deepcopy(allowed);changed['source_pts_ms']=1.;write(new,changed)
        assert compare_pair(old,new)['status']=='failed'
        write(new,allowed,complete=False);assert compare_pair(old,new)['status']=='incomplete'
        print('PASS: allowed region addition; forbidden range/detector/PTS changes; unfinished run remains incomplete')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path);parser.add_argument('--run',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/aviation_ui_regression.json')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return 0
    if (args.baseline is None)!=(args.run is None):parser.error('--baseline and --run must be supplied together')
    pairs=[(args.baseline,args.run)] if args.baseline is not None else [(ROOT/a,ROOT/b) for a,b in DEFAULT_PAIRS]
    results=[]
    for baseline,run in pairs:
        try:results.append(compare_pair(baseline,run))
        except (OSError,ValueError,TypeError,KeyError) as error:
            results.append({'baseline':str(baseline),'run':str(run),'status':'failed',
                            'equivalent_except_attention_regions':False,'error':f'{type(error).__name__}: {error}'})
    states=[result['status'] for result in results]
    state='failed' if 'failed' in states else 'incomplete' if 'incomplete' in states else 'passed'
    report={'kind':'strict UI-only completed LIVE-run regression','timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'state':state,'all_runs_equivalent_except_attention_regions':state=='passed',
            'scope':'No model execution, GT access, tolerances, rounded metrics or ignored decision fields beyond the single declared new field',
            'script_sha256':sha(Path(__file__)),'pairs':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'state':state,'output':str(args.output),
                      'pairs':[{'run':r['run'],'status':r['status'],'decision_mismatches':r.get('decisions',{}).get('mismatch_count')} for r in results]},indent=2))
    return 0 if state=='passed' else 2 if state=='incomplete' else 1


if __name__=='__main__':sys.exit(main())
