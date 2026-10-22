"""Numerical agreement of exact supplied P2 ONNX and its local FP32 TRT rebuild.

Development images only; this functional check makes no throughput claim.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import av
import numpy as np
from airavoid.detector_p2 import DetectorP2,letterbox
from airavoid.assessment_runtime import detector_contract
from airavoid.provenance import file_sha256


def compare():
    audit=json.loads((ROOT/'reports/stage2_dataset_audit.json').read_text())
    onnx=Path('X:/MODELS/yolo26m-p2-1440/best.onnx')
    engine=ROOT/'artifacts/stage2/yolo26m_p2_local_fp32.engine'
    trt=DetectorP2(engine,.03)
    reference=DetectorP2(onnx,.03,backend='cpu')
    contract=detector_contract(trt,'B')
    report={'scope':'Numerical functional agreement only; concurrent CPU research tasks; no FPS claim',
        'onnx_sha256':file_sha256(onnx),'engine_sha256':file_sha256(engine),
        'reference_provider':reference.providers,'runtime_provider':trt.providers,
        'contract':contract,'samples':[]}
    for clip,indices in [('Clip_032',[0,30,60,90,120,149]),('Clip_001',[0,60,120,149])]:
        source=audit['clips'][clip]['video_path']
        with av.open(source) as container:
            stream=container.streams.video[0]
            for index,frame in enumerate(container.decode(stream)):
                if index not in indices:continue
                image=frame.to_ndarray(format='bgr24');h,w=image.shape[:2]
                x,ratio,left,top=letterbox(image)
                ref=reference.session.run(None,{reference.input_name:x})[0][0]
                decoded_trt=trt._infer(image)
                raw_trt=trt.buffers[trt.output_name].float().cpu().numpy()[0].copy()
                decoded_ref=reference._decode(ref.copy(),h,w,ratio,left,top)
                # End-to-end top-k ordering can permute equal-score rows. Match
                # accepted boxes by minimum coordinate error, once each.
                pairs=[];used=set()
                for box in decoded_ref:
                    choices=[(float(np.max(np.abs(box[:4]-other[:4]))),j)
                             for j,other in enumerate(decoded_trt) if j not in used]
                    if not choices:continue
                    error,j=min(choices);used.add(j)
                    pairs.append({'max_coordinate_error_px':error,
                        'score_error':float(abs(box[4]-decoded_trt[j,4])),
                        'class_equal':bool(box[5]==decoded_trt[j,5])})
                record={'clip':clip,'frame':index,'source_pts_s':float(frame.pts*frame.time_base),
                    'resolution':[w,h],'letterbox_ratio':ratio,'left_pad':left,'top_pad':top,
                    'reference_accepted_boxes':len(decoded_ref),'trt_accepted_boxes':len(decoded_trt),
                    'pairs':pairs,'accepted_counts_equal':len(decoded_ref)==len(decoded_trt),
                    'top_score_onnx':float(np.max(ref[:,4])),
                    'top_score_trt':float(np.max(raw_trt[:,4])),
                    'all_accepted_classes_equal':all(pair['class_equal'] for pair in pairs)}
                report['samples'].append(record)
                print(clip,index,len(decoded_ref),len(decoded_trt),flush=True)
                if index==max(indices):break
    pairs=[p for sample in report['samples'] for p in sample['pairs']]
    report['summary']={'development_frames':len(report['samples']),
        'all_accepted_counts_equal':all(r['accepted_counts_equal'] for r in report['samples']),
        'all_accepted_classes_equal':all(r['all_accepted_classes_equal'] for r in report['samples']),
        'matched_boxes':len(pairs),
        'max_coordinate_error_px':max((p['max_coordinate_error_px'] for p in pairs),default=None),
        'max_score_error':max((p['score_error'] for p in pairs),default=None),
        'limitation':'Ten development frames do not establish equivalence on all inputs; low-score threshold crossings are reported if observed'}
    (ROOT/'reports/stage2_p2_agreement.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':compare()
