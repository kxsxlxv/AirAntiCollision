"""Focused independent ORB/homography camera-plane control on identical predicted tracks."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from stage2_assess_trace import replay
from stage2_threat_score import score
from airavoid.assessment import METHODS
from airavoid.provenance import file_sha256


def main():
    audit=json.loads((ROOT/'reports/stage2_dataset_audit.json').read_text())
    matrix=json.loads((ROOT/'reports/stage2_threat_matrix.json').read_text())
    report={'condition':'A/custom/independent_ORB_homography','clips':{},
            'claim_scope':'background-plane camera hypothesis only; no recovered camera rotation or target depth',
            'truth':'NPS 2D identity/boxes only; no physical collision accuracy'}
    for clip in ['Clip_032','Clip_043']:
        src=ROOT/f'artifacts/stage2/camera_orb/{clip}.jsonl';normal=ROOT/f'artifacts/stage2/tracks/A/custom/s1/{clip}.jsonl'
        n=0
        with src.open() as a,normal.open() as b:
            for xa,xb in zip(a,b,strict=True):
                ra=json.loads(xa);rb=json.loads(xb)
                for key in ['frame','time','source_pts_s','measurements','tracks']:
                    if ra[key]!=rb[key]:raise ValueError('Camera ablation changed normal observations: '+key)
                n+=1
        w,h=audit['clips'][clip]['resolution']
        out=ROOT/f'artifacts/stage2/assessments/A/custom/orb_homography/{clip}.jsonl.gz'
        summary=replay(src,out,w,h,METHODS,'auto','homography')
        metrics=score(out,clip);out.with_suffix('.score.json').write_text(json.dumps(metrics,indent=2))
        report['clips'][clip]={'identical_predicted_observation_frames':n,'camera_trace_sha256':file_sha256(src),
                               'normal_trace_sha256':file_sha256(normal),'replay':summary,
                               'ORB_homography':metrics,
                               'raw':matrix['conditions']['A/custom/raw']['per_sequence'][clip],
                               'KLT_similarity':matrix['conditions']['A/custom/similarity']['per_sequence'][clip]}
    (ROOT/'reports/stage2_camera_threat_control.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
