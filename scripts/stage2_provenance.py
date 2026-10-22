"""Record exact inputs/revisions/deviations without distributing upstream assets."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from airavoid.provenance import file_sha256


def write():
    methods={
      'bauer_disc':{'source':['https://eprints.sztaki.hu/8949/1/Bauer_1168_3096377_ny.pdf',
                              'https://real.mtak.hu/100246/1/Hindawi_2019.pdf'],
                    'implementation':'Independent horizontal disc-corrected LS, 2016 Eq.8 and 2019 Eq.4/7',
                    'deviations':['Vertical extension pending inspection of JIRS derivation','Common 2/5 diameter, 60-second research thresholds, not flight threshold transfer'],
                    'prerequisites':['verified focal length and camera axes','disc-like constant horizontal dimensions','trajectory-frame compensation']},
      'lai_lin':{'source':['https://doi.org/10.3390/rs16050756'],'implementation':'Independent Eq.2–6; paper weights .5/.333/.167 and >.5 warning',
                 'deviations':['detector rectangle area replaces Mask R-CNN silhouette','undefined zero-motion cosine propagated as interval, not guessed','shared temporal policy reported separately from published index'],
                 'prerequisites':['first measured area anchor','measured image positions','no physical CPA semantics']},
      'airtrack_cues':{'source':['https://arxiv.org/html/2209.12849v3'],'implementation':'III-E exact pairwise area formula at later sample; 4-second endpoint pair',
                       'deviations':['no author Kalman acceleration state','no angular degrees/pixel without calibration','no collision classifier invented'],
                       'prerequisites':['constant projected target shape for depth-plane time interpretation']},
      'wcv_taumod':{'source':['https://github.com/nasa/daidalus/tree/1c7b58e5525d5cca91cbbf4a8b7337a488243769'],
                    'implementation':'Independent NASA modified-tau/DCPA and vertical TCOA predicates, exact future event partition',
                    'deviations':['not full alerting levels or maneuver bands','analytical test uses explicit 1.5m/35s thresholds, source SI defaults retained in API'],
                    'prerequisites':['verified relative metric state','explicit axes/units','constant-velocity horizon'],
                    'normal_NPS_QUT_status':'missing state, uncertain; analytical oracle-state comparator only'},
      'rv_c3bf':{'source':['https://arxiv.org/html/2610.21374v1'],'implementation':'Independent Eq.2/5 calibrated feature-flow regression and Eq.7 nominal horizontal cone',
                 'deviations':['single-block observer diagnostic, no complete FIFO observer','no kinematic/statistical certification bound','point observer, no verified footprint inflation','no QP/control outputs'],
                 'prerequisites':['calibrated de-rotated actual target feature flow','adequate rank/excitation'],
                 'normal_NPS_QUT_status':'missing calibration/proprioception/target flow; uncertain; not a full RV-C3BF reproduction'}
    }
    papers={p.name:{'sha256':file_sha256(p),'bytes':p.stat().st_size} for p in (ROOT/'external/stage2').glob('*.pdf')}
    previous=json.loads((ROOT/'reports/mot_provenance.json').read_text())
    upstream={k:previous[k] for k in ['TrackTrack','PuTR','weights','isolated_dependencies']}
    for name in ['TrackTrack','PuTR']:
        license_file=ROOT/'external/mot'/name/'LICENSE'
        upstream[name]['license']={'id':'MIT','sha256':file_sha256(license_file)}
    weights={}
    for name,p in [('A_engine',ROOT/'artifacts/yolo26m_local.engine'),
                   ('B_onnx',Path('X:/MODELS/yolo26m-p2-1440/best.onnx')),
                   ('B_local_fp32_engine',ROOT/'artifacts/stage2/yolo26m_p2_local_fp32.engine')]:
        if p.exists():weights[name]={'path':str(p),'sha256':file_sha256(p),'bytes':p.stat().st_size}
    report={'source_papers_local_digests':papers,'methods':methods,'upstream_MOT':upstream,
            'detectors':weights,'detector_B_rebuild':'Provided TensorRT serialization 244 vs runtime 245 incompatible; local FP32 engine built from exact user ONNX in 127.27 seconds. No retraining or model substitution.',
            'licenses':{'NASA':'NOSA 1.3 reviewed from pinned LICENSES/DAIDALUS-NOSA.pdf; no upstream code vendored',
                        'Lai_paper':'CC BY','Bauer_2019_paper':'CC BY','Bauer_2016_manuscript':'IEEE author version; independent equations, no code copying',
                        'YOLO_export_metadata':'AGPL-3.0; model binaries excluded from Git',
                        'ColANet':'No top-level license or inference weights found in pinned recursive tree; reference only'},
            'no_AOT_data_used':True,'no_simulator_run':True,'no_environment_package_changes':True,
            'implementation_sha256':file_sha256(ROOT/'airavoid/assessment.py')}
    (ROOT/'reports/stage2_method_provenance.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':write()
