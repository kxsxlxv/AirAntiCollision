"""Generate the compact NPS report strictly from measured frozen-run JSON artifacts."""
from collections import Counter
import json
from pathlib import Path
import math
ROOT=Path(__file__).resolve().parents[1]


def pct(x):return f'{100*x:.2f}%'


def main():
    r=json.loads((ROOT/'reports/stage2_detection_mot_s1.json').read_text())
    t=json.loads((ROOT/'reports/stage2_threat_matrix.json').read_text())
    status=json.loads((ROOT/'reports/stage2_assessment_campaign_status.json').read_text())
    if status['state']!='complete' or len(status['complete'])!=status['planned_replays']:raise ValueError('Incomplete matrix')
    lines=['# Stage 2 — controlled NPS comparison (2026-10-22)','',
           'This is a focused, frozen six-clip comparison with recorded RGB, verified within-clip IDs and DogFight 2D boxes. It supplies no physical collision or CPA truth. Development: 001/008/032; final evaluation: 020/034/043. Both detectors have user-confirmed NPS training independence; no independent training manifest audit was available.','',
           'Each detector processed exactly 6,644 source frames with original MOV presentation timestamps. Detector A retains its original five-class 416×704 TensorRT model and tiled preprocessing; only class 0 (airplane) is included in the matched fixed-wing condition. Detector B is the supplied fixed-wing-only P2 export, actual input 800×1440, one call per frame. The user confirmed on 2026-10-22 that all targets in these six selected clips are fixed-wing. Native CSVs contain no per-object subtype field, so this confirmation is scoped to the selected clips and is not extrapolated to the whole NPS corpus. Both thresholds were frozen at .03; confidence values are not cross-model calibrated. A five-class production default remains unchanged.','',
           '## Detection on the reserved evaluation clips','',
           '| Detector | IoU | Matched / GT | Predictions | Recall | Precision against supplied labels |','|---|---:|---:|---:|---:|---:|']
    summary={'detectors':{},'MOT':{},'oracle_scope':'matched predicted measurements only for identity oracle; all visible GT boxes for box+ID oracle'}
    for detector,p in r['detectors'].items():
        summary['detectors'][detector]={}
        for th in ['0.25','0.5','0.75']:
            counts=Counter()
            for clip in r['split']['evaluation']:
                for k in ['tp','gt','pred','fp','fn']:counts[k]+=p[clip][th][k]
            recall=counts['tp']/counts['gt'];precision=counts['tp']/counts['pred']
            summary['detectors'][detector][th]={**counts,'recall':recall,'precision_wrt_supplied_labels':precision}
            lines.append(f"| {detector} | {th} | {counts['tp']} / {counts['gt']} | {counts['pred']} | {pct(recall)} | {pct(precision)} |")
    lines+=['','Precision is conditional on supplied visible-target annotation completeness; unmatched predictions are not individually proven false positives. At IoU .5, B has lower aggregate recall against the supplied fixed-wing labels and far fewer unmatched predictions. The largest recall difference is Clip_020 (A 91.55%, B 64.27%); B is not a universally better detector. Per-video localization and log-span errors, pixel-size/confidence strata, and A all-class diagnostics are retained in `stage2_detection_mot_s1.json`.','',
            '## Identity and observed-box continuity','',
            '| Detector / tracker | IDF1 | HOTA | AssA | CLEAR IDSW | CLEAR Frag | Fresh-box coverage | Dominant-ID coverage |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,p in r['MOT'].items():
        d=p['combined']['evaluation'];opps=matches=dominant=0
        for clip in r['split']['evaluation']:
            c=p['per_sequence'][clip]['measured_box_continuity'];n=c['target_frame_opportunities'];opps+=n;matches+=c['matched_fresh_measurements'];dominant+=c['dominant_ID_fraction']*n
        q={'IDF1':d['Identity']['IDF1'],'HOTA':d['HOTA']['HOTA'],'AssA':d['HOTA']['AssA'],'IDSW':int(d['CLEAR']['IDSW']),'Frag':int(d['CLEAR']['Frag']),
           'fresh_box_coverage':matches/opps,'dominant_ID_coverage':dominant/opps}
        summary['MOT'][name]=q
        lines.append(f"| {name} | {pct(q['IDF1'])} | {pct(q['HOTA'])} | {pct(q['AssA'])} | {q['IDSW']} | {q['Frag']} | {pct(q['fresh_box_coverage'])} | {pct(q['dominant_ID_coverage'])} |")
    lines+=['','Metrics use the pinned original TrackEval HOTA/Identity/CLEAR implementations, continuous-box IoU, explicit 0-based source to 1-based GT conversion, unchanged metric equations, and online outputs only. MOT boxes are the upstream emitted boxes; the last two columns independently match fresh, actual detector-measured boxes. Low switch/fragment counts alone can reward missing tracks, so coverage accompanies them. GT IDs never entered normal association. Learned components use the existing pinned DanceTrack checkpoints/adapters, without aircraft-specific or NPS retraining.','',
            'On this reserved set, TrackTrack improves A scored IDF1/HOTA while fresh observation coverage falls; dominant-ID coverage is nearly unchanged. Its metric gain therefore does not mean more labelled aircraft are observed continuously. With B, the original tracker has higher IDF1/HOTA and fewer ID switches than TrackTrack or PuTR. PuTR loses substantial observation coverage in both conditions. These observations do not establish a universal tracker ranking.','',
            '## Shared threat-method observations','',
            'The complete matrix contains A/B × original/TrackTrack/PuTR × raw/KLT similarity, each across all six clips, at full source cadence. All ten methods receive the exact same trace within each condition, source acquisition times and fresh measured extents. The same .24 s warning and 1.2 s apparent-nonconflict temporal policy is applied. Camera compensation is a background-image-plane hypothesis; it cannot resolve rotation, ego translation or target shape.','',
            '| Condition (evaluation) | Method | Warning / target opportunities | Scoped apparent nonconflict | Untracked | Raw apparent nonconflict (pre-policy) | All predicted warning episodes / camera hour |',
            '|---|---|---:|---:|---:|---:|---:|']
    shown=['lai_lin','log','projective','bounded','growth']
    for name,p in sorted(t['conditions'].items()):
        if 'oracle_' in name:continue
        for method in shown:
            d=p['combined']['evaluation']['methods'][method];s=d['states'];n=d['target_frame_opportunities']
            lines.append(f"| {name} | {method} | {s.get('potential_conflict',0)} / {n} | {s.get('apparent_nonconflict',0)} | {s.get('untracked',0)} | {d['raw_candidate_states'].get('apparent_nonconflict',0)} | {d['all_predicted_warning_episodes_per_camera_hour']:.2f} |")
    lines+=['','Raw candidate counts precede moving-camera scoping and temporal confirmation; final state columns follow both. Target-frame untracked means no fresh measured IoU≥.5 match against the visible label, and is never counted as low risk; a stale tracker hypothesis may still exist. Separate all-predicted-track-sample counters retain uncertain versus unobserved, with their different denominator. Warning episodes count every predicted hypothesis, including unmatched hypotheses; they are not measured false-warning rates. Per-target warning onset, duration and persistence are referenced to the first/last visible annotation, never to an inferred impact.','',
            'Bauer full 3D disc geometry and calibrated custom CPA require verified intrinsics; NASA well-clear requires verified metric relative state; the RV-C3BF observable requires calibrated, independently de-rotated target feature flow. Those signals are not present in these NPS traces and are not invented. AirTrack contributes an expansion/intercept cue without a complete published collision classifier. Missing-prerequisite abstention is reported, not counted as successful collision discrimination. Moving-camera raw apparent-nonconflict candidates are preserved and scoped to uncertain without independently verified motion. Physical collision recall, false-safe rates and warning precision are unmeasured.','',
            '## Focused oracle controls','',
            'For development Clip_032 and evaluation Clip_043, identity-only relabels each IoU-matched emitted measurement by verified GT identity, clears its association ambiguity, preserves its exact measured box/confidence, and preserves unmatched hypotheses under disjoint predicted IDs. It cannot restore missed detections or delayed births. Box+ID replaces measurements with every visible GT box and identity, removes unmatched hypotheses and uses no detector confidence. Both keep camera metadata/transforms frozen from the normal original-tracker trace; no GT camera-motion estimate is made. These are offline privileged controls, never deployable inference.','',
            '| Detector / input (Clip_043, raw camera) | Method | Warning | Uncertain | Untracked | Target opportunities |',
            '|---|---|---:|---:|---:|---:|']
    for detector in ['A','B']:
        for truth in ['custom','oracle_identity_only','oracle_box_and_identity']:
            p=t['conditions'][f'{detector}/{truth}/raw']['per_sequence']['Clip_043']
            for m in shown:
                d=p['methods'][m];s=d['states']
                lines.append(f"| {detector}/{truth} | {m} | {s.get('potential_conflict',0)} | {s.get('uncertain',0)} | {s.get('untracked',0)} | {d['target_frame_opportunities']} |")
    lines+=['','All raw and similarity oracle results, measured log-scale increment error, pixel span, detector-confidence, camera-validity and normalized-pixel-offset strata are retained in `stage2_threat_matrix.json`. Pixel offset is not bearing angle without calibration. The oracle controls identify downstream sensitivity to observation/identity quality; their warnings have no collision truth labels.','']
    camera_path=ROOT/'reports/stage2_camera_threat_control.json'
    if camera_path.exists():
        camera=json.loads(camera_path.read_text())
        lines+=['## Independent background-camera control','',
                'The A/original-tracker observations, predicted IDs, measured boxes and source times were verified exactly equal before changing only the camera metadata and transform. ORB/RANSAC homography is an independent background-plane hypothesis. Its target-center coordinates use homogeneous division and its extent adaptation uses the local Jacobian; neither proves recovered camera rotation or target stabilization.','',
                '| Clip | Camera coordinates | Lai–Lin confirmed warning samples | Growth confirmed warning samples | Labelled target opportunities |',
                '|---|---|---:|---:|---:|']
        for clip,entry in camera['clips'].items():
            for condition in ['raw','KLT_similarity','ORB_homography']:
                metrics=entry[condition]['methods'];a=metrics['lai_lin'];b=metrics['growth']
                lines.append(f"| {clip} | {condition} | {a['states'].get('potential_conflict',0)} | {b['states'].get('potential_conflict',0)} | {a['target_frame_opportunities']} |")
        lines+=['','Camera-model choice changes published-proxy warnings on identical target observations. Lower warning counts may reflect changed evidence or reference resets and cannot establish better collision discrimination. No physical CPA truth or independently verified rotation is added. Full records are in `stage2_camera_threat_control.json`.','']
    lines+=[
            '## Reproduction and limits','',
            'After recreating the pinned detector/tracker caches, run:','',
            '```powershell','.venv/Scripts/python.exe scripts/stage2_score.py',
            '.venv/Scripts/python.exe scripts/stage2_assessment_campaign.py --oracle --workers 2',
            '.venv/Scripts/python.exe scripts/stage2_write_nps_results.py','```','',
            'Bulk assessment traces are lossless `.jsonl.gz`; use `stage2_assess_trace.text_open`. Each output retains source and implementation hashes. All normal and oracle GT joins happen in offline scoring/oracle modules. `stage2_assess_trace.py` has no GT loader. Cardinality-first Hungarian matching uses a bonus greater than the maximum possible IoU-sum advantage, with an adversarial regression test. Causality tests reject repeated/reversed timestamps and predicted extents promoted to measurements.','',
            'This report supplies cached-trace comparison, not neural end-to-end FPS. Actual detector/tracker live timings, independently sourced ORB/homography camera control, QUT cadence proxy results, analytical physical falsification, environment fingerprint and final reproducibility handoff are separate Stage 2 artifacts. No metric CPA or maneuver guarantee is inferred from NPS boxes.']
    (ROOT/'reports/stage2_nps_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'reports/stage2_nps_summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
