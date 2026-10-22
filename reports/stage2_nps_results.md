# Stage 2 — controlled NPS comparison (2026-10-22)

This is a focused, frozen six-clip comparison with recorded RGB, verified within-clip IDs and DogFight 2D boxes. It supplies no physical collision or CPA truth. Development: 001/008/032; final evaluation: 020/034/043. Both detectors have user-confirmed NPS training independence; no independent training manifest audit was available.

Each detector processed exactly 6,644 source frames with original MOV presentation timestamps. Detector A retains its original five-class 416×704 TensorRT model and tiled preprocessing; only class 0 (airplane) is included in the matched fixed-wing condition. Detector B is the supplied fixed-wing-only P2 export, actual input 800×1440, one call per frame. The user confirmed on 2026-10-22 that all targets in these six selected clips are fixed-wing. Native CSVs contain no per-object subtype field, so this confirmation is scoped to the selected clips and is not extrapolated to the whole NPS corpus. Both thresholds were frozen at .03; confidence values are not cross-model calibrated. A five-class production default remains unchanged.

## Detection on the reserved evaluation clips

| Detector | IoU | Matched / GT | Predictions | Recall | Precision against supplied labels |
|---|---:|---:|---:|---:|---:|
| A | 0.25 | 3365 / 3534 | 17285 | 95.22% | 19.47% |
| A | 0.5 | 3167 / 3534 | 17285 | 89.62% | 18.32% |
| A | 0.75 | 971 / 3534 | 17285 | 27.48% | 5.62% |
| B | 0.25 | 2991 / 3534 | 3418 | 84.63% | 87.51% |
| B | 0.5 | 2928 / 3534 | 3418 | 82.85% | 85.66% |
| B | 0.75 | 754 / 3534 | 3418 | 21.34% | 22.06% |

Precision is conditional on supplied visible-target annotation completeness; unmatched predictions are not individually proven false positives. At IoU .5, B has lower aggregate recall against the supplied fixed-wing labels and far fewer unmatched predictions. The largest recall difference is Clip_020 (A 91.55%, B 64.27%); B is not a universally better detector. Per-video localization and log-span errors, pixel-size/confidence strata, and A all-class diagnostics are retained in `stage2_detection_mot_s1.json`.

## Identity and observed-box continuity

| Detector / tracker | IDF1 | HOTA | AssA | CLEAR IDSW | CLEAR Frag | Fresh-box coverage | Dominant-ID coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| A/custom/s1 | 25.90% | 24.50% | 46.35% | 77 | 130 | 89.62% | 75.66% |
| A/tracktrack/s1 | 37.00% | 30.66% | 49.98% | 33 | 124 | 84.72% | 75.98% |
| A/putr/s1 | 19.38% | 15.42% | 12.87% | 124 | 111 | 36.39% | 17.37% |
| B/custom/s1 | 82.59% | 54.72% | 59.75% | 5 | 110 | 82.85% | 81.81% |
| B/tracktrack/s1 | 77.60% | 50.31% | 51.74% | 14 | 82 | 78.98% | 74.62% |
| B/putr/s1 | 34.60% | 25.71% | 21.19% | 106 | 68 | 47.28% | 26.23% |

Metrics use the pinned original TrackEval HOTA/Identity/CLEAR implementations, continuous-box IoU, explicit 0-based source to 1-based GT conversion, unchanged metric equations, and online outputs only. MOT boxes are the upstream emitted boxes; the last two columns independently match fresh, actual detector-measured boxes. Low switch/fragment counts alone can reward missing tracks, so coverage accompanies them. GT IDs never entered normal association. Learned components use the existing pinned DanceTrack checkpoints/adapters, without aircraft-specific or NPS retraining.

On this reserved set, TrackTrack improves A scored IDF1/HOTA while fresh observation coverage falls; dominant-ID coverage is nearly unchanged. Its metric gain therefore does not mean more labelled aircraft are observed continuously. With B, the original tracker has higher IDF1/HOTA and fewer ID switches than TrackTrack or PuTR. PuTR loses substantial observation coverage in both conditions. These observations do not establish a universal tracker ranking.

## Shared threat-method observations

The complete matrix contains A/B × original/TrackTrack/PuTR × raw/KLT similarity, each across all six clips, at full source cadence. All ten methods receive the exact same trace within each condition, source acquisition times and fresh measured extents. The same .24 s warning and 1.2 s apparent-nonconflict temporal policy is applied. Camera compensation is a background-image-plane hypothesis; it cannot resolve rotation, ego translation or target shape.

| Condition (evaluation) | Method | Warning / target opportunities | Scoped apparent nonconflict | Untracked | Raw apparent nonconflict (pre-policy) | All predicted warning episodes / camera hour |
|---|---|---:|---:|---:|---:|---:|
| A/custom/raw | lai_lin | 99 / 3534 | 0 | 367 | 763 | 1182.24 |
| A/custom/raw | log | 0 / 3534 | 0 | 367 | 101 | 0.00 |
| A/custom/raw | projective | 0 / 3534 | 0 | 367 | 85 | 0.00 |
| A/custom/raw | bounded | 0 / 3534 | 0 | 367 | 0 | 0.00 |
| A/custom/raw | growth | 0 / 3534 | 0 | 367 | 0 | 0.00 |
| A/custom/similarity | lai_lin | 7 / 3534 | 0 | 367 | 805 | 697.22 |
| A/custom/similarity | log | 0 / 3534 | 0 | 367 | 63 | 0.00 |
| A/custom/similarity | projective | 0 / 3534 | 0 | 367 | 65 | 0.00 |
| A/custom/similarity | bounded | 0 / 3534 | 0 | 367 | 0 | 0.00 |
| A/custom/similarity | growth | 0 / 3534 | 0 | 367 | 0 | 30.31 |
| A/putr/raw | lai_lin | 0 / 3534 | 0 | 2248 | 1138 | 60.63 |
| A/putr/raw | log | 0 / 3534 | 0 | 2248 | 19 | 0.00 |
| A/putr/raw | projective | 0 / 3534 | 0 | 2248 | 18 | 0.00 |
| A/putr/raw | bounded | 0 / 3534 | 0 | 2248 | 0 | 0.00 |
| A/putr/raw | growth | 0 / 3534 | 0 | 2248 | 0 | 0.00 |
| A/putr/similarity | lai_lin | 0 / 3534 | 0 | 2248 | 1135 | 0.00 |
| A/putr/similarity | log | 0 / 3534 | 0 | 2248 | 21 | 0.00 |
| A/putr/similarity | projective | 0 / 3534 | 0 | 2248 | 16 | 0.00 |
| A/putr/similarity | bounded | 0 / 3534 | 0 | 2248 | 0 | 0.00 |
| A/putr/similarity | growth | 0 / 3534 | 0 | 2248 | 0 | 0.00 |
| A/tracktrack/raw | lai_lin | 59 / 3534 | 0 | 540 | 2700 | 454.71 |
| A/tracktrack/raw | log | 0 / 3534 | 0 | 540 | 596 | 0.00 |
| A/tracktrack/raw | projective | 0 / 3534 | 0 | 540 | 524 | 0.00 |
| A/tracktrack/raw | bounded | 0 / 3534 | 0 | 540 | 0 | 0.00 |
| A/tracktrack/raw | growth | 0 / 3534 | 0 | 540 | 0 | 0.00 |
| A/tracktrack/similarity | lai_lin | 42 / 3534 | 0 | 540 | 2779 | 272.82 |
| A/tracktrack/similarity | log | 0 / 3534 | 0 | 540 | 542 | 0.00 |
| A/tracktrack/similarity | projective | 0 / 3534 | 0 | 540 | 318 | 0.00 |
| A/tracktrack/similarity | bounded | 0 / 3534 | 0 | 540 | 0 | 0.00 |
| A/tracktrack/similarity | growth | 0 / 3534 | 0 | 540 | 0 | 30.31 |
| B/custom/raw | lai_lin | 0 / 3534 | 0 | 606 | 2774 | 0.00 |
| B/custom/raw | log | 0 / 3534 | 0 | 606 | 377 | 0.00 |
| B/custom/raw | projective | 0 / 3534 | 0 | 606 | 342 | 0.00 |
| B/custom/raw | bounded | 0 / 3534 | 0 | 606 | 0 | 0.00 |
| B/custom/raw | growth | 0 / 3534 | 0 | 606 | 0 | 0.00 |
| B/custom/similarity | lai_lin | 0 / 3534 | 0 | 606 | 2759 | 0.00 |
| B/custom/similarity | log | 0 / 3534 | 0 | 606 | 363 | 0.00 |
| B/custom/similarity | projective | 0 / 3534 | 0 | 606 | 106 | 0.00 |
| B/custom/similarity | bounded | 0 / 3534 | 0 | 606 | 38 | 0.00 |
| B/custom/similarity | growth | 0 / 3534 | 0 | 606 | 0 | 0.00 |
| B/putr/raw | lai_lin | 0 / 3534 | 0 | 1863 | 1540 | 0.00 |
| B/putr/raw | log | 0 / 3534 | 0 | 1863 | 20 | 0.00 |
| B/putr/raw | projective | 0 / 3534 | 0 | 1863 | 0 | 0.00 |
| B/putr/raw | bounded | 0 / 3534 | 0 | 1863 | 0 | 0.00 |
| B/putr/raw | growth | 0 / 3534 | 0 | 1863 | 0 | 0.00 |
| B/putr/similarity | lai_lin | 0 / 3534 | 0 | 1863 | 1531 | 0.00 |
| B/putr/similarity | log | 0 / 3534 | 0 | 1863 | 74 | 0.00 |
| B/putr/similarity | projective | 0 / 3534 | 0 | 1863 | 0 | 0.00 |
| B/putr/similarity | bounded | 0 / 3534 | 0 | 1863 | 20 | 0.00 |
| B/putr/similarity | growth | 0 / 3534 | 0 | 1863 | 0 | 0.00 |
| B/tracktrack/raw | lai_lin | 0 / 3534 | 0 | 743 | 2698 | 0.00 |
| B/tracktrack/raw | log | 0 / 3534 | 0 | 743 | 443 | 0.00 |
| B/tracktrack/raw | projective | 0 / 3534 | 0 | 743 | 325 | 0.00 |
| B/tracktrack/raw | bounded | 0 / 3534 | 0 | 743 | 0 | 0.00 |
| B/tracktrack/raw | growth | 0 / 3534 | 0 | 743 | 0 | 0.00 |
| B/tracktrack/similarity | lai_lin | 0 / 3534 | 0 | 743 | 2690 | 0.00 |
| B/tracktrack/similarity | log | 0 / 3534 | 0 | 743 | 424 | 0.00 |
| B/tracktrack/similarity | projective | 0 / 3534 | 0 | 743 | 102 | 0.00 |
| B/tracktrack/similarity | bounded | 0 / 3534 | 0 | 743 | 19 | 0.00 |
| B/tracktrack/similarity | growth | 0 / 3534 | 0 | 743 | 0 | 0.00 |

Raw candidate counts precede moving-camera scoping and temporal confirmation; final state columns follow both. Target-frame untracked means no fresh measured IoU≥.5 match against the visible label, and is never counted as low risk; a stale tracker hypothesis may still exist. Separate all-predicted-track-sample counters retain uncertain versus unobserved, with their different denominator. Warning episodes count every predicted hypothesis, including unmatched hypotheses; they are not measured false-warning rates. Per-target warning onset, duration and persistence are referenced to the first/last visible annotation, never to an inferred impact.

Bauer full 3D disc geometry and calibrated custom CPA require verified intrinsics; NASA well-clear requires verified metric relative state; the RV-C3BF observable requires calibrated, independently de-rotated target feature flow. Those signals are not present in these NPS traces and are not invented. AirTrack contributes an expansion/intercept cue without a complete published collision classifier. Missing-prerequisite abstention is reported, not counted as successful collision discrimination. Moving-camera raw apparent-nonconflict candidates are preserved and scoped to uncertain without independently verified motion. Physical collision recall, false-safe rates and warning precision are unmeasured.

## Focused oracle controls

For development Clip_032 and evaluation Clip_043, identity-only relabels each IoU-matched emitted measurement by verified GT identity, clears its association ambiguity, preserves its exact measured box/confidence, and preserves unmatched hypotheses under disjoint predicted IDs. It cannot restore missed detections or delayed births. Box+ID replaces measurements with every visible GT box and identity, removes unmatched hypotheses and uses no detector confidence. Both keep camera metadata/transforms frozen from the normal original-tracker trace; no GT camera-motion estimate is made. These are offline privileged controls, never deployable inference.

| Detector / input (Clip_043, raw camera) | Method | Warning | Uncertain | Untracked | Target opportunities |
|---|---|---:|---:|---:|---:|
| A/custom | lai_lin | 99 | 932 | 238 | 1269 |
| A/custom | log | 0 | 1031 | 238 | 1269 |
| A/custom | projective | 0 | 1031 | 238 | 1269 |
| A/custom | bounded | 0 | 1031 | 238 | 1269 |
| A/custom | growth | 0 | 1031 | 238 | 1269 |
| A/oracle_identity_only | lai_lin | 78 | 953 | 238 | 1269 |
| A/oracle_identity_only | log | 0 | 1031 | 238 | 1269 |
| A/oracle_identity_only | projective | 0 | 1031 | 238 | 1269 |
| A/oracle_identity_only | bounded | 0 | 1031 | 238 | 1269 |
| A/oracle_identity_only | growth | 0 | 1031 | 238 | 1269 |
| A/oracle_box_and_identity | lai_lin | 0 | 1269 | 0 | 1269 |
| A/oracle_box_and_identity | log | 0 | 1269 | 0 | 1269 |
| A/oracle_box_and_identity | projective | 0 | 1269 | 0 | 1269 |
| A/oracle_box_and_identity | bounded | 0 | 1269 | 0 | 1269 |
| A/oracle_box_and_identity | growth | 0 | 1269 | 0 | 1269 |
| B/custom | lai_lin | 0 | 1120 | 149 | 1269 |
| B/custom | log | 0 | 1120 | 149 | 1269 |
| B/custom | projective | 0 | 1120 | 149 | 1269 |
| B/custom | bounded | 0 | 1120 | 149 | 1269 |
| B/custom | growth | 0 | 1120 | 149 | 1269 |
| B/oracle_identity_only | lai_lin | 0 | 1120 | 149 | 1269 |
| B/oracle_identity_only | log | 0 | 1120 | 149 | 1269 |
| B/oracle_identity_only | projective | 0 | 1120 | 149 | 1269 |
| B/oracle_identity_only | bounded | 0 | 1120 | 149 | 1269 |
| B/oracle_identity_only | growth | 0 | 1120 | 149 | 1269 |
| B/oracle_box_and_identity | lai_lin | 0 | 1269 | 0 | 1269 |
| B/oracle_box_and_identity | log | 0 | 1269 | 0 | 1269 |
| B/oracle_box_and_identity | projective | 0 | 1269 | 0 | 1269 |
| B/oracle_box_and_identity | bounded | 0 | 1269 | 0 | 1269 |
| B/oracle_box_and_identity | growth | 0 | 1269 | 0 | 1269 |

All raw and similarity oracle results, measured log-scale increment error, pixel span, detector-confidence, camera-validity and normalized-pixel-offset strata are retained in `stage2_threat_matrix.json`. Pixel offset is not bearing angle without calibration. The oracle controls identify downstream sensitivity to observation/identity quality; their warnings have no collision truth labels.

## Independent background-camera control

The A/original-tracker observations, predicted IDs, measured boxes and source times were verified exactly equal before changing only the camera metadata and transform. ORB/RANSAC homography is an independent background-plane hypothesis. Its target-center coordinates use homogeneous division and its extent adaptation uses the local Jacobian; neither proves recovered camera rotation or target stabilization.

| Clip | Camera coordinates | Lai–Lin confirmed warning samples | Growth confirmed warning samples | Labelled target opportunities |
|---|---|---:|---:|---:|
| Clip_032 | raw | 0 | 0 | 1465 |
| Clip_032 | KLT_similarity | 0 | 0 | 1465 |
| Clip_032 | ORB_homography | 19 | 1 | 1465 |
| Clip_043 | raw | 99 | 0 | 1269 |
| Clip_043 | KLT_similarity | 7 | 0 | 1269 |
| Clip_043 | ORB_homography | 23 | 0 | 1269 |

Camera-model choice changes published-proxy warnings on identical target observations. Lower warning counts may reflect changed evidence or reference resets and cannot establish better collision discrimination. No physical CPA truth or independently verified rotation is added. Full records are in `stage2_camera_threat_control.json`.

## Reproduction and limits

After recreating the pinned detector/tracker caches, run:

```powershell
.venv/Scripts/python.exe scripts/stage2_score.py
.venv/Scripts/python.exe scripts/stage2_assessment_campaign.py --oracle --workers 2
.venv/Scripts/python.exe scripts/stage2_write_nps_results.py
```

Bulk assessment traces are lossless `.jsonl.gz`; use `stage2_assess_trace.text_open`. Each output retains source and implementation hashes. All normal and oracle GT joins happen in offline scoring/oracle modules. `stage2_assess_trace.py` has no GT loader. Cardinality-first Hungarian matching uses a bonus greater than the maximum possible IoU-sum advantage, with an adversarial regression test. Causality tests reject repeated/reversed timestamps and predicted extents promoted to measurements.

This report supplies cached-trace comparison, not neural end-to-end FPS. Actual detector/tracker live timings, independently sourced ORB/homography camera control, QUT cadence proxy results, analytical physical falsification, environment fingerprint and final reproducibility handoff are separate Stage 2 artifacts. No metric CPA or maneuver guarantee is inferred from NPS boxes.
