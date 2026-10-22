# Stage 2 geometry replication and red-team findings — 2026-10-22

The Bauer implementation now includes the full **model-based 3D** extension. It does not infer metric range, arbitrary target shape, or ownship attitude. The original horizontal-only run remains in `reports/stage2_falsification_initial.json`; current results are in `reports/stage2_falsification.json` and the generated readable `reports/stage2_falsification.md`.

## Equations and implementation

The [JIRS author manuscript](https://eprints.sztaki.hu/9817/1/Bauer_261_3349048_ny.pdf), pp. 10–11, was visually checked against its PDF. Eqs. 9–10 transfer the measured horizontal disc half-angle to a virtual vertical disc; Eq. 11 corrects vertical image extent and midpoint; Eqs. 12–14 use the average reciprocal horizontal/virtual-vertical size and normalized vertical position. The resulting samples are

`[X/R,Y/R,Z/R] = [2*xbar/Sbar_x, 2*ybar/Sbar_y, f*(1/Sbar_x+1/Sbar_y)]`.

Their least-squares 3D line is minimized directly. This is algebraically equivalent to the special-point interpolation in JIRS Eqs. 26–29 / the 2019 flight paper's construction, while also handling constant vertical separation without dividing by zero vertical velocity. The previous horizontal line, its CPA and TCPA remain separate diagnostics.

The vertical transfer is a **published approximation**, retaining the symmetric ellipse/disc assumption and depth much larger than the target radius. The vertical view angle uses `atan(measured_bbox_midpoint_y/f)`; no inaccessible true target centre is supplied. A horizontal diameter/height ratio is not needed for normalized 3D CPA; conversion of its vertical component into height units would require a separately justified ratio and is not performed. Common 2/5 diameter and 60-second research thresholds are explicitly different from reproducing the authors' flight decision thresholds.

Independent tests project horizontal circle and vertical ellipse tangencies directly, then check the physical vector CPA, a changing vertical velocity, constant vertical separation, and a singular virtual projection. Fourteen focused assessment tests pass; see `reports/stage2_assessment_tests.log`.

## Analytical results and their limits

The current sweep contains **1,599 cases**: 123 physical constant-velocity trajectories × 13 observation variants. Added vertical trajectories and an exact independent vertical-ellipse projection complement the existing head-on, crossing, recession, lateral, wide-bearing, cadence, aspect/noise, clipping/missing and camera perturbations. The analytical canvas was enlarged vertically so off-centre targets are not silently outside the image. These are projected mathematical fixtures, not recorded-flight evidence or a simulator.

| Method | Raw critical false-safe | After moving-camera scope |
|---|---:|---:|
| Bauer full 3D disc model | 231 | 118 |
| Lai–Lin rectangle-area adaptation | 430 | 346 |
| AirTrack cue, no invented classifier | 0 | 0 |
| NASA WCV with oracle metric state | 0 | 0 |
| Nominal horizontal point-observer RV cone | 337 | 337 |
| Custom log | 153 | 52 |
| Custom projective | 173 | 69 |
| Custom bounded | 7 | 6 |
| Custom calibrated-span model | 28 | 0 |
| Growth-only warning baseline | 0 | 0 |

These are diagnostic counts, **not an equal-information accuracy ranking**. WCV receives privileged oracle state; RV receives analytical calibrated patch flow but does not implement the complete robust observer, FIFO history, certified uncertainty or finite ownship footprint. AirTrack and growth never issue nonconflict, so their zero false-safe count is not evidence of a complete threat solution. Missing/clipped observations still withhold decisions. Raw failures are retained in `artifacts/stage2/falsification/false_safe_cases.jsonl` even when camera scope subsequently abstains.

Bauer has **zero critical false-safe in either ideal-disc variant**, each containing 64 physical near encounters. On the 123 exact disc/ellipse cases its absolute CPA error is median **0.01000 diameter**, maximum **0.09243 diameter**; TCPA error is median **0.01267 s**, maximum **0.06854 s**. Finite approximation error is retained. Outside those assumptions, it remains falsifiable: for `bearing_80_z15.0_intercept6.0/parallel_ideal`, true CPA is 1.11830 m and TCPA 12.09870 s, while Bauer estimates 7.10997 diameters and 13.40327 s and incorrectly emits nonconflict. This parallel-segment target violates the disc projection assumption.

For the camera-parallax perturbation of `bearing_80_z25.0_intercept1.0`, true CPA is 0.17484 m; Bauer's raw model gives 9.11283 diameters and nonconflict. The moving-camera scope reports uncertain, retaining that raw counterexample. The initial fixture incorrectly declared these perturbations fixed-camera; corrected metadata and both raw/scoped counts make that distinction explicit.

No result establishes physical CPA observability on NPS/QUT. Their user-verified IDs and boxes do not supply calibration, target dimensions, synchronized attitude or 3D collision truth. In normal RGB traces, Bauer's calibration prerequisite, NASA's metric-state prerequisite, and the RV flow/rotation prerequisites remain explicit missing-input reasons. QUT encounter categories remain proxies.

The historical 1,368-case horizontal checkpoint was independently reconstructed from commit `e4c56133d714d7513bbca969306892a99ca69d8b` into ignored `artifacts/stage2/falsification_horizontal_reproduction`. Every metric reproduces that revision's `reports/stage2_falsification.json` exactly. Its raw cases and original source/summary snapshots have an archival manifest in `reports/stage2_falsification_initial_archive.json`; reproduce with `.venv/Scripts/python.exe scripts/stage2_archive_initial.py`. The still-earlier committed `stage2_falsification_initial.json` predates the future-WCV correction and contains 210 NASA false-safe. Its source was not separately committed, so the archive does not claim equivalence to that earlier implementation. Both committed original summaries remain unchanged in Git history.

## Read-only review of shared inference and scoring

- Normal detector/tracker modules do not load GT annotation CSVs. Their tracks and measurement indices are causal; upstream raw detector boxes are matched exactly before threat assessment. Smoothed MOT boxes are scored separately. GT access is confined to scoring and clearly labelled oracle transforms; oracle identity alone does not restore detector misses or tracker birth delays.
- Detector observations use original decoded presentation timestamps and source hashes. The audited NPS sources start at PTS zero. Camera homography/similarity ablations explicitly describe background-plane coordinates and local area-equivalent scale, without calling them recovered attitude or new independent measurements.
- Found and reported a live/offline inconsistency: a 500-sample live deque discarded Lai–Lin's first-detection area anchor. The integration owner repaired it with a separate valid measured anchor and added a 620-sample regression. Invalid clipped/ambiguous samples cannot become the next area anchor.
- Found and reported that `-1-IoU` assignment cost was not strictly cardinality-first for arbitrary larger bipartite match sets, despite the scorer's stated contract. The scoring owner repaired it with valid-edge bonus `min(n_gt,n_pred)+1`, which dominates all possible IoU-sum changes, and added an adversarial 5×5 assignment regression.
- Requested strict frame/time monotonicity, per-observation acquisition-time validation and honoring the explicit measured flag in imported replay traces. The scoring owner added these guards, unique measured-ID validation, rejection of unknown measured sources, prediction exclusion and an input-hash recheck after replay. Eight scoring/oracle tests pass, including noncausal/predicted-input rejection. Existing assessment outputs were discarded and the campaign restarted under the frozen corrected source.

No GPU, package installation, detector retraining, AOT data or flight simulator was used for this work. Exact source PDF, implementation, test and script SHA-256 digests are recorded in `reports/stage2_method_provenance.json`.
