# Stage 2 analytical falsification, 2026-10-22

1,599 closed-form projected cases: 123 constant-velocity trajectories × 13 observation variants. These are analytical fixtures, not recorded flight evidence or a 3D flight simulator. Physical CPA is independently minimized from the supplied Cartesian trajectory. A near encounter means future TCPA inside 60 s and Euclidean CPA <1.5 m; R=1 m only in fixture truth. Every vision method sees the same boxes, timestamps and explicitly analytical camera intrinsics, without the physical R or state.

| Method | Potential conflict | Apparent nonconflict | Uncertain | Unobserved | Raw false-safe | After camera scope |
|---|---:|---:|---:|---:|---:|---:|
| bauer_disc | 377 | 698 | 401 | 123 | 231 | 118 |
| lai_lin | 556 | 749 | 171 | 123 | 430 | 346 |
| airtrack_cues | 0 | 0 | 1476 | 123 | 0 | 0 |
| wcv_taumod | 748 | 605 | 123 | 123 | 0 | 0 |
| rv_c3bf | 293 | 814 | 369 | 123 | 337 | 337 |
| log | 352 | 510 | 614 | 123 | 153 | 52 |
| projective | 366 | 572 | 538 | 123 | 173 | 69 |
| bounded | 48 | 68 | 1360 | 123 | 7 | 6 |
| calibrated | 181 | 257 | 1038 | 123 | 28 | 0 |
| growth | 398 | 0 | 1078 | 123 | 0 | 0 |

The columns are diagnostic counts, not an equal-information accuracy ranking: WCV receives verified oracle metric state; RV-C3BF here is only a nominal horizontal point-observer cone using analytical calibrated patch flow; AirTrack supplies a cue without an invented classifier. Missing or clipped measurements remain unobserved/uncertain even in the state comparator tier. The shared image-method thresholds (2/5 diameter units and 60 s) are research thresholds, not transferred flight decision thresholds.

Bauer includes the published approximate virtual vertical disc (JIRS Eqs. 9–14), reciprocal-size average and full 3D line minimum. The independent disc/ellipse variants each contain 64 physical near encounters; Bauer has zero false-safe there. Its 123-case disc/ellipse absolute CPA errors have median 0.01000 diameter and maximum 0.09243 diameter; TCPA median 0.01267 s and maximum 0.06854 s. The reported finite residual is retained; the virtual disc is an approximation. Horizontal-only diagnostics remain available alongside full 3D results.

Bauer still fails under parallel-segment shape mismatch, altered scale/aspect, detector noise and unresolved moving-camera perturbations. The point-observer cone does not represent a finite near-miss volume or include the paper’s robust observer/certified uncertainty bound. A zero-failure state-oracle or ideal-shape result does not establish safety on uncalibrated moving-camera RGB. No assumptions were removed from the raw failure accounting.

All cases, estimates and raw/scoped decisions: `artifacts/stage2/falsification/cases.jsonl`. All raw false-safe cases and assumption violations: `artifacts/stage2/falsification/false_safe_cases.jsonl`. Aggregate machine-readable counts and implementation/script SHA-256 digests: `reports/stage2_falsification.json`. The earlier 1,368-case horizontal-only run remains in `reports/stage2_falsification_initial.json`; the new suite adds an exact vertical-ellipse projection and vertical trajectories, enlarges the analytical canvas and fixes moving-camera metadata.

Reproduce: `.venv/Scripts/python.exe scripts/stage2_falsify.py`.
