# Stage 2 QUT category-proxy results

Frozen original Detector A observations; fresh causal association at each cadence. Six complete available recordings: M1, M2, M6, M7, S1, S2.

These recordings influenced prior development. Approximate target-centre annotations are joined offline with the historical 20 px / .25 s freshness contract. Approach is not verified collision; other is not verified safety. This subset does not reproduce the previous complete 15-recording score.

KLT camera diagnostics are frozen from the original full-rate fixed-camera run. Stride changes association and assessment only. All methods receive identical fresh measured boxes/IDs/timestamps within each condition. No category or GT identity enters normal tracking or assessment.

| Stride | Method | Approach warning | Approach nonconflict | Approach uncertain | Approach untracked | Other warning | Other nonconflict | Other uncertain | Other untracked |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | bauer_disc | 0 | 0 | 5571 | 2266 | 0 | 0 | 5484 | 1449 |
| 1 | lai_lin | 1544 | 670 | 3357 | 2266 | 1 | 1406 | 4077 | 1449 |
| 1 | airtrack_cues | 0 | 0 | 5571 | 2266 | 0 | 0 | 5484 | 1449 |
| 1 | wcv_taumod | 0 | 0 | 5571 | 2266 | 0 | 0 | 5484 | 1449 |
| 1 | rv_c3bf | 0 | 0 | 5571 | 2266 | 0 | 0 | 5484 | 1449 |
| 1 | log | 1664 | 0 | 3907 | 2266 | 42 | 812 | 4630 | 1449 |
| 1 | projective | 1818 | 0 | 3753 | 2266 | 44 | 767 | 4673 | 1449 |
| 1 | bounded | 429 | 0 | 5142 | 2266 | 0 | 89 | 5395 | 1449 |
| 1 | calibrated | 0 | 0 | 5571 | 2266 | 0 | 0 | 5484 | 1449 |
| 1 | growth | 2437 | 0 | 3134 | 2266 | 86 | 0 | 5398 | 1449 |
| 2 | bauer_disc | 0 | 0 | 2751 | 1168 | 0 | 0 | 2663 | 803 |
| 2 | lai_lin | 876 | 457 | 1418 | 1168 | 0 | 1163 | 1500 | 803 |
| 2 | airtrack_cues | 0 | 0 | 2751 | 1168 | 0 | 0 | 2663 | 803 |
| 2 | wcv_taumod | 0 | 0 | 2751 | 1168 | 0 | 0 | 2663 | 803 |
| 2 | rv_c3bf | 0 | 0 | 2751 | 1168 | 0 | 0 | 2663 | 803 |
| 2 | log | 814 | 0 | 1937 | 1168 | 12 | 321 | 2330 | 803 |
| 2 | projective | 842 | 0 | 1909 | 1168 | 15 | 346 | 2302 | 803 |
| 2 | bounded | 260 | 0 | 2491 | 1168 | 0 | 45 | 2618 | 803 |
| 2 | calibrated | 0 | 0 | 2751 | 1168 | 0 | 0 | 2663 | 803 |
| 2 | growth | 1284 | 0 | 1467 | 1168 | 45 | 0 | 2618 | 803 |
| 5 | bauer_disc | 0 | 0 | 1077 | 491 | 0 | 0 | 1030 | 357 |
| 5 | lai_lin | 435 | 291 | 351 | 491 | 0 | 668 | 362 | 357 |
| 5 | airtrack_cues | 0 | 0 | 1077 | 491 | 0 | 0 | 1030 | 357 |
| 5 | wcv_taumod | 0 | 0 | 1077 | 491 | 0 | 0 | 1030 | 357 |
| 5 | rv_c3bf | 0 | 0 | 1077 | 491 | 0 | 0 | 1030 | 357 |
| 5 | log | 304 | 0 | 773 | 491 | 0 | 109 | 921 | 357 |
| 5 | projective | 306 | 0 | 771 | 491 | 0 | 110 | 920 | 357 |
| 5 | bounded | 108 | 0 | 969 | 491 | 0 | 13 | 1017 | 357 |
| 5 | calibrated | 0 | 0 | 1077 | 491 | 0 | 0 | 1030 | 357 |
| 5 | growth | 503 | 0 | 574 | 491 | 23 | 0 | 1007 | 357 |

For Lai–Lin at full rate, 670 approach-category target samples receive apparent nonconflict despite 1,544 warnings. This falsifies interpreting a low image-risk index as clearance on this proxy; it does not measure a physical collision false-safe rate. The live adapter withholds that clearance interpretation; the research table retains the raw reproduction.

Bauer requires verified calibration; NASA requires metric state; RV requires calibrated, independently de-rotated target feature flow. Their abstentions here are missing-prerequisite outcomes. AirTrack has no invented collision classifier. Zero warnings or clearances is not successful discrimination.

Per-sequence counts, raw candidates, warning episodes per camera hour and warning timing relative to first/last available annotation are retained in [stage2_qut_comparison.json](stage2_qut_comparison.json). Timing is never referenced to inferred impact. Source-video, historical-cache and resulting observation/assessment hashes are included per run.

Reproduce the research outputs with `scripts/stage2_qut.py`; regenerate this table with `scripts/stage2_write_qut_results.py`.

Source report SHA256: `e0a49c5fac8ac0df90feece02a6e61d6e8a7de7e2ca4c2f38ee1d4f4912227cc`.
