# Aviation indicator UI

The opt-in showcase uses a **3840×1080 aviation-style indicator** with fewer words, per-object identifiers/classes, automatically sized crops for every displayed object, and grouped image-attention regions. The original scientific assessment, tracker, detector variants and cue rules remain separate from this presentation change.

Core implementation: `97ce7a4236c5dc7ea1edb07fe1e779f28f184f58`.

Actual local videos: [NPS primary](C:/Users/kisel/PycharmProjects/AirAntiCollision/demo/aviation_nps/annotated.mp4), [NPS contrast](C:/Users/kisel/PycharmProjects/AirAntiCollision/demo/aviation_contrast/annotated.mp4), [QUT approach](C:/Users/kisel/PycharmProjects/AirAntiCollision/demo/aviation_approach/annotated.mp4). The [local gallery](../demo/AVIATION_UI.html) groups completed videos and real screenshots. Video binaries remain local; the commands below recreate them.

## Display meaning

- Object markers, classes, IDs and solid trails come from the recorded online detector/tracker outputs. A solid trail contains fresh measured detector centres, not predicted positions. Dashed motion is the existing local 0.5 s image projection.
- Each ROI is a crop of the original RGB around an observed object, resized to fit the available layout. A bigger displayed crop does not add native source detail. The layout must account for all objects rather than silently dropping the remaining objects when a fixed number of cards is filled.
- Grouped regions summarize **image-space attention** around related observed targets. Group membership, hulls and attention/convergence styling do not establish physical collision volumes, metric proximity, or certified danger.
- Empty or unhighlighted image space is **unobserved/unspecified**, not proved safe. A missing target, weak measurement or unresolved direction does not become an all-clear indicator.
- Direction remains the existing conditional image-separation illustration, in camera-image axes. It is not a verified aircraft maneuver or body-axis command.

The left viewport keeps the original RGB aspect ratio. The middle indicator is a normalized **image field**, without invented attitude, heading, altitude or a world-map scale. The right viewport packs every valid object ID/class into a dynamically sized ROI grid: one crop can expand, two use stacked wide cards, and denser scenes use multiple columns/rows. The native bbox pixel extent remains visible so enlargement is not confused with additional source detail.

| Indicator | Meaning |
|---|---|
| Cyan diamond / `КОНТРОЛЬ` | Existing `MONITORING` image cue; no physical all-clear claim |
| Amber triangle / `ВОЗМ. СБЛИЖЕНИЕ` | Existing `POSSIBLE CONVERGENCE` cue, not verified collision truth |
| Muted cross / `НЕТ ДАННЫХ` | Insufficient cue support or stale observation |
| White focus outline | Current display focus, not a calibrated danger ranking |
| ROI growth glyph and `%/с` | Rounded measured bbox log-span trend, not metric closing speed |
| Group `×N` badge | Number of eligible observed members in that attention group |

In a multi-object group, nonfocus field labels are replaced by its compact count badge; every object symbol and every ROI remains. Nearby unsupported objects retain their own muted representation and are not counted as reliable group members. A stale object cannot keep a fresh RGB crop or an illustrative arrow.

Grouping uses current reliable `MONITORING` / `POSSIBLE CONVERGENCE` measurements with finite local velocity and at least six-pixel span. Each bbox is padded by `max(8 px, 20% of span)` and swept for 0.5 s along its measured image velocity. Overlapping swept envelopes form connected groups; their convex visual hull can bridge pixels without a target observation. The number of groups emerges from overlap, without a chosen clustering K. Stale, clipped, ambiguous or unsupported targets cannot create an attention envelope. These are display conventions, not uncertainty or physical-occupancy bounds.

For the user-authorized NPS range preset, approximate distances remain explicitly estimates. Its inputs are the user-confirmed 1.5 m target span and GoPro HERO3 Ultra Wide family, with the declared approximate 120° horizontal fisheye mapping. Aspect, measured bbox extent and lens mismatch can bias the result. Range does not enter grouping, tracking, collision assessment or the separation direction. QUT does not inherit this NPS preset.

## Recreate the opt-in layout

The new layout is selected only inside showcase mode. `--demo-layout aviation` is the showcase default; `--demo-layout classic` retains the earlier 1920×1080 showcase. Running without `--demo-showcase` keeps the normal scientific/default presentation path.

```powershell
.venv/Scripts/python.exe -m airavoid "X:/DATASETS/UAV/TRACKING/RGB/NPS-Drones/Videos/Clip_32.mov" --demo-showcase --demo-layout aviation --nps-range-demo --detector-variant B --camera auto --encoder h264_nvenc --async-render --output demo/aviation_nps
.venv/Scripts/python.exe -m airavoid "X:/DATASETS/UAV/TRACKING/RGB/NPS-Drones/Videos/Clip_43.mov" --demo-showcase --demo-layout aviation --nps-range-demo --detector-variant B --camera auto --encoder h264_nvenc --async-render --output demo/aviation_contrast
.venv/Scripts/python.exe -m airavoid data/qut/M2.mp4 --demo-showcase --demo-layout aviation --detector-variant A --camera fixed --encoder h264_nvenc --async-render --output demo/aviation_approach
```

These commands open original RGB and use live detector inference. No detection cache, GT box or GT identity argument is supplied. The first source is a moving-camera NPS recording; the second is the existing stationary-camera QUT approach-category example, which is not physical collision ground truth.

## Fast visual iteration from existing evidence

The preview tool reads an original source frame and the exact corresponding completed online decision record, including its cue and optional range fields. If historical records lack attention groups, it derives only the new presentation groups in a cloned payload. It does not rerun detection, association, assessment or ranging and does not load GT.

```powershell
.venv/Scripts/python.exe scripts/aviation_ui_preview.py --suite
.venv/Scripts/python.exe scripts/aviation_ui_preview_validate.py
.venv/Scripts/python.exe scripts/aviation_ui_preview.py --run demo/nps_range_showcase --frame 307 --output artifacts/aviation_ui/previews/nps_primary.png
.venv/Scripts/python.exe scripts/aviation_ui_preview.py --run demo/nps_range_contrast --frame 522 --output artifacts/aviation_ui/previews/nps_contrast.png
.venv/Scripts/python.exe scripts/aviation_ui_preview.py --run demo/final_showcase_approach --frame 817 --output artifacts/aviation_ui/previews/qut_approach.png
```

The selected real frames show two NPS targets at 10.586 s, three NPS targets at 18.069 s, and the actual QUT aircraft with the previously recorded convergence cue at 32.68 s. These are real recorded states, not staged warning events. Sidecar `.preview.json` files and `preview_manifest.json` preserve original RGB/trace hashes, source time, observed IDs, derived groups and renderer hashes. A preview's single-frame CPU rendering time is not a live neural FPS measurement.

The suite also produces **8/16/32-object synthetic layout fixtures**. Their flat schematic source and burned-in `SYNTHETIC LAYOUT FIXTURE / NOT RECORDED FLIGHT` banner are deliberate. Artificial boxes, classes and states exist only to exercise dense ROI/group layout; they are not detector output, recorded encounters, additional dataset evidence or collision-accuracy tests. They stay under ignored `artifacts/aviation_ui/previews/` and must not replace real delivery screenshots.

Cached visual QA on **24 October 2026** produced these actual packing checks:

| Preview kind | Input IDs / displayed ROIs | Attention groups | ROI overlap / off-canvas cells |
|---|---:|---:|---:|
| Recorded NPS primary | 2 / 2 | 2 | 0 / 0 |
| Recorded NPS contrast | 3 / 3 | 3 | 0 / 0 |
| Recorded QUT approach | 1 / 1 | 1 | 0 / 0 |
| Synthetic layout fixture | 8 / 8 | 2 | 0 / 0 |
| Synthetic layout fixture | 16 / 16 | 4 | 0 / 0 |
| Synthetic layout fixture | 32 / 32 | 8 | 0 / 0 |

All six PNGs are exactly 3840×1080. The recorded examples and 32-object maximum-density fixture were visually inspected: the original RGB retains its aspect ratio, all expected IDs/classes have ROI cells, source crops remain actual RGB, and the field is explicitly in image coordinates. `reports/aviation_ui_preview_validation.json` records the hashes and packing checks. It rejects stale previews after a renderer/grouping change, lost/duplicated IDs, overlapping or out-of-canvas ROIs, and a missing synthetic-fixture banner. This remains a cached presentation check, not evidence of a newly completed live model run.

## Actual export validation

After completed live renders, a new wrapper delegates decoding/provenance checks to the unchanged earlier validator, then requires the exact aviation canvas and checks every recorded region payload:

```powershell
.venv/Scripts/python.exe scripts/aviation_ui_live_validate.py --run demo/aviation_nps --output reports/aviation_nps_validation.json
.venv/Scripts/python.exe scripts/aviation_ui_live_validate.py --run demo/aviation_contrast --output reports/aviation_contrast_validation.json
.venv/Scripts/python.exe scripts/aviation_ui_live_validate.py --run demo/aviation_approach --detector-variant A --output reports/aviation_approach_validation.json
.venv/Scripts/python.exe scripts/aviation_ui_live_layout_check.py
```

The report must record **3840×1080**, complete original-frame processing, real declared model providers and hashes, decoded H.264 frames and source/output PTS agreement. Source-video duration and playback timing remain unchanged. Real screenshots must be extracted from the newly encoded output, rather than presented as completed live evidence merely because the cached preview looks correct.

All three new **live neural exports passed on 24 October 2026**. Every original and encoded frame was decoded/checked; actual output is 3840×1080 H.264 with original PTS and duration preserved.

| Completed recording | Frames | Encoded duration | Max encoded PTS error | Maximum displayed / fresh IDs |
|---|---:|---:|---:|---:|
| NPS primary, detector B | 1,023 | 35.2758556 s | 5.364 μs | 3 / 3 |
| NPS contrast, detector B | 803 | 27.7950778 s | 5.539 μs | 3 / 3 |
| QUT approach, detector A | 875 | 35.0000000 s | 0 s | 9 / 7 |

Every one of the **2,701 recorded region payloads** exactly matches recomputation by the frozen presentation helper, and all member IDs refer to recorded objects. The three reports are `reports/aviation_nps_validation.json`, `reports/aviation_contrast_validation.json` and `reports/aviation_approach_validation.json`.

`reports/aviation_live_layout_validation.json` additionally checks ROI-grid cardinality, bounds and nonoverlap for **every object-count variation actually present**: 0–3 objects in each NPS run and 0–9 in QUT. These are valid predicted display IDs, not independently established true-aircraft counts. QUT can retain stale hypotheses: such cards explicitly show no fresh RGB instead of cropping current background at an old position. This grid-math check is not a claim that every encoded ROI pixel was individually inspected.

The actual encoded examples were visually reviewed, and the following unmodified screenshots are stored at 3840×1080:

- NPS primary: [two stacked ROIs](../demo/aviation_nps/stable_tracking.png), 10.586 s; [expanded single ROI](../demo/aviation_nps/encounter_cue.png), 35.138 s.
- NPS contrast: [three-object layout](../demo/aviation_contrast/stable_tracking.png), 18.069 s; [two-object layout](../demo/aviation_contrast/encounter_cue.png), 26.237 s.
- QUT: [tracking before confirmation](../demo/aviation_approach/stable_tracking.png), 29.88 s, including explicit stale placeholders; [actual recorded amber cue](../demo/aviation_approach/encounter_cue.png), 32.68 s, real aircraft ID 115. The QUT source is the existing stationary-camera approach-category recording, not physical collision truth. NPS shows monitoring illustrations without an added convergence warning.

All **32 aviation UI/group/CLI tests**, **25 classic demo tests** and **13 existing runtime tests** passed (`reports/aviation_ui_tests.log`, `reports/aviation_classic_tests.log`, `reports/aviation_runtime_tests.log`). The 150-frame normal/default decision trace remains byte-identical to its baseline, SHA-256 `976dd726a8d387366dfdacc1facb2d58709d1278e9891124eb5f8c0769310978`. Full-run UI regression also matches all inference and cue fields after removing only the newly added `advisory.demo.attention_regions` field; see `reports/aviation_ui_regression_nps.json`.

No new collision accuracy, physical-range accuracy or model-speed comparison is established by this UI change.
