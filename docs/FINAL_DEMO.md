# Final live showcase

## NPS approximate-range preset (user-confirmed camera and size)

The user subsequently confirmed **GoPro HERO3 Ultra Wide** and an aircraft size
of **1.5 m** for the NPS clips, and authorized a rough demonstration estimate.
The opt-in `--nps-range-demo` treats 1.5 m as transverse wingspan and uses a
deliberately approximate **120° horizontal FOV**, not a quoted diagonal angle or
verified calibration. An equidistant fisheye ray mapping accounts approximately
for position in the frame; range is `1.5 / (2*tan(angular_extent/2))`. The longer
measured bbox axis is the projected-span proxy. A causal 0.4 s median reduces
jitter. The UI rounds to 5 m above 20 m and explicitly displays **≈ / оценка**.
Foreshortening, aircraft aspect, bbox errors and lens mismatch can substantially
bias the estimate; no error bound or physical collision state is inferred.

```powershell
.venv/Scripts/python.exe -m airavoid "X:/DATASETS/UAV/TRACKING/RGB/NPS-Drones/Videos/Clip_32.mov" --demo-showcase --nps-range-demo --detector-variant B --camera auto --encoder h264_nvenc --async-render --output demo/nps_range_showcase
.venv/Scripts/python.exe -m airavoid "X:/DATASETS/UAV/TRACKING/RGB/NPS-Drones/Videos/Clip_43.mov" --demo-showcase --nps-range-demo --detector-variant B --camera auto --encoder h264_nvenc --async-render --output demo/nps_range_contrast
```

This flag explicitly declares NPS footage. It is not used for the QUT example or
enabled by default. The camera model and 1.5 m size are user inputs; 120° is a
rounded visualization assumption. Overrides are `--nps-target-span-m` and
`--nps-hfov-deg`. Range is computed only from fresh, unclipped, unambiguous
detector extents and is retained in JSON as `range_approx_m`, `range_display_m`,
and the `range_preset` contract. It never enters tracking, encounter assessment,
the separation direction, or the research calibration API. The earlier no-range
exports and their evidence below remain historical results.

Updated local exports: [NPS primary with approximate range](C:/Users/kisel/PycharmProjects/AirAntiCollision/demo/nps_range_showcase/annotated.mp4)
and [NPS contrast with approximate range](C:/Users/kisel/PycharmProjects/AirAntiCollision/demo/nps_range_contrast/annotated.mp4).

The final showcase is a whole recorded NPS moving-camera clip processed by the supplied fixed-wing YOLO26m-P2 detector, the existing predicted-ID tracker, and an opt-in image-plane encounter cue. It is an offline RGB demonstration. Its cue and arrow do not establish physical collision probability, metric CPA, or a safe aircraft maneuver.

Core implementation commit: [`5b4cbd7387a9e981a4dfaa1b0f9ee54925058056`](https://github.com/kxsxlxv/AirAntiCollision/commit/5b4cbd7387a9e981a4dfaa1b0f9ee54925058056). Review: [PR #3](https://github.com/kxsxlxv/AirAntiCollision/pull/3).

Local completed videos: [moving-camera primary](C:/Users/kisel/PycharmProjects/AirAntiCollision/demo/final_showcase/annotated.mp4), [moving-camera contrast](C:/Users/kisel/PycharmProjects/AirAntiCollision/demo/final_showcase_contrast/annotated.mp4), and [recorded approach cue](C:/Users/kisel/PycharmProjects/AirAntiCollision/demo/final_showcase_approach/annotated.mp4). The [local gallery](../demo/FINAL_DEMO.html) groups the actual exports and screenshots. MP4 links refer to existing local artifacts; footage binaries are not represented as tracked GitHub files.

## Source and camera evidence

The primary source is `Clip_32.mov`: **1280×960, 29 fps, 1,023 frames, 35.275862 s**, MPEG-4 Part 2. SHA-256: `12682055927dee85db715a36eb25bedd8e4df1ea04d65f3f56d37f3dee007365`. Local source: `X:/DATASETS/UAV/TRACKING/RGB/NPS-Drones/Videos/Clip_32.mov`. Independent FFprobe output is preserved in `reports/final_demo_source_ffprobe.json`. The alternative Clip_43 is 1280×960, 28.89 fps, 803 frames and 27.795085 s.

Purdue's [original dataset page](https://engineering.purdue.edu/~bouman/UAV_Dataset/) identifies a GoPro 3 mounted on a custom delta-wing aircraft. The page's generic description includes 1280×1060; the actual selected MOV reports 1280×960, which determines this run's raster. The final source has visible camera banking and aircraft observed against sky and terrain. The user confirmed the selected NPS targets are fixed-wing. Those identity/box annotations are used for the separate Stage 2 evaluation and are not inputs to this live demo.

The [official HERO3 Black manual](https://gopro.com/content/dam/help/hero3-black-edition/manuals/HERO3_UM_Black_ENG_REVD_WEB.pdf), printed page 22, lists 960p as 1280×960 with Ultra Wide FOV and lists multiple FOV options for 1080p. Its 960p frame-rate choices differ from this converted 29 fps file. The file contains the encoder tag `Lavf56.40.101`, without camera-model, FOV-mode, intrinsic-calibration or distortion-coefficient tags. Therefore the documentation supports the GoPro camera family, but does not verify the exact original capture mode, Black edition, focal length, lens correction, or a calibrated pixel-to-angle mapping. No numerical FOV or focal length is assigned.

## Recreate the actual live run

Use the existing custom CUDA OpenCV environment and the registered local P2 TensorRT engine. Do not install an OpenCV wheel or run an environment resynchronization.

```powershell
.venv/Scripts/python.exe -m airavoid "X:/DATASETS/UAV/TRACKING/RGB/NPS-Drones/Videos/Clip_32.mov" --demo-showcase --detector-variant B --camera auto --encoder h264_nvenc --async-render --output demo/final_showcase
.venv/Scripts/python.exe scripts/final_demo_validate.py --run demo/final_showcase
```

For a short functional pilot, use a separate output; it does not satisfy the final duration requirement:

```powershell
.venv/Scripts/python.exe -m airavoid "X:/DATASETS/UAV/TRACKING/RGB/NPS-Drones/Videos/Clip_32.mov" --demo-showcase --detector-variant B --camera auto --encoder h264_nvenc --async-render --max-frames 150 --output demo/final_showcase_pilot
.venv/Scripts/python.exe scripts/final_demo_validate.py --run demo/final_showcase_pilot --pilot --output reports/final_demo_pilot_validation.json
```

No `--detections` cache, oracle boxes or GT identities are supplied. Variant B is opt-in and retains its actual 800×1440 export shape and fixed-wing-only taxonomy; the existing five-class variant A remains the default outside this explicit command. The showcase flag selects the new presentation and conditional cue; the conservative research/production assessment remains accessible separately.

## Read the presentation

- **MONITORING:** a tracked observation without the current stronger convergence cue; this does not mean verified physical safety.
- **POSSIBLE CONVERGENCE:** the measured image-plane history supports the declared conditional growth/drift cue.
- **INSUFFICIENT EVIDENCE:** the necessary recent measured support is unavailable or unstable.

IDs come from the existing online tracker. **Solid trails contain only fresh detector-measured centres**, while the dashed path is a local 0.5 s robust linear image projection; predicted positions do not become trail or size evidence. The zoom is an enlarged crop of the actual source RGB. The early-convergence rule and the independent illustrative direction are different outputs: a stable `MONITORING` target may support an image-separation illustration without any convergence warning. A direction is displayed only when `advisory.demo.direction.status` is `resolved`; its declared meaning is `illustrative_image_plane_cue_not_safe_maneuver`. Missing range, ownship dynamics, verified de-rotation and calibration still prevent a safe-maneuver claim. The actual demo cue, reasons, support window, measured trail and priority target are retained in every `decisions.jsonl` record.

The displayed count of independent measurements means fresh detector observations instead of tracker-predicted extents. Successive detections may be correlated; the count is not a statistical-independence or calibrated-confidence claim.

The declared presentation rule uses a **0.6 s measured window**, at least 0.3 s support and five observations, minimum six-pixel span, and a median-pairwise slope of log maximum bbox span. A convergence candidate needs growth at least **0.035/s**, fitted log growth `growth × support_duration > max(0.015, 0.5 / median span, 2 × residual MAD)`, stable scale/aspect, and normalized drift `norm(image velocity) / median span <= max(0.08, 2 × growth)`. It becomes `POSSIBLE CONVERGENCE` only after 0.16 s confirmation. Scale residual MAD must be at most `max(0.025, 1 / median span)` and the log-aspect P90–P10 range at most 0.25. Camera-fit loss, clipping, ambiguity, missing or discontinuous measurements reset or withhold support. These are fixed image-presentation operating rules, not calibrated physical-collision thresholds.

The illustrative vector is the **negative normalized image-centre offset of the priority target's 0.5 s linear projection**, quantized to 45° directions and independently confirmed for 0.16 s. Central symmetry, incompatible other-target directions, occupied image directions or insufficient support leave it unresolved. It uses raw image coordinates, without a focal length, range estimate or camera-to-aircraft body-axis mapping.

## Deliverables and validation

The final output directory is `demo/final_showcase/`. The expected deliverables are `annotated.mp4`, `decisions.jsonl`, `detections.jsonl`, `benchmark.json`, `run_status.json`, `stable_tracking.png`, and `encounter_cue.png`. The video is regenerated by the command above; large video binaries remain local.

`reports/final_demo_validation.json` records the completed live TensorRT run, exact source/engine/runtime-code hashes, every decoded source/output frame, source and output PTS errors, H.264 dimensions/duration, actual measured processing FPS, and screenshot trace evidence. The final canvas must be at least 1920×1080 and the complete export 20–45 seconds. Original source cadence and duration are preserved. Processing throughput is reported as measured, without a speed acceptance gate; offline FPS is not a streaming-latency guarantee.

The validator selects two **unmodified full-resolution frames from the encoded output**, based on actual trace support. The first emphasizes sustained tracking. The second prefers a genuine recorded convergence cue; if no warning is observed, it may show a resolved illustrative direction during `MONITORING`, explicitly labelled **illustration without a convergence warning**. If neither is present, it selects a contrasting tracked frame and records **no observed warning/arrow**. It never invents an encounter or adds an arrow to a screenshot.

The completed primary run passed validation on 22 October 2026:

| Measurement | Actual result |
|---|---:|
| Source / fully decoded output frames | 1,023 / 1,023 |
| Annotated canvas / codec | 1920×1080 / H.264 NVENC |
| Encoded duration | 35.2758556 s |
| Processing throughput, including render/encode drain | 12.40 FPS |
| Throughput including initialization and all run overhead | 11.82 FPS |
| Source-to-trace timestamp error | 0 s |
| Maximum encoded timestamp quantization error | 5.364 μs |

Hardware was RTX 4060 with TensorRT 11.4.0.106 and the preserved custom CUDA OpenCV 5.0.0. All 1,023 processed frames used actual detector B inference; no observation cache was supplied. Playback retains the source's 29 fps timing and duration, independently of slower offline processing.

The trace contains 1,107 `MONITORING` and 435 `INSUFFICIENT EVIDENCE` target-frame samples, with **zero `POSSIBLE CONVERGENCE` warning samples**. Multiple targets can occur in one frame, so these are not video-frame counts. The conditional arrows are monitoring illustrations, not evidence that a collision threat was detected.

- [Stable tracking screenshot](../demo/final_showcase/stable_tracking.png): frame 307, 10.586 s, two visible tracked targets; priority ID 2, illustrative up-left image direction.
- [Contrasting illustration screenshot](../demo/final_showcase/encounter_cue.png): frame 1,019, 35.138 s; priority ID 7, illustrative left image direction. This shows `MONITORING`, not a convergence warning.

Both PNGs are unmodified 1920×1080 decoded frames from the final video and were visually inspected at full resolution. Hashes and selected trace evidence are in `reports/final_demo_validation.json`.

## Optional contrasting recording

The same frozen mode was also run live on the complete `Clip_43.mov`, with no method tuning: **803 frames, 27.7950778 s, 1920×1080 H.264**, 12.75 processing FPS and 11.98 FPS including initialization. Every source/output frame decoded successfully; the maximum encoded PTS error was 5.539 μs. The trace contains 796 `MONITORING`, 580 `INSUFFICIENT EVIDENCE` and zero `POSSIBLE CONVERGENCE` target-frame samples. Its arrows are also illustrations without convergence warnings.

```powershell
.venv/Scripts/python.exe -m airavoid "X:/DATASETS/UAV/TRACKING/RGB/NPS-Drones/Videos/Clip_43.mov" --demo-showcase --detector-variant B --camera auto --encoder h264_nvenc --async-render --output demo/final_showcase_contrast
.venv/Scripts/python.exe scripts/final_demo_validate.py --run demo/final_showcase_contrast --output reports/final_demo_contrast_validation.json
```

The additional [tracking frame](../demo/final_showcase_contrast/stable_tracking.png) at 18.069 s shows three tracks with the direction temporarily unresolved. The [illustration frame](../demo/final_showcase_contrast/encounter_cue.png) at 26.237 s shows a resolved up-left image direction. Both were extracted and inspected at full resolution; `reports/final_demo_contrast_validation.json` preserves their exact trace evidence and hashes.

## Additional recorded approach example with a genuine cue

The unchanged cue was also run live on the existing **QUT M2 35 s selected recording**, using the original five-class detector A and the declared **stationary ground-camera** setup. This is a separate previously used encounter-category example, not a moving-camera NPS test or an independently held-out collision trial. Its category describes an approaching aircraft and does not supply physical CPA/collision truth. Selection provenance is in `data/qut/M2.manifest.json`: source frames 8916–9790, 875 frames at 25 fps, starting 120.64 s into the available full M2 sequence. Source SHA-256 is `9c44cecd3d818089f3d2b22c3f71bd139e5e720eae1935c664cc65a891ae0bb7`.

```powershell
.venv/Scripts/python.exe -m airavoid data/qut/M2.mp4 --demo-showcase --detector-variant A --camera fixed --encoder h264_nvenc --async-render --output demo/final_showcase_approach
.venv/Scripts/python.exe scripts/final_demo_validate.py --run demo/final_showcase_approach --detector-variant A --output reports/final_demo_approach_validation.json
```

All **875 source and output frames** decoded successfully: **1920×1080 H.264, 35.000 s**, original PTS preserved with zero measured error, 18.13 processing FPS and 17.15 FPS including initialization. Detector A made seven actual network calls per frame through its existing tiled path. There was no detection replay or GT input and no warning-threshold tuning for this example.

The actual trace contains **276 `POSSIBLE CONVERGENCE` target-frame samples**, alongside 601 `MONITORING` and 2,629 `INSUFFICIENT EVIDENCE` samples. Those counts include every predicted hypothesis and are not collision-accuracy measurements. The selected amber warning visibly corresponds to the real approaching aircraft, not a tiny unmatched speck:

- [Tracked aircraft before confirmation](../demo/final_showcase_approach/stable_tracking.png): frame 747, 29.88 s, ID 115, `MONITORING` while confirming growth.
- [Recorded convergence cue and illustrative arrow](../demo/final_showcase_approach/encounter_cue.png): frame 817, 32.68 s, ID 115, amber `POSSIBLE CONVERGENCE`, upward image-direction illustration. Its longest consecutive selected warning/arrow interval is 29.92–32.68 s. At the selected frame the measured log-span growth is 0.340/s and normalized drift 0.295/s.

Both screenshots are unmodified encoded frames inspected at full resolution. The warning is a genuine output of the fixed image cue; the upward arrow remains an image-plane illustration, not a verified safe maneuver. `reports/final_demo_approach_validation.json` stores its exact evidence and hashes. The screenshot selector prefers a prominent measured aircraft-class prediction for the A example instead of selecting a collection of tiny bird-class hypotheses; inference and the saved trace are unchanged.

## Regression checks

All **21 demo cue/render tests** and **13 existing Stage 2 runtime tests** passed; logs are `reports/final_demo_tests.log` and `reports/final_demo_runtime_regression.log`. The 150-frame default-mode decision trace is byte-identical to baseline commit `d6e146a00836fb281ca2edcf353e1fac63b97e74`: SHA-256 `976dd726a8d387366dfdacc1facb2d58709d1278e9891124eb5f8c0769310978`. This default comparison replays the same actual detector cache over original RGB to check behavior; it is not a neural-speed measurement. The three showcased exports above were separate live neural runs, each verified against original source PTS.

The final [artifact inventory and measured-support audit](../reports/final_demo_artifacts.json)
checks all 2,701 actual neural updates: fresh cue boxes match actual detector
outputs, support timestamps are causal, and resolved arrows/warnings have fresh,
unclipped, unambiguous evidence with a valid declared camera diagnostic. No
support errors were found. Independent image-geometry checks confirm that every
displayed arrow increases the declared image-plane separation criterion; they do
not establish a physically safe maneuver. The [environment check](../reports/final_demo_environment.json)
preserves the exact custom CUDA OpenCV path/build. Recreate the final inventory
with `scripts/final_demo_audit.py` after completing all three exports.

The showcase is selected demonstration evidence. Detector errors, false
short-lived hypotheses, association changes and image-cue errors remain possible;
the lower-threshold five-class QUT run contains extra hypotheses. No detection,
warning, identity or physical-safety accuracy is inferred from these selected
videos. The original scientific estimators, experiment reports and default CLI
are preserved separately. The [sprint record](../reports/final_demo_sprint.json)
records the execution window and account-wide usage checkpoint.
