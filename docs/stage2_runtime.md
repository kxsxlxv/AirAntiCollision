# Stage 2 practical runtime integration

The existing five-class detector and custom log geometry remain the defaults.
The controlled research has not justified promoting an RGB-only method to a
physical collision classifier. Explicit research selections are available as
`--assessment METHOD`; the common result remains in
`objects[].assessment.research`, including decision, physical meaning, evidence
age, validity conditions, reasons and uncalibrated confidence (`null`).

`--detector-variant A` uses the supplied five-class 416×704 detector: airplane,
helicopter, bird, balloon and parachute. Its conservative configuration retains
full-view plus overlapping native crops and CUDA OpenCV preprocessing.
`--detector-variant B` opts into the supplied fixed-wing-only P2 export, input
1×3×800×1440 and output 1×300×6. Export class `item` maps to fixed-wing UAV as
specified by the user; it does not inherit the five-class names. It uses one
full-frame RGB NCHW `/255` input, linear resize, rounded dimensions and centred
114 padding. Incompatible tiled, CUDA-preprocess, sky, zoom and SOT flags are
rejected. Actual runtime tensor shapes, emitted classes and registered supplied
weight hashes are validated and recorded for each live run. TensorRT model
execution still uses the GPU when B preprocessing uses CPU OpenCV.

The user also confirmed that all targets in the six selected NPS clips are
fixed-wing. This is user-confirmed taxonomy, not independently annotated
per-frame class ground truth.

The original B TensorRT serialization is incompatible with this runtime. The
local FP32 engine was rebuilt from the exact supplied ONNX without retraining.
Asset SHA256 values and rebuilding details are in
`reports/stage2_method_provenance.json`. Engines, source footage and generated
videos remain local; the commands below generate real outputs rather than
linking to absent repository media.

```powershell
# Original baseline, retaining every detector class and the existing geometry.
.venv/Scripts/python.exe -m airavoid "X:/DATASETS/UAV/TRACKING/RGB/NPS-Drones/Videos/Clip_32.mov" --output demo/stage2_A --max-frames 150 --encoder h264_nvenc

# Explicit fixed-wing detector and published image-risk adaptation.
.venv/Scripts/python.exe -m airavoid "X:/DATASETS/UAV/TRACKING/RGB/NPS-Drones/Videos/Clip_32.mov" --detector-variant B --assessment lai_lin --output demo/stage2_B_lai --max-frames 150 --encoder h264_nvenc

# Published AirTrack expansion/depth-plane cues, with no invented classifier.
.venv/Scripts/python.exe -m airavoid "X:/DATASETS/UAV/TRACKING/RGB/NPS-Drones/Videos/Clip_32.mov" --detector-variant B --assessment airtrack_cues --output demo/stage2_B_airtrack --max-frames 150 --encoder h264_nvenc

# Complete sequential live benchmark and full decoded-frame/PTS QA.
.venv/Scripts/python.exe scripts/stage2_live_benchmark.py --research --async-controls --output artifacts/stage2/live_runtime_final
```

Every example preserves original source presentation timestamps. `--no-render`
measures decode→detector→tracking→assessment without video export. Each run saves
`benchmark.json`, `detections.jsonl`, `decisions.jsonl`, and, when enabled,
`annotated.mp4`. Replay is explicitly separate from neural inference: use the
matching `--detector-variant`, `--detections RUN/detections.jsonl`, a distinct
output directory and the same original source. P2 replay requires the upstream
detector contract. No GT-ID CSV is opened by the runtime pipeline.

```powershell
.venv/Scripts/python.exe -m airavoid "X:/DATASETS/UAV/TRACKING/RGB/NPS-Drones/Videos/Clip_32.mov" --detector-variant B --assessment lai_lin --detections artifacts/stage2/live_runtime_final/Clip_032_B_lai_lin_nvenc/detections.jsonl --output demo/stage2_B_lai_replay --max-frames 150 --no-render
```

This replay was executed: source identity is SHA256-verified, all 150 sampled
cache entries are present, and its decisions SHA256 is exactly identical to
the live encoded run (`reports/stage2_runtime_replay.json`). Replay throughput
is not neural-inference FPS. Choose a new output directory for timing reruns;
`--resume` validates previously completed outputs without rerunning them.

The research adapter retains raw measured XYXY detector extents, separately from
tracker prediction, and never interprets background KLT/RANSAC as independently
verified rotation. SOT extents cannot create fresh expansion evidence. Lai/Lin
rectangle-area adaptation is an image-risk index; AirTrack emits cues only;
Bauer requires supplied verified intrinsics and model assumptions. The existing
temporal policy is applied to displayed decisions while the raw research result
and its candidate decision remain in JSON. No physically actionable steering is
selected without the required ownship/range information.

The live runtime separately withholds Lai/Lin low-index clearance, including on
a declared fixed camera (`low_index_is_not_clearance_evidence`), because the
analytical physical counterexamples invalidate its physical-clearance
interpretation; QUT separately exposes encounter-category disagreement.
A nominal RV-C3BF point cone likewise cannot establish
finite-aircraft clearance without a verified footprint. Raw published candidate
decisions and risk indices remain in JSON and remain unchanged in the research
comparison. This is a deployment restriction, not a corrected reproduction.

Measured runtime results are in `reports/stage2_runtime.json`, regenerated by
the benchmark command. It records both detector variants on the same first 150
frames of two original videos, source and sampled cadence, all network calls,
model-load/first-inference/five-update warmup, FPS including encoder drain,
per-stage mean/p50/p95/p99, and sequential sample latency. The percentiles cover
decode through render submission and trace write; the final encoder drain is
included in total throughput, separately from frame percentiles. The benchmark
uses serial rendering for the primary matrix and separately labelled bounded
render-thread controls for A. CUDA device,
OpenCV path and source/weight/implementation digests are saved in each run.
Actual Jetson Orin Nano Super performance is **unmeasured**.

The exact supplied P2 ONNX was also checked against the local FP32 TensorRT
rebuild on ten development frames. All accepted detection counts/classes
matched; 19 matched boxes differed by at most 0.001343 pixels and scores by
0.000816. These samples do not establish equivalence on every input. The
functional check is reproduced by `scripts/stage2_p2_agreement.py` and recorded
in `reports/stage2_p2_agreement.json`.

Focused regression tests: `python -m unittest discover -s tests -p test_stage2_runtime.py -v`.
These check detector isolation, P2 letterbox geometry,
taxonomy rejection, model tensor validation, backend selection, missing fresh
evidence after SOT prediction, and moving-camera scope.

<!-- MEASURED_RUNTIME -->

## Actual RTX 4060 results

Sequential fresh-process measurements, 22 October 2026; 150 consecutive source frames per condition. A uses 10 model calls/update at 1280×960 and 17 at 1920×1080; B uses one. Source cadence is 29 Hz for Clip 032 and 29.97 Hz for Clip 001; stride is one. Reported FPS is offline processing throughput and includes the final encoder drain. These short samples do not certify sustained streaming or tail latency.

| Clip | Detector | Assessment | Export | FPS | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---:|---:|---:|---:|
| Clip_032 | A | legacy_log | none | 21.60 | 45.8 | 49.8 | 53.7 |
| Clip_032 | A | legacy_log | NVENC serial | 12.14 | 78.4 | 99.5 | 137.5 |
| Clip_032 | B | legacy_log | none | 17.56 | 56.4 | 59.4 | 62.4 |
| Clip_032 | B | legacy_log | NVENC serial | 10.82 | 87.2 | 119.4 | 168.0 |
| Clip_001 | A | legacy_log | none | 14.32 | 68.0 | 75.3 | 90.4 |
| Clip_001 | A | legacy_log | NVENC serial | 9.80 | 99.9 | 114.9 | 135.6 |
| Clip_001 | B | legacy_log | none | 16.78 | 57.7 | 67.3 | 92.3 |
| Clip_001 | B | legacy_log | NVENC serial | 11.55 | 84.9 | 92.8 | 116.9 |
| Clip_032 | A | lai_lin | NVENC serial | 12.33 | 79.8 | 87.4 | 104.0 |
| Clip_032 | B | lai_lin | NVENC serial | 11.27 | 87.5 | 94.7 | 107.4 |
| Clip_032 | A | airtrack_cues | NVENC serial | 12.24 | 78.5 | 94.4 | 135.3 |
| Clip_032 | B | airtrack_cues | NVENC serial | 10.97 | 88.2 | 105.7 | 147.4 |
| Clip_032 | A | legacy_log | NVENC threaded | 15.63 | 62.5 | 72.3 | 81.8 |
| Clip_001 | A | legacy_log | NVENC threaded | 11.81 | 82.9 | 91.2 | 105.4 |

Cold-start fields are recorded separately for each condition in the JSON/CSV. The first inference is a complete detector update, including all A crops; it is not one network invocation. Processes are fresh, while the operating-system file cache is uncontrolled. The environment remains the original custom CUDA-enabled OpenCV 5 / Torch 2.12 / TensorRT 11.4 stack.

Threaded controls use the existing bounded two-frame render/export worker. Their FPS includes completion of all queued work and encoder drain, while frame percentiles measure decode through submission/backpressure, not each frame’s completed encoding latency.

The matrix performed 2100 genuine neural-detector updates. All 1500 annotated frames were fully decoded and checked against original PTS, including frame count, H.264 codec and presentation duration. Serial encoding invariance is recorded per matched A/B condition in the JSON.

The source of the table is `reports/stage2_runtime.json`; `scripts/stage2_runtime_summary.py` regenerates it and the compact CSV. No Jetson timing is inferred from these desktop measurements.
