# RGB airborne encounter research demonstrator

Start with the [Russian video gallery](demo/index.html),
[visual guide](demo/README.md) and [measured results](RESULTS.md).

The [Stage 2 research results](docs/STAGE2_RESULTS.md) add controlled NPS
detector/identity comparisons, independent collision-method reproductions,
analytical failures and opt-in P2/assessment backends. The
[runtime guide](docs/stage2_runtime.md) contains actual RTX 4060 measurements and
reproducible export commands. Physical collision accuracy on moving-camera RGB
remains unproven; the acceptance gaps are explicit in that report.

Uses one RGB file, the supplied five-class YOLO26m detector, optional overlapping
crops, conservative motion-gated tracking, temporal image drift/expansion analysis,
camera-motion validity checks and explainable Russian video annotations.
States: potential threat, apparently non-threatening, insufficient evidence.
No metric range, focal length, physical aircraft size, ownship speed or collision
probability is invented. Avoidance illustration compares hypothetical image-flow
effects across targets and **abstains from physical maneuver recommendations**.

## Run in the supplied environment

```powershell
.venv/Scripts/python.exe -m airavoid data/qut/M2.mp4 --camera fixed --stride 2 --encoder h264_nvenc --render-process --output demo/my_run
.venv/Scripts/python.exe -m airavoid C:/path/input.mp4 --camera auto --output demo/arbitrary
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The local TensorRT engine is selected automatically when present. `--camera fixed`
is a declared acquisition assumption appropriate for the QUT grounded tripod
footage; do not apply it silently to airborne video. `auto` attempts background
similarity correction and abstains on unreliable motion. `unknown` always abstains.
`--policy raw` disables temporal confirmation; default `confirmed`
requires 0.24 s warning evidence and 1.2 s non-threat evidence. Missing/ambiguous
observations cancel confirmation. These are prototype design settings, not safety
standards or calibrated probabilities.

The default `--profile conservative` uses detector threshold 0.03, complete
native crop coverage, CUDA preprocessing for compatible TensorRT/CUDA, internal
crop margin 2 px, source-boundary geometry margin 1 px and HAC sensitivity bandwidth 1 s.
`--profile reference` restores the earlier 0.18 threshold, full-frame CPU preprocessing
and zero edge/HAC margins; explicit flags override profile values. Use `--full-frame`
to disable native crops or `--cpu-preprocess` to select CPU input conversion.

Automatic background stabilization does not identify camera rotation versus
translation or bound parallax. The conservative profile therefore withholds
conditional-pass states in `--camera auto`, even after a good background fit.
Potential warnings still depend on image-model assumptions. The research override
`--allow-similarity-pass` restores the unverified background-motion assumption;
the reference profile enables it. It does not establish physical clearance.

For the measured RTX4060 render path, add `--encoder h264_nvenc --render-process`.
An ordered render thread is also available with `--async-render`. Workers have a
two-frame limit and preserve source PTS; these are offline throughput options.
Actual decoded-frame, PTS and decision equivalence checks for thread/process
export are recorded in `reports/`. This is offline throughput, not streaming latency.

Each run saves H.264 MP4, PNG preview, JSONL model decisions and detections, and a
stage timing/environment/configuration report. `--no-render` omits video generation.
`run_status.json` marks running, failed or complete; a failed rerun removes stale
success timing. Output must not overwrite the source or the input replay cache.
`--detections` replays a model-generated cache for ablations; its throughput **is not
neural end-to-end inference performance**. GT is read only by separate scripts.

## Preserve CUDA OpenCV

Use `.venv/Scripts/python.exe` directly. Do **not** blindly `pip install -r` external
repositories or `uv sync`: OpenCV is externally provided and untracked by the
package resolver. Only `av==18.1.0` was added, with `--no-deps` after a one-package
dry run; `reports/environment_after_av.json` confirms the custom build is intact.
Required modules: cv2, numpy, torch, TensorRT, Pillow, av; FFmpeg/FFprobe on PATH.
The initial full fingerprint is `reports/environment_initial.json`:
OpenCV 5.0.0 at `C:/Data/DEV/Libs/opencv_cuda_5/install/python/cv2/__init__.py`, one
CUDA device, RTX 4060. Never replace it with an OpenCV PyPI wheel.

Supplied detector weights reside in `../anti_collision_system/`. Their ONNX
metadata identifies Ultralytics YOLO26m and AGPL-3.0 licensing; ownership/licensing
of the custom training data is external and has not been audited. Weights are not
copied into this project. The locally generated engine is machine/runtime specific:

```powershell
.venv/Scripts/python.exe scripts/build_engine.py
```

The supplied prebuilt engine was incompatible with TRT 11.4. Local build succeeded
in 110.09 s. ONNX Runtime CUDA hit a cuDNN library mismatch; use the verified direct
TensorRT path. An explicit ONNX path is accepted, but that GPU backend is not yet
validated in this environment.

## Data / evaluation

```powershell
.venv/Scripts/python.exe scripts/prepare_qut.py --sequences M1 M2 --seconds 35
.venv/Scripts/python.exe -m scripts.evaluate demo/M2_tiled_initial/decisions.jsonl data/qut/M2.gt.json
.venv/Scripts/python.exe -m scripts.analytic_sweep
.venv/Scripts/python.exe -m scripts.oracle_analysis
```

QUT source: [Martin et al. 2022](https://arxiv.org/html/2112.02735v1),
[dataset](https://open.qcr.ai/dataset/aircraft-collision-course/), BSD-3-Clause.
Local TIFF input is encoded at the published 25 FPS. Manifests record source frame
IDs, exact selection, split and video hashes. Last 35 s is a fixed selection rule;
some negatives visible only earlier are therefore absent. Centroids are approximate
and interpolated. An approaching target label is an encounter-category proxy,
not proof of physical collision; no true CPA labels are available.

See [RESEARCH.md](RESEARCH.md), [EXPERIMENTS.md](EXPERIMENTS.md),
[MISSION_STATUS.md](MISSION_STATUS.md), and generated reports for evidence and
limitations. Detector training provenance is unknown, so even untouched sequences
cannot be claimed detector-training-independent. No AOT or 3D simulator is used.

**Offline research prototype, not flight-certified or flight-ready safety equipment.**


## Optional experiments and source timing
`--gpu-preprocess --tiled` uses the preserved CUDA OpenCV build for actual TensorRT
input conversion. Native tiles cover the full raster at any resolution; inference
cost grows with image size (1280x720:7 calls including full image;3840x2160:50).
`--adaptive-tiles` reduces average calls with periodic discovery; it can miss targets
between discovery frames. `--appearance-bridge` runs the supplied external LoRAT
tracker between detector observations; independent spans are available through
`--detector-scale-only`. Both remain optional because measured gains are mixed.

Default export uses actual source presentation timestamps through PyAV and
libx264. `--video-timing cfr` selects the legacy nominal-rate writer. Variable frame
intervals are preserved, including final packet duration; invalid/nonmonotonic
source timestamps fail explicitly. If a decoder exposes no timestamps, the
benchmark records the nominal-rate fallback. Sky-mask alignment currently requires
CFR; `scripts.run_sky_demo` is an offline two-pass workflow, not a streaming claim.

`--geometry projective` fits an exact constant-shape inverse-span model;
`--geometry bounded` tests assumed centroid/span error bounds;
`--correlation-seconds .5` tests temporal covariance sensitivity of log geometry.
These methods have substantial abstention/model limits; see EXPERIMENTS.md.
Apparent contraction alone is uncertain by default. The legacy
`--allow-recession-clearance` flag reproduces the rejected stable-shape assumption.

`--geometry calibrated --camera fixed --calibration camera.json` additionally
requires a JSON object with `width`, `height`, `focal_px`, `cx`, `cy`, `rectified:true`
and a nonempty `provenance` string. Values must describe the actual original RGB
raster with square pixels and corrected distortion. The file is never inferred
from detector classes. CPA remains in assumed constant projected-target-span units,
not metres; uncertainty excludes systematic calibration and shape errors. This
alternative has analytical validation only, because no verified QUT intrinsics
were found. It does not authorize a flight-control recommendation.

## Measured failure boundaries and optional recall recovery

The all-source score is a category proxy at stride 5. Processing every frame on
all 15 complete studied sources finds 40 other-category warning samples: a
1.04-second pulse in M6 and a 0.56-second pulse in M7. The selected settings therefore do not eliminate other-category warnings.
[RESULTS.md](RESULTS.md) includes both cadences, abstentions and warning timing.

`--zoom-proposals` adds optional all-pixel tiny-object proposals followed by actual
YOLO inference on original RGB crops. It does not classify sky or call a proposal
a detection. On S2, its detection and decision traces exactly match the sky/all-region
controls after crop-boundary filtering. This path remains optional because the evidence is limited and it costs extra inference.

```powershell
.venv/Scripts/python.exe -m airavoid data/qut/S2.mp4 --camera fixed --stride 2 --zoom-proposals --encoder h264_nvenc --render-process --output demo/zoom_repeat
.venv/Scripts/python.exe -m scripts.run_sky_demo data/qut/S2.mp4 --camera fixed --encoder h264_nvenc --render-process --output demo/sky_repeat
```

The DINO workflow keeps its external environment isolated from OpenCV's Windows
DLL search-path changes. It preflights CFR before segmentation and binds the mask
to source/checkpoint hashes. Direct legacy masks without a manifest are explicitly
unverified; FPS and frame count alone cannot prove identity. The diagnostic
`--sky-all-control` flag ignores the mask region while retaining the same proposal
and crop inference path. Source/mask identity and final status appear in benchmarks.

For separate detection diagnostics, `scripts.prepare_air2land_boxes` needs SciPy;
it was run using the existing sibling environment without installing SciPy here.
`scripts.air2land_detection` uses only existing rendered RGB and supplied boxes.
No verified timing, calibration or physical collision labels are inferred from
those files. No simulator is operated. The primary evidence remains real RGB.

The final clipped-history repair excludes invalid clipped sizes rather than
letting the last one seed later growth. A full-rate actual-observation replay changes
one S1 warning to uncertainty; known other-category warning counts stay unchanged.
Current videos and the 76-test suite include this repair. Earlier benchmark runs
retain their exact source fingerprints and archived trace references.
