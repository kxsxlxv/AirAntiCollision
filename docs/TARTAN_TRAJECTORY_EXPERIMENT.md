# Real trajectory experiment: TartanAviation

24 October 2026. This is a selected recorded-data diagnostic, not collision
classification validation. The experiment ran genuine neural inference and
preserves failures. No training, new network, simulator or dependency installation
was used. The normal CLI, scientific estimators and detector A/B selections are
unchanged.

## Result

The selected Airbus A319-112 approaches a stationary ground camera, passes at a
large lateral offset, and recedes. The nearest **sampled horizontal ADS-B range
is 5,586.78 m**; the associated supplied slant proxy is 5,697.87 m. These are
reference measurements, **not distances predicted by our detector**. The
horizontal path is useful even though the altitude datum is not sufficiently
verified for physical vertical separation.

The experiment also exposes substantial weaknesses of the current five-class
model/tracker on this recording. At the existing confidence threshold 0.03:

| Observation | Measured result |
|---|---:|
| Original decoded source frames | 1,933 |
| Genuine inference updates, stride 2 | 967 |
| Actual model calls, including full view and native crops | 34,812 |
| Sampled reference aircraft boxes | 629 |
| Raw detections matched at IoU >= 0.3, any predicted class | 524 / 629 |
| Airplane classification among those matched detections | 213 / 524 |
| Fresh confirmed tracks matched at IoU >= 0.3 | 518 / 629 |
| Matched A319 cue: insufficient / monitoring / possible convergence | 320 / 198 / 0 |
| Matched samples with positive estimated visual growth > 0.035/s | 70 |
| All possible-convergence track samples, including unmatched objects | 152 |
| Predicted-ID changes between adjacent matched observations < 1 s apart | 61 |
| Growth-sign agreement with reference horizontal-range trend, eligible samples | 105 / 211 (49.76%) |
| Offline processing, including startup and encoder completion | 5.41 FPS |

The 61 ID changes use the explicitly stated short-gap diagnostic; they are not
an official MOT identity-switch metric. IoU 0.3 is a permissive diagnostic, not
AP or standard benchmark recall. Any-class localization matches do not establish
correct classification. The measured matched spans are only 10.5–30.8 source
pixels; short-window bounding-box expansion is noisy at this scale.

There is no warning on the matched A319 observations, including 70 positive
growth samples. This illustrates that growth alone does not trigger the cue.
It does **not** prove correct physical rejection: many observations are
insufficient, and the video has no collision labels. The 152 warning samples
do not overlap the reference aircraft at the scoring threshold. Manual review
shows background detections among these cases; annotation completeness for all
other objects has not been established, so they are not all assigned a false
positive label.

**Engineering implication:** even with a known reference trajectory, noisy
extent measurements, unstable classes and associations limit this experiment.
Adding IMU alone would not fix these failures. This recording does not justify
promoting the existing image cue to physical collision assessment.

## Source, time and reference audit

Source: `1_2022-06-26-08-32-07`, TartanAviation, static KAGC ground camera.
Aircraft ICAO decimal `11166922`, Airbus A319-112. RGB source is 2448x2048,
encoded at 24 fps; source SHA256
`8c8b6f772f21f9c48ea44a7716eacf73d7d21094e974d71ec4ed3387e8d2610a`.
The NPS GoPro/1.5 m range preset is **not applied** to this camera or aircraft.

The clip has 1,258 annotated frames, 68 paired spatial ADS-B records and 1,934
SRT RT clock records. Paired-record gaps have a median of 1.015 s and maximum
of 3.014 s. `.label` distance is horizontal kilometres. The supplied paired
vector contains north, east, vertical proxy and its slant norm in metres;
this norm is numerically checked. Label/reference horizontal distances differ
by a median 2.88 m at 44 overlapping sampled reference frames. Both derive
from ADS-B, so this agreement is a consistency check, not independent accuracy.

The separate raw receiver CSV provides only nine deduplicated reports within
the video time window. Only one paired coordinate exactly matches these raw
reports, with a 2.019 s timestamp difference. It cannot independently validate
the dense paired trajectory's timing. Its available altitude flags are
non-GNSS; no verified vertical-clearance claim is made.

The [official extraction convention](https://github.com/castacks/TartanAviation/blob/main/vision/download.py)
uses 24 fps and starts at `2.png`. This experiment maps zero-based decoded frame
`i` to ordinal `i+2`, checking that video frames equal SRT records minus one.
The offset was **not optimized using GT**. A +/-2 ordinal sensitivity check
gives 510–520 matched track samples, versus 518 for the declared mapping.
Counts and label alignment do not independently calibrate RT/exposure delay;
one/two-frame timing uncertainty remains.

The SRT display intervals have startup discontinuities. The experimental runtime
uses increasing RT acquisition timestamps, normalized to ordinal 2, rather than
those intervals or `frame/24`. Runtime source span is 78.874 s including the
estimated last-frame duration, whereas the original encoded duration is 80.542 s.
Original decoded frame order/proportions are preserved; exported video follows
RT intervals. Trace rows retain acquisition UTC, ordinal and original MP4 PTS.

## Reproduce

Run from the repository root with the existing CUDA OpenCV / TensorRT environment.
The downloader requests only original MP4 and metadata, skips the labelled AVI,
checks member CRC and hashes, and resumes interrupted compressed members. The
raw day uses the already installed `C:/Program Files/7-Zip/7z.exe` for Deflate64.
No full dataset or year archive needs to be downloaded.

```powershell
.venv/Scripts/python.exe scripts/tartan_sample_fetch.py --sample 1_2022-06-26-08-32-07 --output artifacts/tartan_jun26 --video --adsb-date 06-26-22

.venv/Scripts/python.exe scripts/tartan_visual_experiment.py --video artifacts/tartan_jun26/1_2022-06-26-08-32-07.mp4 --clock artifacts/tartan_jun26/1_2022-06-26-08-32-07_subtitle.srt --output demo/tartan_jun26_rerun --stride 2

.venv/Scripts/python.exe scripts/tartan_audit.py --folder artifacts/tartan_jun26 --run demo/tartan_jun26_rerun --output reports/tartan_jun26_rerun

.venv/Scripts/python.exe scripts/tartan_review.py --folder artifacts/tartan_jun26 --run demo/tartan_jun26_rerun --output demo/tartan_trajectory_review_rerun
```

The experimental runtime receives only RGB, its acquisition clock and detector
weights. It reuses the original five-class YOLO26m, custom tracker, early visual
cue and aviation renderer. It never reads GT boxes, identities, ADS-B positions,
distances or events. Scientific physical assessment is explicitly unassessed
in this separate opt-in experiment. Only the **offline audit and review** read
references. White A319 markers and the metric path in the review are clearly
labelled reference data; they do not appear as neural detections in the runtime.

## Completed local outputs

Workspace root: `C:/Users/kisel/PycharmProjects/AirAntiCollision/`.

- [Normal inference video, 3840x1080](../demo/tartan_jun26/annotated.mp4)
- [Offline reference review, 1920x1080](../demo/tartan_trajectory_review/evaluation_reference.mp4)
- [Approach screenshot](../demo/tartan_trajectory_review/approach.png)
- [Nearest-pass screenshot](../demo/tartan_trajectory_review/nearest_pass.png)
- [Receding screenshot](../demo/tartan_trajectory_review/receding.png)
- [Reference ground-track diagram](../reports/tartan_jun26/trajectory_reference.png)
- [Audit / measured results](../reports/tartan_jun26/audit.json)
- [Matched observations and cue reasons](../reports/tartan_jun26/matched_observations.csv)
- [Paired reference points](../reports/tartan_jun26/reference_points.csv)
- [Causal reference-only horizontal CPA diagnostic](../reports/tartan_jun26/reference_causal_horizontal_cpa.csv)
- [Live model provenance](../demo/tartan_jun26/benchmark.json)

Large original footage, MP4s and uncompressed `decisions.jsonl` / `detections.jsonl`
remain local and reproducible. Lossless gzip copies of both full live traces
are included in `reports/tartan_jun26/`. Their uncompressed SHA256 values are
recorded in `trace_manifest.json`. No absent media file is represented as a
GitHub-hosted video. Open the [local review gallery](../demo/TARTAN_EXPERIMENT.html)
to compare the two exports.

## Verification and limits

39 focused tests passed: eight new reference parsing/matching/causality tests,
14 existing early-cue tests, four CLI-default tests and 13 Stage 2 runtime tests.
Logs are in `reports/tartan_jun26/tests.log`. Full decoded media checks are in
`reports/tartan_jun26/video_qa.json` and `review_video_qa.json`.

Training-set overlap for this Tartan recording is unknown. This is a selected
diagnostic, not a held-out benchmark. Static-camera aircraft reference trajectories
allow geometry sanity checks but do not test moving-ownship IMU compensation,
safe maneuvers or physical collision classification. Camera calibration and
independently verified exposure/telemetry synchronization remain unavailable.

Data attribution: Jay Patrikar et al., **Image, speech, and ADS-B trajectory
datasets for terminal airspace operations**, Scientific Data 12, 468 (2025),
[DOI / paper](https://doi.org/10.1038/s41597-025-04775-6).
[Dataset and download source](https://github.com/castacks/TartanAviation/).
Data is CC BY 4.0 according to the official repository; code is BSD-3-Clause.
Review remains in [draft PR #3](https://github.com/kxsxlxv/AirAntiCollision/pull/3).
